"""Background thread that runs SAM/DINO/matching/persistence off the video loop.

Subfunctions:
- Own a bounded job queue and an unbounded result queue; submit() never blocks the
  caller (put_nowait), returning False when full or the worker is unavailable.
- Run one background thread that pulls jobs, calls SceneProcessor (SAM + DINO +
  gallery matching) or persists a reference, and pushes one IdentificationResult
  per job.
- Own the in-memory ReID gallery snapshot used for matching; refreshed in this
  thread only, right after a reference is actually saved, so no cross-thread
  mutation of gallery state is possible.
- When a ReidDiagnostics recorder is supplied, record every resolve job's query
  row, all candidate scores and its crops (disk writes stay off the video thread).
- Report initial gallery-loading failure to the main thread before startup returns.
- Request shutdown via an event and sentinel, with a bounded wait for active work.

This module never touches tracker, event-engine, or state-machine state; it only
calls SceneProcessor and DatabaseStore. project_auto.events.coordinator (main
thread) is the only place identification results are turned into item/track
bindings.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from time import monotonic
from typing import Protocol

import numpy as np

from project_auto.events.identification import IdentificationJob, IdentificationResult
from project_auto.events.reference_policy import CandidateKind
from project_auto.memory.store import DatabaseStore
from project_auto.perception.scene_processor import IdentityDecision, SceneProcessor
from project_auto.utils.logging import Category, log_action
from project_auto.utils.reid_diagnostics import CandidateRecord, QueryRecord, ReidDiagnostics

logger = logging.getLogger(__name__)

# Sentinel placed on the job queue to unblock the worker thread's get() on shutdown.
_SHUTDOWN = None


class IdentificationWorkerProtocol(Protocol):
    """What IdentityCoordinator needs from a worker; real or test double."""

    def start(self) -> None: ...

    def stop(self, timeout: float | None = None) -> None: ...

    def submit(self, job: IdentificationJob) -> bool: ...

    def poll_results(self) -> list[IdentificationResult]: ...


@dataclass(slots=True)
class IdentificationWorker:
    """Run identification/reference jobs on one background thread."""

    scene_processor: SceneProcessor
    store: DatabaseStore
    reid_model_name: str
    # Hard ceiling on stored references per item. TODO(gallery): V1 simply stops
    # accepting new references at the cap; a diversity-aware replacement (evict the
    # most redundant member to admit a genuinely new appearance) is future work.
    max_references_per_item: int
    job_queue_max_size: int = 8
    # Expensive-stage reference-quality gate, applied only to capture jobs so that
    # identity resolution keeps working on views too poor to learn from.
    min_mask_score: float = 0.0
    min_mask_occupancy: float = 0.0
    # A candidate at or above this similarity to an existing reference of the same
    # item adds no appearance diversity and is rejected as redundant.
    reference_novelty_threshold: float = 1.0
    # Optional ground-truth diagnostics; None disables recording entirely.
    diagnostics: ReidDiagnostics | None = None
    _job_queue: queue.Queue = field(init=False, repr=False)
    _result_queue: queue.Queue = field(init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _startup_complete: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _startup_error: BaseException | None = field(default=None, init=False, repr=False)
    _stop_requested: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _gallery: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._job_queue = queue.Queue(maxsize=self.job_queue_max_size)
        self._result_queue = queue.Queue()

    def start(self) -> None:
        """Wait for worker-owned gallery loading; propagate startup failure to the caller."""
        if self._startup_error is not None:
            raise RuntimeError(
                "Identification worker failed to load its gallery"
            ) from self._startup_error
        if self._stop_requested.is_set():
            raise RuntimeError("A stopped identification worker cannot be restarted")
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="identification-worker", daemon=True
        )
        self._thread.start()
        self._startup_complete.wait()
        if self._startup_error is not None:
            self._thread.join()
            raise RuntimeError(
                "Identification worker failed to load its gallery"
            ) from self._startup_error

    def stop(self, timeout: float | None = 5.0) -> None:
        """Finish the current job, discard queued work, and wait at most timeout seconds.

        Running inference cannot be interrupted; retain the thread so a later stop
        can join it again if the timeout expires.
        """
        self._stop_requested.set()
        if self._thread is None:
            return
        if self._thread.is_alive():
            try:
                self._job_queue.put_nowait(_SHUTDOWN)
            except queue.Full:
                pass  # The stop event ends the loop after the current job.
        self._thread.join(timeout)
        if not self._thread.is_alive():
            self._thread = None

    def submit(self, job: IdentificationJob) -> bool:
        """Enqueue without blocking; reject a full queue or an unavailable worker."""
        if (
            not self._startup_complete.is_set()
            or self._startup_error is not None
            or self._stop_requested.is_set()
            or self._thread is None
            or not self._thread.is_alive()
        ):
            return False
        try:
            self._job_queue.put_nowait(job)
            return True
        except queue.Full:
            return False

    def poll_results(self) -> list[IdentificationResult]:
        """Drain every result currently available, without blocking."""
        results: list[IdentificationResult] = []
        while True:
            try:
                results.append(self._result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def refresh_gallery(self) -> None:
        """Reload the in-memory gallery snapshot used for identity matching."""
        self._gallery = self.store.load_reid_gallery(self.reid_model_name, statuses=None)

    def _run(self) -> None:
        try:
            self.refresh_gallery()
        except BaseException as exc:  # Always release the startup waiter, even on thread exit.
            self._startup_error = exc
            return
        finally:
            self._startup_complete.set()
        while not self._stop_requested.is_set():
            job = self._job_queue.get()
            if job is _SHUTDOWN:
                return
            self._result_queue.put(self._process_safely(job))

    def _process_safely(self, job: IdentificationJob) -> IdentificationResult:
        started_at = monotonic()
        try:
            result = self._process(job)
        except Exception as exc:  # noqa: BLE001 - a bad job must not kill the worker
            logger.exception("identification.worker_error track_id=%s kind=%s", job.track_id, job.kind)
            log_action(Category.ERROR, track=job.track_id, kind=job.kind, error=repr(exc))
            result = IdentificationResult(
                kind=job.kind,
                track_id=job.track_id,
                status="error",
                item_id=job.item_id,
                failure_reason=repr(exc),
                candidate_kind=job.candidate_kind,
                query_id=job.query_id,
            )
        logger.debug(
            "identification.job_done track_id=%s kind=%s status=%s duration_s=%.3f",
            job.track_id,
            job.kind,
            result.status,
            monotonic() - started_at,
        )
        return result

    def _process(self, job: IdentificationJob) -> IdentificationResult:
        if job.kind == "resolve":
            return self._process_resolve(job)
        return self._process_capture(job)

    def _process_resolve(self, job: IdentificationJob) -> IdentificationResult:
        if job.frame is None or job.box is None:
            raise ValueError("A resolve job requires a frame and a box")

        decision = self.scene_processor.process(job.frame, job.box, job.track_id, self._gallery)
        self._record_resolve(job, decision)
        if decision.decision == "pending":
            return IdentificationResult(
                kind="resolve",
                track_id=job.track_id,
                status="pending",
                failure_reason="unusable_reference",
                query_id=job.query_id,
            )

        status: str = "new" if decision.decision == "new" else "existing"
        return IdentificationResult(
            kind="resolve",
            track_id=job.track_id,
            status=status,  # type: ignore[arg-type]
            item_id=decision.item_id,
            similarity=decision.similarity,
            reference=decision.reference,
            query_id=job.query_id,
        )

    def _record_resolve(self, job: IdentificationJob, decision: IdentityDecision) -> None:
        """Write one resolve job's query row, candidate rows and crops, if enabled."""
        if self.diagnostics is None or job.query_id is None or job.frame is None or job.box is None:
            return
        reference = decision.reference
        if decision.decision == "pending":
            label = "DEFER"
        elif decision.decision == "existing":
            label = "MATCH"
        elif decision.reason == "low_margin":
            label = "AMBIG"
        else:
            label = "NEW"

        frame_height, frame_width = job.frame.shape[:2]
        x1, y1, x2, y2 = job.box
        raw_crop = job.frame[
            max(0, y1) : min(frame_height, y2), max(0, x1) : min(frame_width, x2)
        ]
        self.diagnostics.record_query(
            QueryRecord(
                query_id=job.query_id,
                frame_index=job.frame_index,
                track_id=job.track_id,
                box=job.box,
                det_confidence=job.detection.confidence if job.detection is not None else None,
                sam_score=reference.sam_score if reference is not None else None,
                mask_occupancy=reference.mask_occupancy if reference is not None else None,
                gallery_size=len(self._gallery),
                decision=label,
                decision_reason=decision.reason,
                matched_item_id=decision.item_id,
            ),
            [
                CandidateRecord(
                    item_id=candidate.item_id,
                    final_score=candidate.final_score,
                    dino_score=candidate.dino_score,
                    color_score=candidate.color_score,
                    aspect_score=candidate.aspect_score,
                    n_references=candidate.n_references,
                )
                for candidate in decision.candidates
            ],
            raw_crop_bgr=raw_crop,
            masked_crop=reference.crop if reference is not None else None,
        )

    def _reject_capture(
        self, job: IdentificationJob, reason: str, **evidence: object
    ) -> IdentificationResult:
        """Report one candidate that will not become a reference.

        Returning before add_reference_if_needed is what keeps a rejected
        candidate from touching the item's stored prototype.
        """
        log_action(
            Category.CAPTURE,
            track=job.track_id,
            item=job.item_id,
            saved=False,
            kind=job.candidate_kind.value if job.candidate_kind else None,
            reason=reason,
            **evidence,
        )
        return IdentificationResult(
            kind="capture",
            track_id=job.track_id,
            status="skipped",
            item_id=job.item_id,
            failure_reason=reason,
            candidate_kind=job.candidate_kind,
        )

    def _best_existing_similarity(self, item_id: int, embedding) -> float:
        """Return the highest similarity to this item's stored references, or 0.0.

        Embeddings are unit-normalized on the way in, so the dot product is their
        cosine similarity. An item with no references yet is maximally novel.
        """
        for entry in self._gallery:
            if entry.item_id != item_id or len(entry.embeddings) == 0:
                continue
            return float(np.max(entry.embeddings @ np.asarray(embedding, dtype=np.float32)))
        return 0.0

    def _process_capture(self, job: IdentificationJob) -> IdentificationResult:
        if job.item_id is None:
            raise ValueError("A capture job requires an item_id")

        reference = job.precomputed_reference
        if reference is None:
            if job.frame is None or job.box is None:
                raise ValueError("A capture job needs a precomputed reference or frame+box")
            reference = self.scene_processor.prepare_reference(job.frame, job.box, job.track_id)

        if reference is None:
            return IdentificationResult(
                kind="capture",
                track_id=job.track_id,
                status="pending",
                item_id=job.item_id,
                failure_reason="unusable_reference",
                candidate_kind=job.candidate_kind,
            )

        if reference.sam_score < self.min_mask_score:
            return self._reject_capture(
                job, "low_mask_score", sam_score=reference.sam_score
            )
        if reference.mask_occupancy < self.min_mask_occupancy:
            return self._reject_capture(
                job, "low_mask_occupancy", mask_occupancy=reference.mask_occupancy
            )

        # Baseline references are deliberately not novelty-gated: the two initial
        # views of a newly added item are expected to look alike.
        if job.candidate_kind is not CandidateKind.INITIAL:
            similarity = self._best_existing_similarity(job.item_id, reference.embedding)
            if similarity >= self.reference_novelty_threshold:
                return self._reject_capture(job, "redundant", similarity=similarity)

        saved, _ = self.store.add_reference_if_needed(
            item_id=job.item_id,
            embedding=reference.embedding,
            model_name=self.reid_model_name,
            target_count=self.max_references_per_item,
            aspect_ratio=reference.aspect_ratio,
            color_histogram=reference.color_histogram,
        )
        if not saved:
            return self._reject_capture(job, "gallery_full")

        log_action(
            Category.CAPTURE,
            track=job.track_id,
            item=job.item_id,
            saved=True,
            kind=job.candidate_kind.value if job.candidate_kind else None,
        )
        self.refresh_gallery()
        return IdentificationResult(
            kind="capture",
            track_id=job.track_id,
            status="captured",
            item_id=job.item_id,
            reference=reference,
            candidate_kind=job.candidate_kind,
        )
