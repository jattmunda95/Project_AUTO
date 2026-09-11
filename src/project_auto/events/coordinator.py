"""Coordinate scene processing, identity resolution, and reference capture.

Subfunctions:
- Submit a lightweight IdentificationJob for each newly confirmed (ADD) track and
  return immediately; the real-time loop never waits on SAM/DINO/matching/persistence.
- Drain completed IdentificationResults every frame and apply them: create/associate
  permanent items, route NEW/EXISTING through the event layer, and schedule spaced
  reference captures, all on the main thread.
- Own ReferenceManager so a track never has more than one outstanding job, and a
  bad result (unusable mask, full queue, claimed item) defers retry with a cooldown
  instead of resubmitting every frame.
- Apply cheap pre-SAM quality filtering (box area, confidence, frame-edge clipping)
  before ever submitting a job.
- Route MOVED/REMOVE signals straight to the event engine; a REMOVE also forgets
  the retired track's identification state so a reused track ID starts fresh.

Owns the in-memory ReID gallery refresh trigger indirectly: the worker owns the
actual gallery snapshot (see identification_worker.py) since only it reads/writes
references. No tracker-stage inspection or direct database access beyond gallery-
free item/status lookups; the tracker only supplies signals and per-frame detections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import monotonic
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from project_auto.events.event_engine import EventEngine
from project_auto.events.identification import (
    IdentificationJob,
    IdentificationResult,
    ReferenceManager,
)
from project_auto.events.identification_worker import (
    IdentificationWorker,
    IdentificationWorkerProtocol,
)
from project_auto.memory.models import ItemStatus
from project_auto.memory.store import DatabaseStore
from project_auto.perception.detector import Detection
from project_auto.perception.scene_processor import PreparedReference, SceneProcessor
from project_auto.perception.tracker import TrackSignal, TrackSignalType

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IdentityCoordinator:
    """Submit identification work asynchronously and apply results as they land."""

    scene_processor: SceneProcessor
    event_engine: EventEngine
    store: DatabaseStore
    reid_model_name: str
    reference_target_count: int
    capture_interval_seconds: float
    bad_mask_cooldown_seconds: float = 5.0
    queue_full_retry_seconds: float = 0.5
    min_box_area: int = 0
    min_detector_confidence: float = 0.0
    edge_margin_pixels: int = 0
    job_queue_max_size: int = 8
    clock: Callable[[], float] = field(default=monotonic, repr=False)
    worker: IdentificationWorkerProtocol | None = None
    _reference_manager: ReferenceManager = field(init=False, repr=False)
    _active_track_ids: set[int] = field(default_factory=set, init=False, repr=False)
    _last_detection_by_track: dict[int, Detection] = field(
        default_factory=dict, init=False, repr=False
    )
    _last_capture_at: dict[int, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._reference_manager = ReferenceManager(clock=self.clock)
        if self.worker is None:
            self.worker = IdentificationWorker(
                scene_processor=self.scene_processor,
                store=self.store,
                reid_model_name=self.reid_model_name,
                reference_target_count=self.reference_target_count,
                job_queue_max_size=self.job_queue_max_size,
            )
        self.worker.start()

    def shutdown(self, timeout: float | None = 5.0) -> None:
        """Stop the background worker cleanly; safe to call once at app exit."""
        if self.worker is not None:
            self.worker.stop(timeout)

    def handle_frame(
        self,
        frame: NDArray[np.uint8],
        signals: list[TrackSignal],
        detections_by_track_id: dict[int, Detection],
    ) -> None:
        """Apply pending results, dispatch this frame's signals, and submit/retry work.

        Never blocks on SAM, DINO, matching, or persistence: submission is a
        non-blocking queue put, and results already computed by the worker are
        drained with a non-blocking get.
        """
        now = self.clock()
        assert self.worker is not None
        self._apply_results(self.worker.poll_results())

        for signal in signals:
            if signal.signal_type is TrackSignalType.ADD:
                self._active_track_ids.add(signal.track_id)
                self._last_detection_by_track[signal.track_id] = signal.detection
                self._try_submit_resolve(frame, signal.track_id, signal.detection, now)
            elif signal.signal_type is TrackSignalType.REMOVE:
                self._active_track_ids.discard(signal.track_id)
                self._reference_manager.forget(signal.track_id)
                self._last_detection_by_track.pop(signal.track_id, None)
                # A track that retired while still unresolved never got a permanent item.
                if self.event_engine.item_id_for_track(signal.track_id) is not None:
                    self.event_engine.process_signal(signal)
            elif self.event_engine.item_id_for_track(signal.track_id) is not None:
                self.event_engine.process_signal(signal)
            else:
                # Identity resolution for this track has not completed yet (it is
                # still QUEUED/PROCESSING/DEFERRED in the background worker), so
                # there is no item to attach this MOVED signal to. Drop it rather
                # than block or crash; the item's location is still accurate as
                # of its next confirmed placement once identity resolves.
                logger.info(
                    "identification.signal_dropped_unidentified track_id=%s signal_type=%s",
                    signal.track_id,
                    signal.signal_type,
                )

        for track_id in self._reference_manager.pending_track_ids():
            detection = detections_by_track_id.get(track_id)
            if detection is None:
                continue
            self._last_detection_by_track[track_id] = detection
            self._try_submit_resolve(frame, track_id, detection, now)

        for track_id, detection in detections_by_track_id.items():
            self._last_detection_by_track[track_id] = detection
            item_id = self.event_engine.item_id_for_track(track_id)
            if item_id is None:
                continue
            self._active_track_ids.add(track_id)
            self._maybe_capture(frame, detection, item_id, now)

    def _passes_prefilter(
        self, frame: NDArray[np.uint8], detection: Detection
    ) -> tuple[bool, str]:
        """Cheap reject before ever invoking SAM/DINO; returns (usable, reason)."""
        x1, y1, x2, y2 = detection.box
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return False, "invalid_crop_dimensions"
        if width * height < self.min_box_area:
            return False, "box_too_small"
        if detection.confidence < self.min_detector_confidence:
            return False, "low_detector_confidence"
        frame_height, frame_width = frame.shape[:2]
        margin = self.edge_margin_pixels
        if x1 <= margin or y1 <= margin or x2 >= frame_width - margin or y2 >= frame_height - margin:
            return False, "clipped_by_frame_edge"
        return True, ""

    def _try_submit_resolve(
        self,
        frame: NDArray[np.uint8],
        track_id: int,
        detection: Detection,
        now: float,
    ) -> None:
        """Submit one identity-resolution job if this track is currently eligible."""
        assert self.worker is not None
        if not self._reference_manager.can_submit(track_id, now):
            return

        usable, reason = self._passes_prefilter(frame, detection)
        if not usable:
            self._reference_manager.mark_deferred(track_id, now, self.bad_mask_cooldown_seconds, reason)
            print(f"[Coordinator] track_id={track_id} -> prefilter rejected reason={reason}, deferring")
            return

        job = IdentificationJob(
            kind="resolve", track_id=track_id, frame=frame.copy(), box=detection.box
        )
        if self.worker.submit(job):
            self._reference_manager.mark_queued(track_id)
            print(f"[Coordinator] track_id={track_id} -> submitted resolve job")
        else:
            self._reference_manager.mark_queue_full(track_id, now, self.queue_full_retry_seconds)
            print(f"[Coordinator] track_id={track_id} -> job queue full, will retry shortly")

    def _maybe_capture(
        self,
        frame: NDArray[np.uint8],
        detection: Detection,
        item_id: int,
        now: float,
    ) -> None:
        """Submit one spaced additional-reference capture for a resolved item."""
        assert self.worker is not None
        track_id = detection.track_id
        if track_id is None:
            return
        last_capture_at = self._last_capture_at.get(item_id)
        if last_capture_at is not None and now - last_capture_at < self.capture_interval_seconds:
            return
        if not self._reference_manager.can_submit(track_id, now):
            return

        job = IdentificationJob(
            kind="capture", track_id=track_id, item_id=item_id, frame=frame.copy(), box=detection.box
        )
        if self.worker.submit(job):
            self._reference_manager.mark_queued(track_id)
        else:
            logger.warning("identification.queue_full track_id=%s kind=capture", track_id)

    def _apply_results(self, results: list[IdentificationResult]) -> None:
        """Turn completed background work into item/track state, on this thread."""
        for result in results:
            if result.track_id not in self._active_track_ids:
                logger.info(
                    "identification.stale_result track_id=%s kind=%s status=%s",
                    result.track_id,
                    result.kind,
                    result.status,
                )
                self._reference_manager.forget(result.track_id)
                continue

            if result.kind == "resolve":
                self._apply_resolve_result(result)
            else:
                self._apply_capture_result(result)

    def _apply_resolve_result(self, result: IdentificationResult) -> None:
        track_id = result.track_id
        now = self.clock()

        if result.status in ("pending", "error"):
            print(
                f"[Coordinator] track_id={track_id} -> resolve result status={result.status} "
                f"reason={result.failure_reason}, deferring for {self.bad_mask_cooldown_seconds}s"
            )
            self._reference_manager.mark_deferred(
                track_id, now, self.bad_mask_cooldown_seconds, result.failure_reason or result.status
            )
            return

        detection = self._last_detection_by_track.get(track_id)
        if detection is None:
            # The track vanished between submission and this result; nothing to bind.
            self._reference_manager.forget(track_id)
            return

        if result.status == "new":
            print(f"[Coordinator] track_id={track_id} -> creating NEW item (no gallery match)")
            add_signal = TrackSignal(
                signal_type=TrackSignalType.ADD, track_id=track_id, detection=detection
            )
            item, _ = self.event_engine.process_signal(add_signal)
            print(f"[Coordinator] track_id={track_id} -> item_id={item.id} created")
            self._reference_manager.mark_identified(track_id)
            if result.reference is not None:
                self._submit_capture(track_id, item.id, result.reference)
            return

        item_id = result.item_id
        if item_id is None:
            raise RuntimeError("An EXISTING identity result requires a permanent item_id")
        print(
            f"[Coordinator] track_id={track_id} -> EXISTING match item_id={item_id} "
            f"similarity={result.similarity:.4f}"
        )
        if self.event_engine.is_item_claimed(item_id, excluding_track_id=track_id):
            # Another visible track already owns this item; keep retrying instead.
            print(f"[Coordinator] track_id={track_id} -> item_id={item_id} already claimed, deferring")
            self._reference_manager.mark_deferred(
                track_id, now, self.bad_mask_cooldown_seconds, "item_already_claimed"
            )
            return

        item = self.store.get_item(item_id)
        if item is None:
            raise RuntimeError(f"Matched permanent item {item_id} no longer exists")

        if item.status is ItemStatus.REMOVED:
            print(f"[Coordinator] track_id={track_id} -> item_id={item_id} was REMOVED, marking RETURNED")
            return_signal = TrackSignal(
                signal_type=TrackSignalType.RETURNED,
                track_id=track_id,
                detection=detection,
                item_id=item_id,
            )
            self.event_engine.process_return(return_signal)
        else:
            print(
                f"[Coordinator] track_id={track_id} -> item_id={item_id} status={item.status.value}, "
                "associating without a new event"
            )
            self.event_engine.associate_existing_item(track_id, item_id)

        self._reference_manager.mark_identified(track_id)
        if result.reference is not None:
            self._submit_capture(track_id, item_id, result.reference)

    def _submit_capture(self, track_id: int, item_id: int, reference: PreparedReference) -> None:
        """Persist a reference already computed by a resolve job, off this thread."""
        assert self.worker is not None
        job = IdentificationJob(
            kind="capture", track_id=track_id, item_id=item_id, precomputed_reference=reference
        )
        if self.worker.submit(job):
            self._reference_manager.mark_queued(track_id)
        else:
            logger.warning("identification.queue_full track_id=%s kind=capture_initial", track_id)

    def _apply_capture_result(self, result: IdentificationResult) -> None:
        if result.status == "captured" and result.item_id is not None:
            self._last_capture_at[result.item_id] = self.clock()
        elif result.status == "error":
            logger.warning(
                "identification.capture_error track_id=%s reason=%s",
                result.track_id,
                result.failure_reason,
            )

        # The item is already identified regardless of this capture's outcome.
        self._reference_manager.mark_identified(result.track_id)
