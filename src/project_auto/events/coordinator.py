"""Coordinate scene processing, identity resolution, and reference capture.

Subfunctions:
- Submit a lightweight IdentificationJob for each newly confirmed (ADD) track and
  return immediately; the real-time loop never waits on SAM/DINO/matching/persistence.
- Drain completed IdentificationResults every frame and apply them: create/associate
  permanent items, route NEW/EXISTING through the event layer, and drive the
  event-driven reference policy, all on the main thread.
- Own ReferenceManager so a track never has more than one outstanding job, and a
  bad result (unusable mask, full queue, claimed item) defers retry with a cooldown
  instead of resubmitting every frame.
- Own ReferencePolicy, which decides which frames are worth expensive reference
  work. A static item captures nothing: after its baseline references, only
  MOVE_START/MOVE_END re-arm learning, so new appearances are what earn references.
- Apply cheap pre-SAM quality filtering (box area, confidence, visible fraction of
  the predicted box, sharpness) before ever submitting a job.
- Route MOVED/REMOVE signals straight to the event engine; a REMOVE also forgets
  the retired track's identification state so a reused track ID starts fresh.
  MOVE_START/MOVE_END are runtime tracking transitions only: they arm reference
  learning here and are deliberately never forwarded to the event engine.

Owns the in-memory ReID gallery refresh trigger indirectly: the worker owns the
actual gallery snapshot (see identification_worker.py) since only it reads/writes
references. No tracker-stage inspection or direct database access beyond gallery-
free item/status lookups; the tracker only supplies signals and per-frame detections.
"""

from __future__ import annotations

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
from project_auto.events.reference_policy import (
    CandidateKind,
    ReferencePolicy,
)
from project_auto.memory.models import ItemStatus
from project_auto.memory.store import DatabaseStore
from project_auto.perception.descriptors import (
    clip_box_to_frame,
    frame_visibility_ratio,
    sharpness,
)
from project_auto.perception.detector import Detection
from project_auto.perception.scene_processor import PreparedReference, SceneProcessor
from project_auto.perception.tracker import TrackSignal, TrackSignalType
from project_auto.utils.logging import Category, log_action
from project_auto.utils.reid_diagnostics import ReidDiagnostics


@dataclass(slots=True)
class IdentityCoordinator:
    """Submit identification work asynchronously and apply results as they land."""

    scene_processor: SceneProcessor
    event_engine: EventEngine
    store: DatabaseStore
    reid_model_name: str
    max_references_per_item: int
    bad_mask_cooldown_seconds: float = 5.0
    queue_full_retry_seconds: float = 0.5
    min_box_area: int = 0
    min_detector_confidence: float = 0.0
    # Reference-capture policy timing, in frames (see events/reference_policy.py).
    initial_reference_count: int = 2
    initial_capture_spacing_frames: int = 15
    move_candidate_delay_frames: int = 5
    candidate_retry_frames: int = 10
    max_movement_reference_attempts: int = 3
    # Cheap gate, applied on the video thread to nominated candidates only.
    min_reference_sharpness: float = 0.0
    min_reference_frame_visibility: float = 0.0
    # Expensive gate, forwarded to the worker and applied after SAM/DINO.
    min_mask_score: float = 0.0
    min_mask_occupancy: float = 0.0
    reference_novelty_threshold: float = 1.0
    job_queue_max_size: int = 8
    # Optional ground-truth diagnostics for resolve jobs; None disables recording.
    diagnostics: ReidDiagnostics | None = None
    clock: Callable[[], float] = field(default=monotonic, repr=False)
    worker: IdentificationWorkerProtocol | None = None
    reference_policy: ReferencePolicy = field(init=False, repr=False)
    _reference_manager: ReferenceManager = field(init=False, repr=False)
    _active_track_ids: set[int] = field(default_factory=set, init=False, repr=False)
    _last_detection_by_track: dict[int, Detection] = field(
        default_factory=dict, init=False, repr=False
    )
    # Monotonic frame counter: the reference policy schedules in frames, not seconds.
    _frame_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._reference_manager = ReferenceManager(clock=self.clock)
        self.reference_policy = ReferencePolicy(
            initial_reference_count=self.initial_reference_count,
            initial_capture_spacing_frames=self.initial_capture_spacing_frames,
            move_candidate_delay_frames=self.move_candidate_delay_frames,
            candidate_retry_frames=self.candidate_retry_frames,
            max_movement_reference_attempts=self.max_movement_reference_attempts,
        )
        if self.worker is None:
            self.worker = IdentificationWorker(
                scene_processor=self.scene_processor,
                store=self.store,
                reid_model_name=self.reid_model_name,
                max_references_per_item=self.max_references_per_item,
                job_queue_max_size=self.job_queue_max_size,
                min_mask_score=self.min_mask_score,
                min_mask_occupancy=self.min_mask_occupancy,
                reference_novelty_threshold=self.reference_novelty_threshold,
                diagnostics=self.diagnostics,
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
        self._frame_index += 1
        self.event_engine.frame_size = (frame.shape[1], frame.shape[0])
        self._apply_results(self.worker.poll_results())

        for signal in signals:
            if signal.signal_type is TrackSignalType.ADD:
                self._active_track_ids.add(signal.track_id)
                self._last_detection_by_track[signal.track_id] = signal.detection
                self._try_submit_resolve(frame, signal.track_id, signal.detection, now)
            elif signal.signal_type is TrackSignalType.REMOVE:
                self._active_track_ids.discard(signal.track_id)
                self._reference_manager.forget(signal.track_id)
                self.reference_policy.forget(signal.track_id)
                self._last_detection_by_track.pop(signal.track_id, None)
                # A track that retired while still unresolved never got a permanent item.
                if self.event_engine.item_id_for_track(signal.track_id) is not None:
                    self.event_engine.process_signal(signal)
            elif signal.signal_type is TrackSignalType.MOVE_START:
                # Runtime tracking transitions, never persisted ItemEvents: they only
                # arm reference learning, so they must not reach the event engine.
                self.reference_policy.on_move_start(signal.track_id, self._frame_index)
            elif signal.signal_type is TrackSignalType.MOVE_END:
                self.reference_policy.on_move_end(signal.track_id, self._frame_index)
            elif self.event_engine.item_id_for_track(signal.track_id) is not None:
                self.event_engine.process_signal(signal)
            else:
                # Identity resolution for this track has not completed yet (it is
                # still QUEUED/PROCESSING/DEFERRED in the background worker), so
                # there is no item to attach this MOVED signal to. Drop it rather
                # than block or crash; the item's location is still accurate as
                # of its next confirmed placement once identity resolves.
                # TODO(coordinator): review whether this dropped signal should instead be
                # buffered and replayed once identity resolves, and whether stale results
                # need generation-aware correlation to a track (e.g. a retired-then-reused
                # track ID) rather than being applied unconditionally in _apply_results.
                log_action(
                    Category.DEFER,
                    track=signal.track_id,
                    reason="unidentified",
                    signal=signal.signal_type.value,
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
            self._maybe_nominate_reference(frame, detection, item_id, now)

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
        # A box that merely reaches the image edge can still be almost entirely
        # visible, so judge the visible fraction of the predicted box instead of
        # rejecting every box that touches the boundary.
        frame_height, frame_width = frame.shape[:2]
        visibility = frame_visibility_ratio(detection.box, (frame_width, frame_height))
        if visibility < self.min_reference_frame_visibility:
            return False, "low_frame_visibility"
        return True, ""

    def _passes_cheap_reference_checks(
        self, frame: NDArray[np.uint8], detection: Detection
    ) -> tuple[bool, str]:
        """Stage A gate: measurable, inexpensive checks only, no SAM and no DINO.

        Runs on the video thread, so it only ever runs for a frame the reference
        policy actually nominated, never for every frame of a tracked object.
        """
        usable, reason = self._passes_prefilter(frame, detection)
        if not usable:
            return False, reason

        frame_height, frame_width = frame.shape[:2]
        visible_box = clip_box_to_frame(detection.box, (frame_width, frame_height))
        if visible_box is None:
            return False, "outside_frame"

        x1, y1, x2, y2 = visible_box
        if sharpness(frame[y1:y2, x1:x2]) < self.min_reference_sharpness:
            return False, "blurred"
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
            log_action(Category.REJECT, track=track_id, stage="prefilter", reason=reason)
            return

        job = IdentificationJob(
            kind="resolve",
            track_id=track_id,
            frame=frame.copy(),
            box=detection.box,
            detection=detection,
            query_id=self.diagnostics.new_query_id() if self.diagnostics is not None else None,
            frame_index=self._frame_index,
        )
        if self.worker.submit(job):
            self._reference_manager.mark_queued(track_id)
            log_action(Category.QUEUE, track=track_id, action="submitted", kind="resolve")
        else:
            self._reference_manager.mark_queue_full(track_id, now, self.queue_full_retry_seconds)
            log_action(Category.DEFER, track=track_id, reason="queue_full", kind="resolve")

    def _maybe_nominate_reference(
        self,
        frame: NDArray[np.uint8],
        detection: Detection,
        item_id: int,
        now: float,
    ) -> None:
        """Ask the policy whether this frame is worth expensive reference work.

        A static item nominates nothing: the policy only offers candidates for the
        remaining baseline references and for frames earned by movement, so the
        normal per-frame cost here is one dictionary lookup.
        """
        assert self.worker is not None
        track_id = detection.track_id
        if track_id is None:
            return
        if not self._reference_manager.can_submit(track_id, now):
            return

        decision = self.reference_policy.should_nominate(track_id, self._frame_index)
        if not decision.nominate or decision.kind is None:
            return

        usable, reason = self._passes_cheap_reference_checks(frame, detection)
        if not usable:
            # Stage A rejected it, so SAM and DINO never run for this candidate.
            self.reference_policy.on_candidate_rejected(track_id, decision.kind)
            log_action(
                Category.REJECT,
                track=track_id,
                item=item_id,
                stage="reference_cheap",
                kind=decision.kind.value,
                reason=reason,
            )
            return

        job = IdentificationJob(
            kind="capture",
            track_id=track_id,
            item_id=item_id,
            frame=frame.copy(),
            box=detection.box,
            candidate_kind=decision.kind,
        )
        if self.worker.submit(job):
            self._reference_manager.mark_queued(track_id)
            log_action(
                Category.QUEUE,
                track=track_id,
                item=item_id,
                action="submitted",
                kind=f"capture:{decision.kind.value}",
            )
        else:
            log_action(Category.DEFER, track=track_id, reason="queue_full", kind="capture")

    def _apply_results(self, results: list[IdentificationResult]) -> None:
        """Turn completed background work into item/track state, on this thread."""
        for result in results:
            if result.track_id not in self._active_track_ids:
                log_action(
                    Category.DEFER,
                    track=result.track_id,
                    reason="stale_result",
                    kind=result.kind,
                    status=result.status,
                )
                self._record_outcome(result, None, "STALE")
                self._reference_manager.forget(result.track_id)
                continue

            if result.kind == "resolve":
                self._apply_resolve_result(result)
            else:
                self._apply_capture_result(result)

    def _record_outcome(self, result: IdentificationResult, item_id: int | None, event: str) -> None:
        """Record what this thread did with one resolve result, if diagnostics are on."""
        if self.diagnostics is None or result.kind != "resolve" or result.query_id is None:
            return
        self.diagnostics.record_outcome(result.query_id, item_id, event)

    def _apply_resolve_result(self, result: IdentificationResult) -> None:
        track_id = result.track_id
        now = self.clock()

        if result.status in ("pending", "error"):
            self._record_outcome(result, None, "ERROR" if result.status == "error" else "DEFER")
            log_action(
                Category.DEFER,
                track=track_id,
                reason=result.failure_reason or result.status,
                cooldown_s=self.bad_mask_cooldown_seconds,
            )
            self._reference_manager.mark_deferred(
                track_id, now, self.bad_mask_cooldown_seconds, result.failure_reason or result.status
            )
            return

        detection = self._last_detection_by_track.get(track_id)
        if detection is None:
            # The track vanished between submission and this result; nothing to bind.
            self._record_outcome(result, None, "DROPPED")
            self._reference_manager.forget(track_id)
            return

        if result.status == "new":
            # reid.py already logged the NEW decision; event_engine.process_add logs the
            # persisted EVENT once the item actually exists.
            add_signal = TrackSignal(
                signal_type=TrackSignalType.ADD, track_id=track_id, detection=detection
            )
            item, _ = self.event_engine.process_signal(add_signal)
            self._record_outcome(result, item.id, "ADDED")
            self._reference_manager.mark_identified(track_id)
            # This resolve embedding becomes baseline reference 1 at no extra cost;
            # the policy schedules whatever baseline references still remain.
            self.reference_policy.on_item_created(track_id, self._frame_index)
            if result.reference is not None:
                self._submit_capture(track_id, item.id, result.reference)
            return

        # reid.py already logged the MATCH decision with scores; only the branch below
        # (RETURNED event vs silent ASSOC) is decided here.
        item_id = result.item_id
        if item_id is None:
            raise RuntimeError("An EXISTING identity result requires a permanent item_id")
        if self.event_engine.is_item_claimed(item_id, excluding_track_id=track_id):
            # Another visible track already owns this item; keep retrying instead.
            log_action(Category.DEFER, track=track_id, item=item_id, reason="item_already_claimed")
            self._record_outcome(result, item_id, "DEFER_CLAIMED")
            self._reference_manager.mark_deferred(
                track_id, now, self.bad_mask_cooldown_seconds, "item_already_claimed"
            )
            return

        item = self.store.get_item(item_id)
        if item is None:
            raise RuntimeError(f"Matched permanent item {item_id} no longer exists")

        if item.status is ItemStatus.REMOVED:
            return_signal = TrackSignal(
                signal_type=TrackSignalType.RETURNED,
                track_id=track_id,
                detection=detection,
                item_id=item_id,
            )
            self.event_engine.process_return(return_signal)
            self._record_outcome(result, item_id, "RETURNED")
        else:
            self.event_engine.associate_existing_item(track_id, item_id)
            self._record_outcome(result, item_id, "ASSOC")

        self._reference_manager.mark_identified(track_id)
        # An already-known item keeps its existing baseline and learns only from
        # movement, so it starts at rest rather than scheduling baseline captures.
        self.reference_policy.on_item_associated(track_id)
        if result.reference is not None:
            self._submit_capture(track_id, item_id, result.reference)

    def _submit_capture(self, track_id: int, item_id: int, reference: PreparedReference) -> None:
        """Persist a reference already computed by a resolve job, off this thread."""
        assert self.worker is not None
        job = IdentificationJob(
            kind="capture",
            track_id=track_id,
            item_id=item_id,
            precomputed_reference=reference,
            candidate_kind=CandidateKind.INITIAL,
        )
        if self.worker.submit(job):
            self._reference_manager.mark_queued(track_id)
        else:
            log_action(Category.DEFER, track=track_id, reason="queue_full", kind="capture_initial")

    def _apply_capture_result(self, result: IdentificationResult) -> None:
        # An "error" status here was already logged as ERROR in identification_worker.py
        # when the exception was caught; no need to log it twice.
        if result.candidate_kind is not None:
            if result.status == "captured":
                self.reference_policy.on_candidate_accepted(
                    result.track_id, result.candidate_kind
                )
            else:
                self.reference_policy.on_candidate_rejected(
                    result.track_id, result.candidate_kind
                )

        # The item is already identified regardless of this capture's outcome.
        self._reference_manager.mark_identified(result.track_id)
