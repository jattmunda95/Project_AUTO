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
- Apply cheap pre-SAM quality filtering (box area floor and ceiling, confidence,
  visible fraction of the predicted box, sharpness) before ever submitting a job.
  The ceiling rejects scenery the camera looks at (the table, a desk mat) by
  geometry; filtering by class name is deliberately avoided, since any class
  excluded by name can never be identified at all.
- Route stable MOVED signals to the event engine. REMOVE retires a temporary ID;
  RemovalPolicy owns item absence, ReID handoff deadlines, and permanent removal.
  MOVE_START/MOVE_END are runtime tracking transitions only: they arm reference
  learning here and are deliberately never forwarded to the event engine.

Owns the in-memory ReID gallery refresh trigger indirectly: the worker owns the
actual gallery snapshot (see identification_worker.py) since only it reads/writes
references. No tracker-stage inspection or direct database access beyond gallery-
free item/status lookups; the tracker only supplies signals and per-frame detections.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from time import monotonic

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
from project_auto.events.removal_policy import RemovalConfig, RemovalPolicy, center_distance
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
    # Fraction of the frame a detection may cover before it is treated as scenery
    # (a table surface, a desk mat, a wall) rather than a placeable object. 1.0
    # disables the ceiling.
    max_box_area_fraction: float = 1.0
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
    min_reference_consistency: float = 0.0
    job_queue_max_size: int = 8
    # Caps the worker's torch thread pool; see identification_worker.py.
    worker_torch_threads: int = 4
    removal_config: RemovalConfig = field(default_factory=RemovalConfig)
    # Optional ground-truth diagnostics for resolve jobs; None disables recording.
    diagnostics: ReidDiagnostics | None = None
    clock: Callable[[], float] = field(default=monotonic, repr=False)
    worker: IdentificationWorkerProtocol | None = None
    reference_policy: ReferencePolicy = field(init=False, repr=False)
    _reference_manager: ReferenceManager = field(init=False, repr=False)
    _active_track_ids: set[int] = field(default_factory=set, init=False, repr=False)
    # Monotonic frame counter: the reference policy schedules in frames, not seconds.
    _frame_index: int = field(default=0, init=False, repr=False)
    removal_policy: RemovalPolicy = field(init=False, repr=False)
    _visible: dict[int, Detection] = field(default_factory=dict, init=False, repr=False)
    _early_track_ids: set[int] = field(default_factory=set, init=False, repr=False)
    _confirmed_track_ids: set[int] = field(default_factory=set, init=False, repr=False)
    _handoff_targets: dict[int, set[int]] = field(default_factory=dict, init=False, repr=False)
    _handoff_job_ids: set[int] = field(default_factory=set, init=False, repr=False)
    # A pre-ADD handoff candidate is a fresh, low-confidence ID that routinely drops
    # out for a frame or two. It is only forgotten after being unseen for a grace
    # period, so its in-flight job and handoff target survive a flicker.
    _early_last_seen: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    # An accepted match whose candidate is briefly invisible waits here to be applied.
    _held_matches: dict[int, IdentificationResult] = field(
        default_factory=dict, init=False, repr=False
    )
    _inflight: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _next_job_id: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._reference_manager = ReferenceManager(clock=self.clock)
        self.removal_policy = RemovalPolicy(self.removal_config)
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
                handoff_queue_max_size=self.removal_config.max_inflight,
                min_mask_score=self.min_mask_score,
                min_mask_occupancy=self.min_mask_occupancy,
                reference_novelty_threshold=self.reference_novelty_threshold,
                min_reference_consistency=self.min_reference_consistency,
                worker_torch_threads=self.worker_torch_threads,
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
        retired = {
            signal.track_id for signal in signals if signal.signal_type is TrackSignalType.REMOVE
        }
        self._visible = {
            key: value for key, value in detections_by_track_id.items() if key not in retired
        }
        self._active_track_ids.difference_update(retired)
        self.removal_policy.observe(self._visible, retired, now, datetime.now(timezone.utc))
        # New IDs are eligible for handoff checks before normal ADD confirmation.
        bound_ids = {entry.track_id for entry in self.removal_policy.items.values()}
        self.removal_policy.remember_candidates(self._visible, bound_ids)
        for track_id in retired:
            self._confirmed_track_ids.discard(track_id)
            self._forget_track(track_id)
        for track_id in self._early_track_ids & self._visible.keys():
            self._early_last_seen[track_id] = now
        for track_id in list(self._early_track_ids - self._visible.keys()):
            unseen_for = now - self._early_last_seen.get(track_id, 0.0)
            if unseen_for >= self.removal_config.absence_seconds:
                self._forget_track(track_id)
        # The hard deadline applies even when an old result arrives in this frame.
        self._expire_missing(now, hard_only=True)
        self._schedule_handoffs(frame, now)
        self._apply_held_matches()
        self._apply_results(self.worker.poll_results())

        for signal in signals:
            if signal.signal_type is TrackSignalType.ADD:
                self._confirmed_track_ids.add(signal.track_id)
                was_early = signal.track_id in self._early_track_ids
                self._early_track_ids.discard(signal.track_id)
                if was_early and signal.track_id not in self._inflight:
                    self._reference_manager.forget(signal.track_id)
                self._active_track_ids.add(signal.track_id)
                self._try_submit_resolve(frame, signal.track_id, signal.detection, now)
            elif signal.signal_type is TrackSignalType.REMOVE:
                # Tracker retirement is not evidence that the physical item left.
                # Its binding stays reserved until handoff or item-level timeout.
                continue
            elif signal.signal_type is TrackSignalType.MOVE_START:
                # Runtime tracking transitions, never persisted ItemEvents: they only
                # arm reference learning, so they must not reach the event engine.
                self.reference_policy.on_move_start(signal.track_id, self._frame_index)
                item_id = self.event_engine.item_id_for_track(signal.track_id)
                if item_id in self.removal_policy.items:
                    self.removal_policy.items[item_id].moving = True
            elif signal.signal_type is TrackSignalType.MOVE_END:
                self.reference_policy.on_move_end(signal.track_id, self._frame_index)
            elif self.event_engine.item_id_for_track(signal.track_id) is not None:
                item_id = self.event_engine.item_id_for_track(signal.track_id)
                entry = self.removal_policy.items.get(item_id)
                if entry is None or entry.relocation_started_at is None:
                    self.event_engine.process_signal(signal)
                    if entry is not None:
                        entry.placement_box = signal.destination_box or signal.detection.box
                        entry.moving = False
            else:
                # Identity resolution for this track has not completed yet (it is
                # still QUEUED/PROCESSING/DEFERRED in the background worker), so
                # there is no item to attach this MOVED signal to. Drop it rather
                # than block or crash; the item's location is still accurate as
                # of its next confirmed placement once identity resolves.
                # TODO(coordinator): ordinary unresolved movement is still dropped;
                # handoff relocation has its own retained source box and settling.
                log_action(
                    Category.DEFER,
                    track=signal.track_id,
                    reason="unidentified",
                    signal=signal.signal_type.value,
                )

        for track_id in self._reference_manager.pending_track_ids():
            if track_id in self._early_track_ids:
                continue
            detection = detections_by_track_id.get(track_id)
            if detection is None:
                continue
            self._try_submit_resolve(frame, track_id, detection, now)

        for track_id, detection in detections_by_track_id.items():
            item_id = self.event_engine.item_id_for_track(track_id)
            if item_id is None:
                # The item timeout can precede tracker retirement. A returning
                # still-confirmed ID therefore needs ReID even without a new ADD.
                if track_id in self._confirmed_track_ids:
                    self._active_track_ids.add(track_id)
                    self._try_submit_resolve(frame, track_id, detection, now)
                continue
            self._active_track_ids.add(track_id)
            self._maybe_nominate_reference(frame, detection, item_id, now)

        self._finish_handoff_movements(now)
        self._expire_missing(now)

    def _forget_track(self, track_id: int) -> None:
        """Retire runtime work while leaving item ownership to the removal policy."""
        self._active_track_ids.discard(track_id)
        self._early_track_ids.discard(track_id)
        self._inflight.pop(track_id, None)
        self._handoff_targets.pop(track_id, None)
        self._early_last_seen.pop(track_id, None)
        held = self._held_matches.pop(track_id, None)
        if held is not None:
            self._record_outcome(held, None, "STALE")
        self._reference_manager.forget(track_id)
        self.reference_policy.forget(track_id)

    def _submit_job(self, job: IdentificationJob) -> bool:
        assert self.worker is not None
        self._next_job_id += 1
        job = replace(job, job_id=self._next_job_id)
        if not self.worker.submit(job):
            return False
        self._inflight[job.track_id] = job.job_id
        if job.priority:
            self._handoff_job_ids.add(job.job_id)
        return True

    def _schedule_handoffs(self, frame: NDArray[np.uint8], now: float) -> None:
        assert self.worker is not None
        for entry in sorted(self.removal_policy.items.values(), key=lambda entry: entry.item_id):
            if entry.missing_since is None:
                continue
            for track_id in sorted(entry.candidates - entry.attempted):
                if (
                    track_id not in self._visible
                    or self.event_engine.item_id_for_track(track_id) is not None
                ):
                    continue
                if track_id in self._handoff_targets:
                    self._handoff_targets[track_id].add(entry.item_id)
                    entry.attempted.add(track_id)
                    continue
                if len(self._handoff_job_ids) >= self.removal_config.max_inflight:
                    return
                self._handoff_targets[track_id] = {entry.item_id}
                job_id = self._inflight.get(track_id)
                if job_id is not None:
                    self.worker.promote(job_id)
                    self._handoff_job_ids.add(job_id)
                else:
                    if track_id not in self._active_track_ids:
                        self._early_track_ids.add(track_id)
                        self._early_last_seen[track_id] = now
                        self._active_track_ids.add(track_id)
                    self._try_submit_resolve(frame, track_id, self._visible[track_id], now)
                if track_id in self._inflight:
                    entry.attempted.add(track_id)
                    log_action(
                        Category.DEFER,
                        item=entry.item_id,
                        track=track_id,
                        reason="checking_handoff",
                    )
                else:
                    self._handoff_targets.pop(track_id, None)

    def _expire_missing(self, now: float, hard_only: bool = False) -> None:
        for item_id, entry in list(self.removal_policy.items.items()):
            checking = (
                hard_only
                # An outstanding check keeps the grace alive through a candidate
                # flicker; it ends when the candidate is forgotten or the hard
                # deadline passes, so it cannot hold an item forever.
                or any(item_id in targets for targets in self._handoff_targets.values())
                or any(
                    track_id in self._visible
                    and self.event_engine.item_id_for_track(track_id) is None
                    for track_id in entry.candidates - entry.attempted
                )
            )
            if not self.removal_policy.expired(item_id, now, checking):
                continue
            signal = TrackSignal(TrackSignalType.REMOVE, entry.track_id, entry.detection)
            self.event_engine.process_signal(signal)
            self.removal_policy.items.pop(item_id)
            self._forget_track(entry.track_id)
            for targets in self._handoff_targets.values():
                targets.discard(item_id)

    def _finish_handoff_movements(self, now: float) -> None:
        for item_id, entry in self.removal_policy.items.items():
            if not self.removal_policy.settled(item_id, now):
                continue
            box = entry.detection.box
            if (
                center_distance(entry.placement_box, box)
                > self.removal_config.movement_tolerance_pixels
            ):
                signal = TrackSignal(
                    TrackSignalType.MOVED,
                    entry.track_id,
                    entry.detection,
                    started_at=entry.relocation_started_at,
                    finished_at=datetime.now(timezone.utc),
                    source_box=entry.placement_box,
                    destination_box=box,
                )
                self.event_engine.process_signal(signal)
                entry.placement_box = box
            entry.relocation_started_at = None
            entry.moving = False
            self.reference_policy.on_move_end(entry.track_id, self._frame_index)

    def _passes_prefilter(self, frame: NDArray[np.uint8], detection: Detection) -> tuple[bool, str]:
        """Cheap reject before ever invoking SAM/DINO; returns (usable, reason)."""
        x1, y1, x2, y2 = detection.box
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return False, "invalid_crop_dimensions"
        if width * height < self.min_box_area:
            return False, "box_too_small"
        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_height * frame_width
        if frame_area > 0 and width * height > self.max_box_area_fraction * frame_area:
            # Scenery, not a placeable object: the table itself, a desk mat, a
            # chair back filling the view. Segmenting it wastes a worker slot and
            # produces an "item" that can never meaningfully be added or removed.
            return False, "box_too_large"
        if detection.confidence < self.min_detector_confidence:
            return False, "low_detector_confidence"
        # A box that merely reaches the image edge can still be almost entirely
        # visible, so judge the visible fraction of the predicted box instead of
        # rejecting every box that touches the boundary.
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
        if self.event_engine.item_id_for_track(track_id) is not None:
            return
        if not self._reference_manager.can_submit(track_id, now):
            return

        usable, reason = self._passes_prefilter(frame, detection)
        if not usable:
            self._reference_manager.mark_deferred(
                track_id, now, self.bad_mask_cooldown_seconds, reason
            )
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
            priority=bool(self._handoff_targets.get(track_id)),
        )
        if self._submit_job(job):
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
        if self._submit_job(job):
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
            current_job_id = self._inflight.get(result.track_id)
            self._handoff_job_ids.discard(result.job_id)
            if current_job_id is None or result.job_id != current_job_id:
                self._record_outcome(result, None, "STALE")
                log_action(
                    Category.DEFER, track=result.track_id, reason="stale_job", job=result.job_id
                )
                continue
            self._inflight.pop(result.track_id, None)
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

            if self._should_hold_match(result):
                self._held_matches[result.track_id] = result
                continue
            if result.kind == "resolve":
                self._apply_resolve_result(result)
            else:
                self._apply_capture_result(result)

    def _should_hold_match(self, result: IdentificationResult) -> bool:
        """Whether a handoff match arrived while its candidate is briefly not visible."""
        return (
            result.kind == "resolve"
            and result.status == "existing"
            and result.track_id in self._early_track_ids
            and result.track_id not in self._visible
            and result.item_id in self._handoff_targets.get(result.track_id, ())
        )

    def _apply_held_matches(self) -> None:
        """Apply held matches once their candidate is visible again."""
        for track_id in list(self._held_matches):
            if track_id in self._visible:
                self._apply_resolve_result(self._held_matches.pop(track_id))

    def _record_outcome(
        self, result: IdentificationResult, item_id: int | None, event: str
    ) -> None:
        """Record what this thread did with one resolve result, if diagnostics are on."""
        if self.diagnostics is None or result.kind != "resolve" or result.query_id is None:
            return
        self.diagnostics.record_outcome(result.query_id, item_id, event)

    def _apply_resolve_result(self, result: IdentificationResult) -> None:
        track_id = result.track_id
        now = self.clock()
        targets = self._handoff_targets.pop(track_id, set())
        entry = self.removal_policy.items.get(result.item_id)
        if (
            result.status == "existing"
            and result.item_id in targets
            and entry is not None
            and entry.missing_since is not None
            and track_id in entry.candidates
            and track_id in self._visible
            and self.event_engine.item_id_for_track(track_id) is None
        ):
            old_track_id = entry.track_id
            detection = self._visible[track_id]
            self.event_engine.transfer_item(entry.item_id, old_track_id, track_id)
            self.removal_policy.handoff(entry.item_id, detection, datetime.now(timezone.utc))
            self._forget_track(old_track_id)
            self._reference_manager.mark_identified(track_id)
            self.reference_policy.on_item_associated(track_id)
            self._record_outcome(result, entry.item_id, "HANDOFF")
            # A handoff match does not automatically teach its crop to the gallery.
            # Normal movement/settling capture still passes all reference gates.
            return

        if track_id in self._early_track_ids:
            # An early probe can only transfer a verified existing item. NEW and
            # unrelated matches must wait for normal tracker ADD confirmation.
            self._record_outcome(result, result.item_id, "DEFER_HANDOFF")
            self._reference_manager.mark_deferred(
                track_id, now, self.bad_mask_cooldown_seconds, "handoff_unconfirmed"
            )
            return

        if result.status in ("pending", "error"):
            self._record_outcome(result, None, "ERROR" if result.status == "error" else "DEFER")
            log_action(
                Category.DEFER,
                track=track_id,
                reason=result.failure_reason or result.status,
                cooldown_s=self.bad_mask_cooldown_seconds,
            )
            self._reference_manager.mark_deferred(
                track_id,
                now,
                self.bad_mask_cooldown_seconds,
                result.failure_reason or result.status,
            )
            return

        detection = self._visible.get(track_id)
        if detection is None:
            # The submitted crop is historical evidence. A currently missing
            # detection must not create/return an item, even before ID retirement.
            self._record_outcome(result, None, "DROPPED")
            log_action(
                Category.DEFER,
                track=track_id,
                reason="not_visible",
                kind=result.kind,
                status=result.status,
            )
            self._reference_manager.mark_deferred(
                track_id, now, self.queue_full_retry_seconds, "not_visible"
            )
            return

        if result.status == "new":
            # reid.py already logged the NEW decision; event_engine.process_add logs the
            # persisted EVENT once the item actually exists.
            add_signal = TrackSignal(
                signal_type=TrackSignalType.ADD, track_id=track_id, detection=detection
            )
            item, _ = self.event_engine.process_signal(add_signal)
            self.removal_policy.bind(item.id, detection)
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

        placement_box = (
            tuple(item.current_box)
            if item.current_box and item.status is not ItemStatus.REMOVED
            else detection.box
        )
        self.removal_policy.bind(item_id, detection, placement_box)

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
        if self._submit_job(job):
            self._reference_manager.mark_queued(track_id)
        else:
            log_action(Category.DEFER, track=track_id, reason="queue_full", kind="capture_initial")

    def _apply_capture_result(self, result: IdentificationResult) -> None:
        # An "error" status here was already logged as ERROR in identification_worker.py
        # when the exception was caught; no need to log it twice.
        if result.candidate_kind is not None:
            if result.status == "captured":
                self.reference_policy.on_candidate_accepted(result.track_id, result.candidate_kind)
            else:
                self.reference_policy.on_candidate_rejected(result.track_id, result.candidate_kind)

        # The item is already identified regardless of this capture's outcome.
        self._reference_manager.mark_identified(result.track_id)
