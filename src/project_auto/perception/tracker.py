"""Track structured detections across consecutive video frames.

Subfunctions:
- Index detections by temporary source-track ID and maintain candidate/confirmed state.
- Confirm persistent candidates and emit the current one-time ADD confirmation signal.
- Detect stable placement, movement, stopping, disappearance, and removal.
- Return lifecycle signals to app.py without database writes or image inference.

Temporal confirmation does not establish permanent identity. Planned app coordination
will resolve identity before creating an item. Never call scene_processor, SAM, or ReID
from this tracker; RETURNED resolution belongs outside it. The return stub is inactive.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import hypot
from time import monotonic
from typing import Callable

from project_auto.perception.detector import Detection


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for persisted lifecycle events."""
    return datetime.now(timezone.utc)


def _make_buffer_box(
    box: tuple[int, int, int, int],
    scale: float,
) -> tuple[float, float, float, float]:
    """Return a centered buffer scaled from one detection bounding box."""
    if scale <= 1.0:
        raise ValueError("movement_buffer_scale must be greater than 1")

    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    half_buffer_width = (x2 - x1) * scale / 2
    half_buffer_height = (y2 - y1) * scale / 2

    return (
        center_x - half_buffer_width,
        center_y - half_buffer_height,
        center_x + half_buffer_width,
        center_y + half_buffer_height,
    )


def _is_box_inside(
    box: tuple[int, int, int, int],
    buffer_box: tuple[float, float, float, float],
) -> bool:
    """Return whether every edge of a detection is inside its placement buffer."""
    x1, y1, x2, y2 = box
    buffer_x1, buffer_y1, buffer_x2, buffer_y2 = buffer_box

    return (
        x1 >= buffer_x1
        and y1 >= buffer_y1
        and x2 <= buffer_x2
        and y2 <= buffer_y2
    )


def _box_center_displacement(
    reference_box: tuple[int, int, int, int],
    current_box: tuple[int, int, int, int],
) -> float:
    """Return the Euclidean distance between two bounding-box centers."""
    reference_x1, reference_y1, reference_x2, reference_y2 = reference_box
    current_x1, current_y1, current_x2, current_y2 = current_box
    reference_center_x = (reference_x1 + reference_x2) / 2
    reference_center_y = (reference_y1 + reference_y2) / 2
    current_center_x = (current_x1 + current_x2) / 2
    current_center_y = (current_y1 + current_y2) / 2

    return hypot(
        current_center_x - reference_center_x,
        current_center_y - reference_center_y,
    )


class TrackStatus(str, Enum):
    """Lifecycle states for a temporary track."""

    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    STABLE = "stable"
    MOVING = "moving"
    MISSING = "missing"


class TrackSignalType(str, Enum):
    """Meaningful lifecycle signals emitted for downstream decisions."""

    ADD = "add"
    MOVED = "moved"
    REMOVE = "remove"
    # TODO(ReID): Keep RETURNED emission inactive until associative memory resolves
    # an observation to a permanent item_id; a tracker ID alone is insufficient.
    RETURNED = "returned"
    # Runtime tracking transitions, not persisted ItemEvents. MOVE_START fires once
    # when a STABLE track exits its placement buffer; MOVE_END fires once when a
    # MOVING track re-stabilizes (in the same frame as, and immediately before, any
    # MOVED signal). MOVED remains the authoritative persisted "meaningful location
    # change" semantic; a MOVE_START/MOVE_END pair can occur without MOVED ever
    # applying should the caller decide the net displacement was not meaningful.
    MOVE_START = "move_start"
    MOVE_END = "move_end"


@dataclass(frozen=True, slots=True)
class TrackSignal:
    """One meaningful tracker result for the state machine."""

    signal_type: TrackSignalType
    track_id: int
    detection: Detection
    # TODO(ReID): Require this permanent identity when constructing RETURNED;
    # leave it unset for tracker-only lifecycle signals.
    item_id: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    source_box: tuple[int, int, int, int] | None = None
    destination_box: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class _ActiveTrack:
    """Mutable frame-to-frame state for one temporary track."""

    detection: Detection
    confirmation_started_at: float
    candidate_missing_frames: int = 0
    status: TrackStatus = TrackStatus.CANDIDATE
    missing_since: float | None = None
    status_before_missing: TrackStatus | None = None
    stable_box: tuple[int, int, int, int] | None = None
    buffer_box: tuple[float, float, float, float] | None = None
    movement_started_at: datetime | None = None
    stability_reference_box: tuple[int, int, int, int] | None = None
    stopped_since: float | None = None
    stopped_at: datetime | None = None


@dataclass(slots=True)
class DetectionTracker:
    """Hold lifecycle state for BoT-SORT tracks across video frames."""

    candidate_confirmation_seconds: float = 2.0
    max_candidate_missing_frames: int = 15
    removal_timeout_seconds: float = 2.0
    movement_buffer_scale: float = 1.2
    movement_stop_tolerance_pixels: float = 5.0
    movement_stopped_confirmation_seconds: float = 1.0
    clock: Callable[[], float] = field(default=monotonic, repr=False)
    timestamp_clock: Callable[[], datetime] = field(default=_utc_now, repr=False)
    _tracks: dict[int, _ActiveTrack] = field(
        default_factory=dict,
        init=False,
    )

    @staticmethod
    def _index_detections(detections: list[Detection]) -> dict[int, Detection]:
        """Return tracked detections keyed by their BoT-SORT IDs."""
        return {
            detection.track_id: detection
            for detection in detections
            if detection.track_id is not None
        }

    def _check_return(self, detection: Detection) -> None:
        """Reserve a ReID-backed return check without activating it yet."""
        # TODO(identity): Retire this inactive stub when coordinator routing is added.
        # The app requests scene processing after confirmation; this tracker must
        # never call ReID, SAM, or the database. The event layer resolves RETURNED.
        pass

    def _update_add_lifecycle(
        self,
        track_id: int,
        detection: Detection,
        now: float,
    ) -> list[TrackSignal]:
        """Emit ADD after one visible candidate completes its timed attempt."""
        active_track = self._tracks.get(track_id)
        if active_track is None:
            active_track = _ActiveTrack(
                detection=detection,
                confirmation_started_at=now,
            )
            self._tracks[track_id] = active_track
        elif active_track.status is TrackStatus.MISSING:
            return self._restore_missing_track(track_id, active_track, detection)
        elif active_track.status is TrackStatus.STABLE:
            signal = self._update_stable_track(track_id, active_track, detection)
            return [signal] if signal is not None else []
        elif active_track.status is TrackStatus.MOVING:
            return self._update_moving_track(
                track_id,
                active_track,
                detection,
                now,
            )
        else:
            active_track.detection = detection

        if (
            active_track.status is TrackStatus.CANDIDATE
            and now - active_track.confirmation_started_at
            >= self.candidate_confirmation_seconds
        ):
            active_track.status = TrackStatus.STABLE
            active_track.stable_box = detection.box
            active_track.buffer_box = _make_buffer_box(
                detection.box,
                self.movement_buffer_scale,
            )
            return [
                TrackSignal(
                    signal_type=TrackSignalType.ADD,
                    track_id=track_id,
                    detection=detection,
                )
            ]

        return []

    def _update_stable_track(
        self,
        track_id: int,
        active_track: _ActiveTrack,
        detection: Detection,
    ) -> TrackSignal | None:
        """Refresh a stable track and begin movement after its buffer is exited."""
        active_track.detection = detection
        if active_track.buffer_box is None:
            raise ValueError("A stable track requires a placement buffer")
        if _is_box_inside(detection.box, active_track.buffer_box):
            return None

        active_track.status = TrackStatus.MOVING
        active_track.movement_started_at = self.timestamp_clock()
        active_track.stability_reference_box = detection.box
        active_track.stopped_since = None
        active_track.stopped_at = None

        return TrackSignal(
            signal_type=TrackSignalType.MOVE_START,
            track_id=track_id,
            detection=detection,
        )

    def _update_moving_track(
        self,
        track_id: int,
        active_track: _ActiveTrack,
        detection: Detection,
        now: float,
    ) -> list[TrackSignal]:
        """Emit MOVE_END then MOVED after a visible moving track stops long enough."""
        reference_box = active_track.stability_reference_box
        if reference_box is None:
            raise ValueError("A moving track requires a stability reference box")

        active_track.detection = detection
        displacement = _box_center_displacement(reference_box, detection.box)
        if displacement > self.movement_stop_tolerance_pixels:
            active_track.stability_reference_box = detection.box
            active_track.stopped_since = None
            active_track.stopped_at = None
            return []

        if active_track.stopped_since is None:
            active_track.stopped_since = now
            active_track.stopped_at = self.timestamp_clock()
            return []

        elapsed = now - active_track.stopped_since
        if elapsed < self.movement_stopped_confirmation_seconds:
            return []
        if active_track.movement_started_at is None:
            raise ValueError("A moving track requires a movement start timestamp")
        if active_track.stopped_at is None:
            raise ValueError("A stopped track requires a stop timestamp")
        if active_track.stable_box is None:
            raise ValueError("A moving track requires a source placement box")

        move_end_signal = TrackSignal(
            signal_type=TrackSignalType.MOVE_END,
            track_id=track_id,
            detection=detection,
        )
        moved_signal = TrackSignal(
            signal_type=TrackSignalType.MOVED,
            track_id=track_id,
            detection=detection,
            started_at=active_track.movement_started_at,
            finished_at=active_track.stopped_at,
            source_box=active_track.stable_box,
            destination_box=detection.box,
        )
        active_track.status = TrackStatus.STABLE
        active_track.stable_box = detection.box
        active_track.buffer_box = _make_buffer_box(
            detection.box,
            self.movement_buffer_scale,
        )
        active_track.movement_started_at = None
        active_track.stability_reference_box = None
        active_track.stopped_since = None
        active_track.stopped_at = None

        return [move_end_signal, moved_signal]

    def _restore_missing_track(
        self,
        track_id: int,
        active_track: _ActiveTrack,
        detection: Detection,
    ) -> list[TrackSignal]:
        """Restore a visible track to the lifecycle state preceding its absence."""
        restored_status = active_track.status_before_missing or TrackStatus.STABLE
        active_track.status = restored_status
        active_track.status_before_missing = None
        active_track.missing_since = None

        if restored_status is TrackStatus.STABLE:
            # The track may have drifted outside its buffer while undetected; this
            # can itself be a genuine STABLE -> MOVING transition, so propagate it.
            signal = self._update_stable_track(track_id, active_track, detection)
            return [signal] if signal is not None else []

        active_track.detection = detection
        active_track.stability_reference_box = detection.box
        active_track.stopped_since = None
        active_track.stopped_at = None
        return []

    def _update_missing_lifecycle(
        self,
        track_id: int,
        now: float,
    ) -> TrackSignal | None:
        """Update one absent track and emit REMOVE after its confirmed timeout."""
        active_track = self._tracks[track_id]

        if active_track.status is TrackStatus.CANDIDATE:
            active_track.candidate_missing_frames += 1
            if (
                active_track.candidate_missing_frames
                > self.max_candidate_missing_frames
            ):
                del self._tracks[track_id]
            return None

        if active_track.status in {
            TrackStatus.CONFIRMED,
            TrackStatus.STABLE,
            TrackStatus.MOVING,
        }:
            active_track.status_before_missing = active_track.status
            active_track.status = TrackStatus.MISSING
            active_track.missing_since = now
            return None

        if active_track.missing_since is None:
            active_track.missing_since = now
            return None

        elapsed = now - active_track.missing_since
        if elapsed < self.removal_timeout_seconds:
            return None

        signal = TrackSignal(
            signal_type=TrackSignalType.REMOVE,
            track_id=track_id,
            detection=active_track.detection,
        )
        del self._tracks[track_id]
        return signal

    def update(self, detections: list[Detection]) -> list[TrackSignal]:
        """Update track lifecycles and return newly confirmed meaningful signals."""
        if self.candidate_confirmation_seconds <= 0:
            raise ValueError("candidate_confirmation_seconds must be positive")
        if self.max_candidate_missing_frames < 0:
            raise ValueError("max_candidate_missing_frames must be non-negative")
        if self.removal_timeout_seconds <= 0:
            raise ValueError("removal_timeout_seconds must be positive")
        if self.movement_buffer_scale <= 1.0:
            raise ValueError("movement_buffer_scale must be greater than 1")
        if self.movement_stop_tolerance_pixels < 0:
            raise ValueError("movement_stop_tolerance_pixels must be non-negative")
        if self.movement_stopped_confirmation_seconds <= 0:
            raise ValueError(
                "movement_stopped_confirmation_seconds must be positive"
            )

        tracked_detections = self._index_detections(detections)
        now = self.clock()
        signals: list[TrackSignal] = []

        for track_id in set(self._tracks) - set(tracked_detections):
            signal = self._update_missing_lifecycle(track_id, now)
            if signal is not None:
                signals.append(signal)

        for track_id, detection in tracked_detections.items():
            signals.extend(self._update_add_lifecycle(track_id, detection, now))

        return signals
