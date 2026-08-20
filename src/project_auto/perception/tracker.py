"""Track structured detections across consecutive video frames."""

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Callable

from project_auto.perception.detector import Detection


class TrackStatus(str, Enum):
    """Lifecycle states for a temporary track."""

    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    MISSING = "missing"


class TrackSignalType(str, Enum):
    """Meaningful lifecycle signals emitted for downstream decisions."""

    ADD = "add"
    REMOVE = "remove"


@dataclass(frozen=True, slots=True)
class TrackSignal:
    """One meaningful tracker result for the state machine."""

    signal_type: TrackSignalType
    track_id: int
    detection: Detection


@dataclass(slots=True)
class _ActiveTrack:
    """Mutable frame-to-frame state for one temporary track."""

    detection: Detection
    seen_frames: int = 1
    missed_frames: int = 0
    status: TrackStatus = TrackStatus.CANDIDATE
    missing_since: float | None = None


@dataclass(slots=True)
class DetectionTracker:
    """Hold lifecycle state for BoT-SORT tracks across video frames."""

    confirmation_frames: int = 30
    max_missed_frames: int = 10
    removal_timeout_seconds: float = 2.0
    clock: Callable[[], float] = field(default=monotonic, repr=False)
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

    def _update_add_lifecycle(
        self,
        track_id: int,
        detection: Detection,
    ) -> TrackSignal | None:
        """Update one visible track and emit ADD when its candidate is confirmed."""
        active_track = self._tracks.get(track_id)
        if active_track is None:
            active_track = _ActiveTrack(detection=detection)
            self._tracks[track_id] = active_track
        elif active_track.status in {TrackStatus.CONFIRMED, TrackStatus.MISSING}:
            self._update_confirmed_track(active_track, detection)
            return None
        else:
            active_track.detection = detection
            active_track.missed_frames = 0
            active_track.seen_frames += 1

        if (
            active_track.status is TrackStatus.CANDIDATE
            and active_track.seen_frames >= self.confirmation_frames
        ):
            active_track.status = TrackStatus.CONFIRMED
            return TrackSignal(
                signal_type=TrackSignalType.ADD,
                track_id=track_id,
                detection=detection,
            )

        return None

    @staticmethod
    def _update_confirmed_track(
        active_track: _ActiveTrack,
        detection: Detection,
    ) -> None:
        """Refresh a confirmed track and cancel any active missing timer."""
        active_track.detection = detection
        active_track.missed_frames = 0
        active_track.status = TrackStatus.CONFIRMED
        active_track.missing_since = None

    def _update_missing_lifecycle(
        self,
        track_id: int,
        now: float,
    ) -> TrackSignal | None:
        """Update one absent track and emit REMOVE after its confirmed timeout."""
        active_track = self._tracks[track_id]

        if active_track.status is TrackStatus.CANDIDATE:
            active_track.missed_frames += 1
            if active_track.missed_frames > self.max_missed_frames:
                del self._tracks[track_id]
            return None

        if active_track.status is TrackStatus.CONFIRMED:
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
        """Update candidate tracks and emit ADD once a track is confirmed."""
        if self.confirmation_frames < 1:
            raise ValueError("confirmation_frames must be positive")
        if self.max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        if self.removal_timeout_seconds <= 0:
            raise ValueError("removal_timeout_seconds must be positive")

        tracked_detections = self._index_detections(detections)
        now = self.clock()
        signals: list[TrackSignal] = []

        for track_id in set(self._tracks) - set(tracked_detections):
            signal = self._update_missing_lifecycle(track_id, now)
            if signal is not None:
                signals.append(signal)

        for track_id, detection in tracked_detections.items():
            signal = self._update_add_lifecycle(track_id, detection)
            if signal is not None:
                signals.append(signal)

        return signals
