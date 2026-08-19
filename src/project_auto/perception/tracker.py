"""Track structured detections across consecutive video frames."""

from dataclasses import dataclass, field
from enum import Enum

from project_auto.perception.detector import Detection


class TrackStatus(str, Enum):
    """Lifecycle states for a temporary track."""

    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"


class TrackSignalType(str, Enum):
    """Meaningful lifecycle signals emitted for downstream decisions."""

    ADD = "add"


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


@dataclass(slots=True)
class DetectionTracker:
    """Hold lifecycle state for BoT-SORT tracks across video frames."""

    confirmation_frames: int = 30
    max_missed_frames: int = 10
    _tracks: dict[int, _ActiveTrack] = field(
        default_factory=dict,
        init=False,
    )

    def update(self, detections: list[Detection]) -> list[TrackSignal]:
        """Update candidate tracks and emit ADD once a track is confirmed."""
        if self.confirmation_frames < 1:
            raise ValueError("confirmation_frames must be positive")
        if self.max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")

        # list[Detection] comprehension custom dictionary to map track_id to detection for all detections that have a track_id
        tracked_detections = {
            detection.track_id: detection
            for detection in detections
            if detection.track_id is not None
        }

        for track_id in set(self._tracks) - set(tracked_detections): # find the tracks that are no longer detected
            active_track = self._tracks[track_id] # make the active track the track that is no longer detected
            active_track.missed_frames += 1 # add missing frame
            if ( 
                active_track.status is TrackStatus.CANDIDATE
                and active_track.missed_frames > self.max_missed_frames
            ):
                del self._tracks[track_id] # remove the track from the active tracks if it has been missed for too long

        signals: list[TrackSignal] = [] # empty list to hold signals to be returned
        for track_id, detection in tracked_detections.items():
            active_track = self._tracks.get(track_id) # get the active track for the current track_id
            if active_track is None:
                active_track = _ActiveTrack(detection=detection)
                self._tracks[track_id] = active_track
            else:
                active_track.detection = detection
                active_track.seen_frames += 1
                active_track.missed_frames = 0

            if (
                active_track.status is TrackStatus.CANDIDATE
                and active_track.seen_frames >= self.confirmation_frames
            ):
                active_track.status = TrackStatus.CONFIRMED
                signals.append(
                    TrackSignal(
                        signal_type=TrackSignalType.ADD,
                        track_id=track_id,
                        detection=detection,
                    )
                )

        return signals
