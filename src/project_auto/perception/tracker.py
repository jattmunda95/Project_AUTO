"""Track structured detections across consecutive video frames."""

from dataclasses import dataclass, field
from enum import Enum

from project_auto.perception.detector import Detection


class TrackStatus(str, Enum):
    """Lifecycle states for a temporary track."""

    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """One detection paired with a temporary, session-local track ID."""

    track_id: int
    detection: Detection


@dataclass(slots=True)
class _ActiveTrack:
    """Mutable frame-to-frame state for one temporary track."""

    detection: Detection
    seen_frames: int = 1
    missed_frames: int = 0


@dataclass(slots=True)
class DetectionTracker:
    """Hold temporary tracks across consecutive video frames."""

    iou_threshold: float = 0.3
    confirmation_frames: int = 30
    max_missed_frames: int = 10
    _next_track_id: int = field(default=1, init=False)
    _tracks: dict[int, _ActiveTrack] = field(default_factory=dict, init=False)
