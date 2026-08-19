from __future__ import annotations

import pytest

from project_auto.perception.detector import Detection
from project_auto.perception.tracker import DetectionTracker, TrackSignalType


def make_detection(track_id: int | None = 7) -> Detection:
    return Detection(
        track_id=track_id,
        class_id=41,
        class_name="cup",
        confidence=0.9,
        box=(10, 20, 110, 220),
    )


def test_add_signal_is_emitted_once_on_thirtieth_sighting() -> None:
    tracker = DetectionTracker()
    detection = make_detection()

    for _ in range(29):
        assert tracker.update([detection]) == []

    signals = tracker.update([detection])

    assert len(signals) == 1
    assert signals[0].signal_type is TrackSignalType.ADD
    assert signals[0].track_id == 7
    assert signals[0].detection is detection
    assert tracker.update([detection]) == []


def test_detection_without_track_id_is_ignored() -> None:
    tracker = DetectionTracker(confirmation_frames=1)

    assert tracker.update([make_detection(track_id=None)]) == []


def test_candidate_tolerates_ten_consecutive_missed_frames() -> None:
    tracker = DetectionTracker(confirmation_frames=2, max_missed_frames=10)
    detection = make_detection()
    tracker.update([detection])

    for _ in range(10):
        assert tracker.update([]) == []

    signals = tracker.update([detection])

    assert len(signals) == 1
    assert signals[0].signal_type is TrackSignalType.ADD


def test_candidate_expires_after_eleventh_consecutive_missed_frame() -> None:
    tracker = DetectionTracker(confirmation_frames=2, max_missed_frames=10)
    detection = make_detection()
    tracker.update([detection])

    for _ in range(11):
        tracker.update([])

    assert tracker.update([detection]) == []


@pytest.mark.parametrize(
    "tracker",
    [
        DetectionTracker(confirmation_frames=0),
        DetectionTracker(max_missed_frames=-1),
    ],
)
def test_invalid_configuration_is_rejected(tracker: DetectionTracker) -> None:
    with pytest.raises(ValueError):
        tracker.update([])
