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


def test_confirmed_track_emits_remove_once_after_timeout() -> None:
    now = [0.0]
    tracker = DetectionTracker(
        confirmation_frames=1,
        removal_timeout_seconds=2.0,
        clock=lambda: now[0],
    )
    detection = make_detection()
    tracker.update([detection])

    assert tracker.update([]) == []
    now[0] = 1.99
    assert tracker.update([]) == []

    now[0] = 2.0
    signals = tracker.update([])

    assert len(signals) == 1
    assert signals[0].signal_type is TrackSignalType.REMOVE
    assert signals[0].track_id == 7
    assert signals[0].detection is detection
    assert tracker.update([]) == []


def test_same_track_id_reappearing_before_timeout_cancels_removal() -> None:
    now = [0.0]
    tracker = DetectionTracker(
        confirmation_frames=1,
        removal_timeout_seconds=2.0,
        clock=lambda: now[0],
    )
    detection = make_detection()
    tracker.update([detection])
    tracker.update([])

    now[0] = 1.5
    assert tracker.update([detection]) == []

    now[0] = 3.0
    assert tracker.update([]) == []
    now[0] = 4.99
    assert tracker.update([]) == []
    now[0] = 5.0
    assert tracker.update([])[0].signal_type is TrackSignalType.REMOVE


@pytest.mark.parametrize(
    "tracker",
    [
        DetectionTracker(confirmation_frames=0),
        DetectionTracker(max_missed_frames=-1),
        DetectionTracker(removal_timeout_seconds=0),
    ],
)
def test_invalid_configuration_is_rejected(tracker: DetectionTracker) -> None:
    with pytest.raises(ValueError):
        tracker.update([])
