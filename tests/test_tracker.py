from __future__ import annotations

from datetime import datetime, timezone

import pytest

from project_auto.perception.detector import Detection
from project_auto.perception.tracker import (
    DetectionTracker,
    TrackSignalType,
    TrackStatus,
)


def make_detection(
    track_id: int | None = 7,
    box: tuple[int, int, int, int] = (10, 20, 110, 220),
) -> Detection:
    return Detection(
        track_id=track_id,
        class_id=41,
        class_name="cup",
        confidence=0.9,
        box=box,
    )


def test_add_signal_is_emitted_once_after_two_seconds() -> None:
    now = [0.0]
    tracker = DetectionTracker(clock=lambda: now[0])
    detection = make_detection()

    assert tracker.update([detection]) == []
    now[0] = 1.99
    assert tracker.update([detection]) == []

    now[0] = 2.0
    signals = tracker.update([detection])

    assert len(signals) == 1
    assert signals[0].signal_type is TrackSignalType.ADD
    assert signals[0].track_id == 7
    assert signals[0].detection is detection
    assert tracker.update([detection]) == []


def test_detection_without_track_id_is_ignored() -> None:
    tracker = DetectionTracker(candidate_confirmation_seconds=1.0)

    assert tracker.update([make_detection(track_id=None)]) == []


def test_candidate_tolerates_fifteen_cumulative_missing_frames() -> None:
    now = [0.0]
    tracker = DetectionTracker(clock=lambda: now[0])
    detection = make_detection()
    assert tracker.update([detection]) == []

    for _ in range(10):
        assert tracker.update([]) == []
    assert tracker.update([detection]) == []
    for _ in range(5):
        assert tracker.update([]) == []

    now[0] = 2.0
    signals = tracker.update([detection])

    assert len(signals) == 1
    assert signals[0].signal_type is TrackSignalType.ADD


def test_candidate_restarts_after_sixteenth_cumulative_missing_frame() -> None:
    now = [0.0]
    tracker = DetectionTracker(clock=lambda: now[0])
    detection = make_detection()
    assert tracker.update([detection]) == []

    for _ in range(8):
        assert tracker.update([]) == []
    assert tracker.update([detection]) == []
    for _ in range(8):
        assert tracker.update([]) == []

    now[0] = 2.0
    assert tracker.update([detection]) == []
    now[0] = 3.99
    assert tracker.update([detection]) == []

    now[0] = 4.0
    signals = tracker.update([detection])

    assert len(signals) == 1
    assert signals[0].signal_type is TrackSignalType.ADD


def test_candidate_emits_add_only_when_visible() -> None:
    now = [0.0]
    tracker = DetectionTracker(clock=lambda: now[0])
    detection = make_detection()
    tracker.update([detection])

    now[0] = 2.0
    assert tracker.update([]) == []

    now[0] = 2.1
    assert tracker.update([detection])[0].signal_type is TrackSignalType.ADD


def test_confirmed_track_emits_remove_once_after_timeout() -> None:
    now = [0.0]
    tracker = DetectionTracker(
        candidate_confirmation_seconds=0.1,
        removal_timeout_seconds=2.0,
        clock=lambda: now[0],
    )
    detection = make_detection()
    tracker.update([detection])
    now[0] = 0.1
    tracker.update([detection])

    assert tracker.update([]) == []
    now[0] = 2.09
    assert tracker.update([]) == []

    now[0] = 2.1
    signals = tracker.update([])

    assert len(signals) == 1
    assert signals[0].signal_type is TrackSignalType.REMOVE
    assert signals[0].track_id == 7
    assert signals[0].detection is detection
    assert tracker.update([]) == []


def test_same_track_id_reappearing_before_timeout_cancels_removal() -> None:
    now = [0.0]
    tracker = DetectionTracker(
        candidate_confirmation_seconds=0.1,
        removal_timeout_seconds=2.0,
        clock=lambda: now[0],
    )
    detection = make_detection()
    tracker.update([detection])
    now[0] = 0.1
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


def test_add_establishes_stable_placement_and_buffer() -> None:
    now = [0.0]
    tracker = DetectionTracker(
        candidate_confirmation_seconds=0.1,
        clock=lambda: now[0],
    )
    detection = make_detection()
    tracker.update([detection])

    now[0] = 0.1
    tracker.update([detection])
    active_track = tracker._tracks[7]

    assert active_track.status is TrackStatus.STABLE
    assert active_track.stable_box == detection.box
    assert active_track.buffer_box == (0.0, 0.0, 120.0, 240.0)


def test_stable_track_starts_moving_only_after_exiting_buffer() -> None:
    now = [0.0]
    movement_started_at = datetime(2026, 8, 22, tzinfo=timezone.utc)
    tracker = DetectionTracker(
        candidate_confirmation_seconds=0.1,
        clock=lambda: now[0],
        timestamp_clock=lambda: movement_started_at,
    )
    detection = make_detection()
    tracker.update([detection])
    now[0] = 0.1
    tracker.update([detection])

    boundary_detection = make_detection(box=(0, 0, 120, 240))
    assert tracker.update([boundary_detection]) == []
    assert tracker._tracks[7].status is TrackStatus.STABLE

    exiting_detection = make_detection(box=(1, 0, 121, 240))
    assert tracker.update([exiting_detection]) == []
    active_track = tracker._tracks[7]

    assert active_track.status is TrackStatus.MOVING
    assert active_track.movement_started_at == movement_started_at
    assert active_track.stable_box == detection.box


def test_moving_track_emits_one_timed_moved_signal_after_stopping() -> None:
    now = [0.0]
    movement_started_at = datetime(2026, 8, 22, 1, 0, tzinfo=timezone.utc)
    stopped_at = datetime(2026, 8, 22, 1, 0, 2, tzinfo=timezone.utc)
    timestamps = iter([movement_started_at, stopped_at])
    tracker = DetectionTracker(
        candidate_confirmation_seconds=0.1,
        clock=lambda: now[0],
        timestamp_clock=lambda: next(timestamps),
    )
    source_detection = make_detection()
    tracker.update([source_detection])
    now[0] = 0.1
    tracker.update([source_detection])

    now[0] = 0.2
    tracker.update([make_detection(box=(21, 20, 121, 220))])
    now[0] = 0.3
    tracker.update([make_detection(box=(31, 20, 131, 220))])
    now[0] = 0.4
    destination_detection = make_detection(box=(34, 20, 134, 220))
    tracker.update([destination_detection])
    now[0] = 1.39
    assert tracker.update([destination_detection]) == []

    now[0] = 1.4001
    signals = tracker.update([destination_detection])

    assert len(signals) == 1
    assert signals[0].signal_type is TrackSignalType.MOVED
    assert signals[0].started_at == movement_started_at
    assert signals[0].finished_at == stopped_at
    assert signals[0].source_box == source_detection.box
    assert signals[0].destination_box == destination_detection.box
    active_track = tracker._tracks[7]
    assert active_track.status is TrackStatus.STABLE
    assert active_track.stable_box == destination_detection.box
    assert tracker.update([destination_detection]) == []


def test_moving_track_resets_stop_attempt_when_displacement_exceeds_tolerance() -> None:
    now = [0.0]
    tracker = DetectionTracker(
        candidate_confirmation_seconds=0.1,
        clock=lambda: now[0],
    )
    detection = make_detection()
    tracker.update([detection])
    now[0] = 0.1
    tracker.update([detection])
    tracker.update([make_detection(box=(21, 20, 121, 220))])

    now[0] = 0.2
    tracker.update([make_detection(box=(24, 20, 124, 220))])
    assert tracker._tracks[7].stopped_since == pytest.approx(0.2)

    now[0] = 1.19
    assert tracker.update([make_detection(box=(26, 20, 126, 220))]) == []
    assert tracker._tracks[7].stopped_since == pytest.approx(0.2)

    tracker.update([make_detection(box=(27, 20, 127, 220))])
    assert tracker._tracks[7].stopped_since is None


def test_moving_track_missing_timeout_emits_remove_not_moved() -> None:
    now = [0.0]
    tracker = DetectionTracker(
        candidate_confirmation_seconds=0.1,
        removal_timeout_seconds=2.0,
        clock=lambda: now[0],
    )
    detection = make_detection()
    tracker.update([detection])
    now[0] = 0.1
    tracker.update([detection])
    tracker.update([make_detection(box=(21, 20, 121, 220))])

    now[0] = 0.2
    assert tracker.update([]) == []
    assert tracker._tracks[7].status is TrackStatus.MISSING
    now[0] = 2.19
    assert tracker.update([]) == []

    now[0] = 2.2
    signals = tracker.update([])

    assert len(signals) == 1
    assert signals[0].signal_type is TrackSignalType.REMOVE


def test_moving_track_return_restarts_visible_stop_confirmation() -> None:
    now = [0.0]
    movement_started_at = datetime(2026, 8, 22, 2, 0, tzinfo=timezone.utc)
    abandoned_stop_at = datetime(2026, 8, 22, 2, 0, 1, tzinfo=timezone.utc)
    resumed_stop_at = datetime(2026, 8, 22, 2, 0, 2, tzinfo=timezone.utc)
    timestamps = iter([movement_started_at, abandoned_stop_at, resumed_stop_at])
    tracker = DetectionTracker(
        candidate_confirmation_seconds=0.1,
        clock=lambda: now[0],
        timestamp_clock=lambda: next(timestamps),
    )
    source_detection = make_detection()
    moving_detection = make_detection(box=(21, 20, 121, 220))
    tracker.update([source_detection])
    now[0] = 0.1
    tracker.update([source_detection])
    now[0] = 0.2
    tracker.update([moving_detection])
    now[0] = 0.3
    tracker.update([moving_detection])

    now[0] = 0.8
    tracker.update([])
    now[0] = 1.0
    assert tracker.update([moving_detection]) == []
    assert tracker._tracks[7].status is TrackStatus.MOVING
    assert tracker._tracks[7].stopped_since is None

    now[0] = 2.0
    assert tracker.update([moving_detection]) == []
    now[0] = 2.999
    assert tracker.update([moving_detection]) == []
    now[0] = 3.0
    signals = tracker.update([moving_detection])

    assert len(signals) == 1
    assert signals[0].signal_type is TrackSignalType.MOVED
    assert signals[0].finished_at == resumed_stop_at


@pytest.mark.parametrize(
    "tracker",
    [
        DetectionTracker(candidate_confirmation_seconds=0),
        DetectionTracker(max_candidate_missing_frames=-1),
        DetectionTracker(removal_timeout_seconds=0),
        DetectionTracker(movement_buffer_scale=1.0),
        DetectionTracker(movement_stop_tolerance_pixels=-1),
        DetectionTracker(movement_stopped_confirmation_seconds=0),
    ],
)
def test_invalid_configuration_is_rejected(tracker: DetectionTracker) -> None:
    with pytest.raises(ValueError):
        tracker.update([])
