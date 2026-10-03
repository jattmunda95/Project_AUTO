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
    signals = tracker.update([exiting_detection])
    active_track = tracker._tracks[7]

    assert len(signals) == 1
    assert signals[0].signal_type is TrackSignalType.MOVE_START
    assert signals[0].track_id == 7
    assert signals[0].detection is exiting_detection
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

    assert len(signals) == 2
    assert signals[0].signal_type is TrackSignalType.MOVE_END
    assert signals[0].track_id == 7
    assert signals[0].detection is destination_detection
    assert signals[1].signal_type is TrackSignalType.MOVED
    assert signals[1].started_at == movement_started_at
    assert signals[1].finished_at == stopped_at
    assert signals[1].source_box == source_detection.box
    assert signals[1].destination_box == destination_detection.box
    active_track = tracker._tracks[7]
    assert active_track.status is TrackStatus.STABLE
    assert active_track.stable_box == destination_detection.box
    # No duplicate MOVE_END/MOVED on later frames once stable again.
    assert tracker.update([destination_detection]) == []
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

    assert len(signals) == 2
    assert signals[0].signal_type is TrackSignalType.MOVE_END
    assert signals[1].signal_type is TrackSignalType.MOVED
    assert signals[1].finished_at == resumed_stop_at


def test_moving_track_does_not_repeat_move_start_while_still_moving() -> None:
    now = [0.0]
    tracker = DetectionTracker(
        candidate_confirmation_seconds=0.1,
        clock=lambda: now[0],
    )
    detection = make_detection()
    tracker.update([detection])
    now[0] = 0.1
    tracker.update([detection])

    exiting_detection = make_detection(box=(1, 0, 121, 240))
    signals = tracker.update([exiting_detection])
    assert len(signals) == 1
    assert signals[0].signal_type is TrackSignalType.MOVE_START

    # Still moving (displacement keeps resetting the stop attempt): no repeat signal.
    assert tracker.update([make_detection(box=(30, 0, 150, 240))]) == []
    assert tracker.update([make_detection(box=(60, 0, 180, 240))]) == []
    assert tracker._tracks[7].status is TrackStatus.MOVING


def test_stable_track_does_not_emit_move_end_without_prior_movement() -> None:
    now = [0.0]
    tracker = DetectionTracker(
        candidate_confirmation_seconds=0.1,
        clock=lambda: now[0],
    )
    detection = make_detection()
    tracker.update([detection])
    now[0] = 0.1
    tracker.update([detection])
    assert tracker._tracks[7].status is TrackStatus.STABLE

    # Remaining inside the buffer on many subsequent frames emits nothing.
    for _ in range(5):
        assert tracker.update([detection]) == []
    assert tracker._tracks[7].status is TrackStatus.STABLE


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


# --- Stillness-gated ADD -----------------------------------------------------


def shifted(x: int) -> Detection:
    return make_detection(box=(x, 20, x + 100, 220))


def test_a_moving_candidate_is_never_added() -> None:
    now = [0.0]
    tracker = DetectionTracker(clock=lambda: now[0])

    # 30px per half second for 6s: always further than the tolerance from the anchor.
    for step in range(13):
        now[0] = step * 0.5
        assert tracker.update([shifted(10 + step * 30)]) == []


def test_jitter_inside_the_tolerance_still_confirms_on_time() -> None:
    now = [0.0]
    tracker = DetectionTracker(clock=lambda: now[0])

    for step, offset in enumerate((0, 4, -3, 5, -4, 3, 0, 4, -2, 2)):
        now[0] = step * 0.2
        assert tracker.update([shifted(10 + offset)]) == []

    now[0] = 2.0
    signals = tracker.update([shifted(10)])

    assert [signal.signal_type for signal in signals] == [TrackSignalType.ADD]


def test_slow_drift_beyond_the_tolerance_restarts_the_clock() -> None:
    now = [0.0]
    tracker = DetectionTracker(clock=lambda: now[0])

    # 5px per 0.25s never exceeds the tolerance frame to frame, but it does against
    # the fixed anchor, so the clock must restart rather than confirm at 2.0s.
    signals = []
    for step in range(9):
        now[0] = step * 0.25
        signals += tracker.update([shifted(10 + step * 5)])

    assert signals == []


def test_the_clock_restarts_from_the_moment_the_candidate_stops() -> None:
    now = [0.0]
    tracker = DetectionTracker(clock=lambda: now[0])
    assert tracker.update([shifted(10)]) == []

    now[0] = 1.0
    assert tracker.update([shifted(200)]) == []  # moved: clock restarts at 1.0

    now[0] = 2.0
    assert tracker.update([shifted(200)]) == []  # only 1.0s still

    now[0] = 2.99
    assert tracker.update([shifted(200)]) == []

    now[0] = 3.0
    signals = tracker.update([shifted(200)])
    assert [signal.signal_type for signal in signals] == [TrackSignalType.ADD]


def test_a_candidate_that_stops_moving_is_added_after_the_still_period() -> None:
    now = [0.0]
    tracker = DetectionTracker(clock=lambda: now[0])
    for step in range(5):
        now[0] = step * 0.5
        assert tracker.update([shifted(10 + step * 40)]) == []

    still_since = now[0]
    final = shifted(10 + 4 * 40)
    now[0] = still_since + 1.99
    assert tracker.update([final]) == []
    now[0] = still_since + 2.0
    assert [s.signal_type for s in tracker.update([final])] == [TrackSignalType.ADD]


def test_stillness_survives_brief_candidate_dropouts() -> None:
    now = [0.0]
    tracker = DetectionTracker(clock=lambda: now[0])
    assert tracker.update([shifted(10)]) == []
    now[0] = 0.5
    assert tracker.update([]) == []  # a missing frame must not reset the anchor
    now[0] = 2.0
    signals = tracker.update([shifted(10)])

    assert [signal.signal_type for signal in signals] == [TrackSignalType.ADD]


def test_a_restarted_clock_is_logged_with_its_reason(capsys: pytest.CaptureFixture[str]) -> None:
    now = [0.0]
    tracker = DetectionTracker(clock=lambda: now[0])
    tracker.update([shifted(10)])
    now[0] = 0.5
    tracker.update([shifted(100)])

    line = capsys.readouterr().out
    assert "DEFER" in line
    assert "track=7" in line
    assert "reason=not_still" in line
    assert "resets=1" in line
    assert "visible_s=0.5000" in line
    assert "tolerance_px=12.0000" in line


def test_restart_logging_is_limited_to_once_per_confirmation_window(
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = [0.0]
    tracker = DetectionTracker(clock=lambda: now[0])
    tracker.update([shifted(10)])

    # Ten resets within the first 2s window must produce one line, not ten.
    for step in range(1, 11):
        now[0] = step * 0.15
        tracker.update([shifted(10 + step * 40)])
    assert capsys.readouterr().out.count("reason=not_still") == 1

    # Still moving after the window: another line, carrying the running reset count.
    for step in range(11, 25):
        now[0] = step * 0.15
        tracker.update([shifted(10 + step * 40)])
    output = capsys.readouterr().out
    assert output.count("reason=not_still") >= 1
    assert "resets=1 " not in output


def test_a_still_candidate_logs_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    now = [0.0]
    tracker = DetectionTracker(clock=lambda: now[0])
    for step in range(5):
        now[0] = step * 0.5
        tracker.update([shifted(10)])

    assert "not_still" not in capsys.readouterr().out


def test_negative_stillness_tolerance_is_rejected() -> None:
    tracker = DetectionTracker(candidate_stillness_tolerance_pixels=-1.0)

    with pytest.raises(ValueError, match="candidate_stillness_tolerance_pixels"):
        tracker.update([shifted(10)])
