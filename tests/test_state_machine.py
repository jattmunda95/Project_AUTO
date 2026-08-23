from project_auto.memory.models import ItemEventType, ItemStatus
import pytest

from project_auto.memory.state_machine import decide_return_signal, decide_track_signal
from project_auto.perception.detector import Detection
from project_auto.perception.tracker import TrackSignal, TrackSignalType


def test_moved_signal_keeps_item_present_and_creates_moved_decision() -> None:
    detection = Detection(
        track_id=7,
        class_id=41,
        class_name="cup",
        confidence=0.9,
        box=(30, 20, 130, 220),
    )
    signal = TrackSignal(
        signal_type=TrackSignalType.MOVED,
        track_id=7,
        detection=detection,
    )

    decision = decide_track_signal(signal)

    assert decision.status is ItemStatus.PRESENT
    assert decision.event_type is ItemEventType.MOVED


def test_resolved_return_signal_makes_item_present() -> None:
    detection = Detection(
        track_id=12,
        class_id=41,
        class_name="cup",
        confidence=0.88,
        box=(30, 20, 130, 220),
    )
    signal = TrackSignal(
        signal_type=TrackSignalType.RETURNED,
        track_id=12,
        detection=detection,
        item_id=4,
    )

    decision = decide_return_signal(signal)

    assert decision.status is ItemStatus.PRESENT
    assert decision.event_type is ItemEventType.RETURNED


def test_return_signal_requires_permanent_item_id() -> None:
    detection = Detection(
        track_id=12,
        class_id=41,
        class_name="cup",
        confidence=0.88,
        box=(30, 20, 130, 220),
    )
    signal = TrackSignal(
        signal_type=TrackSignalType.RETURNED,
        track_id=12,
        detection=detection,
    )

    with pytest.raises(ValueError, match="requires a permanent item_id"):
        decide_return_signal(signal)
