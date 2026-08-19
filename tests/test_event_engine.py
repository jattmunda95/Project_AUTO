from __future__ import annotations

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from project_auto.events.event_engine import EventEngine
from project_auto.memory.models import Base, ItemEventType, ItemStatus
from project_auto.memory.store import DatabaseStore
from project_auto.perception.detector import Detection
from project_auto.perception.tracker import TrackSignal, TrackSignalType


@pytest.fixture
def store() -> DatabaseStore:
    engine: Engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    database_store = DatabaseStore.__new__(DatabaseStore)
    database_store.engine = engine
    database_store.session_factory = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
    )
    yield database_store
    engine.dispose()


def make_add_signal(signal_track_id: int = 7, detection_track_id: int = 7) -> TrackSignal:
    detection = Detection(
        track_id=detection_track_id,
        class_id=41,
        class_name="cup",
        confidence=0.91,
        box=(10, 20, 110, 220),
    )
    return TrackSignal(
        signal_type=TrackSignalType.ADD,
        track_id=signal_track_id,
        detection=detection,
    )


def test_add_signal_creates_linked_item_and_event(store: DatabaseStore) -> None:
    engine = EventEngine(store)

    item, added_event = engine.process_signal(make_add_signal())

    assert item.class_name == "cup"
    assert item.status is ItemStatus.PRESENT
    assert added_event.item_id == item.id
    assert added_event.event_type is ItemEventType.ADDED
    assert added_event.source_track_id == 7
    assert added_event.detector_confidence == pytest.approx(0.91)
    assert engine._item_ids_by_track_id == {7: item.id}
    assert store.count_items() == 1


def test_add_signal_for_associated_track_is_rejected(store: DatabaseStore) -> None:
    engine = EventEngine(store)
    signal = make_add_signal()
    engine.process_signal(signal)

    with pytest.raises(ValueError, match="Track 7 is already associated"):
        engine.process_signal(signal)

    assert store.count_items() == 1


def test_mismatched_signal_and_detection_track_ids_are_rejected(
    store: DatabaseStore,
) -> None:
    engine = EventEngine(store)

    with pytest.raises(ValueError, match="track IDs must match"):
        engine.process_signal(make_add_signal(signal_track_id=7, detection_track_id=8))

    assert store.count_items() == 0
