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


def make_return_signal(
    item_id: int | None,
    signal_track_id: int = 19,
    detection_track_id: int = 19,
) -> TrackSignal:
    detection = Detection(
        track_id=detection_track_id,
        class_id=41,
        class_name="cup",
        confidence=0.87,
        box=(30, 20, 130, 220),
    )
    return TrackSignal(
        signal_type=TrackSignalType.RETURNED,
        track_id=signal_track_id,
        detection=detection,
        item_id=item_id,
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


def test_explicit_return_persists_against_resolved_permanent_item(
    store: DatabaseStore,
) -> None:
    item = store.create_item("cup", status=ItemStatus.REMOVED)
    engine = EventEngine(store)

    returned_event = engine.process_return(make_return_signal(item.id))

    saved_item = store.get_item(item.id)
    assert saved_item is not None
    assert saved_item.status is ItemStatus.PRESENT
    assert returned_event.item_id == item.id
    assert returned_event.event_type is ItemEventType.RETURNED
    assert returned_event.source_track_id == 19
    assert returned_event.detector_confidence == pytest.approx(0.87)
    assert engine._item_ids_by_track_id == {}


def test_explicit_return_requires_permanent_item_id(store: DatabaseStore) -> None:
    engine = EventEngine(store)

    with pytest.raises(ValueError, match="requires a permanent item_id"):
        engine.process_return(make_return_signal(None))


def test_explicit_return_rejects_mismatched_track_ids(store: DatabaseStore) -> None:
    item = store.create_item("cup", status=ItemStatus.REMOVED)
    engine = EventEngine(store)

    with pytest.raises(ValueError, match="track IDs must match"):
        engine.process_return(
            make_return_signal(
                item.id,
                signal_track_id=19,
                detection_track_id=20,
            )
        )

    saved_item = store.get_item(item.id)
    assert saved_item is not None
    assert saved_item.status is ItemStatus.REMOVED
    assert store.get_item_history(item.id) == []
