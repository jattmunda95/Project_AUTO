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


def make_remove_signal(signal_track_id: int, detection_track_id: int | None = None) -> TrackSignal:
    detection = Detection(
        track_id=detection_track_id if detection_track_id is not None else signal_track_id,
        class_id=41,
        class_name="cup",
        confidence=0.5,
        box=(10, 20, 110, 220),
    )
    return TrackSignal(
        signal_type=TrackSignalType.REMOVE,
        track_id=signal_track_id,
        detection=detection,
    )


def test_remove_signal_marks_item_removed_and_releases_the_track_binding(
    store: DatabaseStore,
) -> None:
    engine = EventEngine(store)
    item, _ = engine.process_signal(make_add_signal(signal_track_id=7, detection_track_id=7))

    removed_event = engine.process_signal(make_remove_signal(7))

    saved_item = store.get_item(item.id)
    assert saved_item is not None
    assert saved_item.status is ItemStatus.REMOVED
    assert removed_event.event_type is ItemEventType.REMOVED
    # The retired track_id must not keep claiming the item, or no future track
    # could ever be recognized as this item returning (see process_return below).
    assert engine.item_id_for_track(7) is None
    assert engine.is_item_claimed(item.id) is False


def test_a_new_track_can_return_an_item_after_its_old_track_was_removed(
    store: DatabaseStore,
) -> None:
    """Regression test: a stale track binding used to make is_item_claimed report
    a removed item as permanently claimed, silently blocking every future RETURNED.
    """
    engine = EventEngine(store)
    item, _ = engine.process_signal(make_add_signal(signal_track_id=7, detection_track_id=7))
    engine.process_signal(make_remove_signal(7))

    assert engine.is_item_claimed(item.id, excluding_track_id=99) is False
    returned_event = engine.process_return(make_return_signal(item.id, signal_track_id=99, detection_track_id=99))

    saved_item = store.get_item(item.id)
    assert saved_item is not None
    assert saved_item.status is ItemStatus.PRESENT
    assert returned_event.event_type is ItemEventType.RETURNED
    assert engine.item_id_for_track(99) == item.id


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
    assert engine._item_ids_by_track_id == {19: item.id}


def test_explicit_return_requires_permanent_item_id(store: DatabaseStore) -> None:
    engine = EventEngine(store)

    with pytest.raises(ValueError, match="requires a permanent item_id"):
        engine.process_return(make_return_signal(None))


def test_explicit_return_rejects_already_associated_track(store: DatabaseStore) -> None:
    item = store.create_item("cup", status=ItemStatus.REMOVED)
    other_item = store.create_item("mug", status=ItemStatus.REMOVED)
    engine = EventEngine(store)
    engine.process_return(make_return_signal(item.id))

    with pytest.raises(ValueError, match="Track 19 is already associated"):
        engine.process_return(make_return_signal(other_item.id))


def test_explicit_return_rejects_item_claimed_by_another_track(store: DatabaseStore) -> None:
    item = store.create_item("cup", status=ItemStatus.REMOVED)
    engine = EventEngine(store)
    engine.associate_existing_item(track_id=3, item_id=item.id)

    with pytest.raises(ValueError, match="already claimed by a visible track"):
        engine.process_return(make_return_signal(item.id))


def test_associate_existing_item_binds_track_without_event(store: DatabaseStore) -> None:
    item = store.create_item("cup")
    engine = EventEngine(store)

    engine.associate_existing_item(track_id=5, item_id=item.id)

    assert engine.item_id_for_track(5) == item.id
    assert store.get_item_history(item.id) == []


def test_associate_existing_item_rejects_reused_track(store: DatabaseStore) -> None:
    item = store.create_item("cup")
    other_item = store.create_item("mug")
    engine = EventEngine(store)
    engine.associate_existing_item(track_id=5, item_id=item.id)

    with pytest.raises(ValueError, match="Track 5 is already associated"):
        engine.associate_existing_item(track_id=5, item_id=other_item.id)


def test_associate_existing_item_rejects_double_claim(store: DatabaseStore) -> None:
    item = store.create_item("cup")
    engine = EventEngine(store)
    engine.associate_existing_item(track_id=5, item_id=item.id)

    with pytest.raises(ValueError, match="already claimed by a visible track"):
        engine.associate_existing_item(track_id=6, item_id=item.id)


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
