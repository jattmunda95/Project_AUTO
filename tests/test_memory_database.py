from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, event, inspect, select, text
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session, sessionmaker

from project_auto.memory.models import Base, Item, ItemEvent, ItemEventType, ItemStatus
from project_auto.memory.store import DatabaseStore


@pytest.fixture
def engine() -> Engine:
    database_engine = create_engine("sqlite:///:memory:")

    @event.listens_for(database_engine, "connect")
    def enable_foreign_keys(database_connection: object, _: object) -> None:
        cursor = database_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(database_engine)
    yield database_engine
    database_engine.dispose()


@pytest.fixture
def store(engine: Engine) -> DatabaseStore:
    database_store = DatabaseStore.__new__(DatabaseStore)
    database_store.engine = engine
    database_store.session_factory = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
    )
    return database_store


def test_schema_creates_required_tables_and_composite_index(engine: Engine) -> None:
    database_inspector = inspect(engine)

    assert set(database_inspector.get_table_names()) == {"items", "item_events"}
    event_indexes = {index["name"] for index in database_inspector.get_indexes("item_events")}
    assert "ix_item_events_item_id_occurred_at" in event_indexes


def test_item_and_event_relationship_works_both_directions(engine: Engine) -> None:
    with Session(engine) as session:
        item = Item(class_name="cup", display_name="blue mug")
        event_record = ItemEvent(event_type=ItemEventType.ADDED, source_track_id=12)
        item.events.append(event_record)
        session.add(item)
        session.commit()

        assert event_record.item is item
        assert item.events == [event_record]
        assert event_record.item_id == item.id


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (ItemStatus.PRESENT, True),
        (ItemStatus.OCCLUDED, True),
        (ItemStatus.REMOVED, False),
    ],
)
def test_is_present_for_every_status(status: ItemStatus, expected: bool) -> None:
    assert Item(class_name="cup", status=status).is_present is expected


def test_invalid_enum_value_is_rejected(engine: Engine) -> None:
    with Session(engine) as session:
        session.add(Item(class_name="cup", status="invalid"))  # type: ignore[arg-type]

        with pytest.raises(StatementError):
            session.commit()


def test_enum_values_are_stored_as_lowercase_strings(engine: Engine) -> None:
    with Session(engine) as session:
        item = Item(class_name="cup", status=ItemStatus.OCCLUDED)
        item.events.append(ItemEvent(event_type=ItemEventType.STATUS_CHANGED))
        session.add(item)
        session.commit()

    with engine.connect() as connection:
        stored_status = connection.scalar(text("SELECT status FROM items"))
        stored_event_type = connection.scalar(text("SELECT event_type FROM item_events"))

    assert stored_status == "occluded"
    assert stored_event_type == "status_changed"


def test_foreign_key_rejects_event_for_unknown_item(engine: Engine) -> None:
    with Session(engine) as session:
        session.add(ItemEvent(item_id=999, event_type=ItemEventType.ADDED))

        with pytest.raises(IntegrityError):
            session.commit()


def test_deleting_item_cascades_to_events(store: DatabaseStore) -> None:
    item = store.create_item("cup")
    event_record = store.record_event(item.id, ItemEventType.ADDED)

    with store.session_factory() as session:
        saved_item = session.get(Item, item.id)
        assert saved_item is not None
        session.delete(saved_item)
        session.commit()

    with store.session_factory() as session:
        assert session.get(ItemEvent, event_record.id) is None


def test_history_is_chronological(store: DatabaseStore) -> None:
    item = store.create_item("cup")
    first = store.record_event(item.id, ItemEventType.ADDED)
    second = store.record_event(item.id, ItemEventType.MOVED)

    history = store.get_item_history(item.id)

    assert [event.id for event in history] == [first.id, second.id]


def test_mark_present_records_only_real_status_changes(store: DatabaseStore) -> None:
    item = store.create_item("cup", status=ItemStatus.OCCLUDED)

    changed_event = store.mark_present(item.id)
    duplicate_event = store.mark_present(item.id)

    assert changed_event is not None
    assert changed_event.event_type is ItemEventType.STATUS_CHANGED
    assert duplicate_event is None
    assert len(store.get_item_history(item.id)) == 1


def test_status_functions_reject_unknown_item(store: DatabaseStore) -> None:
    with pytest.raises(ValueError, match="Item 999 does not exist"):
        store.mark_removed(999)


def test_present_item_listing_excludes_removed_items(store: DatabaseStore) -> None:
    present = store.create_item("cup", status=ItemStatus.PRESENT)
    occluded = store.create_item("book", status=ItemStatus.OCCLUDED)
    store.create_item("bottle", status=ItemStatus.REMOVED)

    listed_ids = {item.id for item in store.list_present_items()}

    assert listed_ids == {present.id, occluded.id}
    assert store.count_items() == 3


def test_event_query_can_filter_by_permanent_item_id(engine: Engine) -> None:
    with Session(engine) as session:
        first_item = Item(class_name="cup")
        second_item = Item(class_name="book")
        first_item.events.append(ItemEvent(event_type=ItemEventType.ADDED))
        second_item.events.append(ItemEvent(event_type=ItemEventType.ADDED))
        session.add_all([first_item, second_item])
        session.commit()

        first_history = list(
            session.scalars(select(ItemEvent).where(ItemEvent.item_id == first_item.id))
        )

    assert len(first_history) == 1
    assert first_history[0].item_id == first_item.id
