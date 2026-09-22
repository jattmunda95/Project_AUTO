from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, event, inspect, select, text
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session, sessionmaker

from project_auto.memory.models import Base, Item, ItemEmbedding, ItemEvent, ItemEventType, ItemStatus
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

    assert set(database_inspector.get_table_names()) == {
        "items", "item_events", "item_embeddings", "regions"
    }
    event_indexes = {index["name"] for index in database_inspector.get_indexes("item_events")}
    assert "ix_item_events_item_id_occurred_at" in event_indexes


def test_item_embeddings_round_trip_and_orphan_removal(engine: Engine) -> None:
    with Session(engine) as session:
        item = Item(class_name="cup")
        item.embeddings = [
            ItemEmbedding(model_name="test-model", embedding=[1.0, 0.0]),
            ItemEmbedding(
                model_name="test-model", embedding=[0.6, 0.8], object_image_path="crops/cup.jpg"
            ),
        ]
        session.add(item)
        session.commit()
        item_id = item.id

    with Session(engine) as session:
        item = session.get(Item, item_id)
        references = sorted(item.embeddings, key=lambda reference: reference.id)
        assert len(references) == 2
        assert references[0].embedding == [1.0, 0.0]
        assert references[0].object_image_path is None
        assert references[1].embedding == [0.6, 0.8]
        assert references[1].object_image_path == "crops/cup.jpg"
        assert all(reference.item is item for reference in references)
        assert all(reference.created_at is not None for reference in references)
        removed_id = references[0].id
        item.embeddings.remove(references[0])
        session.commit()
        assert session.get(ItemEmbedding, removed_id) is None

    # A fresh session leaves the collection unloaded, exercising database cascade.
    with Session(engine) as session:
        session.delete(session.get(Item, item_id))
        session.commit()
        assert list(session.scalars(select(ItemEmbedding))) == []


def test_embedding_rejects_unknown_permanent_item(engine: Engine) -> None:
    with Session(engine) as session:
        session.add(ItemEmbedding(item_id=999, model_name="test-model", embedding=[1.0, 0.0]))
        with pytest.raises(IntegrityError):
            session.commit()


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


def test_mark_returned_updates_removed_item_and_records_event(
    store: DatabaseStore,
) -> None:
    item = store.create_item("cup", status=ItemStatus.REMOVED)

    returned_event = store.mark_returned(
        item.id,
        source_track_id=19,
        detector_confidence=0.88,
    )

    saved_item = store.get_item(item.id)
    assert saved_item is not None
    assert saved_item.status is ItemStatus.PRESENT
    assert returned_event.item_id == item.id
    assert returned_event.event_type is ItemEventType.RETURNED
    assert returned_event.source_track_id == 19
    assert returned_event.detector_confidence == pytest.approx(0.88)
    assert [event.id for event in store.get_item_history(item.id)] == [returned_event.id]


def test_mark_returned_rejects_item_that_is_not_removed(
    store: DatabaseStore,
) -> None:
    item = store.create_item("cup", status=ItemStatus.PRESENT)

    with pytest.raises(ValueError, match=f"Item {item.id} is not removed"):
        store.mark_returned(item.id)

    assert store.get_item_history(item.id) == []


def test_mark_returned_rejects_unknown_item(store: DatabaseStore) -> None:
    with pytest.raises(ValueError, match="Item 999 does not exist"):
        store.mark_returned(999)


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


def test_count_item_embeddings_counts_only_matching_model(store: DatabaseStore) -> None:
    item = store.create_item("cup")
    other_item = store.create_item("mug")
    store.save_item_embedding(item.id, [1.0, 0.0], "model-a")
    store.save_item_embedding(item.id, [0.0, 1.0], "model-a")
    store.save_item_embedding(other_item.id, [1.0, 0.0, 0.0], "model-b")

    assert store.count_item_embeddings(item.id, "model-a") == 2
    assert store.count_item_embeddings(item.id, "model-b") == 0
    assert store.count_item_embeddings(other_item.id, "model-b") == 1


def test_add_reference_if_needed_saves_below_target(store: DatabaseStore) -> None:
    item = store.create_item("cup")

    saved, count = store.add_reference_if_needed(item.id, [1.0, 0.0], "model-a", target_count=2)

    assert saved is True
    assert count == 1
    assert store.count_item_embeddings(item.id, "model-a") == 1
    saved_item = store.get_item(item.id)
    assert saved_item is not None
    assert saved_item.item_prototype == pytest.approx([1.0, 0.0])


def test_add_reference_if_needed_stops_at_target(store: DatabaseStore) -> None:
    item = store.create_item("cup")
    store.add_reference_if_needed(item.id, [1.0, 0.0], "model-a", target_count=1)

    saved, count = store.add_reference_if_needed(item.id, [0.0, 1.0], "model-a", target_count=1)

    assert saved is False
    assert count == 1
    assert store.count_item_embeddings(item.id, "model-a") == 1


def test_add_reference_if_needed_rejects_unknown_item(store: DatabaseStore) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        store.add_reference_if_needed(999, [1.0, 0.0], "model-a", target_count=6)


def test_add_reference_if_needed_rejects_mismatched_dimensions(store: DatabaseStore) -> None:
    item = store.create_item("cup")
    store.add_reference_if_needed(item.id, [1.0, 0.0], "model-a", target_count=6)

    with pytest.raises(ValueError, match="matching dimensions"):
        store.add_reference_if_needed(item.id, [1.0, 0.0, 0.0], "model-a", target_count=6)


def test_load_reid_gallery_includes_each_items_prototype(store: DatabaseStore) -> None:
    item = store.create_item("cup")
    store.save_item_embedding(item.id, [1.0, 0.0], "model-a")
    store.save_item_embedding(item.id, [0.0, 1.0], "model-a")
    unreferenced_item = store.create_item("mug")

    gallery = store.load_reid_gallery("model-a")

    assert len(gallery) == 1
    entry = gallery[0]
    assert entry.item_id == item.id
    assert entry.prototype is not None
    assert entry.prototype == pytest.approx(store.get_item(item.id).item_prototype)
    assert unreferenced_item.id not in {e.item_id for e in gallery}
