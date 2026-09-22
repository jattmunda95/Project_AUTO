"""Spatial lifecycle integration against isolated SQLite databases."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import event, inspect
from sqlalchemy.exc import IntegrityError

from project_auto.events.event_engine import EventEngine
from project_auto.memory.models import Item, ItemEvent, ItemEventType, ItemStatus, Region
from project_auto.memory.store import DatabaseStore
from project_auto.perception.detector import Detection
from project_auto.perception.tracker import TrackSignal, TrackSignalType


@pytest.fixture
def store(tmp_path):
    store = DatabaseStore(tmp_path / "regions.db")
    store.create_schema()
    yield store
    store.engine.dispose()


def regions(store):
    left = store.create_region("left tray", [[0, 0], [100, 0], [100, 100], [0, 100]])
    right = store.create_region("centre table", [[100, 0], [200, 0], [200, 100], [100, 100]])
    return left, right


def signal(kind, box, track=7, **kwargs):
    return TrackSignal(signal_type=kind, track_id=track, detection=Detection(
        class_id=1, class_name="calculator", confidence=0.9, box=box, track_id=track,
    ), **kwargs)


def test_complete_spatial_lifecycle_and_queries(store):
    left, right = regions(store)
    engine = EventEngine(store)
    engine.frame_size = (200, 100)
    source, destination = (10, 10, 30, 30), (110, 10, 150, 30)
    item, added = engine.process_signal(signal(TrackSignalType.ADD, source))
    assert added.destination_region_id == left.id
    assert added.destination_region == "left tray"
    assert added.destination_box_area_fraction == pytest.approx(0.02)
    assert store.get_item(item.id).current_box == list(source)
    assert store.get_item_current_region(item.id).id == left.id
    assert [i.id for i in store.get_items_in_region(left.id)] == [item.id]

    now = datetime.now(timezone.utc)
    moved = engine.process_signal(signal(TrackSignalType.MOVED, destination,
        source_box=source, destination_box=destination, started_at=now, finished_at=now))
    assert (moved.source_region_id, moved.destination_region_id) == (left.id, right.id)
    assert (moved.source_region, moved.destination_region) == (left.name, right.name)
    assert moved.source_box == list(source)
    assert moved.destination_box == list(destination)
    assert moved.source_box_area_fraction == pytest.approx(0.02)
    assert moved.destination_box_area_fraction == pytest.approx(0.04)
    assert store.get_item(item.id).current_box == list(destination)
    assert store.get_item_current_region(item.id).id == right.id
    assert not store.get_items_in_region(left.id)
    assert store.get_region_last_activity(left.id).id == moved.id

    removed = engine.process_signal(signal(TrackSignalType.REMOVE, destination))
    assert removed.source_region_id == right.id
    assert removed.source_region == right.name
    assert removed.source_box == list(destination)
    assert removed.source_box_area_fraction == pytest.approx(0.04)
    assert store.get_item(item.id).current_region_id is None
    assert store.get_item(item.id).current_box is None
    assert store.get_item_current_region(item.id) is None
    assert not store.get_items_in_region(right.id)
    assert [i.id for i in store.get_items_associated_with_region(right.id)] == [item.id]

    returned = engine.process_return(signal(TrackSignalType.RETURNED, source,
                                            track=9, item_id=item.id))
    assert returned.destination_region_id == left.id
    assert returned.destination_region == left.name
    assert returned.destination_box == list(source)
    assert store.get_item_current_region(item.id).id == left.id
    assert store.get_item(item.id).current_box == list(source)
    assert store.get_region_last_activity(left.id).id == returned.id
    assert [e.id for e in store.get_region_recent_events(left.id)] == [returned.id, moved.id, added.id]
    assert len(store.get_items_associated_with_region(left.id)) == 1


def test_region_crud_constraints_and_deletion_preserve_history(store):
    left, _ = regions(store)
    assert store.list_regions()[0].area_px == 10000
    assert store.get_region(left.id).polygon == left.polygon
    assert store.get_region_by_name("left tray").id == left.id
    assert store.resolve_box_region((10, 10, 20, 20)).id == left.id
    with pytest.raises(IntegrityError):
        store.create_region(left.name, left.polygon)
    with pytest.raises(ValueError):
        store.create_region("bad", [[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        store.create_region(" ", left.polygon)
    item, added = EventEngine(store).process_signal(signal(TrackSignalType.ADD, (1, 1, 5, 5)))
    with store.session_factory() as session:
        session.get(Region, left.id).name = "electronics tray"
        session.commit()
    assert store.get_item_current_region(item.id).name == "electronics tray"
    assert store.get_item_history(item.id)[0].destination_region == "left tray"
    assert store.get_region_recent_events(left.id)[0].id == added.id
    removed = store.mark_removed(item.id)
    assert removed.source_region_id == left.id
    store.mark_returned(item.id, destination_box=(1, 1, 5, 5))
    assert store.delete_region(left.id)
    assert store.get_item(item.id).current_region_id is None
    history = store.get_item_history(item.id)
    assert len(history) == 3
    assert history[0].destination_region_id is None
    assert history[0].destination_region == "left tray"
    assert history[1].source_region_id is None
    assert history[1].source_region == "electronics tray"
    assert not store.delete_region(left.id)
    assert store.get_item_current_region(item.id) is None


def test_unknown_destination_does_not_revive_previous_region(store):
    left, _ = regions(store)
    engine = EventEngine(store)
    item, _ = engine.process_signal(signal(TrackSignalType.ADD, (1, 1, 5, 5)))
    store.record_movement(item.id, source_box=(1, 1, 5, 5), destination_box=(300, 300, 310, 310))
    assert store.get_item_current_region(item.id) is None
    assert not store.get_items_in_region(left.id)
    assert store.get_item(item.id).current_box == [300, 300, 310, 310]


def test_legacy_fallback_respects_status_and_latest_location_event(store):
    left, _ = regions(store)
    item = store.create_item("legacy")
    with store.session_factory() as session:
        session.add(ItemEvent(item_id=item.id, event_type=ItemEventType.ADDED,
                              destination_region_id=left.id, destination_region=left.name))
        session.commit()
    assert store.get_item_current_region(item.id).id == left.id
    assert not store.get_items_in_region(left.id)  # contents never use fallback
    store.record_event(item.id, ItemEventType.STATUS_CHANGED)
    assert store.get_item_current_region(item.id).id == left.id
    store.record_event(item.id, ItemEventType.MOVED)
    assert store.get_item_current_region(item.id) is None
    with store.session_factory() as session:
        session.get(Item, item.id).status = ItemStatus.REMOVED
        session.get(Item, item.id).current_region_id = left.id
        session.commit()
    assert store.get_item_current_region(item.id) is None
    assert not store.get_items_in_region(left.id)


def test_spatial_failure_rolls_back_event_and_item(store):
    left, _ = regions(store)
    engine = EventEngine(store)
    engine.frame_size = (100, 100)
    item, _ = engine.process_signal(signal(TrackSignalType.ADD, (1, 1, 5, 5)))
    before = store.get_item(item.id)
    engine.frame_size = (0, 100)
    with pytest.raises(ValueError):
        engine.process_signal(signal(TrackSignalType.REMOVE, (1, 1, 5, 5)))
    after = store.get_item(item.id)
    assert after.current_region_id == left.id
    assert after.current_box == before.current_box
    assert after.status == ItemStatus.PRESENT
    assert after.last_seen_at == before.last_seen_at
    assert len(store.get_item_history(item.id)) == 1
    assert engine.item_id_for_track(7) == item.id


def test_queries_do_not_write_and_limits_are_deterministic(store):
    left, _ = regions(store)
    item, _ = EventEngine(store).process_signal(signal(TrackSignalType.ADD, (1, 1, 5, 5)))
    writes = []

    @event.listens_for(store.engine, "before_cursor_execute")
    def track_writes(connection, cursor, statement, parameters, context, executemany):
        if statement.split()[0].upper() in {"INSERT", "UPDATE", "DELETE"}:
            writes.append(statement)

    for _ in range(3):
        store.resolve_box_region((1, 1, 5, 5))
        store.get_item_current_region(item.id)
        store.get_items_in_region(left.id)
        store.get_region_recent_events(left.id)
        store.get_region_last_activity(left.id)
        store.get_items_associated_with_region(left.id)
    assert writes == []
    assert store.get_region_recent_events(left.id, limit=0) == []
    assert store.get_region_last_activity(999) is None
    with pytest.raises(ValueError):
        store.get_region_recent_events(left.id, limit=-1)
    for table, columns in (("items", {"current_region_id"}),
                           ("item_events", {"source_region_id", "destination_region_id"})):
        fks = inspect(store.engine).get_foreign_keys(table)
        region_fks = [fk for fk in fks if fk["referred_table"] == "regions"]
        assert {fk["constrained_columns"][0] for fk in region_fks} == columns
        assert all(fk["options"]["ondelete"] == "SET NULL" for fk in region_fks)
