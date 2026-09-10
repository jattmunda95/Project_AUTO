"""Standalone identity-coordinator tests using a real store and a mocked scene processor."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from unittest.mock import Mock

from project_auto.events.coordinator import IdentityCoordinator
from project_auto.events.event_engine import EventEngine
from project_auto.memory.models import Base, ItemStatus
from project_auto.memory.store import DatabaseStore
from project_auto.perception.detector import Detection
from project_auto.perception.scene_processor import IdentityDecision, PreparedReference
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


@pytest.fixture
def scene_processor() -> Mock:
    return Mock()


def reference() -> PreparedReference:
    return PreparedReference(crop=Image.new("RGB", (4, 4)), embedding=np.array([1.0, 0.0]))


def make_detection(track_id: int) -> Detection:
    return Detection(
        track_id=track_id,
        class_id=41,
        class_name="cup",
        confidence=0.9,
        box=(10, 20, 110, 220),
    )


def add_signal(track_id: int) -> TrackSignal:
    return TrackSignal(TrackSignalType.ADD, track_id, make_detection(track_id))


def remove_signal(track_id: int, detection: Detection | None = None) -> TrackSignal:
    return TrackSignal(TrackSignalType.REMOVE, track_id, detection or make_detection(track_id))


@pytest.fixture
def coordinator(store: DatabaseStore, scene_processor: Mock) -> IdentityCoordinator:
    event_engine = EventEngine(store)
    return IdentityCoordinator(
        scene_processor=scene_processor,
        event_engine=event_engine,
        store=store,
        reid_model_name="test-model",
        reference_target_count=2,
        capture_interval_seconds=0.0,
        pending_retry_interval_seconds=0.0,
        clock=Mock(return_value=0.0),
    )


def test_new_decision_creates_item_and_saves_first_reference(
    coordinator: IdentityCoordinator, scene_processor: Mock, store: DatabaseStore
) -> None:
    scene_processor.process.return_value = IdentityDecision("new", 7, None, 0.0, reference())

    coordinator.handle_frame(np.zeros((4, 4, 3), dtype=np.uint8), [add_signal(7)], {})

    assert store.count_items() == 1
    item = store.list_present_items()[0]
    assert coordinator.event_engine.item_id_for_track(7) == item.id
    assert store.count_item_embeddings(item.id, "test-model") == 1


def test_pending_decision_is_retried_on_a_later_visible_frame(
    coordinator: IdentityCoordinator, scene_processor: Mock, store: DatabaseStore
) -> None:
    scene_processor.process.return_value = IdentityDecision("pending", 7, None, 0.0, None)

    coordinator.handle_frame(np.zeros((4, 4, 3), dtype=np.uint8), [add_signal(7)], {})
    assert store.count_items() == 0

    scene_processor.process.return_value = IdentityDecision("new", 7, None, 0.0, reference())
    scene_processor.prepare_reference.return_value = reference()
    coordinator.handle_frame(np.zeros((4, 4, 3), dtype=np.uint8), [], {7: make_detection(7)})

    assert store.count_items() == 1


def test_pending_retries_are_throttled_by_interval(
    store: DatabaseStore, scene_processor: Mock
) -> None:
    fake_time = [0.0]
    event_engine = EventEngine(store)
    coordinator = IdentityCoordinator(
        scene_processor=scene_processor,
        event_engine=event_engine,
        store=store,
        reid_model_name="test-model",
        reference_target_count=2,
        capture_interval_seconds=0.0,
        pending_retry_interval_seconds=5.0,
        clock=lambda: fake_time[0],
    )
    scene_processor.process.return_value = IdentityDecision("pending", 7, None, 0.0, None)
    coordinator.handle_frame(np.zeros((4, 4, 3), dtype=np.uint8), [add_signal(7)], {})
    assert scene_processor.process.call_count == 1

    # Still visible one second later: within the 5s interval, must not retry yet.
    fake_time[0] = 1.0
    coordinator.handle_frame(np.zeros((4, 4, 3), dtype=np.uint8), [], {7: make_detection(7)})
    assert scene_processor.process.call_count == 1

    # Past the interval: retries.
    fake_time[0] = 6.0
    coordinator.handle_frame(np.zeros((4, 4, 3), dtype=np.uint8), [], {7: make_detection(7)})
    assert scene_processor.process.call_count == 2


def test_pending_request_is_cancelled_on_retirement(
    coordinator: IdentityCoordinator, scene_processor: Mock, store: DatabaseStore
) -> None:
    scene_processor.process.return_value = IdentityDecision("pending", 7, None, 0.0, None)
    coordinator.handle_frame(np.zeros((4, 4, 3), dtype=np.uint8), [add_signal(7)], {})

    coordinator.handle_frame(np.zeros((4, 4, 3), dtype=np.uint8), [remove_signal(7)], {})
    # No longer pending: a later sighting under the same track ID starts fresh, not a retry.
    scene_processor.process.reset_mock()
    coordinator.handle_frame(np.zeros((4, 4, 3), dtype=np.uint8), [], {7: make_detection(7)})

    scene_processor.process.assert_not_called()


def test_existing_match_against_removed_item_returns_it(
    coordinator: IdentityCoordinator, scene_processor: Mock, store: DatabaseStore
) -> None:
    item = store.create_item("cup", status=ItemStatus.REMOVED)
    scene_processor.process.return_value = IdentityDecision("existing", 7, item.id, 0.9, reference())

    coordinator.handle_frame(np.zeros((4, 4, 3), dtype=np.uint8), [add_signal(7)], {})

    saved_item = store.get_item(item.id)
    assert saved_item is not None
    assert saved_item.status is ItemStatus.PRESENT
    assert coordinator.event_engine.item_id_for_track(7) == item.id


def test_existing_match_against_present_item_only_associates(
    coordinator: IdentityCoordinator, scene_processor: Mock, store: DatabaseStore
) -> None:
    item = store.create_item("cup", status=ItemStatus.PRESENT)
    scene_processor.process.return_value = IdentityDecision("existing", 7, item.id, 0.9, reference())

    coordinator.handle_frame(np.zeros((4, 4, 3), dtype=np.uint8), [add_signal(7)], {})

    assert store.get_item_history(item.id) == []
    assert coordinator.event_engine.item_id_for_track(7) == item.id


def test_double_claim_on_same_item_keeps_second_track_pending(
    coordinator: IdentityCoordinator, scene_processor: Mock, store: DatabaseStore
) -> None:
    item = store.create_item("cup", status=ItemStatus.PRESENT)
    scene_processor.process.return_value = IdentityDecision("existing", 7, item.id, 0.9, reference())
    coordinator.handle_frame(np.zeros((4, 4, 3), dtype=np.uint8), [add_signal(7)], {})

    scene_processor.process.return_value = IdentityDecision("existing", 8, item.id, 0.9, reference())
    coordinator.handle_frame(np.zeros((4, 4, 3), dtype=np.uint8), [add_signal(8)], {})

    assert coordinator.event_engine.item_id_for_track(8) is None
    assert 8 in coordinator._pending_track_ids


def test_reference_capture_stops_at_target_count(
    coordinator: IdentityCoordinator, scene_processor: Mock, store: DatabaseStore
) -> None:
    scene_processor.process.return_value = IdentityDecision("new", 7, None, 0.0, reference())
    coordinator.handle_frame(np.zeros((4, 4, 3), dtype=np.uint8), [add_signal(7)], {})
    item_id = store.list_present_items()[0].id
    assert store.count_item_embeddings(item_id, "test-model") == 1

    scene_processor.prepare_reference.return_value = reference()
    coordinator.handle_frame(np.zeros((4, 4, 3), dtype=np.uint8), [], {7: make_detection(7)})
    assert store.count_item_embeddings(item_id, "test-model") == 2

    scene_processor.prepare_reference.reset_mock()
    coordinator.handle_frame(np.zeros((4, 4, 3), dtype=np.uint8), [], {7: make_detection(7)})
    assert store.count_item_embeddings(item_id, "test-model") == 2
    scene_processor.prepare_reference.assert_not_called()
