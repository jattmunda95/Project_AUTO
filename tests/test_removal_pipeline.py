"""Deterministic item-removal and handoff tests with real in-memory persistence."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest
import test_coordinator as shared
from test_coordinator import (
    add_signal,
    frame,
    make_detection,
    move_start_signal,
    remove_signal,
)

from project_auto.events.identification import IdentificationResult, IdentificationState
from project_auto.events.removal_policy import RemovalConfig, RemovalPolicy
from project_auto.memory.models import ItemEventType, ItemStatus
from project_auto.perception.tracker import DetectionTracker, TrackSignal, TrackSignalType

# Register the shared fixtures explicitly; no camera, models, or live database.
store = shared.store
worker = shared.worker
coordinator = shared.coordinator


def present(coordinator, store, worker, track_id=7, box=(10, 20, 110, 220)):
    detection = make_detection(track_id, box)
    coordinator.handle_frame(
        frame(), [TrackSignal(TrackSignalType.ADD, track_id, detection)], {track_id: detection}
    )
    job = worker.submitted[-1]
    worker.push_result(IdentificationResult("resolve", track_id, "new", job_id=job.job_id))
    coordinator.handle_frame(frame(), [], {track_id: detection})
    return coordinator.event_engine.item_id_for_track(track_id)


def tick(coordinator, now, detections=None, signals=None):
    coordinator.clock.return_value = now
    coordinator.handle_frame(frame(), signals or [], detections or {})


def result_for(worker, track_id, status, item_id=None):
    job = next(
        job
        for job in reversed(worker.submitted)
        if job.track_id == track_id and job.kind == "resolve"
    )
    worker.push_result(
        IdentificationResult("resolve", track_id, status, item_id=item_id, job_id=job.job_id)
    )


def events(store, item_id):
    return [event.event_type for event in store.get_item_history(item_id)]


def test_item_timeout_works_without_tracker_remove(coordinator, store, worker):
    item_id = present(coordinator, store, worker)
    tick(coordinator, 0.1)
    tick(coordinator, 2.0)
    assert store.get_item(item_id).status is ItemStatus.PRESENT
    tick(coordinator, 2.1)
    tick(coordinator, 8, signals=[remove_signal(7)])
    assert events(store, item_id).count(ItemEventType.REMOVED) == 1
    assert coordinator.event_engine.item_id_for_track(7) is None


def test_tracker_retirement_alone_never_removes_item(coordinator, store, worker):
    item_id = present(coordinator, store, worker)
    tick(coordinator, 0.1, signals=[remove_signal(7)])
    assert store.get_item(item_id).status is ItemStatus.PRESENT


def test_same_id_reappearance_before_timeout_cancels_absence(coordinator, store, worker):
    item_id = present(coordinator, store, worker)
    tick(coordinator, 0.1)
    tick(coordinator, 1, {7: make_detection(7)})
    tick(coordinator, 8, {7: make_detection(7)})
    assert ItemEventType.REMOVED not in events(store, item_id)


def test_handoff_before_add_transfers_item_and_records_settled_move(coordinator, store, worker):
    item_id = present(coordinator, store, worker)
    tick(coordinator, 0.1, {8: make_detection(8)})
    assert worker.submitted[-1].priority
    assert worker.submitted[-1].track_id == 8
    # A candidate nominated at the old box remains valid after moving away.
    destination = make_detection(8, (210, 20, 310, 220))
    result_for(worker, 8, "existing", item_id)
    tick(coordinator, 0.3, {8: destination})
    assert coordinator.event_engine.item_id_for_track(7) is None
    assert coordinator.event_engine.item_id_for_track(8) == item_id
    assert events(store, item_id) == [ItemEventType.ADDED]
    tick(coordinator, 1.4, {8: destination}, [remove_signal(7)])
    history = store.get_item_history(item_id)
    assert [event.event_type for event in history].count(ItemEventType.MOVED) == 1
    move = next(event for event in history if event.event_type is ItemEventType.MOVED)
    assert move.source_box == [10, 20, 110, 220]
    assert move.destination_box == list(destination.box)
    assert store.get_item(item_id).current_box == list(destination.box)
    tick(coordinator, 2.5, {8: destination}, [TrackSignal(TrackSignalType.ADD, 8, destination)])
    assert events(store, item_id).count(ItemEventType.MOVED) == 1
    assert ItemEventType.REMOVED not in events(store, item_id)
    assert ItemEventType.RETURNED not in events(store, item_id)


@pytest.mark.parametrize("status", ["new", "pending", "error"])
def test_failed_early_probe_never_creates_or_binds_item(coordinator, store, worker, status):
    item_id = present(coordinator, store, worker)
    tick(coordinator, 0.1, {8: make_detection(8)})
    result_for(worker, 8, status)
    tick(coordinator, 0.2, {8: make_detection(8)})
    assert store.count_items() == 1
    assert coordinator.event_engine.item_id_for_track(8) is None
    tick(coordinator, 2.1, {8: make_detection(8)})
    assert store.get_item(item_id).status is ItemStatus.REMOVED


def test_candidate_that_disappeared_cannot_keep_item_present(coordinator, store, worker):
    item_id = present(coordinator, store, worker)
    tick(coordinator, 0.1, {8: make_detection(8)})
    result_for(worker, 8, "existing", item_id)
    tick(coordinator, 2.1)
    assert store.get_item(item_id).status is ItemStatus.REMOVED
    assert coordinator.event_engine.item_id_for_track(8) is None


def test_grace_is_bounded_and_late_result_can_only_return_after_confirmation(
    coordinator, store, worker
):
    item_id = present(coordinator, store, worker)
    tick(coordinator, 0.1, {8: make_detection(8)})
    tick(coordinator, 2.2, {8: make_detection(8)}, [add_signal(8), remove_signal(7)])
    assert store.get_item(item_id).status is ItemStatus.PRESENT
    tick(coordinator, 5.1, {8: make_detection(8)})
    assert store.get_item(item_id).status is ItemStatus.REMOVED
    result_for(worker, 8, "existing", item_id)
    tick(coordinator, 5.2, {8: make_detection(8)})
    assert events(store, item_id).count(ItemEventType.REMOVED) == 1
    assert events(store, item_id).count(ItemEventType.RETURNED) == 1


def test_reuses_already_queued_resolve(coordinator, store, worker):
    item_id = present(coordinator, store, worker)
    tick(coordinator, 0.05, {7: make_detection(7), 8: make_detection(8)}, [add_signal(8)])
    count = len(worker.submitted)
    tick(coordinator, 0.1, {8: make_detection(8)})
    assert len(worker.submitted) == count
    result_for(worker, 8, "existing", item_id)
    tick(coordinator, 0.2, {8: make_detection(8)})
    assert coordinator.event_engine.item_id_for_track(8) == item_id


def test_old_owner_reappears_before_match_prevents_handoff(coordinator, store, worker):
    item_id = present(coordinator, store, worker)
    tick(coordinator, 0.1, {8: make_detection(8)})
    result_for(worker, 8, "existing", item_id)
    tick(coordinator, 0.2, {7: make_detection(7), 8: make_detection(8)})
    assert coordinator.event_engine.item_id_for_track(7) == item_id
    assert coordinator.event_engine.item_id_for_track(8) is None


def test_distant_stationary_candidate_is_not_sent_for_reid(coordinator, store, worker):
    present(coordinator, store, worker)
    count = len(worker.submitted)
    tick(coordinator, 0.1, {8: make_detection(8, (410, 20, 510, 220))})
    assert len(worker.submitted) == count


def test_moving_candidate_can_be_checked_without_overlap(coordinator, store, worker):
    present(coordinator, store, worker)
    tick(coordinator, 0.05, {7: make_detection(7)}, [move_start_signal(7)])
    tick(coordinator, 0.1, {8: make_detection(8, (120, 20, 220, 220))})
    assert worker.submitted[-1].track_id == 8
    assert worker.submitted[-1].priority


def test_handoff_capacity_and_queue_failure_do_not_hold_removal_forever(coordinator, store, worker):
    item_id = present(coordinator, store, worker)
    worker.max_size = 0
    tick(coordinator, 0.1, {8: make_detection(8)})
    tick(coordinator, 2.2, {8: make_detection(8)})
    assert store.get_item(item_id).status is ItemStatus.PRESENT
    tick(coordinator, 5.1, {8: make_detection(8)})
    assert store.get_item(item_id).status is ItemStatus.REMOVED


def test_retiring_track_result_cannot_create_or_return_item(coordinator, store, worker):
    item = store.create_item("cup", status=ItemStatus.REMOVED)
    tick(coordinator, 0, {7: make_detection(7)}, [add_signal(7)])
    result_for(worker, 7, "existing", item.id)
    tick(coordinator, 1, signals=[remove_signal(7)])
    assert store.get_item_history(item.id) == []
    assert store.get_item(item.id).status is ItemStatus.REMOVED


def test_old_job_cannot_bind_reused_id_or_clear_its_new_job(coordinator, store, worker):
    tick(coordinator, 0, {7: make_detection(7)}, [add_signal(7)])
    old_job = worker.submitted[-1]
    tick(coordinator, 1, signals=[remove_signal(7)])
    tick(coordinator, 2, {7: make_detection(7)}, [add_signal(7)])
    new_job = worker.submitted[-1]
    worker.push_result(IdentificationResult("resolve", 7, "new", job_id=old_job.job_id))
    tick(coordinator, 2.1, {7: make_detection(7)})
    assert store.count_items() == 0
    assert coordinator._inflight[7] == new_job.job_id
    assert coordinator._reference_manager.state(7) is IdentificationState.QUEUED


def test_same_track_can_return_after_item_timeout_without_another_add(coordinator, store, worker):
    item_id = present(coordinator, store, worker)
    tick(coordinator, 0.1)
    tick(coordinator, 2.1)
    tick(coordinator, 2.2, {7: make_detection(7)})
    assert worker.submitted[-1].kind == "resolve"
    result_for(worker, 7, "existing", item_id)
    tick(coordinator, 2.3, {7: make_detection(7)})
    assert store.get_item(item_id).status is ItemStatus.PRESENT
    assert events(store, item_id).count(ItemEventType.RETURNED) == 1


def test_geometry_never_uses_class_or_regions():
    policy = RemovalPolicy()
    policy.bind(1, make_detection(7))
    policy.observe({}, set(), 0, datetime.now(timezone.utc))
    candidate = replace(make_detection(8), class_name="person")
    policy.remember_candidates({8: candidate}, {7})
    assert policy.items[1].candidates == {8}


def test_missing_unretired_track_cannot_be_bound_by_old_crop(coordinator, store, worker):
    tick(coordinator, 0, {7: make_detection(7)}, [add_signal(7)])
    result_for(worker, 7, "new")
    tick(coordinator, 0.1)
    assert store.count_items() == 0
    assert coordinator._reference_manager.reason(7) == "not_visible"


def test_real_tracker_id_change_produces_one_move_without_removal(coordinator, store, worker):
    tracker = DetectionTracker(clock=coordinator.clock, removal_timeout_seconds=3.0)

    def step(now, detection):
        coordinator.clock.return_value = now
        signals = tracker.update([detection])
        coordinator.handle_frame(frame(), signals, {detection.track_id: detection})

    step(0, make_detection(7))
    step(2, make_detection(7))
    result_for(worker, 7, "new")
    step(2.1, make_detection(7))
    item_id = coordinator.event_engine.item_id_for_track(7)
    step(2.2, make_detection(8))
    assert worker.submitted[-1].priority
    result_for(worker, 8, "existing", item_id)
    destination = make_detection(8, (210, 20, 310, 220))
    step(2.3, destination)
    step(3.4, destination)
    step(4.3, destination)  # new ID's ADD must not create or resolve another item
    step(5.3, destination)  # old ID's REMOVE must not remove the transferred item
    assert store.count_items() == 1
    assert events(store, item_id).count(ItemEventType.MOVED) == 1
    assert ItemEventType.REMOVED not in events(store, item_id)
    assert ItemEventType.RETURNED not in events(store, item_id)


def test_handoff_at_same_location_emits_no_movement(coordinator, store, worker):
    item_id = present(coordinator, store, worker)
    tick(coordinator, 0.1, {8: make_detection(8)})
    result_for(worker, 8, "existing", item_id)
    tick(coordinator, 0.2, {8: make_detection(8)})
    tick(coordinator, 1.3, {8: make_detection(8)})
    assert events(store, item_id) == [ItemEventType.ADDED]


def test_no_more_than_two_handoff_jobs_are_outstanding(coordinator, store, worker):
    present(coordinator, store, worker)
    tick(coordinator, 0.1, {n: make_detection(n) for n in [8, 9, 10]})
    assert len([job for job in worker.submitted if job.priority]) == 2
    assert len(coordinator._handoff_job_ids) == 2


def test_wrong_item_match_never_transfers_missing_items_binding(coordinator, store, worker):
    item_id = present(coordinator, store, worker)
    other = store.create_item("cup", status=ItemStatus.REMOVED)
    tick(coordinator, 0.1, {8: make_detection(8)})
    result_for(worker, 8, "existing", other.id)
    tick(coordinator, 0.2, {8: make_detection(8)})
    assert coordinator.event_engine.item_id_for_track(7) == item_id
    assert coordinator.event_engine.item_id_for_track(8) is None
    assert store.get_item(other.id).status is ItemStatus.REMOVED


def test_single_frame_candidate_dropout_does_not_invalidate_or_duplicate_job(
    coordinator, store, worker
):
    item_id = present(coordinator, store, worker)
    tick(coordinator, 0.1, {8: make_detection(8)})
    first_job = worker.submitted[-1]
    # Original owner returns for a frame, then disappears again. The candidate
    # also flickers; this reproduced the duplicate track=257 jobs in the live log.
    tick(coordinator, 0.15, {7: make_detection(7)})
    tick(coordinator, 0.2, {8: make_detection(8)})
    assert coordinator._inflight[8] == first_job.job_id
    assert len([job for job in worker.submitted if job.track_id == 8]) == 1
    worker.push_result(IdentificationResult("resolve", 8, "existing", item_id=item_id,
                                          job_id=first_job.job_id))
    tick(coordinator, 0.3, {8: make_detection(8)})
    assert coordinator.event_engine.item_id_for_track(8) == item_id
    assert events(store, item_id) == [ItemEventType.ADDED]


def test_match_arriving_during_brief_dropout_waits_for_visible_candidate(
    coordinator, store, worker
):
    item_id = present(coordinator, store, worker)
    tick(coordinator, 0.1, {8: make_detection(8)})
    result_for(worker, 8, "existing", item_id)
    tick(coordinator, 0.15)
    assert coordinator.event_engine.item_id_for_track(8) is None
    tick(coordinator, 0.2, {8: make_detection(8)})
    assert coordinator.event_engine.item_id_for_track(8) == item_id
    assert events(store, item_id) == [ItemEventType.ADDED]


def test_brief_candidate_dropout_at_absence_boundary_does_not_remove_item(
    coordinator, store, worker
):
    item_id = present(coordinator, store, worker)
    tick(coordinator, 0.1, {8: make_detection(8)})
    tick(coordinator, 2, {8: make_detection(8)})
    tick(coordinator, 2.15)
    assert store.get_item(item_id).status is ItemStatus.PRESENT
    result_for(worker, 8, "existing", item_id)
    tick(coordinator, 2.2, {8: make_detection(8)})
    assert coordinator.event_engine.item_id_for_track(8) == item_id


@pytest.mark.parametrize("kwargs", [{"absence_seconds": 0}, {"max_inflight": 0}, {"min_iou": 2}])
def test_invalid_policy_settings_fail_at_startup(kwargs):
    with pytest.raises(ValueError):
        RemovalConfig(**kwargs)
