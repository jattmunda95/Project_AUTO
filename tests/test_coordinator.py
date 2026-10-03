"""Standalone identity-coordinator tests: async submission, dedup, and result apply.

Uses a real store and a FakeWorker double (no real thread) so job submission and
result application can be driven deterministically, one step at a time.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from project_auto.events.coordinator import IdentityCoordinator
from project_auto.events.event_engine import EventEngine
from project_auto.events.identification import (
    IdentificationJob,
    IdentificationResult,
    IdentificationState,
)
from project_auto.events.reference_policy import CandidateKind, ReferencePolicyState
from project_auto.memory.models import Base, ItemStatus
from project_auto.memory.store import DatabaseStore
from project_auto.perception.detector import Detection
from project_auto.perception.scene_processor import PreparedReference
from project_auto.perception.tracker import TrackSignal, TrackSignalType


class FakeWorker:
    """A deterministic worker double: records submissions, returns queued results."""

    def __init__(self, max_size: int = 8) -> None:
        self.max_size = max_size
        self.submitted: list[IdentificationJob] = []
        self._queued: list[IdentificationJob] = []
        self._pending_results: list[IdentificationResult] = []
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float | None = None) -> None:
        self.stopped = True

    def submit(self, job: IdentificationJob) -> bool:
        self.submitted.append(job)
        if len(self._queued) >= self.max_size:
            return False
        self._queued.append(job)
        return True

    def poll_results(self) -> list[IdentificationResult]:
        results, self._pending_results = self._pending_results, []
        return results

    def promote(self, job_id: int) -> bool:
        return any(job.job_id == job_id for job in self._queued)

    def push_result(self, result: IdentificationResult) -> None:
        if result.job_id is None:
            job = next(
                (
                    job
                    for job in reversed(self.submitted)
                    if job.track_id == result.track_id and job.kind == result.kind
                ),
                None,
            )
            if job is not None:
                result = replace(result, job_id=job.job_id)
        self._pending_results.append(result)


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
def worker() -> FakeWorker:
    return FakeWorker()


def reference() -> PreparedReference:
    return PreparedReference(crop=Image.new("RGB", (4, 4)), embedding=np.array([1.0, 0.0]))


def make_detection(track_id: int, box: tuple[int, int, int, int] = (10, 20, 110, 220)) -> Detection:
    return Detection(track_id=track_id, class_id=41, class_name="cup", confidence=0.9, box=box)


def add_signal(track_id: int) -> TrackSignal:
    return TrackSignal(TrackSignalType.ADD, track_id, make_detection(track_id))


def remove_signal(track_id: int, detection: Detection | None = None) -> TrackSignal:
    return TrackSignal(TrackSignalType.REMOVE, track_id, detection or make_detection(track_id))


def move_start_signal(track_id: int) -> TrackSignal:
    return TrackSignal(TrackSignalType.MOVE_START, track_id, make_detection(track_id))


def move_end_signal(track_id: int) -> TrackSignal:
    return TrackSignal(TrackSignalType.MOVE_END, track_id, make_detection(track_id))


def moved_signal(track_id: int) -> TrackSignal:
    import datetime as _dt

    detection = make_detection(track_id)
    now = _dt.datetime.now(_dt.timezone.utc)
    return TrackSignal(
        TrackSignalType.MOVED,
        track_id,
        detection,
        started_at=now,
        finished_at=now,
        source_box=detection.box,
        destination_box=detection.box,
    )


@pytest.fixture
def coordinator(store: DatabaseStore, worker: FakeWorker) -> IdentityCoordinator:
    event_engine = EventEngine(store)
    return IdentityCoordinator(
        scene_processor=Mock(),
        event_engine=event_engine,
        store=store,
        reid_model_name="test-model",
        max_references_per_item=2,
        initial_capture_spacing_frames=0,
        move_candidate_delay_frames=0,
        candidate_retry_frames=0,
        bad_mask_cooldown_seconds=5.0,
        queue_full_retry_seconds=0.5,
        clock=Mock(return_value=0.0),
        worker=worker,
    )


def frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_worker_is_started_on_construction(
    coordinator: IdentityCoordinator, worker: FakeWorker
) -> None:
    assert worker.started is True


def test_shutdown_stops_the_worker(coordinator: IdentityCoordinator, worker: FakeWorker) -> None:
    coordinator.shutdown()
    assert worker.stopped is True


def test_a_new_track_submits_exactly_one_resolve_job(
    coordinator: IdentityCoordinator, worker: FakeWorker
) -> None:
    coordinator.handle_frame(frame(), [add_signal(7)], {})

    assert len(worker.submitted) == 1
    assert worker.submitted[0].kind == "resolve"
    assert worker.submitted[0].track_id == 7


def test_repeated_frames_for_a_queued_track_do_not_resubmit(
    coordinator: IdentityCoordinator, worker: FakeWorker
) -> None:
    coordinator.handle_frame(frame(), [add_signal(7)], {})
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})

    # Only the initial ADD submits; a QUEUED track is not in pending_track_ids(),
    # so the retry loop over detections_by_track_id never re-submits it either.
    assert len(worker.submitted) == 1


def test_submission_does_not_block_when_the_queue_is_full(
    coordinator: IdentityCoordinator, worker: FakeWorker
) -> None:
    worker.max_size = 0  # every submit() reports the queue as full

    coordinator.handle_frame(frame(), [add_signal(7)], {})

    assert len(worker.submitted) == 1  # attempted once, did not raise or block
    assert coordinator._reference_manager.state(7) is IdentificationState.WAITING_FOR_BETTER_VIEW


def test_queue_full_track_becomes_eligible_again_after_its_short_cooldown(
    store: DatabaseStore, worker: FakeWorker
) -> None:
    fake_time = [0.0]
    worker.max_size = 0
    event_engine = EventEngine(store)
    coordinator = IdentityCoordinator(
        scene_processor=Mock(),
        event_engine=event_engine,
        store=store,
        reid_model_name="test-model",
        max_references_per_item=2,
        initial_capture_spacing_frames=0,
        move_candidate_delay_frames=0,
        candidate_retry_frames=0,
        bad_mask_cooldown_seconds=5.0,
        queue_full_retry_seconds=0.5,
        clock=lambda: fake_time[0],
        worker=worker,
    )

    coordinator.handle_frame(frame(), [add_signal(7)], {})
    assert len(worker.submitted) == 1

    fake_time[0] = 0.2
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    assert len(worker.submitted) == 1  # still cooling down

    worker.max_size = 8
    fake_time[0] = 0.6
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    assert len(worker.submitted) == 2  # cooldown elapsed, worker has room now


def test_new_result_creates_item_and_queues_the_first_reference_capture(
    coordinator: IdentityCoordinator, worker: FakeWorker, store: DatabaseStore
) -> None:
    coordinator.handle_frame(frame(), [add_signal(7)], {})

    worker.push_result(
        IdentificationResult(kind="resolve", track_id=7, status="new", reference=reference())
    )
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})

    assert store.count_items() == 1
    item = store.list_present_items()[0]
    assert coordinator.event_engine.item_id_for_track(7) == item.id
    capture_jobs = [job for job in worker.submitted if job.kind == "capture"]
    assert len(capture_jobs) == 1
    assert capture_jobs[0].precomputed_reference is not None


def test_pending_result_defers_the_track_instead_of_retrying_immediately(
    coordinator: IdentityCoordinator, worker: FakeWorker, store: DatabaseStore
) -> None:
    coordinator.handle_frame(frame(), [add_signal(7)], {})

    worker.push_result(IdentificationResult(kind="resolve", track_id=7, status="pending"))
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})

    assert store.count_items() == 0
    assert coordinator._reference_manager.state(7) is IdentificationState.DEFERRED
    # Still visible next frame, but cooling down: must not resubmit.
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    assert len(worker.submitted) == 1


def test_deferred_track_is_resubmitted_only_after_its_cooldown(
    store: DatabaseStore, worker: FakeWorker
) -> None:
    fake_time = [0.0]
    event_engine = EventEngine(store)
    coordinator = IdentityCoordinator(
        scene_processor=Mock(),
        event_engine=event_engine,
        store=store,
        reid_model_name="test-model",
        max_references_per_item=2,
        initial_capture_spacing_frames=0,
        move_candidate_delay_frames=0,
        candidate_retry_frames=0,
        bad_mask_cooldown_seconds=5.0,
        queue_full_retry_seconds=0.5,
        clock=lambda: fake_time[0],
        worker=worker,
    )

    coordinator.handle_frame(frame(), [add_signal(7)], {})
    worker.push_result(IdentificationResult(kind="resolve", track_id=7, status="pending"))
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    assert len(worker.submitted) == 1

    fake_time[0] = 1.0
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    assert len(worker.submitted) == 1  # within the 5s cooldown

    fake_time[0] = 6.0
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    assert len(worker.submitted) == 2  # cooldown elapsed


def test_a_bad_mask_never_causes_an_immediate_retry_loop(
    coordinator: IdentityCoordinator, worker: FakeWorker
) -> None:
    coordinator.handle_frame(frame(), [add_signal(7)], {})
    worker.push_result(IdentificationResult(kind="resolve", track_id=7, status="pending"))

    for _ in range(40):
        coordinator.handle_frame(frame(), [], {7: make_detection(7)})

    # Same-frame cooldown never elapses (clock is fixed at 0.0): exactly one attempt total.
    assert len(worker.submitted) == 1


def test_removed_track_is_forgotten_so_a_reused_id_starts_fresh(
    coordinator: IdentityCoordinator, worker: FakeWorker
) -> None:
    coordinator.handle_frame(frame(), [add_signal(7)], {})
    worker.push_result(IdentificationResult(kind="resolve", track_id=7, status="pending"))
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})

    coordinator.handle_frame(frame(), [remove_signal(7)], {})
    worker.submitted.clear()
    coordinator.handle_frame(frame(), [add_signal(7)], {})

    assert len(worker.submitted) == 1  # a fresh submission, not blocked by old deferred state


def test_existing_match_against_removed_item_returns_it(
    coordinator: IdentityCoordinator, worker: FakeWorker, store: DatabaseStore
) -> None:
    item = store.create_item("cup", status=ItemStatus.REMOVED)
    coordinator.handle_frame(frame(), [add_signal(7)], {})

    worker.push_result(
        IdentificationResult(
            kind="resolve", track_id=7, status="existing", item_id=item.id, reference=reference()
        )
    )
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})

    saved_item = store.get_item(item.id)
    assert saved_item is not None
    assert saved_item.status is ItemStatus.PRESENT
    assert coordinator.event_engine.item_id_for_track(7) == item.id


def test_existing_match_against_present_item_only_associates(
    coordinator: IdentityCoordinator, worker: FakeWorker, store: DatabaseStore
) -> None:
    item = store.create_item("cup", status=ItemStatus.PRESENT)
    coordinator.handle_frame(frame(), [add_signal(7)], {})

    worker.push_result(
        IdentificationResult(
            kind="resolve", track_id=7, status="existing", item_id=item.id, reference=reference()
        )
    )
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})

    assert store.get_item_history(item.id) == []
    assert coordinator.event_engine.item_id_for_track(7) == item.id


def test_double_claim_on_same_item_defers_the_second_track(
    coordinator: IdentityCoordinator, worker: FakeWorker, store: DatabaseStore
) -> None:
    item = store.create_item("cup", status=ItemStatus.PRESENT)
    coordinator.handle_frame(frame(), [add_signal(7)], {})
    worker.push_result(
        IdentificationResult(
            kind="resolve", track_id=7, status="existing", item_id=item.id, reference=reference()
        )
    )
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})

    coordinator.handle_frame(frame(), [add_signal(8)], {})
    worker.push_result(
        IdentificationResult(
            kind="resolve", track_id=8, status="existing", item_id=item.id, reference=reference()
        )
    )
    coordinator.handle_frame(frame(), [], {7: make_detection(7), 8: make_detection(8)})

    assert coordinator.event_engine.item_id_for_track(8) is None
    assert coordinator._reference_manager.state(8) is IdentificationState.DEFERRED


def test_a_result_for_a_track_that_disappeared_is_handled_safely(
    coordinator: IdentityCoordinator, worker: FakeWorker, store: DatabaseStore
) -> None:
    coordinator.handle_frame(frame(), [add_signal(7)], {})
    coordinator.handle_frame(frame(), [remove_signal(7)], {})

    worker.push_result(
        IdentificationResult(kind="resolve", track_id=7, status="new", reference=reference())
    )
    # Must not raise, and must not create an item for a track that no longer exists.
    coordinator.handle_frame(frame(), [], {})

    assert store.count_items() == 0


def test_capture_job_is_not_duplicated_while_one_is_outstanding(
    coordinator: IdentityCoordinator, worker: FakeWorker, store: DatabaseStore
) -> None:
    coordinator.handle_frame(frame(), [add_signal(7)], {})
    worker.push_result(
        IdentificationResult(kind="resolve", track_id=7, status="new", reference=reference())
    )
    # Applying the "new" result itself queues the first capture job.
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    capture_jobs = [job for job in worker.submitted if job.kind == "capture"]
    assert len(capture_jobs) == 1

    # The item is now IDENTIFIED with a capture QUEUED; further frames must not
    # submit a second capture job until the outstanding one's result lands.
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    capture_jobs = [job for job in worker.submitted if job.kind == "capture"]
    assert len(capture_jobs) == 1

    # The outstanding capture resolves without completing the baseline (this result
    # carries no candidate_kind), so the still-owed baseline reference is nominated.
    item_id = store.list_present_items()[0].id
    worker.push_result(
        IdentificationResult(
            kind="capture", track_id=7, status="captured", item_id=item_id, reference=reference()
        )
    )
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    capture_jobs = [job for job in worker.submitted if job.kind == "capture"]
    assert len(capture_jobs) == 2  # eligible again now that the prior capture resolved


def test_static_item_stops_capturing_once_its_baseline_is_complete(
    coordinator: IdentityCoordinator, worker: FakeWorker, store: DatabaseStore
) -> None:
    """A still object must not keep producing near-identical references forever."""
    coordinator.handle_frame(frame(), [add_signal(7)], {})
    worker.push_result(
        IdentificationResult(kind="resolve", track_id=7, status="new", reference=reference())
    )
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    item_id = store.list_present_items()[0].id

    # Baseline reference 1 is the resolve embedding; completing reference 2 ends
    # the baseline (initial_reference_count defaults to 2).
    worker.push_result(
        IdentificationResult(
            kind="capture",
            track_id=7,
            status="captured",
            item_id=item_id,
            reference=reference(),
            candidate_kind=CandidateKind.INITIAL,
        )
    )
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    baseline_jobs = len([job for job in worker.submitted if job.kind == "capture"])

    for _ in range(50):
        coordinator.handle_frame(frame(), [], {7: make_detection(7)})

    assert coordinator.reference_policy.state(7) is ReferencePolicyState.IDLE
    assert len([job for job in worker.submitted if job.kind == "capture"]) == baseline_jobs


def test_move_start_arms_capture_and_is_never_persisted_as_an_event(
    coordinator: IdentityCoordinator, worker: FakeWorker, store: DatabaseStore
) -> None:
    coordinator.handle_frame(frame(), [add_signal(7)], {})
    worker.push_result(
        IdentificationResult(kind="resolve", track_id=7, status="new", reference=reference())
    )
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    item_id = store.list_present_items()[0].id
    worker.push_result(
        IdentificationResult(
            kind="capture",
            track_id=7,
            status="captured",
            item_id=item_id,
            reference=reference(),
            candidate_kind=CandidateKind.INITIAL,
        )
    )
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    history_before = len(store.get_item_history(item_id))
    settled_jobs = len([job for job in worker.submitted if job.kind == "capture"])

    coordinator.handle_frame(frame(), [move_start_signal(7)], {7: make_detection(7)})

    assert coordinator.reference_policy.state(7) is ReferencePolicyState.ARMED
    movement_jobs = [
        job
        for job in worker.submitted
        if job.kind == "capture" and job.candidate_kind is CandidateKind.MOVEMENT
    ]
    assert len(movement_jobs) == 1
    assert len([job for job in worker.submitted if job.kind == "capture"]) == settled_jobs + 1
    # A runtime transition only: it must not write an ItemEvent.
    assert len(store.get_item_history(item_id)) == history_before


def test_movement_candidates_are_bounded_per_movement_event(
    coordinator: IdentityCoordinator, worker: FakeWorker, store: DatabaseStore
) -> None:
    coordinator.handle_frame(frame(), [add_signal(7)], {})
    worker.push_result(
        IdentificationResult(kind="resolve", track_id=7, status="new", reference=reference())
    )
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    item_id = store.list_present_items()[0].id
    coordinator.handle_frame(frame(), [move_start_signal(7)], {})

    # Each frame resolves the outstanding job immediately, so only the policy's
    # attempt budget limits how many candidates one movement can produce.
    for _ in range(30):
        worker.push_result(
            IdentificationResult(
                kind="capture",
                track_id=7,
                status="skipped",
                item_id=item_id,
                candidate_kind=CandidateKind.MOVEMENT,
            )
        )
        coordinator.handle_frame(frame(), [], {7: make_detection(7)})

    movement_jobs = [
        job
        for job in worker.submitted
        if job.kind == "capture" and job.candidate_kind is CandidateKind.MOVEMENT
    ]
    assert len(movement_jobs) == coordinator.max_movement_reference_attempts
    assert coordinator.reference_policy.state(7) is ReferencePolicyState.EXHAUSTED


def test_move_end_nominates_exactly_one_final_candidate_then_idles(
    coordinator: IdentityCoordinator, worker: FakeWorker, store: DatabaseStore
) -> None:
    coordinator.handle_frame(frame(), [add_signal(7)], {})
    worker.push_result(
        IdentificationResult(kind="resolve", track_id=7, status="new", reference=reference())
    )
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    item_id = store.list_present_items()[0].id
    worker.push_result(
        IdentificationResult(
            kind="capture",
            track_id=7,
            status="captured",
            item_id=item_id,
            reference=reference(),
            candidate_kind=CandidateKind.INITIAL,
        )
    )
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})

    coordinator.handle_frame(frame(), [move_end_signal(7)], {7: make_detection(7)})
    worker.push_result(
        IdentificationResult(
            kind="capture",
            track_id=7,
            status="captured",
            item_id=item_id,
            reference=reference(),
            candidate_kind=CandidateKind.MOVE_END,
        )
    )
    for _ in range(20):
        coordinator.handle_frame(frame(), [], {7: make_detection(7)})

    move_end_jobs = [
        job
        for job in worker.submitted
        if job.kind == "capture" and job.candidate_kind is CandidateKind.MOVE_END
    ]
    assert len(move_end_jobs) == 1
    assert coordinator.reference_policy.state(7) is ReferencePolicyState.IDLE


def test_moved_signal_before_identity_resolves_is_dropped_not_crashed(
    coordinator: IdentityCoordinator, worker: FakeWorker, store: DatabaseStore
) -> None:
    # Track 7 is ADDed but its resolve job has not returned a result yet, so it
    # has no item_id. The tracker can still emit MOVED (placement/movement logic
    # is independent of identity resolution); this must not raise.
    coordinator.handle_frame(frame(), [add_signal(7)], {})

    coordinator.handle_frame(frame(), [moved_signal(7)], {7: make_detection(7)})

    assert coordinator.event_engine.item_id_for_track(7) is None


def test_moved_signal_after_identity_resolves_is_recorded_normally(
    coordinator: IdentityCoordinator, worker: FakeWorker, store: DatabaseStore
) -> None:
    coordinator.handle_frame(frame(), [add_signal(7)], {})
    worker.push_result(
        IdentificationResult(kind="resolve", track_id=7, status="new", reference=reference())
    )
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    item_id = store.list_present_items()[0].id

    coordinator.handle_frame(frame(), [moved_signal(7)], {7: make_detection(7)})

    assert store.get_item_history(item_id)[-1].event_type.value == "moved"


def test_resolve_jobs_carry_query_id_detection_and_frame_index_when_diagnostics_are_on(
    coordinator: IdentityCoordinator, worker: FakeWorker
) -> None:
    diagnostics = Mock()
    diagnostics.new_query_id.return_value = "run-q00001"
    coordinator.diagnostics = diagnostics

    coordinator.handle_frame(frame(), [add_signal(7)], {})

    job = worker.submitted[0]
    assert job.query_id == "run-q00001"
    assert job.frame_index == 1
    assert job.detection is not None and job.detection.track_id == 7


def test_resolve_jobs_have_no_query_id_when_diagnostics_are_off(
    coordinator: IdentityCoordinator, worker: FakeWorker
) -> None:
    coordinator.handle_frame(frame(), [add_signal(7)], {})

    assert worker.submitted[0].query_id is None


def test_applied_outcomes_are_recorded_against_their_query(
    coordinator: IdentityCoordinator, worker: FakeWorker, store: DatabaseStore
) -> None:
    diagnostics = Mock()
    coordinator.diagnostics = diagnostics
    coordinator.handle_frame(frame(), [add_signal(7)], {})

    worker.push_result(
        IdentificationResult(
            kind="resolve", track_id=7, status="new", reference=reference(), query_id="q1"
        )
    )
    coordinator.handle_frame(frame(), [], {7: make_detection(7)})
    item_id = store.list_present_items()[0].id
    coordinator.handle_frame(frame(), [remove_signal(7)], {})

    # Tracker retirement starts/continues item absence; it no longer writes REMOVED.
    coordinator.clock.return_value = 2.0
    coordinator.handle_frame(frame(), [], {})

    coordinator.handle_frame(frame(), [add_signal(8)], {})
    worker.push_result(
        IdentificationResult(
            kind="resolve", track_id=8, status="existing", item_id=item_id, query_id="q2"
        )
    )
    coordinator.handle_frame(frame(), [], {8: make_detection(8)})

    worker.push_result(
        IdentificationResult(kind="resolve", track_id=99, status="new", query_id="q3")
    )
    coordinator.handle_frame(frame(), [], {})

    assert [call.args for call in diagnostics.record_outcome.call_args_list] == [
        ("q1", item_id, "ADDED"),
        ("q2", item_id, "RETURNED"),
        ("q3", None, "STALE"),
    ]


# --- Scenery rejection and track-level class voting (Phase A step 2) ---------


def test_a_box_covering_most_of_the_frame_is_rejected_as_scenery(
    store: DatabaseStore, worker: FakeWorker
) -> None:
    coordinator = IdentityCoordinator(
        scene_processor=Mock(),
        event_engine=EventEngine(store),
        store=store,
        reid_model_name="test-model",
        max_references_per_item=2,
        max_box_area_fraction=0.35,
        clock=Mock(return_value=0.0),
        worker=worker,
    )
    # 500x400 of a 640x480 frame: 65% of the view, i.e. the table, not an object.
    table = TrackSignal(
        TrackSignalType.ADD,
        7,
        make_detection(7, box=(10, 10, 510, 410)),
    )

    coordinator.handle_frame(frame(), [table], {})

    assert worker.submitted == []
    assert coordinator._reference_manager.reason(7) == "box_too_large"


def test_a_normal_object_is_unaffected_by_the_size_ceiling(
    store: DatabaseStore, worker: FakeWorker
) -> None:
    coordinator = IdentityCoordinator(
        scene_processor=Mock(),
        event_engine=EventEngine(store),
        store=store,
        reid_model_name="test-model",
        max_references_per_item=2,
        max_box_area_fraction=0.35,
        clock=Mock(return_value=0.0),
        worker=worker,
    )

    coordinator.handle_frame(frame(), [add_signal(7)], {})

    assert len(worker.submitted) == 1


def test_the_size_ceiling_is_disabled_at_one(store: DatabaseStore, worker: FakeWorker) -> None:
    coordinator = IdentityCoordinator(
        scene_processor=Mock(),
        event_engine=EventEngine(store),
        store=store,
        reid_model_name="test-model",
        max_references_per_item=2,
        max_box_area_fraction=1.0,
        clock=Mock(return_value=0.0),
        worker=worker,
    )
    whole_frame = TrackSignal(TrackSignalType.ADD, 7, make_detection(7, box=(0, 0, 640, 480)))

    coordinator.handle_frame(frame(), [whole_frame], {})

    assert len(worker.submitted) == 1


def test_a_flickering_class_label_never_withholds_identification(
    coordinator: IdentityCoordinator, worker: FakeWorker
) -> None:
    # Identification is class-agnostic because detector labels are unreliable: a
    # track whose label keeps flipping (a knife read as scissors and back) must
    # still be identified normally.
    for class_name in ("knife", "scissors", "knife", "scissors"):
        detection = Detection(
            track_id=7, class_id=43, class_name=class_name, confidence=0.9, box=(10, 20, 110, 220)
        )
        coordinator.handle_frame(frame(), [], {7: detection})

    coordinator.handle_frame(frame(), [add_signal(7)], {})

    assert len(worker.submitted) == 1
