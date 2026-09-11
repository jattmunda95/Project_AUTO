"""Standalone IdentificationWorker tests: job->result mapping, queue, and shutdown."""

from __future__ import annotations

import time
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from project_auto.events.identification import IdentificationJob, IdentificationResult
from project_auto.events.identification_worker import IdentificationWorker
from project_auto.perception.scene_processor import IdentityDecision, PreparedReference


def reference() -> PreparedReference:
    return PreparedReference(crop=Image.new("RGB", (4, 4)), embedding=np.array([1.0, 0.0]))


@pytest.fixture
def scene_processor() -> Mock:
    return Mock()


@pytest.fixture
def store() -> Mock:
    return Mock()


@pytest.fixture
def worker(scene_processor: Mock, store: Mock) -> IdentificationWorker:
    store.load_reid_gallery.return_value = []
    return IdentificationWorker(
        scene_processor=scene_processor,
        store=store,
        reid_model_name="test-model",
        reference_target_count=2,
        job_queue_max_size=2,
    )


def test_submit_returns_false_when_queue_is_full(worker: IdentificationWorker) -> None:
    job = IdentificationJob(kind="resolve", track_id=1, frame=np.zeros((4, 4, 3), dtype=np.uint8), box=(0, 0, 4, 4))

    assert worker.submit(job) is True
    assert worker.submit(job) is True
    assert worker.submit(job) is False  # queue_max_size=2


def test_process_resolve_maps_new_decision(worker: IdentificationWorker, scene_processor: Mock) -> None:
    scene_processor.process.return_value = IdentityDecision("new", 7, None, 0.2, reference())
    job = IdentificationJob(kind="resolve", track_id=7, frame=np.zeros((4, 4, 3), dtype=np.uint8), box=(0, 0, 4, 4))

    result = worker._process(job)

    assert result.kind == "resolve"
    assert result.track_id == 7
    assert result.status == "new"
    assert result.item_id is None
    assert result.similarity == pytest.approx(0.2)
    assert result.reference is not None


def test_process_resolve_maps_existing_decision(worker: IdentificationWorker, scene_processor: Mock) -> None:
    scene_processor.process.return_value = IdentityDecision("existing", 7, 42, 0.9, reference())
    job = IdentificationJob(kind="resolve", track_id=7, frame=np.zeros((4, 4, 3), dtype=np.uint8), box=(0, 0, 4, 4))

    result = worker._process(job)

    assert result.status == "existing"
    assert result.item_id == 42
    assert result.similarity == pytest.approx(0.9)


def test_process_resolve_maps_pending_decision_without_running_capture(
    worker: IdentificationWorker, scene_processor: Mock, store: Mock
) -> None:
    scene_processor.process.return_value = IdentityDecision("pending", 7, None, 0.0, None)
    job = IdentificationJob(kind="resolve", track_id=7, frame=np.zeros((4, 4, 3), dtype=np.uint8), box=(0, 0, 4, 4))

    result = worker._process(job)

    assert result.status == "pending"
    assert result.reference is None
    store.add_reference_if_needed.assert_not_called()


def test_process_capture_with_precomputed_reference_skips_resegmenting(
    worker: IdentificationWorker, scene_processor: Mock, store: Mock
) -> None:
    store.add_reference_if_needed.return_value = (True, 1)
    job = IdentificationJob(kind="capture", track_id=7, item_id=9, precomputed_reference=reference())

    result = worker._process(job)

    assert result.status == "captured"
    scene_processor.prepare_reference.assert_not_called()
    store.add_reference_if_needed.assert_called_once()


def test_process_capture_below_target_is_skipped_when_store_declines(
    worker: IdentificationWorker, store: Mock
) -> None:
    store.add_reference_if_needed.return_value = (False, 2)
    job = IdentificationJob(kind="capture", track_id=7, item_id=9, precomputed_reference=reference())

    result = worker._process(job)

    assert result.status == "skipped"


def test_process_capture_with_bad_mask_returns_pending_without_saving(
    worker: IdentificationWorker, scene_processor: Mock, store: Mock
) -> None:
    scene_processor.prepare_reference.return_value = None
    job = IdentificationJob(
        kind="capture", track_id=7, item_id=9, frame=np.zeros((4, 4, 3), dtype=np.uint8), box=(0, 0, 4, 4)
    )

    result = worker._process(job)

    assert result.status == "pending"
    store.add_reference_if_needed.assert_not_called()


def test_a_raising_job_yields_an_error_result_instead_of_crashing_the_worker(
    worker: IdentificationWorker, scene_processor: Mock
) -> None:
    scene_processor.process.side_effect = RuntimeError("boom")
    job = IdentificationJob(kind="resolve", track_id=7, frame=np.zeros((4, 4, 3), dtype=np.uint8), box=(0, 0, 4, 4))

    result = worker._process_safely(job)

    assert result.status == "error"
    assert result.track_id == 7


def test_worker_thread_processes_a_job_end_to_end_and_shuts_down_cleanly(
    worker: IdentificationWorker, scene_processor: Mock
) -> None:
    scene_processor.process.return_value = IdentityDecision("new", 7, None, 0.0, reference())
    job = IdentificationJob(kind="resolve", track_id=7, frame=np.zeros((4, 4, 3), dtype=np.uint8), box=(0, 0, 4, 4))

    worker.start()
    try:
        assert worker.submit(job) is True
        deadline = time.monotonic() + 2.0
        results: list[IdentificationResult] = []
        while time.monotonic() < deadline and not results:
            results = worker.poll_results()
            if not results:
                time.sleep(0.01)
        assert len(results) == 1
        assert results[0].status == "new"
    finally:
        worker.stop(timeout=2.0)


def test_stop_is_idempotent_and_does_not_hang(worker: IdentificationWorker) -> None:
    worker.start()
    worker.stop(timeout=2.0)
    worker.stop(timeout=2.0)  # must not raise or block
