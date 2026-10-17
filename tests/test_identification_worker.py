"""Standalone IdentificationWorker tests: job->result mapping, queue, and shutdown."""

from __future__ import annotations

import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from project_auto.events.identification import IdentificationJob, IdentificationResult
from project_auto.events.identification_worker import IdentificationWorker
from project_auto.memory.reid import CandidateScore
from project_auto.perception.detector import Detection
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
        max_references_per_item=2,
        job_queue_max_size=2,
    )


def test_submit_returns_false_when_queue_is_full(
    worker: IdentificationWorker, scene_processor: Mock
) -> None:
    job = IdentificationJob(
        kind="resolve", track_id=1, frame=np.zeros((4, 4, 3), dtype=np.uint8), box=(0, 0, 4, 4)
    )
    entered = threading.Event()
    release = threading.Event()

    def process(*args):
        entered.set()
        assert release.wait(5.0)
        return IdentityDecision("pending", 1, None, 0.0, None)

    scene_processor.process.side_effect = process
    worker.start()
    try:
        assert worker.submit(job) is True
        assert entered.wait(2.0)
        assert worker.submit(job) is True
        assert worker.submit(job) is True
        assert worker.submit(job) is False  # queue_max_size=2
        started = time.monotonic()
        worker.stop(timeout=0.05)
        assert time.monotonic() - started < 1.0
        assert worker._thread is not None
        assert worker._thread.is_alive()
        assert worker.submit(job) is False
    finally:
        release.set()
        worker.stop(timeout=2.0)
    assert worker._thread is None
    assert scene_processor.process.call_count == 1


def test_start_propagates_gallery_failure(worker: IdentificationWorker, store: Mock) -> None:
    failure = RuntimeError("missing gallery column")
    store.load_reid_gallery.side_effect = failure
    with pytest.raises(RuntimeError, match="failed to load its gallery") as caught:
        worker.start()
    assert caught.value.__cause__ is failure
    assert worker.submit(IdentificationJob(kind="resolve", track_id=1)) is False
    assert worker.poll_results() == []
    with pytest.raises(RuntimeError, match="failed to load its gallery"):
        worker.start()
    worker.stop(timeout=0.1)
    worker.stop(timeout=0.1)


def test_start_waits_for_gallery_loaded_on_worker(
    worker: IdentificationWorker, store: Mock
) -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    errors = []
    gallery = [object()]

    def load(*args, **kwargs):
        assert threading.current_thread() is worker._thread
        entered.set()
        assert release.wait(5.0)
        return gallery

    def start():
        try:
            worker.start()
        except BaseException as exc:  # noqa: BLE001 - collect any failure from the test thread
            errors.append(exc)
        finally:
            finished.set()

    store.load_reid_gallery.side_effect = load
    caller = threading.Thread(target=start)
    caller.start()
    try:
        assert entered.wait(2.0)
        assert not finished.wait(0.05)
        assert worker.submit(IdentificationJob(kind="resolve", track_id=1)) is False
        release.set()
        assert finished.wait(2.0)
        assert errors == []
        assert worker._gallery is gallery
        worker.start()  # Already started: do not reload.
        assert store.load_reid_gallery.call_count == 1
    finally:
        release.set()
        caller.join(2.0)
        worker.stop(timeout=2.0)


def test_submit_rejects_unstarted_and_stopped_worker(worker: IdentificationWorker) -> None:
    job = IdentificationJob(kind="resolve", track_id=1)
    assert worker.submit(job) is False
    worker.start()
    worker.stop(timeout=2.0)
    assert worker.submit(job) is False
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        worker.start()


def test_process_resolve_maps_new_decision(
    worker: IdentificationWorker, scene_processor: Mock
) -> None:
    scene_processor.process.return_value = IdentityDecision("new", 7, None, 0.2, reference())
    job = IdentificationJob(
        kind="resolve", track_id=7, frame=np.zeros((4, 4, 3), dtype=np.uint8), box=(0, 0, 4, 4)
    )

    result = worker._process(job)

    assert result.kind == "resolve"
    assert result.track_id == 7
    assert result.status == "new"
    assert result.item_id is None
    assert result.similarity == pytest.approx(0.2)
    assert result.reference is not None


def test_process_resolve_maps_existing_decision(
    worker: IdentificationWorker, scene_processor: Mock
) -> None:
    scene_processor.process.return_value = IdentityDecision("existing", 7, 42, 0.9, reference())
    job = IdentificationJob(
        kind="resolve", track_id=7, frame=np.zeros((4, 4, 3), dtype=np.uint8), box=(0, 0, 4, 4)
    )

    result = worker._process(job)

    assert result.status == "existing"
    assert result.item_id == 42
    assert result.similarity == pytest.approx(0.9)


def test_process_resolve_maps_pending_decision_without_running_capture(
    worker: IdentificationWorker, scene_processor: Mock, store: Mock
) -> None:
    scene_processor.process.return_value = IdentityDecision("pending", 7, None, 0.0, None)
    job = IdentificationJob(
        kind="resolve", track_id=7, frame=np.zeros((4, 4, 3), dtype=np.uint8), box=(0, 0, 4, 4)
    )

    result = worker._process(job)

    assert result.status == "pending"
    assert result.reference is None
    store.add_reference_if_needed.assert_not_called()


def test_resolve_job_with_query_id_records_query_candidates_and_crops(
    worker: IdentificationWorker, scene_processor: Mock
) -> None:
    diagnostics = Mock()
    worker.diagnostics = diagnostics
    candidates = (
        CandidateScore(42, 0.9, 0.8, 0.7, 0.95, 3),
        CandidateScore(9, 0.3, 0.2, 0.4, 0.9, 2),
    )
    scene_processor.process.return_value = IdentityDecision(
        "existing", 7, 42, 0.9, reference(), candidates=candidates, reason="accepted"
    )
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    detection = Detection(
        track_id=7, class_id=41, class_name="cup", confidence=0.8, box=(2, 3, 12, 8)
    )
    job = IdentificationJob(
        kind="resolve",
        track_id=7,
        frame=frame,
        box=(2, 3, 12, 8),  # extends past the frame; the raw crop is clipped
        detection=detection,
        query_id="run-q00001",
        frame_index=55,
    )

    result = worker._process(job)

    assert result.query_id == "run-q00001"
    query, recorded = diagnostics.record_query.call_args.args
    assert query.query_id == "run-q00001"
    assert query.frame_index == 55
    assert query.det_confidence == pytest.approx(0.8)
    assert query.decision == "MATCH"
    assert query.matched_item_id == 42
    assert [candidate.item_id for candidate in recorded] == [42, 9]
    assert recorded[0].n_references == 3
    kwargs = diagnostics.record_query.call_args.kwargs
    assert kwargs["raw_crop_bgr"].shape == (5, 8, 3)
    assert kwargs["masked_crop"] is not None


@pytest.mark.parametrize(
    ("decision", "reason", "label"),
    [
        ("new", "low_margin", "AMBIG"),
        ("new", "below_threshold", "NEW"),
        ("pending", "no_mask", "DEFER"),
    ],
)
def test_recorded_decision_label_separates_ambiguity_from_new(
    worker: IdentificationWorker, scene_processor: Mock, decision: str, reason: str, label: str
) -> None:
    worker.diagnostics = Mock()
    ref = None if decision == "pending" else reference()
    scene_processor.process.return_value = IdentityDecision(
        decision, 7, None, 0.0, ref, reason=reason
    )
    job = IdentificationJob(
        kind="resolve",
        track_id=7,
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        box=(0, 0, 4, 4),
        query_id="q",
    )

    worker._process(job)

    query, _ = worker.diagnostics.record_query.call_args.args
    assert query.decision == label
    assert query.decision_reason == reason


def test_nothing_is_recorded_without_a_query_id_or_for_capture_jobs(
    worker: IdentificationWorker, scene_processor: Mock, store: Mock
) -> None:
    worker.diagnostics = Mock()
    scene_processor.process.return_value = IdentityDecision("new", 7, None, 0.2, reference())
    store.add_reference_if_needed.return_value = (True, 1)

    worker._process(
        IdentificationJob(
            kind="resolve", track_id=7, frame=np.zeros((4, 4, 3), dtype=np.uint8), box=(0, 0, 4, 4)
        )
    )
    worker._process(
        IdentificationJob(
            kind="capture", track_id=7, item_id=9, precomputed_reference=reference(), query_id="q"
        )
    )

    worker.diagnostics.record_query.assert_not_called()


def test_process_capture_with_precomputed_reference_skips_resegmenting(
    worker: IdentificationWorker, scene_processor: Mock, store: Mock
) -> None:
    store.add_reference_if_needed.return_value = (True, 1)
    job = IdentificationJob(
        kind="capture", track_id=7, item_id=9, precomputed_reference=reference()
    )

    result = worker._process(job)

    assert result.status == "captured"
    scene_processor.prepare_reference.assert_not_called()
    store.add_reference_if_needed.assert_called_once()


def test_process_capture_below_target_is_skipped_when_store_declines(
    worker: IdentificationWorker, store: Mock
) -> None:
    store.add_reference_if_needed.return_value = (False, 2)
    job = IdentificationJob(
        kind="capture", track_id=7, item_id=9, precomputed_reference=reference()
    )

    result = worker._process(job)

    assert result.status == "skipped"


def test_process_capture_with_bad_mask_returns_pending_without_saving(
    worker: IdentificationWorker, scene_processor: Mock, store: Mock
) -> None:
    scene_processor.prepare_reference.return_value = None
    job = IdentificationJob(
        kind="capture",
        track_id=7,
        item_id=9,
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        box=(0, 0, 4, 4),
    )

    result = worker._process(job)

    assert result.status == "pending"
    store.add_reference_if_needed.assert_not_called()


def test_a_raising_job_yields_an_error_result_instead_of_crashing_the_worker(
    worker: IdentificationWorker, scene_processor: Mock
) -> None:
    scene_processor.process.side_effect = RuntimeError("boom")
    job = IdentificationJob(
        kind="resolve", track_id=7, frame=np.zeros((4, 4, 3), dtype=np.uint8), box=(0, 0, 4, 4)
    )

    result = worker._process_safely(job)

    assert result.status == "error"
    assert result.track_id == 7


def test_worker_thread_processes_a_job_end_to_end_and_shuts_down_cleanly(
    worker: IdentificationWorker, scene_processor: Mock
) -> None:
    scene_processor.process.return_value = IdentityDecision("new", 7, None, 0.0, reference())
    job = IdentificationJob(
        kind="resolve", track_id=7, frame=np.zeros((4, 4, 3), dtype=np.uint8), box=(0, 0, 4, 4)
    )

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


@pytest.mark.parametrize("raises", [False, True])
def test_worker_echoes_job_id_on_success_and_error(worker, scene_processor, raises):
    if raises:
        scene_processor.process.side_effect = RuntimeError("failed view")
    else:
        scene_processor.process.return_value = IdentityDecision("pending", 7, None, 0.0, None)
    job = IdentificationJob("resolve", 7, frame=np.zeros((4, 4, 3)), box=(0, 0, 4, 4), job_id=123)
    assert worker._process_safely(job).job_id == 123


def test_priority_handoff_runs_after_active_job_before_waiting_regular_job(worker, scene_processor):
    entered = threading.Event()
    release = threading.Event()
    order = []

    def process(frame, box, track_id, gallery):
        order.append(track_id)
        if track_id == 1:
            entered.set()
            assert release.wait(5)
        return IdentityDecision("pending", track_id, None, 0.0, None)

    scene_processor.process.side_effect = process
    worker.start()
    try:

        def job(track_id, priority=False):
            return IdentificationJob(
                "resolve",
                track_id,
                frame=np.zeros((4, 4, 3)),
                box=(0, 0, 4, 4),
                priority=priority,
                job_id=track_id,
            )

        assert worker.submit(job(1))
        assert entered.wait(2)
        assert worker.submit(job(2))
        assert worker.submit(job(3, True))
        release.set()
        results = []
        deadline = time.monotonic() + 3
        while len(results) < 3 and time.monotonic() < deadline:
            results.extend(worker.poll_results())
            if len(results) < 3:
                time.sleep(0.01)
        assert order == [1, 3, 2]
        assert [result.job_id for result in results] == [1, 3, 2]
    finally:
        release.set()
        worker.stop(2)


class TestReferenceThumbnails:
    """The worker writes a picture beside each stored reference, and only then."""

    @pytest.fixture
    def images(self, tmp_path):
        from project_auto.utils.reference_images import ReferenceImageStore

        return ReferenceImageStore(root=tmp_path / "images")

    def capture_job(self) -> IdentificationJob:
        return IdentificationJob(
            kind="capture", track_id=7, item_id=9, precomputed_reference=reference()
        )

    def test_a_stored_reference_records_the_path_of_its_thumbnail(
        self, worker: IdentificationWorker, store: Mock, images
    ) -> None:
        worker.reference_images = images
        store.add_reference_if_needed.return_value = (True, 1)

        worker._process(self.capture_job())

        recorded = store.add_reference_if_needed.call_args.kwargs["object_image_path"]
        assert recorded is not None
        assert (images.root / recorded).is_file()

    def test_without_a_thumbnail_store_references_are_stored_as_before(
        self, worker: IdentificationWorker, store: Mock
    ) -> None:
        store.add_reference_if_needed.return_value = (True, 1)

        worker._process(self.capture_job())

        assert store.add_reference_if_needed.call_args.kwargs["object_image_path"] is None

    def test_a_refused_reference_leaves_no_orphan_picture(
        self, worker: IdentificationWorker, store: Mock, images
    ) -> None:
        """The gallery-full path: the picture is written first, so it must be removed."""
        worker.reference_images = images
        store.add_reference_if_needed.return_value = (False, 2)

        result = worker._process(self.capture_job())

        assert result.status == "skipped"
        assert not list(images.root.rglob("*.png"))

    def test_a_failed_picture_never_loses_the_reference(
        self, worker: IdentificationWorker, store: Mock, tmp_path
    ) -> None:
        from project_auto.utils.reference_images import ReferenceImageStore

        blocked = tmp_path / "blocked"
        blocked.write_text("a file, not a folder", encoding="utf-8")
        worker.reference_images = ReferenceImageStore(root=blocked)
        store.add_reference_if_needed.return_value = (True, 1)

        result = worker._process(self.capture_job())

        assert result.status == "captured"
        assert store.add_reference_if_needed.call_args.kwargs["object_image_path"] is None

    def test_a_rejected_candidate_writes_no_picture(
        self, worker: IdentificationWorker, store: Mock, images
    ) -> None:
        """Rejected views never reach storage, so they must not reach the disk either."""
        worker.reference_images = images
        worker.min_mask_score = 0.9  # reference() reports sam_score 0.0

        result = worker._process(self.capture_job())

        assert result.status == "skipped"
        store.add_reference_if_needed.assert_not_called()
        assert not images.root.exists()

    def test_the_picture_is_written_on_the_worker_thread_not_the_caller(
        self, worker: IdentificationWorker, store: Mock, tmp_path
    ) -> None:
        from project_auto.utils.reference_images import ReferenceImageStore

        threads: list[str] = []

        def name() -> str:
            # Called from inside save(), so it reports which thread is doing the disk write.
            threads.append(threading.current_thread().name)
            return "view"

        worker.reference_images = ReferenceImageStore(root=tmp_path / "images", name_factory=name)
        store.add_reference_if_needed.return_value = (True, 1)
        worker.start()
        try:
            worker.submit(self.capture_job())
            deadline = time.monotonic() + 5
            while not threads and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            worker.stop()

        assert threads == ["identification-worker"]


class TestConfirmedViews:
    """A view the user confirmed skips the similarity gates, and only those."""

    @pytest.fixture
    def gated(self, worker: IdentificationWorker):
        """A worker with the real gates on and item 9 already holding one reference."""
        from project_auto.memory.reid import GalleryEntry

        worker.min_reference_consistency = 0.5
        worker.reference_novelty_threshold = 0.93
        worker.min_mask_score = 0.6
        worker.min_mask_occupancy = 0.1
        worker._gallery = [GalleryEntry(item_id=9, embeddings=np.array([[1.0, 0.0]], np.float32))]
        return worker

    def job(self, embedding, confirmed: bool, sam: float = 0.9, occupancy: float = 0.7):
        return IdentificationJob(
            kind="capture",
            track_id=7,
            item_id=9,
            confirmed=confirmed,
            precomputed_reference=PreparedReference(
                crop=Image.new("RGB", (4, 4)),
                embedding=np.array(embedding, dtype=np.float32),
                sam_score=sam,
                mask_occupancy=occupancy,
            ),
        )

    def test_a_view_too_different_is_rejected_unless_the_user_confirmed_it(
        self, gated: IdentificationWorker, store: Mock
    ) -> None:
        """The phone's back looks nothing like its front, which is exactly the point."""
        store.add_reference_if_needed.return_value = (True, 2)
        different = [0.0, 1.0]

        assert gated._process(self.job(different, confirmed=False)).failure_reason == "inconsistent"
        assert gated._process(self.job(different, confirmed=True)).status == "captured"

    def test_a_view_too_similar_is_rejected_unless_the_user_confirmed_it(
        self, gated: IdentificationWorker, store: Mock
    ) -> None:
        store.add_reference_if_needed.return_value = (True, 2)
        same = [1.0, 0.0]

        assert gated._process(self.job(same, confirmed=False)).failure_reason == "redundant"
        assert gated._process(self.job(same, confirmed=True)).status == "captured"

    def test_a_confirmed_view_still_needs_a_usable_mask(
        self, gated: IdentificationWorker, store: Mock
    ) -> None:
        """Being told it is the right item does not make a bad crop worth learning from."""
        different = [0.0, 1.0]

        poor_score = gated._process(self.job(different, confirmed=True, sam=0.3))
        thin_mask = gated._process(self.job(different, confirmed=True, occupancy=0.02))

        assert poor_score.failure_reason == "low_mask_score"
        assert thin_mask.failure_reason == "low_mask_occupancy"
        store.add_reference_if_needed.assert_not_called()

    def test_a_confirmed_view_still_respects_the_reference_cap(
        self, gated: IdentificationWorker, store: Mock
    ) -> None:
        store.add_reference_if_needed.return_value = (False, 8)

        result = gated._process(self.job([0.0, 1.0], confirmed=True))

        assert result.status == "skipped" and result.failure_reason == "gallery_full"
