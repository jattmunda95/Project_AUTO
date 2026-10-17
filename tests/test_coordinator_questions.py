"""Coordinator + question tests: the whole ask-and-answer flow, driven end to end.

The real coordinator, policy, registry and event engine run against an in-memory database
and a fake worker, with time advanced by hand. The scores are real ones from
logs/reid_runs/20261016T030339Z. Each test follows one object through one outcome.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import test_coordinator as shared
from test_coordinator import FakeWorker, add_signal, frame, make_detection, remove_signal
from test_coordinator_policy import good_reference, near_miss_result, score

from project_auto.events.coordinator import IdentityCoordinator
from project_auto.events.event_engine import EventEngine
from project_auto.events.identification import IdentificationResult, IdentificationState
from project_auto.events.identity_policy import IdentityPolicy, IdentityPolicyConfig
from project_auto.events.question_registry import Answer, QuestionRegistry, QuestionState
from project_auto.memory.models import ItemStatus
from project_auto.memory.store import DatabaseStore

# Register the shared fixtures explicitly; no camera, models, or live database.
store = shared.store
worker = shared.worker

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "identity_policy.yaml"
TRACK = 7


@pytest.fixture
def config() -> IdentityPolicyConfig:
    return replace(IdentityPolicyConfig.from_yaml(CONFIG_PATH), mode="enforce")


@pytest.fixture
def seeded(store: DatabaseStore) -> DatabaseStore:
    """A store holding items 1..8 so candidate ids refer to real rows."""
    for _ in range(8):
        store.create_item("thing")
    return store


@pytest.fixture
def registry(config: IdentityPolicyConfig) -> QuestionRegistry:
    return QuestionRegistry(config)


@pytest.fixture
def coordinator(
    seeded: DatabaseStore,
    worker: FakeWorker,
    config: IdentityPolicyConfig,
    registry: QuestionRegistry,
) -> IdentityCoordinator:
    return IdentityCoordinator(
        scene_processor=Mock(),
        event_engine=EventEngine(seeded),
        store=seeded,
        reid_model_name="test-model",
        max_references_per_item=2,
        initial_capture_spacing_frames=0,
        move_candidate_delay_frames=0,
        candidate_retry_frames=0,
        bad_mask_cooldown_seconds=5.0,
        queue_full_retry_seconds=0.5,
        clock=Mock(return_value=0.0),
        worker=worker,
        identity_policy=IdentityPolicy(
            config=config, acceptance_threshold=0.68, margin_threshold=0.06
        ),
        question_registry=registry,
    )


def resolve_jobs(worker: FakeWorker) -> int:
    return len([job for job in worker.submitted if job.kind == "resolve"])


def present(coordinator: IdentityCoordinator, worker: FakeWorker, track: int = TRACK) -> None:
    """Confirm a track so a resolve job is submitted for it."""
    coordinator.handle_frame(frame(), [add_signal(track)], {})


def deliver(
    coordinator: IdentityCoordinator,
    worker: FakeWorker,
    result: IdentificationResult,
    at: float | None = None,
    visible: bool = True,
) -> None:
    """Push a worker result and run one frame so the coordinator applies it."""
    if at is not None:
        coordinator.clock.return_value = at
    worker.push_result(result)
    detections = {result.track_id: make_detection(result.track_id)} if visible else {}
    coordinator.handle_frame(frame(), [], detections)


def tick(coordinator: IdentityCoordinator, at: float, track: int = TRACK) -> None:
    """Advance time by one frame with the object in view."""
    coordinator.clock.return_value = at
    coordinator.handle_frame(frame(), [], {track: make_detection(track)})


def ask_about(coordinator: IdentityCoordinator, worker: FakeWorker, result=None):
    """Drive one track to an open question and return it."""
    present(coordinator, worker)
    deliver(coordinator, worker, result or near_miss_result())
    question = coordinator.question_registry.open_for_track(TRACK)
    assert question is not None, "the near miss should have opened a question"
    return question


class TestAsking:
    def test_a_near_miss_opens_a_question_instead_of_adding_a_duplicate(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        items_before = seeded.count_items()

        question = ask_about(coordinator, worker)

        assert seeded.count_items() == items_before
        assert question.track_id == TRACK
        assert question.offered_item_ids == (7, 6)
        assert question.is_open

    def test_the_question_carries_the_crop_and_the_reasons_in_plain_english(
        self, coordinator: IdentityCoordinator, worker: FakeWorker
    ) -> None:
        question = ask_about(coordinator, worker)

        assert question.crop is not None
        assert question.reasons
        for reason in question.reasons:
            assert reason.endswith(".")
            # An item may be named ("item 7"), but no score or threshold may ever appear.
            assert not re.search(r"\d", re.sub(r"item \d+", "", reason, flags=re.IGNORECASE))

    def test_the_track_waits_for_the_answer_instead_of_retrying(
        self, coordinator: IdentityCoordinator, worker: FakeWorker
    ) -> None:
        ask_about(coordinator, worker)
        submitted = resolve_jobs(worker)

        tick(coordinator, 5.0)
        tick(coordinator, 20.0)

        assert coordinator._reference_manager.state(TRACK) is IdentificationState.DEFERRED
        assert coordinator._reference_manager.reason(TRACK) == "awaiting_answer"
        assert resolve_jobs(worker) == submitted

    def test_each_candidates_stored_picture_is_attached(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        seeded.save_item_embedding(7, [1.0, 0.0], "m", object_image_path=Path("7/a.png"))
        seeded.save_item_embedding(7, [0.0, 1.0], "m", object_image_path=Path("7/b.png"))

        question = ask_about(coordinator, worker)

        assert question.candidate_images[7] == Path("7/b.png"), "the newest picture is shown"
        assert question.candidate_images[6] is None, "item 6 has no picture: a placeholder"

    def test_a_candidate_on_another_visible_track_is_not_offered(
        self, coordinator: IdentityCoordinator, worker: FakeWorker
    ) -> None:
        coordinator.event_engine.associate_existing_item(99, 7)
        result = replace(near_miss_result(), candidates=(score(7, 0.654), score(6, 0.60)))

        question = ask_about(coordinator, worker, result)

        assert question.offered_item_ids == (6,)

    def test_a_confident_match_never_opens_a_question(
        self, coordinator: IdentityCoordinator, worker: FakeWorker
    ) -> None:
        present(coordinator, worker)
        match = IdentificationResult(
            kind="resolve",
            track_id=TRACK,
            status="existing",
            item_id=7,
            reference=good_reference(),
            candidates=(score(7, 0.86), score(1, 0.28)),
            decision_reason="accepted",
        )

        deliver(coordinator, worker, match)

        assert coordinator.question_registry.open_count() == 0
        assert coordinator.event_engine.item_id_for_track(TRACK) == 7

    def test_observe_mode_opens_no_question(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, config
    ) -> None:
        coordinator.identity_policy = IdentityPolicy(
            config=replace(config, mode="observe"), acceptance_threshold=0.68, margin_threshold=0.06
        )
        present(coordinator, worker)

        deliver(coordinator, worker, near_miss_result())

        assert coordinator.question_registry.open_count() == 0
        assert coordinator.event_engine.item_id_for_track(TRACK) is not None, "legacy add applies"

    def test_without_a_registry_a_near_miss_still_waits_rather_than_guesses(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        coordinator.question_registry = None
        before = seeded.count_items()
        present(coordinator, worker)

        deliver(coordinator, worker, near_miss_result())

        assert seeded.count_items() == before
        assert coordinator._reference_manager.reason(TRACK) == "policy_ask"


class TestAnswerPick:
    def test_picking_an_item_links_the_object_to_it(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        question = ask_about(coordinator, worker)
        before = seeded.count_items()

        closed = coordinator.answer_question(question.id, Answer.pick(7), now=2.0)

        assert closed is question and question.state is QuestionState.ANSWERED
        assert coordinator.event_engine.item_id_for_track(TRACK) == 7
        assert seeded.count_items() == before, "no duplicate item may be created"
        assert coordinator._reference_manager.state(TRACK) is not IdentificationState.DEFERRED

    def test_picking_saves_the_picture_as_a_confirmed_view(
        self, coordinator: IdentityCoordinator, worker: FakeWorker
    ) -> None:
        """Confirmed views skip the similarity gates and spend no reference-policy budget."""
        question = ask_about(coordinator, worker)

        coordinator.answer_question(question.id, Answer.pick(7), now=2.0)

        capture = [job for job in worker.submitted if job.kind == "capture"][-1]
        assert capture.item_id == 7
        assert capture.confirmed is True
        assert capture.candidate_kind is None
        assert capture.precomputed_reference is not None

    def test_a_legacy_match_is_not_marked_confirmed(
        self, coordinator: IdentityCoordinator, worker: FakeWorker
    ) -> None:
        present(coordinator, worker)
        match = IdentificationResult(
            kind="resolve",
            track_id=TRACK,
            status="existing",
            item_id=7,
            reference=good_reference(),
            candidates=(score(7, 0.86), score(1, 0.28)),
            decision_reason="accepted",
        )

        deliver(coordinator, worker, match)

        capture = [job for job in worker.submitted if job.kind == "capture"][-1]
        assert capture.confirmed is False
        assert capture.candidate_kind is not None

    def test_picking_an_item_that_was_removed_returns_it(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        seeded.mark_removed(7)
        question = ask_about(coordinator, worker)

        coordinator.answer_question(question.id, Answer.pick(7), now=2.0)

        assert coordinator.event_engine.item_id_for_track(TRACK) == 7
        assert seeded.get_item(7).status is ItemStatus.PRESENT


class TestAnswerNew:
    def test_new_item_adds_exactly_one_item(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        question = ask_about(coordinator, worker)
        before = seeded.count_items()

        coordinator.answer_question(question.id, Answer.new_item(), now=2.0)

        assert seeded.count_items() == before + 1
        assert coordinator.event_engine.item_id_for_track(TRACK) is not None

    def test_a_new_item_is_not_marked_confirmed(
        self, coordinator: IdentityCoordinator, worker: FakeWorker
    ) -> None:
        """Its first view is the ordinary baseline reference, as for any new item."""
        question = ask_about(coordinator, worker)

        coordinator.answer_question(question.id, Answer.new_item(), now=2.0)

        capture = [job for job in worker.submitted if job.kind == "capture"][-1]
        assert capture.confirmed is False


class TestAnswerSkip:
    def test_skip_adds_and_links_nothing(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        question = ask_about(coordinator, worker)
        before = seeded.count_items()

        coordinator.answer_question(question.id, Answer.skip(), now=2.0)

        assert seeded.count_items() == before
        assert coordinator.event_engine.item_id_for_track(TRACK) is None
        assert question.state is QuestionState.SKIPPED

    def test_a_skipped_track_is_not_asked_again_until_the_cooldown_passes(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, config
    ) -> None:
        question = ask_about(coordinator, worker)
        coordinator.answer_question(question.id, Answer.skip(), now=2.0)
        submitted = resolve_jobs(worker)

        tick(coordinator, 2.0 + config.track_cooldown_seconds - 1)
        assert resolve_jobs(worker) == submitted

        tick(coordinator, 2.0 + config.track_cooldown_seconds + 1)
        assert resolve_jobs(worker) == submitted + 1


class TestAnswerNotAnItem:
    def test_the_object_is_ignored_and_never_added(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        question = ask_about(coordinator, worker)
        before = seeded.count_items()

        coordinator.answer_question(question.id, Answer.not_an_item(), now=2.0)

        assert seeded.count_items() == before
        assert coordinator.event_engine.item_id_for_track(TRACK) is None

    def test_it_is_never_identified_again_while_the_track_lives(
        self, coordinator: IdentityCoordinator, worker: FakeWorker
    ) -> None:
        """A hand stays a hand: no amount of waiting makes it a candidate again."""
        question = ask_about(coordinator, worker)
        coordinator.answer_question(question.id, Answer.not_an_item(), now=2.0)
        submitted = resolve_jobs(worker)

        for at in (100.0, 10_000.0, 1_000_000.0):
            tick(coordinator, at)

        assert resolve_jobs(worker) == submitted


class TestStaleAndHeldAnswers:
    def test_an_answer_after_the_question_expired_is_ignored(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore, config
    ) -> None:
        question = ask_about(coordinator, worker)
        before = seeded.count_items()

        late = coordinator.answer_question(
            question.id, Answer.pick(7), now=config.question_ttl_seconds + 5
        )

        assert late is None
        assert seeded.count_items() == before
        assert coordinator.event_engine.item_id_for_track(TRACK) is None

    def test_answering_twice_applies_only_the_first(
        self, coordinator: IdentityCoordinator, worker: FakeWorker
    ) -> None:
        question = ask_about(coordinator, worker)
        coordinator.answer_question(question.id, Answer.pick(7), now=2.0)

        assert coordinator.answer_question(question.id, Answer.pick(6), now=3.0) is None
        assert coordinator.event_engine.item_id_for_track(TRACK) == 7

    def test_an_unknown_question_is_ignored(self, coordinator: IdentityCoordinator) -> None:
        assert coordinator.answer_question(404, Answer.skip(), now=1.0) is None

    def test_an_answer_while_the_object_is_out_of_view_waits_for_it(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        question = ask_about(coordinator, worker)
        # The object drops out of view for a frame (not retired: just not detected).
        coordinator.clock.return_value = 3.0
        coordinator.handle_frame(frame(), [], {})

        coordinator.answer_question(question.id, Answer.pick(7), now=3.0)
        assert coordinator.event_engine.item_id_for_track(TRACK) is None, "not applied yet"

        tick(coordinator, 4.0)

        assert coordinator.event_engine.item_id_for_track(TRACK) == 7


class TestExpiryAndCancel:
    def test_an_unanswered_question_closes_after_its_time(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, config
    ) -> None:
        question = ask_about(coordinator, worker)

        tick(coordinator, config.question_ttl_seconds + 1)

        assert question.state is QuestionState.EXPIRED
        assert coordinator.question_registry.open_count() == 0
        assert not coordinator._asked

    def test_an_expired_question_is_never_guessed_for(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore, config
    ) -> None:
        question = ask_about(coordinator, worker)
        before = seeded.count_items()

        tick(coordinator, config.question_ttl_seconds + 1)

        assert seeded.count_items() == before
        assert question.answer is None

    def test_the_track_retries_after_the_question_closes(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, config
    ) -> None:
        ask_about(coordinator, worker)
        submitted = resolve_jobs(worker)

        tick(coordinator, config.question_ttl_seconds + 1)

        assert resolve_jobs(worker) == submitted + 1

    def test_a_retiring_track_cancels_its_open_question(
        self, coordinator: IdentityCoordinator, worker: FakeWorker
    ) -> None:
        question = ask_about(coordinator, worker)

        coordinator.handle_frame(frame(), [remove_signal(TRACK)], {})

        assert question.state is QuestionState.CANCELLED
        assert coordinator.question_registry.open_count() == 0
        assert not coordinator._asked

    def test_an_answer_to_a_cancelled_question_does_nothing(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        question = ask_about(coordinator, worker)
        coordinator.handle_frame(frame(), [remove_signal(TRACK)], {})
        before = seeded.count_items()

        assert coordinator.answer_question(question.id, Answer.new_item(), now=2.0) is None
        assert seeded.count_items() == before

    def test_a_held_answer_is_dropped_when_its_track_retires(
        self, coordinator: IdentityCoordinator, worker: FakeWorker
    ) -> None:
        question = ask_about(coordinator, worker)
        coordinator.clock.return_value = 3.0
        coordinator.handle_frame(frame(), [], {})
        coordinator.answer_question(question.id, Answer.pick(7), now=3.0)
        assert TRACK in coordinator._held_answers

        coordinator.handle_frame(frame(), [remove_signal(TRACK)], {})

        assert TRACK not in coordinator._held_answers


class TestLimits:
    def test_no_more_than_the_limit_are_open_at_once(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, config
    ) -> None:
        pairs = [(1, 2), (3, 4), (5, 6), (7, 8)]
        for track, (a, b) in enumerate(pairs, start=10):
            coordinator.handle_frame(frame(), [add_signal(track)], {})
            result = replace(
                near_miss_result(),
                track_id=track,
                candidates=(score(a, 0.654), score(b, 0.322)),
            )
            worker.push_result(result)
            coordinator.handle_frame(frame(), [], {track: make_detection(track)})

        assert coordinator.question_registry.open_count() == config.max_open_questions

    def test_a_question_that_cannot_open_waits_and_adds_nothing(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore, config
    ) -> None:
        for track, (a, b) in enumerate([(1, 2), (3, 4), (5, 6)], start=10):
            coordinator.handle_frame(frame(), [add_signal(track)], {})
            worker.push_result(
                replace(
                    near_miss_result(),
                    track_id=track,
                    candidates=(score(a, 0.654), score(b, 0.322)),
                )
            )
            coordinator.handle_frame(frame(), [], {track: make_detection(track)})
        before = seeded.count_items()

        coordinator.handle_frame(frame(), [add_signal(20)], {})
        worker.push_result(
            replace(near_miss_result(), track_id=20, candidates=(score(7, 0.654), score(8, 0.322)))
        )
        coordinator.handle_frame(frame(), [], {20: make_detection(20)})

        assert seeded.count_items() == before
        assert coordinator._reference_manager.reason(20) == "policy_ask_suppressed_limit"


def poor_near_miss() -> IdentificationResult:
    """The near miss, seen through a mask too thin to learn from (occupancy under 0.40)."""
    return replace(near_miss_result(), reference=replace(good_reference(), mask_occupancy=0.2))


def retry(coordinator: IdentityCoordinator, worker: FakeWorker, at: float) -> None:
    """Advance past a bad-view wait and check the track really was resubmitted."""
    submitted = resolve_jobs(worker)
    tick(coordinator, at)
    assert resolve_jobs(worker) == submitted + 1, "the track should have retried by now"


class TestPoorViewQuestions:
    """A poor view is asked about, but never kept: the answer decides who, not what is saved."""

    def test_a_poor_view_is_asked_about_and_marked(
        self, coordinator: IdentityCoordinator, worker: FakeWorker
    ) -> None:
        question = ask_about(coordinator, worker, poor_near_miss())

        assert question.poor_view is True
        assert "can't see this clearly" in question.reasons[0]

    def test_a_good_view_question_is_not_marked(
        self, coordinator: IdentityCoordinator, worker: FakeWorker
    ) -> None:
        assert ask_about(coordinator, worker).poor_view is False

    def test_a_pick_links_the_object_but_saves_no_picture(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        question = ask_about(coordinator, worker, poor_near_miss())
        before = seeded.count_items()

        coordinator.answer_question(question.id, Answer.pick(7), now=2.0)

        assert coordinator.event_engine.item_id_for_track(TRACK) == 7
        assert seeded.count_items() == before
        assert not [job for job in worker.submitted if job.kind == "capture"], (
            "the worker's own mask gate is looser than the policy's, so nothing may be sent"
        )

    def test_a_new_answer_adds_nothing_until_a_good_view(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        question = ask_about(coordinator, worker, poor_near_miss())
        before = seeded.count_items()

        coordinator.answer_question(question.id, Answer.new_item(), now=2.0)

        assert seeded.count_items() == before
        assert coordinator.event_engine.item_id_for_track(TRACK) is None
        assert coordinator._reference_manager.reason(TRACK) == "awaiting_good_view"

    def test_the_new_item_is_added_from_the_first_good_view(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        question = ask_about(coordinator, worker, poor_near_miss())
        coordinator.answer_question(question.id, Answer.new_item(), now=2.0)
        before = seeded.count_items()

        retry(coordinator, worker, 2.0 + coordinator.bad_mask_cooldown_seconds + 1)
        deliver(coordinator, worker, near_miss_result())

        assert seeded.count_items() == before + 1
        assert coordinator.event_engine.item_id_for_track(TRACK) not in (None, 6, 7)
        capture = [job for job in worker.submitted if job.kind == "capture"][-1]
        assert capture.precomputed_reference.mask_occupancy >= 0.40, "the good view is kept"
        assert TRACK not in coordinator._awaiting_good_view

    def test_the_users_answer_stands_even_if_the_good_view_now_matches(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        """Nothing is asked again and no match overrides 'this is new'."""
        question = ask_about(coordinator, worker, poor_near_miss())
        coordinator.answer_question(question.id, Answer.new_item(), now=2.0)
        before = seeded.count_items()
        confident = IdentificationResult(
            kind="resolve",
            track_id=TRACK,
            status="existing",
            item_id=7,
            reference=good_reference(),
            candidates=(score(7, 0.86), score(1, 0.28)),
            decision_reason="accepted",
        )

        retry(coordinator, worker, 2.0 + coordinator.bad_mask_cooldown_seconds + 1)
        deliver(coordinator, worker, confident)

        assert seeded.count_items() == before + 1
        assert coordinator.event_engine.item_id_for_track(TRACK) != 7
        assert coordinator.question_registry.open_count() == 0

    def test_more_poor_views_wait_without_asking_again(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        question = ask_about(coordinator, worker, poor_near_miss())
        coordinator.answer_question(question.id, Answer.new_item(), now=2.0)
        before = seeded.count_items()

        retry(coordinator, worker, 2.0 + coordinator.bad_mask_cooldown_seconds + 1)
        deliver(coordinator, worker, poor_near_miss())

        assert seeded.count_items() == before
        assert coordinator.question_registry.open_count() == 0
        assert coordinator._reference_manager.reason(TRACK) == "awaiting_good_view"

    def test_a_new_item_that_is_never_seen_well_is_added_in_the_end(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore, config
    ) -> None:
        """An object never added is lost for good; a poor first picture can be improved."""
        question = ask_about(coordinator, worker, poor_near_miss())
        coordinator.answer_question(question.id, Answer.new_item(), now=2.0)
        before = seeded.count_items()
        at = 2.0

        for attempt in range(config.give_up_after_attempts):
            assert seeded.count_items() == before, f"added too early, poor view {attempt}"
            at += coordinator.bad_mask_cooldown_seconds + 1
            retry(coordinator, worker, at)
            deliver(coordinator, worker, poor_near_miss())

        assert seeded.count_items() == before + 1
        assert coordinator.event_engine.item_id_for_track(TRACK) is not None
        assert TRACK not in coordinator._awaiting_good_view

    def test_a_failed_view_still_counts_and_the_asked_view_is_used(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore, config
    ) -> None:
        """A result with no picture at all (a failed mask) cannot stall it either."""
        question = ask_about(coordinator, worker, poor_near_miss())
        coordinator.answer_question(question.id, Answer.new_item(), now=2.0)
        failed = IdentificationResult(
            kind="resolve", track_id=TRACK, status="pending", failure_reason="low_mask_score"
        )
        before = seeded.count_items()
        at = 2.0

        for _ in range(config.give_up_after_attempts):
            at += coordinator.bad_mask_cooldown_seconds + 1
            retry(coordinator, worker, at)
            deliver(coordinator, worker, failed)

        assert seeded.count_items() == before + 1
        capture = [job for job in worker.submitted if job.kind == "capture"][-1]
        assert capture.precomputed_reference is not None

    def test_a_lost_track_adds_nothing(
        self,
        coordinator: IdentityCoordinator,
        worker: FakeWorker,
        seeded: DatabaseStore,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        question = ask_about(coordinator, worker, poor_near_miss())
        coordinator.answer_question(question.id, Answer.new_item(), now=2.0)
        before = seeded.count_items()

        coordinator.handle_frame(frame(), [remove_signal(TRACK)], {})

        assert seeded.count_items() == before
        assert TRACK not in coordinator._awaiting_good_view
        assert "action=new_item_dropped" in capsys.readouterr().out

    def test_not_an_item_on_a_poor_view_stores_no_example(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        question = ask_about(coordinator, worker, poor_near_miss())

        coordinator.answer_question(question.id, Answer.not_an_item(), now=2.0)

        assert seeded.count_negative_examples() == 0
        assert coordinator._reference_manager.reason(TRACK) == "not_an_item", "still ignored"

    def test_not_an_item_on_a_good_view_stores_an_example(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore
    ) -> None:
        question = ask_about(coordinator, worker)

        coordinator.answer_question(question.id, Answer.not_an_item(), now=2.0)

        assert seeded.count_negative_examples() == 1

    def test_the_answer_log_says_the_view_was_poor(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, capsys
    ) -> None:
        question = ask_about(coordinator, worker, poor_near_miss())

        coordinator.answer_question(question.id, Answer.pick(7), now=2.0)

        assert "poor_view=True" in capsys.readouterr().out


class TestGivingUp:
    def poor(self) -> IdentificationResult:
        return poor_near_miss()

    def test_a_view_that_never_improves_is_eventually_added(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore, config
    ) -> None:
        """Three poor views in a row: stop waiting and remember the object.

        Waiting on a poor view only happens when view_blocks_match is switched on; by
        default a poor view is asked about instead (see TestPoorViewQuestions).
        """
        coordinator.identity_policy = IdentityPolicy(
            config=replace(config, view_blocks_match=True),
            acceptance_threshold=0.68,
            margin_threshold=0.06,
        )
        present(coordinator, worker)
        before = seeded.count_items()
        at = 0.0

        for attempt in range(config.give_up_after_attempts):
            assert seeded.count_items() == before, f"added too early, attempt {attempt}"
            deliver(coordinator, worker, self.poor(), at=at)
            at += coordinator.bad_mask_cooldown_seconds + 1
            tick(coordinator, at)

        assert seeded.count_items() == before + 1
        assert coordinator.event_engine.item_id_for_track(TRACK) is not None

    def test_a_weak_likeness_that_never_resolves_is_eventually_added(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore, config
    ) -> None:
        """The phone case: a plausible but unconvincing score that no view improves."""
        present(coordinator, worker)
        weak = replace(
            near_miss_result(),
            candidates=(score(1, 0.49), score(2, 0.18)),
            reference=good_reference(),
        )
        before = seeded.count_items()
        at = 0.0

        for _ in range(config.give_up_after_attempts + 1):
            deliver(coordinator, worker, weak, at=at)
            at += config.track_cooldown_seconds + 1
            tick(coordinator, at)

        assert seeded.count_items() == before + 1

    def test_unanswered_questions_eventually_add_the_object(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore, config
    ) -> None:
        """Nobody answers: after the configured number of expiries it is added, not lost."""
        before = seeded.count_items()
        asked = 0
        delivered = 0
        coordinator.clock.return_value = 0.0
        present(coordinator, worker)

        for at in range(0, 2000, 5):
            tick(coordinator, float(at))
            if coordinator.event_engine.item_id_for_track(TRACK) is not None:
                break
            if resolve_jobs(worker) > delivered:
                delivered = resolve_jobs(worker)
                had = coordinator.question_registry.open_count()
                deliver(coordinator, worker, near_miss_result(), at=float(at))
                asked += coordinator.question_registry.open_count() - had

        assert seeded.count_items() == before + 1
        assert asked == config.give_up_after_attempts, "it asked the configured number of times"

    def test_a_skip_never_counts_towards_giving_up(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, seeded: DatabaseStore, config
    ) -> None:
        """'Not now' is an answer: the plan is to never guess against it."""
        present(coordinator, worker)
        before = seeded.count_items()
        at = 0.0

        skipped = 0
        for _ in range(config.give_up_after_attempts + 3):
            question = coordinator.question_registry.open_for_track(TRACK)
            if question is None:
                tick(coordinator, at)
                if resolve_jobs(worker) > 0:
                    deliver(coordinator, worker, self._fresh(worker), at=at)
                question = coordinator.question_registry.open_for_track(TRACK)
            if question is not None:
                coordinator.answer_question(question.id, Answer.skip(), now=at + 1)
                skipped += 1
            at += config.item_pair_cooldown_seconds + config.track_cooldown_seconds + 5

        assert skipped > config.give_up_after_attempts, "it must have been skipped past the limit"
        assert seeded.count_items() == before
        assert coordinator.event_engine.item_id_for_track(TRACK) is None

    @staticmethod
    def _fresh(worker: FakeWorker) -> IdentificationResult:
        return near_miss_result()


class TestWhatIsLogged:
    def test_asking_and_answering_are_each_logged_once(
        self, coordinator: IdentityCoordinator, worker: FakeWorker, capsys
    ) -> None:
        question = ask_about(coordinator, worker)
        coordinator.answer_question(question.id, Answer.pick(7), now=2.0)

        lines = [row for row in capsys.readouterr().out.splitlines() if "ASK" in row]

        assert sum("action=asked" in row for row in lines) == 1
        assert sum("action=answered" in row for row in lines) == 1
        assert any("why=" in row and "look" in row for row in lines)

    def test_a_stale_answer_is_logged_not_applied(
        self, coordinator: IdentityCoordinator, capsys
    ) -> None:
        coordinator.answer_question(404, Answer.skip(), now=1.0)

        assert "stale_answer_ignored" in capsys.readouterr().out
