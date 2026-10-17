"""Question-registry tests: what may be asked, what closes a question, when to give up.

Decisions come from the real identity policy and the shipped configuration, so a test of
"what happens to an ASK" is built from an actual ASK. Time is passed in as plain numbers:
the registry reads no clock, so nothing here sleeps or patches time.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from project_auto.events.identity_policy import (
    IdentityOutcome,
    IdentityPolicy,
    IdentityPolicyConfig,
    PolicyCandidate,
    PolicyDecision,
    ViewQuality,
)
from project_auto.events.question_registry import (
    Answer,
    AnswerKind,
    QuestionRegistry,
    QuestionState,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "identity_policy.yaml"
GOOD_VIEW = ViewQuality(mask_occupancy=0.70, mask_score=0.95)


@pytest.fixture
def config() -> IdentityPolicyConfig:
    return IdentityPolicyConfig.from_yaml(CONFIG_PATH)


@pytest.fixture
def policy(config: IdentityPolicyConfig) -> IdentityPolicy:
    return IdentityPolicy(config=config, acceptance_threshold=0.68, margin_threshold=0.06)


@pytest.fixture
def registry(config: IdentityPolicyConfig) -> QuestionRegistry:
    return QuestionRegistry(config)


def candidate(item_id: int, final: float) -> PolicyCandidate:
    return PolicyCandidate(
        item_id=item_id,
        final_score=final,
        dino_score=final,
        color_score=final,
        aspect_score=0.5,
        n_references=5,
    )


def ask_decision(policy: IdentityPolicy, first: int = 7, second: int = 6) -> PolicyDecision:
    """A real ASK: a near miss at 0.654 (q00029 of the run) with one rival."""
    decision = policy.decide((candidate(first, 0.654), candidate(second, 0.322)), GOOD_VIEW)
    assert decision.outcome is IdentityOutcome.ASK
    return decision


class TestOpening:
    def test_an_ask_decision_opens_a_question_carrying_its_evidence(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        decision = ask_decision(policy)

        question = registry.ask(1, decision, now=100.0, crop="the-crop")

        assert question is not None
        assert question.is_open
        assert question.track_id == 1
        assert question.offered_item_ids == (7, 6)
        assert question.reasons == decision.reasons
        assert question.evidence == decision.evidence
        assert question.crop == "the-crop"

    def test_a_question_expires_after_the_configured_time(
        self, registry: QuestionRegistry, policy: IdentityPolicy, config: IdentityPolicyConfig
    ) -> None:
        question = registry.ask(1, ask_decision(policy), now=100.0)

        assert question is not None
        assert question.expires_at == pytest.approx(100.0 + config.question_ttl_seconds)

    def test_thumbnail_paths_are_carried_not_loaded(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        images = {7: Path("7/a.png"), 6: None}

        question = registry.ask(1, ask_decision(policy), now=0.0, candidate_images=images)

        assert question is not None
        assert question.candidate_images == images

    @pytest.mark.parametrize("outcome", [IdentityOutcome.MATCH, IdentityOutcome.WAIT])
    def test_only_an_ask_decision_can_open_a_question(
        self, registry: QuestionRegistry, outcome: IdentityOutcome
    ) -> None:
        with pytest.raises(ValueError, match="only an ASK"):
            registry.ask(1, PolicyDecision(outcome=outcome, code="x"), now=0.0)

    def test_ids_are_unique_and_increasing(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        first = registry.ask(1, ask_decision(policy, 7, 6), now=0.0)
        second = registry.ask(2, ask_decision(policy, 3, 4), now=0.0)

        assert first is not None and second is not None
        assert second.id > first.id


class TestOneQuestionPerTrack:
    def test_asking_again_returns_the_same_open_question(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        first = registry.ask(1, ask_decision(policy), now=0.0)
        again = registry.ask(1, ask_decision(policy), now=1.0)

        assert again is first
        assert registry.open_count() == 1

    def test_a_track_with_a_waiting_question_is_reported_as_on_cooldown(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        """So the policy waits instead of asking the same track a second question."""
        registry.ask(1, ask_decision(policy), now=0.0)

        assert registry.context(1, now=1.0).track_on_cooldown
        assert not registry.context(2, now=1.0).track_on_cooldown


class TestOpenQuestionLimit:
    def test_no_more_than_the_limit_are_open_at_once(
        self, registry: QuestionRegistry, policy: IdentityPolicy, config: IdentityPolicyConfig
    ) -> None:
        for track, (a, b) in enumerate([(1, 2), (3, 4), (5, 6)], start=1):
            assert registry.ask(track, ask_decision(policy, a, b), now=0.0) is not None
        assert registry.open_count() == config.max_open_questions

        assert registry.ask(9, ask_decision(policy, 7, 8), now=0.0) is None
        assert registry.open_count() == config.max_open_questions

    def test_the_context_reports_how_many_are_open(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        registry.ask(1, ask_decision(policy, 1, 2), now=0.0)
        registry.ask(2, ask_decision(policy, 3, 4), now=0.0)

        assert registry.context(9, now=0.0).open_questions == 2

    def test_the_policy_waits_when_the_registry_is_full(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        """The two modules agree: past the limit the policy defers rather than asks."""
        for track, (a, b) in enumerate([(1, 2), (3, 4), (5, 6)], start=1):
            registry.ask(track, ask_decision(policy, a, b), now=0.0)

        decision = policy.decide(
            (candidate(7, 0.654), candidate(8, 0.322)), GOOD_VIEW, registry.context(9, now=0.0)
        )

        assert decision.outcome is IdentityOutcome.WAIT
        assert decision.code == "ask_suppressed_limit"


class TestAnswering:
    def test_picking_an_offered_item_closes_the_question(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None

        closed = registry.answer(question.id, Answer.pick(7), now=5.0)

        assert closed is question
        assert closed.state is QuestionState.ANSWERED
        assert closed.answer == Answer.pick(7)
        assert closed.closed_at == 5.0
        assert registry.open_count() == 0
        assert registry.open_for_track(1) is None

    @pytest.mark.parametrize("answer", [Answer.new_item(), Answer.not_an_item()])
    def test_new_and_not_an_item_answer_the_question(
        self, registry: QuestionRegistry, policy: IdentityPolicy, answer: Answer
    ) -> None:
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None

        closed = registry.answer(question.id, answer, now=5.0)

        assert closed is not None and closed.state is QuestionState.ANSWERED

    def test_skip_closes_the_question_as_skipped(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None

        closed = registry.answer(question.id, Answer.skip(), now=5.0)

        assert closed is not None and closed.state is QuestionState.SKIPPED

    def test_picking_an_item_that_was_not_offered_is_refused(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None

        with pytest.raises(ValueError, match="not offered"):
            registry.answer(question.id, Answer.pick(99), now=5.0)

        assert question.is_open, "a refused answer must leave the question open"

    @pytest.mark.parametrize("kind", [AnswerKind.NEW, AnswerKind.SKIP, AnswerKind.NOT_AN_ITEM])
    def test_only_a_pick_may_name_an_item(
        self, registry: QuestionRegistry, policy: IdentityPolicy, kind: AnswerKind
    ) -> None:
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None

        with pytest.raises(ValueError, match="cannot name an item"):
            registry.answer(question.id, Answer(kind, item_id=7), now=5.0)

    def test_a_pick_without_an_item_is_refused(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None

        with pytest.raises(ValueError):
            registry.answer(question.id, Answer(AnswerKind.PICK), now=5.0)


class TestStaleAnswersAreIgnored:
    def test_an_answer_to_an_unknown_question_is_ignored(self, registry: QuestionRegistry) -> None:
        assert registry.answer(404, Answer.skip(), now=0.0) is None

    def test_an_answer_to_an_already_answered_question_is_ignored(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None
        registry.answer(question.id, Answer.pick(7), now=5.0)

        assert registry.answer(question.id, Answer.pick(6), now=6.0) is None
        assert question.answer == Answer.pick(7), "the first answer must stand"

    def test_an_answer_arriving_after_the_time_ran_out_is_not_applied(
        self, registry: QuestionRegistry, policy: IdentityPolicy, config: IdentityPolicyConfig
    ) -> None:
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None

        late = registry.answer(question.id, Answer.pick(7), now=config.question_ttl_seconds + 1)

        assert late is None
        assert question.state is QuestionState.EXPIRED
        assert question.answer is None

    def test_an_answer_to_a_cancelled_question_is_ignored(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None
        registry.forget_track(1)

        assert registry.answer(question.id, Answer.pick(7), now=1.0) is None


class TestExpiry:
    def test_a_question_past_its_time_is_expired_and_returned(
        self, registry: QuestionRegistry, policy: IdentityPolicy, config: IdentityPolicyConfig
    ) -> None:
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None

        assert registry.expire(now=config.question_ttl_seconds - 1) == []
        expired = registry.expire(now=config.question_ttl_seconds)

        assert expired == [question]
        assert question.state is QuestionState.EXPIRED
        assert registry.open_count() == 0

    def test_an_expired_question_frees_its_slot(
        self, registry: QuestionRegistry, policy: IdentityPolicy, config: IdentityPolicyConfig
    ) -> None:
        for track, (a, b) in enumerate([(1, 2), (3, 4), (5, 6)], start=1):
            registry.ask(track, ask_decision(policy, a, b), now=0.0)
        assert registry.ask(9, ask_decision(policy, 7, 8), now=1.0) is None

        registry.expire(now=config.question_ttl_seconds)

        assert registry.ask(9, ask_decision(policy, 7, 8), now=config.question_ttl_seconds)

    def test_expiry_leaves_the_track_on_cooldown(
        self, registry: QuestionRegistry, policy: IdentityPolicy, config: IdentityPolicyConfig
    ) -> None:
        registry.ask(1, ask_decision(policy), now=0.0)
        end = config.question_ttl_seconds
        registry.expire(now=end)

        assert registry.context(1, now=end + 1).track_on_cooldown
        assert not registry.context(
            1, now=end + config.track_cooldown_seconds + 1
        ).track_on_cooldown

    def test_lapsed_cooldowns_are_forgotten_so_the_registry_does_not_grow(
        self, registry: QuestionRegistry, policy: IdentityPolicy, config: IdentityPolicyConfig
    ) -> None:
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None
        registry.answer(question.id, Answer.skip(), now=1.0)

        registry.expire(now=1.0 + config.item_pair_cooldown_seconds + 1)

        assert not registry._track_quiet_until
        assert not registry._pair_quiet_until


class TestTrackCooldown:
    def test_a_skipped_track_is_not_asked_again_until_the_cooldown_passes(
        self, registry: QuestionRegistry, policy: IdentityPolicy, config: IdentityPolicyConfig
    ) -> None:
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None
        registry.answer(question.id, Answer.skip(), now=5.0)

        assert registry.ask(1, ask_decision(policy, 3, 4), now=6.0) is None
        later = 5.0 + config.track_cooldown_seconds + 1
        assert registry.ask(1, ask_decision(policy, 3, 4), now=later) is not None

    def test_the_policy_waits_while_the_track_is_cooling_down(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None
        registry.answer(question.id, Answer.skip(), now=5.0)

        decision = policy.decide(
            (candidate(3, 0.654), candidate(4, 0.322)), GOOD_VIEW, registry.context(1, now=6.0)
        )

        assert decision.outcome is IdentityOutcome.WAIT
        assert decision.code == "ask_suppressed_cooldown"


class TestPairCooldown:
    """The same confusion returning is not a fresh question: this is how it asks less."""

    def test_a_pair_just_asked_about_is_not_asked_about_on_another_track(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        question = registry.ask(1, ask_decision(policy, 7, 6), now=0.0)
        assert question is not None
        registry.answer(question.id, Answer.pick(7), now=5.0)

        # The same object reappears under a brand-new track ID.
        assert registry.ask(2, ask_decision(policy, 7, 6), now=10.0) is None

    def test_the_pair_is_the_same_whichever_item_leads(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        question = registry.ask(1, ask_decision(policy, 7, 6), now=0.0)
        assert question is not None
        registry.answer(question.id, Answer.pick(7), now=5.0)

        assert registry.ask(2, ask_decision(policy, 6, 7), now=10.0) is None

    def test_a_different_pair_is_still_asked(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        question = registry.ask(1, ask_decision(policy, 7, 6), now=0.0)
        assert question is not None
        registry.answer(question.id, Answer.pick(7), now=5.0)

        assert registry.ask(2, ask_decision(policy, 3, 4), now=10.0) is not None

    def test_the_pair_is_asked_again_once_its_cooldown_passes(
        self, registry: QuestionRegistry, policy: IdentityPolicy, config: IdentityPolicyConfig
    ) -> None:
        question = registry.ask(1, ask_decision(policy, 7, 6), now=0.0)
        assert question is not None
        registry.answer(question.id, Answer.pick(7), now=5.0)

        later = 5.0 + config.item_pair_cooldown_seconds + 1
        assert registry.ask(2, ask_decision(policy, 7, 6), now=later) is not None

    def test_the_policy_waits_on_a_pair_already_asked_about(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        question = registry.ask(1, ask_decision(policy, 7, 6), now=0.0)
        assert question is not None
        registry.answer(question.id, Answer.pick(7), now=5.0)

        decision = policy.decide(
            (candidate(7, 0.654), candidate(6, 0.322)),
            GOOD_VIEW,
            registry.context(2, now=10.0, candidate_item_ids=(7, 6)),
        )

        assert decision.outcome is IdentityOutcome.WAIT
        assert decision.code == "ask_suppressed_cooldown"

    def test_an_expired_question_does_not_silence_the_pair(
        self, registry: QuestionRegistry, policy: IdentityPolicy, config: IdentityPolicyConfig
    ) -> None:
        """Nobody answered, so nothing was learned; another track may still be asked."""
        registry.ask(1, ask_decision(policy, 7, 6), now=0.0)
        end = config.question_ttl_seconds
        registry.expire(now=end)

        assert registry.ask(2, ask_decision(policy, 7, 6), now=end + 1) is not None


class TestGivingUp:
    def test_a_fresh_object_has_not_stalled(self, registry: QuestionRegistry) -> None:
        assert registry.unresolved_attempts(1) == 0
        assert not registry.should_give_up(1)

    def test_unanswered_questions_count_towards_giving_up(
        self, registry: QuestionRegistry, policy: IdentityPolicy, config: IdentityPolicyConfig
    ) -> None:
        now = 0.0
        for attempt in range(config.give_up_after_attempts):
            assert not registry.should_give_up(1)
            question = registry.ask(1, ask_decision(policy, 10 + attempt, 20 + attempt), now=now)
            assert question is not None
            now = question.expires_at
            registry.expire(now=now)
            now += config.track_cooldown_seconds + 1

        assert registry.should_give_up(1)

    def test_a_policy_deferral_counts_too(
        self, registry: QuestionRegistry, config: IdentityPolicyConfig
    ) -> None:
        """A view that never improves stalls an object just as an unanswered question does."""
        for _ in range(config.give_up_after_attempts):
            registry.note_unresolved(1)

        assert registry.should_give_up(1)

    def test_a_skip_never_counts_towards_giving_up(
        self, registry: QuestionRegistry, policy: IdentityPolicy, config: IdentityPolicyConfig
    ) -> None:
        """A skip is the user saying 'not now'; the system must not guess against it."""
        now = 0.0
        for attempt in range(config.give_up_after_attempts + 2):
            question = registry.ask(1, ask_decision(policy, 10 + attempt, 20 + attempt), now=now)
            assert question is not None
            registry.answer(question.id, Answer.skip(), now=now + 1)
            now += config.track_cooldown_seconds + 5

        assert registry.unresolved_attempts(1) == 0
        assert not registry.should_give_up(1)

    def test_a_real_answer_clears_the_count(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        registry.note_unresolved(1)
        registry.note_unresolved(1)
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None

        registry.answer(question.id, Answer.pick(7), now=1.0)

        assert registry.unresolved_attempts(1) == 0

    def test_counts_are_per_track(self, registry: QuestionRegistry) -> None:
        registry.note_unresolved(1)

        assert registry.unresolved_attempts(2) == 0


class TestForgettingATrack:
    def test_a_retiring_track_cancels_its_open_question(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None

        cancelled = registry.forget_track(1)

        assert cancelled is question
        assert question.state is QuestionState.CANCELLED
        assert registry.open_count() == 0

    def test_a_reused_track_id_starts_fresh(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        """No cooldown or attempt count may leak onto a different object with the same ID."""
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None
        registry.answer(question.id, Answer.skip(), now=1.0)
        registry.note_unresolved(1)

        registry.forget_track(1)

        assert not registry.context(1, now=2.0).track_on_cooldown
        assert registry.unresolved_attempts(1) == 0

    def test_forgetting_a_track_with_nothing_open_is_harmless(
        self, registry: QuestionRegistry
    ) -> None:
        assert registry.forget_track(404) is None


class TestListing:
    def test_open_questions_are_listed_oldest_first(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        first = registry.ask(1, ask_decision(policy, 1, 2), now=0.0)
        second = registry.ask(2, ask_decision(policy, 3, 4), now=1.0)

        assert registry.open_questions() == (first, second)

    def test_closed_questions_are_not_retained(
        self, registry: QuestionRegistry, policy: IdentityPolicy
    ) -> None:
        question = registry.ask(1, ask_decision(policy), now=0.0)
        assert question is not None
        registry.answer(question.id, Answer.pick(7), now=1.0)

        assert registry.get(question.id) is None
        assert registry.open_questions() == ()


class TestConfiguration:
    def test_the_shipped_settings_are_the_agreed_defaults(
        self, config: IdentityPolicyConfig
    ) -> None:
        assert config.question_ttl_seconds == pytest.approx(60.0)
        assert config.give_up_after_attempts == 3
        assert config.max_open_questions == 3

    def test_the_registry_reads_its_limits_from_the_config_it_is_given(
        self, config: IdentityPolicyConfig, policy: IdentityPolicy
    ) -> None:
        tight = QuestionRegistry(replace(config, max_open_questions=1))

        assert tight.ask(1, ask_decision(policy, 1, 2), now=0.0) is not None
        assert tight.ask(2, ask_decision(policy, 3, 4), now=0.0) is None
