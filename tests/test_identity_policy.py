"""Identity-policy tests: when the system matches, asks, waits, or adds a new item.

Every score in here is taken from a real query in logs/reid_runs/20261016T030339Z so the
table documents what the policy does to actual data rather than to invented numbers. The
query id is named in each case.

The policy is pure arithmetic, so these tests construct it directly and need no store,
worker, frame or clock.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from project_auto.events.identity_policy import (
    EVIDENCE_CLOSE_RACE,
    EVIDENCE_DISAGREEMENT,
    EVIDENCE_FEW_VIEWS,
    EVIDENCE_NEAR_ACCEPTANCE,
    EVIDENCE_PLAUSIBLE,
    EVIDENCE_REGION_MATCH,
    POOR_VIEW_REASON,
    REASON_ACCEPTED,
    REASON_ALL_CLAIMED,
    REASON_ASK_COOLDOWN,
    REASON_ASK_LIMIT,
    REASON_BAD_VIEW,
    REASON_BELOW_FLOOR,
    REASON_GALLERY_EMPTY,
    REASON_UNCONVINCING,
    AskContext,
    IdentityOutcome,
    IdentityPolicy,
    IdentityPolicyConfig,
    PolicyCandidate,
    ViewQuality,
)

ACCEPTANCE = 0.68
MARGIN = 0.06
GOOD_VIEW = ViewQuality(mask_occupancy=0.70, mask_score=0.95)
POOR_VIEW = ViewQuality(mask_occupancy=0.36, mask_score=0.93)


@pytest.fixture
def policy() -> IdentityPolicy:
    """The policy as the shipped configuration file actually configures it."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "identity_policy.yaml"
    return IdentityPolicy(
        config=IdentityPolicyConfig.from_yaml(config_path),
        acceptance_threshold=ACCEPTANCE,
        margin_threshold=MARGIN,
    )


def candidate(
    item_id: int,
    final: float,
    dino: float | None = None,
    colour: float | None = None,
    views: int = 5,
    region_matches: bool = False,
) -> PolicyCandidate:
    """Build one candidate; dino/colour default to the final score when irrelevant."""
    return PolicyCandidate(
        item_id=item_id,
        final_score=final,
        dino_score=final if dino is None else dino,
        color_score=final if colour is None else colour,
        aspect_score=0.5,
        n_references=views,
        region_matches=region_matches,
    )


class TestClearCases:
    """Confident scores must keep behaving exactly as they do today."""

    def test_a_clear_winner_matches(self, policy: IdentityPolicy) -> None:
        # q00105 of the run: item 7 at 0.86, nothing else close.
        decision = policy.decide((candidate(7, 0.86), candidate(1, 0.28)), GOOD_VIEW)

        assert decision.outcome is IdentityOutcome.MATCH
        assert decision.code == REASON_ACCEPTED
        assert decision.item_id == 7

    def test_nothing_like_it_is_a_new_item(self, policy: IdentityPolicy) -> None:
        # q00017 of the run: best candidate 0.34, no evidence of a confusion.
        decision = policy.decide((candidate(8, 0.34), candidate(5, 0.21)), GOOD_VIEW)

        assert decision.outcome is IdentityOutcome.NEW
        assert decision.code == REASON_BELOW_FLOOR
        assert decision.confidence == 0.0

    def test_an_empty_gallery_is_a_new_item(self, policy: IdentityPolicy) -> None:
        decision = policy.decide((), GOOD_VIEW)

        assert decision.outcome is IdentityOutcome.NEW
        assert decision.code == REASON_GALLERY_EMPTY

    def test_a_match_still_needs_its_margin(self, policy: IdentityPolicy) -> None:
        """Above acceptance but in a dead heat is not a match; it is a question.

        This is today's AMBIG/low_margin case, which currently defers and retries. No
        query in the replayed run hit it (every accepted match led by at least 0.14), so
        the behaviour here is reasoned, not measured.
        """
        decision = policy.decide((candidate(3, 0.80), candidate(4, 0.78)), GOOD_VIEW)

        assert decision.outcome is IdentityOutcome.ASK
        assert EVIDENCE_NEAR_ACCEPTANCE in decision.evidence
        assert EVIDENCE_CLOSE_RACE in decision.evidence
        assert "not safe to choose between them" in decision.reasons[0]
        # One sentence for the pair, not the same pair described twice.
        assert len(decision.reasons) == 1


class TestTheNearMissBand:
    """The case the whole flow exists for: a clear winner the threshold rejects."""

    @pytest.mark.parametrize(
        ("query", "item", "score", "views"),
        [
            ("q00048", 3, 0.678, 8),
            ("q00029", 7, 0.654, 5),
            ("q00012", 4, 0.653, 3),
            ("q00042", 7, 0.641, 5),
            ("q00039", 7, 0.637, 5),
            ("q00023", 7, 0.623, 5),
            ("q00007", 5, 0.615, 1),
            ("q00070", 3, 0.566, 8),
            ("q00063", 12, 0.553, 2),
        ],
    )
    def test_every_near_miss_in_the_run_asks(
        self, policy: IdentityPolicy, query: str, item: int, score: float, views: int
    ) -> None:
        """All nine 0.55-to-0.68 rejections become questions, not duplicate items."""
        decision = policy.decide(
            (candidate(item, score, views=views), candidate(99, score - 0.30)), GOOD_VIEW
        )

        assert decision.outcome is IdentityOutcome.ASK, query
        assert EVIDENCE_NEAR_ACCEPTANCE in decision.evidence

    def test_the_phone_seen_from_both_faces_never_becomes_a_duplicate(
        self, policy: IdentityPolicy
    ) -> None:
        """The case the whole flow was written for: one phone that became three items.

        Its screen side and its back score about 0.49 against each other. The plausible
        floor is set to 0.45 so that score counts as a weak likeness rather than as
        nothing: on its own it waits for a better view, and with any rival beside it the
        user is asked. Neither path silently adds a second phone.
        """
        alone = policy.decide((candidate(1, 0.49), candidate(2, 0.18)), GOOD_VIEW)

        assert alone.outcome is IdentityOutcome.WAIT
        assert alone.outcome is not IdentityOutcome.NEW

        with_a_rival = policy.decide((candidate(1, 0.49), candidate(2, 0.44)), GOOD_VIEW)

        assert with_a_rival.outcome is IdentityOutcome.ASK

    def test_the_floor_is_what_stops_that_duplicate(self, policy: IdentityPolicy) -> None:
        """Guards the choice itself: at the old 0.50 floor the phone duplicated silently."""
        old_floor = replace(policy, config=replace(policy.config, plausible_floor=0.50))

        decision = old_floor.decide((candidate(1, 0.49), candidate(2, 0.18)), GOOD_VIEW)

        assert decision.outcome is IdentityOutcome.NEW


class TestEvidence:
    """Each trigger on its own, and the rule that modifiers cannot trigger alone."""

    def test_a_two_way_tie_with_a_real_score_asks(self, policy: IdentityPolicy) -> None:
        # q00015 of the run: items 1 and 6 within a hair of each other.
        decision = policy.decide((candidate(1, 0.51), candidate(6, 0.47)), GOOD_VIEW)

        assert decision.outcome is IdentityOutcome.ASK
        assert EVIDENCE_CLOSE_RACE in decision.evidence

    def test_a_tie_between_two_weak_candidates_is_a_new_item(self, policy: IdentityPolicy) -> None:
        """Twenty near ties in the run looked like this: both candidates weak, nothing to ask."""
        decision = policy.decide((candidate(1, 0.12), candidate(2, 0.11)), GOOD_VIEW)

        assert decision.outcome is IdentityOutcome.NEW
        assert EVIDENCE_CLOSE_RACE not in decision.evidence

    def test_appearance_and_colour_pointing_different_ways_asks(
        self, policy: IdentityPolicy
    ) -> None:
        decision = policy.decide(
            (
                candidate(3, 0.56, dino=0.62, colour=0.33),
                candidate(12, 0.52, dino=0.40, colour=0.71),
            ),
            GOOD_VIEW,
        )

        assert decision.outcome is IdentityOutcome.ASK
        assert EVIDENCE_DISAGREEMENT in decision.evidence

    def test_disagreement_among_weak_candidates_is_not_evidence(
        self, policy: IdentityPolicy
    ) -> None:
        """Below the floor the components disagree because nothing matches at all."""
        decision = policy.decide(
            (
                candidate(3, 0.24, dino=0.30, colour=0.08),
                candidate(9, 0.19, dino=0.11, colour=0.44),
            ),
            GOOD_VIEW,
        )

        assert decision.outcome is IdentityOutcome.NEW
        assert EVIDENCE_DISAGREEMENT not in decision.evidence

    def test_a_dino_only_fallback_cannot_disagree(self, policy: IdentityPolicy) -> None:
        """With no colour score there is nothing to compare, so the trigger stays silent."""
        thin = PolicyCandidate(item_id=3, final_score=0.52, dino_score=0.52, n_references=5)
        other = PolicyCandidate(item_id=9, final_score=0.20, dino_score=0.20, n_references=5)

        decision = policy.decide((thin, other), GOOD_VIEW)

        assert EVIDENCE_DISAGREEMENT not in decision.evidence

    def test_few_stored_views_alone_never_causes_a_question(self, policy: IdentityPolicy) -> None:
        """55% of the run's queries had a thin gallery; on its own that asks nothing."""
        decision = policy.decide((candidate(10, 0.30, views=1), candidate(4, 0.12)), GOOD_VIEW)

        assert decision.outcome is IdentityOutcome.NEW
        assert EVIDENCE_FEW_VIEWS not in decision.evidence

    def test_few_stored_views_strengthens_a_question_that_already_fired(
        self, policy: IdentityPolicy
    ) -> None:
        # q00007 of the run: item 5 at 0.615 with a single stored view.
        thin = policy.decide((candidate(5, 0.615, views=1), candidate(1, 0.16)), GOOD_VIEW)
        well_known = policy.decide((candidate(5, 0.615, views=8), candidate(1, 0.16)), GOOD_VIEW)

        assert EVIDENCE_FEW_VIEWS in thin.evidence
        assert EVIDENCE_FEW_VIEWS not in well_known.evidence
        assert thin.confidence > well_known.confidence

    def test_the_same_region_alone_never_causes_a_question(self, policy: IdentityPolicy) -> None:
        """A different object put down in the same place must not raise a question."""
        decision = policy.decide(
            (candidate(4, 0.31, region_matches=True), candidate(3, 0.20)), GOOD_VIEW
        )

        assert decision.outcome is IdentityOutcome.NEW
        assert EVIDENCE_REGION_MATCH not in decision.evidence

    def test_the_same_region_strengthens_a_question_that_already_fired(
        self, policy: IdentityPolicy
    ) -> None:
        """A look-alike where that item was last seen is a genuine point of confusion."""
        elsewhere = policy.decide((candidate(4, 0.57), candidate(3, 0.26)), GOOD_VIEW)
        same_place = policy.decide(
            (candidate(4, 0.57, region_matches=True), candidate(3, 0.26)), GOOD_VIEW
        )

        assert EVIDENCE_REGION_MATCH in same_place.evidence
        assert same_place.confidence > elsewhere.confidence


class TestLiveRunPhone:
    """Track 214 of the 17 October live run: a returning phone that never asked.

    It scored 0.5482 and then 0.5374 against item 7, which had only 2 stored views, and
    the only evidence that fired was the plausible band plus few views (0.25 + 0.15 = 0.40).
    """

    @pytest.mark.parametrize("score", [0.5482, 0.5374])
    def test_a_returning_phone_with_few_views_now_asks(
        self, policy: IdentityPolicy, score: float
    ) -> None:
        decision = policy.decide((candidate(7, score, views=2),), GOOD_VIEW)

        assert decision.outcome is IdentityOutcome.ASK
        assert decision.candidates[0].item_id == 7

    def test_a_weak_likeness_on_a_mature_item_still_does_not_ask(
        self, policy: IdentityPolicy
    ) -> None:
        """The raised weight fades as galleries mature: with 5 views nothing else fires."""
        decision = policy.decide((candidate(7, 0.47, views=5),), GOOD_VIEW)

        assert decision.outcome is IdentityOutcome.WAIT
        assert decision.code == REASON_UNCONVINCING

    def test_the_weight_alone_reaches_the_threshold_with_few_views(
        self, policy: IdentityPolicy
    ) -> None:
        """Plausible 0.35 + few views 0.15 = 0.50, with the floor change taken out."""
        old_floor = replace(policy.config, near_acceptance_floor=0.55)
        decision = replace(policy, config=old_floor).decide(
            (candidate(7, 0.52, views=2),), GOOD_VIEW
        )

        assert decision.outcome is IdentityOutcome.ASK
        assert decision.evidence == (EVIDENCE_PLAUSIBLE, EVIDENCE_FEW_VIEWS)

    def test_the_floor_alone_asks_regardless_of_views(self, policy: IdentityPolicy) -> None:
        old_weight = replace(policy.config, weight_plausible=0.25)
        decision = replace(policy, config=old_weight).decide(
            (candidate(7, 0.52, views=9),), GOOD_VIEW
        )

        assert decision.outcome is IdentityOutcome.ASK
        assert EVIDENCE_NEAR_ACCEPTANCE in decision.evidence


class TestViewQuality:
    """A poor view still gets its question, but is marked so it is never learned from."""

    def test_a_poor_view_still_asks_and_is_marked(self, policy: IdentityPolicy) -> None:
        """A live run held a phone at 0.60 back as a bad view and never asked at all."""
        decision = policy.decide((candidate(7, 0.62), candidate(6, 0.31)), POOR_VIEW)

        assert decision.outcome is IdentityOutcome.ASK
        assert decision.poor_view is True
        assert [item.item_id for item in decision.candidates] == [7, 6]

    def test_a_good_view_question_is_not_marked(self, policy: IdentityPolicy) -> None:
        decision = policy.decide((candidate(7, 0.62), candidate(6, 0.31)), GOOD_VIEW)

        assert decision.outcome is IdentityOutcome.ASK
        assert decision.poor_view is False

    def test_a_poor_view_question_says_so_first(self, policy: IdentityPolicy) -> None:
        poor = policy.decide((candidate(7, 0.62), candidate(6, 0.31)), POOR_VIEW)
        good = policy.decide((candidate(7, 0.62), candidate(6, 0.31)), GOOD_VIEW)

        assert poor.reasons[0] == POOR_VIEW_REASON
        assert poor.reasons[1:] == good.reasons
        assert POOR_VIEW_REASON not in good.reasons

    def test_a_poor_view_does_not_lower_the_bar_for_asking(self, policy: IdentityPolicy) -> None:
        """Weak evidence on a poor view is treated exactly as on a good one."""
        for scores in ((0.34, 0.21), (0.47,)):
            candidates = tuple(candidate(7 - i, s) for i, s in enumerate(scores))
            poor = policy.decide(candidates, POOR_VIEW)
            good = policy.decide(candidates, GOOD_VIEW)

            assert poor.outcome is good.outcome
            assert poor.outcome is not IdentityOutcome.ASK
            assert poor.code == good.code

    def test_a_poor_view_never_stops_a_clearly_new_object_being_added(
        self, policy: IdentityPolicy
    ) -> None:
        """Nothing resembles this object, so there is nothing to ask and nothing to wait for.

        Waiting here would stop roughly a quarter of new objects ever being added, because
        thin masks are common: 21 of the run's 76 queries were below the mask-fill floor.
        """
        decision = policy.decide((candidate(8, 0.34), candidate(5, 0.21)), POOR_VIEW)

        assert decision.outcome is IdentityOutcome.NEW
        assert decision.code == REASON_BELOW_FLOOR

    def test_a_poor_mask_score_also_marks_the_question(self, policy: IdentityPolicy) -> None:
        decision = policy.decide(
            (candidate(7, 0.62), candidate(6, 0.31)),
            ViewQuality(mask_occupancy=0.70, mask_score=0.56),
        )

        assert decision.outcome is IdentityOutcome.ASK
        assert decision.poor_view is True

    def test_view_is_usable_uses_both_floors(self, policy: IdentityPolicy) -> None:
        assert policy.view_is_usable(GOOD_VIEW)
        assert not policy.view_is_usable(POOR_VIEW)
        assert not policy.view_is_usable(ViewQuality(mask_occupancy=0.70, mask_score=0.56))
        assert policy.view_is_usable(ViewQuality(mask_occupancy=0.40, mask_score=0.60))

    def test_a_cooldown_still_suppresses_a_poor_view_question(self, policy: IdentityPolicy) -> None:
        decision = policy.decide(
            (candidate(7, 0.62), candidate(6, 0.31)),
            POOR_VIEW,
            AskContext(track_on_cooldown=True),
        )

        assert decision.outcome is IdentityOutcome.WAIT
        assert decision.code == REASON_ASK_COOLDOWN

    def test_a_confident_match_survives_a_poor_view_by_default(
        self, policy: IdentityPolicy
    ) -> None:
        """Blocking these cost 5 of the run's 37 matches and changed no question."""
        decision = policy.decide((candidate(7, 0.86), candidate(1, 0.28)), POOR_VIEW)

        assert decision.outcome is IdentityOutcome.MATCH

    def test_the_view_gate_can_be_made_to_block_matching(self, policy: IdentityPolicy) -> None:
        strict = replace(policy, config=replace(policy.config, view_blocks_match=True))

        decision = strict.decide((candidate(7, 0.86), candidate(1, 0.28)), POOR_VIEW)

        assert decision.outcome is IdentityOutcome.WAIT
        assert decision.code == REASON_BAD_VIEW


class TestImpossibleCandidates:
    """An item visible on another track right now cannot also be this object."""

    def test_a_claimed_item_is_not_offered(self, policy: IdentityPolicy) -> None:
        decision = policy.decide(
            (candidate(7, 0.62), candidate(6, 0.58)),
            GOOD_VIEW,
            AskContext(claimed_item_ids=frozenset({7})),
        )

        assert [item.item_id for item in decision.candidates] == [6]

    def test_when_every_look_alike_is_elsewhere_it_is_a_new_item(
        self, policy: IdentityPolicy
    ) -> None:
        decision = policy.decide(
            (candidate(7, 0.62), candidate(6, 0.58)),
            GOOD_VIEW,
            AskContext(claimed_item_ids=frozenset({6, 7})),
        )

        assert decision.outcome is IdentityOutcome.NEW
        assert decision.code == REASON_ALL_CLAIMED


class TestSuppressionNeverGuesses:
    """A question that cannot be asked waits. It never silently becomes a new item."""

    def test_a_track_on_cooldown_waits(self, policy: IdentityPolicy) -> None:
        decision = policy.decide(
            (candidate(7, 0.62), candidate(6, 0.31)),
            GOOD_VIEW,
            AskContext(track_on_cooldown=True),
        )

        assert decision.outcome is IdentityOutcome.WAIT
        assert decision.code == REASON_ASK_COOLDOWN

    def test_an_item_already_asked_about_waits(self, policy: IdentityPolicy) -> None:
        decision = policy.decide(
            (candidate(7, 0.62), candidate(6, 0.31)),
            GOOD_VIEW,
            AskContext(items_on_cooldown=frozenset({7})),
        )

        assert decision.outcome is IdentityOutcome.WAIT
        assert decision.code == REASON_ASK_COOLDOWN

    def test_too_many_open_questions_waits(self, policy: IdentityPolicy) -> None:
        decision = policy.decide(
            (candidate(7, 0.62), candidate(6, 0.31)),
            GOOD_VIEW,
            AskContext(open_questions=policy.config.max_open_questions),
        )

        assert decision.outcome is IdentityOutcome.WAIT
        assert decision.code == REASON_ASK_LIMIT


class TestWhatTheUserIsTold:
    """The reasons are shown to a person, so they carry no scores or thresholds."""

    def test_every_reason_is_a_plain_sentence(self, policy: IdentityPolicy) -> None:
        decision = policy.decide(
            (candidate(7, 0.62, views=2, region_matches=True), candidate(6, 0.58)),
            GOOD_VIEW,
        )

        assert decision.reasons
        for reason in decision.reasons:
            assert reason[0].isupper() and reason.endswith(".")
            # An item may be named ("item 7") when it has no label, but no score,
            # threshold or other number may ever appear in text shown to the user.
            named = re.sub(r"item \d+", "", reason, flags=re.IGNORECASE)
            assert not re.search(r"\d", named), reason

    def test_the_reasons_name_the_evidence_that_fired(self, policy: IdentityPolicy) -> None:
        decision = policy.decide(
            (candidate(7, 0.62, views=2), candidate(6, 0.58)),
            GOOD_VIEW,
        )

        assert len(decision.reasons) == len(decision.evidence)

    def test_an_item_label_is_used_in_place_of_its_number(self, policy: IdentityPolicy) -> None:
        labelled = PolicyCandidate(
            item_id=7,
            final_score=0.62,
            dino_score=0.62,
            color_score=0.62,
            n_references=5,
            item_label="the black phone",
        )

        decision = policy.decide((labelled, candidate(6, 0.31)), GOOD_VIEW)

        assert "the black phone" in decision.reasons[0]

    def test_a_waiting_decision_explains_itself_too(self, policy: IdentityPolicy) -> None:
        strict = replace(policy, config=replace(policy.config, view_blocks_match=True))

        decision = strict.decide((candidate(7, 0.62),), POOR_VIEW)

        assert decision.reasons and "too poor" in decision.reasons[0]


class TestDetectorLabelsAreIrrelevant:
    """Project rule: identity is never decided or named by a detector class."""

    def test_the_decision_depends_only_on_scores_views_region_and_view_quality(
        self, policy: IdentityPolicy
    ) -> None:
        """A flickering class label cannot reach the policy: there is nowhere to put one."""
        fields = set(PolicyCandidate.__dataclass_fields__)

        assert "class_name" not in fields
        assert "detector_class" not in fields

    def test_a_descriptive_label_never_changes_the_outcome(self, policy: IdentityPolicy) -> None:
        plain = policy.decide((candidate(7, 0.62), candidate(6, 0.31)), GOOD_VIEW)
        labelled = policy.decide(
            (replace(candidate(7, 0.62), item_label="mouse"), candidate(6, 0.31)), GOOD_VIEW
        )

        assert plain.outcome is labelled.outcome
        assert plain.confidence == labelled.confidence


class TestConfiguration:
    """The shipped file must load, and its thresholds must not drift apart silently."""

    def test_the_shipped_config_loads(self, policy: IdentityPolicy) -> None:
        assert policy.config.ask_threshold == pytest.approx(0.50)
        assert policy.config.near_acceptance_floor == pytest.approx(0.50)
        assert policy.config.weight_plausible == pytest.approx(0.35)
        assert policy.config.plausible_floor == pytest.approx(0.45)
        assert policy.config.close_race_margin == pytest.approx(0.15)
        assert policy.config.few_views_below == 4

    def test_the_acceptance_threshold_is_not_duplicated_in_the_policy_config(
        self, policy: IdentityPolicy
    ) -> None:
        """It comes from ReidConfig, so the matcher and the policy cannot disagree."""
        fields = set(IdentityPolicyConfig.__dataclass_fields__)

        assert "acceptance_threshold" not in fields
        assert "margin_threshold" not in fields

    def test_the_near_miss_band_sits_under_acceptance(self, policy: IdentityPolicy) -> None:
        assert (
            policy.config.plausible_floor
            <= policy.config.near_acceptance_floor
            < policy.acceptance_threshold
        )
