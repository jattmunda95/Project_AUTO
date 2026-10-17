"""Decide what one identity resolve result means: match it, ask the user, wait, or add it.

Subfunctions:
- IdentityPolicyConfig loads the ask-confidence weights, bands and limits from
  configs/identity_policy.yaml. The acceptance and margin thresholds are NOT in that
  file: they are supplied from the already-loaded ReidConfig, so the matcher and this
  policy can never disagree about what counts as a match.
- PolicyCandidate, ViewQuality and AskContext are the plain values the caller supplies;
  PolicyDecision is the verdict, carrying the candidates to offer and the reasons to show.
- IdentityPolicy.decide applies the checks in a fixed order: candidates already visible
  elsewhere are dropped, a clear winner matches, and anything left over is scored for
  evidence that a question is worth asking. A poor view does not stop a question; it marks
  the decision poor_view so the answer is applied without storing that view.

Why a question at all: the matcher's acceptance threshold is a hard line through a soft
distribution. One phone seen from its screen side and from its back scores about 0.49
against itself while acceptance is 0.68, so it becomes two items. Rather than guessing or
waiting forever, the system asks once and learns a confirmed view from the answer.

This module is pure arithmetic. It reads no clock, no database and no frame, runs no model,
imports nothing from the perception or memory layers, and performs no side effects -- the
caller resolves cooldowns into booleans and applies whatever comes back. Reasons are plain
English and deliberately carry no scores or thresholds: they are written to be shown to the
person answering, who needs to know why they are being asked, not what the numbers were.

Every threshold behind the decision is a hand-set estimate replayed against
logs/reid_runs/20261016T030339Z, not a value fitted to labelled data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml

# off      -> the coordinator ignores the policy; behaviour is exactly as before it existed.
# observe  -> the policy runs on every result and logs what it WOULD do, but the previous
#             behaviour still applies. Safe before the answer overlay exists.
# enforce  -> ASK and WAIT outcomes defer the track instead of adding or guessing. Only
#             meaningful once something can actually put a question in front of the user.
POLICY_MODES = ("off", "observe", "enforce")


class IdentityOutcome(str, Enum):
    """What should happen to one resolved track."""

    MATCH = "match"
    NEW = "new"
    WAIT = "wait"
    ASK = "ask"


@dataclass(frozen=True, slots=True)
class PolicyCandidate:
    """One scored gallery item as the policy sees it.

    Mirrors memory.reid.CandidateScore, plus the two pieces of context the matcher does
    not have: whether this item was last seen where the object now sits, and a label to
    name it in a question. The duplication is deliberate -- importing CandidateScore would
    pull torch and transformers into a module that is only arithmetic.

    color_score and aspect_score are None when the matcher fell back to DINO alone.
    item_label is descriptive text only and is never used to decide anything.
    """

    item_id: int
    final_score: float
    dino_score: float
    color_score: float | None = None
    aspect_score: float | None = None
    n_references: int = 0
    region_matches: bool = False
    item_label: str | None = None

    @classmethod
    def from_score(
        cls, score: object, region_matches: bool = False, item_label: str | None = None
    ) -> PolicyCandidate:
        """Build a candidate from a matcher score without importing the matcher.

        score is anything with the attributes of memory.reid.CandidateScore. Duck-typed on
        purpose: importing the real class would pull torch into this pure module.
        """
        return cls(
            item_id=score.item_id,  # type: ignore[attr-defined]
            final_score=score.final_score,  # type: ignore[attr-defined]
            dino_score=score.dino_score,  # type: ignore[attr-defined]
            color_score=score.color_score,  # type: ignore[attr-defined]
            aspect_score=score.aspect_score,  # type: ignore[attr-defined]
            n_references=score.n_references,  # type: ignore[attr-defined]
            region_matches=region_matches,
            item_label=item_label,
        )

    @property
    def name(self) -> str:
        """Return how this item should be named in a question shown to the user."""
        return self.item_label or f"item {self.item_id}"


@dataclass(frozen=True, slots=True)
class ViewQuality:
    """How usable the crop behind this query was.

    mask_occupancy is masked foreground pixels over segmented box area: a mask-sanity
    measure, not occlusion detection.
    """

    mask_occupancy: float
    mask_score: float


@dataclass(frozen=True, slots=True)
class AskContext:
    """What else is going on, resolved to plain values by the caller.

    claimed_item_ids are items currently held by another visible track, so they cannot
    also be this object. items_on_cooldown are items this same track was already asked
    about recently. Both cooldown flags are booleans, not times, so the policy stays
    deterministic under test and never reads a clock.
    """

    claimed_item_ids: frozenset[int] = frozenset()
    items_on_cooldown: frozenset[int] = frozenset()
    open_questions: int = 0
    track_on_cooldown: bool = False


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """One verdict. candidates is ranked best first and is what a question would offer."""

    outcome: IdentityOutcome
    code: str
    reasons: tuple[str, ...] = ()
    item_id: int | None = None
    candidates: tuple[PolicyCandidate, ...] = ()
    confidence: float = 0.0
    # Machine-readable names of the evidence that fired, for logging and diagnostics.
    evidence: tuple[str, ...] = ()
    # The question was asked about a view too poor to learn from. The answer still decides
    # who the object is, but that view must never be stored as a picture of it.
    poor_view: bool = False


@dataclass(frozen=True, slots=True)
class IdentityPolicyConfig:
    """Ask-confidence settings; see configs/identity_policy.yaml for what each one cost."""

    # See POLICY_MODES. observe is the safe default until the answer overlay exists.
    mode: str = "observe"
    min_mask_occupancy: float = 0.40
    min_mask_score: float = 0.60
    # A poor view always blocks a question; by default it does not block a confident
    # match. See configs/identity_policy.yaml for what measuring the alternative cost.
    view_blocks_match: bool = False
    ask_threshold: float = 0.50
    near_acceptance_floor: float = 0.50
    # 0.45, not 0.50, so the two faces of one phone (about 0.49 apart) stop becoming a
    # duplicate item. See configs/identity_policy.yaml.
    plausible_floor: float = 0.45
    close_race_margin: float = 0.15
    few_views_below: int = 4
    weight_near_acceptance: float = 0.50
    weight_plausible: float = 0.35
    weight_close_race: float = 0.30
    weight_component_disagreement: float = 0.20
    weight_few_views: float = 0.15
    weight_region_match: float = 0.15
    max_open_questions: int = 3
    # Read by the question registry, not by this module, but kept here so every setting
    # for the ask-and-answer flow lives in one file.
    track_cooldown_seconds: float = 30.0
    item_pair_cooldown_seconds: float = 120.0
    question_ttl_seconds: float = 60.0
    give_up_after_attempts: int = 3

    @classmethod
    def from_yaml(cls, config_path: Path) -> IdentityPolicyConfig:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        view = config.get("view", {})
        bands = config.get("bands", {})
        evidence = config.get("evidence", {})
        limits = config.get("limits", {})
        defaults = cls()

        mode = str(config.get("mode", defaults.mode))
        if mode not in POLICY_MODES:
            raise ValueError(f"identity policy mode must be one of {POLICY_MODES}, got {mode!r}")

        return cls(
            mode=mode,
            min_mask_occupancy=float(view.get("min_mask_occupancy", defaults.min_mask_occupancy)),
            min_mask_score=float(view.get("min_mask_score", defaults.min_mask_score)),
            view_blocks_match=bool(view.get("blocks_match", defaults.view_blocks_match)),
            ask_threshold=float(config.get("ask_threshold", defaults.ask_threshold)),
            near_acceptance_floor=float(
                bands.get("near_acceptance_floor", defaults.near_acceptance_floor)
            ),
            plausible_floor=float(bands.get("plausible_floor", defaults.plausible_floor)),
            close_race_margin=float(bands.get("close_race_margin", defaults.close_race_margin)),
            few_views_below=int(bands.get("few_views_below", defaults.few_views_below)),
            weight_near_acceptance=float(
                evidence.get("near_acceptance", defaults.weight_near_acceptance)
            ),
            weight_plausible=float(evidence.get("plausible", defaults.weight_plausible)),
            weight_close_race=float(evidence.get("close_race", defaults.weight_close_race)),
            weight_component_disagreement=float(
                evidence.get("component_disagreement", defaults.weight_component_disagreement)
            ),
            weight_few_views=float(evidence.get("few_views", defaults.weight_few_views)),
            weight_region_match=float(evidence.get("region_match", defaults.weight_region_match)),
            max_open_questions=int(limits.get("max_open_questions", defaults.max_open_questions)),
            track_cooldown_seconds=float(
                limits.get("track_cooldown_seconds", defaults.track_cooldown_seconds)
            ),
            item_pair_cooldown_seconds=float(
                limits.get("item_pair_cooldown_seconds", defaults.item_pair_cooldown_seconds)
            ),
            question_ttl_seconds=float(
                limits.get("question_ttl_seconds", defaults.question_ttl_seconds)
            ),
            give_up_after_attempts=int(
                limits.get("give_up_after_attempts", defaults.give_up_after_attempts)
            ),
        )


# Reason codes, so callers log and test against names rather than prose.
REASON_GALLERY_EMPTY = "gallery_empty"
REASON_BAD_VIEW = "bad_view"
REASON_ACCEPTED = "accepted"
REASON_ALL_CLAIMED = "all_candidates_claimed"
REASON_BELOW_FLOOR = "below_floor"
REASON_UNCONVINCING = "unconvincing"
REASON_ASK = "ask"
REASON_ASK_COOLDOWN = "ask_suppressed_cooldown"
REASON_ASK_LIMIT = "ask_suppressed_limit"

EVIDENCE_NEAR_ACCEPTANCE = "near_acceptance"
EVIDENCE_PLAUSIBLE = "plausible"
EVIDENCE_CLOSE_RACE = "close_race"
EVIDENCE_DISAGREEMENT = "component_disagreement"
EVIDENCE_FEW_VIEWS = "few_views"
EVIDENCE_REGION_MATCH = "region_match"


# Shown first when a question is asked about a poor view, so the person looks at the
# object itself rather than trusting the cropped picture.
POOR_VIEW_REASON = "I can't see this clearly, so check the object itself before answering."


def _sentence(text: str) -> str:
    """Capitalise a reason that begins with an item's name, which may be lower case."""
    return text[:1].upper() + text[1:] if text else text


NO_ASK_CONTEXT = AskContext()

_TRIGGERS = (
    EVIDENCE_NEAR_ACCEPTANCE,
    EVIDENCE_PLAUSIBLE,
    EVIDENCE_CLOSE_RACE,
    EVIDENCE_DISAGREEMENT,
)


@dataclass(frozen=True, slots=True)
class IdentityPolicy:
    """Turn one set of candidate scores into a decision, with reasons a person can read.

    acceptance_threshold and margin_threshold come from the ReidConfig the matcher is
    already using; passing them in rather than re-reading a YAML keeps one source of truth.
    """

    config: IdentityPolicyConfig
    acceptance_threshold: float
    margin_threshold: float = 0.0
    _max_offered: int = field(default=3, repr=False)

    def decide(
        self,
        candidates: tuple[PolicyCandidate, ...],
        view: ViewQuality,
        context: AskContext = NO_ASK_CONTEXT,
    ) -> PolicyDecision:
        """Return what to do with one resolve result; candidates may be in any order."""
        ranked = tuple(sorted(candidates, key=lambda item: item.final_score, reverse=True))
        view_is_usable = self.view_is_usable(view)

        if not ranked:
            return PolicyDecision(
                outcome=IdentityOutcome.NEW,
                code=REASON_GALLERY_EMPTY,
                reasons=("Nothing is saved yet, so this is a new object.",),
            )

        if not view_is_usable and self.config.view_blocks_match:
            return self._wait_on_view()

        best = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        margin = best.final_score - (runner_up.final_score if runner_up else 0.0)
        if best.final_score >= self.acceptance_threshold and margin >= self.margin_threshold:
            return PolicyDecision(
                outcome=IdentityOutcome.MATCH,
                code=REASON_ACCEPTED,
                item_id=best.item_id,
                candidates=ranked,
                reasons=(_sentence(f"{best.name} matches clearly."),),
            )

        possible = tuple(item for item in ranked if item.item_id not in context.claimed_item_ids)
        if not possible:
            return PolicyDecision(
                outcome=IdentityOutcome.NEW,
                code=REASON_ALL_CLAIMED,
                candidates=ranked,
                reasons=(
                    (
                        "Everything saved that looks like this is already somewhere else in "
                        "view, so this is a different object."
                    ),
                ),
            )

        confidence, evidence, reasons = self._score_evidence(possible)
        offered = possible[: self._max_offered]

        if confidence < self.config.ask_threshold:
            if possible[0].final_score < self.config.plausible_floor:
                return PolicyDecision(
                    outcome=IdentityOutcome.NEW,
                    code=REASON_BELOW_FLOOR,
                    candidates=offered,
                    confidence=confidence,
                    evidence=evidence,
                    reasons=("Nothing saved looks like this object.",),
                )
            return PolicyDecision(
                outcome=IdentityOutcome.WAIT,
                code=REASON_UNCONVINCING,
                candidates=offered,
                confidence=confidence,
                evidence=evidence,
                reasons=(
                    _sentence(
                        f"{possible[0].name} is only a loose likeness, so this is worth "
                        "another look."
                    ),
                ),
            )

        # Evidence is sufficient, so the question is asked even on a poor view: the person
        # can see the object, and waiting for a better frame stalled the very objects this
        # flow exists for. The decision is flagged instead, so the answer links the object
        # without ever saving the poor view as a picture of it.
        poor_view = not view_is_usable
        if poor_view:
            reasons = (POOR_VIEW_REASON, *reasons)

        # Never fall through to a guess: a suppressed question waits.
        if context.track_on_cooldown or possible[0].item_id in context.items_on_cooldown:
            return PolicyDecision(
                outcome=IdentityOutcome.WAIT,
                code=REASON_ASK_COOLDOWN,
                candidates=offered,
                confidence=confidence,
                evidence=evidence,
                reasons=("This object was already asked about recently.",),
            )
        if context.open_questions >= self.config.max_open_questions:
            return PolicyDecision(
                outcome=IdentityOutcome.WAIT,
                code=REASON_ASK_LIMIT,
                candidates=offered,
                confidence=confidence,
                evidence=evidence,
                reasons=("There are already as many questions waiting as the limit allows.",),
            )

        return PolicyDecision(
            outcome=IdentityOutcome.ASK,
            code=REASON_ASK,
            candidates=offered,
            confidence=confidence,
            evidence=evidence,
            reasons=reasons,
            poor_view=poor_view,
        )

    def view_is_usable(self, view: ViewQuality) -> bool:
        """Return whether the crop was good enough to judge an identity from, or learn from."""
        return (
            view.mask_occupancy >= self.config.min_mask_occupancy
            and view.mask_score >= self.config.min_mask_score
        )

    def _wait_on_view(self) -> PolicyDecision:
        """Wait for a better frame; only used when view_blocks_match is switched on."""
        return PolicyDecision(
            outcome=IdentityOutcome.WAIT,
            code=REASON_BAD_VIEW,
            reasons=(
                (
                    "The view is too poor to judge: the outline is unclear or the object is "
                    "partly hidden."
                ),
            ),
        )

    def _score_evidence(
        self, possible: tuple[PolicyCandidate, ...]
    ) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
        """Add up the evidence that a question is worth asking.

        Triggers stand on their own. Modifiers (how many views are stored, whether the
        object sits where that item was last seen) only count once a trigger has fired, so
        neither can cause a question by itself.
        """
        config = self.config
        best = possible[0]
        runner_up = possible[1] if len(possible) > 1 else None
        margin = best.final_score - (runner_up.final_score if runner_up else 0.0)
        above_floor = best.final_score >= config.plausible_floor

        confidence = 0.0
        evidence: list[str] = []
        reasons: list[str] = []

        # A score this high only reaches here when the match was refused, either by the
        # acceptance threshold or -- at or above it -- by the margin against a rival that
        # scored almost as well. Both are strong evidence that a question is worth asking.
        if best.final_score >= config.near_acceptance_floor:
            confidence += config.weight_near_acceptance
            evidence.append(EVIDENCE_NEAR_ACCEPTANCE)
            if best.final_score >= self.acceptance_threshold and runner_up is not None:
                reasons.append(
                    _sentence(
                        f"{best.name} and {runner_up.name} both look so much like this that "
                        "it is not safe to choose between them."
                    )
                )
            else:
                reasons.append(f"This looks like {best.name}, but not clearly enough to be sure.")
        elif config.plausible_floor <= best.final_score < config.near_acceptance_floor:
            confidence += config.weight_plausible
            evidence.append(EVIDENCE_PLAUSIBLE)
            reasons.append(f"This resembles {best.name}, though the likeness is weak.")

        if above_floor and runner_up is not None and margin < config.close_race_margin:
            confidence += config.weight_close_race
            evidence.append(EVIDENCE_CLOSE_RACE)
            # A refused tie above acceptance has already been explained in these words; the
            # evidence still counts, but the person should not read the same pair twice.
            if best.final_score < self.acceptance_threshold:
                reasons.append(
                    _sentence(f"{best.name} and {runner_up.name} look about equally alike.")
                )

        disagreement = self._disagreement(possible) if above_floor else None
        if disagreement is not None:
            confidence += config.weight_component_disagreement
            evidence.append(EVIDENCE_DISAGREEMENT)
            reasons.append(_sentence(disagreement))

        if any(name in _TRIGGERS for name in evidence):
            if 0 < best.n_references < config.few_views_below:
                confidence += config.weight_few_views
                evidence.append(EVIDENCE_FEW_VIEWS)
                reasons.append(
                    f"Only a few views of {best.name} are saved, so a close match is harder "
                    "to confirm."
                )
            region_match = next((item for item in possible if item.region_matches), None)
            if region_match is not None:
                confidence += config.weight_region_match
                evidence.append(EVIDENCE_REGION_MATCH)
                reasons.append(f"It is sitting where {region_match.name} was last seen.")

        return round(confidence, 4), tuple(evidence), tuple(reasons)

    def _disagreement(self, possible: tuple[PolicyCandidate, ...]) -> str | None:
        """Return a plain-English line when appearance and colour point different ways.

        Returns None when they agree with the winner, or when the matcher fell back to
        DINO alone and there is no colour score to compare.
        """
        if any(item.color_score is None for item in possible):
            return None

        best = possible[0]
        by_appearance = max(possible, key=lambda item: item.dino_score)
        by_colour = max(possible, key=lambda item: item.color_score or 0.0)
        if by_appearance.item_id == by_colour.item_id == best.item_id:
            return None

        if by_appearance.item_id != by_colour.item_id:
            return (
                f"Its overall appearance points to {by_appearance.name}, but its colour "
                f"points to {by_colour.name}."
            )
        return (
            f"{best.name} scores highest overall, but both its appearance and its colour "
            f"point to {by_appearance.name}."
        )  # _sentence() capitalises this at the call site
