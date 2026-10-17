"""Hold the questions the system has asked the user, and the rules about asking again.

Subfunctions:
- Question is one open or closed question: the track it is about, the candidates offered,
  the plain-English reasons, the new object's crop and its state.
- Answer is what the user decided: pick a candidate, call it a new item, say it is not an
  item, or skip.
- QuestionRegistry owns the lifecycle. It opens a question only when the rules allow it
  (one per track, a limit on how many are open at once, a cooldown per track and per pair of
  look-alike items), closes it on an answer, an expiry or the track leaving view, and counts
  how many times an object has stayed unresolved so the caller can give up on it.

Why a registry: the identity policy decides *that* a question is warranted, but it is pure
arithmetic and remembers nothing. Something has to remember what is already open, what was
just asked and skipped, and how many times one object has stalled. That is this module, and
it is also what lets the system ask less as it matures: a confusion already answered or
skipped is put on a cooldown instead of being asked again every time the object reappears.

Like the policy, this module is pure bookkeeping. It reads no clock (the caller passes the
time in), touches no camera, model, database or screen, and logs nothing: the coordinator
that applies a question's outcome logs it, so each action is still logged once, by the layer
that acts on it. A closed question is returned to the caller and then forgotten, so a long
session does not accumulate history.

Answers that arrive late are ignored, never applied: an answer to a question that has already
expired, been cancelled or been answered returns None.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from project_auto.events.identity_policy import (
    AskContext,
    IdentityOutcome,
    IdentityPolicyConfig,
    PolicyCandidate,
    PolicyDecision,
)


class QuestionState(str, Enum):
    """Where one question stands."""

    OPEN = "open"
    ANSWERED = "answered"
    SKIPPED = "skipped"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class AnswerKind(str, Enum):
    """What the user decided."""

    PICK = "pick"
    NEW = "new"
    NOT_AN_ITEM = "not_an_item"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class Answer:
    """One user decision. item_id is set for PICK and for nothing else."""

    kind: AnswerKind
    item_id: int | None = None

    @classmethod
    def pick(cls, item_id: int) -> Answer:
        """Return the answer "it is this item"."""
        return cls(AnswerKind.PICK, item_id)

    @classmethod
    def new_item(cls) -> Answer:
        """Return the answer "it is a new item"."""
        return cls(AnswerKind.NEW)

    @classmethod
    def not_an_item(cls) -> Answer:
        """Return the answer "this is not an item at all" (a hand, the table)."""
        return cls(AnswerKind.NOT_AN_ITEM)

    @classmethod
    def skip(cls) -> Answer:
        """Return the answer "not now"; the system must not guess against it."""
        return cls(AnswerKind.SKIP)


@dataclass(slots=True)
class Question:
    """One question put to the user.

    crop is the new object's picture, held so the overlay can show it; candidate_images maps
    each offered item to the path of its thumbnail (None when it has none). Both are carried,
    never loaded here, so this module stays free of image and disk work. poor_view is copied
    from the decision: the answer decides who the object is, but that view is never stored.
    """

    id: int
    track_id: int
    created_at: float
    expires_at: float
    candidates: tuple[PolicyCandidate, ...]
    reasons: tuple[str, ...]
    evidence: tuple[str, ...] = ()
    crop: object | None = None
    candidate_images: Mapping[int, Path | None] = field(default_factory=dict)
    poor_view: bool = False
    state: QuestionState = QuestionState.OPEN
    answer: Answer | None = None
    closed_at: float | None = None

    @property
    def is_open(self) -> bool:
        """Return whether this question is still waiting for an answer."""
        return self.state is QuestionState.OPEN

    @property
    def offered_item_ids(self) -> tuple[int, ...]:
        """Return the offered items, best candidate first."""
        return tuple(candidate.item_id for candidate in self.candidates)


def _pair_key(item_ids: tuple[int, ...]) -> frozenset[int] | None:
    """Return the key that identifies one confusion: the two best candidates, unordered."""
    return frozenset(item_ids[:2]) if item_ids else None


@dataclass(slots=True)
class QuestionRegistry:
    """Own every open question and every rule about asking again.

    Every method that depends on time takes it as a parameter, in the same units as the
    coordinator's clock, so the registry is deterministic under test.
    """

    config: IdentityPolicyConfig
    _open: dict[int, Question] = field(default_factory=dict, init=False, repr=False)
    _open_by_track: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _track_quiet_until: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _pair_quiet_until: dict[frozenset[int], float] = field(
        default_factory=dict, init=False, repr=False
    )
    _unresolved: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _next_id: int = field(default=1, init=False, repr=False)

    # --- Asking -----------------------------------------------------------------------

    def context(
        self,
        track_id: int,
        now: float,
        candidate_item_ids: tuple[int, ...] = (),
        claimed_item_ids: frozenset[int] = frozenset(),
    ) -> AskContext:
        """Return what the policy needs to know about the state of asking.

        candidate_item_ids are the items the policy is about to weigh, best first: if the
        two best were asked about recently, the best is reported as on cooldown.
        claimed_item_ids is the caller's knowledge of which items are on other visible
        tracks; the registry cannot know that, so it passes it through.
        """
        pair = _pair_key(candidate_item_ids)
        pair_quiet = pair is not None and self._pair_quiet_until.get(pair, 0.0) > now
        return AskContext(
            claimed_item_ids=claimed_item_ids,
            items_on_cooldown=frozenset(candidate_item_ids[:1]) if pair_quiet else frozenset(),
            open_questions=len(self._open),
            # A track with a question already waiting counts as on cooldown: asking it again
            # would only ever duplicate the question.
            track_on_cooldown=(
                self._track_quiet_until.get(track_id, 0.0) > now or track_id in self._open_by_track
            ),
        )

    def ask(
        self,
        track_id: int,
        decision: PolicyDecision,
        now: float,
        crop: object | None = None,
        candidate_images: Mapping[int, Path | None] | None = None,
    ) -> Question | None:
        """Open a question for an ASK decision, or return None if the rules forbid it.

        A track that already has an open question gets that same question back, so a repeat
        call is harmless. Past the open-question limit, or inside a track or pair cooldown,
        nothing is opened and the caller should defer the track instead of guessing.
        """
        if decision.outcome is not IdentityOutcome.ASK:
            raise ValueError(f"only an ASK decision can open a question, got {decision.outcome}")

        existing = self._open_by_track.get(track_id)
        if existing is not None:
            return self._open[existing]
        if self._track_quiet_until.get(track_id, 0.0) > now:
            return None
        if len(self._open) >= self.config.max_open_questions:
            return None
        offered = tuple(candidate.item_id for candidate in decision.candidates)
        pair = _pair_key(offered)
        if pair is not None and self._pair_quiet_until.get(pair, 0.0) > now:
            return None

        question = Question(
            id=self._next_id,
            track_id=track_id,
            created_at=now,
            expires_at=now + self.config.question_ttl_seconds,
            candidates=decision.candidates,
            reasons=decision.reasons,
            evidence=decision.evidence,
            crop=crop,
            candidate_images=dict(candidate_images or {}),
            poor_view=decision.poor_view,
        )
        self._next_id += 1
        self._open[question.id] = question
        self._open_by_track[track_id] = question.id
        return question

    # --- Answering --------------------------------------------------------------------

    def answer(self, question_id: int, answer: Answer, now: float) -> Question | None:
        """Apply the user's answer, or return None if the question is no longer open.

        None covers an unknown id, a question already closed, and one whose time ran out
        before the answer arrived (which is closed as EXPIRED on the spot). A stale answer
        is never applied. Raises ValueError for an answer that makes no sense for this
        question, such as picking an item that was not offered.
        """
        question = self._open.get(question_id)
        if question is None:
            return None
        if now >= question.expires_at:
            self._close(question, QuestionState.EXPIRED, now)
            self._track_quiet_until[question.track_id] = now + self.config.track_cooldown_seconds
            self.note_unresolved(question.track_id)
            return None

        self._validate(question, answer)
        state = QuestionState.SKIPPED if answer.kind is AnswerKind.SKIP else QuestionState.ANSWERED
        question.answer = answer
        self._close(question, state, now)

        # The confusion is now handled either way (settled, or the user asked not to be
        # bothered), so neither this track nor this pair of items is asked about again soon.
        self._track_quiet_until[question.track_id] = now + self.config.track_cooldown_seconds
        pair = _pair_key(question.offered_item_ids)
        if pair is not None:
            self._pair_quiet_until[pair] = now + self.config.item_pair_cooldown_seconds
        if answer.kind is not AnswerKind.SKIP:
            # A real answer resolves the object. A skip is "not now" and deliberately does
            # not count towards giving up: the plan is to never guess against it.
            self._unresolved.pop(question.track_id, None)
        return question

    @staticmethod
    def _validate(question: Question, answer: Answer) -> None:
        """Reject an answer that cannot apply to this question."""
        if answer.kind is AnswerKind.PICK:
            if answer.item_id is None or answer.item_id not in question.offered_item_ids:
                raise ValueError(f"item {answer.item_id} was not offered in question {question.id}")
        elif answer.item_id is not None:
            raise ValueError(f"a {answer.kind.value} answer cannot name an item")

    # --- Time and tracks --------------------------------------------------------------

    def expire(self, now: float) -> list[Question]:
        """Close every question whose time has run out and return them.

        An unanswered question puts its track on cooldown and counts as one unresolved
        attempt. Also forgets cooldowns that have lapsed, so the registry does not grow
        over a long session.
        """
        expired = [q for q in self._open.values() if now >= q.expires_at]
        for question in expired:
            self._close(question, QuestionState.EXPIRED, now)
            self._track_quiet_until[question.track_id] = now + self.config.track_cooldown_seconds
            self.note_unresolved(question.track_id)
        self._track_quiet_until = {t: u for t, u in self._track_quiet_until.items() if u > now}
        self._pair_quiet_until = {p: u for p, u in self._pair_quiet_until.items() if u > now}
        return expired

    def forget_track(self, track_id: int) -> Question | None:
        """Drop everything known about a track that retired; cancel its open question.

        Returns the cancelled question, if there was one. No cooldown is left behind: a
        reused track ID is a different object and starts fresh.
        """
        question = None
        question_id = self._open_by_track.get(track_id)
        if question_id is not None:
            question = self._open[question_id]
            self._close(question, QuestionState.CANCELLED, None)
        self._track_quiet_until.pop(track_id, None)
        self._unresolved.pop(track_id, None)
        return question

    # --- Giving up --------------------------------------------------------------------

    def note_unresolved(self, track_id: int) -> int:
        """Record one more attempt that left this object unresolved; return the total.

        Called for an expired question and by the coordinator when it defers a track
        because the policy said to wait. Never called for a skip.
        """
        self._unresolved[track_id] = self._unresolved.get(track_id, 0) + 1
        return self._unresolved[track_id]

    def unresolved_attempts(self, track_id: int) -> int:
        """Return how many times this object has stayed unresolved."""
        return self._unresolved.get(track_id, 0)

    def should_give_up(self, track_id: int) -> bool:
        """Return whether this object has stalled long enough to be added as new."""
        return self._unresolved.get(track_id, 0) >= self.config.give_up_after_attempts

    # --- Looking ----------------------------------------------------------------------

    def get(self, question_id: int) -> Question | None:
        """Return an open question by id, or None once it is closed or unknown."""
        return self._open.get(question_id)

    def open_for_track(self, track_id: int) -> Question | None:
        """Return the track's open question, if any."""
        question_id = self._open_by_track.get(track_id)
        return self._open[question_id] if question_id is not None else None

    def open_questions(self) -> tuple[Question, ...]:
        """Return every open question, oldest first (the order an overlay should show them)."""
        return tuple(sorted(self._open.values(), key=lambda question: question.id))

    def open_count(self) -> int:
        """Return how many questions are waiting for an answer."""
        return len(self._open)

    def _close(self, question: Question, state: QuestionState, now: float | None) -> None:
        """Move a question out of the open set; the caller keeps the returned object."""
        question.state = state
        question.closed_at = now
        self._open.pop(question.id, None)
        if self._open_by_track.get(question.track_id) == question.id:
            del self._open_by_track[question.track_id]
