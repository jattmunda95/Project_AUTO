"""Decide *when* a frame should become a reference candidate; never what it contains.

Reference learning is sparse and event-driven. A static item earns no new
references no matter how long it stays visible: movement is what produces new
appearances (a new orientation, a flipped side, a different lighting angle), so
movement is what re-arms learning.

Lifecycle for one track:

    item created -> NEEDS_INITIAL -> (baseline reference 2) -> IDLE
    MOVE_START   -> ARMED -> nominate up to max_movement_reference_attempts
                          -> EXHAUSTED once they are spent
    MOVE_END     -> one final stable candidate -> IDLE

The identity resolve job already produces reference 1 for a newly created item,
so only the remaining baseline references are scheduled here.

This module owns timing and attempt budgets only. It performs no segmentation,
no embedding, no quality measurement and no persistence: should_nominate answers
"is this frame worth spending expensive work on", and the coordinator applies the
cheap gate and submits the job. All timing is in frame counts, supplied by the
caller, so the policy never reads a clock and stays deterministic under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ReferencePolicyState(str, Enum):
    """Where one track stands in the sparse reference-learning lifecycle."""

    NEEDS_INITIAL = "needs_initial"
    IDLE = "idle"
    ARMED = "armed"
    EXHAUSTED = "exhausted"


class CandidateKind(str, Enum):
    """Why a frame was nominated; decides how strictly it is later judged."""

    INITIAL = "initial"
    MOVEMENT = "movement"
    MOVE_END = "move_end"


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    """One nomination verdict for one frame."""

    nominate: bool
    kind: CandidateKind | None = None


NO_CANDIDATE = CandidateDecision(nominate=False)


@dataclass(slots=True)
class _TrackPolicyState:
    """Per-track bookkeeping; frame numbers are supplied by the caller."""

    state: ReferencePolicyState
    remaining_initial: int = 0
    next_eligible_frame: int = 0
    initial_attempts: int = 0
    movement_attempts: int = 0
    pending_move_end: bool = False


@dataclass(slots=True)
class ReferencePolicy:
    """Own when each track may nominate a reference candidate.

    Attempt budgets are consumed at nomination time, not when a result comes
    back, so a track cannot keep nominating while earlier expensive work is still
    in flight, and a candidate rejected by a cheap check still costs an attempt.
    """

    initial_reference_count: int = 2
    initial_capture_spacing_frames: int = 15
    move_candidate_delay_frames: int = 5
    candidate_retry_frames: int = 10
    max_movement_reference_attempts: int = 3
    _states: dict[int, _TrackPolicyState] = field(
        default_factory=dict, init=False, repr=False
    )

    def state(self, track_id: int) -> ReferencePolicyState:
        """Return one track's current policy state."""
        track_state = self._states.get(track_id)
        return track_state.state if track_state else ReferencePolicyState.IDLE

    def on_item_created(self, track_id: int, frame_index: int) -> None:
        """Schedule the remaining baseline references for a newly created item.

        The identity resolve job already saved reference 1 from the embedding it
        had to compute anyway, so only initial_reference_count - 1 remain.
        """
        remaining = max(0, self.initial_reference_count - 1)
        self._states[track_id] = _TrackPolicyState(
            state=(
                ReferencePolicyState.NEEDS_INITIAL
                if remaining > 0
                else ReferencePolicyState.IDLE
            ),
            remaining_initial=remaining,
            next_eligible_frame=frame_index + self.initial_capture_spacing_frames,
        )

    def on_item_associated(self, track_id: int) -> None:
        """Start a track bound to an already-known item at rest.

        An existing item already has a baseline appearance, so it learns only
        from movement, never from simply being seen again.
        """
        self._states[track_id] = _TrackPolicyState(state=ReferencePolicyState.IDLE)

    def on_move_start(self, track_id: int, frame_index: int) -> None:
        """Arm movement learning; the first candidate waits out the delay."""
        track_state = self._states.setdefault(
            track_id, _TrackPolicyState(state=ReferencePolicyState.IDLE)
        )
        track_state.state = ReferencePolicyState.ARMED
        track_state.movement_attempts = 0
        track_state.pending_move_end = False
        track_state.next_eligible_frame = frame_index + self.move_candidate_delay_frames

    def on_move_end(self, track_id: int, frame_index: int) -> None:
        """Queue exactly one final candidate for the newly settled appearance."""
        track_state = self._states.setdefault(
            track_id, _TrackPolicyState(state=ReferencePolicyState.IDLE)
        )
        track_state.state = ReferencePolicyState.IDLE
        track_state.movement_attempts = 0
        track_state.pending_move_end = True
        track_state.next_eligible_frame = frame_index

    def should_nominate(self, track_id: int, frame_index: int) -> CandidateDecision:
        """Decide whether this frame becomes a candidate, consuming an attempt.

        A returned nomination is already recorded as spent, so the caller must
        treat every nomination as used even when its cheap checks reject it.
        """
        track_state = self._states.get(track_id)
        if track_state is None:
            return NO_CANDIDATE

        # A settled appearance is a one-shot opportunity and outranks the
        # baseline schedule, which can wait for a later frame.
        if track_state.pending_move_end:
            track_state.pending_move_end = False
            track_state.state = ReferencePolicyState.IDLE
            track_state.next_eligible_frame = frame_index + self.candidate_retry_frames
            return CandidateDecision(nominate=True, kind=CandidateKind.MOVE_END)

        if frame_index < track_state.next_eligible_frame:
            return NO_CANDIDATE

        if (
            track_state.state is ReferencePolicyState.NEEDS_INITIAL
            and track_state.remaining_initial > 0
        ):
            if track_state.initial_attempts >= self.max_movement_reference_attempts:
                # A baseline view that keeps failing is not worth retrying forever;
                # movement will provide better opportunities than a static retry.
                track_state.state = ReferencePolicyState.IDLE
                return NO_CANDIDATE
            track_state.initial_attempts += 1
            track_state.next_eligible_frame = frame_index + self.candidate_retry_frames
            return CandidateDecision(nominate=True, kind=CandidateKind.INITIAL)

        if track_state.state is ReferencePolicyState.ARMED:
            if track_state.movement_attempts >= self.max_movement_reference_attempts:
                track_state.state = ReferencePolicyState.EXHAUSTED
                return NO_CANDIDATE
            track_state.movement_attempts += 1
            track_state.next_eligible_frame = frame_index + self.candidate_retry_frames
            return CandidateDecision(nominate=True, kind=CandidateKind.MOVEMENT)

        return NO_CANDIDATE

    def on_candidate_accepted(self, track_id: int, kind: CandidateKind) -> None:
        """Record that a nominated candidate was actually saved as a reference."""
        track_state = self._states.get(track_id)
        if track_state is None:
            return
        if kind is CandidateKind.INITIAL and track_state.remaining_initial > 0:
            track_state.remaining_initial -= 1
            if track_state.remaining_initial <= 0:
                track_state.state = ReferencePolicyState.IDLE

    def on_candidate_rejected(self, track_id: int, kind: CandidateKind) -> None:
        """Record a nominated candidate that never became a reference.

        The attempt and its retry spacing were already charged at nomination, so
        this only exists to keep the accept/reject paths symmetrical for callers
        and to leave one place to change if rejections should ever cost less.
        """
        return None

    def forget(self, track_id: int) -> None:
        """Drop all policy state for a retired or reused track."""
        self._states.pop(track_id, None)
