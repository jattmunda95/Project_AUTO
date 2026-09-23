"""Standalone reference-policy tests: sparse, event-driven candidate nomination.

The policy is pure bookkeeping over frame numbers, so every scenario here runs
without a clock, a frame buffer, SAM, DINO, or a database.
"""

from __future__ import annotations

from project_auto.events.reference_policy import (
    CandidateKind,
    ReferencePolicy,
    ReferencePolicyState,
)


def make_policy(**overrides) -> ReferencePolicy:
    settings = {
        "initial_reference_count": 2,
        "initial_capture_spacing_frames": 10,
        "move_candidate_delay_frames": 5,
        "candidate_retry_frames": 4,
        "max_movement_reference_attempts": 2,
    }
    settings.update(overrides)
    return ReferencePolicy(**settings)


def test_unknown_track_never_nominates() -> None:
    policy = make_policy()

    assert policy.should_nominate(7, 0).nominate is False


def test_new_item_waits_out_the_spacing_before_its_second_baseline_reference() -> None:
    policy = make_policy()
    policy.on_item_created(7, frame_index=0)

    assert policy.state(7) is ReferencePolicyState.NEEDS_INITIAL
    for frame_index in range(0, 10):
        assert policy.should_nominate(7, frame_index).nominate is False

    decision = policy.should_nominate(7, 10)

    assert decision.nominate is True
    assert decision.kind is CandidateKind.INITIAL


def test_completing_the_baseline_returns_the_track_to_idle() -> None:
    policy = make_policy()
    policy.on_item_created(7, frame_index=0)
    decision = policy.should_nominate(7, 10)
    policy.on_candidate_accepted(7, decision.kind)

    assert policy.state(7) is ReferencePolicyState.IDLE


def test_a_static_item_stops_nominating_once_its_baseline_is_complete() -> None:
    policy = make_policy()
    policy.on_item_created(7, frame_index=0)
    policy.on_candidate_accepted(7, policy.should_nominate(7, 10).kind)

    # However long it stays visible and still, it earns nothing more.
    for frame_index in range(11, 400):
        assert policy.should_nominate(7, frame_index).nominate is False


def test_an_associated_existing_item_takes_no_baseline_references() -> None:
    policy = make_policy()
    policy.on_item_associated(7)

    assert policy.state(7) is ReferencePolicyState.IDLE
    for frame_index in range(0, 100):
        assert policy.should_nominate(7, frame_index).nominate is False


def test_move_start_arms_but_holds_off_for_the_candidate_delay() -> None:
    policy = make_policy()
    policy.on_item_associated(7)
    policy.on_move_start(7, frame_index=100)

    assert policy.state(7) is ReferencePolicyState.ARMED
    for frame_index in range(100, 105):
        assert policy.should_nominate(7, frame_index).nominate is False

    decision = policy.should_nominate(7, 105)

    assert decision.nominate is True
    assert decision.kind is CandidateKind.MOVEMENT


def test_a_failed_movement_candidate_waits_the_retry_interval() -> None:
    policy = make_policy()
    policy.on_item_associated(7)
    policy.on_move_start(7, frame_index=100)
    first = policy.should_nominate(7, 105)
    policy.on_candidate_rejected(7, first.kind)

    for frame_index in range(106, 109):
        assert policy.should_nominate(7, frame_index).nominate is False

    assert policy.should_nominate(7, 109).nominate is True


def test_one_movement_event_cannot_exceed_its_attempt_budget() -> None:
    policy = make_policy()  # max_movement_reference_attempts=2
    policy.on_item_associated(7)
    policy.on_move_start(7, frame_index=100)

    nominations = [
        decision
        for decision in (policy.should_nominate(7, frame_index) for frame_index in range(100, 300))
        if decision.nominate
    ]

    assert len(nominations) == 2
    assert all(decision.kind is CandidateKind.MOVEMENT for decision in nominations)
    assert policy.state(7) is ReferencePolicyState.EXHAUSTED


def test_a_later_movement_event_restores_the_attempt_budget() -> None:
    policy = make_policy()
    policy.on_item_associated(7)
    policy.on_move_start(7, frame_index=100)
    for frame_index in range(100, 300):
        policy.should_nominate(7, frame_index)
    assert policy.state(7) is ReferencePolicyState.EXHAUSTED

    policy.on_move_start(7, frame_index=400)
    nominations = [
        decision
        for decision in (policy.should_nominate(7, frame_index) for frame_index in range(400, 600))
        if decision.nominate
    ]

    assert len(nominations) == 2


def test_move_end_nominates_exactly_one_final_candidate_then_idles() -> None:
    policy = make_policy()
    policy.on_item_associated(7)
    policy.on_move_start(7, frame_index=100)
    policy.on_move_end(7, frame_index=120)

    decision = policy.should_nominate(7, 120)

    assert decision.nominate is True
    assert decision.kind is CandidateKind.MOVE_END
    assert policy.state(7) is ReferencePolicyState.IDLE
    for frame_index in range(121, 400):
        assert policy.should_nominate(7, frame_index).nominate is False


def test_move_end_still_offers_a_candidate_after_the_budget_is_spent() -> None:
    """The settled pose is the most valuable view, so it is never budget-blocked."""
    policy = make_policy()
    policy.on_item_associated(7)
    policy.on_move_start(7, frame_index=100)
    for frame_index in range(100, 300):
        policy.should_nominate(7, frame_index)
    assert policy.state(7) is ReferencePolicyState.EXHAUSTED

    policy.on_move_end(7, frame_index=300)
    decision = policy.should_nominate(7, 300)

    assert decision.nominate is True
    assert decision.kind is CandidateKind.MOVE_END


def test_a_baseline_that_keeps_failing_gives_up_instead_of_retrying_forever() -> None:
    policy = make_policy()  # max attempts = 2
    policy.on_item_created(7, frame_index=0)

    nominations = [
        decision
        for decision in (policy.should_nominate(7, frame_index) for frame_index in range(0, 500))
        if decision.nominate
    ]

    assert len(nominations) == 2
    assert policy.state(7) is ReferencePolicyState.IDLE


def test_forget_drops_all_state_for_a_retired_track() -> None:
    policy = make_policy()
    policy.on_item_created(7, frame_index=0)

    policy.forget(7)

    assert policy.should_nominate(7, 500).nominate is False


def test_a_single_initial_reference_count_needs_no_extra_baseline_capture() -> None:
    """The resolve embedding alone already satisfies a count of one."""
    policy = make_policy(initial_reference_count=1)
    policy.on_item_created(7, frame_index=0)

    assert policy.state(7) is ReferencePolicyState.IDLE
    for frame_index in range(0, 100):
        assert policy.should_nominate(7, frame_index).nominate is False
