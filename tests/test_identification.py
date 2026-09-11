"""Standalone ReferenceManager tests: dedup, cooldown, and queue-full bookkeeping."""

from __future__ import annotations

from project_auto.events.identification import IdentificationState, ReferenceManager


def make_manager(now: float = 0.0) -> tuple[ReferenceManager, list[float]]:
    clock_box = [now]
    manager = ReferenceManager(clock=lambda: clock_box[0])
    return manager, clock_box


def test_unidentified_track_can_submit() -> None:
    manager, _ = make_manager()
    assert manager.state(7) is IdentificationState.UNIDENTIFIED
    assert manager.can_submit(7, now=0.0) is True


def test_queued_track_cannot_be_resubmitted() -> None:
    manager, _ = make_manager()
    manager.mark_queued(7)

    assert manager.state(7) is IdentificationState.QUEUED
    assert manager.can_submit(7, now=0.0) is False


def test_identified_track_can_still_submit_a_capture_job() -> None:
    manager, _ = make_manager()
    manager.mark_identified(7)

    assert manager.can_submit(7, now=0.0) is True


def test_deferred_track_is_ineligible_until_cooldown_elapses() -> None:
    manager, _ = make_manager()
    manager.mark_deferred(7, now=0.0, cooldown_seconds=5.0, reason="bad_mask")

    assert manager.can_submit(7, now=1.0) is False
    assert manager.can_submit(7, now=5.0) is True
    assert manager.reason(7) == "bad_mask"


def test_queue_full_uses_its_own_shorter_cooldown() -> None:
    manager, _ = make_manager()
    manager.mark_queue_full(7, now=0.0, retry_seconds=0.5)

    assert manager.state(7) is IdentificationState.WAITING_FOR_BETTER_VIEW
    assert manager.can_submit(7, now=0.2) is False
    assert manager.can_submit(7, now=0.5) is True


def test_forget_clears_all_state_for_a_retired_track() -> None:
    manager, _ = make_manager()
    manager.mark_deferred(7, now=0.0, cooldown_seconds=5.0, reason="bad_mask")

    manager.forget(7)

    assert manager.state(7) is IdentificationState.UNIDENTIFIED
    assert manager.reason(7) is None
    assert manager.can_submit(7, now=0.0) is True


def test_pending_track_ids_only_lists_cooling_down_tracks() -> None:
    manager, _ = make_manager()
    manager.mark_queued(1)
    manager.mark_identified(2)
    manager.mark_deferred(3, now=0.0, cooldown_seconds=5.0, reason="bad_mask")
    manager.mark_queue_full(4, now=0.0, retry_seconds=0.5)

    assert sorted(manager.pending_track_ids()) == [3, 4]
