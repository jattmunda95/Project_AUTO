"""Decide meaningful item-state changes without performing persistence."""

from __future__ import annotations

from dataclasses import dataclass

from project_auto.memory.models import ItemEventType, ItemStatus
from project_auto.perception.tracker import TrackSignal, TrackSignalType


@dataclass(frozen=True, slots=True)
class StateDecision:
    """The state and event that persistence should record for one decision."""

    status: ItemStatus
    event_type: ItemEventType


def decide_item_added() -> StateDecision:
    """Return the initial state and event for a newly recognized item."""
    return StateDecision(
        status=ItemStatus.PRESENT,
        event_type=ItemEventType.ADDED,
    )


def decide_item_removed() -> StateDecision:
    """Return the state and event for a confirmed item that timed out."""
    return StateDecision(
        status=ItemStatus.REMOVED,
        event_type=ItemEventType.REMOVED,
    )


def decide_item_moved() -> StateDecision:
    """Return the unchanged present state and event for a completed relocation."""
    return StateDecision(
        status=ItemStatus.PRESENT,
        event_type=ItemEventType.MOVED,
    )


def decide_item_returned() -> StateDecision:
    """Return the state and event for a permanently identified returned item."""
    # TODO(ReID): Dispatch this decision only after associative memory resolves an
    # observation to a permanent item_id; never infer a return from a tracker ID.
    return StateDecision(
        status=ItemStatus.PRESENT,
        event_type=ItemEventType.RETURNED,
    )


def decide_return_signal(signal: TrackSignal) -> StateDecision:
    """Decide an explicitly ReID-resolved return signal."""
    if signal.signal_type is not TrackSignalType.RETURNED:
        raise ValueError("decide_return_signal requires a RETURNED signal")
    if signal.item_id is None:
        raise ValueError("A RETURNED signal requires a permanent item_id")

    # TODO(ReID): Call this interface only from the future associative-memory
    # result path after its configured identity-confidence threshold is met.
    return decide_item_returned()


def decide_track_signal(signal: TrackSignal) -> StateDecision:
    """Translate one meaningful tracker signal into a persistence decision."""
    if signal.signal_type is TrackSignalType.ADD:
        return decide_item_added()
    if signal.signal_type is TrackSignalType.MOVED:
        return decide_item_moved()
    if signal.signal_type is TrackSignalType.REMOVE:
        return decide_item_removed()

    raise ValueError(f"Unsupported tracker signal: {signal.signal_type}")
