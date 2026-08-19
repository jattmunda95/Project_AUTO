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


def decide_track_signal(signal: TrackSignal) -> StateDecision:
    """Translate one meaningful tracker signal into a persistence decision."""
    if signal.signal_type is TrackSignalType.ADD:
        return decide_item_added()

    raise ValueError(f"Unsupported tracker signal: {signal.signal_type}")
