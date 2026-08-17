"""Decide meaningful item-state changes without performing persistence."""

from __future__ import annotations

from dataclasses import dataclass

from project_auto.memory.models import ItemEventType, ItemStatus


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
