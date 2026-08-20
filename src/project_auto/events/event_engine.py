"""Coordinate state decisions with persistent item and event storage."""

from project_auto.memory.models import Item, ItemEvent, ItemEventType
from project_auto.memory.state_machine import StateDecision, decide_track_signal
from project_auto.memory.store import DatabaseStore
from project_auto.perception.tracker import TrackSignal, TrackSignalType


class EventEngine:
    """Connect meaningful tracker decisions to permanent item identities."""

    def __init__(self, store: DatabaseStore) -> None:
        """Retain the store and start with no track-to-item associations."""
        self.store = store
        self._item_ids_by_track_id: dict[int, int] = {}

    def process_signal(
        self,
        signal: TrackSignal,
    ) -> tuple[Item, ItemEvent] | ItemEvent | None:
        """Decide what one tracker signal means and dispatch its side effect."""
        decision = decide_track_signal(signal)

        if decision.event_type is ItemEventType.ADDED:
            return self.process_add(signal, decision)
        if decision.event_type is ItemEventType.REMOVED:
            return self.process_remove(signal, decision)

        raise ValueError(f"Unsupported state decision: {decision.event_type}")

    def process_add(
        self,
        signal: TrackSignal,
        decision: StateDecision,
    ) -> tuple[Item, ItemEvent]:
        """Persist one confirmed new item and remember its provisional track binding."""
        if signal.signal_type is not TrackSignalType.ADD:
            raise ValueError("process_add requires an ADD track signal")
        if decision.event_type is not ItemEventType.ADDED:
            raise ValueError("process_add requires an ADDED state decision")
        if signal.detection.track_id != signal.track_id:
            raise ValueError("Signal and detection track IDs must match")
        if signal.track_id in self._item_ids_by_track_id:
            raise ValueError(f"Track {signal.track_id} is already associated with an item")

        item, added_event = self.store.add_item_with_event(
            class_name=signal.detection.class_name,
            status=decision.status,
            source_track_id=signal.track_id,
            detector_confidence=signal.detection.confidence,
        )
        self._item_ids_by_track_id[signal.track_id] = item.id

        return item, added_event

    def process_remove(
        self,
        signal: TrackSignal,
        decision: StateDecision,
    ) -> ItemEvent | None:
        """Persist removal for an associated item while retaining its track binding."""
        if signal.signal_type is not TrackSignalType.REMOVE:
            raise ValueError("process_remove requires a REMOVE track signal")
        if decision.event_type is not ItemEventType.REMOVED:
            raise ValueError("process_remove requires a REMOVED state decision")
        if signal.detection.track_id != signal.track_id:
            raise ValueError("Signal and detection track IDs must match")

        item_id = self._item_ids_by_track_id.get(signal.track_id)
        if item_id is None:
            raise ValueError(f"Track {signal.track_id} is not associated with an item")

        removed_event = self.store.mark_removed(
            item_id=item_id,
            source_track_id=signal.track_id,
            detector_confidence=signal.detection.confidence,
        )

        return removed_event
