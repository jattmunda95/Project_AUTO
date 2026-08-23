"""Coordinate state decisions with persistent item and event storage."""

from project_auto.memory.models import Item, ItemEvent, ItemEventType
from project_auto.memory.state_machine import (
    StateDecision,
    decide_return_signal,
    decide_track_signal,
)
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
        if decision.event_type is ItemEventType.MOVED:
            return self.process_move(signal, decision)
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

    def process_move(
        self,
        signal: TrackSignal,
        decision: StateDecision,
    ) -> ItemEvent:
        """Persist one completed relocation for an associated permanent item."""
        if signal.signal_type is not TrackSignalType.MOVED:
            raise ValueError("process_move requires a MOVED track signal")
        if decision.event_type is not ItemEventType.MOVED:
            raise ValueError("process_move requires a MOVED state decision")
        if signal.detection.track_id != signal.track_id:
            raise ValueError("Signal and detection track IDs must match")
        if signal.started_at is None or signal.finished_at is None:
            raise ValueError("A MOVED signal requires start and finish timestamps")
        if signal.source_box is None or signal.destination_box is None:
            raise ValueError("A MOVED signal requires source and destination boxes")

        item_id = self._item_ids_by_track_id.get(signal.track_id)
        if item_id is None:
            raise ValueError(f"Track {signal.track_id} is not associated with an item")

        return self.store.record_movement(
            item_id=item_id,
            started_at=signal.started_at,
            finished_at=signal.finished_at,
            source_box=signal.source_box,
            destination_box=signal.destination_box,
            source_track_id=signal.track_id,
            detector_confidence=signal.detection.confidence,
        )

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

    def process_return(self, signal: TrackSignal) -> ItemEvent:
        """Persist an explicitly ReID-resolved return without activating dispatch."""
        decision = decide_return_signal(signal)
        if decision.event_type is not ItemEventType.RETURNED:
            raise ValueError("process_return requires a RETURNED state decision")
        if signal.detection.track_id != signal.track_id:
            raise ValueError("Signal and detection track IDs must match")
        if signal.item_id is None:
            raise ValueError("A RETURNED signal requires a permanent item_id")

        # TODO(ReID): Once associative memory is implemented, call this method
        # from its confident-match path and then establish the new provisional
        # track-to-item association for subsequent lifecycle signals.
        return self.store.mark_returned(
            item_id=signal.item_id,
            source_track_id=signal.track_id,
            detector_confidence=signal.detection.confidence,
        )
