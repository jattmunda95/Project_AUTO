"""Coordinate state decisions with persistent item and event storage.

Implemented subfunctions:
- Dispatch ADD, MOVED, and REMOVE through state decisions and atomic store operations.
- Maintain provisional source-track to permanent-item bindings for lifecycle events.
- Provide an explicit RETURNED path and a no-event association for a matched present item.
- Bind resolved tracks after successful persistence and reject duplicate visible claims.

The identity coordinator (project_auto.events.coordinator) supplies identity decisions
from scene processing and calls process_add/process_return/associate_existing_item
directly; it owns retries and gallery refresh. No crop preparation, model inference, or
reference-capture scheduling belongs here.
"""

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
        # Capture dimensions are in-memory context for event-only normalized areas.
        self.frame_size: tuple[int, int] | None = None
        self._item_ids_by_track_id: dict[int, int] = {}

    def item_id_for_track(self, track_id: int) -> int | None:
        """Return the permanent item currently bound to one visible track, if any."""
        return self._item_ids_by_track_id.get(track_id)

    def is_item_claimed(self, item_id: int, excluding_track_id: int | None = None) -> bool:
        """Return whether some other visible track already claims this permanent item."""
        return any(
            associated_item_id == item_id and track_id != excluding_track_id
            for track_id, associated_item_id in self._item_ids_by_track_id.items()
        )

    def associate_existing_item(self, track_id: int, item_id: int) -> None:
        """Bind a visible track to an already-present matched item; no event recorded."""
        if track_id in self._item_ids_by_track_id:
            raise ValueError(f"Track {track_id} is already associated with an item")
        if self.is_item_claimed(item_id):
            raise ValueError(f"Item {item_id} is already claimed by a visible track")

        self._item_ids_by_track_id[track_id] = item_id

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
            destination_box=signal.detection.box,
            frame_size=self.frame_size,
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
            frame_size=self.frame_size,
        )

    def process_remove(
        self,
        signal: TrackSignal,
        decision: StateDecision,
    ) -> ItemEvent | None:
        """Persist removal for an associated item and release its track binding.

        The track itself is retiring (the tracker deletes its own record for it),
        so its binding must not outlive it: a stale entry here would make
        is_item_claimed report this item as claimed forever, blocking any future
        RETURNED/associate resolution on a new track_id for the same item.
        """
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
            source_box=signal.detection.box,
            frame_size=self.frame_size,
        )
        del self._item_ids_by_track_id[signal.track_id]

        return removed_event

    def process_return(self, signal: TrackSignal) -> ItemEvent:
        """Persist a scene-processor-resolved return and bind the track afterward.

        The identity coordinator calls this only for a matched REMOVED item; a
        matched PRESENT item needs associate_existing_item instead, not RETURNED.
        """
        decision = decide_return_signal(signal)
        if decision.event_type is not ItemEventType.RETURNED:
            raise ValueError("process_return requires a RETURNED state decision")
        if signal.detection.track_id != signal.track_id:
            raise ValueError("Signal and detection track IDs must match")
        if signal.item_id is None:
            raise ValueError("A RETURNED signal requires a permanent item_id")
        if signal.track_id in self._item_ids_by_track_id:
            raise ValueError(f"Track {signal.track_id} is already associated with an item")
        if self.is_item_claimed(signal.item_id):
            raise ValueError(f"Item {signal.item_id} is already claimed by a visible track")

        returned_event = self.store.mark_returned(
            item_id=signal.item_id,
            source_track_id=signal.track_id,
            detector_confidence=signal.detection.confidence,
            destination_box=signal.detection.box,
            frame_size=self.frame_size,
        )
        self._item_ids_by_track_id[signal.track_id] = signal.item_id

        return returned_event
