"""SQLite persistence for Project AUTO items and meaningful events."""

from __future__ import annotations

from datetime import datetime
from math import fsum, hypot, isfinite
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, selectinload, sessionmaker

from project_auto.memory.models import Base, Item, ItemEvent, ItemEventType, ItemStatus, utc_now


class DatabaseStore:
    """Own the database engine and create short-lived sessions."""

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        database_url = f"sqlite:///{database_path.resolve().as_posix()}"

        self.engine: Engine = create_engine(database_url)

        @event.listens_for(self.engine, "connect")
        def enable_sqlite_foreign_keys(
            database_connection: Any,
            connection_record: Any,
        ) -> None:
            del connection_record
            cursor = database_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
        )

    def create_schema(self) -> None:
        """Create any missing Project AUTO database tables."""
        Base.metadata.create_all(self.engine)

    def update_item_prototype(self, item_id: int) -> list[float] | None:
        """Save and return the unit-length mean of an item's reference vectors.

        References are expected to be normalized by the embedding extractor and
        use identical model/preprocessing settings. No references clears the
        prototype. Invalid references raise ValueError without changing it.
        Call explicitly after reference changes; this is not an automatic hook.
        """
        with self.session_factory() as session:
            item = session.get(Item, item_id)
            if item is None:
                raise ValueError(f"Item {item_id} does not exist")

            references = item.embeddings
            prototype: list[float] | None = None
            if references:
                if len({reference.model_name for reference in references}) != 1:
                    raise ValueError("Cannot average embeddings from different models")

                vectors = [reference.embedding for reference in references]
                dimension: int | None = None
                for vector in vectors:
                    if not isinstance(vector, list) or not vector:
                        raise ValueError("Embeddings must be nonempty vectors")
                    if dimension is None:
                        dimension = len(vector)
                    if len(vector) != dimension:
                        raise ValueError("Embeddings must have matching dimensions")
                    if any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not isfinite(value)
                        for value in vector
                    ):
                        raise ValueError("Embeddings must contain finite numbers")
                    if hypot(*vector) == 0.0:
                        raise ValueError("Embeddings must have nonzero length")

                count = len(vectors)
                mean = [fsum(value / count for value in column) for column in zip(*vectors)]
                norm = hypot(*mean)
                if not isfinite(norm) or norm == 0.0:
                    raise ValueError("The mean embedding must have finite, nonzero length")
                prototype = [value / norm for value in mean]

            item.item_prototype = prototype
            session.commit()

        return prototype

    @staticmethod
    def _validate_event_interval(
        started_at: datetime | None,
        finished_at: datetime | None,
    ) -> None:
        """Validate an optional complete, timezone-aware event interval."""
        if (started_at is None) is not (finished_at is None):
            raise ValueError("started_at and finished_at must be provided together")
        if started_at is None or finished_at is None:
            return
        if started_at.utcoffset() is None or finished_at.utcoffset() is None:
            raise ValueError("Event interval timestamps must be timezone-aware")
        if started_at > finished_at:
            raise ValueError("started_at must not be later than finished_at")

    @staticmethod
    def _serialize_box(
        box: tuple[int, int, int, int] | None,
    ) -> list[int] | None:
        """Convert an optional bbox tuple into JSON-compatible coordinates."""
        if box is None:
            return None
        if len(box) != 4:
            raise ValueError("A bounding box must contain four coordinates")

        return list(box)

    def create_item(
        self,
        class_name: str,
        display_name: str | None = None,
        identity_confidence: float | None = None,
        status: ItemStatus = ItemStatus.PRESENT,
    ) -> Item:
        """Create and return one permanent physical-object record."""
        item = Item(
            class_name=class_name,
            display_name=display_name,
            identity_confidence=identity_confidence,
            status=status,
        )

        with self.session_factory() as session:
            session.add(item)
            session.commit()

        return item

    def add_item_with_event(
        self,
        class_name: str,
        status: ItemStatus,
        source_track_id: int,
        detector_confidence: float,
    ) -> tuple[Item, ItemEvent]:
        """Create an item and its initial ADDED event in one transaction."""
        item = Item(
            class_name=class_name,
            status=status,
        )

        with self.session_factory() as session:
            session.add(item)
            session.flush()

            added_event = ItemEvent(
                item_id=item.id,
                event_type=ItemEventType.ADDED,
                source_track_id=source_track_id,
                detector_confidence=detector_confidence,
            )
            session.add(added_event)
            session.commit()

        return item, added_event

    def record_event(
        self,
        item_id: int,
        event_type: ItemEventType,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        source_track_id: int | None = None,
        detector_confidence: float | None = None,
        source_region: str | None = None,
        destination_region: str | None = None,
        source_box: tuple[int, int, int, int] | None = None,
        destination_box: tuple[int, int, int, int] | None = None,
        object_image_path: str | None = None,
        context_image_path: str | None = None,
        video_clip_path: str | None = None,
        notes: str | None = None,
    ) -> ItemEvent:
        """Record one meaningful event against a permanent item identity."""
        self._validate_event_interval(started_at, finished_at)
        event = ItemEvent(
            item_id=item_id,
            event_type=event_type,
            started_at=started_at,
            finished_at=finished_at,
            source_track_id=source_track_id,
            detector_confidence=detector_confidence,
            source_region=source_region,
            destination_region=destination_region,
            source_box=self._serialize_box(source_box),
            destination_box=self._serialize_box(destination_box),
            object_image_path=object_image_path,
            context_image_path=context_image_path,
            video_clip_path=video_clip_path,
            notes=notes,
        )

        with self.session_factory() as session:
            session.add(event)
            session.commit()

        return event

    def get_item(self, item_id: int) -> Item | None:
        """Return one permanent item and its event history, if it exists."""
        statement = (
            select(Item)
            .where(Item.id == item_id)
            .options(selectinload(Item.events))
        )

        with self.session_factory() as session:
            return session.scalar(statement)

    def list_present_items(self) -> list[Item]:
        """Return present and occluded items from newest to oldest."""
        statement = (
            select(Item)
            .where(Item.status.in_([ItemStatus.PRESENT, ItemStatus.OCCLUDED]))
            .options(selectinload(Item.events))
            .order_by(Item.last_seen_at.desc())
        )

        with self.session_factory() as session:
            return list(session.scalars(statement))

    def count_items(self) -> int:
        """Return the number of all permanent items in inventory history."""
        statement = select(func.count(Item.id))

        with self.session_factory() as session:
            return int(session.scalar(statement) or 0)

    def mark_removed(
        self,
        item_id: int,
        source_track_id: int | None = None,
        detector_confidence: float | None = None,
        source_region: str | None = None,
        context_image_path: str | None = None,
        video_clip_path: str | None = None,
        notes: str | None = None,
    ) -> ItemEvent | None:
        """Mark an item removed, recording the meaningful transition once."""
        with self.session_factory() as session:
            item = session.get(Item, item_id)
            if item is None:
                raise ValueError(f"Item {item_id} does not exist")
            if item.status is ItemStatus.REMOVED:
                return None

            item.status = ItemStatus.REMOVED
            event = ItemEvent(
                item=item,
                event_type=ItemEventType.REMOVED,
                source_track_id=source_track_id,
                detector_confidence=detector_confidence,
                source_region=source_region,
                context_image_path=context_image_path,
                video_clip_path=video_clip_path,
                notes=notes,
            )
            session.add(event)
            session.commit()

        return event

    def mark_occluded(
        self,
        item_id: int,
        source_track_id: int | None = None,
        source_region: str | None = None,
        context_image_path: str | None = None,
        notes: str | None = None,
    ) -> ItemEvent | None:
        """Mark an item occluded and record the status transition once."""
        with self.session_factory() as session:
            item = session.get(Item, item_id)
            if item is None:
                raise ValueError(f"Item {item_id} does not exist")
            if item.status is ItemStatus.OCCLUDED:
                return None

            item.status = ItemStatus.OCCLUDED
            event = ItemEvent(
                item=item,
                event_type=ItemEventType.STATUS_CHANGED,
                source_track_id=source_track_id,
                source_region=source_region,
                context_image_path=context_image_path,
                notes=notes,
            )
            session.add(event)
            session.commit()

        return event

    def mark_present(
        self,
        item_id: int,
        source_track_id: int | None = None,
        detector_confidence: float | None = None,
        destination_region: str | None = None,
        object_image_path: str | None = None,
        context_image_path: str | None = None,
        notes: str | None = None,
    ) -> ItemEvent | None:
        """Mark an item present and record the corresponding transition once."""
        with self.session_factory() as session:
            item = session.get(Item, item_id)
            if item is None:
                raise ValueError(f"Item {item_id} does not exist")
            if item.status is ItemStatus.PRESENT:
                return None

            event_type = (
                ItemEventType.RETURNED
                if item.status is ItemStatus.REMOVED
                else ItemEventType.STATUS_CHANGED
            )
            item.status = ItemStatus.PRESENT
            item.last_seen_at = utc_now()

            event = ItemEvent(
                item=item,
                event_type=event_type,
                source_track_id=source_track_id,
                detector_confidence=detector_confidence,
                destination_region=destination_region,
                object_image_path=object_image_path,
                context_image_path=context_image_path,
                notes=notes,
            )
            session.add(event)
            session.commit()

        return event

    def mark_returned(
        self,
        item_id: int,
        source_track_id: int | None = None,
        detector_confidence: float | None = None,
    ) -> ItemEvent:
        """Mark a permanently identified removed item as returned."""
        # TODO(ReID): Call this only after associative memory resolves the
        # observation to item_id above its configured confidence threshold.
        with self.session_factory() as session:
            item = session.get(Item, item_id)
            if item is None:
                raise ValueError(f"Item {item_id} does not exist")
            if item.status is not ItemStatus.REMOVED:
                raise ValueError(f"Item {item_id} is not removed")

            item.status = ItemStatus.PRESENT
            item.last_seen_at = utc_now()
            returned_event = ItemEvent(
                item=item,
                event_type=ItemEventType.RETURNED,
                source_track_id=source_track_id,
                detector_confidence=detector_confidence,
            )
            session.add(returned_event)
            session.commit()

        return returned_event

    def record_movement(
        self,
        item_id: int,
        source_region: str | None = None,
        destination_region: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        source_box: tuple[int, int, int, int] | None = None,
        destination_box: tuple[int, int, int, int] | None = None,
        source_track_id: int | None = None,
        detector_confidence: float | None = None,
        object_image_path: str | None = None,
        context_image_path: str | None = None,
        video_clip_path: str | None = None,
        notes: str | None = None,
    ) -> ItemEvent:
        """Record a visible relocation without changing the item's status."""
        self._validate_event_interval(started_at, finished_at)
        if (source_box is None) is not (destination_box is None):
            raise ValueError("source_box and destination_box must be provided together")
        with self.session_factory() as session:
            item = session.get(Item, item_id)
            if item is None:
                raise ValueError(f"Item {item_id} does not exist")

            item.last_seen_at = utc_now()
            event = ItemEvent(
                item=item,
                event_type=ItemEventType.MOVED,
                started_at=started_at,
                finished_at=finished_at,
                source_track_id=source_track_id,
                detector_confidence=detector_confidence,
                source_region=source_region,
                destination_region=destination_region,
                source_box=self._serialize_box(source_box),
                destination_box=self._serialize_box(destination_box),
                object_image_path=object_image_path,
                context_image_path=context_image_path,
                video_clip_path=video_clip_path,
                notes=notes,
            )
            session.add(event)
            session.commit()

        return event

    def get_item_history(self, item_id: int) -> list[ItemEvent]:
        """Return an item's meaningful events from oldest to newest."""
        statement = (
            select(ItemEvent)
            .where(ItemEvent.item_id == item_id)
            .order_by(ItemEvent.occurred_at.asc(), ItemEvent.id.asc())
        )

        with self.session_factory() as session:
            return list(session.scalars(statement))
