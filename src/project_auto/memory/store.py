"""SQLite persistence for Project AUTO items and meaningful events."""

from __future__ import annotations

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

    def record_event(
        self,
        item_id: int,
        event_type: ItemEventType,
        source_track_id: int | None = None,
        detector_confidence: float | None = None,
        source_region: str | None = None,
        destination_region: str | None = None,
        object_image_path: str | None = None,
        context_image_path: str | None = None,
        video_clip_path: str | None = None,
        notes: str | None = None,
    ) -> ItemEvent:
        """Record one meaningful event against a permanent item identity."""
        event = ItemEvent(
            item_id=item_id,
            event_type=event_type,
            source_track_id=source_track_id,
            detector_confidence=detector_confidence,
            source_region=source_region,
            destination_region=destination_region,
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
    ) -> ItemEvent:
        """Mark an item as removed and record its removal event atomically."""
        with self.session_factory() as session:
            item = session.get(Item, item_id)
            if item is None:
                raise ValueError(f"Item {item_id} does not exist")

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
    ) -> ItemEvent:
        """Mark an item as occluded and record the status change atomically."""
        with self.session_factory() as session:
            item = session.get(Item, item_id)
            if item is None:
                raise ValueError(f"Item {item_id} does not exist")

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
        """Mark an item present, recording an event only when its status changes."""
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

    def record_movement(
        self,
        item_id: int,
        source_region: str,
        destination_region: str,
        source_track_id: int | None = None,
        detector_confidence: float | None = None,
        object_image_path: str | None = None,
        context_image_path: str | None = None,
        video_clip_path: str | None = None,
        notes: str | None = None,
    ) -> ItemEvent:
        """Record a visible item moving between regions without changing its status."""
        with self.session_factory() as session:
            item = session.get(Item, item_id)
            if item is None:
                raise ValueError(f"Item {item_id} does not exist")

            item.last_seen_at = utc_now()
            event = ItemEvent(
                item=item,
                event_type=ItemEventType.MOVED,
                source_track_id=source_track_id,
                detector_confidence=detector_confidence,
                source_region=source_region,
                destination_region=destination_region,
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
