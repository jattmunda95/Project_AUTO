"""SQLite persistence for permanent items, lifecycle events, and references.

Implemented subfunctions:
- Own database sessions, schema creation, and atomic item/event operations.
- Validate and normalize embeddings; save_item_embedding saves a reference and prototype.
- update_item_prototype rebuilds the normalized mean, or clears it without references.
- load_reid_gallery filters by model/status and returns grouped float32 arrays in memory.
- count_item_embeddings returns the compatible reference count for a permanent item.
- add_reference_if_needed serializes count/check/save in one transaction against the
  caller's configured target, keeping prototype updates atomic with accepted writes.

save_item_embedding does NOT enforce a reference-count limit; add_reference_if_needed
does. The app owns visibility, identity resolution, timing, and gallery refresh. This
module stores crop paths, not image bytes, and never performs SAM or embedding inference.
"""

from __future__ import annotations

from datetime import datetime
from math import fsum, hypot, isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, selectinload, sessionmaker

from project_auto.memory.models import Base, Item, ItemEmbedding, ItemEvent, ItemEventType, ItemStatus, utc_now


if TYPE_CHECKING:
    from project_auto.memory.reid import GalleryEntry


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

    @staticmethod
    def _normalized_reference(embedding: list[float] | NDArray) -> list[float]:
        """Validate a numeric vector and return an independent unit-length list."""
        vector = np.asarray(embedding)
        if vector.ndim != 1 or vector.size == 0 or vector.dtype.kind not in "fiu":
            raise ValueError("embedding must be a nonempty numeric vector")
        if not np.isfinite(vector).all():
            raise ValueError("embedding must contain finite numbers")
        # Scale first to avoid overflow/underflow when calculating the norm.
        values = vector.astype(np.float64)
        scale = float(np.max(np.abs(values)))
        if not isfinite(scale) or scale == 0.0:
            raise ValueError("embedding must have finite, nonzero length")
        values = values / scale
        return (values / hypot(*values)).tolist()

    @classmethod
    def _reference_prototype(cls, references: list[ItemEmbedding]) -> list[float] | None:
        """Calculate without committing so reference and prototype writes stay atomic."""
        if not references:
            return None
        if len({reference.model_name for reference in references}) != 1:
            raise ValueError("Cannot average embeddings from different models")
        vectors = [cls._normalized_reference(reference.embedding) for reference in references]
        if len({len(vector) for vector in vectors}) != 1:
            raise ValueError("Embeddings must have matching dimensions")
        count = len(vectors)
        mean = [fsum(value / count for value in column) for column in zip(*vectors)]
        if hypot(*mean) <= np.finfo(np.float64).eps * len(mean):
            raise ValueError("Mean embedding cancels to zero within floating-point precision")
        return cls._normalized_reference(mean)

    def update_item_prototype(self, item_id: int) -> list[float] | None:
        """Rebuild a prototype from unit-normalized references; clear it when empty."""
        with self.session_factory() as session:
            item = session.get(Item, item_id)
            if item is None:
                raise ValueError(f"Item {item_id} does not exist")
            prototype = self._reference_prototype(item.embeddings)
            item.item_prototype = prototype
            session.commit()
        return prototype

    def save_item_embedding(
        self,
        item_id: int,
        embedding: list[float] | NDArray,
        model_name: str,
        object_image_path: Path | None = None,
    ) -> ItemEmbedding:
        """Save one normalized reference and its item's prototype atomically.

        The caller owns capture quality, scheduling, and image-file creation. Model
        name must identify the same embedding space; identical preprocessing is
        also required but is not yet recorded in the schema. One model per item
        is supported because Item currently has only one prototype column.
        """
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must not be empty")
        vector = self._normalized_reference(embedding)
        with self.session_factory() as session:
            item = session.get(Item, item_id)
            if item is None:
                raise ValueError(f"Item {item_id} does not exist")
            # Reject dimensions inconsistent with this model anywhere in the gallery.
            existing = session.scalars(
                select(ItemEmbedding).where(ItemEmbedding.model_name == model_name)
            )
            for reference in existing:
                if len(self._normalized_reference(reference.embedding)) != len(vector):
                    raise ValueError("Embeddings for the same model must have matching dimensions")
            reference = ItemEmbedding(
                model_name=model_name,
                embedding=vector,
                object_image_path=str(object_image_path) if object_image_path is not None else None,
            )
            item.embeddings.append(reference)
            item.item_prototype = self._reference_prototype(item.embeddings)
            session.commit()
        return reference

    def count_item_embeddings(self, item_id: int, model_name: str) -> int:
        """Return the number of compatible references saved for a permanent item."""
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must not be empty")
        statement = select(func.count(ItemEmbedding.id)).where(
            ItemEmbedding.item_id == item_id,
            ItemEmbedding.model_name == model_name,
        )
        with self.session_factory() as session:
            return int(session.scalar(statement) or 0)

    def add_reference_if_needed(
        self,
        item_id: int,
        embedding: list[float] | NDArray,
        model_name: str,
        target_count: int,
        object_image_path: Path | None = None,
    ) -> tuple[bool, int]:
        """Save one reference only while below target_count; update the prototype atomically.

        Serializes the count check and the save inside one transaction so concurrent
        callers cannot both push a permanent item's reference count past its target.
        Returns whether a reference was saved and the item's resulting reference count.
        """
        if target_count <= 0:
            raise ValueError("target_count must be positive")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must not be empty")
        vector = self._normalized_reference(embedding)
        with self.session_factory() as session:
            item = session.get(Item, item_id)
            if item is None:
                raise ValueError(f"Item {item_id} does not exist")
            existing = [
                reference for reference in item.embeddings if reference.model_name == model_name
            ]
            if len(existing) >= target_count:
                return False, len(existing)
            for reference in existing:
                if len(self._normalized_reference(reference.embedding)) != len(vector):
                    raise ValueError(
                        "Embeddings for the same model must have matching dimensions"
                    )
            reference = ItemEmbedding(
                model_name=model_name,
                embedding=vector,
                object_image_path=str(object_image_path)
                if object_image_path is not None
                else None,
            )
            item.embeddings.append(reference)
            item.item_prototype = self._reference_prototype(item.embeddings)
            session.commit()
            return True, len(existing) + 1

    def load_reid_gallery(
        self, model_name: str, statuses: tuple[ItemStatus, ...] | None = None
    ) -> list[GalleryEntry]:
        """Load an independent in-memory snapshot of normalized reference arrays.

        None includes all item statuses; an empty tuple includes none. Entries
        are ordered by permanent item ID, references by reference ID. Items with
        no references for this model are omitted. Each entry also carries its
        item's stored prototype (already unit-normalized), or None if unset, for
        prototype-shortlisted matching. No similarity search occurs here. Reuse
        the returned list and explicitly reload after relevant writes.
        Importing GalleryEntry needs the optional reid libraries, but loads no weights.
        """
        from project_auto.memory.reid import GalleryEntry

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must not be empty")
        if statuses is not None and any(not isinstance(status, ItemStatus) for status in statuses):
            raise ValueError("statuses must contain ItemStatus values")
        statement = (
            select(ItemEmbedding, Item.item_prototype)
            .join(Item, Item.id == ItemEmbedding.item_id)
            .where(ItemEmbedding.model_name == model_name)
            .order_by(ItemEmbedding.item_id, ItemEmbedding.id)
        )
        if statuses is not None:
            statement = statement.where(Item.status.in_(statuses))
        grouped: dict[int, list[list[float]]] = {}
        prototypes: dict[int, list[float] | None] = {}
        dimension: int | None = None
        with self.session_factory() as session:
            for reference, prototype in session.execute(statement):
                vector = self._normalized_reference(reference.embedding)
                if dimension is not None and len(vector) != dimension:
                    raise ValueError("Gallery embeddings must have matching dimensions")
                dimension = len(vector)
                grouped.setdefault(reference.item_id, []).append(vector)
                prototypes[reference.item_id] = prototype
        return [
            GalleryEntry(
                item_id,
                np.array(vectors, dtype=np.float32),
                np.array(prototypes[item_id], dtype=np.float32)
                if prototypes[item_id] is not None
                else None,
            )
            for item_id, vectors in grouped.items()
        ]

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
