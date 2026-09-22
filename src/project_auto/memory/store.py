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

import functools
import sqlite3
import time
from datetime import datetime
from math import fsum, hypot, isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from sqlalchemy import create_engine, event, func, or_, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, selectinload, sessionmaker

from project_auto.memory.models import Base, Item, ItemEmbedding, ItemEvent, ItemEventType, ItemStatus, Region, utc_now
from project_auto.memory.regions import Box, box_area_fraction, polygon_area, resolve_region, validate_polygon


if TYPE_CHECKING:
    from project_auto.memory.reid import GalleryEntry


def _retry_on_locked(max_attempts: int = 8, base_delay: float = 0.5) -> Any:
    """Retry a whole write method on a transient SQLite lock (e.g. OneDrive syncing the file).

    Each attempt re-runs the entire method, so it only applies to methods whose body
    opens its own short-lived session and is safe to run again after a failed commit
    left nothing persisted.
    """

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except OperationalError as exc:
                    if "database is locked" not in str(exc).lower() or attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, 8)
            raise AssertionError("unreachable")

        return wrapper

    return decorator


class DatabaseStore:
    """Own the database engine and create short-lived sessions."""

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        database_url = f"sqlite:///{database_path.resolve().as_posix()}"

        self.engine: Engine = create_engine(database_url, connect_args={"timeout": 30})

        @event.listens_for(self.engine, "connect")
        def enable_sqlite_pragmas(
            database_connection: Any,
            connection_record: Any,
        ) -> None:
            del connection_record
            cursor = database_connection.cursor()
            # busy_timeout must be set before anything that can itself hit a lock
            # (journal_mode included), otherwise a transient lock on this pragma
            # raises immediately instead of retrying. OneDrive syncing the .db
            # file underneath the app is a known source of such transient locks.
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA foreign_keys=ON")
            for attempt in range(5):
                try:
                    cursor.execute("PRAGMA journal_mode=WAL")
                    break
                except sqlite3.OperationalError:
                    if attempt == 4:
                        break
                    time.sleep(1)
            cursor.close()

        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
        )

    def create_schema(self) -> None:
        """Create any missing Project AUTO database tables."""
        # Existing Project AUTO databases require manual migration or database recreation
        # before the newly added Item and ItemEvent columns will exist. create_all does
        # not alter existing tables; never delete or automatically migrate user data here.
        Base.metadata.create_all(self.engine)

    @_retry_on_locked()
    def create_region(self, name: str, polygon: list[list[int]]) -> Region:
        """Validate and persist a simple polygon and its precomputed area."""
        name = name.strip()
        if not name or len(name) > 200:
            raise ValueError("Region name must contain 1 to 200 characters")
        points = validate_polygon(polygon)
        region = Region(name=name, polygon=points, area_px=polygon_area(points))
        with self.session_factory() as session:
            session.add(region)
            session.commit()
        return region

    def get_region(self, region_id: int) -> Region | None:
        """Return one configured region by its canonical ID."""
        with self.session_factory() as session:
            return session.get(Region, region_id)

    def get_region_by_name(self, name: str) -> Region | None:
        """Return a region by its current unique name."""
        with self.session_factory() as session:
            return session.scalar(select(Region).where(Region.name == name.strip()))

    def list_regions(self) -> list[Region]:
        """Return configured regions in stable ID order."""
        with self.session_factory() as session:
            return list(session.scalars(select(Region).order_by(Region.id)))

    @_retry_on_locked()
    def delete_region(self, region_id: int) -> bool:
        """Delete geometry; SET NULL retains events and their historical names."""
        with self.session_factory() as session:
            region = session.get(Region, region_id)
            if region is None:
                return False
            session.delete(region)
            session.commit()
        return True

    @staticmethod
    def _resolve_box_region(session: Session, box: Box | None) -> Region | None:
        if box is None:
            return None
        with session.no_autoflush:
            regions = list(session.scalars(select(Region)))
        region_id = resolve_region(box, regions)
        return next((region for region in regions if region.id == region_id), None)

    def resolve_box_region(self, box: Box) -> Region | None:
        """Load region geometry and resolve a centroid without writing state."""
        with self.session_factory() as session:
            return self._resolve_box_region(session, box)

    def _set_event_regions(
        self, session: Session, event_record: ItemEvent,
        frame_size: tuple[int, int] | None,
    ) -> None:
        """Resolve both evidence boxes using the caller's transaction."""
        for side in ("source", "destination"):
            box = getattr(event_record, f"{side}_box")
            region = self._resolve_box_region(session, box)
            setattr(event_record, f"{side}_region_id", region.id if region else None)
            if box is not None:
                setattr(event_record, f"{side}_region", region.name if region else None)
            setattr(event_record, f"{side}_box_area_fraction", box_area_fraction(box, frame_size))

    def _set_event_spatial_state(
        self, session: Session, item: Item, event_record: ItemEvent,
        frame_size: tuple[int, int] | None,
    ) -> None:
        """Resolve evidence and update live location inside the caller's transaction."""
        self._set_event_regions(session, event_record, frame_size)
        if event_record.event_type is ItemEventType.REMOVED or not item.is_present:
            item.current_box = None
            item.current_region_id = None
        elif event_record.event_type in {
            ItemEventType.ADDED, ItemEventType.MOVED, ItemEventType.RETURNED
        }:
            item.last_seen_at = utc_now()
            item.current_box = event_record.destination_box
            item.current_region_id = event_record.destination_region_id

    def find_items(self, name: str) -> list[Item]:
        """Find exact display/class-name matches, including removed items."""
        with self.session_factory() as session:
            return list(session.scalars(select(Item).where(
                or_(Item.display_name == name.strip(), Item.class_name == name.strip())
            ).order_by(Item.id)))

    def get_item_current_region(self, item_id: int) -> Region | None:
        """Return current location, with conservative fallback for legacy items only.

        A current box outside every polygon is an explicit unknown region. For legacy
        rows with neither live field, inspect only the newest location event; never
        skip a removal or an unassigned destination to revive an older location.
        """
        with self.session_factory() as session:
            item = session.get(Item, item_id)
            if item is None or not item.is_present:
                return None
            if item.current_region_id is not None:
                return session.get(Region, item.current_region_id)
            if item.current_box is not None:
                return None
            latest = session.scalar(select(ItemEvent).where(
                ItemEvent.item_id == item_id,
                ItemEvent.event_type.in_([
                    ItemEventType.ADDED, ItemEventType.MOVED,
                    ItemEventType.REMOVED, ItemEventType.RETURNED,
                ]),
            ).order_by(ItemEvent.occurred_at.desc(), ItemEvent.id.desc()).limit(1))
            if latest is None or latest.event_type is ItemEventType.REMOVED:
                return None
            if latest.destination_region_id is None:
                return None
            return session.get(Region, latest.destination_region_id)

    def get_items_in_region(self, region_id: int) -> list[Item]:
        """Return current contents from live item state, never historical events."""
        with self.session_factory() as session:
            return list(session.scalars(select(Item).where(
                Item.current_region_id == region_id,
                Item.status.in_([ItemStatus.PRESENT, ItemStatus.OCCLUDED]),
            ).order_by(Item.id)))

    def get_region_recent_events(self, region_id: int, limit: int = 20) -> list[ItemEvent]:
        """Return recent events involving either region FK, newest first."""
        if limit < 0:
            raise ValueError("limit must be nonnegative")
        with self.session_factory() as session:
            return list(session.scalars(select(ItemEvent).where(or_(
                ItemEvent.source_region_id == region_id,
                ItemEvent.destination_region_id == region_id,
            )).options(selectinload(ItemEvent.item)).order_by(
                ItemEvent.occurred_at.desc(), ItemEvent.id.desc()
            ).limit(limit)))

    def get_region_last_activity(self, region_id: int) -> ItemEvent | None:
        """Return the newest meaningful event involving this region."""
        events = self.get_region_recent_events(region_id, limit=1)
        return events[0] if events else None

    def get_items_associated_with_region(self, region_id: int) -> list[Item]:
        """Return distinct items historically associated through event region IDs."""
        with self.session_factory() as session:
            return list(session.scalars(select(Item).join(ItemEvent).where(or_(
                ItemEvent.source_region_id == region_id,
                ItemEvent.destination_region_id == region_id,
            )).distinct().order_by(Item.id)))

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

    @_retry_on_locked()
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

    @staticmethod
    def _normalized_color_histogram(
        color_histogram: list[float] | NDArray | None,
    ) -> list[float] | None:
        """Validate and flatten an optional histogram for JSON storage."""
        if color_histogram is None:
            return None
        vector = np.asarray(color_histogram, dtype=np.float64).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("color_histogram must be a nonempty array of finite numbers")
        return vector.tolist()

    @_retry_on_locked()
    def save_item_embedding(
        self,
        item_id: int,
        embedding: list[float] | NDArray,
        model_name: str,
        object_image_path: Path | None = None,
        aspect_ratio: float | None = None,
        color_histogram: list[float] | NDArray | None = None,
    ) -> ItemEmbedding:
        """Save one normalized reference and its item's prototype atomically.

        The caller owns capture quality, scheduling, and image-file creation. Model
        name must identify the same embedding space; identical preprocessing is
        also required but is not yet recorded in the schema. One model per item
        is supported because Item currently has only one prototype column.
        aspect_ratio and color_histogram are optional descriptors captured
        alongside the embedding; a flattened histogram is reshaped by callers
        using descriptors.py's fixed bin layout.
        """
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must not be empty")
        vector = self._normalized_reference(embedding)
        histogram = self._normalized_color_histogram(color_histogram)
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
                aspect_ratio=aspect_ratio,
                color_histogram=histogram,
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

    @_retry_on_locked()
    def add_reference_if_needed(
        self,
        item_id: int,
        embedding: list[float] | NDArray,
        model_name: str,
        target_count: int,
        object_image_path: Path | None = None,
        aspect_ratio: float | None = None,
        color_histogram: list[float] | NDArray | None = None,
    ) -> tuple[bool, int]:
        """Save one reference only while below target_count; update the prototype atomically.

        Serializes the count check and the save inside one transaction so concurrent
        callers cannot both push a permanent item's reference count past its target.
        Returns whether a reference was saved and the item's resulting reference count.
        aspect_ratio and color_histogram are optional descriptors captured alongside
        the embedding; see save_item_embedding.
        """
        if target_count <= 0:
            raise ValueError("target_count must be positive")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must not be empty")
        vector = self._normalized_reference(embedding)
        histogram = self._normalized_color_histogram(color_histogram)
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
                aspect_ratio=aspect_ratio,
                color_histogram=histogram,
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
        prototype-shortlisted matching, plus per-reference aspect ratio and
        flattened color histogram arrays (NaN where a reference predates those
        descriptors) for the weighted color/size similarity terms. No similarity
        search occurs here. Reuse the returned list and explicitly reload after
        relevant writes. Importing GalleryEntry needs the optional reid
        libraries, but loads no weights.
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
        aspect_ratios: dict[int, list[float]] = {}
        color_histograms: dict[int, list[list[float] | None]] = {}
        prototypes: dict[int, list[float] | None] = {}
        dimension: int | None = None
        histogram_size: int | None = None
        with self.session_factory() as session:
            for reference, prototype in session.execute(statement):
                vector = self._normalized_reference(reference.embedding)
                if dimension is not None and len(vector) != dimension:
                    raise ValueError("Gallery embeddings must have matching dimensions")
                dimension = len(vector)
                grouped.setdefault(reference.item_id, []).append(vector)
                prototypes[reference.item_id] = prototype
                aspect_ratios.setdefault(reference.item_id, []).append(
                    reference.aspect_ratio if reference.aspect_ratio is not None else np.nan
                )
                if reference.color_histogram is not None:
                    histogram_size = len(reference.color_histogram)
                color_histograms.setdefault(reference.item_id, []).append(
                    reference.color_histogram
                )

        return [
            GalleryEntry(
                item_id,
                np.array(vectors, dtype=np.float32),
                np.array(prototypes[item_id], dtype=np.float32)
                if prototypes[item_id] is not None
                else None,
                np.array(aspect_ratios[item_id], dtype=np.float32),
                np.array(
                    [
                        histogram if histogram is not None else [np.nan] * (histogram_size or 0)
                        for histogram in color_histograms[item_id]
                    ],
                    dtype=np.float32,
                ),
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
        box: Box | None,
    ) -> list[int] | None:
        """Convert an optional bbox tuple into JSON-compatible coordinates."""
        if box is None:
            return None
        if len(box) != 4:
            raise ValueError("A bounding box must contain four coordinates")

        return list(box)

    @_retry_on_locked()
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

    @_retry_on_locked()
    def add_item_with_event(
        self,
        class_name: str,
        status: ItemStatus,
        source_track_id: int,
        detector_confidence: float,
        destination_box: Box | None = None,
        frame_size: tuple[int, int] | None = None,
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
                destination_box=self._serialize_box(destination_box),
            )
            self._set_event_spatial_state(session, item, added_event, frame_size)
            session.add(added_event)
            session.commit()

        return item, added_event

    @_retry_on_locked()
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
        frame_size: tuple[int, int] | None = None,
    ) -> ItemEvent:
        """Append historical evidence; use lifecycle methods to change live item state."""
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
            self._set_event_regions(session, event, frame_size)
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

    @_retry_on_locked()
    def mark_removed(
        self,
        item_id: int,
        source_track_id: int | None = None,
        detector_confidence: float | None = None,
        source_region: str | None = None,
        context_image_path: str | None = None,
        video_clip_path: str | None = None,
        notes: str | None = None,
        source_box: Box | None = None,
        frame_size: tuple[int, int] | None = None,
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
                source_box=self._serialize_box(source_box if source_box is not None else item.current_box),
                context_image_path=context_image_path,
                video_clip_path=video_clip_path,
                notes=notes,
            )
            self._set_event_spatial_state(session, item, event, frame_size)
            session.add(event)
            session.commit()

        return event

    @_retry_on_locked()
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

    @_retry_on_locked()
    def mark_present(
        self,
        item_id: int,
        source_track_id: int | None = None,
        detector_confidence: float | None = None,
        destination_region: str | None = None,
        object_image_path: str | None = None,
        context_image_path: str | None = None,
        notes: str | None = None,
        destination_box: Box | None = None,
        frame_size: tuple[int, int] | None = None,
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
                destination_box=self._serialize_box(destination_box),
                object_image_path=object_image_path,
                context_image_path=context_image_path,
                notes=notes,
            )
            self._set_event_spatial_state(session, item, event, frame_size)
            session.add(event)
            session.commit()

        return event

    @_retry_on_locked()
    def mark_returned(
        self,
        item_id: int,
        source_track_id: int | None = None,
        detector_confidence: float | None = None,
        destination_box: Box | None = None,
        frame_size: tuple[int, int] | None = None,
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
                destination_box=self._serialize_box(destination_box),
            )
            self._set_event_spatial_state(session, item, returned_event, frame_size)
            session.add(returned_event)
            session.commit()

        return returned_event

    @_retry_on_locked()
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
        frame_size: tuple[int, int] | None = None,
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
            self._set_event_spatial_state(session, item, event, frame_size)
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
