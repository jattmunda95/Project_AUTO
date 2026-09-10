"""Persistent database schema for permanent items, events, and references.

Subfunctions:
- Item defines permanent identity, status, timestamps, and nullable prototype storage.
- ItemEmbedding links individual JSON vectors and model/crop metadata to permanent IDs.
- ItemEvent records meaningful lifecycle changes and evidence paths.
- Define enum constraints, indexes, relationships, and cascading child deletion.

Models describe storage; store.py validates vectors and updates prototypes. They do not
schedule captures, calculate image embeddings, or enforce the planned reference count.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import (
    JSON,
    DateTime,
    Enum as SqlEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class ItemStatus(str, Enum):
    PRESENT = "present"
    OCCLUDED = "occluded"
    REMOVED = "removed"


class ItemEventType(str, Enum):
    ADDED = "added"
    RETURNED = "returned"
    REMOVED = "removed"
    MOVED = "moved"
    STATUS_CHANGED = "status_changed"


class Base(DeclarativeBase):
    pass


item_status_type = SqlEnum(
    ItemStatus,
    name="item_status",
    native_enum=False,
    create_constraint=True,
    validate_strings=True,
    values_callable=lambda enum_class: [member.value for member in enum_class],
)

item_event_type = SqlEnum(
    ItemEventType,
    name="item_event_type",
    native_enum=False,
    create_constraint=True,
    validate_strings=True,
    values_callable=lambda enum_class: [member.value for member in enum_class],
)


class Item(Base):
    """One permanent physical object known to Project AUTO."""

    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    class_name: Mapped[str] = mapped_column(String(100), index=True)
    display_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[ItemStatus] = mapped_column(
        item_status_type,
        default=ItemStatus.PRESENT,
        index=True,
    )
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        index=True,
    )
    identity_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Normalized mean of compatible reference embeddings; maintained by the store.
    # Replace the list when updating it; in-place JSON edits are not tracked.
    item_prototype: Mapped[list[float] | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    events: Mapped[list[ItemEvent]] = relationship(
        back_populates="item",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    embeddings: Mapped[list[ItemEmbedding]] = relationship(
        back_populates="item",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @property
    def is_present(self) -> bool:
        """Return whether the item is present or temporarily occluded."""
        return self.status in {ItemStatus.PRESENT, ItemStatus.OCCLUDED}

    def __repr__(self) -> str:
        status = self.status.value if self.status is not None else None
        return (
            f"Item(id={self.id!r}, class_name={self.class_name!r}, "
            f"display_name={self.display_name!r}, status={status!r})"
        )


class ItemEmbedding(Base):
    """One reference embedding for a permanent item, stored as a JSON vector.

    References must use the same model and preprocessing before comparison or
    averaging. Vector validation and prototype updates belong in the store layer.
    Replace the embedding list when updating it; in-place JSON edits are not tracked.
    """

    __tablename__ = "item_embeddings"
    __table_args__ = (Index("ix_item_embeddings_item_id_model_name", "item_id", "model_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"))
    model_name: Mapped[str] = mapped_column(String(200))
    embedding: Mapped[list[float]] = mapped_column(JSON(none_as_null=True))
    object_image_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    item: Mapped[Item] = relationship(back_populates="embeddings")


class ItemEvent(Base):
    """One meaningful state or location event for a permanent item."""

    __tablename__ = "item_events"
    __table_args__ = (Index("ix_item_events_item_id_occurred_at", "item_id", "occurred_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[int] = mapped_column(
        ForeignKey("items.id", ondelete="CASCADE"),
        index=True,
    )
    event_type: Mapped[ItemEventType] = mapped_column(item_event_type, index=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        index=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    source_track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detector_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_region: Mapped[str | None] = mapped_column(String(200), nullable=True)
    destination_region: Mapped[str | None] = mapped_column(String(200), nullable=True)
    source_box: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    destination_box: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    object_image_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    context_image_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    video_clip_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    item: Mapped[Item] = relationship(back_populates="events")

    def __repr__(self) -> str:
        return (
            f"ItemEvent(id={self.id!r}, item_id={self.item_id!r}, "
            f"event_type={self.event_type.value!r}, occurred_at={self.occurred_at!r})"
        )
