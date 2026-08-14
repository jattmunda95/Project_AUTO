"""Persistent database models for permanent items and meaningful events."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import DateTime, Enum as SqlEnum, Float, ForeignKey, Index, Integer, String, Text
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
    REMOVED = "removed"
    RETURNED = "returned"
    STATUS_CHANGED = "status_changed"
    MOVED = "moved"
    PLACED = "placed"


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
    source_track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detector_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_region: Mapped[str | None] = mapped_column(String(200), nullable=True)
    destination_region: Mapped[str | None] = mapped_column(String(200), nullable=True)
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
