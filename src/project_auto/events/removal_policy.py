"""Item presence and handoff geometry; no inference, clocks, or persistence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from math import hypot

from project_auto.perception.detector import Detection

Box = tuple[int, int, int, int]


def center_distance(a: Box, b: Box) -> float:
    return hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)


def box_iou(a: Box, b: Box) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True, slots=True)
class RemovalConfig:
    absence_seconds: float = 2.0
    reid_grace_seconds: float = 3.0
    min_iou: float = 0.5
    moving_distance_scale: float = 1.5
    min_area_ratio: float = 0.4
    max_candidates: int = 4
    max_inflight: int = 2
    settled_seconds: float = 1.0
    movement_tolerance_pixels: float = 5.0

    def __post_init__(self) -> None:
        if self.absence_seconds <= 0 or self.reid_grace_seconds < 0:
            raise ValueError("Removal timeout must be positive and grace nonnegative")
        if not 0 < self.min_iou <= 1 or not 0 < self.min_area_ratio <= 1:
            raise ValueError("Handoff overlap and area ratio must be in (0, 1]")
        if self.moving_distance_scale < 0 or self.max_candidates < 1 or self.max_inflight < 1:
            raise ValueError("Invalid handoff distance or capacity")
        if self.settled_seconds <= 0 or self.movement_tolerance_pixels < 0:
            raise ValueError("Invalid handoff settling settings")


@dataclass(slots=True)
class ItemPresence:
    item_id: int
    track_id: int
    detection: Detection
    placement_box: Box
    recent_boxes: deque[Box] = field(default_factory=lambda: deque(maxlen=4))
    missing_since: float | None = None
    missing_at: datetime | None = None
    retired: bool = False
    moving: bool = False
    candidates: set[int] = field(default_factory=set)
    attempted: set[int] = field(default_factory=set)
    relocation_started_at: datetime | None = None
    settle_box: Box | None = None
    settled_since: float | None = None


@dataclass(slots=True)
class RemovalPolicy:
    config: RemovalConfig = field(default_factory=RemovalConfig)
    items: dict[int, ItemPresence] = field(default_factory=dict)

    def bind(self, item_id: int, detection: Detection, placement_box: Box | None = None) -> None:
        if detection.track_id is None:
            raise ValueError("An item binding requires a track ID")
        entry = ItemPresence(item_id, detection.track_id, detection, placement_box or detection.box)
        entry.recent_boxes.append(detection.box)
        self.items[item_id] = entry

    def observe(
        self, detections: dict[int, Detection], retired: set[int], now: float, timestamp: datetime
    ) -> None:
        """Start absence on the first missing frame; retirement alone never emits an event."""
        for entry in self.items.values():
            entry.retired |= entry.track_id in retired
            detection = detections.get(entry.track_id) if not entry.retired else None
            if detection is None:
                if entry.missing_since is None:
                    entry.missing_since = now
                    entry.missing_at = timestamp
                entry.settled_since = None
                continue
            entry.detection = detection
            entry.recent_boxes.append(detection.box)
            entry.missing_since = None
            entry.missing_at = None
            entry.candidates.clear()
            entry.attempted.clear()

    def remember_candidates(self, detections: dict[int, Detection], bound_ids: set[int]) -> None:
        """Remember a candidate even if it subsequently moves beyond the original box."""
        for entry in self.items.values():
            if entry.missing_since is None:
                continue
            ranked = sorted(
                detections.items(),
                key=lambda pair: (-box_iou(entry.detection.box, pair[1].box), pair[0]),
            )
            for track_id, detection in ranked:
                if track_id in bound_ids or track_id == entry.track_id:
                    continue
                if len(entry.candidates) >= self.config.max_candidates:
                    break
                if self.is_candidate(entry, detection.box):
                    entry.candidates.add(track_id)

    def is_candidate(self, entry: ItemPresence, box: Box) -> bool:
        for previous in entry.recent_boxes:
            if box_iou(previous, box) >= self.config.min_iou:
                return True
            if not entry.moving:
                continue
            old_area = max(0, previous[2] - previous[0]) * max(0, previous[3] - previous[1])
            new_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            ratio = (
                min(old_area, new_area) / max(old_area, new_area) if max(old_area, new_area) else 0
            )
            scale = max(previous[2] - previous[0], previous[3] - previous[1])
            if ratio >= self.config.min_area_ratio and center_distance(previous, box) <= (
                self.config.moving_distance_scale * scale
            ):
                return True
        return False

    def expired(self, item_id: int, now: float, checking: bool) -> bool:
        entry = self.items[item_id]
        if entry.missing_since is None:
            return False
        limit = self.config.absence_seconds
        if checking:
            limit += self.config.reid_grace_seconds
        return now - entry.missing_since >= limit

    def handoff(self, item_id: int, detection: Detection, timestamp: datetime) -> None:
        entry = self.items[item_id]
        source = entry.placement_box
        started_at = entry.relocation_started_at or entry.missing_at or timestamp
        self.bind(item_id, detection, source)
        entry = self.items[item_id]
        entry.relocation_started_at = started_at
        entry.moving = True

    def settled(self, item_id: int, now: float) -> bool:
        entry = self.items[item_id]
        if entry.relocation_started_at is None or entry.missing_since is not None:
            return False
        box = entry.detection.box
        if entry.settle_box is None or center_distance(entry.settle_box, box) > (
            self.config.movement_tolerance_pixels
        ):
            entry.settle_box = box
            entry.settled_since = now
            return False
        if entry.settled_since is None:
            entry.settled_since = now
        return now - entry.settled_since >= self.config.settled_seconds
