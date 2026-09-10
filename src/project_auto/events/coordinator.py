"""Coordinate scene processing, identity resolution, and reference capture.

Subfunctions:
- Resolve ADD confirmations through scene_processor before any item is created.
- Retry PENDING (unusable crop/mask) identity requests on later visible frames.
- Cancel a pending request once its track retires; a reused track ID starts fresh.
- Route NEW to EventEngine.process_add, a REMOVED match to process_return, and a
  PRESENT/OCCLUDED match to associate_existing_item (no event).
- Capture spaced reference frames for resolved, visible items below the configured
  target count, reusing the identity-resolution embedding as the first capture.

Owns the in-memory ReID gallery and refreshes it after every accepted reference.
No tracker-stage inspection or direct database access; the tracker only supplies
signals and per-frame detections, and persistence goes through store/event_engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from project_auto.events.event_engine import EventEngine
from project_auto.memory.models import ItemStatus
from project_auto.memory.reid import GalleryEntry
from project_auto.memory.store import DatabaseStore
from project_auto.perception.detector import Detection
from project_auto.perception.scene_processor import PreparedReference, SceneProcessor
from project_auto.perception.tracker import TrackSignal, TrackSignalType


@dataclass(slots=True)
class IdentityCoordinator:
    """Own identity resolution and reference capture around the event engine."""

    scene_processor: SceneProcessor
    event_engine: EventEngine
    store: DatabaseStore
    reid_model_name: str
    reference_target_count: int
    capture_interval_seconds: float
    pending_retry_interval_seconds: float = 1.0
    clock: Callable[[], float] = field(default=monotonic, repr=False)
    _gallery: list[GalleryEntry] = field(default_factory=list, init=False, repr=False)
    _pending_track_ids: set[int] = field(default_factory=set, init=False, repr=False)
    _last_capture_at: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _last_pending_attempt_at: dict[int, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.refresh_gallery()

    def refresh_gallery(self) -> None:
        """Reload the in-memory gallery snapshot used for identity comparisons."""
        self._gallery = self.store.load_reid_gallery(self.reid_model_name, statuses=None)

    def handle_frame(
        self,
        frame: NDArray[np.uint8],
        signals: list[TrackSignal],
        detections_by_track_id: dict[int, Detection],
    ) -> None:
        """Resolve identity for one frame's signals, retries, and reference captures."""
        for signal in signals:
            if signal.signal_type is TrackSignalType.ADD:
                self._resolve(frame, signal.track_id, signal.detection)
            elif signal.signal_type is TrackSignalType.REMOVE:
                self._pending_track_ids.discard(signal.track_id)
                self._last_pending_attempt_at.pop(signal.track_id, None)
                # A track that retired while still PENDING never got a permanent item.
                if self.event_engine.item_id_for_track(signal.track_id) is not None:
                    self.event_engine.process_signal(signal)
            else:
                self.event_engine.process_signal(signal)

        now = self.clock()
        for track_id in list(self._pending_track_ids):
            detection = detections_by_track_id.get(track_id)
            if detection is None:
                continue
            last_attempt_at = self._last_pending_attempt_at.get(track_id)
            if (
                last_attempt_at is not None
                and now - last_attempt_at < self.pending_retry_interval_seconds
            ):
                continue
            self._last_pending_attempt_at[track_id] = now
            self._resolve(frame, track_id, detection)

        for track_id, detection in detections_by_track_id.items():
            item_id = self.event_engine.item_id_for_track(track_id)
            if item_id is not None:
                self._maybe_capture(frame, detection.box, item_id, now)

    def _resolve(self, frame: NDArray[np.uint8], track_id: int, detection: Detection) -> None:
        """Propose and dispatch one identity decision for a visible confirmed track."""
        decision = self.scene_processor.process(frame, detection.box, track_id, self._gallery)

        if decision.decision == "pending":
            self._pending_track_ids.add(track_id)
            self._last_pending_attempt_at[track_id] = self.clock()
            return
        self._pending_track_ids.discard(track_id)
        self._last_pending_attempt_at.pop(track_id, None)

        if decision.decision == "new":
            add_signal = TrackSignal(
                signal_type=TrackSignalType.ADD,
                track_id=track_id,
                detection=detection,
            )
            item, _ = self.event_engine.process_signal(add_signal)
            self._capture(item.id, decision.reference, self.clock())
            return

        item_id = decision.item_id
        if item_id is None:
            raise RuntimeError("An EXISTING identity decision requires a permanent item_id")
        if self.event_engine.is_item_claimed(item_id, excluding_track_id=track_id):
            # Another visible track already owns this item; keep retrying instead.
            self._pending_track_ids.add(track_id)
            return

        item = self.store.get_item(item_id)
        if item is None:
            raise RuntimeError(f"Matched permanent item {item_id} no longer exists")

        if item.status is ItemStatus.REMOVED:
            return_signal = TrackSignal(
                signal_type=TrackSignalType.RETURNED,
                track_id=track_id,
                detection=detection,
                item_id=item_id,
            )
            self.event_engine.process_return(return_signal)
        else:
            self.event_engine.associate_existing_item(track_id, item_id)

        self._capture(item_id, decision.reference, self.clock())

    def _maybe_capture(self, frame: NDArray[np.uint8], box: tuple[int, int, int, int], item_id: int, now: float) -> None:
        """Capture one spaced reference frame for a resolved item below its target."""
        if self.store.count_item_embeddings(item_id, self.reid_model_name) >= self.reference_target_count:
            return
        last_capture_at = self._last_capture_at.get(item_id)
        if last_capture_at is not None and now - last_capture_at < self.capture_interval_seconds:
            return

        reference = self.scene_processor.prepare_reference(frame, box)
        self._capture(item_id, reference, now)

    def _capture(self, item_id: int, reference: PreparedReference | None, now: float) -> None:
        """Save one reference if the item is still below its target; refresh the gallery."""
        if reference is None:
            return

        saved, _ = self.store.add_reference_if_needed(
            item_id=item_id,
            embedding=reference.embedding,
            model_name=self.reid_model_name,
            target_count=self.reference_target_count,
        )
        if saved:
            self._last_capture_at[item_id] = now
            self.refresh_gallery()
