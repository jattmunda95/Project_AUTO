"""Job/result contracts and per-track state for asynchronous identity resolution.

Subfunctions:
- IdentificationState names where one temporary track stands in the async pipeline.
- IdentificationJob/IdentificationResult are immutable messages crossing the thread
  boundary between the main loop and the background identification worker.
- ReferenceManager owns per-track state and cooldowns so at most one identification
  or capture job is ever outstanding for a given track, and a bad result defers
  retry instead of resubmitting immediately.

No SAM/DINO/store calls happen here; this module is pure bookkeeping shared by
project_auto.events.coordinator (submission/result application, main thread) and
project_auto.events.identification_worker (job execution, background thread).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

from project_auto.perception.detector import Detection
from project_auto.perception.scene_processor import PreparedReference

JobKind = Literal["resolve", "capture"]
ResultStatus = Literal["new", "existing", "pending", "captured", "skipped", "error"]


class IdentificationState(str, Enum):
    """Lifecycle of one temporary track's identification/reference work."""

    UNIDENTIFIED = "unidentified"
    QUEUED = "queued"
    PROCESSING = "processing"
    IDENTIFIED = "identified"
    WAITING_FOR_BETTER_VIEW = "waiting_for_better_view"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class IdentificationJob:
    """One unit of background work; carries only what the worker needs.

    A "resolve" job requires frame and box (identity is unknown yet). A "capture"
    job requires item_id and either precomputed_reference (skip re-segmenting a
    reference already computed by a resolve job) or frame and box (segment fresh).
    """

    kind: JobKind
    track_id: int
    item_id: int | None = None
    frame: NDArray[np.uint8] | None = None
    box: tuple[int, int, int, int] | None = None
    precomputed_reference: PreparedReference | None = None
    detection: Detection | None = None


@dataclass(frozen=True, slots=True)
class IdentificationResult:
    """One completed job's outcome; the main thread applies all side effects."""

    kind: JobKind
    track_id: int
    status: ResultStatus
    item_id: int | None = None
    similarity: float = 0.0
    reference: PreparedReference | None = None
    failure_reason: str | None = None


_OUTSTANDING = (IdentificationState.QUEUED, IdentificationState.PROCESSING)
_COOLING_DOWN = (IdentificationState.WAITING_FOR_BETTER_VIEW, IdentificationState.DEFERRED)


@dataclass(slots=True)
class ReferenceManager:
    """Track per-track identification state so jobs are never duplicated.

    Invariant: at most one outstanding (queued or processing) job per track_id.
    A DEFERRED/WAITING_FOR_BETTER_VIEW track only becomes eligible again after
    its cooldown elapses; nothing here retries on its own initiative.
    """

    clock: Callable[[], float] = field(default=monotonic, repr=False)
    _states: dict[int, IdentificationState] = field(default_factory=dict, init=False, repr=False)
    _eligible_at: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _reasons: dict[int, str] = field(default_factory=dict, init=False, repr=False)

    def state(self, track_id: int) -> IdentificationState:
        """Return one track's current identification state."""
        return self._states.get(track_id, IdentificationState.UNIDENTIFIED)

    def reason(self, track_id: int) -> str | None:
        """Return the recorded reason a track was deferred, if any."""
        return self._reasons.get(track_id)

    def can_submit(self, track_id: int, now: float | None = None) -> bool:
        """Return whether a new job may be submitted for this track right now."""
        state = self.state(track_id)
        if state in _OUTSTANDING:
            return False
        if state in _COOLING_DOWN:
            now = self.clock() if now is None else now
            return now >= self._eligible_at.get(track_id, 0.0)
        return True

    def mark_queued(self, track_id: int) -> None:
        """Record that a job for this track was accepted onto the queue."""
        self._states[track_id] = IdentificationState.QUEUED

    def mark_identified(self, track_id: int) -> None:
        """Record that this track now has a resolved permanent identity."""
        self._states[track_id] = IdentificationState.IDENTIFIED
        self._eligible_at.pop(track_id, None)
        self._reasons.pop(track_id, None)

    def mark_deferred(self, track_id: int, now: float, cooldown_seconds: float, reason: str) -> None:
        """Defer retry after a failed attempt (e.g. a bad mask); ends this attempt."""
        self._states[track_id] = IdentificationState.DEFERRED
        self._eligible_at[track_id] = now + max(0.0, cooldown_seconds)
        self._reasons[track_id] = reason

    def mark_queue_full(self, track_id: int, now: float, retry_seconds: float) -> None:
        """Record a skipped submission because the bounded job queue was full."""
        self._states[track_id] = IdentificationState.WAITING_FOR_BETTER_VIEW
        self._eligible_at[track_id] = now + max(0.0, retry_seconds)
        self._reasons[track_id] = "queue_full"

    def forget(self, track_id: int) -> None:
        """Drop all state for a track that retired or was reused."""
        self._states.pop(track_id, None)
        self._eligible_at.pop(track_id, None)
        self._reasons.pop(track_id, None)

    def pending_track_ids(self) -> list[int]:
        """Return tracks currently cooling down, eligible for a retry check."""
        return [track_id for track_id, state in self._states.items() if state in _COOLING_DOWN]
