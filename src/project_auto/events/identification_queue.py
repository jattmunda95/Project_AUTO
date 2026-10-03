"""Bounded worker queue with reserved handoff capacity and fair priority service."""

from collections import deque
from dataclasses import replace
from queue import Empty, Full
from threading import Condition

from project_auto.events.identification import IdentificationJob


class IdentificationQueue:
    def __init__(self, normal_capacity: int, priority_capacity: int) -> None:
        if normal_capacity < 1 or priority_capacity < 1:
            raise ValueError("Worker queue capacities must be positive")
        self._normal: deque[IdentificationJob] = deque()
        self._priority: deque[IdentificationJob] = deque()
        self._normal_capacity = normal_capacity
        self._priority_capacity = priority_capacity
        self._priority_streak = 0
        self._condition = Condition()
        self._closed = False

    def put_nowait(self, job: IdentificationJob) -> None:
        with self._condition:
            target = self._priority if job.priority else self._normal
            capacity = self._priority_capacity if job.priority else self._normal_capacity
            if self._closed or len(target) >= capacity:
                raise Full
            target.append(job)
            self._condition.notify()

    def promote(self, job_id: int) -> bool:
        """Promote an already queued resolve without copying its frame or duplicating work."""
        with self._condition:
            if any(job.job_id == job_id for job in self._priority):
                return True
            if len(self._priority) >= self._priority_capacity:
                return False
            for job in self._normal:
                if job.job_id == job_id and job.kind == "resolve":
                    self._normal.remove(job)
                    self._priority.append(replace(job, priority=True))
                    return True
            return False  # Already running/completed; its result is still reusable.

    def get(self) -> IdentificationJob:
        with self._condition:
            self._condition.wait_for(lambda: self._closed or self._priority or self._normal)
            if self._closed:
                raise Empty
            if self._priority and (
                not self._normal or self._priority_streak < self._priority_capacity
            ):
                self._priority_streak += 1
                return self._priority.popleft()
            self._priority_streak = 0
            return self._normal.popleft()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._normal.clear()
            self._priority.clear()
            self._condition.notify_all()
