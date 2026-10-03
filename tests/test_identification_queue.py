"""Priority, backpressure, fairness, and shutdown of the actual worker queue."""

from queue import Empty, Full

import pytest

from project_auto.events.identification import IdentificationJob
from project_auto.events.identification_queue import IdentificationQueue


def job(number: int, priority: bool = False) -> IdentificationJob:
    return IdentificationJob("resolve", number, job_id=number, priority=priority)


def test_priority_has_reserved_capacity_and_runs_before_normal():
    jobs = IdentificationQueue(2, 2)
    jobs.put_nowait(job(1))
    jobs.put_nowait(job(2))
    with pytest.raises(Full):
        jobs.put_nowait(job(3))
    jobs.put_nowait(job(4, True))
    jobs.put_nowait(job(5, True))
    with pytest.raises(Full):
        jobs.put_nowait(job(6, True))
    assert jobs.get().job_id == 4
    assert jobs.get().job_id == 5
    # A continuous stream of urgent jobs cannot starve regular work.
    jobs.put_nowait(job(6, True))
    assert jobs.get().job_id == 1
    assert jobs.get().job_id == 6
    assert jobs.get().job_id == 2


def test_promotion_reuses_one_job_and_frees_normal_slot():
    jobs = IdentificationQueue(2, 1)
    jobs.put_nowait(job(1))
    jobs.put_nowait(job(2))
    assert jobs.promote(2)
    assert jobs.promote(2)
    jobs.put_nowait(job(3))
    assert [jobs.get().job_id for _ in range(3)] == [2, 1, 3]
    assert not jobs.promote(2)


def test_full_priority_queue_does_not_drop_unpromoted_job():
    jobs = IdentificationQueue(1, 1)
    jobs.put_nowait(job(1))
    jobs.put_nowait(job(2, True))
    assert not jobs.promote(1)
    assert jobs.get().job_id == 2
    assert jobs.get().job_id == 1


def test_close_discards_pending_work_and_unblocks_get():
    jobs = IdentificationQueue(1, 1)
    jobs.put_nowait(job(1))
    jobs.close()
    with pytest.raises(Empty):
        jobs.get()
    with pytest.raises(Full):
        jobs.put_nowait(job(2))
