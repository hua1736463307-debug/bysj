from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from app.domains import job_service


def test_concurrent_idempotency(sessions):
    barrier = Barrier(2)
    def create():
        with sessions() as db:
            barrier.wait(timeout=10)
            job, created = job_service.create_job(db, job_type="training", owner_id=1,
                owner_username="test", payload={"seed": 42}, idempotency_key="same-request")
            return job.id, created
    with ThreadPoolExecutor(2) as pool:
        first, second = list(pool.map(lambda _: create(), range(2)))
    assert first[0] == second[0]
    assert sorted([first[1], second[1]]) == [False, True]


def test_concurrent_claim(sessions):
    with sessions() as db:
        original, _ = job_service.create_job(db, job_type="training", owner_id=1,
            owner_username="test", payload={})
    barrier = Barrier(2)
    def claim(index):
        with sessions() as db:
            barrier.wait(timeout=10)
            job = job_service.claim_job(db, job_types=["training"], worker_id=f"worker-{index}")
            return job.id if job else None
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(claim, range(2)))
    assert results.count(original.id) == 1
    assert results.count(None) == 1
