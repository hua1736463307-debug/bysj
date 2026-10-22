import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import pytest
from app.core.config import get_settings
from app.core.errors import ApiError
from app.db.base import utcnow
from app.db.models import JobTask, ResourceLease
from app.domains import job_service, lease_service
from app.workers import dispatcher
from app.workers.gpu import acquire, release


def new_job(sessions, worker="one"):
    with sessions() as db:
        job, _ = job_service.create_job(db, job_type="training", owner_id=1, owner_username="test", payload={})
        job = job_service.claim_job(db, job_types=["training"], worker_id=worker, lease_seconds=2)
        return job.id


def test_long_blocking_handler_renews_job_and_gpu(sessions, monkeypatch):
    job_id = new_job(sessions)
    settings = get_settings().model_copy(update={"job_lease_seconds": 2, "gpu_lease_seconds": 2})
    def handler(ctx):
        token = acquire(ctx)
        try:
            time.sleep(4.5)  # no progress callbacks, beyond twice the original TTL
            ctx.lock_for_write()
            return {"ok": True}
        finally:
            release(ctx, token)
    monkeypatch.setattr(dispatcher, "resolve_handler", lambda _: handler)
    def execute():
        with sessions() as db:
            return dispatcher.run_claimed_job(db, db.get(JobTask, job_id), worker_id="one", settings=settings)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(execute)
        time.sleep(3)
        with sessions() as db:
            job = db.get(JobTask, job_id)
            gpu = db.get(ResourceLease, "gpu:0")
            assert job.lease_expires_at > utcnow()
            assert gpu.expires_at > utcnow()
            assert lease_service.acquire_lease(db, "gpu:0", holder="other") is None
            assert job_service.mark_stale_interrupted(db, older_than_seconds=2) == 0
        assert future.result(timeout=10) == "completed"
    with sessions() as db:
        assert lease_service.acquire_lease(db, "gpu:0", holder="next") is not None


def test_old_attempt_cannot_write_even_with_same_worker(sessions):
    job_id = new_job(sessions)
    with sessions() as old, sessions() as replacement:
        stale = old.get(JobTask, job_id)
        row = replacement.get(JobTask, job_id)
        row.attempt += 1
        replacement.commit()
        assert not job_service.heartbeat(old, stale, worker_id="one", expected_attempt=1)
        with pytest.raises(ApiError):
            job_service.finish_job(old, stale, worker_id="one", expected_attempt=1)


def test_interrupted_cancel_is_terminal_and_idempotent(sessions):
    job_id = new_job(sessions)
    with sessions() as db:
        job = db.get(JobTask, job_id)
        job.status = "interrupted"
        db.commit()
        assert job_service.cancel_job(db, job).status == "cancelled"
        assert job_service.cancel_job(db, job).status == "cancelled"


def test_cancelling_stale_job_is_recovered(sessions):
    job_id = new_job(sessions)
    with sessions() as db:
        job = db.get(JobTask, job_id)
        job.status = "cancelling"
        job.heartbeat_at = utcnow() - timedelta(seconds=10)
        job.lease_expires_at = utcnow() - timedelta(seconds=10)
        db.commit()
        assert job_service.mark_stale_interrupted(db, older_than_seconds=2) == 1
        db.refresh(job)
        assert job.status == "interrupted"
