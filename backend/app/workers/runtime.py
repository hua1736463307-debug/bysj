"""Shared worker loop with independent role liveness."""
import logging
import os
import socket
import time
import uuid
from app.core.config import get_settings
from app.db.session import get_session_factory
from app.domains import job_service
from app.workers.dispatcher import run_claimed_job
from app.workers.liveness import WorkerHeartbeat


def run_worker(role: str, job_types):
    settings = get_settings()
    logging.basicConfig(level="INFO")
    logger = logging.getLogger(f"bysj.worker.{role}")
    # Fit the existing JobTask lease_owner String(64), unique across restarts.
    worker_id = f"{role}-{socket.gethostname()[:16]}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    with WorkerHeartbeat(role, worker_id):
        while True:
            try:
                with get_session_factory()() as db:
                    job_service.mark_stale_interrupted(db, older_than_seconds=settings.job_lease_seconds)
                    job = job_service.claim_job(db, job_types=job_types, worker_id=worker_id,
                                                lease_seconds=settings.job_lease_seconds)
                    if job:
                        run_claimed_job(db, job, worker_id=worker_id, settings=settings)
            except Exception:
                logger.exception("worker_iteration_failed")
            time.sleep(settings.worker_poll_seconds)
