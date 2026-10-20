"""Database-backed scheduler and heartbeat for the isolated training worker."""

import threading
import time
from typing import Optional

from sqlalchemy import text

from database import SessionLocal, engine, JobTask
from job_store import (
    claim_pending_training_job,
    fail_orphaned_training_jobs,
    finalize_unclaimed_training_cancellations,
    is_cancel_requested,
    requeue_gpu_job,
)
from training_worker import (
    get_active_training_tasks,
    poll_training_processes,
    start_training_process,
    stop_training_process,
    terminate_all_processes_for_worker_shutdown,
)
from training_worker_health import write_worker_heartbeat
from gpu_runtime import gpu_runtime_is_ready, probe_gpu


WORKER_HEARTBEAT_INTERVAL_SECONDS = 5.0
WORKER_POLL_INTERVAL_SECONDS = 2.0
WORKER_ADVISORY_LOCK_KEY = 4924218313798924622


class TrainingWorkerService:
    def __init__(self):
        self.stop_event = threading.Event()
        self._lease_connection = None
        self._last_heartbeat_at = 0.0
        self._cancel_threads_started = set()

    def stop(self):
        self.stop_event.set()

    def _acquire_singleton_lease(self):
        connection = engine.connect()
        acquired = bool(
            connection.execute(
                text("select pg_try_advisory_lock(:lock_key)"),
                {"lock_key": WORKER_ADVISORY_LOCK_KEY},
            ).scalar()
        )
        if not acquired:
            connection.close()
            raise RuntimeError("another training worker already owns the scheduler lease")
        self._lease_connection = connection

    def _release_singleton_lease(self):
        if self._lease_connection is not None:
            try:
                self._lease_connection.execute(
                    text("select pg_advisory_unlock(:lock_key)"),
                    {"lock_key": WORKER_ADVISORY_LOCK_KEY},
                )
            finally:
                self._lease_connection.close()
                self._lease_connection = None

    def _publish_heartbeat(self, active_task_id: Optional[str], *, force: bool = False):
        now = time.monotonic()
        if force or now - self._last_heartbeat_at >= WORKER_HEARTBEAT_INTERVAL_SECONDS:
            write_worker_heartbeat(active_task_id)
            self._last_heartbeat_at = now

    def _reconcile_previous_worker(self):
        db = SessionLocal()
        try:
            fail_orphaned_training_jobs(db)
            waiting = [row.id for row in db.query(JobTask.id).filter(
                JobTask.type == "training", JobTask.status == "interrupted", JobTask.stage == "waiting_gpu"
            ).all()]
        finally:
            db.close()
        for task_id in waiting:
            requeue_gpu_job(task_id)

    def _claim_next(self, *, gpu_available=True):
        db = SessionLocal()
        try:
            finalize_unclaimed_training_cancellations(db)
            return claim_pending_training_job(db, gpu_available=gpu_available)
        finally:
            db.close()

    def run(self):
        self._acquire_singleton_lease()
        try:
            self._reconcile_previous_worker()
            self._publish_heartbeat(None, force=True)

            while not self.stop_event.is_set():
                poll_training_processes()
                db = SessionLocal()
                try:
                    finalize_unclaimed_training_cancellations(db)
                finally:
                    db.close()
                active_tasks = get_active_training_tasks()
                active_task_id = active_tasks[0] if active_tasks else None

                if active_task_id:
                    if (
                        is_cancel_requested(active_task_id)
                        and active_task_id not in self._cancel_threads_started
                    ):
                        stop_training_process(active_task_id, grace_seconds=20)
                        self._cancel_threads_started.add(active_task_id)
                else:
                    self._cancel_threads_started.clear()
                    job = self._claim_next(gpu_available=gpu_runtime_is_ready(probe_gpu()))
                    if job is not None:
                        input_data = dict(job.input_json or {})
                        active_task_id = job.id
                        start_training_process(
                            job.id,
                            dict(input_data.get("config") or {}),
                            str(input_data.get("dataset_path") or ""),
                        )

                self._publish_heartbeat(active_task_id)
                self.stop_event.wait(WORKER_POLL_INTERVAL_SECONDS)
        finally:
            terminate_all_processes_for_worker_shutdown()
            write_worker_heartbeat(None, status="stopped")
            self._release_singleton_lease()
