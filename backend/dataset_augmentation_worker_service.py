"""Singleton CPU worker for persistent whole-dataset augmentation jobs."""

from __future__ import annotations

import os
import threading
from typing import Optional

from database import SessionLocal
from dataset_augmentation_jobs import execute_dataset_augmentation_job
from job_store import claim_pending_dataset_augmentation_job, reconcile_dataset_augmentation_jobs


POLL_INTERVAL_SECONDS = 2.0


class DatasetAugmentationWorkerService:
    def __init__(self):
        self.stop_event = threading.Event()
        self._active_task_id: Optional[str] = None

    def stop(self):
        self.stop_event.set()

    def _claim_next(self):
        db = SessionLocal()
        try:
            return claim_pending_dataset_augmentation_job(db, worker_pid=os.getpid())
        finally:
            db.close()

    def _reconcile_previous_worker(self):
        db = SessionLocal()
        try:
            reconcile_dataset_augmentation_jobs(db)
        finally:
            db.close()

    def run(self):
        self._reconcile_previous_worker()
        while not self.stop_event.is_set():
            job = self._claim_next()
            if job is None:
                self.stop_event.wait(POLL_INTERVAL_SECONDS)
                continue
            self._active_task_id = job.id
            try:
                execute_dataset_augmentation_job(job.id)
            except Exception:
                # The executor has already persisted the terminal failure/cancel state.
                pass
            finally:
                self._active_task_id = None


def run_dataset_augmentation_worker():
    service = DatasetAugmentationWorkerService()
    try:
        service.run()
    except KeyboardInterrupt:
        service.stop()
