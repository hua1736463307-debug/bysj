"""Independent heartbeats using fresh sessions; never share handler sessions."""
import logging
import threading
from app.db.models import JobTask
from app.db.session import get_session_factory
from app.domains import job_service, lease_service

logger = logging.getLogger(__name__)


class JobHeartbeat:
    def __init__(self, job_id, attempt, worker_id, settings):
        self.job_id, self.attempt, self.worker_id, self.settings = job_id, attempt, worker_id, settings
        self.stop = threading.Event()
        self.lost = threading.Event()
        self.lock = threading.Lock()
        self.gpu = None
        self.thread = threading.Thread(target=self._run, name="job-heartbeat", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)

    def set_gpu(self, holder, token):
        with self.lock:
            self.gpu = (holder, token)

    def clear_gpu(self):
        with self.lock:
            self.gpu = None

    def _run(self):
        interval = max(0.1, min(self.settings.job_lease_seconds, self.settings.gpu_lease_seconds) / 3)
        while not self.stop.wait(interval):
            try:
                with get_session_factory()() as db:
                    job = db.get(JobTask, self.job_id)
                    if job is None or not job_service.heartbeat(db, job, worker_id=self.worker_id,
                            expected_attempt=self.attempt, lease_seconds=self.settings.job_lease_seconds):
                        self.lost.set()
                        return
                    with self.lock:
                        if self.gpu:
                            holder, token = self.gpu
                            if not lease_service.renew_lease(db, lease_service.GPU_RESOURCE,
                                    holder=holder, generation=token, job_id=self.job_id,
                                    ttl_seconds=self.settings.gpu_lease_seconds):
                                self.lost.set()
                                return
            except Exception:
                logger.exception("heartbeat_lost job_id=%s", self.job_id)
                self.lost.set()
                return


class WorkerHeartbeat:
    def __init__(self, role, worker_id):
        self.resource = f"worker:{role}:{worker_id}"
        self.worker_id = worker_id
        self.stop = threading.Event()
        self.token = None
        self.thread = threading.Thread(target=self._run, name="worker-heartbeat", daemon=True)

    def _tick(self):
        with get_session_factory()() as db:
            if self.token and lease_service.renew_lease(db, self.resource, holder=self.worker_id,
                    generation=self.token, ttl_seconds=30):
                return
            self.token = lease_service.acquire_lease(db, self.resource, holder=self.worker_id, ttl_seconds=30)

    def __enter__(self):
        self._tick()
        self.thread.start()
        return self

    def _run(self):
        while not self.stop.wait(8):
            try:
                self._tick()
            except Exception:
                logger.exception("worker_heartbeat_failed resource=%s", self.resource)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.token and not self.thread.is_alive():
            with get_session_factory()() as db:
                lease_service.release_lease(db, self.resource, holder=self.worker_id, generation=self.token)
