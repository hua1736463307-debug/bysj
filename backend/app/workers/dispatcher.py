"""作业执行器：worker 进程与测试内联共用同一处理器注册表。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.db.base import utcnow
from app.db.models import JobTask
from app.db.session import get_session_factory
from app.domains import job_service

logger = logging.getLogger("bysj.worker")

HANDLERS: dict[str, str] = {
    "dataset_import": "app.workers.dataset_worker:handle_dataset_import",
    "dataset_version": "app.workers.dataset_worker:handle_dataset_version",
    "annotation_export": "app.workers.dataset_worker:handle_annotation_export",
    "feedback_export": "app.workers.dataset_worker:handle_feedback_export",
    "training": "app.workers.trainer_worker:handle_training",
    "evaluation": "app.workers.trainer_worker:handle_evaluation",
    "inference": "app.workers.inference_worker:handle_inference",
}


@dataclass
class JobContext:
    db: Session
    job: JobTask
    settings: Settings
    worker_id: str

    def progress(self, value: float, stage: str | None = None, message: str | None = None) -> None:
        job_service.update_progress(
            self.db, self.job, worker_id=self.worker_id, progress=value, stage=stage, message=message
        )
        # 长训练期间必须续租，否则其它 worker 的心跳清扫会误判为 interrupted。
        job_service.heartbeat(
            self.db, self.job, worker_id=self.worker_id, lease_seconds=self.settings.job_lease_seconds
        )

    def event(self, event_type: str, message: str, payload: dict | None = None) -> None:
        job_service.append_event(self.db, self.job, event_type, message, payload)

    def check_cancelled(self) -> None:
        self.db.refresh(self.job)
        if self.job.cancel_requested or self.job.status == "cancelling":
            raise JobCancelled()

    def heartbeat(self) -> None:
        job_service.heartbeat(self.db, self.job, worker_id=self.worker_id, lease_seconds=self.settings.job_lease_seconds)


class JobCancelled(Exception):
    """执行器检测到取消请求。"""


def resolve_handler(job_type: str) -> Callable:
    target = HANDLERS.get(job_type)
    if not target:
        raise ApiError(422, "unsupported_job", f"未知任务类型: {job_type}")
    module_name, _, attr = target.partition(":")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attr)


def run_claimed_job(db: Session, job: JobTask, *, worker_id: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    context = JobContext(db=db, job=job, settings=settings, worker_id=worker_id)
    handler = resolve_handler(job.type)
    try:
        result = handler(context)
    except JobCancelled:
        job_service.finish_job(db, job, worker_id=worker_id, result=None, message="任务已取消")
        return "cancelled"
    except ApiError as exc:
        job_service.fail_job(
            db,
            job,
            worker_id=worker_id,
            code=exc.code,
            detail=exc.message,
            result={"error": exc.data} if exc.data else None,
        )
        return "failed"
    except Exception as exc:  # 记录类型，不吞 traceback 到响应
        logger.exception("job_failed job_id=%s type=%s", job.id, job.type)
        job_service.fail_job(
            db, job, worker_id=worker_id, code="internal_error", detail=f"{type(exc).__name__}: {exc}"
        )
        return "failed"
    job_service.finish_job(db, job, worker_id=worker_id, result=result or {}, message="任务完成")
    return "completed"


def execute_job_inline(job_id: str) -> str:
    """测试/开发用：在调用进程内执行已持久化的任务（作业仍是权威记录）。"""
    factory = get_session_factory()
    with factory() as db:
        job = db.get(JobTask, job_id)
        if job is None:
            return "missing"
        if job.status == "pending":
            now = utcnow()
            job.status = "running"
            job.stage = "starting"
            job.started_at = now
            job.heartbeat_at = now
            job.lease_owner = "inline"
            job.lease_expires_at = now
            db.commit()
        return run_claimed_job(db, job, worker_id="inline")
