"""PostgreSQL-backed job state machine used by training and batch annotation."""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from database import ContinuousMonitor, JobEvent, JobTask, SessionLocal


ACTIVE_STATUSES = {"pending", "running", "cancelling", "interrupted"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
ALLOWED_TRANSITIONS = {
    "pending": {"running", "cancelling", "cancelled", "failed", "interrupted"},
    "running": {"cancelling", "completed", "failed", "cancelled", "interrupted"},
    "cancelling": {"cancelled", "failed", "interrupted"},
    "interrupted": {"pending", "running", "failed", "cancelled"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}


def _append_event_locked(
    db: Session,
    task_id: str,
    *,
    event_type: str,
    message: str,
    level: str = "info",
    payload: Optional[Dict[str, Any]] = None,
) -> JobEvent:
    last_sequence = db.query(func.max(JobEvent.sequence)).filter(JobEvent.task_id == task_id).scalar() or 0
    event = JobEvent(
        task_id=task_id,
        sequence=int(last_sequence) + 1,
        level=level,
        event_type=event_type,
        message=message or "",
        payload_json=payload or {},
    )
    db.add(event)
    return event


def create_job(
    db: Session,
    *,
    job_type: str,
    owner_username: str,
    owner_id: Optional[str] = None,
    input_data: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
    message: str = "任务已创建",
) -> JobTask:
    now = datetime.now()
    task = JobTask(
        id=task_id or str(uuid4()),
        type=job_type,
        status="pending",
        stage="queued",
        progress=0.0,
        message=message,
        owner_id=str(owner_id) if owner_id is not None else None,
        owner_username=owner_username or "unknown",
        input_json=input_data or {},
        result_json={},
        heartbeat_at=now,
        updated_at=now,
    )
    db.add(task)
    db.flush()
    _append_event_locked(db, task.id, event_type="created", message=message, payload={"status": "pending"})
    db.commit()
    db.refresh(task)
    return task


def update_job(
    db: Session,
    task_id: str,
    *,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    error_code: Optional[str] = None,
    error_detail: Optional[str] = None,
    worker_pid: Optional[int] = None,
    heartbeat: bool = True,
    event_type: str = "progress",
    event_payload: Optional[Dict[str, Any]] = None,
    level: str = "info",
) -> JobTask:
    task = db.query(JobTask).filter(JobTask.id == task_id).with_for_update().first()
    if not task:
        raise KeyError(f"job not found: {task_id}")

    now = datetime.now()
    if status and status != task.status:
        if status not in ALLOWED_TRANSITIONS.get(task.status, set()):
            raise ValueError(f"illegal job transition: {task.status} -> {status}")
        if status == "running" and not task.started_at:
            task.started_at = now
        if status in TERMINAL_STATUSES:
            task.finished_at = now
        task.status = status
    if stage is not None:
        task.stage = stage
    if progress is not None:
        task.progress = max(0.0, min(float(progress), 100.0))
    if message is not None:
        task.message = str(message)
    if result is not None:
        task.result_json = result
    if error_code is not None:
        task.error_code = error_code
    if error_detail is not None:
        task.error_detail = error_detail
    if worker_pid is not None:
        task.worker_pid = worker_pid
    if heartbeat:
        task.heartbeat_at = now
    task.updated_at = now

    payload = {
        "status": task.status,
        "stage": task.stage,
        "progress": float(task.progress or 0.0),
        **(event_payload or {}),
    }
    _append_event_locked(
        db,
        task.id,
        event_type=event_type,
        message=task.message,
        level=level,
        payload=payload,
    )
    db.commit()
    db.refresh(task)
    return task


def record_job_update(task_id: str, **kwargs) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        return update_job(db, task_id, **kwargs).to_dict()
    finally:
        db.close()


def append_job_event(
    db: Session,
    task_id: str,
    *,
    event_type: str,
    message: str,
    level: str = "info",
    payload: Optional[Dict[str, Any]] = None,
) -> JobEvent:
    task = db.query(JobTask).filter(JobTask.id == task_id).with_for_update().first()
    if not task:
        raise KeyError(f"job not found: {task_id}")
    event = _append_event_locked(
        db,
        task_id,
        event_type=event_type,
        message=message,
        level=level,
        payload=payload,
    )
    task.heartbeat_at = datetime.now()
    task.updated_at = datetime.now()
    db.commit()
    db.refresh(event)
    return event


def append_job_event_external(task_id: str, **kwargs) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        return append_job_event(db, task_id, **kwargs).to_dict()
    finally:
        db.close()


def request_job_cancel(db: Session, task_id: str) -> JobTask:
    task = db.query(JobTask).filter(JobTask.id == task_id).with_for_update().first()
    if not task:
        raise KeyError(f"job not found: {task_id}")
    if task.status in TERMINAL_STATUSES:
        db.rollback()
        return task
    if task.cancel_requested and task.status == "cancelling":
        db.rollback()
        return task
    task.cancel_requested = True
    previous = task.status
    if task.status in {"pending", "running"}:
        task.status = "cancelling"
    task.message = "正在取消任务..."
    task.updated_at = datetime.now()
    task.heartbeat_at = datetime.now()
    _append_event_locked(
        db,
        task_id,
        event_type="cancel_requested",
        message=task.message,
        payload={"previous_status": previous, "status": task.status},
    )
    db.commit()
    db.refresh(task)
    return task


def is_cancel_requested(task_id: str) -> bool:
    db = SessionLocal()
    try:
        task = db.query(JobTask.cancel_requested, JobTask.status).filter(JobTask.id == task_id).first()
        return bool(task and (task.cancel_requested or task.status == "cancelling"))
    finally:
        db.close()


def claim_pending_training_job(db: Session, *, gpu_available: bool = True) -> Optional[JobTask]:
    """Atomically claim the oldest eligible queued training task."""
    if db.bind and db.bind.dialect.name == "postgresql":
        db.execute(text("select pg_advisory_xact_lock(:lock_key)"), {"lock_key": 4924218313798924623})
    if db.query(ContinuousMonitor.id).filter(
        (ContinuousMonitor.desired_state == "running") | (ContinuousMonitor.status == "stopping")
    ).first():
        db.rollback()
        return None
    active_exists = db.query(JobTask.id).filter(
        JobTask.type.in_({"training", "behavior_analysis", "pose_training", "pose_evaluation"}),
        JobTask.status.in_({"running", "cancelling"}),
    ).first()
    if active_exists:
        db.rollback()
        return None
    candidates = (
        db.query(JobTask)
        .filter(
            JobTask.type == "training",
            JobTask.status == "pending",
            JobTask.cancel_requested.is_(False),
        )
        .order_by(JobTask.created_at.asc())
        .populate_existing()
        .with_for_update(skip_locked=True)
        .all()
    )
    task = next(
        (
            candidate
            for candidate in candidates
            if (gpu_available
                or str(((candidate.input_json or {}).get("config") or {}).get("device", "auto")).strip().lower() == "cpu")
            and float((candidate.input_json or {}).get("_gpu_retry_after", 0)) <= time.time()
        ),
        None,
    )
    if not task:
        db.rollback()
        return None

    now = datetime.now()
    task.status = "running"
    task.stage = "starting"
    task.message = "训练 Worker 已领取任务"
    task.started_at = task.started_at or now
    task.heartbeat_at = now
    task.updated_at = now
    _append_event_locked(
        db,
        task.id,
        event_type="worker_claimed",
        message=task.message,
        payload={"status": task.status, "stage": task.stage},
    )
    db.commit()
    db.refresh(task)
    return task


def wait_for_gpu_job(task_id: str, error: str) -> None:
    """Hold failed GPU work; it cannot be claimed until its owner has exited."""
    db = SessionLocal()
    try:
        task = db.query(JobTask).filter(JobTask.id == task_id).with_for_update().first()
        if task is None or task.status in TERMINAL_STATUSES:
            db.rollback()
            return
        update_job(db, task_id, status="interrupted" if not task.cancel_requested else "cancelled",
                   stage="waiting_gpu" if not task.cancel_requested else "cancelled",
                   message="当前无剩余运算力，GPU 恢复后重新启动本任务" if not task.cancel_requested else "任务已取消",
                   error_code="GPU_RUNTIME_UNAVAILABLE", error_detail=str(error)[-1000:],
                   event_type="waiting_gpu", level="warning")
    finally:
        db.close()


def requeue_gpu_job(task_id: str) -> bool:
    """Called only after the GPU task owner has stopped; preserve cancellation."""
    db = SessionLocal()
    try:
        task = db.query(JobTask).filter(JobTask.id == task_id).with_for_update().first()
        if task is None or task.status != "interrupted" or task.stage != "waiting_gpu":
            db.rollback()
            return False
        task.worker_pid = None
        data = dict(task.input_json or {})
        attempts = int(data.get("_gpu_retry_count", 0)) + 1
        data["_gpu_retry_count"] = attempts
        data["_gpu_retry_after"] = time.time() + min(300, 15 * 2 ** min(attempts - 1, 5))
        task.input_json = data
        task.error_code = None
        task.error_detail = None
        update_job(db, task_id, status="cancelled" if task.cancel_requested else "pending",
                   stage="cancelled" if task.cancel_requested else "waiting_gpu",
                   message="任务已取消" if task.cancel_requested else "当前无剩余运算力，GPU 恢复后重新启动本任务",
                   event_type="gpu_wait_requeued")
        return True
    finally:
        db.close()


def claim_pending_behavior_job(db: Session, *, worker_pid: Optional[int] = None) -> Optional[JobTask]:
    """Atomically claim one queued offline behavior-analysis/temporal task.

    A database advisory lock plus the active-row check keeps GPU concurrency at
    one even if two worker containers are started accidentally.
    """
    if db.bind and db.bind.dialect.name == "postgresql":
        db.execute(text("select pg_advisory_xact_lock(:lock_key)"), {"lock_key": 4924218313798924623})
    if db.query(ContinuousMonitor.id).filter(
        (ContinuousMonitor.desired_state == "running") | (ContinuousMonitor.status == "stopping")
    ).first():
        db.rollback()
        return None
    active_exists = db.query(JobTask.id).filter(
        JobTask.type.in_({"training", "behavior_analysis", "behavior_temporal_training", "pose_training", "pose_evaluation"}),
        JobTask.status.in_({"running", "cancelling"}),
    ).first()
    if active_exists:
        db.rollback()
        return None
    task = (
        db.query(JobTask)
        .filter(
            JobTask.type.in_({"behavior_analysis", "behavior_temporal_training"}),
            JobTask.status == "pending",
            JobTask.cancel_requested.is_(False),
        )
        .order_by(JobTask.created_at.asc())
        .populate_existing()
        .with_for_update(skip_locked=True)
        .all()
    )
    task = next((item for item in task if float((item.input_json or {}).get("_gpu_retry_after", 0)) <= time.time()), None)
    if not task:
        db.rollback()
        return None

    now = datetime.now()
    task.status = "running"
    task.stage = "starting"
    task.message = (
        "行为时序训练 Worker 已领取任务"
        if task.type == "behavior_temporal_training"
        else "行为分析 Worker 已领取任务"
    )
    task.started_at = task.started_at or now
    task.worker_pid = worker_pid
    task.heartbeat_at = now
    task.updated_at = now
    _append_event_locked(
        db,
        task.id,
        event_type="worker_claimed",
        message=task.message,
        payload={"status": task.status, "stage": task.stage, "worker_pid": worker_pid},
    )
    db.commit()
    db.refresh(task)
    return task


def claim_pending_dataset_augmentation_job(db: Session, *, worker_pid: Optional[int] = None) -> Optional[JobTask]:
    """Atomically claim one queued CPU dataset-augmentation task."""
    if db.bind and db.bind.dialect.name == "postgresql":
        db.execute(text("select pg_advisory_xact_lock(:lock_key)"), {"lock_key": 4924218313798924633})
    task = (
        db.query(JobTask)
        .filter(
            JobTask.type == "dataset_augmentation",
            JobTask.status == "pending",
            JobTask.cancel_requested.is_(False),
        )
        .order_by(JobTask.created_at.asc())
        .with_for_update(skip_locked=True)
        .first()
    )
    if not task:
        db.rollback()
        return None

    now = datetime.now()
    task.status = "running"
    task.stage = "starting"
    task.message = "数据增强 Worker 已领取任务"
    task.started_at = task.started_at or now
    task.worker_pid = worker_pid
    task.heartbeat_at = now
    task.updated_at = now
    _append_event_locked(
        db,
        task.id,
        event_type="worker_claimed",
        message=task.message,
        payload={"status": task.status, "stage": task.stage, "worker_pid": worker_pid},
    )
    db.commit()
    db.refresh(task)
    return task


def reconcile_dataset_augmentation_jobs(db: Session) -> Dict[str, Iterable[str]]:
    """Requeue interrupted augmentation jobs after a worker restart."""
    result = {"requeued": [], "cancelled": []}
    tasks = (
        db.query(JobTask)
        .filter(
            JobTask.type == "dataset_augmentation",
            JobTask.status.in_({"running", "cancelling", "interrupted"}),
        )
        .with_for_update(skip_locked=True)
        .all()
    )
    now = datetime.now()
    for task in tasks:
        if task.cancel_requested or task.status == "cancelling":
            task.status = "cancelled"
            task.stage = "cancelled"
            task.message = "数据增强任务已取消"
            task.finished_at = now
            event_type = "cancelled"
            result["cancelled"].append(task.id)
        else:
            task.status = "pending"
            task.stage = "queued"
            task.progress = 0.0
            task.message = "数据增强 Worker 重启，任务已重新排队"
            task.started_at = None
            task.finished_at = None
            task.error_code = None
            task.error_detail = None
            event_type = "requeued_after_restart"
            result["requeued"].append(task.id)
        task.worker_pid = None
        task.heartbeat_at = now
        task.updated_at = now
        _append_event_locked(
            db,
            task.id,
            event_type=event_type,
            message=task.message,
            level="warning" if event_type == "requeued_after_restart" else "info",
            payload={"status": task.status, "stage": task.stage},
        )
    db.commit()
    return result


def reconcile_behavior_jobs(db: Session) -> Dict[str, Iterable[str]]:
    """Requeue interrupted behavior jobs and finalize queued cancellations."""
    result = {"requeued": [], "cancelled": []}
    tasks = (
        db.query(JobTask)
        .filter(
            JobTask.type.in_({"behavior_analysis", "behavior_temporal_training"}),
            JobTask.status.in_({"running", "cancelling", "interrupted"}),
        )
        .with_for_update(skip_locked=True)
        .all()
    )
    now = datetime.now()
    for task in tasks:
        if task.cancel_requested or task.status == "cancelling":
            task.status = "cancelled"
            task.stage = "cancelled"
            task.message = (
                "行为时序训练任务已取消"
                if task.type == "behavior_temporal_training"
                else "行为分析任务已取消"
            )
            task.finished_at = now
            event_type = "cancelled"
            result["cancelled"].append(task.id)
        else:
            task.status = "pending"
            task.stage = "queued"
            task.progress = 0.0
            task.message = (
                "行为时序训练 Worker 重启，任务已重新排队"
                if task.type == "behavior_temporal_training"
                else "行为分析 Worker 重启，任务已重新排队"
            )
            task.started_at = None
            task.finished_at = None
            task.error_code = None
            task.error_detail = None
            event_type = "requeued_after_restart"
            result["requeued"].append(task.id)
        task.worker_pid = None
        task.heartbeat_at = now
        task.updated_at = now
        _append_event_locked(
            db,
            task.id,
            event_type=event_type,
            message=task.message,
            level="warning" if event_type == "requeued_after_restart" else "info",
            payload={"status": task.status, "stage": task.stage},
        )
    db.commit()
    return result


def finalize_unclaimed_training_cancellations(db: Session) -> Iterable[str]:
    """Finish queued cancellations that never acquired a child process."""
    tasks = (
        db.query(JobTask)
        .filter(
            JobTask.type == "training",
            JobTask.status == "cancelling",
            JobTask.cancel_requested.is_(True),
            JobTask.worker_pid.is_(None),
        )
        .with_for_update(skip_locked=True)
        .all()
    )
    now = datetime.now()
    task_ids = []
    for task in tasks:
        task.status = "cancelled"
        task.stage = "cancelled"
        task.message = "训练任务已在启动前取消"
        task.finished_at = now
        task.heartbeat_at = now
        task.updated_at = now
        _append_event_locked(
            db,
            task.id,
            event_type="cancelled",
            message=task.message,
            payload={"status": task.status},
        )
        task_ids.append(task.id)
    db.commit()
    return task_ids


def fail_orphaned_training_jobs(db: Session) -> Iterable[str]:
    """Mark jobs left by a previous training-worker process as worker_lost."""
    tasks = (
        db.query(JobTask)
        .filter(JobTask.type == "training", JobTask.status.in_({"running", "cancelling"}))
        .with_for_update(skip_locked=True)
        .all()
    )
    now = datetime.now()
    task_ids = []
    for task in tasks:
        task.status = "failed"
        task.stage = "failed"
        task.message = "训练 Worker 已重启，任务无法自动续训"
        task.error_code = "worker_lost"
        task.error_detail = "负责该任务的训练 Worker 已退出"
        task.finished_at = now
        task.heartbeat_at = now
        task.updated_at = now
        _append_event_locked(
            db,
            task.id,
            event_type="worker_lost",
            message=task.message,
            level="error",
            payload={"status": task.status},
        )
        task_ids.append(task.id)
    db.commit()
    return task_ids


def delete_terminal_job(
    db: Session,
    task_id: str,
    *,
    job_type: Optional[str] = None,
    commit: bool = True,
) -> Dict[str, Any]:
    """Delete a completed job and its event history from the persistent store."""
    task = db.query(JobTask).filter(JobTask.id == task_id).first()
    if not task:
        raise KeyError("job not found")
    if job_type and task.type != job_type:
        raise KeyError("job type not found")
    if task.status not in TERMINAL_STATUSES:
        raise ValueError("job is not terminal")

    deleted_events = db.query(JobEvent).filter(JobEvent.task_id == task.id).delete(synchronize_session=False)
    db.delete(task)
    if commit:
        db.commit()
    return {"task_id": task_id, "deleted_events": deleted_events}


def get_job_health_snapshot(db: Session, stale_after_seconds: Optional[int] = None) -> Dict[str, Any]:
    timeout = int(stale_after_seconds or os.getenv("JOB_HEARTBEAT_TIMEOUT_SECONDS", "600"))
    timeout = max(30, timeout)
    cutoff = datetime.now() - timedelta(seconds=timeout)
    status_counts = {
        status: int(count)
        for status, count in db.query(JobTask.status, func.count(JobTask.id)).group_by(JobTask.status).all()
    }
    stale_query = (
        db.query(JobTask.id)
        .filter(
            JobTask.status.in_({"running", "cancelling"}),
            JobTask.heartbeat_at.isnot(None),
            JobTask.heartbeat_at < cutoff,
        )
    )
    stale_count = stale_query.count()
    stale = stale_query.order_by(JobTask.heartbeat_at.asc()).limit(100).all()
    return {
        "heartbeat_timeout_seconds": timeout,
        "status_counts": status_counts,
        "active_tasks": sum(status_counts.get(status, 0) for status in ACTIVE_STATUSES),
        "stale_task_count": stale_count,
        "stale_task_ids": [row[0] for row in stale],
    }


def recover_active_jobs(db: Session) -> Dict[str, Iterable[str]]:
    recovered = {"training_failed": [], "annotation_interrupted": []}
    tasks = (
        db.query(JobTask)
        .filter(
            JobTask.type.notin_({"training", "behavior_analysis", "pose_training", "pose_evaluation"}),
            JobTask.status.in_({"pending", "running", "cancelling"}),
        )
        .all()
    )
    for task in tasks:
        task.status = "interrupted"
        task.stage = "interrupted"
        task.message = "服务重启导致任务中断"
        task.error_code = "service_restart"
        task.error_detail = "任务运行期间后端服务重启"
        task.heartbeat_at = datetime.now()
        task.updated_at = datetime.now()
        _append_event_locked(
            db,
            task.id,
            event_type="interrupted",
            message=task.message,
            level="warning",
            payload={"type": task.type},
        )
        if task.type == "training":
            task.status = "failed"
            task.finished_at = datetime.now()
            recovered["training_failed"].append(task.id)
        elif task.type == "annotation":
            recovered["annotation_interrupted"].append(task.id)
    db.commit()
    return recovered
