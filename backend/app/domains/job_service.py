"""持久化任务账本：七态、幂等键、领取、取消与事件。

状态机（CONTRACTS C04）：
pending -> running|cancelling|cancelled|failed|interrupted
running -> cancelling|completed|failed|cancelled|interrupted
cancelling -> cancelled|failed|interrupted
interrupted -> pending|running|failed|cancelled
completed/failed/cancelled -> 终态，无后继
"""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from typing import Any, Iterable

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.errors import ApiError
from app.db.base import utcnow
from app.db.models import JobEvent, JobTask, InferenceRun, EvaluationRun


def _sync_domain_state(db: Session, job: JobTask, status: str) -> None:
    from app.db.models import CxrAnalysisBatch

    mapping = {
        "inference": (InferenceRun, "run_id"),
        "evaluation": (EvaluationRun, "evaluation_id"),
        "cxr_evaluate": (EvaluationRun, "evaluation_id"),
        "cxr_analyze": (CxrAnalysisBatch, "batch_id"),
    }
    model, key = mapping.get(job.type, (None, None))
    if model is None or status == "completed":
        return  # Successful handlers retain their completed/partial result.
    target = (job.input_json or {}).get(key)
    if target:
        values = {"status": status, "job_id": job.id}
        if status in {"failed", "cancelled", "interrupted"}:
            values["finished_at"] = utcnow()
            if model is CxrAnalysisBatch:
                values["error_code"] = status
                values["error_detail"] = job.error_detail or job.message or ""
        elif status in {"pending", "running"}:
            values["finished_at"] = None
        db.query(model).filter(model.id == target).update(values, synchronize_session=False)

TERMINAL_STATES = {"completed", "failed", "cancelled"}
ACTIVE_STATES = {"pending", "running", "cancelling", "interrupted"}
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"running", "cancelling", "cancelled", "failed", "interrupted"},
    "running": {"cancelling", "completed", "failed", "cancelled", "interrupted"},
    "cancelling": {"cancelled", "failed", "interrupted"},
    "interrupted": {"pending", "running", "failed", "cancelled"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}


def canonical_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _transition_allowed(current: str, target: str) -> bool:
    return target in ALLOWED_TRANSITIONS.get(current, set())


def create_job(
    db: Session,
    *,
    job_type: str,
    owner_id: int,
    owner_username: str,
    payload: dict,
    idempotency_key: str | None = None,
) -> tuple[JobTask, bool]:
    """返回 (job, created)。相同键相同请求返回原任务；不同请求 409。"""
    request_hash = canonical_hash(payload)
    if idempotency_key:
        existing = (
            db.query(JobTask)
            .filter(
                JobTask.owner_id == str(owner_id),
                JobTask.type == job_type,
                JobTask.idempotency_key == idempotency_key,
            )
            .one_or_none()
        )
        if existing:
            if existing.request_hash != request_hash:
                raise ApiError(409, "idempotency_conflict", "相同幂等键对应了不同请求")
            return existing, False

    job = JobTask(
        type=job_type,
        status="pending",
        stage="queued",
        owner_id=str(owner_id),
        owner_username=owner_username,
        owner_user_id=owner_id,
        input_json=payload,
        idempotency_key=idempotency_key,
        request_hash=request_hash,
        message="任务已排队",
    )
    db.add(job)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        if idempotency_key:
            existing = (
                db.query(JobTask)
                .filter(
                    JobTask.owner_id == str(owner_id),
                    JobTask.type == job_type,
                    JobTask.idempotency_key == idempotency_key,
                )
                .one_or_none()
            )
            if existing and existing.request_hash == request_hash:
                return existing, False
            raise ApiError(409, "idempotency_conflict", "相同幂等键对应了不同请求") from exc
        raise
    db.refresh(job)
    append_event(db, job, "created", "任务已创建", {"type": job_type})
    return job, True


def append_event(
    db: Session,
    job: JobTask,
    event_type: str,
    message: str,
    payload: dict | None = None,
    level: str = "info",
) -> JobEvent:
    for _ in range(3):
        current = db.query(JobEvent).filter(JobEvent.task_id == job.id).order_by(JobEvent.sequence.desc()).first()
        sequence = (current.sequence + 1) if current else 1
        event = JobEvent(
            task_id=job.id,
            sequence=sequence,
            level=level,
            event_type=event_type,
            message=message,
            payload_json=payload or {},
        )
        db.add(event)
        try:
            db.commit()
            return event
        except IntegrityError:
            db.rollback()
    raise ApiError(503, "event_conflict", "任务事件写入冲突")


def get_job(db: Session, job_id: str, *, owner_id: int | None = None, is_admin: bool = False) -> JobTask:
    job = db.get(JobTask, job_id)
    if job is None or (not is_admin and owner_id is not None and job.owner_id != str(owner_id)):
        raise ApiError(404, "not_found", "任务不存在")
    return job


def list_jobs(
    db: Session,
    *,
    owner_id: int | None,
    is_admin: bool,
    job_type: str | None = None,
    status: str | None = None,
    page: int = 1,
    page_size: int = 50,
    search: str = "", owner: str = "", include_archived: bool = False,
    allowed_types: list[str] | None = None,
) -> tuple[list[JobTask], int]:
    query = db.query(JobTask)
    if allowed_types is not None:
        query = query.filter(JobTask.type.in_(allowed_types))
    if not include_archived:
        query = query.filter(JobTask.archived_at.is_(None))
    if search:
        from sqlalchemy import or_
        query = query.filter(or_(JobTask.id.contains(search, autoescape=True),
                                 JobTask.input_json["name"].as_string().contains(search, autoescape=True)))
    if owner:
        query = query.filter(JobTask.owner_username.contains(owner, autoescape=True))
    if not is_admin and owner_id is not None:
        query = query.filter(JobTask.owner_id == str(owner_id))
    if job_type:
        query = query.filter(JobTask.type == job_type)
    if status:
        query = query.filter(JobTask.status == status)
    total = query.count()
    items = (
        query.order_by(JobTask.created_at.desc(), JobTask.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def claim_job(
    db: Session,
    *,
    job_types: Iterable[str],
    worker_id: str,
    lease_seconds: int = 120,
) -> JobTask | None:
    """条件更新领取任务；SQLite/PostgreSQL 均按 status='pending' 原子竞争。"""
    types = list(job_types)
    if not types:
        return None
    while True:
        candidate = (
            db.query(JobTask)
            .filter(JobTask.type.in_(types), JobTask.status == "pending")
            .order_by(JobTask.created_at.asc(), JobTask.id.asc())
            .first()
        )
        if candidate is None:
            return None
        now = utcnow()
        updated = (
            db.query(JobTask)
            .filter(JobTask.id == candidate.id, JobTask.status == "pending")
            .update(
                {
                    "status": "running",
                    "stage": "starting",
                    "started_at": now,
                    "heartbeat_at": now,
                    "lease_owner": worker_id,
                    "lease_expires_at": now + timedelta(seconds=lease_seconds),
                    "message": "任务开始执行",
                },
                synchronize_session=False,
            )
        )
        if updated:
            _sync_domain_state(db, candidate, "running")
        db.commit()
        if updated:
            db.refresh(candidate)
            append_event(db, candidate, "started", "任务开始执行", {"worker": worker_id})
            return candidate


def heartbeat(db: Session, job: JobTask, *, worker_id: str, lease_seconds: int = 120, expected_attempt: int | None = None) -> bool:
    now = utcnow()
    updated = (
        db.query(JobTask)
        .filter(JobTask.id == job.id, JobTask.lease_owner == worker_id,
                JobTask.attempt == (expected_attempt if expected_attempt is not None else job.attempt),
                JobTask.lease_expires_at > now, JobTask.status.in_(["running", "cancelling"]))
        .update(
            {"heartbeat_at": now, "lease_expires_at": now + timedelta(seconds=lease_seconds)},
            synchronize_session=False,
        )
    )
    db.commit()
    return bool(updated)


def update_progress(
    db: Session,
    job: JobTask,
    *,
    worker_id: str,
    progress: float | None = None,
    stage: str | None = None,
    message: str | None = None,
    expected_attempt: int | None = None,
) -> None:
    values: dict[str, Any] = {}
    if progress is not None:
        values["progress"] = max(0.0, min(1.0, float(progress)))
    if stage is not None:
        values["stage"] = stage
    if message is not None:
        values["message"] = message
    if not values:
        return
    changed = db.query(JobTask).filter(
        JobTask.id == job.id,
        JobTask.lease_owner == worker_id,
        JobTask.attempt == (expected_attempt if expected_attempt is not None else job.attempt),
        JobTask.lease_expires_at > utcnow(),
        JobTask.status.in_(["running", "cancelling"]),
    ).update(values, synchronize_session=False)
    db.commit()
    if not changed:
        raise ApiError(409, "lease_lost", "任务租约已失效，拒绝写入进度")


def finish_job(
    db: Session,
    job: JobTask,
    *,
    worker_id: str,
    result: dict | None = None,
    message: str = "任务完成",
    expected_attempt: int | None = None,
) -> JobTask:
    attempt = expected_attempt if expected_attempt is not None else job.attempt
    db.refresh(job)
    if job.lease_owner != worker_id or job.attempt != attempt:
        raise ApiError(409, "lease_lost", "任务租约已失效，拒绝写入终态")
    target = "cancelled" if job.cancel_requested or job.status == "cancelling" else "completed"
    _apply_terminal(db, job, target, message=message, result=result, worker_id=worker_id, expected_attempt=attempt)
    return job


def fail_job(
    db: Session,
    job: JobTask,
    *,
    worker_id: str,
    code: str,
    detail: str,
    result: dict | None = None,
    expected_attempt: int | None = None,
) -> JobTask:
    attempt = expected_attempt if expected_attempt is not None else job.attempt
    db.refresh(job)
    if job.lease_owner != worker_id or job.attempt != attempt:
        raise ApiError(409, "lease_lost", "任务租约已失效，拒绝写入终态")
    _apply_terminal(db, job, "failed", message=detail[:500], result=result, worker_id=worker_id,
                    expected_attempt=attempt, errors={"error_code": code, "error_detail": detail[:4000]})
    return job


def _apply_terminal(db: Session, job: JobTask, target: str, *, message: str, result: dict | None,
                    worker_id: str | None = None, expected_attempt: int | None = None, errors: dict | None = None) -> None:
    if not _transition_allowed(job.status, target):
        raise ApiError(409, "invalid_transition", f"非法状态转换 {job.status} -> {target}")
    now = utcnow()
    query = db.query(JobTask).filter(JobTask.id == job.id, JobTask.status == job.status,
        JobTask.attempt == (expected_attempt if expected_attempt is not None else job.attempt),
        JobTask.cancel_requested == job.cancel_requested)
    if worker_id is not None:
        query = query.filter(JobTask.lease_owner == worker_id, JobTask.lease_expires_at > now)
    values = dict(status=target, stage=target, message=message, finished_at=now)
    values.update(errors or {})
    if result is not None:
        values["result_json"] = result
    if target in {"completed", "failed", "cancelled"}:
        values["progress"] = 1.0 if target == "completed" else job.progress
    changed = query.update(values, synchronize_session=False)
    if not changed:
        db.rollback()
        raise ApiError(409, "lease_lost", "任务状态或租约已变化，拒绝写入终态")
    _sync_domain_state(db, job, target)
    db.commit()
    db.refresh(job)
    append_event(db, job, target, message, result or {})


def cancel_job(db: Session, job: JobTask) -> JobTask:
    db.refresh(job)
    if job.status in TERMINAL_STATES:
        return job
    if job.status in {"pending", "interrupted"}:
        _apply_terminal(db, job, "cancelled", message="任务已取消", result=None)
        return job
    if job.status in {"running", "cancelling", "interrupted"}:
        if job.status != "cancelling":
            if not _transition_allowed(job.status, "cancelling"):
                raise ApiError(409, "invalid_transition", f"当前状态不可取消: {job.status}")
        changed = db.query(JobTask).filter(JobTask.id == job.id, JobTask.status == job.status,
            JobTask.attempt == job.attempt).update(dict(status="cancelling", cancel_requested=True,
            message="已请求取消，等待执行进程退出"), synchronize_session=False)
        if changed:
            _sync_domain_state(db, job, "cancelling")
        db.commit()
        db.refresh(job)
        if not changed:
            if job.status in TERMINAL_STATES:
                return job
            raise ApiError(409, "state_changed", "任务状态已变化，请刷新")
        append_event(db, job, "cancel_requested", "已请求取消")
        return job
    raise ApiError(409, "invalid_transition", f"当前状态不可取消: {job.status}")


def mark_stale_interrupted(db: Session, *, older_than_seconds: int = 300, job_id: str | None = None) -> int:
    """把租约过期且心跳超时的 running 任务标为 interrupted（不改写活跃 worker）。"""
    now = utcnow()
    cutoff = now - timedelta(seconds=older_than_seconds)
    stale = (
        db.query(JobTask)
        .filter(JobTask.status.in_(["running", "cancelling"]))
        .filter((JobTask.lease_expires_at.is_(None)) | (JobTask.lease_expires_at < now))
        .filter((JobTask.heartbeat_at.is_(None)) | (JobTask.heartbeat_at < cutoff))
        .filter(JobTask.id == job_id if job_id is not None else True)
        .all()
    )
    count = 0
    for job in stale:
        changed = db.query(JobTask).filter(JobTask.id == job.id,
            JobTask.attempt == job.attempt, JobTask.status.in_(["running", "cancelling"]),
            (JobTask.lease_expires_at.is_(None)) | (JobTask.lease_expires_at < now),
            (JobTask.heartbeat_at.is_(None)) | (JobTask.heartbeat_at < cutoff),
        ).update(dict(status="interrupted", stage="interrupted", message="worker 心跳超时，任务中断"), synchronize_session=False)
        if changed:
            _sync_domain_state(db, job, "interrupted")
        db.commit()
        if changed:
            count += 1
            append_event(db, job, "interrupted", "worker 心跳超时，任务中断", level="error")
    return count


def pause_for_resume(db: Session, job: JobTask) -> JobTask:
    """显式中断后重新排队：只有 interrupted 允许，生成新 attempt。"""
    db.refresh(job)
    if job.status != "interrupted":
        raise ApiError(409, "invalid_transition", "只有 interrupted 任务可以重新排队")
    changed = db.query(JobTask).filter(JobTask.id == job.id, JobTask.status == "interrupted",
        JobTask.attempt == job.attempt).update(dict(status="pending", stage="queued", attempt=job.attempt + 1,
        cancel_requested=False, lease_owner=None, lease_expires_at=None, started_at=None, finished_at=None,
        error_code=None, error_detail=None, message="任务重新排队"), synchronize_session=False)
    if not changed:
        db.rollback()
        raise ApiError(409, "state_changed", "任务状态已变化，请刷新")
    _sync_domain_state(db, job, "pending")
    db.commit()
    db.refresh(job)
    append_event(db, job, "requeued", "任务重新排队", {"attempt": job.attempt})
    return job
