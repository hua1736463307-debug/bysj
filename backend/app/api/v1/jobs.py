"""v1 通用任务接口（导入/版本/训练/推理/评估共用）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user
from app.core.errors import ApiError, accepted, ok
from app.db.models import JobEvent
from app.db.session import get_db
from app.domains import job_service
from app.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=None)
def list_jobs(
    type: str | None = Query(default=None),
    status: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    items, total = job_service.list_jobs(
        db,
        owner_id=user.id,
        is_admin=user.is_admin,
        job_type=type,
        status=status,
        page=page,
        page_size=page_size,
    )
    return ok(
        {
            "items": [item.to_public() for item in items],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    )


@router.get("/{job_id}", response_model=None)
def get_job(
    job_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    job = job_service.get_job(db, job_id, owner_id=user.id, is_admin=user.is_admin)
    return ok(job.to_public())


@router.get("/{job_id}/events", response_model=None)
def list_events(
    job_id: str,
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=200),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    job = job_service.get_job(db, job_id, owner_id=user.id, is_admin=user.is_admin)
    events = (
        db.query(JobEvent)
        .filter(JobEvent.task_id == job.id, JobEvent.sequence > after_sequence)
        .order_by(JobEvent.sequence.asc())
        .limit(limit)
        .all()
    )
    return ok(
        {
            "items": [
                {
                    "sequence": event.sequence,
                    "level": event.level,
                    "event_type": event.event_type,
                    "message": event.message,
                    "payload": event.payload_json or {},
                    "created_at": event.created_at.isoformat() if event.created_at else None,
                }
                for event in events
            ],
            "job_id": job.id,
        }
    )


@router.post("/{job_id}/cancel", response_model=None)
def cancel_job(
    job_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    job = job_service.get_job(db, job_id, owner_id=user.id, is_admin=user.is_admin)
    if job.status in job_service.TERMINAL_STATES:
        raise ApiError(409, "job_finished", f"任务已处于终态: {job.status}")
    job = job_service.cancel_job(db, job)
    return accepted(job.to_public(), message="取消请求已处理")


@router.post("/{job_id}/resume", response_model=None)
def resume_job(
    job_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    job = job_service.get_job(db, job_id, owner_id=user.id, is_admin=user.is_admin)
    job = job_service.pause_for_resume(db, job)
    return accepted(job.to_public(), message="任务已重新排队")
