"""v1 训练接口。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user, settings_dep
from app.core.config import Settings
from app.core.errors import ApiError, accepted, ok
from app.db.session import get_db
from app.domains import job_service
from app.domains.training import service as training_service

router = APIRouter(prefix="/training", tags=["training"])


class TrainingParams(BaseModel):
    epochs: int = Field(default=80, ge=1, le=1000)
    batch: int = Field(default=8, ge=1, le=128)
    imgsz: int = Field(default=640, ge=32, le=2048)
    seed: int = Field(default=42, ge=0)
    deterministic: bool = True
    device: str = "0"


class TrainingRequest(BaseModel):
    dataset_version_id: str
    model_id: str | None = None
    model_ref: str | None = None
    task: str = "detect"
    params: TrainingParams = Field(default_factory=TrainingParams)


@router.post("/preflight", response_model=None)
def preflight(
    payload: TrainingRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    body = payload.model_dump()
    body["params"] = payload.params.model_dump()
    report = training_service.preflight(db, settings, user, body)
    return ok(report, message="预检完成" if report["ok"] else "预检发现问题")


@router.post("/jobs", response_model=None)
def create_job(
    payload: TrainingRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    body = payload.model_dump()
    body["params"] = payload.params.model_dump()
    job, created = training_service.create_training_job(
        db, settings, user, body, idempotency_key=idempotency_key
    )
    if settings.inline_jobs and created:
        from app.workers.dispatcher import execute_job_inline

        execute_job_inline(job.id)
        db.refresh(job)
    return accepted(job.to_public(), message="训练任务已接收")


@router.get("/jobs/{job_id}", response_model=None)
def get_training_job(
    job_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    job = job_service.get_job(db, job_id, owner_id=user.id, is_admin=user.is_admin)
    if job.type != "training":
        raise ApiError(404, "not_found", "训练任务不存在")
    return ok(job.to_public())


@router.post("/jobs/{job_id}/cancel", response_model=None)
def cancel_training_job(
    job_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    job = job_service.get_job(db, job_id, owner_id=user.id, is_admin=user.is_admin)
    if job.type != "training":
        raise ApiError(404, "not_found", "训练任务不存在")
    job = job_service.cancel_job(db, job)
    return accepted(job.to_public(), message="取消请求已处理")
