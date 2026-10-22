"""v1 评估接口。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user, settings_dep
from app.core.config import Settings
from app.core.errors import accepted, ok
from app.db.models import EvaluationRun
from app.db.session import get_db
from app.domains.evaluation import service as evaluation_service
from app.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


class EvaluationRequest(BaseModel):
    model_id: str
    dataset_version_id: str
    split: str = "val"
    params: dict = Field(default_factory=dict)
    purpose: str | None = None
    idempotency_key: str | None = None


@router.post("", response_model=None)
def create_evaluation(
    payload: EvaluationRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    run, job, created_now = evaluation_service.create_evaluation(db, settings, user, payload.model_dump())
    if settings.inline_jobs and created_now:
        from app.workers.dispatcher import execute_job_inline

        execute_job_inline(job.id)
        db.refresh(run)
        db.refresh(job)
    return accepted(
        {"evaluation": evaluation_service.serialize(run), "job": job.to_public()},
        message="评估任务已接收",
    )


@router.get("", response_model=None)
def list_evaluations(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(EvaluationRun)
    if not user.is_admin:
        query = query.filter(EvaluationRun.owner_id == user.id)
    total = query.count()
    items = (
        query.order_by(EvaluationRun.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return ok(
        {
            "items": [evaluation_service.serialize(item) for item in items],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    )


@router.get("/{evaluation_id}", response_model=None)
def get_evaluation(
    evaluation_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    run = evaluation_service.get_evaluation(db, evaluation_id, user)
    return ok(evaluation_service.serialize(run))
