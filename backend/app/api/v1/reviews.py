"""v1 人工复核与回流接口。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user, require_reviewer, settings_dep
from app.core.config import Settings
from app.core.errors import ApiError, accepted, created, ok
from app.db.models import ReviewRecord
from app.db.session import get_db
from app.domains import job_service
from app.domains.datasets import service as dataset_service
from app.domains.feedback import review_service

router = APIRouter(prefix="/reviews", tags=["reviews"])


class ReviewItemPayload(BaseModel):
    prediction_id: str | None = None
    verdict: str
    class_id: int | None = None
    corrected: dict | None = None


class ReviewCreateRequest(BaseModel):
    inference_item_id: str | None = None
    image_asset_id: int | None = None
    reason: str | None = None
    items: list[ReviewItemPayload] = Field(default_factory=list)
    final_annotations: list[dict] | None = None


class ReviewUpdateRequest(ReviewCreateRequest):
    expected_revision: int = Field(ge=0)


class DecisionRequest(BaseModel):
    approve: bool
    reason: str | None = None


class ExportVersionRequest(BaseModel):
    parent_version_id: str
    review_ids: list[str]
    idempotency_key: str | None = None


@router.post("", response_model=None)
def create_review(
    payload: ReviewCreateRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    review = review_service.create_review(db, user, payload.model_dump())
    return created(review_service.serialize(db, review), message="复核记录已创建")


@router.get("", response_model=None)
def list_reviews(
    state: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(ReviewRecord)
    if not user.is_admin and user.role != "reviewer":
        query = query.filter(ReviewRecord.owner_id == user.id)
    if state:
        query = query.filter(ReviewRecord.state == state)
    total = query.count()
    items = (
        query.order_by(ReviewRecord.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return ok(
        {
            "items": [review_service.serialize(db, item) for item in items],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    )


@router.post("/export-version", response_model=None)
def export_version(
    payload: ExportVersionRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    parent = dataset_service.get_version(db, payload.parent_version_id, user)
    job, created_now = job_service.create_job(
        db,
        job_type="feedback_export",
        owner_id=user.id,
        owner_username=user.username,
        payload={
            "parent_version_id": parent.id,
            "review_ids": payload.review_ids,
            "idempotency_key": payload.idempotency_key,
            "dataset_version_id": parent.id,
        },
        idempotency_key=payload.idempotency_key,
    )
    if settings.inline_jobs and created_now:
        from app.workers.dispatcher import execute_job_inline

        execute_job_inline(job.id)
        db.refresh(job)
    return accepted(job.to_public(), message="回流导出任务已接收")


@router.get("/{review_id}", response_model=None)
def get_review(
    review_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    review = db.get(ReviewRecord, review_id)
    if review is None or (not user.is_admin and user.role != "reviewer" and review.owner_id != user.id):
        raise ApiError(404, "not_found", "复核记录不存在")
    return ok(review_service.serialize(db, review))


@router.patch("/{review_id}", response_model=None)
def update_review(
    review_id: str,
    payload: ReviewUpdateRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    review = db.get(ReviewRecord, review_id)
    if review is None:
        raise ApiError(404, "not_found", "复核记录不存在")
    review = review_service.update_review(db, review, user, payload.model_dump())
    return ok(review_service.serialize(db, review), message="复核记录已更新")


@router.post("/{review_id}/submit", response_model=None)
def submit_review(
    review_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    review = db.get(ReviewRecord, review_id)
    if review is None:
        raise ApiError(404, "not_found", "复核记录不存在")
    review = review_service.submit_review(db, review, user)
    return ok(review_service.serialize(db, review), message="复核记录已提交")


@router.post("/{review_id}/decision", response_model=None)
def decide_review(
    review_id: str,
    payload: DecisionRequest,
    user: CurrentUser = Depends(require_reviewer),
    db: Session = Depends(get_db),
):
    review = db.get(ReviewRecord, review_id)
    if review is None:
        raise ApiError(404, "not_found", "复核记录不存在")
    review = review_service.decide_review(db, review, user, approve=payload.approve, reason=payload.reason)
    return ok(review_service.serialize(db, review), message="复核已审批")
