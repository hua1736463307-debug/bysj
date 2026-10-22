"""v1 标注接口。"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user, settings_dep
from app.core.config import Settings
from app.core.errors import ApiError, accepted, ok
from app.db.session import get_db
from app.domains import job_service
from app.domains.annotation import service as annotation_service
from app.domains.datasets import service as dataset_service

router = APIRouter(tags=["annotations"])


class BoxPayload(BaseModel):
    client_id: str | None = None
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    source: str = "manual"
    prediction_id: str | None = None


class AnnotationSaveRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    boxes: list[BoxPayload] = Field(default_factory=list)
    deleted_ids: list[str] = Field(default_factory=list)
    page: int = Field(default=1, ge=1)
    geometry: str = "rect"


class ExportRequest(BaseModel):
    dataset_id: str
    version_id: str | None = None
    format: str = "yolo-txt"


@router.get("/annotations/{image_id}", response_model=None)
def get_annotations(
    image_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return ok(annotation_service.get_annotation_state(db, image_id, user))


@router.put("/annotations/{image_id}", response_model=None)
def put_annotations(
    image_id: int,
    payload: AnnotationSaveRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    result = annotation_service.save_annotations(
        db,
        image_id,
        user,
        expected_revision=payload.expected_revision,
        boxes=[box.model_dump() for box in payload.boxes],
        deleted_ids=payload.deleted_ids,
        page=payload.page,
        geometry=payload.geometry,
    )
    return ok(result, message="标注已保存")


@router.post("/annotations/export", response_model=None)
def export_annotations(
    payload: ExportRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    resource = dataset_service.get_dataset(db, payload.dataset_id, user)
    version = (
        dataset_service.get_version(db, payload.version_id, user)
        if payload.version_id
        else dataset_service.latest_version(db, resource.id)
    )
    if version is None:
        raise ApiError(404, "not_found", "数据集还没有版本")
    if payload.format not in {"yolo-txt", "json"}:
        raise ApiError(422, "invalid_format", "导出格式只支持 yolo-txt 或 json")

    job, created = job_service.create_job(
        db,
        job_type="annotation_export",
        owner_id=user.id,
        owner_username=user.username,
        payload={
            "dataset_id": resource.id,
            "version_id": version.id,
            "format": payload.format,
            "dataset_version_id": version.id,
        },
    )
    if settings.inline_jobs and created:
        from app.workers.dispatcher import execute_job_inline

        execute_job_inline(job.id)
        db.refresh(job)
    return accepted(job.to_public(), message="导出任务已接收")
