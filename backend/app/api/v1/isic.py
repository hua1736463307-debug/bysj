"""v1 皮肤镜方向接口（`/api/v1/isic`）。

单标签七类（ISIC 2018 Task 3）。与 `/api/v1/cxr` 并列：独立导入、独立冻结、
独立摘要；不复用胸片四态，也不把七类混入胸片标签列。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user, require_admin, settings_dep
from app.core.config import Settings
from app.core.errors import ApiError, accepted, ok
from app.db.models import DatasetResource, DatasetVersion, IsicSample
from app.db.session import get_db
from app.directions.isic import ISIC_DIRECTION_KEY, ISIC_LABEL_SCHEMA, ISIC_LABELS, ISIC_SPLITS, ISIC_TASK
from app.domains import job_service
from app.domains.datasets import service as dataset_service
from app.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

router = APIRouter(prefix="/isic", tags=["isic"])


class IsicImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    split: str = Field(default="val", pattern=r"^(train|val|test)$")
    name: str = Field(default="", max_length=255)
    license_note: str | None = Field(default=None, max_length=2000)
    max_images: int | None = Field(default=None, ge=1, le=20000)


class IsicFreezeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split_seed: int = Field(default=42, ge=0, le=2**31 - 1)
    ratios: list[float] = Field(default_factory=lambda: [0.7, 0.1, 0.1, 0.1])

    @field_validator("ratios")
    @classmethod
    def _check_ratios(cls, value: list[float]) -> list[float]:
        if len(value) != len(ISIC_SPLITS) or any(item <= 0 for item in value):
            raise ValueError("四集比例必须是四个正数")
        if abs(sum(value) - 1.0) > 1e-6:
            raise ValueError("四集比例之和必须为 1")
        return value


@router.get("/schema", response_model=None)
def get_schema(user: CurrentUser = Depends(get_current_user)):
    return ok(
        {
            "direction": ISIC_DIRECTION_KEY,
            "task": ISIC_TASK,
            "label_schema": ISIC_LABEL_SCHEMA,
            "task_kind": "single_label_classification",
            "labels": [
                {"index": label.index, "key": label.key, "display_name": label.display_name}
                for label in ISIC_LABELS
            ],
            "splits": list(ISIC_SPLITS),
            "group_level": "lesion",
            "note": "单标签互斥：输出 softmax 分布（和为 1），与胸片多标签独立 sigmoid 不同。",
        }
    )


@router.post("/datasets/import", response_model=None)
def import_dataset(
    payload: IsicImportRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=128),
    user: CurrentUser = Depends(require_admin),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    job, created_now = job_service.create_job(
        db,
        job_type="isic_import",
        owner_id=user.id,
        owner_username=user.username,
        payload=payload.model_dump(),
        idempotency_key=idempotency_key,
    )
    if settings.inline_jobs and created_now:
        from app.workers.dispatcher import execute_job_inline

        execute_job_inline(job.id)
        db.refresh(job)
    return accepted(job.to_public(), message="皮肤镜导入任务已接收")


@router.get("/datasets", response_model=None)
def list_datasets(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(DatasetResource).filter(
        DatasetResource.task_type == ISIC_TASK, DatasetResource.deleted_at.is_(None)
    )
    if not user.is_admin:
        query = query.filter(DatasetResource.owner_user_id == user.id)
    total = query.count()
    items = (
        query.order_by(DatasetResource.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    )
    payload = []
    for item in items:
        version = (
            db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == item.id, DatasetVersion.deleted_at.is_(None))
            .order_by(DatasetVersion.created_at.desc())
            .first()
        )
        payload.append(
            {
                "dataset_id": item.id,
                "dataset_code": item.dataset_id,
                "name": item.name,
                "task_type": item.task_type,
                "group_level": item.group_level,
                "source_version": item.source_version,
                "license": item.license,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "latest_version": None
                if version is None
                else {
                    "version_id": version.id,
                    "state": version.state,
                    "asset_count": version.asset_count,
                    "split_counts": version.split_counts_json or {},
                    "manifest_sha256": version.manifest_sha256,
                    "class_gate": (version.metadata_json or {}).get("class_gate"),
                    "rare_class_warnings": (version.metadata_json or {}).get("rare_class_warnings"),
                },
            }
        )
    return ok({"items": payload, "total": total, "page": page, "page_size": page_size})


@router.get("/datasets/{dataset_id}", response_model=None)
def get_dataset(
    dataset_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    resource = dataset_service.get_dataset(db, dataset_id, user)
    if resource.task_type != ISIC_TASK:
        raise ApiError(409, "task_mismatch", "该数据集不是皮肤镜数据集", {"task_type": resource.task_type})
    version = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == resource.id, DatasetVersion.deleted_at.is_(None))
        .order_by(DatasetVersion.created_at.desc())
        .first()
    )
    sample_count = (
        db.query(IsicSample).filter(IsicSample.dataset_version_id == version.id).count() if version else 0
    )
    return ok(
        {
            "dataset_id": resource.id,
            "dataset_code": resource.dataset_id,
            "name": resource.name,
            "task_type": resource.task_type,
            "group_level": resource.group_level,
            "license": resource.license,
            "sample_count": sample_count,
            "latest_version_id": version.id if version else None,
            "latest_version_state": version.state if version else None,
        }
    )


@router.post("/datasets/{dataset_id}/versions", response_model=None)
def freeze_version(
    dataset_id: str,
    payload: IsicFreezeRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=128),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    resource = dataset_service.get_dataset(db, dataset_id, user)
    if resource.task_type != ISIC_TASK:
        raise ApiError(409, "task_mismatch", "该数据集不是皮肤镜数据集", {"task_type": resource.task_type})
    version = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == resource.id, DatasetVersion.deleted_at.is_(None))
        .order_by(DatasetVersion.created_at.desc())
        .first()
    )
    if version is None:
        raise ApiError(404, "not_found", "数据集还没有版本")
    if version.state == "frozen":
        raise ApiError(409, "version_frozen", "版本已冻结，内容不可修改")
    job, created_now = job_service.create_job(
        db,
        job_type="isic_freeze",
        owner_id=user.id,
        owner_username=user.username,
        payload={
            "dataset_id": resource.id,
            "version_id": version.id,
            "dataset_version_id": version.id,
            "split_seed": payload.split_seed,
            "ratios": payload.ratios,
        },
        idempotency_key=idempotency_key,
    )
    if settings.inline_jobs and created_now:
        from app.workers.dispatcher import execute_job_inline

        execute_job_inline(job.id)
        db.refresh(job)
    return accepted(job.to_public(), message="皮肤镜版本冻结任务已接收")


@router.get("/datasets/{dataset_id}/versions/{version_id}/summary", response_model=None)
def version_summary(
    dataset_id: str,
    version_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    resource = dataset_service.get_dataset(db, dataset_id, user)
    if resource.task_type != ISIC_TASK:
        raise ApiError(409, "task_mismatch", "该数据集不是皮肤镜数据集")
    version = dataset_service.get_version(db, version_id, user)
    if version.dataset_id != resource.id:
        raise ApiError(404, "not_found", "该版本不属于此数据集")
    if version.state != "frozen":
        raise ApiError(409, "version_not_frozen", "版本尚未冻结，没有四集摘要")
    import json
    from pathlib import Path

    manifest = json.loads(Path(version.manifest_path).read_text(encoding="utf-8"))
    return ok(
        {
            "version": {
                "version_id": version.id,
                "state": version.state,
                "asset_count": version.asset_count,
                "split_counts": version.split_counts_json,
                "split_config": version.split_config_json,
                "manifest_sha256": version.manifest_sha256,
                "split_manifest_hash": version.split_manifest_hash,
            },
            "counts": manifest.get("counts"),
            "class_counts": manifest.get("class_counts"),
            "group_counts": manifest.get("group_counts"),
            "group_level": manifest.get("group_level"),
            "class_gate": (version.metadata_json or {}).get("class_gate"),
            "rare_class_warnings": (version.metadata_json or {}).get("rare_class_warnings"),
            "source": manifest.get("source"),
            "note": "单标签七类；总体准确率会被多数类主导，须同时看逐类指标。",
        }
    )
