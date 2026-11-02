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
from app.schemas.isic import IsicEvaluationRequest

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


# ---------------------------------------------------------------------------
# 训练与评价（七类单标签）
# ---------------------------------------------------------------------------


class IsicTrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version_id: str
    base_model_id: str | None = None
    params: dict = Field(default_factory=dict)
    #: P0：实验用途必须显式声明；缺失即"未分类"，产物不具备论文资格。
    experiment_purpose: str | None = Field(default=None, max_length=16, pattern=r"^(smoke|formal)$")
    #: formal 必须绑定冻结协议（`docs/datasets/ISIC-FORMAL-PROTOCOL-v1.md`）及其字节哈希。
    protocol_id: str | None = Field(default=None, max_length=64)
    protocol_sha256: str | None = Field(default=None, max_length=64)


def _run_inline(settings: Settings, job, created_now: bool) -> None:
    if settings.inline_jobs and created_now:
        from app.workers.dispatcher import execute_job_inline

        execute_job_inline(job.id)


@router.post("/training/preflight", response_model=None)
def training_preflight(
    payload: IsicTrainingRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    from app.domains.isic import training_service

    return ok(training_service.preflight(db, settings, user, payload.model_dump()))


@router.post("/training/jobs", response_model=None)
def create_training(
    payload: IsicTrainingRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=128),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    from app.domains.isic import training_service

    job, created_now = training_service.create_training_job(
        db, settings, user, payload.model_dump(), idempotency_key=idempotency_key
    )
    _run_inline(settings, job, created_now)
    db.refresh(job)
    return accepted(job.to_public(), message="皮肤镜训练任务已接收")


@router.post("/evaluations", response_model=None)
def create_evaluation(
    payload: IsicEvaluationRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=128),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    from app.domains.isic import evaluation_service

    job, created_now = evaluation_service.create_evaluation_job(
        db, settings, user, payload.model_dump(), idempotency_key=idempotency_key
    )
    _run_inline(settings, job, created_now)
    db.refresh(job)
    return accepted(job.to_public(), message="皮肤镜评价任务已接收")


def _model_paper_eligibility(model) -> dict:
    """P0：论文资格按模型元数据重新计算，不信任任何已存布尔字段。"""
    from app.domains.isic.experiment_purpose import paper_eligibility

    if model is None:
        return {"paper_eligible": False, "experiment_purpose": None, "reasons": ["model_missing"]}
    return paper_eligibility(model.metadata_json)


@router.get("/evaluations", response_model=None)
def list_evaluations(
    dataset_version_id: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from app.db.models import EvaluationRun, ModelRegistry

    query = db.query(EvaluationRun).join(ModelRegistry, ModelRegistry.id == EvaluationRun.model_id).filter(
        ModelRegistry.model_task == ISIC_TASK
    ).order_by(EvaluationRun.created_at.desc())
    if dataset_version_id:
        query = query.filter(EvaluationRun.dataset_version_id == dataset_version_id)
    if not user.is_admin:
        query = query.filter(EvaluationRun.owner_id == user.id)
    total = query.count()
    rows = query.offset((page - 1) * page_size).limit(page_size).all()
    models = {
        item.id: item
        for item in db.query(ModelRegistry).filter(ModelRegistry.id.in_({row.model_id for row in rows})).all()
    } if rows else {}
    items = [
        {
            "evaluation_id": row.id,
            "job_id": row.job_id,
            "model_id": row.model_id,
            "model_name": (models.get(row.model_id).name if models.get(row.model_id) else None),
            "dataset_version_id": row.dataset_version_id,
            "split": row.split,
            "status": row.status,
            "purpose": row.purpose,
            "paper_eligibility": _model_paper_eligibility(models.get(row.model_id)),
            "metrics": row.metrics_json,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        }
        for row in rows
    ]
    return ok({"items": items, "total": total, "page": page, "page_size": page_size})


@router.get("/models", response_model=None)
def list_isic_models(
    task: str = Query(default="isic_multiclass"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from app.db.models import ModelRegistry

    if task not in {"isic_multiclass", "isic_backbone"}:
        raise ApiError(422, "isic_model_task", "task 只支持 isic_multiclass 或 isic_backbone")
    query = db.query(ModelRegistry).filter(ModelRegistry.status == "active", ModelRegistry.model_task == task)
    if not user.is_admin:
        query = query.filter(ModelRegistry.owner_id == user.id)
    total = query.count()
    rows = query.order_by(ModelRegistry.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return ok(
        {
            "items": [
                {
                    "model_id": row.id,
                    "name": row.name,
                    "model_task": row.model_task,
                    "sha256": row.sha256,
                    "size": row.size,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "label_schema": (row.metadata_json or {}).get("label_schema"),
                    "class_order": (row.metadata_json or {}).get("class_order"),
                    "research_valid": (row.metadata_json or {}).get("research_valid"),
                    "paper_eligibility": _model_paper_eligibility(row),
                    "metrics": (row.metadata_json or {}).get("metrics"),
                }
                for row in rows
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    )


@router.get("/results/paper-summary", response_model=None)
def paper_summary(
    dataset_version_id: str | None = Query(default=None),
    user: CurrentUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """P0：论文汇总只收 `paper_eligible=true` 的 test 评价；其余单列原因，数值不进入汇总。"""
    from app.db.models import EvaluationRun, ModelRegistry
    from app.domains.isic.experiment_purpose import FORMAL_PROTOCOL_ID

    query = db.query(EvaluationRun).join(ModelRegistry, ModelRegistry.id == EvaluationRun.model_id).filter(
        ModelRegistry.model_task == ISIC_TASK,
        EvaluationRun.split == "test", EvaluationRun.status == "completed"
    )
    if dataset_version_id:
        query = query.filter(EvaluationRun.dataset_version_id == dataset_version_id)
    rows = query.order_by(EvaluationRun.created_at.asc()).all()
    models = {
        item.id: item
        for item in db.query(ModelRegistry)
        .filter(ModelRegistry.id.in_({row.model_id for row in rows}))
        .all()
    } if rows else {}

    included: list[dict] = []
    excluded: list[dict] = []
    for row in rows:
        report = _model_paper_eligibility(models.get(row.model_id))
        base = {
            "evaluation_id": row.id,
            "model_id": row.model_id,
            "dataset_version_id": row.dataset_version_id,
            "paper_eligibility": report,
        }
        if report["paper_eligible"]:
            included.append({**base, "metrics": row.metrics_json})
        else:
            excluded.append(base)

    return ok(
        {
            "protocol_id": FORMAL_PROTOCOL_ID,
            "included": included,
            "excluded": excluded,
            "included_count": len(included),
            "excluded_count": len(excluded),
            "note": "仅收 experiment_purpose=formal 且通过冻结协议/数据/权重/seed 门禁的 test 评价；排除项不提供指标。",
        }
    )
