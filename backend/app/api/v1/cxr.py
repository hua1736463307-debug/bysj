"""v1 胸片五标签辅助分析接口（`/api/v1/cxr`）。

沿用 `code/message/data/request_id` envelope；认证必需。
权限：管理员管来源配置/可信模型/test 评价；researcher 只操作自身资源；
reviewer 按授权复核。软件权限不代表临床执业资格。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Header, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user, require_admin, settings_dep
from app.core.config import Settings
from app.core.errors import ApiError, accepted, created, ok
from app.db.models import (
    CxrAnalysis,
    CxrAnalysisBatch,
    CxrReport,
    CxrReview,
    DatasetResource,
    DatasetVersion,
    EvaluationRun,
    ModelRegistry,
)
from app.db.session import get_db
from app.domains import job_service
from app.domains.cxr import (
    analysis_service,
    evaluation_service,
    report_service,
    review_service,
    service as cxr_service,
    training_service,
)
from app.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from app.schemas.cxr import (
    CXR_TASK,
    CxrAnalysisBatchRequest,
    CxrDecisionRequest,
    CxrEvaluationRequest,
    CxrExplanationRequest,
    CxrFreezeRequest,
    CxrImportRequest,
    CxrReportRequest,
    CxrReviewCreateRequest,
    CxrReviewUpdateRequest,
    CxrTrainingRequest,
    schema_payload,
)

router = APIRouter(prefix="/cxr", tags=["cxr"])


def _run_inline(settings: Settings, job, created_now: bool) -> None:
    if settings.inline_jobs and created_now:
        from app.workers.dispatcher import execute_job_inline

        execute_job_inline(job.id)


def _dataset_dict(resource: DatasetResource, version: DatasetVersion | None) -> dict:
    return {
        "dataset_id": resource.id,
        "dataset_code": resource.dataset_id,
        "name": resource.name,
        "task_type": resource.task_type,
        "group_level": resource.group_level,
        "modality": resource.modality,
        "source_version": resource.source_version,
        "license": resource.license,
        "owner": resource.owner_username,
        "created_at": resource.created_at.isoformat() if resource.created_at else None,
        "latest_version": _version_dict(version) if version else None,
    }


def _version_dict(version: DatasetVersion) -> dict:
    return {
        "version_id": version.id,
        "dataset_id": version.dataset_id,
        "state": version.state,
        "asset_count": version.asset_count,
        "split_counts": version.split_counts_json or {},
        "split_config": version.split_config_json or {},
        "manifest_sha256": version.manifest_sha256,
        "split_manifest_hash": version.split_manifest_hash,
        "label_schema_hash": version.class_map_hash,
        "frozen_at": version.frozen_at.isoformat() if version.frozen_at else None,
        "created_at": version.created_at.isoformat() if version.created_at else None,
    }


@router.get("/schema", response_model=None)
def get_schema(user: CurrentUser = Depends(get_current_user)):
    return ok(schema_payload())


@router.get("/dashboard", response_model=None)
def dashboard(user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)):
    dataset_query = db.query(DatasetResource).filter(
        DatasetResource.task_type == CXR_TASK, DatasetResource.deleted_at.is_(None)
    )
    analysis_query = db.query(CxrAnalysis)
    review_query = db.query(CxrReview)
    report_query = db.query(CxrReport)
    if not user.is_admin:
        dataset_query = dataset_query.filter(DatasetResource.owner_user_id == user.id)
        analysis_query = analysis_query.filter(CxrAnalysis.owner_id == user.id)
        review_query = review_query.filter(CxrReview.owner_id == user.id)
        report_query = report_query.filter(CxrReport.owner_id == user.id)
    version_count = (
        db.query(func.count(DatasetVersion.id))
        .join(DatasetResource, DatasetResource.id == DatasetVersion.dataset_id)
        .filter(DatasetResource.task_type == CXR_TASK, DatasetVersion.deleted_at.is_(None))
        .scalar()
        or 0
    )
    return ok(
        {
            "task": CXR_TASK,
            "datasets": dataset_query.count(),
            "versions": int(version_count),
            "analyses": {
                "total": analysis_query.count(),
                "completed": analysis_query.filter(CxrAnalysis.status == "completed").count(),
                "failed": analysis_query.filter(CxrAnalysis.status == "failed").count(),
            },
            "reviews": {
                "draft": review_query.filter(CxrReview.state == "draft").count(),
                "submitted": review_query.filter(CxrReview.state == "submitted").count(),
                "approved": review_query.filter(CxrReview.state == "approved").count(),
                "rejected": review_query.filter(CxrReview.state == "rejected").count(),
            },
            "reports": {
                "draft": report_query.filter(CxrReport.state == "draft").count(),
                "final": report_query.filter(CxrReport.state == "final").count(),
            },
            "note": "统计来自真实记录；没有数据时显示 0，不生成虚构病例或准确率。",
        }
    )


@router.post("/datasets/import", response_model=None)
def import_dataset(
    payload: CxrImportRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=128),
    user: CurrentUser = Depends(require_admin),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    job, created_now = job_service.create_job(
        db,
        job_type="cxr_import",
        owner_id=user.id,
        owner_username=user.username,
        payload=payload.model_dump(),
        idempotency_key=idempotency_key,
    )
    _run_inline(settings, job, created_now)
    db.refresh(job)
    return accepted(job.to_public(), message="胸片导入任务已接收")


@router.get("/datasets", response_model=None)
def list_datasets(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(DatasetResource).filter(
        DatasetResource.task_type == CXR_TASK, DatasetResource.deleted_at.is_(None)
    )
    if not user.is_admin:
        query = query.filter(DatasetResource.owner_user_id == user.id)
    total = query.count()
    items = (
        query.order_by(DatasetResource.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    latest = {}
    for item in items:
        version = (
            db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == item.id, DatasetVersion.deleted_at.is_(None))
            .order_by(DatasetVersion.created_at.desc())
            .first()
        )
        latest[item.id] = version
    return ok(
        {
            "items": [_dataset_dict(item, latest.get(item.id)) for item in items],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    )


@router.get("/datasets/{dataset_id}", response_model=None)
def get_dataset(
    dataset_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    resource = cxr_service.get_cxr_dataset(db, dataset_id, user)
    version = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == resource.id, DatasetVersion.deleted_at.is_(None))
        .order_by(DatasetVersion.created_at.desc())
        .first()
    )
    return ok(_dataset_dict(resource, version))


@router.post("/datasets/{dataset_id}/versions", response_model=None)
def freeze_version(
    dataset_id: str,
    payload: CxrFreezeRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=128),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    resource = cxr_service.get_cxr_dataset(db, dataset_id, user)
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
        job_type="cxr_freeze",
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
    _run_inline(settings, job, created_now)
    db.refresh(job)
    return accepted(job.to_public(), message="胸片版本冻结任务已接收")


@router.get("/datasets/{dataset_id}/versions/{version_id}/summary", response_model=None)
def version_summary(
    dataset_id: str,
    version_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    resource = cxr_service.get_cxr_dataset(db, dataset_id, user)
    version = cxr_service.get_cxr_version(db, version_id, user)
    if version.dataset_id != resource.id:
        raise ApiError(404, "not_found", "该版本不属于此数据集")
    if version.state != "frozen":
        raise ApiError(409, "version_not_frozen", "版本尚未冻结，没有四集摘要")
    manifest = cxr_service.load_manifest(version)
    return ok(
        {
            "version": _version_dict(version),
            "counts": manifest.get("counts"),
            "patient_counts": manifest.get("patient_counts"),
            "component_counts": manifest.get("component_counts"),
            "label_coverage": manifest.get("label_coverage"),
            "masked_support": manifest.get("masked_support"),
            "duplicate_candidates": manifest.get("duplicate_candidates"),
            "training_gate": training_service.training_gate(manifest),
            "source": {
                "source_id": (manifest.get("source") or {}).get("source_version"),
                "manifest_schema": manifest.get("schema"),
            },
            "note": "患者与检查只显示伪标识计数；原始 CSV/路径不通过本接口下发。",
        }
    )


@router.post("/training/preflight", response_model=None)
def training_preflight(
    payload: CxrTrainingRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    report = training_service.preflight(db, settings, user, payload.model_dump())
    return ok(report)


@router.post("/training/jobs", response_model=None)
def create_training(
    payload: CxrTrainingRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=128),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    job, created_now = training_service.create_training_job(
        db, settings, user, payload.model_dump(), idempotency_key=idempotency_key
    )
    _run_inline(settings, job, created_now)
    db.refresh(job)
    return accepted(job.to_public(), message="胸片训练任务已接收")


@router.get("/models", response_model=None)
def list_cxr_models(
    task: str = Query(default="cxr_multilabel"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if task not in {"cxr_multilabel", "cxr_backbone"}:
        raise ApiError(422, "cxr_model_task", "task 只支持 cxr_multilabel 或 cxr_backbone")
    query = db.query(ModelRegistry).filter(ModelRegistry.status == "active", ModelRegistry.model_task == task)
    if not user.is_admin:
        query = query.filter(
            (ModelRegistry.owner_id == user.id) | (ModelRegistry.trusted_source == "base")
        )
    total = query.count()
    items = (
        query.order_by(ModelRegistry.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return ok(
        {
            "items": [
                {
                    **item.to_public(),
                    "label_schema": (item.metadata_json or {}).get("label_schema"),
                    "backend": (item.metadata_json or {}).get("backend"),
                    "research_valid": (item.metadata_json or {}).get("research_valid", True),
                }
                for item in items
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    )


@router.post("/evaluations", response_model=None)
def create_evaluation(
    payload: CxrEvaluationRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=128),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    job, created_now = evaluation_service.create_evaluation_job(
        db, settings, user, payload.model_dump(), idempotency_key=idempotency_key
    )
    _run_inline(settings, job, created_now)
    db.refresh(job)
    return accepted(job.to_public(), message="胸片评价任务已接收")


@router.get("/evaluations", response_model=None)
def list_evaluations(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(EvaluationRun).filter(
        EvaluationRun.params_json.isnot(None)
    )
    if not user.is_admin:
        query = query.filter(EvaluationRun.owner_id == user.id)
    runs = query.order_by(EvaluationRun.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    items = []
    for run in runs:
        params = run.params_json or {}
        if params.get("task") != CXR_TASK:
            continue
        items.append(
            {
                "evaluation_id": run.id,
                "model_id": run.model_id,
                "dataset_version_id": run.dataset_version_id,
                "split": run.split,
                "action": params.get("action"),
                "status": run.status,
                "purpose": run.purpose,
                "metrics": evaluation_service.summary_metrics(run.metrics_json or {}),
                "created_at": run.created_at.isoformat() if run.created_at else None,
            }
        )
    return ok({"items": items, "page": page, "page_size": page_size})


@router.post("/analysis-batches", response_model=None)
def create_analysis_batch(
    payload: CxrAnalysisBatchRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=128),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    batch, job, created_now = analysis_service.create_batch(
        db,
        settings,
        user,
        model_id=payload.model_id,
        asset_ids=payload.asset_ids,
        idempotency_key=idempotency_key,
    )
    _run_inline(settings, job, created_now)
    db.refresh(batch)
    return accepted(
        {**cxr_service.serialize_batch(batch), "job": job.to_public()},
        message="胸片分析批次已接收",
    )


@router.post("/analysis-batches/upload", response_model=None)
def create_analysis_batch_upload(
    model_id: str = Form(...),
    files: list[UploadFile] = File(...),
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=128),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    if not files:
        raise ApiError(422, "no_files", "至少上传一张胸片")
    if len(files) > settings.max_batch_inference:
        raise ApiError(422, "cxr_batch_too_large", f"单批最多 {settings.max_batch_inference} 张，未截断输入")
    batch_id = __import__("uuid").uuid4().hex
    uploads: list[dict] = []
    for upload in files:
        if not upload.filename:
            continue
        ref, digest, width, height = analysis_service.save_upload(settings, batch_id, upload.filename, upload.file)
        uploads.append({"upload_ref": ref, "sha256": digest, "width": width, "height": height})
    if not uploads:
        raise ApiError(422, "no_files", "上传文件均无文件名")
    batch, job, created_now = analysis_service.create_batch(
        db, settings, user, model_id=model_id, uploads=uploads, idempotency_key=idempotency_key
    )
    _run_inline(settings, job, created_now)
    db.refresh(batch)
    return accepted(
        {**cxr_service.serialize_batch(batch), "job": job.to_public()},
        message="胸片分析批次已接收",
    )


@router.get("/analysis-batches", response_model=None)
def list_batches(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(CxrAnalysisBatch)
    if not user.is_admin:
        query = query.filter(CxrAnalysisBatch.owner_id == user.id)
    total = query.count()
    items = (
        query.order_by(CxrAnalysisBatch.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return ok(
        {
            "items": [cxr_service.serialize_batch(item) for item in items],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    )


@router.get("/analysis-batches/{batch_id}", response_model=None)
def get_batch(
    batch_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    batch = cxr_service.get_batch(db, batch_id, user)
    query = db.query(CxrAnalysis).filter(CxrAnalysis.batch_id == batch.id)
    total = query.count()
    items = (
        query.order_by(CxrAnalysis.created_at.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return ok(
        {
            "batch": cxr_service.serialize_batch(batch),
            "items": [cxr_service.serialize_analysis(item) for item in items],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    )


@router.get("/analyses/{analysis_id}", response_model=None)
def get_analysis(
    analysis_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    analysis = cxr_service.get_analysis(db, analysis_id, user)
    return ok(cxr_service.serialize_analysis(analysis))


@router.get("/analyses/{analysis_id}/image", response_model=None)
def analysis_image(
    analysis_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    """回读分析所用原图；只返回图像字节，不暴露服务器路径。"""
    from fastapi.responses import FileResponse

    analysis = cxr_service.get_analysis(db, analysis_id, user)
    path = analysis_service.resolve_image_path(db, settings, analysis)
    return FileResponse(path, headers={"Cache-Control": "private, no-store"})


@router.post("/analyses/{analysis_id}/explanations", response_model=None)
def create_explanation(
    analysis_id: str,
    payload: CxrExplanationRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=128),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    job, created_now = analysis_service.create_explanation_job(
        db,
        settings,
        user,
        analysis_id=analysis_id,
        label_key=payload.label_key,
        method=payload.method,
        idempotency_key=idempotency_key,
    )
    _run_inline(settings, job, created_now)
    db.refresh(job)
    return accepted(job.to_public(), message="解释生成任务已接收")


@router.post("/reviews", response_model=None)
def create_review(
    payload: CxrReviewCreateRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    review = review_service.create_review(db, user, payload.model_dump())
    return created(cxr_service.serialize_review(review), message="复核草稿已创建")


@router.get("/reviews", response_model=None)
def list_reviews(
    state: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(CxrReview)
    if not user.is_admin:
        query = query.filter(CxrReview.owner_id == user.id)
    if state:
        query = query.filter(CxrReview.state == state)
    total = query.count()
    items = (
        query.order_by(CxrReview.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return ok(
        {
            "items": [cxr_service.serialize_review(item) for item in items],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    )


@router.get("/reviews/{review_id}", response_model=None)
def get_review(
    review_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    review = cxr_service.get_review(db, review_id, user)
    payload = cxr_service.serialize_review(review)
    payload["matrix"] = review_service.label_matrix(review)
    return ok(payload)


@router.patch("/reviews/{review_id}", response_model=None)
def update_review(
    review_id: str,
    payload: CxrReviewUpdateRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    review = cxr_service.get_review(db, review_id, user)
    review = review_service.update_review(db, review, user, payload.model_dump())
    return ok(cxr_service.serialize_review(review), message="复核草稿已保存")


@router.post("/reviews/{review_id}/submit", response_model=None)
def submit_review(
    review_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    review = cxr_service.get_review(db, review_id, user)
    review = review_service.submit_review(db, review, user)
    return ok(cxr_service.serialize_review(review), message="复核已提交")


@router.post("/reviews/{review_id}/decision", response_model=None)
def decide_review(
    review_id: str,
    payload: CxrDecisionRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    review = cxr_service.get_review(db, review_id, user)
    review = review_service.decide_review(db, review, user, approve=payload.approve, reason=payload.reason)
    return ok(cxr_service.serialize_review(review), message="复核已审批")


@router.post("/reports", response_model=None)
def create_report(
    payload: CxrReportRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    report = report_service.create_report(db, settings, user, payload.model_dump())
    return created(cxr_service.serialize_report(report), message="报告快照已生成")


@router.get("/reports", response_model=None)
def list_reports(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(CxrReport)
    if not user.is_admin:
        query = query.filter(CxrReport.owner_id == user.id)
    total = query.count()
    items = (
        query.order_by(CxrReport.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return ok({"items": [cxr_service.serialize_report(item) for item in items], "total": total,
               "page": page, "page_size": page_size})


@router.get("/reports/{report_id}", response_model=None)
def get_report(
    report_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    report = cxr_service.get_report(db, report_id, user)
    return ok({**cxr_service.serialize_report(report), "snapshot": report.snapshot_json})


@router.get("/reports/{report_id}/export", response_model=None)
def export_report(
    report_id: str,
    format: str = Query(default="html"),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    report = cxr_service.get_report(db, report_id, user)
    body, filename, mime = report_service.export(db, settings, report, format)
    return Response(
        content=body,
        media_type=mime,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
