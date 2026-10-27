"""CXR 域共享查询、序列化与任务类型守卫。"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.core.config import Settings
from app.core.errors import ApiError
from app.core.storage import sha256_file
from app.db.models import (
    CxrAnalysis,
    CxrAnalysisBatch,
    CxrReport,
    CxrReview,
    DatasetResource,
    DatasetVersion,
    ModelRegistry,
)
from app.domains.datasets import service as dataset_service
from app.schemas.cxr import CXR_LABEL_SCHEMA, CXR_TASK, build_label_results


def get_cxr_dataset(db: Session, dataset_id: str, user: CurrentUser) -> DatasetResource:
    resource = dataset_service.get_dataset(db, dataset_id, user)
    if resource.task_type != CXR_TASK:
        raise ApiError(
            409,
            "task_mismatch",
            "该数据集不是胸片五标签数据集，不能用于 CXR 分析",
            {"task_type": resource.task_type, "expected": CXR_TASK},
        )
    return resource


def get_cxr_version(db: Session, version_id: str, user: CurrentUser) -> DatasetVersion:
    version = dataset_service.get_version(db, version_id, user)
    resource = dataset_service.get_dataset(db, version.dataset_id, user)
    if resource.task_type != CXR_TASK:
        raise ApiError(
            409,
            "task_mismatch",
            "该版本不是胸片五标签版本",
            {"task_type": resource.task_type, "expected": CXR_TASK},
        )
    return version


def manifest_path(version: DatasetVersion) -> Path:
    if not version.manifest_path:
        raise ApiError(409, "version_not_frozen", "版本还没有 CXR manifest，请先冻结")
    path = Path(version.manifest_path)
    if not path.is_file():
        raise ApiError(409, "manifest_missing", "版本 manifest 文件缺失")
    return path


def load_manifest(version: DatasetVersion) -> dict:
    path = manifest_path(version)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("label_schema") != CXR_LABEL_SCHEMA:
        raise ApiError(409, "cxr_schema_mismatch", "manifest 标签 schema 与当前契约不一致")
    return payload


def get_analysis(db: Session, analysis_id: str, user: CurrentUser) -> CxrAnalysis:
    analysis = db.get(CxrAnalysis, analysis_id)
    if analysis is None:
        raise ApiError(404, "not_found", "分析记录不存在")
    if not user.is_admin and analysis.owner_id != user.id:
        raise ApiError(404, "not_found", "分析记录不存在")
    return analysis


def get_batch(db: Session, batch_id: str, user: CurrentUser) -> CxrAnalysisBatch:
    batch = db.get(CxrAnalysisBatch, batch_id)
    if batch is None:
        raise ApiError(404, "not_found", "分析批次不存在")
    if not user.is_admin and batch.owner_id != user.id:
        raise ApiError(404, "not_found", "分析批次不存在")
    return batch


def get_review(db: Session, review_id: str, user: CurrentUser) -> CxrReview:
    review = db.get(CxrReview, review_id)
    if review is None:
        raise ApiError(404, "not_found", "复核记录不存在")
    if not user.is_admin and review.owner_id != user.id:
        raise ApiError(404, "not_found", "复核记录不存在")
    return review


def get_report(db: Session, report_id: str, user: CurrentUser) -> CxrReport:
    report = db.get(CxrReport, report_id)
    if report is None:
        raise ApiError(404, "not_found", "报告不存在")
    if not user.is_admin and report.owner_id != user.id:
        raise ApiError(404, "not_found", "报告不存在")
    return report


def get_cxr_model(db: Session, model_id: str, user: CurrentUser) -> ModelRegistry:
    model = db.get(ModelRegistry, model_id)
    if model is None or model.status != "active":
        raise ApiError(404, "not_found", "模型不存在")
    if not user.is_admin and model.owner_id != user.id and model.trusted_source != "base":
        raise ApiError(404, "not_found", "模型不存在")
    return model


def get_asset_for_analysis(db: Session, asset_id: int, user: CurrentUser):
    """胸片分析只接受属于 CXR 版本的资产；跨任务/跨用户统一 404/409。"""
    from app.db.models import DatasetAsset

    asset = db.get(DatasetAsset, asset_id)
    if asset is None or asset.deleted_at is not None:
        raise ApiError(404, "not_found", "图像资产不存在")
    version = get_cxr_version(db, asset.dataset_version_id, user)
    if asset.dataset_version_id != version.id:
        raise ApiError(404, "not_found", "图像资产不存在")
    return asset


def record_artifact(db: Session, settings: Settings, *, owner_id: int | None, resource_type: str,
                    resource_id: str, path: Path, mime: str):
    from app.domains.inference.service import record_artifact as _record

    return _record(
        db, settings, owner_id=owner_id, resource_type=resource_type,
        resource_id=resource_id, path=path, mime=mime,
    )


def analysis_labels(analysis: CxrAnalysis) -> list[dict]:
    return build_label_results(
        analysis.raw_scores_json or {},
        analysis.calibrated_scores_json,
        analysis.thresholds_json,
        calibration_status=analysis.calibration_status,
        threshold_source=analysis.threshold_source,
    )


def serialize_analysis(analysis: CxrAnalysis, *, include_labels: bool = True) -> dict:
    payload = {
        "analysis_id": analysis.id,
        "batch_id": analysis.batch_id,
        "asset_id": analysis.asset_id,
        "image": {
            "sha256": analysis.image_sha256,
            "width": analysis.image_width,
            "height": analysis.image_height,
        },
        "model": {
            "model_id": analysis.model_id,
            "sha256": analysis.model_sha256,
            "backend": analysis.backend,
        },
        "task": analysis.task,
        "label_schema": analysis.label_schema,
        "calibration": {
            "status": analysis.calibration_status,
            "threshold_source": analysis.threshold_source,
            "artifact_id": analysis.calibration_artifact_id,
        },
        "quality": analysis.quality_json or {},
        "explanation": analysis.explanation_json or {},
        "explanation_artifact_id": analysis.explanation_artifact_id,
        "status": analysis.status,
        "attempt": analysis.attempt,
        "error": analysis.error,
        "timing": analysis.timing_json or {},
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
        "finished_at": analysis.finished_at.isoformat() if analysis.finished_at else None,
    }
    if include_labels:
        payload["labels"] = analysis_labels(analysis)
    return payload


def serialize_batch(batch: CxrAnalysisBatch) -> dict:
    return {
        "batch_id": batch.id,
        "job_id": batch.job_id,
        "model_id": batch.model_id,
        "model_sha256": batch.model_sha256,
        "status": batch.status,
        "total": batch.total,
        "completed": batch.completed,
        "failed": batch.failed,
        "error": {"code": batch.error_code, "detail": batch.error_detail} if batch.error_code else None,
        "created_at": batch.created_at.isoformat() if batch.created_at else None,
        "finished_at": batch.finished_at.isoformat() if batch.finished_at else None,
    }


def serialize_review(review: CxrReview) -> dict:
    return {
        "review_id": review.id,
        "analysis_id": review.analysis_id,
        "state": review.state,
        "revision": review.revision,
        "reviewer": review.reviewer_username or None,
        "items": review.items_json or [],
        "model_snapshot": review.model_snapshot_json or {},
        "reason": review.reason,
        "created_at": review.created_at.isoformat() if review.created_at else None,
        "updated_at": review.updated_at.isoformat() if review.updated_at else None,
        "submitted_at": review.submitted_at.isoformat() if review.submitted_at else None,
        "decided_at": review.decided_at.isoformat() if review.decided_at else None,
        "decided_by": review.decided_by_username,
    }


def serialize_report(report: CxrReport) -> dict:
    return {
        "report_id": report.id,
        "analysis_id": report.analysis_id,
        "review_id": report.review_id,
        "review_revision": report.review_revision,
        "state": report.state,
        "version": report.version,
        "supersedes_id": report.supersedes_id,
        "snapshot_sha256": report.snapshot_sha256,
        "artifact_id": report.artifact_id,
        "issued_by": report.issued_by_username,
        "issued_at": report.issued_at.isoformat() if report.issued_at else None,
        "created_at": report.created_at.isoformat() if report.created_at else None,
    }


def verify_model_hash(model: ModelRegistry, settings: Settings) -> Path:
    from app.domains.models import service as model_service

    return model_service.verify_model_file(model, settings)


def file_sha256(path: Path) -> str:
    return sha256_file(path)
