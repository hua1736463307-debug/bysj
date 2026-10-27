"""CXR 独立评价、校准与阈值。

规则（02 契约 G）：
- 逐标签只对 mask=1 样本计算；单类/缺类返回 null + reason，不填 0；
- best checkpoint 只由 val 选；温度与阈值只由 calibration 拟合；
- test 只在模型/预处理/标签策略/阈值方案冻结后读取，且需 admin + purpose；
- 指标不写 map50 字段；dummy 后端指标一律 null + reason。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.core.config import Settings
from app.core.errors import ApiError
from app.core.storage import atomic_write_bytes, ensure_dir
from app.db.base import utcnow
from app.db.models import DatasetVersion, EvaluationRun, JobTask, ModelRegistry
from app.domains import job_service
from app.domains.cxr import service as cxr_service
from app.domains.cxr.runner import run_cxr_subprocess
from app.ml.cxr_adapter import assert_analyzable_model, get_cxr_adapter
from app.ml.cxr_metrics import bootstrap_ci, evaluate_labels, macro
from app.schemas.cxr import (
    CXR_LABEL_SCHEMA,
    CXR_METRICS_SCHEMA,
    CXR_TASK,
    DEFAULT_THRESHOLD,
    LABEL_KEYS,
)

CALIBRATION_ACTIONS = {"calibrate", "evaluate"}
DEFAULT_THRESHOLD_SOURCE = "default_no_calibration"


def _context_hash(*, model_sha: str, manifest_sha: str, split: str, action: str, extra: dict | None = None) -> str:
    payload = {
        "task": CXR_TASK,
        "label_schema": CXR_LABEL_SCHEMA,
        "metrics_schema": CXR_METRICS_SCHEMA,
        "model_sha256": model_sha,
        "manifest_sha256": manifest_sha,
        "split": split,
        "action": action,
        "extra": extra or {},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_request(db: Session, settings: Settings, user: CurrentUser, payload: dict) -> dict:
    model = cxr_service.get_cxr_model(db, payload["model_id"], user)
    assert_analyzable_model(model, settings=settings)
    version = cxr_service.get_cxr_version(db, payload["dataset_version_id"], user)
    if version.state != "frozen":
        raise ApiError(409, "version_not_frozen", "评价只接受已冻结的胸片版本")
    split = payload.get("split") or "val"
    action = payload.get("action") or "evaluate"
    if action == "calibrate" and split != "calibration":
        raise ApiError(422, "cxr_calibration_split", "温度与阈值只能在 calibration 集拟合")
    if split == "test":
        if not user.is_admin:
            raise ApiError(403, "forbidden", "test 评价需要管理员权限")
        if not (payload.get("purpose") or "").strip():
            raise ApiError(422, "cxr_test_purpose_required", "test 评价必须写明预登记目的")
    context = _context_hash(
        model_sha=model.sha256,
        manifest_sha=version.manifest_sha256 or "",
        split=split,
        action=action,
        extra={"purpose": payload.get("purpose") or ""},
    )
    return {
        "model": model,
        "version": version,
        "split": split,
        "action": action,
        "context_hash": context,
    }


def create_evaluation_job(
    db: Session,
    settings: Settings,
    user: CurrentUser,
    payload: dict,
    *,
    idempotency_key: str | None,
) -> tuple[JobTask, bool]:
    if not idempotency_key:
        raise ApiError(422, "idempotency_key_required", "胸片评价必须提供 Idempotency-Key")
    checked = validate_request(db, settings, user, payload)
    run = EvaluationRun(
        owner_id=user.id,
        model_id=checked["model"].id,
        dataset_version_id=checked["version"].id,
        split=checked["split"],
        status="pending",
        purpose=payload.get("purpose"),
        params_json={
            "task": CXR_TASK,
            "label_schema": CXR_LABEL_SCHEMA,
            "metrics_schema": CXR_METRICS_SCHEMA,
            "action": checked["action"],
            "purpose": payload.get("purpose"),
        },
        context_hash=checked["context_hash"],
        approved_by=user.username if checked["split"] == "test" else None,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    job, created = job_service.create_job(
        db,
        job_type="cxr_evaluate",
        owner_id=user.id,
        owner_username=user.username,
        payload={
            "evaluation_id": run.id,
            "model_id": checked["model"].id,
            "dataset_version_id": checked["version"].id,
            "split": checked["split"],
            "action": checked["action"],
            "context_hash": checked["context_hash"],
            "task": CXR_TASK,
            "label_schema": CXR_LABEL_SCHEMA,
        },
        idempotency_key=idempotency_key,
    )
    if run.job_id != job.id:
        run.job_id = job.id
        db.commit()
    return job, created


def latest_calibration(db: Session, *, model_id: str, dataset_version_id: str) -> dict | None:
    """只读已完成的 calibration 产物；换模型必须重新拟合。"""
    runs = (
        db.query(EvaluationRun)
        .filter(
            EvaluationRun.model_id == model_id,
            EvaluationRun.dataset_version_id == dataset_version_id,
            EvaluationRun.split == "calibration",
            EvaluationRun.status == "completed",
        )
        .order_by(EvaluationRun.created_at.desc())
        .all()
    )
    for run in runs:
        params = run.params_json or {}
        if params.get("action") != "calibrate":
            continue
        artifacts = (run.artifacts_json or {}).get("calibration") or {}
        if artifacts.get("artifact_id"):
            return {
                "evaluation_id": run.id,
                "artifact_id": artifacts["artifact_id"],
                "sha256": artifacts.get("sha256"),
                "temperatures": artifacts.get("temperatures") or {},
                "thresholds": artifacts.get("thresholds") or {},
                "threshold_sources": artifacts.get("threshold_sources") or {},
                "calibration_status": artifacts.get("calibration_status") or {},
                "model_sha256": artifacts.get("model_sha256"),
                "manifest_sha256": artifacts.get("manifest_sha256"),
            }
    return None


def thresholds_for(db: Session, *, model_id: str, dataset_version_id: str) -> tuple[dict, str, str | None]:
    calibration = latest_calibration(db, model_id=model_id, dataset_version_id=dataset_version_id)
    if calibration is None:
        return {key: DEFAULT_THRESHOLD for key in LABEL_KEYS}, DEFAULT_THRESHOLD_SOURCE, None
    thresholds = {key: float((calibration["thresholds"] or {}).get(key, DEFAULT_THRESHOLD)) for key in LABEL_KEYS}
    sources = calibration["threshold_sources"] or {}
    primary = sorted({str(value) for value in sources.values()}) or [DEFAULT_THRESHOLD_SOURCE]
    return thresholds, ",".join(primary), calibration["artifact_id"]


def compute_metrics(
    *,
    scores: dict[str, list[float]],
    truth: dict[str, list[float]],
    mask: dict[str, list[int]],
    groups: list[str],
    thresholds: dict[str, float],
    bootstrap_rounds: int = 1000,
    bootstrap_seed: int = 42,
) -> dict:
    """纯函数：从原始评分算逐标签指标，供 worker 与测试共用。"""
    per_label = evaluate_labels(truth, scores, mask, thresholds)
    for key in LABEL_KEYS:
        from app.ml.cxr_metrics import auroc

        per_label[key]["auroc_ci"] = bootstrap_ci(
            groups, truth[key], scores[key], mask[key], auroc,
            rounds=bootstrap_rounds, seed=bootstrap_seed,
        )
    return {
        "metrics_schema": CXR_METRICS_SCHEMA,
        "label_schema": CXR_LABEL_SCHEMA,
        "per_label": per_label,
        "macro_auroc": macro(per_label, "auroc"),
        "macro_auprc": macro(per_label, "auprc"),
        "macro_brier": macro(per_label, "brier"),
        "macro_ece": macro(per_label, "ece"),
        "thresholds": {key: float(thresholds.get(key, DEFAULT_THRESHOLD)) for key in LABEL_KEYS},
        "bootstrap": {"unit": "patient_component", "rounds": bootstrap_rounds, "seed": bootstrap_seed},
        "note": "逐标签只对 mask=1 样本计算；uncertain/unmentioned 不参与监督，也不被当作阴性。",
    }


def _dummy_metrics(reason: str) -> dict:
    return {
        "metrics_schema": CXR_METRICS_SCHEMA,
        "label_schema": CXR_LABEL_SCHEMA,
        "per_label": {
            key: {
                "auroc": None, "auroc_reason": reason,
                "auprc": None, "auprc_reason": reason,
                "brier": None, "brier_reason": reason,
                "ece": None, "ece_reason": reason,
                "binary": {"threshold": DEFAULT_THRESHOLD, "support": 0, "reason": reason},
                "masked_samples": 0,
            }
            for key in LABEL_KEYS
        },
        "macro_auroc": {"value": None, "valid_labels": 0, "total_labels": len(LABEL_KEYS), "reason": reason},
        "macro_auprc": {"value": None, "valid_labels": 0, "total_labels": len(LABEL_KEYS), "reason": reason},
        "macro_brier": {"value": None, "valid_labels": 0, "total_labels": len(LABEL_KEYS), "reason": reason},
        "macro_ece": {"value": None, "valid_labels": 0, "total_labels": len(LABEL_KEYS), "reason": reason},
        "thresholds": {key: DEFAULT_THRESHOLD for key in LABEL_KEYS},
        "research_valid": False,
        "note": reason,
    }


def _calibrated_scores(scores: dict[str, list[float]], temperatures: dict[str, float]) -> dict[str, list[float]]:
    import math

    output: dict[str, list[float]] = {}
    for key, values in scores.items():
        temperature = float(temperatures.get(key) or 1.0)
        converted = []
        for value in values:
            clipped = min(max(float(value), 1e-6), 1 - 1e-6)
            logit = math.log(clipped / (1 - clipped))
            converted.append(1.0 / (1.0 + math.exp(-logit / temperature)))
        output[key] = converted
    return output


def run_evaluation(
    db: Session,
    settings: Settings,
    *,
    job: JobTask,
    workdir: Path,
    should_stop=None,
    progress=None,
    attempt: int = 1,
    lock_for_write=None,
) -> dict:
    payload = job.input_json or {}
    run = db.get(EvaluationRun, payload["evaluation_id"])
    if run is None:
        raise ApiError(404, "not_found", "评价记录不存在")
    model = db.get(ModelRegistry, run.model_id)
    version = db.get(DatasetVersion, run.dataset_version_id)
    if model is None or version is None:
        raise ApiError(404, "not_found", "模型或数据版本不存在")
    assert_analyzable_model(model, settings=settings)

    # 相同协议且已完成：直接复用，不重复消耗 GPU。
    existing = (
        db.query(EvaluationRun)
        .filter(
            EvaluationRun.id != run.id,
            EvaluationRun.model_id == run.model_id,
            EvaluationRun.dataset_version_id == run.dataset_version_id,
            EvaluationRun.split == run.split,
            EvaluationRun.context_hash == run.context_hash,
            EvaluationRun.status == "completed",
        )
        .order_by(EvaluationRun.created_at.asc())
        .first()
    )
    if existing is not None:
        run.status = "completed"
        run.metrics_json = existing.metrics_json
        run.artifacts_json = {**(existing.artifacts_json or {}), "reused_from": existing.id}
        run.finished_at = utcnow()
        db.commit()
        return {"evaluation_id": run.id, "reused_from": existing.id, "metrics": run.metrics_json}

    adapter = get_cxr_adapter(settings)
    action = payload.get("action") or "evaluate"
    split = payload.get("split") or run.split
    request = {
        "task": "calibrate" if action == "calibrate" else "score",
        "model_path": model.path,
        "dataset_root": str(version.root_path),
        "manifest": str(version.manifest_path),
        "split": split,
        "batch": 8,
        "image_size": int((model.metadata_json or {}).get("image_size") or 224),
        "label_schema": CXR_LABEL_SCHEMA,
    }
    if progress:
        progress(0.2, "scoring", f"在 {split} 集计算胸片评分")

    if adapter.backend == "dummy":
        result = {
            "backend": "dummy",
            "split": split,
            "samples": 0,
            "scores": {key: [] for key in LABEL_KEYS},
            "truth": {key: [] for key in LABEL_KEYS},
            "mask": {key: [] for key in LABEL_KEYS},
            "groups": [],
        }
    else:
        result = run_cxr_subprocess(request, workdir, should_stop=should_stop, progress=progress)

    if lock_for_write:
        lock_for_write()
    artifact_dir = ensure_dir(Path(settings.artifact_root) / "cxr" / "evaluations" / run.id / str(attempt))
    artifacts: dict = {
        "backend": result.get("backend"),
        "model_sha256": model.sha256,
        "manifest_sha256": version.manifest_sha256,
        "split": split,
        "action": action,
        "label_schema": CXR_LABEL_SCHEMA,
        "environment": {"backend": result.get("backend")},
    }

    if adapter.backend == "dummy":
        run.metrics_json = _dummy_metrics("dummy_backend：工程契约后端不产生真实指标")
        run.artifacts_json = artifacts
        run.context_hash = run.context_hash or _context_hash(
            model_sha=model.sha256, manifest_sha=version.manifest_sha256 or "", split=split, action=action
        )
        run.status = "completed"
        run.finished_at = utcnow()
        db.commit()
        return {"evaluation_id": run.id, "metrics": run.metrics_json, "backend": "dummy"}

    scores = result["scores"]
    truth = result["truth"]
    mask = result["mask"]
    groups = result["groups"]

    if action == "calibrate":
        temperatures = {key: float((result.get("per_label") or {}).get(key, {}).get("temperature", 1.0)) for key in LABEL_KEYS}
        thresholds = {key: float((result.get("per_label") or {}).get(key, {}).get("threshold", DEFAULT_THRESHOLD)) for key in LABEL_KEYS}
        threshold_sources = {key: str((result.get("per_label") or {}).get(key, {}).get("threshold_selection", {}).get("source", DEFAULT_THRESHOLD_SOURCE)) for key in LABEL_KEYS}
        calibration_status = {
            key: str((result.get("per_label") or {}).get(key, {}).get("calibration", {}).get("status", "not_calibrated"))
            for key in LABEL_KEYS
        }
        payload_calibration = {
            "schema": "cxr-calibration-v1",
            "model_id": model.id,
            "model_sha256": model.sha256,
            "dataset_version_id": version.id,
            "manifest_sha256": version.manifest_sha256,
            "split_manifest_hash": version.split_manifest_hash,
            "split": "calibration",
            "label_schema": CXR_LABEL_SCHEMA,
            "temperatures": temperatures,
            "thresholds": thresholds,
            "threshold_sources": threshold_sources,
            "calibration_status": calibration_status,
            "per_label": result.get("per_label") or {},
            "samples": result.get("samples"),
            "fit_code": "app/ml/cxr_metrics.py:fit_temperature+select_threshold",
            "fit_code_sha256": cxr_service.file_sha256(Path(__file__).resolve().parents[2] / "ml" / "cxr_metrics.py"),
            "bootstrap": {"unit": "patient_component", "rounds": 1000, "seed": 42},
        }
        artifact_path = artifact_dir / "calibration.json"
        atomic_write_bytes(
            artifact_path,
            json.dumps(payload_calibration, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
        )
        record = cxr_service.record_artifact(
            db, settings, owner_id=run.owner_id, resource_type="cxr_calibration",
            resource_id=run.id, path=artifact_path, mime="application/json",
        )
        calibrated = _calibrated_scores(scores, temperatures)
        metrics = compute_metrics(
            scores=calibrated, truth=truth, mask=mask, groups=groups, thresholds=thresholds
        )
        metrics["calibration_status"] = calibration_status
        metrics["temperatures"] = temperatures
        metrics["evaluated_on"] = "calibrated_scores"
        artifacts["calibration"] = {
            "artifact_id": record.id,
            "sha256": record.sha256,
            "temperatures": temperatures,
            "thresholds": thresholds,
            "threshold_sources": threshold_sources,
            "calibration_status": calibration_status,
            "model_sha256": model.sha256,
            "manifest_sha256": version.manifest_sha256,
        }
        run.metrics_json = metrics
        run.artifacts_json = artifacts
        run.status = "completed"
        run.finished_at = utcnow()
        db.commit()
        if progress:
            progress(1.0, "completed", "校准与阈值已生成")
        return {
            "evaluation_id": run.id,
            "temperatures": temperatures,
            "thresholds": thresholds,
            "calibration_status": calibration_status,
            "metrics": metrics,
        }

    # action=evaluate：使用已存在的校准/阈值产物；没有就显式回退到 0.5。
    thresholds, threshold_source, calibration_artifact_id = thresholds_for(
        db, model_id=model.id, dataset_version_id=version.id
    )
    calibration = latest_calibration(db, model_id=model.id, dataset_version_id=version.id)
    evaluated_scores = scores
    evaluated_on = "raw_scores"
    if calibration is not None:
        temperatures = {key: float((calibration["temperatures"] or {}).get(key, 1.0)) for key in LABEL_KEYS}
        evaluated_scores = _calibrated_scores(scores, temperatures)
        evaluated_on = "calibrated_scores"
    metrics = compute_metrics(
        scores=evaluated_scores, truth=truth, mask=mask, groups=groups, thresholds=thresholds
    )
    metrics["threshold_source"] = threshold_source
    metrics["calibration_artifact_id"] = calibration_artifact_id
    metrics["evaluated_on"] = evaluated_on
    metrics["calibration_status"] = (
        calibration["calibration_status"] if calibration else {key: "not_calibrated" for key in LABEL_KEYS}
    )
    artifacts["calibration_artifact_id"] = calibration_artifact_id
    artifacts["threshold_source"] = threshold_source

    scores_path = artifact_dir / "scores.json"
    atomic_write_bytes(
        scores_path,
        json.dumps(
            {
                "schema": "cxr-evaluation-scores-v1",
                "scores": evaluated_scores,
                "raw_scores": scores,
                "truth": truth,
                "mask": mask,
                "groups": groups,
            },
            ensure_ascii=False,
        ).encode("utf-8"),
    )
    record = cxr_service.record_artifact(
        db, settings, owner_id=run.owner_id, resource_type="cxr_evaluation",
        resource_id=run.id, path=scores_path, mime="application/json",
    )
    artifacts["scores"] = {"artifact_id": record.id, "sha256": record.sha256}
    run.metrics_json = metrics
    run.artifacts_json = artifacts
    run.status = "completed"
    run.finished_at = utcnow()
    db.commit()
    if progress:
        progress(1.0, "completed", f"{split} 评价完成")
    return {"evaluation_id": run.id, "split": split, "metrics": metrics}


def summary_metrics(metrics: dict) -> dict:
    """对外摘要：只给总体与逐标签关键指标，不伪造缺失项。"""
    per_label = metrics.get("per_label") or {}
    return {
        "macro_auroc": metrics.get("macro_auroc"),
        "macro_auprc": metrics.get("macro_auprc"),
        "per_label": {
            key: {
                "auroc": (per_label.get(key) or {}).get("auroc"),
                "auprc": (per_label.get(key) or {}).get("auprc"),
                "sensitivity": ((per_label.get(key) or {}).get("binary") or {}).get("sensitivity"),
                "specificity": ((per_label.get(key) or {}).get("binary") or {}).get("specificity"),
                "support": (per_label.get(key) or {}).get("masked_samples"),
            }
            for key in LABEL_KEYS
        },
        "thresholds": metrics.get("thresholds"),
        "research_valid": metrics.get("research_valid", True),
    }
