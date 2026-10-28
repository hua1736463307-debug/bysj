"""CXR 分析批次：上传/受控资产 → 五标签评分 → 批次与逐项状态。

约束：
- 一图一 analysis，不把同患者多图擅自融合；
- 单批最多 `max_batch_inference`（当前 100），不静默截断；
- 一张失败不丢弃其他成功项；
- 无可用模型时明确 `model_unavailable` / `unsupported_task`，不自动用骨干替代；
- 输出五项完整结果，不是只返回 top1。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.core.config import Settings
from app.core.errors import ApiError
from app.core.storage import atomic_write_bytes, ensure_dir, resolve_under
from app.db.base import utcnow
from app.db.models import (
    CxrAnalysis,
    CxrAnalysisBatch,
    CxrSample,
    DatasetAsset,
    DatasetVersion,
    JobTask,
    ModelRegistry,
)
from app.domains import job_service
from app.domains.cxr import service as cxr_service
from app.domains.cxr.evaluation_service import latest_calibration, thresholds_for
from app.domains.cxr.runner import run_cxr_subprocess
from app.ml.cxr_adapter import assert_analyzable_model, get_cxr_adapter
from app.schemas.cxr import (
    CXR_LABEL_SCHEMA,
    CXR_TASK,
    DEFAULT_THRESHOLD,
    LABEL_KEYS,
    build_label_results,
)


def _analysis_root(settings: Settings) -> Path:
    return ensure_dir(Path(settings.data_root) / "cxr" / "uploads")


def save_upload(settings: Settings, batch_id: str, filename: str, stream) -> tuple[str, str, int, int]:
    """保存上传图到受控目录，返回 (相对引用, sha256, 宽, 高)。"""
    root = ensure_dir(_analysis_root(settings) / batch_id)
    data = stream.read()
    if not data:
        raise ApiError(422, "empty_upload", "上传文件为空")
    if len(data) > settings.max_image_bytes:
        raise ApiError(413, "image_too_large", "图像超过大小上限")
    digest = hashlib.sha256(data).hexdigest()
    suffix = Path(filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ApiError(415, "unsupported_format", "首期仅支持 JPG/PNG 胸片")
    target = root / f"{digest[:16]}{suffix}"
    atomic_write_bytes(target, data)
    try:
        with Image.open(target) as image:
            image.load()
            if (image.format or "").upper() not in {"JPEG", "PNG"}:
                raise ApiError(415, "unsupported_format", "首期仅支持 8bit JPG/PNG")
            width, height = image.size
    except ApiError:
        target.unlink(missing_ok=True)
        raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise ApiError(422, "image_unreadable", "上传图像无法解码") from exc
    if width * height > settings.max_image_pixels:
        target.unlink(missing_ok=True)
        raise ApiError(413, "image_too_large", "图像像素数超过上限")
    return f"{batch_id}/{target.name}", digest, width, height


def create_batch(
    db: Session,
    settings: Settings,
    user: CurrentUser,
    *,
    model_id: str,
    asset_ids: list[int] | None = None,
    uploads: list[dict] | None = None,
    idempotency_key: str | None = None,
) -> tuple[CxrAnalysisBatch, JobTask, bool]:
    if not idempotency_key:
        raise ApiError(422, "idempotency_key_required", "胸片分析必须提供 Idempotency-Key")
    model = cxr_service.get_cxr_model(db, model_id, user)
    assert_analyzable_model(model, settings=settings)
    asset_ids = asset_ids or []
    uploads = uploads or []
    if bool(asset_ids) == bool(uploads):
        raise ApiError(422, "cxr_input_conflict", "asset_ids 与上传文件必须二选一")
    total = len(asset_ids) or len(uploads)
    if total == 0:
        raise ApiError(422, "cxr_no_input", "至少提供一张胸片")
    if total > settings.max_batch_inference:
        raise ApiError(422, "cxr_batch_too_large", f"单批最多 {settings.max_batch_inference} 张，未截断输入")

    request_payload = {
        "model_id": model.id,
        "asset_ids": sorted(int(item) for item in asset_ids),
        "uploads": [item["sha256"] for item in uploads],
    }
    request_hash = hashlib.sha256(
        json.dumps(request_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    existing = (
        db.query(CxrAnalysisBatch)
        .filter(CxrAnalysisBatch.owner_id == user.id, CxrAnalysisBatch.idempotency_key == idempotency_key)
        .one_or_none()
    )
    if existing is not None:
        if existing.request_hash != request_hash:
            raise ApiError(409, "idempotency_conflict", "相同幂等键对应了不同分析请求")
        job = db.get(JobTask, existing.job_id) if existing.job_id else None
        if job is None:
            raise ApiError(409, "cxr_batch_inconsistent", "批次缺少关联作业，请人工核对")
        return existing, job, False

    batch = CxrAnalysisBatch(
        owner_id=user.id,
        owner_username=user.username,
        model_id=model.id,
        model_sha256=model.sha256,
        request_hash=request_hash,
        idempotency_key=idempotency_key,
        status="pending",
        total=total,
    )
    db.add(batch)
    db.flush()

    if asset_ids:
        for asset_id in asset_ids:
            asset = cxr_service.get_asset_for_analysis(db, asset_id, user)
            db.add(
                CxrAnalysis(
                    batch_id=batch.id,
                    owner_id=user.id,
                    asset_id=asset.id,
                    item_key=f"asset:{asset.id}",
                    image_sha256=asset.content_hash,
                    image_width=asset.width,
                    image_height=asset.height,
                    model_id=model.id,
                    model_sha256=model.sha256,
                    task=CXR_TASK,
                    label_schema=CXR_LABEL_SCHEMA,
                    status="pending",
                    attempt=1,
                )
            )
    else:
        # 同批重复上传同一文件只保留一条分析，避免唯一约束冲突。
        seen: set[str] = set()
        for upload in uploads:
            if upload["sha256"] in seen:
                continue
            seen.add(upload["sha256"])
            db.add(
                CxrAnalysis(
                    batch_id=batch.id,
                    owner_id=user.id,
                    asset_id=None,
                    item_key=f"upload:{upload['sha256']}",
                    upload_ref=upload["upload_ref"],
                    image_sha256=upload["sha256"],
                    image_width=upload.get("width"),
                    image_height=upload.get("height"),
                    model_id=model.id,
                    model_sha256=model.sha256,
                    task=CXR_TASK,
                    label_schema=CXR_LABEL_SCHEMA,
                    status="pending",
                    attempt=1,
                )
            )
        batch.total = len(seen)
    db.commit()

    job, created = job_service.create_job(
        db,
        job_type="cxr_analyze",
        owner_id=user.id,
        owner_username=user.username,
        payload={
            "batch_id": batch.id,
            "model_id": model.id,
            "task": CXR_TASK,
            "label_schema": CXR_LABEL_SCHEMA,
        },
        idempotency_key=idempotency_key,
    )
    if batch.job_id != job.id:
        batch.job_id = job.id
        db.commit()
    db.refresh(batch)
    return batch, job, created


def resolve_image_path(db: Session, settings: Settings, analysis: CxrAnalysis) -> Path:
    if analysis.upload_ref:
        return resolve_under(_analysis_root(settings), analysis.upload_ref, must_exist=True)
    if analysis.asset_id is None:
        raise ApiError(409, "cxr_input_missing", "分析记录没有输入图像")
    asset = db.get(DatasetAsset, analysis.asset_id)
    if asset is None or asset.deleted_at is not None:
        raise ApiError(404, "not_found", "图像资产不存在或已删除")
    version = db.get(DatasetVersion, asset.dataset_version_id)
    if version is None:
        raise ApiError(404, "not_found", "图像所属版本不存在")
    root = Path(version.root_path).resolve()
    try:
        root.relative_to(Path(settings.data_root).resolve())
    except ValueError as exc:
        raise ApiError(409, "invalid_path", "版本目录超出数据目录") from exc
    return resolve_under(root / "images", asset.relative_path, must_exist=True)


def run_batch(
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
    batch = db.get(CxrAnalysisBatch, payload["batch_id"])
    if batch is None:
        raise ApiError(404, "not_found", "分析批次不存在")
    model = db.get(ModelRegistry, batch.model_id)
    if model is None:
        raise ApiError(404, "not_found", "模型不存在")
    assert_analyzable_model(model, settings=settings)

    thresholds, threshold_source, calibration_artifact_id = thresholds_for(
        db, model_id=model.id, dataset_version_id=_version_hint(db, batch)
    )
    calibration = latest_calibration(db, model_id=model.id, dataset_version_id=_version_hint(db, batch))
    temperatures = {key: float(((calibration or {}).get("temperatures") or {}).get(key, 1.0)) for key in LABEL_KEYS}
    calibration_status = (
        (calibration or {}).get("calibration_status")
        or {key: "not_calibrated" for key in LABEL_KEYS}
    )

    items = (
        db.query(CxrAnalysis)
        .filter(CxrAnalysis.batch_id == batch.id, CxrAnalysis.status.in_(["pending", "failed"]))
        .order_by(CxrAnalysis.created_at.asc())
        .all()
    )
    adapter = get_cxr_adapter(settings)
    backend = adapter.backend
    queue_ms = max(0, (job.started_at - job.created_at).total_seconds() * 1000) if job.started_at else None

    # 一次加载模型逐项处理（warm 样本）；dummy 也走同一契约路径。
    handle = adapter.load(Path(model.path), model.metadata_json or {})
    try:
        for index, item in enumerate(items, start=1):
            if should_stop and should_stop():
                from app.workers.dispatcher import JobCancelled

                raise JobCancelled()
            started = time.perf_counter()
            try:
                source = resolve_image_path(db, settings, item)
                prediction = adapter.predict(handle, source)
            except Exception as exc:
                if lock_for_write:
                    lock_for_write()
                item.status = "failed"
                item.error = f"{type(exc).__name__}: {exc}"[:1000]
                item.finished_at = utcnow()
                batch.failed = int(batch.failed or 0) + 1
                db.commit()
                if progress:
                    progress(index / max(1, len(items)), "analyzing", f"第 {index} 张失败")
                continue

            raw_scores = {key: float(prediction["raw_scores"].get(key, 0.0)) for key in LABEL_KEYS}
            calibrated_scores = None
            if calibration is not None:
                calibrated_scores = _apply_temperature(raw_scores, temperatures)
            effective = calibrated_scores or raw_scores
            labels = build_label_results(
                raw_scores,
                calibrated_scores,
                thresholds,
                calibration_status="temperature_scaled" if calibration is not None else "not_calibrated",
                threshold_source=threshold_source,
            )
            if lock_for_write:
                lock_for_write()
            item.raw_scores_json = raw_scores
            item.calibrated_scores_json = calibrated_scores
            item.thresholds_json = thresholds
            item.calibration_status = "temperature_scaled" if calibration is not None else "not_calibrated"
            item.threshold_source = threshold_source
            item.calibration_artifact_id = calibration_artifact_id
            item.quality_json = prediction.get("quality") or {}
            item.backend = backend
            item.status = "completed"
            item.error = None
            item.finished_at = utcnow()
            item.timing_json = {
                "queue_ms": queue_ms,
                "processing_ms": round((time.perf_counter() - started) * 1000, 3),
                **(prediction.get("timing") or {}),
                "phase": "cold_model" if index == 1 else "warm_model",
            }
            item.explanation_json = {
                **(item.explanation_json or {}),
                "labels": labels,
                "limitations": prediction.get("limitations") or [],
                "score_semantics": "模型评分，不是临床确诊概率；未超过阈值不等于排除该征象或其他疾病",
            }
            batch.completed = int(batch.completed or 0) + 1
            db.commit()
            if progress:
                progress(index / max(1, len(items)), "analyzing", f"分析 {index}/{len(items)}")
    finally:
        adapter.release(handle)

    if lock_for_write:
        lock_for_write()
    batch.completed = (
        db.query(CxrAnalysis)
        .filter(CxrAnalysis.batch_id == batch.id, CxrAnalysis.status == "completed")
        .count()
    )
    batch.failed = (
        db.query(CxrAnalysis)
        .filter(CxrAnalysis.batch_id == batch.id, CxrAnalysis.status == "failed")
        .count()
    )
    batch.status = "partial" if int(batch.failed or 0) else "completed"
    batch.finished_at = utcnow()
    db.commit()
    return {
        "batch_id": batch.id,
        "total": batch.total,
        "completed": batch.completed,
        "failed": batch.failed,
        "status": batch.status,
        "backend": backend,
    }


def _apply_temperature(scores: dict[str, float], temperatures: dict[str, float]) -> dict[str, float]:
    import math

    output: dict[str, float] = {}
    for key, value in scores.items():
        temperature = float(temperatures.get(key) or 1.0)
        clipped = min(max(float(value), 1e-6), 1 - 1e-6)
        logit = math.log(clipped / (1 - clipped))
        output[key] = float(1.0 / (1.0 + math.exp(-logit / temperature)))
    return output


def _version_hint(db: Session, batch: CxrAnalysisBatch) -> str:
    """校准产物按 模型+数据版本 绑定；上传图没有版本时用空串（即只用默认阈值）。"""
    sample = (
        db.query(CxrAnalysis)
        .filter(CxrAnalysis.batch_id == batch.id, CxrAnalysis.asset_id.isnot(None))
        .first()
    )
    if sample is None or sample.asset_id is None:
        return ""
    asset = db.get(DatasetAsset, sample.asset_id)
    if asset is None:
        return ""
    cxr_sample = db.query(CxrSample).filter(CxrSample.asset_id == asset.id).one_or_none()
    if cxr_sample is None:
        return ""
    return cxr_sample.dataset_version_id


def create_explanation_job(
    db: Session,
    settings: Settings,
    user: CurrentUser,
    *,
    analysis_id: str,
    label_key: str,
    method: str = "gradcam",
    idempotency_key: str | None = None,
) -> tuple[JobTask, bool]:
    analysis = cxr_service.get_analysis(db, analysis_id, user)
    if analysis.status != "completed":
        raise ApiError(409, "cxr_analysis_not_ready", "只有已完成的分析才能生成解释")
    model = db.get(ModelRegistry, analysis.model_id)
    if model is None:
        raise ApiError(404, "not_found", "模型不存在")
    assert_analyzable_model(model, settings=settings)
    key = idempotency_key or hashlib.sha256(
        f"{analysis.id}:{label_key}:{method}:{model.sha256}".encode("utf-8")
    ).hexdigest()[:64]
    return job_service.create_job(
        db,
        job_type="cxr_explain",
        owner_id=user.id,
        owner_username=user.username,
        payload={
            "analysis_id": analysis.id,
            "label_key": label_key,
            "method": method,
            "model_id": model.id,
            "model_sha256": model.sha256,
            "image_sha256": analysis.image_sha256,
            "task": CXR_TASK,
        },
        idempotency_key=key,
    )


def run_explanation(
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
    analysis = db.get(CxrAnalysis, payload["analysis_id"])
    if analysis is None:
        raise ApiError(404, "not_found", "分析记录不存在")
    model = db.get(ModelRegistry, analysis.model_id)
    if model is None:
        raise ApiError(404, "not_found", "模型不存在")
    label_key = payload["label_key"]
    method = payload.get("method") or "gradcam"

    existing = (analysis.explanation_json or {}).get("artifacts") or {}
    cached = existing.get(label_key)
    if cached and cached.get("model_sha256") == model.sha256 and cached.get("image_sha256") == analysis.image_sha256:
        return {"analysis_id": analysis.id, "label_key": label_key, "reused": True, "artifact_id": cached.get("artifact_id")}

    source = resolve_image_path(db, settings, analysis)
    adapter = get_cxr_adapter(settings)
    if adapter.backend == "dummy":
        from app.ml.cxr_adapter import DummyCxrAdapter

        adapter = DummyCxrAdapter()
    handle = adapter.load(Path(model.path), model.metadata_json or {})
    try:
        if should_stop and should_stop():
            from app.workers.dispatcher import JobCancelled

            raise JobCancelled()
        try:
            result = adapter.explain(handle, source, label_key)
        except ApiError as exc:
            # 解释失败不阻断评分与结构化报告：记录原因，热力图保持 null。
            if lock_for_write:
                lock_for_write()
            failures = dict((analysis.explanation_json or {}).get("failures") or {})
            failures[label_key] = {"code": exc.code, "reason": exc.message}
            analysis.explanation_json = {
                **(analysis.explanation_json or {}),
                "failures": failures,
                "definition": "模型关注区域；无定位真值，不计算 IoU/Dice，不代表病灶定位准确",
            }
            db.commit()
            raise
    finally:
        adapter.release(handle)

    cam = result["cam"]
    meta = result["meta"]
    target_dir = ensure_dir(Path(settings.artifact_root) / "cxr" / "explanations" / analysis.id / str(attempt))
    png_path = target_dir / f"{label_key}.png"
    _write_overlay(source, cam, png_path)
    if lock_for_write:
        lock_for_write()
    record = cxr_service.record_artifact(
        db, settings, owner_id=analysis.owner_id, resource_type="cxr_explanation",
        resource_id=analysis.id, path=png_path, mime="image/png",
    )
    artifacts = dict(existing)
    artifacts[label_key] = {
        "artifact_id": record.id,
        "sha256": record.sha256,
        "method": method,
        "label_key": label_key,
        "model_sha256": model.sha256,
        "image_sha256": analysis.image_sha256,
        "target_layer": result.get("target_layer") or "dummy",
        "preprocess": {key: meta.get(key) for key in ("image_size", "scale", "pad_left", "pad_top", "schema")},
        "limitations": result.get("limitations") or [],
    }
    analysis.explanation_json = {
        **(analysis.explanation_json or {}),
        "artifacts": artifacts,
        "method": method,
        "definition": "模型关注区域；无定位真值，不计算 IoU/Dice，不代表病灶定位准确",
    }
    analysis.explanation_artifact_id = record.id
    db.commit()
    if progress:
        progress(1.0, "completed", "解释已生成")
    return {"analysis_id": analysis.id, "label_key": label_key, "artifact_id": record.id, "sha256": record.sha256}


def _write_overlay(source: Path, cam, target: Path) -> None:
    """把关注图叠到原图上（只做可视化，不改变原图文件）。"""
    import numpy as np

    with Image.open(source) as image:
        base = image.convert("RGB")
    array = np.asarray(base, dtype=np.float32)
    heat = np.asarray(cam, dtype=np.float32)
    if heat.shape != array.shape[:2]:
        heat_image = Image.fromarray(heat, mode="F").resize((base.width, base.height), resample=Image.BILINEAR)
        heat = np.asarray(heat_image, dtype=np.float32)
    if not np.isfinite(heat).all():
        raise ApiError(500, "cxr_explain_failed", "关注区域图含非有限值")
    span = float(heat.max() - heat.min())
    normalized = (heat - float(heat.min())) / span if span > 1e-8 else np.zeros_like(heat)
    # 简单红色叠加：强度映射到 alpha，避免伪造纹理。
    overlay = array.copy()
    overlay[..., 0] = np.clip(array[..., 0] + normalized * 160.0, 0, 255)
    overlay[..., 1] = np.clip(array[..., 1] * (1 - normalized * 0.45), 0, 255)
    overlay[..., 2] = np.clip(array[..., 2] * (1 - normalized * 0.45), 0, 255)
    ensure_dir(target.parent)
    Image.fromarray(overlay.astype("uint8"), mode="RGB").save(target, format="PNG")
