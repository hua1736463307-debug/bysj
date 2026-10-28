"""ISIC 评价域：在指定 split 上计算七类指标。

策略（与 CXR 同构，语义独立）：
- `test` 只允许管理员执行，且必须写明预登记目的；
- 评价只读冻结版本的 manifest 与图像，不改数据、不写模型；
- dummy 后端产生的指标带 `research_valid=false` 与原因，不进入任何汇总。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.core.config import Settings
from app.core.errors import ApiError
from app.db.models import DatasetVersion, EvaluationRun, JobTask, ModelRegistry
from app.directions.isic import ISIC_LABEL_SCHEMA, ISIC_TASK
from app.domains import job_service
from app.domains.isic import runner
from app.domains.isic import training_service
from app.domains.models import service as model_service
from app.ml.isic_adapter import get_isic_adapter
from app.ml.isic_metrics import evaluate as evaluate_metrics
from app.schemas.isic import ISIC_CLASS_COUNT, ISIC_CLASS_KEYS, ISIC_MODEL_TASK


def preflight(db: Session, settings: Settings, user: CurrentUser, payload: dict) -> dict:
    checks: list[dict] = []
    version: DatasetVersion | None = None
    manifest: dict | None = None
    split = payload.get("split") or "val"

    version_id = payload.get("dataset_version_id")
    if not version_id:
        checks.append({"name": "dataset_version", "status": "fail", "detail": "必须提供 dataset_version_id"})
    else:
        try:
            version, _resource = training_service.get_version(db, version_id)
        except ApiError as exc:
            checks.append({"name": "dataset_version", "status": "fail", "detail": exc.message})
        else:
            checks.append(
                {
                    "name": "dataset_version",
                    "status": "pass" if version.state == "frozen" else "fail",
                    "detail": f"state={version.state}",
                }
            )
            if version.state == "frozen":
                try:
                    manifest = training_service.load_manifest(version)
                except ApiError as exc:
                    checks.append({"name": "manifest", "status": "fail", "detail": exc.message})

    if manifest is not None:
        counts = manifest.get("counts") or {}
        checks.append(
            {
                "name": "split",
                "status": "pass" if counts.get(split) else "fail",
                "detail": f"{split}={counts.get(split)}",
            }
        )

    model: ModelRegistry | None = None
    model_id = payload.get("model_id")
    if not model_id:
        checks.append({"name": "model", "status": "fail", "detail": "必须提供 model_id"})
    else:
        model = db.get(ModelRegistry, model_id)
        if model is None or model.status != "active":
            checks.append({"name": "model", "status": "fail", "detail": "模型不存在或未启用"})
        elif model.model_task != ISIC_MODEL_TASK:
            checks.append(
                {
                    "name": "model",
                    "status": "fail",
                    "detail": f"该模型任务为 {model.model_task}，不能用于皮肤镜七类评价",
                }
            )
        else:
            try:
                path = model_service.verify_model_file(model, settings)
                checks.append({"name": "model", "status": "pass", "detail": f"sha256={model.sha256[:12]} size={path.stat().st_size}"})
            except ApiError as exc:
                checks.append({"name": "model", "status": "fail", "detail": exc.message})

    if split == "test":
        if not user.is_admin:
            raise ApiError(403, "forbidden", "test 评价需要管理员权限")
        if not (payload.get("purpose") or "").strip():
            raise ApiError(422, "isic_test_purpose_required", "test 评价必须写明预登记目的")

    adapter = get_isic_adapter(settings)
    checks.append(
        {
            "name": "backend",
            "status": "pass" if adapter.backend == "densenet121" else "warn",
            "detail": f"backend={adapter.backend}"
                      + ("（dummy 只验证契约，不产生研究指标）" if adapter.backend == "dummy" else ""),
        }
    )

    return {
        "ok": all(item["status"] != "fail" for item in checks),
        "checks": checks,
        "split": split,
        "dataset_version_id": version.id if version else None,
        "model_id": model.id if model else None,
        "backend": adapter.backend,
        "label_schema": ISIC_LABEL_SCHEMA,
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
        raise ApiError(422, "idempotency_key_required", "皮肤镜评价任务必须提供 Idempotency-Key")
    report = preflight(db, settings, user, payload)
    if not report["ok"]:
        raise ApiError(422, "isic_evaluation_preflight_failed", "皮肤镜评价预检未通过", {"checks": report["checks"]})

    purpose = (payload.get("purpose") or "").strip() or None
    context_hash = hashlib.sha256(
        json.dumps(
            {
                "model_sha": (db.get(ModelRegistry, report["model_id"]).sha256 if report["model_id"] else ""),
                "dataset_version_id": report["dataset_version_id"],
                "split": report["split"],
                "purpose": purpose or "",
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    run = EvaluationRun(
        owner_id=user.id,
        model_id=report["model_id"],
        dataset_version_id=report["dataset_version_id"],
        split=report["split"],
        status="pending",
        purpose=purpose,
        params_json={
            "task": ISIC_TASK,
            "label_schema": ISIC_LABEL_SCHEMA,
            "class_order": list(ISIC_CLASS_KEYS),
            "purpose": purpose,
        },
        context_hash=context_hash,
    )
    db.add(run)
    db.flush()

    job, created = job_service.create_job(
        db,
        job_type="isic_evaluate",
        owner_id=user.id,
        owner_username=user.username,
        payload={
            "dataset_version_id": report["dataset_version_id"],
            "model_id": report["model_id"],
            "split": report["split"],
            "purpose": purpose,
            "evaluation_id": run.id,
            "label_schema": ISIC_LABEL_SCHEMA,
            "class_order": list(ISIC_CLASS_KEYS),
        },
        idempotency_key=idempotency_key,
    )
    run.job_id = job.id
    db.commit()
    return job, created


def _load_items(version: DatasetVersion, split: str) -> list[dict]:
    manifest = training_service.load_manifest(version)
    root = Path(version.root_path)
    items = []
    for asset in manifest["assets"]:
        if asset["split"] != split:
            continue
        items.append(
            {
                "path": root / "images" / asset["image"],
                "class_index": int(asset["class_index"]),
            }
        )
    return items


def _dummy_metrics(items: list[dict]) -> dict:
    """确定性伪概率：只用于验证指标链路，绝不代表模型能力。"""
    probabilities = np.zeros((len(items), ISIC_CLASS_COUNT), dtype=np.float64)
    for index, item in enumerate(items):
        digest = hashlib.sha256(Path(item["path"]).read_bytes()).digest()
        raw = np.frombuffer(digest[:ISIC_CLASS_COUNT], dtype=np.uint8).astype(np.float64) + 1.0
        probabilities[index] = raw / raw.sum()
    truth = [item["class_index"] for item in items]
    metrics = evaluate_metrics(truth, probabilities.argmax(axis=1).tolist(), probabilities=probabilities)
    metrics["research_valid"] = False
    metrics["reason"] = "dummy backend 的伪概率，仅用于验证指标链路，不得作为模型性能"
    return metrics


def run(
    settings: Settings,
    db: Session,
    *,
    job_id: str,
    version: DatasetVersion,
    model: ModelRegistry,
    split: str,
    workdir: Path,
    should_stop=None,
    progress=None,
) -> dict:
    adapter = get_isic_adapter(settings)
    if adapter.backend == "dummy":
        if should_stop and should_stop():
            from app.workers.dispatcher import JobCancelled

            raise JobCancelled()
        items = _load_items(version, split)
        if progress:
            progress(0.5, "evaluating", f"dummy 评价 {split} {len(items)} 张")
        metrics = _dummy_metrics(items)
        return {
            "backend": "dummy",
            "split": split,
            "counts": len(items),
            "metrics": metrics,
            "dataset_manifest_sha256": version.manifest_sha256,
            "model_sha256": model.sha256,
            "label_schema": ISIC_LABEL_SCHEMA,
            "class_order": list(ISIC_CLASS_KEYS),
            "environment": {"backend": "dummy"},
        }

    request = {
        "task": "evaluate",
        "job_id": job_id,
        "dataset_root": str(version.root_path),
        "manifest": str(version.manifest_path),
        "dataset_manifest_sha256": version.manifest_sha256,
        "model_path": model.path,
        "model_sha256": model.sha256,
        "split": split,
        "label_schema": ISIC_LABEL_SCHEMA,
        "class_order": list(ISIC_CLASS_KEYS),
        "workdir": str(workdir),
        "device": (model.metadata_json or {}).get("device", "0"),
    }
    result = runner.run_isic_subprocess(request, workdir, should_stop=should_stop, progress=progress)
    result["dataset_manifest_sha256"] = version.manifest_sha256
    result["model_sha256"] = model.sha256
    result["label_schema"] = ISIC_LABEL_SCHEMA
    result["class_order"] = list(ISIC_CLASS_KEYS)
    (workdir / "evaluate-request-record.json").write_text(
        json.dumps({**request, "model_path": "<controlled>"}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result
