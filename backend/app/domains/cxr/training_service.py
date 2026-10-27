"""CXR 训练域：独立预检、参数白名单与可取消训练执行。

detect 的 TrainingParams/预检只认 detect，本模块不复用它们，也不把
detect 参数放开为任意 dict。
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.core.config import Settings
from app.core.errors import ApiError
from app.core.storage import ensure_dir
from app.db.models import DatasetVersion, JobTask, ModelRegistry
from app.domains import job_service
from app.domains.cxr import service as cxr_service
from app.domains.cxr.runner import run_cxr_subprocess
from app.domains.cxr.split_service import training_gate
from app.ml.cxr_adapter import get_cxr_adapter
from app.schemas.cxr import CXR_LABEL_SCHEMA, CXR_TASK, CxrTrainingParams


class CxrTrainingPreflightError(ApiError):
    def __init__(self, checks: list[dict]) -> None:
        super().__init__(422, "cxr_preflight_failed", "胸片训练预检未通过", {"checks": checks})
        self.checks = checks


def preflight(db: Session, settings: Settings, user: CurrentUser, payload: dict) -> dict:
    checks: list[dict] = []
    version_id = payload.get("dataset_version_id")
    version: DatasetVersion | None = None
    manifest: dict | None = None
    if not version_id:
        checks.append({"name": "dataset_version", "status": "fail", "detail": "必须提供 dataset_version_id"})
    else:
        try:
            version = cxr_service.get_cxr_version(db, version_id, user)
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
                    manifest = cxr_service.load_manifest(version)
                    checks.append(
                        {
                            "name": "manifest",
                            "status": "pass",
                            "detail": f"assets={version.asset_count} counts={version.split_counts_json}",
                        }
                    )
                except ApiError as exc:
                    checks.append({"name": "manifest", "status": "fail", "detail": exc.message})

    if manifest is not None:
        counts = manifest.get("counts") or {}
        checks.append(
            {
                "name": "splits",
                "status": "pass" if counts.get("train") and counts.get("val") else "fail",
                "detail": f"train={counts.get('train')} val={counts.get('val')} "
                          f"calibration={counts.get('calibration')} test={counts.get('test')}",
            }
        )
        gate = training_gate(manifest)
        checks.append(
            {
                "name": "label_support",
                "status": "pass" if not gate else "fail",
                "detail": "train 每个标签均有明确正例与负例" if not gate else gate,
            }
        )

    params: dict = {}
    try:
        params = CxrTrainingParams.model_validate(payload.get("params") or {}).model_dump()
        checks.append({"name": "params", "status": "pass", "detail": "CXR 参数白名单校验通过"})
    except ValidationError as exc:
        checks.append({"name": "params", "status": "fail", "detail": str(exc)})

    base_model: ModelRegistry | None = None
    base_id = payload.get("base_model_id")
    if base_id:
        try:
            base_model = cxr_service.get_cxr_model(db, base_id, user)
        except ApiError as exc:
            checks.append({"name": "base_model", "status": "fail", "detail": exc.message})
        else:
            if base_model.model_task != "cxr_backbone":
                checks.append(
                    {
                        "name": "base_model",
                        "status": "fail",
                        "detail": f"初始权重必须是 cxr_backbone，当前为 {base_model.model_task}",
                    }
                )
            else:
                try:
                    path = cxr_service.verify_model_hash(base_model, settings)
                    checks.append(
                        {"name": "base_model", "status": "pass", "detail": f"sha256={base_model.sha256[:12]} size={path.stat().st_size}"}
                    )
                except ApiError as exc:
                    checks.append({"name": "base_model", "status": "fail", "detail": exc.message})
    else:
        checks.append(
            {
                "name": "base_model",
                "status": "fail",
                "detail": "必须提供已登记的 ImageNet 骨干（cxr_backbone）作为初始化权重",
            }
        )

    adapter = get_cxr_adapter(settings)
    checks.append(
        {
            "name": "backend",
            "status": "pass" if adapter.backend == "densenet121" else "warn",
            "detail": f"backend={adapter.backend}"
                      + ("（dummy 只验证契约，不产生研究指标）" if adapter.backend == "dummy" else ""),
        }
    )

    ok = all(item["status"] != "fail" for item in checks)
    return {
        "ok": ok,
        "checks": checks,
        "params": params,
        "dataset_version_id": version.id if version else None,
        "base_model_id": base_model.id if base_model else None,
        "backend": adapter.backend,
        "label_schema": CXR_LABEL_SCHEMA,
    }


def create_training_job(
    db: Session,
    settings: Settings,
    user: CurrentUser,
    payload: dict,
    *,
    idempotency_key: str | None,
) -> tuple[JobTask, bool]:
    if not idempotency_key:
        raise ApiError(422, "idempotency_key_required", "胸片训练任务必须提供 Idempotency-Key")
    report = preflight(db, settings, user, payload)
    if not report["ok"]:
        raise CxrTrainingPreflightError(report["checks"])

    active = (
        db.query(JobTask)
        .filter(
            JobTask.owner_id == str(user.id),
            JobTask.type == "cxr_train",
            JobTask.status.in_(["pending", "running", "cancelling"]),
        )
        .count()
    )
    if active >= settings.max_pending_training_per_user:
        raise ApiError(429, "queue_limit", "待执行胸片训练任务已达上限")

    return job_service.create_job(
        db,
        job_type="cxr_train",
        owner_id=user.id,
        owner_username=user.username,
        payload={
            "dataset_version_id": report["dataset_version_id"],
            "base_model_id": report["base_model_id"],
            "task": CXR_TASK,
            "label_schema": CXR_LABEL_SCHEMA,
            "params": report["params"],
        },
        idempotency_key=idempotency_key,
    )


def _run_dummy(request: dict, workdir: Path, progress, should_stop) -> dict:
    target = workdir / "best.json"
    payload = {
        "backend": "dummy",
        "arch": "densenet121",
        "label_schema": CXR_LABEL_SCHEMA,
        "label_order": ["atelectasis", "cardiomegaly", "consolidation", "edema", "pleural_effusion"],
        "job_id": request["job_id"],
        "note": "contract-test artifact; not a trainable network and not research evidence",
    }
    target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    for step in range(1, 4):
        if should_stop and should_stop():
            from app.workers.dispatcher import JobCancelled

            raise JobCancelled()
        if progress:
            progress(step / 4, "training", f"dummy 胸片训练 {step}/3")
    return {
        "backend": "dummy",
        "best_path": str(target),
        "best_epoch": 0,
        "epochs_completed": 0,
        "metrics": {
            "reason": "dummy backend 不产生真实指标，仅用于工程契约测试",
            "macro_auroc": {"value": None, "reason": "dummy_backend"},
            "per_label": {},
        },
        "effective_args": request.get("params") or {},
        "environment": {"backend": "dummy"},
    }


def train(
    settings: Settings,
    db: Session,
    *,
    job_id: str,
    version: DatasetVersion,
    base_model: ModelRegistry | None,
    params: dict,
    workdir: Path,
    should_stop=None,
    progress=None,
) -> dict:
    adapter = get_cxr_adapter(settings)
    request = {
        "task": "train",
        "job_id": job_id,
        "dataset_root": str(version.root_path),
        "manifest": str(version.manifest_path),
        "dataset_manifest_sha256": version.manifest_sha256,
        "label_schema": CXR_LABEL_SCHEMA,
        "init_state_dict": base_model.path if base_model else None,
        "base_model_sha256": base_model.sha256 if base_model else None,
        "class_weight": params.get("class_weight", "none"),
        "params": params,
        "workdir": str(workdir),
    }
    if adapter.backend == "dummy":
        result = _run_dummy(request, workdir, progress, should_stop)
    else:
        result = run_cxr_subprocess(request, workdir, should_stop=should_stop, progress=progress)
    result["dataset_manifest_sha256"] = version.manifest_sha256
    result["label_schema"] = CXR_LABEL_SCHEMA
    result["request_sha256"] = __import__("hashlib").sha256(
        json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    (workdir / "train-request-record.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def workspace(settings: Settings, job_id: str) -> Path:
    return ensure_dir(Path(settings.data_root) / "runs" / job_id)
