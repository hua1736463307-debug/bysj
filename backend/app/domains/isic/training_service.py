"""ISIC 训练域：独立预检、参数白名单与可取消训练执行。

detect 的 TrainingParams 与 CXR 的 CxrTrainingParams 都不适用于七类单标签，
本模块使用 `IsicTrainingParams` 白名单，不把参数放开为任意 dict。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.core.config import Settings
from app.core.errors import ApiError
from app.core.storage import ensure_dir
from app.db.models import DatasetResource, DatasetVersion, JobTask, ModelRegistry
from app.directions.isic import ISIC_LABEL_SCHEMA, ISIC_LABELS, ISIC_TASK
from app.domains import job_service
from app.domains.isic import runner
from app.domains.isic.experiment_purpose import (
    EXPERIMENT_PURPOSE_FORMAL,
    normalize_purpose,
    validate_formal_dataset,
    validate_formal_protocol,
)
from app.domains.models import service as model_service
from app.ml.isic_adapter import get_isic_adapter
from app.schemas.isic import ISIC_CLASS_KEYS, ISIC_BACKBONE_TASK, IsicTrainingParams


class IsicTrainingPreflightError(ApiError):
    def __init__(self, checks: list[dict]) -> None:
        super().__init__(422, "isic_preflight_failed", "皮肤镜训练预检未通过", {"checks": checks})
        self.checks = checks


def load_manifest(version: DatasetVersion) -> dict:
    path = Path(version.manifest_path)
    if not path.exists():
        raise ApiError(409, "isic_manifest_missing", "数据版本缺少 manifest")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("label_schema") != ISIC_LABEL_SCHEMA:
        raise ApiError(409, "isic_schema_mismatch", "manifest 标签 schema 与当前契约不一致")
    order = [item["key"] for item in payload.get("labels") or []]
    if order and tuple(order) != ISIC_CLASS_KEYS:
        raise ApiError(409, "isic_class_order_mismatch", "manifest 类别顺序与固定契约不一致")
    return payload


def get_version(db: Session, version_id: str) -> tuple[DatasetVersion, DatasetResource]:
    version = db.get(DatasetVersion, version_id)
    if version is None or version.deleted_at is not None:
        raise ApiError(404, "not_found", "数据版本不存在")
    resource = db.get(DatasetResource, version.dataset_id)
    if resource is None or resource.deleted_at is not None:
        raise ApiError(404, "not_found", "数据集不存在")
    if resource.task_type != ISIC_TASK:
        raise ApiError(409, "task_mismatch", "该数据集不是皮肤镜数据集")
    return version, resource


def preflight(db: Session, settings: Settings, user: CurrentUser, payload: dict) -> dict:
    checks: list[dict] = []
    version: DatasetVersion | None = None
    manifest: dict | None = None

    version_id = payload.get("dataset_version_id")
    if not version_id:
        checks.append({"name": "dataset_version", "status": "fail", "detail": "必须提供 dataset_version_id"})
    else:
        try:
            version, _resource = get_version(db, version_id)
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
                    manifest = load_manifest(version)
                except ApiError as exc:
                    checks.append({"name": "manifest", "status": "fail", "detail": exc.message})
                else:
                    checks.append(
                        {
                            "name": "manifest",
                            "status": "pass",
                            "detail": f"assets={version.asset_count} counts={version.split_counts_json}",
                        }
                    )

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
        train_counts = (manifest.get("class_counts") or {}).get("train") or {}
        missing = [key for key in ISIC_CLASS_KEYS if not train_counts.get(key)]
        checks.append(
            {
                "name": "class_coverage",
                # 缺类不是硬失败：softmax 仍可训练，只是该类别永远不会被预测；
                # 但必须在预检里显式暴露，避免把"某类 F1=0"误读为模型问题。
                "status": "warn" if missing else "pass",
                "detail": "train 每类均有样本" if not missing else f"train 缺少类别：{missing}",
            }
        )

    params: dict = {}
    try:
        params = IsicTrainingParams.model_validate(payload.get("params") or {}).model_dump()
        checks.append({"name": "params", "status": "pass", "detail": "ISIC 参数白名单校验通过"})
    except ValidationError as exc:
        checks.append({"name": "params", "status": "fail", "detail": str(exc)})

    # P0：实验用途是提交时必须显式声明的独立语义，不再用 research_valid 兼任。
    # smoke 允许随机初始化；formal 必须绑定冻结协议、冻结数据版本与预训练权重。
    raw_purpose = payload.get("experiment_purpose")
    purpose = normalize_purpose(raw_purpose)
    if raw_purpose in (None, ""):
        checks.append(
            {
                "name": "experiment_purpose",
                "status": "warn",
                "detail": "未声明 experiment_purpose：按未分类处理，产物不具备论文资格（paper_eligible=false）",
            }
        )
    elif not purpose:
        checks.append(
            {
                "name": "experiment_purpose",
                "status": "fail",
                "detail": "experiment_purpose 只能是 smoke 或 formal",
            }
        )
    else:
        checks.append({"name": "experiment_purpose", "status": "pass", "detail": f"experiment_purpose={purpose}"})

    if purpose == EXPERIMENT_PURPOSE_FORMAL:
        formal_reasons = validate_formal_protocol(payload.get("protocol_id"), payload.get("protocol_sha256"))
        if payload.get("dataset_version_id"):
            version_for_dataset = db.get(DatasetVersion, payload["dataset_version_id"])
            if version_for_dataset is not None:
                formal_reasons.extend(validate_formal_dataset(version_for_dataset.manifest_sha256))
        if not payload.get("base_model_id"):
            formal_reasons.append("base_model_id_missing")
        checks.append(
            {
                "name": "formal_protocol",
                "status": "fail" if formal_reasons else "pass",
                "detail": "formal 门禁通过" if not formal_reasons else f"未满足：{formal_reasons}",
            }
        )

    base_model: ModelRegistry | None = None
    base_id = payload.get("base_model_id")
    if base_id:
        base_model = db.get(ModelRegistry, base_id)
        if base_model is None or base_model.status != "active":
            checks.append({"name": "base_model", "status": "fail", "detail": "初始权重模型不存在或未启用"})
        elif base_model.model_task != ISIC_BACKBONE_TASK:
            checks.append(
                {
                    "name": "base_model",
                    "status": "fail",
                    "detail": f"初始权重必须是 {ISIC_BACKBONE_TASK}，当前为 {base_model.model_task}",
                }
            )
        else:
            try:
                path = model_service.verify_model_file(base_model, settings)
                checks.append(
                    {"name": "base_model", "status": "pass", "detail": f"sha256={base_model.sha256[:12]} size={path.stat().st_size}"}
                )
            except ApiError as exc:
                checks.append({"name": "base_model", "status": "fail", "detail": exc.message})
    else:
        checks.append(
            {"name": "base_model", "status": "pass", "detail": "未提供初始权重：从随机初始化开始（smoke 允许）"}
        )

    # 预检在 API 进程里跑，而 API 镜像按契约不含 torch；真正的训练在 trainer worker 里。
    # 因此这里报告**配置值**并说明由谁决定，不能因为 API 没有 torch 就断言"后端是 dummy"。
    configured = (getattr(settings, "isic_ml_backend", "auto") or "auto").lower()
    if configured == "dummy":
        checks.append({"name": "backend", "status": "warn",
                       "detail": "显式配置 ISIC_ML_BACKEND=dummy：只验证契约，不产生研究指标"})
    elif configured == "auto":
        checks.append({"name": "backend", "status": "pass",
                       "detail": "ISIC_ML_BACKEND=auto：实际后端由 trainer worker 决定（API 容器不含 torch）"})
    else:
        checks.append({"name": "backend", "status": "pass", "detail": f"ISIC_ML_BACKEND={configured}"})

    return {
        "ok": all(item["status"] != "fail" for item in checks),
        "checks": checks,
        "params": params,
        "dataset_version_id": version.id if version else None,
        "base_model_id": base_model.id if base_model else None,
        "backend": configured,
        "label_schema": ISIC_LABEL_SCHEMA,
        "class_order": list(ISIC_CLASS_KEYS),
        "experiment_purpose": purpose or None,
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
        raise ApiError(422, "idempotency_key_required", "皮肤镜训练任务必须提供 Idempotency-Key")
    report = preflight(db, settings, user, payload)
    if not report["ok"]:
        raise IsicTrainingPreflightError(report["checks"])

    active = (
        db.query(JobTask)
        .filter(
            JobTask.owner_id == str(user.id),
            JobTask.type == "isic_train",
            JobTask.status.in_(["pending", "running", "cancelling"]),
        )
        .count()
    )
    if active >= settings.max_pending_training_per_user:
        raise ApiError(429, "queue_limit", "待执行皮肤镜训练任务已达上限")

    return job_service.create_job(
        db,
        job_type="isic_train",
        owner_id=user.id,
        owner_username=user.username,
        payload={
            "dataset_version_id": report["dataset_version_id"],
            "base_model_id": report["base_model_id"],
            "task": ISIC_TASK,
            "label_schema": ISIC_LABEL_SCHEMA,
            "class_order": list(ISIC_CLASS_KEYS),
            "params": report["params"],
            "experiment_purpose": report["experiment_purpose"],
            "protocol_id": payload.get("protocol_id"),
            "protocol_sha256": payload.get("protocol_sha256"),
        },
        idempotency_key=idempotency_key,
    )


def _run_dummy(request: dict, workdir: Path, progress, should_stop) -> dict:
    target = workdir / "best.json"
    payload = {
        "backend": "dummy",
        "arch": "densenet121",
        "label_schema": ISIC_LABEL_SCHEMA,
        "class_order": list(ISIC_CLASS_KEYS),
        "job_id": request["job_id"],
        "note": "contract-test artifact; not a trainable network and not research evidence",
    }
    target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    for step in range(1, 4):
        if should_stop and should_stop():
            from app.workers.dispatcher import JobCancelled

            raise JobCancelled()
        if progress:
            progress(step / 4, "training", f"dummy 皮肤镜训练 {step}/3")
    return {
        "backend": "dummy",
        "best_path": str(target),
        "best_epoch": 0,
        "epochs_completed": 0,
        "metrics": {
            "reason": "dummy backend 不产生真实指标，仅用于工程契约测试",
            "macro_f1": {"value": None, "reason": "dummy_backend"},
            "per_class": [],
        },
        "effective_args": request.get("params") or {},
        "environment": {"backend": "dummy"},
        "label_schema": ISIC_LABEL_SCHEMA,
        "class_order": list(ISIC_CLASS_KEYS),
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
    adapter = get_isic_adapter(settings)
    request = {
        "task": "train",
        "job_id": job_id,
        "dataset_root": str(version.root_path),
        "manifest": str(version.manifest_path),
        "dataset_manifest_sha256": version.manifest_sha256,
        "label_schema": ISIC_LABEL_SCHEMA,
        "class_order": list(ISIC_CLASS_KEYS),
        "init_state_dict": base_model.path if base_model else None,
        "base_model_sha256": base_model.sha256 if base_model else None,
        "class_weight": params.get("class_weight", "none"),
        "params": params,
        "workdir": str(workdir),
    }
    if adapter.backend == "dummy":
        result = _run_dummy(request, workdir, progress, should_stop)
    else:
        result = runner.run_isic_subprocess(request, workdir, should_stop=should_stop, progress=progress)
    result["dataset_manifest_sha256"] = version.manifest_sha256
    result["label_schema"] = ISIC_LABEL_SCHEMA
    result["class_order"] = list(ISIC_CLASS_KEYS)
    result["request_sha256"] = hashlib.sha256(
        json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    (workdir / "train-request-record.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def workspace(settings: Settings, job_id: str) -> Path:
    return ensure_dir(Path(settings.data_root) / "runs" / job_id)


def class_map() -> dict:
    return {label.key: label.display_name for label in ISIC_LABELS}
