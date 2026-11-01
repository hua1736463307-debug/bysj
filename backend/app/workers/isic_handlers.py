"""皮肤镜方向作业处理器：导入、冻结、训练、评价。

角色分配（不新建第二套作业系统）：
- dataset : isic_import, isic_freeze（纯 I/O 与 CPU，不占 GPU）
- trainer : isic_train, isic_evaluate（GPU 租约 + 受控子进程）

训练/评价复用既有 JobTask、GPU 租约与 fencing，取消与进程回收由
`app.domains.child_runner` 统一处理。
"""
from __future__ import annotations

from pathlib import Path

from app.core.errors import ApiError
from app.core.storage import ensure_dir
from app.db.models import (
    DatasetResource,
    DatasetVersion,
    EvaluationRun,
    ExperimentRun,
    ModelRegistry,
)
from app.directions.isic import ISIC_LABEL_SCHEMA, ISIC_TASK
from app.domains.isic import evaluation_service, import_service, split_service, training_service
from app.domains.isic.experiment_purpose import eligibility_metadata, paper_eligibility
from app.domains.models import service as model_service
from app.ml.isic_adapter import get_isic_adapter
from app.ml.yolo_adapter import environment_snapshot
from app.schemas.isic import (
    ISIC_CLASS_KEYS,
    ISIC_METRICS_SCHEMA,
    ISIC_MODEL_TASK,
    ISIC_PREPROCESS_SCHEMA,
)
from app.workers.dispatcher import JobContext
from app.db.models import utcnow


def handle_isic_import(ctx: JobContext) -> dict:
    payload = ctx.job.input_json or {}
    root = import_service.resolve_source(ctx.settings, payload["source_id"])
    split = payload.get("split") or "val"
    ctx.progress(0.2, "auditing", "校验官方 GroundTruth 与图像一致性")
    entries, report = import_service.plan_import(
        root=root, split=split, max_images=payload.get("max_images")
    )
    ctx.progress(0.6, "materializing", "复制图像到受控数据集目录")
    ctx.lock_for_write()
    result = import_service.materialize(
        ctx.db,
        ctx.settings,
        root=root,
        split=split,
        entries=entries,
        report=report,
        name=payload.get("name") or "",
        license_note=payload.get("license_note"),
        owner_id=int(ctx.job.owner_user_id or 0),
        owner_username=ctx.job.owner_username or "system",
    )
    ctx.db.commit()
    ctx.progress(1.0, "imported", "皮肤镜导入完成")
    return result


def handle_isic_freeze(ctx: JobContext) -> dict:
    payload = ctx.job.input_json or {}
    version = ctx.db.get(DatasetVersion, payload["version_id"])
    if version is None:
        raise ApiError(404, "not_found", "数据版本不存在")
    resource = ctx.db.get(DatasetResource, version.dataset_id)
    if resource is None:
        raise ApiError(404, "not_found", "数据集不存在")
    if resource.task_type != ISIC_TASK:
        raise ApiError(409, "task_mismatch", "该数据集不是皮肤镜数据集")
    ctx.progress(0.2, "splitting", "按 lesion/单图分组划分四集")
    return split_service.freeze_version(
        ctx.db,
        ctx.settings,
        version=version,
        resource=resource,
        split_seed=int(payload.get("split_seed") or 42),
        ratios=payload.get("ratios") or [0.7, 0.1, 0.1, 0.1],
        progress=ctx.progress,
    )


# ---------------------------------------------------------------------------
# GPU 作业：训练与评价
#
# 与 CXR 共用同一套 JobTask、GPU 租约（app.workers.gpu）与 fencing：
# 不新建队列、不新建调度器，取消/回收由 child_runner 统一处理。
# ---------------------------------------------------------------------------


def _acquire_gpu(ctx: JobContext):
    from app.workers.gpu import acquire

    return acquire(ctx)


def handle_isic_train(ctx: JobContext) -> dict:
    payload = ctx.job.input_json or {}
    version = ctx.db.get(DatasetVersion, payload["dataset_version_id"])
    if version is None:
        raise ApiError(404, "not_found", "数据版本不存在")
    base_model = ctx.db.get(ModelRegistry, payload["base_model_id"]) if payload.get("base_model_id") else None
    workspace = training_service.workspace(ctx.settings, ctx.job.id)

    holder = _acquire_gpu(ctx)
    try:
        existing = (
            ctx.db.query(ModelRegistry).filter(ModelRegistry.training_job_id == ctx.job.id).one_or_none()
        )
        if existing is not None:
            model_service.verify_model_file(existing, ctx.settings)
            result = (existing.metadata_json or {}).get("training_result")
            if not result:
                raise ApiError(409, "recovery_metadata_missing", "已有模型缺少完整训练记录，拒绝覆盖；需人工核对")
            return {
                "model_id": existing.id,
                "model_sha256": existing.sha256,
                "reused": True,
                "metrics": (existing.metadata_json or {}).get("metrics"),
                "paper_eligible": paper_eligibility(existing.metadata_json)["paper_eligible"],
            }

        result = training_service.train(
            ctx.settings,
            ctx.db,
            job_id=ctx.job.id,
            version=version,
            base_model=base_model,
            params=payload.get("params") or {},
            workdir=workspace,
            should_stop=ctx.should_stop,
            progress=ctx.progress,
        )
        ctx.check_cancelled()
        best_path = Path(result["best_path"])
        try:
            best_path.resolve().relative_to(workspace.resolve())
        except ValueError as exc:
            raise ApiError(500, "isic_child_path_escape", "训练产物路径越界") from exc

        ctx.lock_for_write()
        backend = result.get("backend") or get_isic_adapter(ctx.settings).backend
        params = payload.get("params") or {}
        # P0：论文资格在落库时用门禁函数重新计算，不接受调用方传入的布尔值。
        purpose = payload.get("experiment_purpose")
        protocol_id = payload.get("protocol_id")
        protocol_sha256 = payload.get("protocol_sha256")
        eligibility = eligibility_metadata(
            {
                "experiment_purpose": purpose,
                "research_valid": backend != "dummy",
                "backend": backend,
                "synthetic": False,
                "dataset_manifest_sha256": version.manifest_sha256,
                "protocol_id": protocol_id,
                "protocol_sha256": protocol_sha256,
                "init_weights_sha256": base_model.sha256 if base_model else None,
                "seed": params.get("seed", 42),
            }
        )
        registered = model_service.register_trained_model(
            ctx.db,
            ctx.settings,
            job_id=ctx.job.id,
            version_id=version.id,
            owner_id=int(ctx.job.owner_user_id or 0),
            owner_username=ctx.job.owner_username,
            name=f"isic-{version.dataset_id[:8]}-{ctx.job.id[:8]}",
            best_path=best_path,
            class_map=training_service.class_map(),
            metrics=result.get("metrics") or {},
            backend=backend,
            training_result=result,
            model_task=ISIC_MODEL_TASK,
            file_format="json" if backend == "dummy" else "pt",
            metadata_extra={
                "label_schema": ISIC_LABEL_SCHEMA,
                "class_order": list(ISIC_CLASS_KEYS),
                "arch": "densenet121",
                "image_size": int(params.get("image_size") or 224),
                "preprocess_schema": ISIC_PREPROCESS_SCHEMA,
                "metrics_schema": ISIC_METRICS_SCHEMA,
                "class_weight": params.get("class_weight", "none"),
                "seed": params.get("seed", 42),
                "dataset_manifest_sha256": version.manifest_sha256,
                "effective_args": result.get("effective_args"),
                "history": result.get("history"),
                "best_epoch": result.get("best_epoch"),
                "init_weights_sha256": base_model.sha256 if base_model else None,
                "environment": result.get("environment"),
                "research_valid": backend != "dummy",
                "experiment_purpose": purpose,
                "protocol_id": protocol_id,
                "protocol_sha256": protocol_sha256,
                **eligibility,
            },
        )
        experiment = ctx.db.query(ExperimentRun).filter(ExperimentRun.training_job_id == ctx.job.id).one_or_none()
        if experiment is None:
            experiment = ExperimentRun(
                owner_id=int(ctx.job.owner_user_id or 0),
                training_job_id=ctx.job.id,
                model_id=registered.id,
                dataset_version_id=version.id,
                code_sha=environment_snapshot().get("code_sha"),
                env_json=environment_snapshot(),
                config_json={
                    "task": ISIC_TASK,
                    "label_schema": ISIC_LABEL_SCHEMA,
                    "class_order": list(ISIC_CLASS_KEYS),
                    "params": params,
                    "dataset_manifest_sha256": version.manifest_sha256,
                    "init_weights_sha256": base_model.sha256 if base_model else None,
                    "effective_args": result.get("effective_args"),
                    "seed": params.get("seed", 42),
                    "preprocess_schema": ISIC_PREPROCESS_SCHEMA,
                    "experiment_purpose": purpose,
                    "protocol_id": protocol_id,
                    "protocol_sha256": protocol_sha256,
                    "paper_eligible": eligibility["paper_eligible"],
                },
                metrics_json=result.get("metrics") or {},
            )
            ctx.db.add(experiment)
        ctx.db.commit()
        return {
            "model_id": registered.id,
            "model_sha256": registered.sha256,
            "backend": backend,
            "best_epoch": result.get("best_epoch"),
            "metrics": result.get("metrics"),
            "research_valid": backend != "dummy",
            "experiment_purpose": purpose,
            "paper_eligible": eligibility["paper_eligible"],
            "paper_ineligible_reasons": eligibility["paper_ineligible_reasons"],
        }
    finally:
        from app.workers.gpu import release

        release(ctx, holder)


def handle_isic_evaluate(ctx: JobContext) -> dict:
    payload = ctx.job.input_json or {}
    version = ctx.db.get(DatasetVersion, payload["dataset_version_id"])
    model = ctx.db.get(ModelRegistry, payload["model_id"])
    if version is None or model is None:
        raise ApiError(404, "not_found", "数据版本或模型不存在")
    workspace = ensure_dir(Path(ctx.settings.data_root) / "runs" / ctx.job.id)
    run = ctx.db.get(EvaluationRun, payload["evaluation_id"]) if payload.get("evaluation_id") else None

    holder = _acquire_gpu(ctx)
    try:
        if run is not None:
            run.status = "running"
            ctx.db.commit()
        result = evaluation_service.run(
            ctx.settings,
            ctx.db,
            job_id=ctx.job.id,
            version=version,
            model=model,
            split=payload.get("split") or "val",
            workdir=workspace,
            should_stop=ctx.should_stop,
            progress=ctx.progress,
        )
        ctx.check_cancelled()
        if run is not None:
            ctx.lock_for_write()
            run.metrics_json = result.get("metrics") or {}
            run.status = "completed"
            run.finished_at = utcnow()
            run.artifacts_json = {"workspace": str(workspace.name)}
            ctx.db.commit()
        return {
            "evaluation_id": run.id if run is not None else None,
            "split": result.get("split"),
            "counts": result.get("counts"),
            "metrics": result.get("metrics"),
            "backend": result.get("backend"),
        }
    except Exception:
        if run is not None:
            run.status = "failed"
            run.finished_at = utcnow()
            ctx.db.commit()
        raise
    finally:
        from app.workers.gpu import release

        release(ctx, holder)
