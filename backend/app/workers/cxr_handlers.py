"""CXR 作业处理器：注册到既有 dispatcher，复用 JobTask / GPU 租约 / fencing。

角色分配（不新建第二套作业系统）：
- dataset  : cxr_import, cxr_freeze
- trainer  : cxr_train, cxr_evaluate
- inference: cxr_analyze, cxr_explain
"""
from __future__ import annotations

from pathlib import Path

from app.api.deps import CurrentUser
from app.core.errors import ApiError
from app.core.storage import ensure_dir
from app.db.models import (
    DatasetResource,
    DatasetVersion,
    ExperimentRun,
    ModelRegistry,
)
from app.domains.cxr import (
    analysis_service,
    evaluation_service,
    freeze_service,
    import_service,
    training_service,
)
from app.domains.models import service as model_service
from app.ml.cxr_adapter import get_cxr_adapter
from app.ml.yolo_adapter import environment_snapshot
from app.schemas.cxr import CXR_LABEL_SCHEMA, CXR_TASK, LABEL_DISPLAY
from app.workers.dispatcher import JobContext


def _owner(ctx: JobContext) -> CurrentUser:
    return CurrentUser(
        id=int(ctx.job.owner_user_id or 0),
        username=ctx.job.owner_username or "system",
        role="admin",
        raw_role="admin",
    )


def _acquire_gpu(ctx: JobContext):
    from app.workers.gpu import acquire

    return acquire(ctx)


def handle_cxr_import(ctx: JobContext) -> dict:
    payload = ctx.job.input_json or {}
    source_root = import_service.resolve_source_root(ctx.settings, payload["source_id"])
    synthetic = import_service.assert_source_allowed(source_root, ctx.settings)
    csv_path = import_service._resolve_inside(source_root, payload.get("source_csv") or "train.csv")
    rows = import_service.read_source_csv(csv_path)
    ctx.progress(0.2, "auditing", "审计来源 CSV 与图像")
    namespace = f"cxr:{payload['source_id']}"
    selected, report = import_service.plan_import(
        source_root=source_root,
        rows=rows,
        max_patients=int(payload.get("max_patients") or 10000),
        namespace=namespace,
    )
    if payload.get("dry_run"):
        return {"dry_run": True, "report": report, "selected": len(selected)}

    ctx.progress(0.6, "materializing", "复制图像到受控数据集目录")
    ctx.lock_for_write()
    result = import_service.materialize(
        ctx.db,
        ctx.settings,
        source_root=source_root,
        source_csv=csv_path,
        source_id=payload["source_id"],
        selected=selected,
        report=report,
        name=payload.get("name") or "",
        license_note=payload.get("license_note"),
        owner_id=int(ctx.job.owner_user_id or 0),
        owner_username=ctx.job.owner_username or "system",
        synthetic=synthetic,
    )
    ctx.db.commit()
    ctx.progress(1.0, "imported", "胸片导入完成")
    return result


def handle_cxr_freeze(ctx: JobContext) -> dict:
    payload = ctx.job.input_json or {}
    version = ctx.db.get(DatasetVersion, payload["version_id"])
    if version is None:
        raise ApiError(404, "not_found", "数据版本不存在")
    resource = ctx.db.get(DatasetResource, version.dataset_id)
    if resource is None:
        raise ApiError(404, "not_found", "数据集不存在")
    if resource.task_type != CXR_TASK:
        raise ApiError(409, "task_mismatch", "该数据集不是胸片五标签数据集")
    ctx.progress(0.2, "splitting", "按患者 component 划分四集")
    result = freeze_service.freeze_cxr_version(
        ctx.db,
        ctx.settings,
        version=version,
        resource=resource,
        split_seed=int(payload.get("split_seed") or 42),
        ratios=payload.get("ratios") or [0.7, 0.1, 0.1, 0.1],
        progress=ctx.progress,
    )
    return result


def handle_cxr_train(ctx: JobContext) -> dict:
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
            raise ApiError(500, "cxr_child_path_escape", "训练产物路径越界") from exc
        ctx.lock_for_write()
        backend = result.get("backend") or get_cxr_adapter(ctx.settings).backend
        registered = model_service.register_trained_model(
            ctx.db,
            ctx.settings,
            job_id=ctx.job.id,
            version_id=version.id,
            owner_id=int(ctx.job.owner_user_id or 0),
            owner_username=ctx.job.owner_username,
            name=f"cxr-{version.dataset_id[:8]}-{ctx.job.id[:8]}",
            best_path=best_path,
            class_map=dict(LABEL_DISPLAY),
            metrics=result.get("metrics") or {},
            backend=backend,
            training_result=result,
            model_task="cxr_multilabel",
            file_format="json" if backend == "dummy" else "pt",
            metadata_extra={
                "label_schema": CXR_LABEL_SCHEMA,
                "arch": "densenet121",
                "image_size": int((payload.get("params") or {}).get("image_size") or 224),
                "preprocess_schema": "cxr-preprocess-v1",
                "class_weight": (payload.get("params") or {}).get("class_weight", "none"),
                "seed": (payload.get("params") or {}).get("seed", 42),
                "dataset_manifest_sha256": version.manifest_sha256,
                "effective_args": result.get("effective_args"),
                "history": result.get("history"),
                "best_epoch": result.get("best_epoch"),
                "init_weights_sha256": base_model.sha256 if base_model else None,
                "environment": result.get("environment"),
                "research_valid": backend != "dummy",
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
                    "task": CXR_TASK,
                    "label_schema": CXR_LABEL_SCHEMA,
                    "params": payload.get("params"),
                    "dataset_manifest_sha256": version.manifest_sha256,
                    "init_weights_sha256": base_model.sha256 if base_model else None,
                    "effective_args": result.get("effective_args"),
                },
                metrics_json=result.get("metrics") or {},
                artifact_refs={"model_sha256": registered.sha256, "best_path": str(best_path)},
            )
        ctx.lock_for_write()
        ctx.db.add(experiment)
        ctx.db.commit()
        return {
            "model_id": registered.id,
            "model_sha256": registered.sha256,
            "backend": backend,
            "metrics": result.get("metrics"),
            "best_epoch": result.get("best_epoch"),
            "epochs_completed": result.get("epochs_completed"),
            "experiment_id": experiment.id,
            "dataset_manifest_sha256": version.manifest_sha256,
            "research_valid": backend != "dummy",
        }
    finally:
        from app.workers.gpu import release

        release(ctx, holder)


def handle_cxr_evaluate(ctx: JobContext) -> dict:
    workspace = ensure_dir(Path(ctx.settings.data_root) / "runs" / ctx.job.id)
    holder = _acquire_gpu(ctx)
    try:
        return evaluation_service.run_evaluation(
            ctx.db,
            ctx.settings,
            job=ctx.job,
            workdir=workspace,
            should_stop=ctx.should_stop,
            progress=ctx.progress,
            attempt=ctx.attempt,
            lock_for_write=ctx.lock_for_write,
        )
    finally:
        from app.workers.gpu import release

        release(ctx, holder)


def handle_cxr_analyze(ctx: JobContext) -> dict:
    workspace = ensure_dir(Path(ctx.settings.data_root) / "runs" / ctx.job.id)
    holder = _acquire_gpu(ctx)
    try:
        return analysis_service.run_batch(
            ctx.db,
            ctx.settings,
            job=ctx.job,
            workdir=workspace,
            should_stop=ctx.should_stop,
            progress=ctx.progress,
            attempt=ctx.attempt,
            lock_for_write=ctx.lock_for_write,
        )
    finally:
        from app.workers.gpu import release

        release(ctx, holder)


def handle_cxr_explain(ctx: JobContext) -> dict:
    workspace = ensure_dir(Path(ctx.settings.data_root) / "runs" / ctx.job.id)
    holder = _acquire_gpu(ctx)
    try:
        return analysis_service.run_explanation(
            ctx.db,
            ctx.settings,
            job=ctx.job,
            workdir=workspace,
            should_stop=ctx.should_stop,
            progress=ctx.progress,
            attempt=ctx.attempt,
            lock_for_write=ctx.lock_for_write,
        )
    finally:
        from app.workers.gpu import release

        release(ctx, holder)
