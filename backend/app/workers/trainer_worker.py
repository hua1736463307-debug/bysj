"""训练与评估 worker（GPU 单卡租约）。

命令：``python -m app.workers.trainer_worker``
"""
from __future__ import annotations

import logging
import os
import socket
import time
from pathlib import Path

from app.api.deps import CurrentUser
from app.core.errors import ApiError
from app.core.storage import ensure_dir
from app.db.base import utcnow
from app.db.models import DatasetVersion, EvaluationRun, ExperimentRun, JobTask, ModelRegistry
from app.db.session import get_session_factory
from app.domains import job_service, lease_service
from app.domains.models import service as model_service
from app.domains.training import service as training_service
from app.ml.yolo_adapter import compute_sha256, environment_snapshot, get_adapter
from app.workers.dispatcher import JobCancelled, JobContext, run_claimed_job

logger = logging.getLogger("bysj.worker.trainer")

WORKER_TYPES = ("training", "evaluation", "cxr_train", "cxr_evaluate", "isic_train", "isic_evaluate")


def _cancel_checker(job_id: str):
    def _check() -> bool:
        factory = get_session_factory()
        with factory() as db:
            job = db.get(JobTask, job_id)
            if job is None:
                return True
            return bool(
                job.cancel_requested
                or job.status in {"cancelling", "interrupted", "failed", "cancelled"}
            )

    return _check


def _acquire_gpu(ctx: JobContext):
    from app.workers.gpu import acquire
    return acquire(ctx)


def handle_training(ctx: JobContext) -> dict:
    payload = ctx.job.input_json or {}
    version = ctx.db.get(DatasetVersion, payload["dataset_version_id"])
    if version is None:
        raise ApiError(404, "not_found", "数据版本不存在")
    model = ctx.db.get(ModelRegistry, payload.get("model_id")) if payload.get("model_id") else None

    workspace = ensure_dir(Path(ctx.settings.data_root) / "runs" / ctx.job.id)
    holder = _acquire_gpu(ctx)
    try:
        ctx.progress(0.05, "preparing", "准备训练数据")
        existing = ctx.db.query(ModelRegistry).filter(ModelRegistry.training_job_id == ctx.job.id).one_or_none()
        if existing:
            model_service.verify_model_file(existing, ctx.settings)
            result = (existing.metadata_json or {}).get("training_result")
            if not result:
                raise ApiError(409, "recovery_metadata_missing", "已有模型缺少完整训练记录，拒绝覆盖；需人工核对")
        else:
            result = training_service.train(
                ctx.settings,
                ctx.db,
                job_id=ctx.job.id,
                version=version,
                model=model,
                class_map=version.class_map_json or {},
                params=payload.get("params") or {},
                workdir=workspace,
                should_stop=ctx.should_stop,
                progress=ctx.progress,
            )
        ctx.check_cancelled()
        best_path = Path(result["best_path"])
        metrics = result.get("metrics") or {}
        ctx.lock_for_write()
        registered = model_service.register_trained_model(
            ctx.db,
            ctx.settings,
            job_id=ctx.job.id,
            version_id=version.id,
            owner_id=int(ctx.job.owner_user_id or 0),
            owner_username=ctx.job.owner_username,
            name=f"{version.dataset_id[:8]}-{ctx.job.id[:8]}",
            best_path=best_path,
            class_map=version.class_map_json or {},
            metrics=metrics,
            backend=result.get("backend", "unknown"),
            training_result=result,
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
                    "params": payload.get("params"),
                    "dataset_yaml": result.get("dataset_yaml"),
                    "dataset_manifest_sha256": result.get("dataset_manifest_sha256"),
                    "request_sha256": result.get("request_sha256"),
                    "effective_args": result.get("effective_args"),
                    "training_seconds": result.get("training_seconds"),
                    "peak_gpu_allocated_bytes": result.get("peak_gpu_allocated_bytes"),
                    "peak_gpu_reserved_bytes": result.get("peak_gpu_reserved_bytes"),
                },
                metrics_json=metrics,
                artifact_refs={
                    "model_sha256": registered.sha256,
                    "best_path": str(best_path),
                },
            )
        ctx.lock_for_write()
        ctx.db.add(experiment)
        ctx.db.commit()
        return {
            "model_id": registered.id,
            "model_sha256": registered.sha256,
            "metrics": metrics,
            "backend": result.get("backend"),
            "epochs_completed": result.get("epochs_completed"),
            "experiment_id": experiment.id,
            "dataset_manifest_sha256": version.manifest_sha256,
            "effective_args": result.get("effective_args"),
            "environment": result.get("environment") or environment_snapshot(),
            "training_seconds": result.get("training_seconds"),
            "peak_gpu_allocated_bytes": result.get("peak_gpu_allocated_bytes"),
            "peak_gpu_reserved_bytes": result.get("peak_gpu_reserved_bytes"),
        }
    finally:
        from app.workers.gpu import release
        release(ctx, holder)


def handle_evaluation(ctx: JobContext) -> dict:
    payload = ctx.job.input_json or {}
    run = ctx.db.get(EvaluationRun, payload["evaluation_id"])
    if run is None:
        raise ApiError(404, "not_found", "评估记录不存在")
    model = ctx.db.get(ModelRegistry, run.model_id)
    version = ctx.db.get(DatasetVersion, run.dataset_version_id)
    if model is None or version is None:
        raise ApiError(404, "not_found", "模型或数据版本不存在")
    path = model_service.verify_model_file(model, ctx.settings)

    workspace = ensure_dir(Path(ctx.settings.data_root) / "runs" / ctx.job.id)
    dataset_yaml = training_service.build_dataset_yaml(ctx.db, version, workspace)
    holder = _acquire_gpu(ctx)
    try:
        adapter = get_adapter(ctx.settings)
        handle = adapter.load(path, model.class_map_json or version.class_map_json or {})
        try:
            output = adapter.evaluate(handle, dataset_yaml, run.split, {**(run.params_json or {}), "_output_dir": str(workspace)})
        finally:
            adapter.release(handle)
        from app.domains.inference.service import record_artifact
        from app.core.storage import atomic_write_bytes
        import json
        import shutil
        artifacts = []
        metrics = output.get("metrics") or {}
        destination = ensure_dir(ctx.settings.artifact_root / "evaluations" / run.id / str(ctx.attempt))
        curves = metrics.pop("curves", None)
        if curves is not None:
            curve_path = destination / "curves.json"
            atomic_write_bytes(curve_path, json.dumps(curves, allow_nan=False).encode("utf-8"))
            ctx.lock_for_write()
            record = record_artifact(ctx.db, ctx.settings, owner_id=run.owner_id, resource_type="evaluation", resource_id=run.id, path=curve_path, mime="application/json")
            metrics["curves"] = {"artifact_id": record.id, "sha256": record.sha256}
        else:
            metrics["curves"] = None
        for raw in output.get("plots", []):
            source = Path(raw).resolve()
            source.relative_to(workspace.resolve())
            target = destination / source.name
            shutil.copy2(source, target)
            ctx.lock_for_write()
            record = record_artifact(ctx.db, ctx.settings, owner_id=run.owner_id, resource_type="evaluation", resource_id=run.id, path=target, mime="image/png")
            artifacts.append({"name": source.name, "artifact_id": record.id, "sha256": record.sha256})
        ctx.lock_for_write()
        run.status = "completed"
        run.metrics_json = metrics
        run.artifacts_json = {"backend": output.get("backend"), "reason": output.get("reason"), "plots": artifacts,
                              "environment": environment_snapshot(), "model_sha256": model.sha256,
                              "manifest_sha256": version.manifest_sha256, "split": run.split}
        run.context_hash = compute_sha256(dataset_yaml)
        run.finished_at = utcnow()
        ctx.db.commit()
        return {
            "evaluation_id": run.id,
            "split": run.split,
            "metrics": run.metrics_json,
            "reason": output.get("reason"),
        }
    finally:
        from app.workers.gpu import release
        release(ctx, holder)


def run_forever() -> None:  # pragma: no cover
    from app.workers.runtime import run_worker
    run_worker("trainer", WORKER_TYPES)


if __name__ == "__main__":  # pragma: no cover
    run_forever()
