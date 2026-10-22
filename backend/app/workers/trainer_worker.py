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

WORKER_TYPES = ("training", "evaluation")


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


def _acquire_gpu(ctx: JobContext, *, timeout_seconds: int = 120) -> str:
    holder = f"trainer:{os.getpid()}:{ctx.job.id[:8]}"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        ctx.check_cancelled()
        if lease_service.acquire_lease(
            ctx.db, lease_service.GPU_RESOURCE, holder=holder, job_id=ctx.job.id, ttl_seconds=ctx.settings.gpu_lease_seconds
        ):
            return holder
        ctx.progress(None, "waiting_gpu", "等待 GPU 租约")
        time.sleep(2)
    raise ApiError(503, "gpu_busy", "等待 GPU 租约超时")


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
        result = training_service.train(
            ctx.settings,
            ctx.db,
            job_id=ctx.job.id,
            version=version,
            model=model,
            class_map=version.class_map_json or {},
            params=payload.get("params") or {},
            workdir=workspace,
            should_stop=_cancel_checker(ctx.job.id),
            progress=ctx.progress,
        )
        ctx.check_cancelled()
        best_path = Path(result["best_path"])
        metrics = result.get("metrics") or {}
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
        )
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
            },
            metrics_json=metrics,
            artifact_refs={
                "model_sha256": registered.sha256,
                "best_path": str(best_path),
            },
        )
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
        }
    finally:
        lease_service.release_lease(ctx.db, lease_service.GPU_RESOURCE, holder=holder)


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
            output = adapter.evaluate(handle, dataset_yaml, run.split, run.params_json or {})
        finally:
            adapter.release(handle)
        run.status = "completed"
        run.metrics_json = output.get("metrics") or {}
        run.artifacts_json = {"backend": output.get("backend"), "reason": output.get("reason"), "dataset_yaml": str(dataset_yaml)}
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
        lease_service.release_lease(ctx.db, lease_service.GPU_RESOURCE, holder=holder)


def run_forever() -> None:  # pragma: no cover
    settings = __import__("app.core.config", fromlist=["get_settings"]).get_settings()
    logging.basicConfig(level="INFO")
    worker_id = f"trainer-{socket.gethostname()}-{os.getpid()}"
    factory = get_session_factory()
    lease = f"worker:trainer:{worker_id}"
    logger.info("trainer_worker_start id=%s", worker_id)
    while True:
        with factory() as db:
            lease_service.acquire_lease(db, lease, holder=worker_id, ttl_seconds=30)
            job_service.mark_stale_interrupted(db, older_than_seconds=max(120, settings.job_lease_seconds))
            job = job_service.claim_job(
                db, job_types=WORKER_TYPES, worker_id=worker_id, lease_seconds=settings.job_lease_seconds
            )
            if job:
                run_claimed_job(db, job, worker_id=worker_id, settings=settings)
            else:
                db.commit()
        time.sleep(settings.worker_poll_seconds)


if __name__ == "__main__":  # pragma: no cover
    run_forever()
