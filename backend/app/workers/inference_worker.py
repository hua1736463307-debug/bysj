"""推理 worker：单卡 GPU 租约、逐项持久化、部分失败保留成功项。"""
from __future__ import annotations

import logging
import os
import socket
import time
from pathlib import Path

from app.core.errors import ApiError
from app.core.storage import ensure_dir
from app.db.base import utcnow
from app.db.models import DatasetAsset, InferenceItem, InferencePrediction, InferenceRun, ModelRegistry
from app.db.session import get_session_factory
from app.domains import job_service, lease_service
from app.domains.inference import service as inference_service
from app.domains.models import service as model_service
from app.ml.yolo_adapter import get_adapter
from app.workers.dispatcher import JobCancelled, JobContext, run_claimed_job

logger = logging.getLogger("bysj.worker.inference")

WORKER_TYPES = ("inference",)


def _acquire_gpu(ctx: JobContext, *, timeout_seconds: int = 120) -> str:
    holder = f"inference:{os.getpid()}:{ctx.job.id[:8]}"
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


def handle_inference(ctx: JobContext) -> dict:
    payload = ctx.job.input_json or {}
    run = ctx.db.get(InferenceRun, payload["run_id"])
    if run is None:
        raise ApiError(404, "not_found", "推理记录不存在")
    model = ctx.db.get(ModelRegistry, run.model_id)
    if model is None:
        raise ApiError(404, "not_found", "模型不存在")
    model_path = model_service.verify_model_file(model, ctx.settings)

    class_map = model.class_map_json or {}
    adapter = get_adapter(ctx.settings)
    holder = _acquire_gpu(ctx)
    try:
        load_start = time.perf_counter()
        handle = adapter.load(model_path, class_map)
        load_ms = (time.perf_counter() - load_start) * 1000

        items = (
            ctx.db.query(InferenceItem)
            .filter(InferenceItem.run_id == run.id, InferenceItem.status.in_(["pending", "failed"]))
            .order_by(InferenceItem.created_at.asc())
            .all()
        )
        for index, item in enumerate(items, start=1):
            try:
                ctx.check_cancelled()
            except JobCancelled:
                raise
            source = inference_service.resolve_item_path(ctx.db, ctx.settings, item)
            if not source.exists():
                raise FileNotFoundError("源图像不存在")
            process_start = time.perf_counter()
            try:
                predictions = adapter.predict_image(handle, source, run.params_json or {})
            except Exception as exc:
                item.status = "failed"
                item.error = f"{type(exc).__name__}: {exc}"[:1000]
                item.finished_at = utcnow()
                run.failed_count = int(run.failed_count or 0) + 1
                ctx.db.commit()
                ctx.event("item_failed", f"第 {index} 张失败", {"item_id": item.id})
                continue
            process_ms = (time.perf_counter() - process_start) * 1000

            target = ensure_dir(Path(ctx.settings.artifact_root) / "inference" / run.id) / f"{item.id}.jpg"
            try:
                inference_service.save_annotated_image(source, predictions, target)
            except Exception:
                target = None  # 结果图失败不影响预测落库

            for prediction in predictions:
                ctx.db.add(
                    InferencePrediction(
                        item_id=item.id,
                        class_id=int(prediction["class_id"]),
                        class_name=prediction.get("class_name"),
                        confidence=float(prediction.get("confidence") or 0.0),
                        bbox_json=[round(float(value), 2) for value in prediction["bbox_xyxy"]],
                        raw_json=prediction,
                    )
                )
            if target is not None:
                artifact = inference_service.record_artifact(
                    ctx.db,
                    ctx.settings,
                    owner_id=run.owner_id,
                    resource_type="inference_item",
                    resource_id=item.id,
                    path=target,
                    mime="image/jpeg",
                )
                item.artifact_id = artifact.id

            try:
                from PIL import Image

                with Image.open(source) as image:
                    item.image_width, item.image_height = image.size
            except Exception:
                pass
            item.status = "completed"
            item.error = None
            item.image_sha256 = item.image_sha256 or None
            item.timing_json = {
                "queue_ms": 0.0,
                "load_ms": round(load_ms, 3) if index == 1 else 0.0,
                "model_ms": round(process_ms, 3),
                "total_ms": round(process_ms, 3),
            }
            item.finished_at = utcnow()
            run.completed_count = int(run.completed_count or 0) + 1
            ctx.db.commit()
            ctx.progress(index / max(1, len(items)), "inferring", f"推理 {index}/{len(items)}")
        adapter.release(handle)

        run.status = "partial" if int(run.failed_count or 0) else "completed"
        run.finished_at = utcnow()
        ctx.db.commit()
        return {
            "run_id": run.id,
            "total": run.total_count,
            "completed": run.completed_count,
            "failed": run.failed_count,
            "status": run.status,
        }
    finally:
        lease_service.release_lease(ctx.db, lease_service.GPU_RESOURCE, holder=holder)


def run_forever() -> None:  # pragma: no cover
    from app.core.config import get_settings

    settings = get_settings()
    logging.basicConfig(level="INFO")
    worker_id = f"inference-{socket.gethostname()}-{os.getpid()}"
    factory = get_session_factory()
    lease = f"worker:inference:{worker_id}"
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
