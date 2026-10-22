"""非 GPU 数据集 worker：导入、冻结、导出。

不触碰 GPU 领取函数；命令：
``python -m app.workers.dataset_worker``（容器内）
"""
from __future__ import annotations

import logging
import os
import socket
import time

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import ApiError
from app.db.models import DatasetAsset, DatasetVersion, JobTask
from app.domains import job_service, lease_service
from app.domains.datasets import import_service, service, version_service
from app.workers.dispatcher import JobContext, JobCancelled, run_claimed_job

logger = logging.getLogger("bysj.worker.dataset")

WORKER_TYPES = ("dataset_import", "dataset_version", "annotation_export", "feedback_export", "dataset_export")


def handle_dataset_import(ctx: JobContext) -> dict:
    payload = ctx.job.input_json or {}
    ctx.progress(0.05, "validating", "校验上传内容")
    ctx.check_cancelled()
    result = import_service.import_dataset(
        ctx.db,
        ctx.settings,
        staging_token=payload["staging_token"],
        payload=payload,
        owner_id=int(ctx.job.owner_user_id or 0),
        owner_username=ctx.job.owner_username,
        progress=ctx.progress,
    )
    ctx.check_cancelled()
    return result


def handle_dataset_version(ctx: JobContext) -> dict:
    payload = ctx.job.input_json or {}
    version = ctx.db.get(DatasetVersion, payload["version_id"])
    if version is None:
        raise ApiError(404, "not_found", "数据版本不存在")
    ctx.progress(0.1, "validating", "检查资产与来源组")
    result = version_service.freeze_version(
        ctx.db,
        ctx.settings,
        version=version,
        seed=int(payload.get("split_seed") or 42),
        ratios=payload.get("ratios") or (0.8, 0.1, 0.1),
        progress=ctx.progress,
    )
    return result


def handle_annotation_export(ctx: JobContext) -> dict:
    """把冻结/草稿版本导出为 YOLO 目录（后台执行，避免前端拼巨量 ZIP）。"""
    payload = ctx.job.input_json or {}
    version = ctx.db.get(DatasetVersion, payload["version_id"])
    if version is None:
        raise ApiError(404, "not_found", "数据版本不存在")
    assets = (
        ctx.db.query(DatasetAsset)
        .filter(DatasetAsset.dataset_version_id == version.id, DatasetAsset.deleted_at.is_(None))
        .order_by(DatasetAsset.relative_path.asc())
        .all()
    )
    lines = []
    for index, asset in enumerate(assets, start=1):
        for ann in asset.annotations_json or []:
            lines.append(
                f"{asset.relative_path} {ann['class_id']} {ann['cx']:.6f} {ann['cy']:.6f} {ann['w']:.6f} {ann['h']:.6f}"
            )
        ctx.progress(index / max(1, len(assets)), "exporting", f"导出 {index}/{len(assets)}")
    body = "\n".join(lines) + ("\n" if lines else "")
    return {
        "version_id": version.id,
        "asset_count": len(assets),
        "annotation_count": len(lines),
        "format": "yolo-txt",
        "content_preview_sha256": __import__("hashlib").sha256(body.encode()).hexdigest(),
    }


def handle_feedback_export(ctx: JobContext) -> dict:
    from app.domains.feedback import export_service

    payload = ctx.job.input_json or {}
    return export_service.export_feedback_version(ctx.db, ctx.settings, payload, ctx.progress)


def run_forever() -> None:  # pragma: no cover - 进程入口
    settings = get_settings()
    logging.basicConfig(level="INFO")
    worker_id = f"dataset-{socket.gethostname()}-{os.getpid()}"
    from app.db.session import get_session_factory

    factory = get_session_factory()
    lease = f"worker:dataset:{worker_id}"
    logger.info("dataset_worker_start id=%s types=%s", worker_id, WORKER_TYPES)
    while True:
        with factory() as db:
            lease_service.acquire_lease(db, lease, holder=worker_id, ttl_seconds=30)
            job_service.mark_stale_interrupted(db, older_than_seconds=max(120, settings.job_lease_seconds))
            job = job_service.claim_job(
                db, job_types=WORKER_TYPES, worker_id=worker_id, lease_seconds=settings.job_lease_seconds
            )
            if job:
                logger.info("claimed job_id=%s type=%s", job.id, job.type)
                run_claimed_job(db, job, worker_id=worker_id, settings=settings)
            else:
                db.commit()
        time.sleep(settings.worker_poll_seconds)


if __name__ == "__main__":  # pragma: no cover
    run_forever()
