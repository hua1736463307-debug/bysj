"""非 GPU 数据集 worker：导入、冻结、导出。

不触碰 GPU 领取函数；命令：
``python -m app.workers.dataset_worker``（容器内）
"""
from __future__ import annotations

import logging
import os
import socket
import time
import json
import zipfile
from pathlib import Path, PurePosixPath

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import ApiError
from app.db.models import DatasetAsset, DatasetVersion, JobTask
from app.domains import job_service, lease_service
from app.domains.datasets import import_service, service, version_service
from app.workers.dispatcher import JobContext, JobCancelled, run_claimed_job

logger = logging.getLogger("bysj.worker.dataset")

WORKER_TYPES = ("dataset_import", "dataset_version", "dataset_process", "annotation_export", "feedback_export", "dataset_export")


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
        seed=int(payload["split_seed"] if payload.get("split_seed") is not None else 42),
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
    snapshot = []
    for index, asset in enumerate(assets, start=1):
        snapshot.append({"asset_id": asset.id, "relative_path": asset.relative_path,
                         "sha256": asset.content_hash, "annotations": asset.annotations_json or []})
        ctx.progress(index / max(1, len(assets)), "exporting", f"导出 {index}/{len(assets)}")
    from app.core.storage import atomic_write_bytes, ensure_dir, normalize_relative
    from app.domains.inference.service import record_artifact
    directory = ensure_dir(ctx.settings.artifact_root / "annotations" / ctx.job.id / str(ctx.attempt))
    export = {"schema_version": 1, "dataset_id": version.dataset_id, "version_id": version.id,
              "class_map": version.class_map_json, "assets": snapshot}
    fmt = payload.get("format", "yolo-txt")
    if fmt == "json":
        path = directory / "annotations.json"
        atomic_write_bytes(path, json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"))
        mime = "application/json"
    else:
        path = directory / "labels.zip"
        names = set()
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(export, ensure_ascii=False, indent=2))
            for item in snapshot:
                ctx.check_cancelled()
                name = "labels/" + normalize_relative(item["relative_path"]).with_suffix(".txt").as_posix()
                if name in names:
                    raise ApiError(409, "label_name_collision", "不同图像导出为同名标签，请使用 JSON 导出")
                names.add(name)
                body = "".join(f"{a['class_id']} {a['cx']:.6f} {a['cy']:.6f} {a['w']:.6f} {a['h']:.6f}\n" for a in item["annotations"])
                archive.writestr(name, body)
        mime = "application/zip"
    ctx.lock_for_write()
    record = record_artifact(ctx.db, ctx.settings, owner_id=ctx.job.owner_user_id,
        resource_type="annotation_export", resource_id=ctx.job.id, path=path, mime=mime)
    return {
        "version_id": version.id,
        "asset_count": len(assets),
        "annotation_count": sum(len(item["annotations"]) for item in snapshot),
        "format": fmt,
        "artifact_id": record.id,
        "sha256": record.sha256,
        "download_url": f"/api/v1/artifacts/{record.id}",
    }


def handle_feedback_export(ctx: JobContext) -> dict:
    from app.domains.feedback import export_service

    payload = ctx.job.input_json or {}
    return export_service.export_feedback_version(ctx.db, ctx.settings, payload, ctx.progress)


def run_forever() -> None:  # pragma: no cover
    from app.workers.runtime import run_worker
    run_worker("dataset", WORKER_TYPES)


if __name__ == "__main__":  # pragma: no cover
    run_forever()
