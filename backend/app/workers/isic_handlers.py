"""皮肤镜方向作业处理器：导入与冻结。

复用既有 JobTask 与 dataset 角色 worker；不新建队列，也不占用 GPU
（导入/冻结是纯 I/O 与 CPU 工作）。
"""
from __future__ import annotations

from app.core.errors import ApiError
from app.db.models import DatasetResource, DatasetVersion
from app.directions.isic import ISIC_TASK
from app.domains.isic import import_service, split_service
from app.workers.dispatcher import JobContext


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
