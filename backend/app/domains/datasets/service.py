"""数据集查询、归属校验与删除保护。"""
from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.core.errors import ApiError
from app.db.base import utcnow
from app.db.models import DatasetAsset, DatasetResource, DatasetVersion, JobTask, ModelRegistry


def get_dataset(db: Session, dataset_id: str, user: CurrentUser, *, manage: bool = False) -> DatasetResource:
    resource = db.get(DatasetResource, dataset_id)
    if resource is None or (resource.deleted_at is not None and not manage):
        raise ApiError(404, "not_found", "数据集不存在")
    if not user.is_admin and resource.owner_user_id != user.id:
        # 统一 404，避免通过状态码枚举他人资源
        raise ApiError(404, "not_found", "数据集不存在")
    return resource


def list_datasets(
    db: Session, user: CurrentUser, *, page: int, page_size: int, include_deleted: bool = False
) -> tuple[list[DatasetResource], int]:
    query = db.query(DatasetResource)
    if not user.is_admin:
        query = query.filter(DatasetResource.owner_user_id == user.id)
    if not include_deleted:
        query = query.filter(DatasetResource.deleted_at.is_(None))
    total = query.count()
    items = (
        query.order_by(DatasetResource.created_at.desc(), DatasetResource.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def get_version(db: Session, version_id: str, user: CurrentUser) -> DatasetVersion:
    version = db.get(DatasetVersion, version_id)
    if version is None or version.deleted_at is not None:
        raise ApiError(404, "not_found", "数据版本不存在")
    get_dataset(db, version.dataset_id, user)
    return version


def list_versions(db: Session, dataset_id: str, user: CurrentUser) -> list[DatasetVersion]:
    get_dataset(db, dataset_id, user)
    return (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset_id, DatasetVersion.deleted_at.is_(None))
        .order_by(DatasetVersion.created_at.desc())
        .all()
    )


def latest_version(db: Session, dataset_id: str) -> DatasetVersion | None:
    return (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset_id, DatasetVersion.deleted_at.is_(None))
        .order_by(DatasetVersion.created_at.desc())
        .first()
    )


def list_assets(
    db: Session, version_id: str, *, page: int, page_size: int
) -> tuple[list[DatasetAsset], int]:
    query = db.query(DatasetAsset).filter(
        DatasetAsset.dataset_version_id == version_id, DatasetAsset.deleted_at.is_(None)
    )
    total = query.count()
    items = (
        query.order_by(DatasetAsset.relative_path.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def _referenced(db: Session, version_ids: list[str]) -> list[str]:
    if not version_ids:
        return []
    usages: list[str] = []
    models = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.dataset_version_id.in_(version_ids))
        .count()
    )
    if models:
        usages.append(f"models:{models}")
    # JSON 查询在不同方言上行为不一致，这里对活跃作业做有界扫描（作业量可控）。
    wanted = set(version_ids)
    active_jobs = (
        db.query(JobTask)
        .filter(JobTask.status.in_(["pending", "running", "cancelling"]))
        .limit(1000)
        .all()
    )
    jobs = sum(1 for job in active_jobs if (job.input_json or {}).get("dataset_version_id") in wanted)
    if jobs:
        usages.append(f"jobs:{jobs}")
    children = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.parent_version_id.in_(version_ids), DatasetVersion.deleted_at.is_(None))
        .count()
    )
    if children:
        usages.append(f"child_versions:{children}")
    return usages


def delete_dataset(db: Session, resource: DatasetResource) -> None:
    versions = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == resource.id, DatasetVersion.deleted_at.is_(None))
        .all()
    )
    usages = _referenced(db, [version.id for version in versions])
    if usages:
        raise ApiError(409, "resource_referenced", "数据集已被训练/模型/子版本引用，不能删除", {"usages": usages})
    now = utcnow()
    resource.deleted_at = now
    for version in versions:
        version.deleted_at = now
    db.commit()


def restore_dataset(db: Session, resource: DatasetResource) -> None:
    resource.deleted_at = None
    for version in db.query(DatasetVersion).filter(DatasetVersion.dataset_id == resource.id).all():
        version.deleted_at = None
    db.commit()


def get_asset(db: Session, asset_id: int, user: CurrentUser) -> DatasetAsset:
    asset = db.get(DatasetAsset, asset_id)
    if asset is None or asset.deleted_at is not None:
        raise ApiError(404, "not_found", "图像资产不存在")
    get_version(db, asset.dataset_version_id, user)
    return asset


def delete_asset(db: Session, asset: DatasetAsset) -> None:
    version = db.get(DatasetVersion, asset.dataset_version_id)
    if version and version.state == "frozen":
        raise ApiError(409, "version_frozen", "冻结版本内容只读；请创建子版本后删除")
    if version and version.parent_version_id:
        pass
    asset.deleted_at = utcnow()
    db.commit()


def restore_asset(db: Session, asset: DatasetAsset) -> None:
    asset.deleted_at = None
    db.commit()
