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
    db: Session, user: CurrentUser, *, page: int, page_size: int, include_deleted: bool = False,
    task_type: str | None = None, search: str = "", owner: str = "",
) -> tuple[list[DatasetResource], int]:
    query = db.query(DatasetResource)
    if not user.is_admin:
        query = query.filter(DatasetResource.owner_user_id == user.id)
    if not include_deleted:
        query = query.filter(DatasetResource.deleted_at.is_(None))
    if task_type:
        query = query.filter(DatasetResource.task_type == task_type)
    if search.strip():
        query = query.filter(DatasetResource.name.contains(search.strip(), autoescape=True))
    if owner.strip():
        query = query.filter(DatasetResource.owner_username.contains(owner.strip(), autoescape=True))
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


def task_type_of_version(db: Session, version: DatasetVersion) -> str:
    resource = db.get(DatasetResource, version.dataset_id)
    return resource.task_type if resource is not None else "detect"


def assert_detect_task(db: Session, version: DatasetVersion) -> None:
    """BBox 冻结/切片/标注只服务 detect；CXR 版本必须走 `/api/v1/cxr`。"""
    task = task_type_of_version(db, version)
    if task != "detect":
        raise ApiError(
            409,
            "task_mismatch",
            "该版本不是检测任务版本，不能使用 BBox 冻结/切片/标注接口",
            {"task_type": task, "expected": "detect"},
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
    # Stream JSON references so secondary merge parents remain protected.
    wanted = set(version_ids)
    active_jobs = (
        db.query(JobTask)
        .filter(JobTask.status.in_(["pending", "running", "cancelling"]))
        .yield_per(200)
    )
    jobs = sum(1 for job in active_jobs if wanted.intersection(
        [(job.input_json or {}).get("dataset_version_id"), (job.input_json or {}).get("parent_version_id")] +
        list((job.input_json or {}).get("version_ids") or [])
    ))
    if jobs:
        usages.append(f"jobs:{jobs}")
    children = sum(1 for child in db.query(DatasetVersion).filter(DatasetVersion.deleted_at.is_(None)).yield_per(200)
                   if wanted.intersection([child.parent_version_id] + list((child.metadata_json or {}).get("parent_version_ids") or [])))
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


def get_asset(db: Session, asset_id: int, user: CurrentUser, *, include_deleted: bool = False) -> DatasetAsset:
    asset = db.get(DatasetAsset, asset_id)
    if asset is None or (asset.deleted_at is not None and not (include_deleted and user.is_admin)):
        raise ApiError(404, "not_found", "图像资产不存在")
    get_version(db, asset.dataset_version_id, user)
    return asset


def delete_asset(db: Session, asset: DatasetAsset) -> None:
    version = db.query(DatasetVersion).filter(DatasetVersion.id == asset.dataset_version_id).populate_existing().with_for_update().one()
    if version.state not in {"draft", "failed"}:
        raise ApiError(409, "version_frozen", "冻结版本内容只读；请创建子版本后删除")
    if version and version.parent_version_id:
        pass
    asset.deleted_at = utcnow()
    db.commit()


def restore_asset(db: Session, asset: DatasetAsset) -> None:
    version = db.query(DatasetVersion).filter(DatasetVersion.id == asset.dataset_version_id).populate_existing().with_for_update().one()
    if version.state not in {"draft", "failed"}:
        raise ApiError(409, "version_frozen", "冻结或冻结中的版本只读")
    asset.deleted_at = None
    db.commit()


def create_annotation_version(db, settings, dataset_id, base_version_id, user):
    """Fork an immutable detection version without changing its images or labels."""
    import copy
    import shutil
    import uuid
    from pathlib import Path
    from app.core.storage import resolve_under

    parent = get_version(db, base_version_id, user)
    if parent.dataset_id != dataset_id:
        raise ApiError(404, 'not_found', '该版本不属于此数据集')
    assert_detect_task(db, parent)
    if parent.state != 'frozen':
        raise ApiError(409, 'version_invalid_state', '请选择已冻结版本创建标注副本')
    source = Path(parent.root_path).resolve()
    try:
        relative = source.relative_to(settings.data_root.resolve())
        if not relative.parts:
            raise ValueError()
    except ValueError as exc:
        raise ApiError(422, 'invalid_version_root', '版本目录不在数据根目录内') from exc
    child_id = str(uuid.uuid4())
    target = resolve_under(settings.data_root, f'annotation-versions/{child_id}')
    assets = db.query(DatasetAsset).filter(
        DatasetAsset.dataset_version_id == parent.id, DatasetAsset.deleted_at.is_(None)
    ).order_by(DatasetAsset.id).all()
    try:
        target.mkdir(parents=True)
        child = DatasetVersion(
            id=child_id, dataset_id=dataset_id, parent_version_id=parent.id,
            dataset_role=parent.dataset_role, status='ready', state='draft',
            root_path=str(target), class_map_json=copy.deepcopy(parent.class_map_json),
            metadata_json={**copy.deepcopy(parent.metadata_json or {}), 'annotation_parent_version_id': parent.id, 'preserve_parent_splits': True},
            split_config_json=copy.deepcopy(parent.split_config_json or {}),
            split_counts_json=copy.deepcopy(parent.split_counts_json or {}),
            content_hash=child_id, asset_count=len(assets), created_by=user.username,
            created_by_user_id=user.id,
        )
        db.add(child)
        for asset in assets:
            src = resolve_under(source, f'images/{asset.relative_path}', must_exist=True)
            dst = resolve_under(target, f'images/{asset.relative_path}')
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            db.add(DatasetAsset(
                dataset_version_id=child_id, relative_path=asset.relative_path,
                basename=asset.basename, size=asset.size, width=asset.width, height=asset.height,
                source_group=asset.source_group, content_hash=asset.content_hash, split=asset.split,
                annotations_json=copy.deepcopy(asset.annotations_json or []),
                root_asset_id=asset.root_asset_id or asset.id, original_asset_id=asset.id,
                group_level=asset.group_level, transform_json=copy.deepcopy(asset.transform_json or {}),
            ))
        db.commit()
        return child
    except Exception:
        db.rollback()
        if target.exists():
            shutil.rmtree(target)
        raise
