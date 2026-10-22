"""v1 数据集与版本接口。"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user, settings_dep
from app.core.config import Settings
from app.core.errors import ApiError, accepted, ok
from app.core.storage import remove_tree, resolve_under
from app.db.models import DatasetAsset, DatasetResource, DatasetVersion
from app.db.session import get_db
from app.domains import job_service
from app.domains.datasets import import_service, service
from app.schemas.common import MAX_PAGE_SIZE, DEFAULT_PAGE_SIZE

router = APIRouter(tags=["datasets"])


def dataset_dict(resource: DatasetResource, version: DatasetVersion | None) -> dict:
    return {
        "dataset_id": resource.id,
        "name": resource.name,
        "description": resource.description,
        "modality": resource.modality,
        "task_type": resource.task_type,
        "group_level": resource.group_level,
        "source": {
            "url": resource.source_url,
            "version": resource.source_version,
            "license": resource.license,
        },
        "owner": resource.owner_username,
        "created_at": resource.created_at.isoformat() if resource.created_at else None,
        "latest_version": version_dict(version) if version else None,
    }


def version_dict(version: DatasetVersion) -> dict:
    return {
        "version_id": version.id,
        "dataset_id": version.dataset_id,
        "parent_version_id": version.parent_version_id,
        "state": version.state,
        "asset_count": version.asset_count,
        "class_map": version.class_map_json or {},
        "split_counts": version.split_counts_json or {},
        "split_config": version.split_config_json or {},
        "manifest_sha256": version.manifest_sha256,
        "split_manifest_hash": version.split_manifest_hash,
        "class_map_hash": version.class_map_hash,
        "frozen_at": version.frozen_at.isoformat() if version.frozen_at else None,
        "created_at": version.created_at.isoformat() if version.created_at else None,
    }


def asset_dict(asset: DatasetAsset) -> dict:
    return {
        "asset_id": asset.id,
        "relative_path": asset.relative_path,
        "basename": asset.basename,
        "width": asset.width,
        "height": asset.height,
        "size": asset.size,
        "sha256": asset.content_hash,
        "source_group": asset.source_group,
        "split": asset.split,
        "annotations": asset.annotations_json or [],
    }


@router.post("/datasets/import", response_model=None)
def import_dataset(
    files: list[UploadFile] = File(...),
    name: str = Form(default=""),
    source_url: str | None = Form(default=None),
    source_version: str | None = Form(default=None),
    license_text: str | None = Form(default=None, alias="license"),
    modality: str | None = Form(default=None),
    group_level: str = Form(default="original_image"),
    class_map: str | None = Form(default=None),
    allow_empty_labels: str = Form(default="true"),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    if not files:
        raise ApiError(422, "no_files", "至少上传一个文件")
    token = str(uuid.uuid4())
    staging = import_service.staging_root(settings, token)
    try:
        for upload in files:
            if not upload.filename:
                continue
            import_service.save_upload(settings, staging, upload.filename, upload.file)
    except ApiError:
        remove_tree(staging, allowed_root=settings.data_root / "staging")
        raise

    parsed_class_map = None
    if class_map:
        try:
            parsed_class_map = json.loads(class_map)
        except json.JSONDecodeError as exc:
            remove_tree(staging, allowed_root=settings.data_root / "staging")
            raise ApiError(422, "invalid_class_map", "class_map 必须是 JSON 对象") from exc

    payload = {
        "staging_token": token,
        "name": name or "导入数据集",
        "source_url": source_url,
        "source_version": source_version,
        "license": license_text,
        "modality": modality,
        "group_level": group_level,
        "class_map": parsed_class_map,
        "allow_empty_labels": str(allow_empty_labels).strip().lower() not in {"false", "0", "no"},
    }
    job, created = job_service.create_job(
        db,
        job_type="dataset_import",
        owner_id=user.id,
        owner_username=user.username,
        payload=payload,
    )
    if settings.inline_jobs and created:
        from app.workers.dispatcher import execute_job_inline

        execute_job_inline(job.id)
        db.refresh(job)
    return accepted(job.to_public(), message="导入任务已接收")


@router.get("/datasets", response_model=None)
def list_datasets(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    items, total = service.list_datasets(db, user, page=page, page_size=page_size)
    return ok(
        {
            "items": [
                dataset_dict(item, service.latest_version(db, item.id)) for item in items
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    )


@router.get("/datasets/{dataset_id}", response_model=None)
def get_dataset(
    dataset_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    resource = service.get_dataset(db, dataset_id, user)
    return ok(dataset_dict(resource, service.latest_version(db, resource.id)))


@router.get("/datasets/{dataset_id}/versions", response_model=None)
def list_versions(
    dataset_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    versions = service.list_versions(db, dataset_id, user)
    return ok({"items": [version_dict(version) for version in versions], "total": len(versions)})


@router.get("/datasets/{dataset_id}/assets", response_model=None)
def list_assets(
    dataset_id: str,
    version_id: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    resource = service.get_dataset(db, dataset_id, user)
    version = service.get_version(db, version_id, user) if version_id else service.latest_version(db, resource.id)
    if version is None:
        raise ApiError(404, "not_found", "数据集还没有版本")
    if version.dataset_id != resource.id:
        raise ApiError(404, "not_found", "该版本不属于此数据集")
    items, total = service.list_assets(db, version.id, page=page, page_size=page_size)
    return ok(
        {
            "items": [asset_dict(item) for item in items],
            "total": total,
            "page": page,
            "page_size": page_size,
            "version_id": version.id,
        }
    )


class VersionCreateRequest(BaseModel):
    base_version_id: str | None = None
    split_seed: int = Field(default=42, ge=0)
    ratios: list[float] = Field(default_factory=lambda: [0.8, 0.1, 0.1])


class TrainTilesRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    parent_version_id: str
    tile_size: int = Field(default=320, ge=64, le=2048)
    overlap: float = Field(default=.2, ge=0, le=.75)
    min_visibility: float = Field(default=.5, gt=0, le=1)


@router.post('/datasets/{dataset_id}/train-tile-versions', response_model=None)
def create_train_tiles(dataset_id: str, payload: TrainTilesRequest,
    idempotency_key: str = Header(..., alias='Idempotency-Key', min_length=1, max_length=128),
    user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db), settings: Settings = Depends(settings_dep)):
    service.get_dataset(db,dataset_id,user)
    parent=service.get_version(db,payload.parent_version_id,user)
    if parent.dataset_id!=dataset_id:raise ApiError(404,'not_found','父版本不属于该数据集')
    if parent.state!='frozen':raise ApiError(409,'parent_not_frozen','先冻结父版本再创建训练切片')
    job,created=job_service.create_job(db,job_type='dataset_process',owner_id=user.id,owner_username=user.username,
        payload=payload.model_dump(),idempotency_key=idempotency_key)
    if created and settings.inline_jobs:
        from app.workers.dispatcher import execute_job_inline
        execute_job_inline(job.id);db.refresh(job)
    return accepted(job.to_public(),message='训练集切片任务已接收，验证/测试图像保持原样')


@router.post("/datasets/{dataset_id}/versions", response_model=None)
def create_version(
    dataset_id: str,
    payload: VersionCreateRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    resource = service.get_dataset(db, dataset_id, user)
    version = (
        service.get_version(db, payload.base_version_id, user)
        if payload.base_version_id
        else service.latest_version(db, resource.id)
    )
    if version is None:
        raise ApiError(404, "not_found", "数据集还没有版本")
    if len(payload.ratios) != 3 or any(value <= 0 for value in payload.ratios):
        raise ApiError(422, "invalid_ratios", "三集比例必须是三个正数")
    if version.dataset_id != resource.id:
        raise ApiError(404, "not_found", "该版本不属于此数据集")

    job, created = job_service.create_job(
        db,
        job_type="dataset_version",
        owner_id=user.id,
        owner_username=user.username,
        payload={
            "dataset_id": resource.id,
            "version_id": version.id,
            "dataset_version_id": version.id,
            "split_seed": payload.split_seed,
            "ratios": payload.ratios,
        },
    )
    if settings.inline_jobs and created:
        from app.workers.dispatcher import execute_job_inline

        execute_job_inline(job.id)
        db.refresh(job)
    return accepted(job.to_public(), message="版本冻结任务已接收")


@router.get("/dataset-versions/{version_id}", response_model=None)
def get_version(
    version_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    version = service.get_version(db, version_id, user)
    return ok(version_dict(version))


@router.delete("/datasets/{dataset_id}", response_model=None)
def delete_dataset(
    dataset_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    resource = service.get_dataset(db, dataset_id, user)
    service.delete_dataset(db, resource)
    return ok({"dataset_id": dataset_id, "deleted": True}, message="数据集已删除")


@router.post("/datasets/{dataset_id}/restore", response_model=None)
def restore_dataset(
    dataset_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not user.is_admin:
        raise ApiError(403, "forbidden", "只有管理员可以恢复数据集")
    resource = service.get_dataset(db, dataset_id, user, manage=True)
    service.restore_dataset(db, resource)
    return ok({"dataset_id": dataset_id, "restored": True}, message="数据集已恢复")


@router.delete("/assets/{asset_id}", response_model=None)
def delete_asset(
    asset_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    asset = service.get_asset(db, asset_id, user)
    service.delete_asset(db, asset)
    return ok({"asset_id": asset_id, "deleted": True}, message="图像资产已删除")


@router.get("/assets/{asset_id}/image", response_model=None)
def asset_image(asset_id: int, user: CurrentUser = Depends(get_current_user),
                db: Session = Depends(get_db), settings: Settings = Depends(settings_dep)):
    asset = service.get_asset(db, asset_id, user)
    version = service.get_version(db, asset.dataset_version_id, user)
    root = Path(version.root_path).resolve()
    try:
        root.relative_to(settings.data_root.resolve())
    except ValueError as exc:
        raise ApiError(409, "invalid_path", "图像路径超出数据目录") from exc
    path = resolve_under(root / "images", asset.relative_path, must_exist=True)
    return FileResponse(path, headers={"Cache-Control": "private, no-store"})


@router.post("/assets/{asset_id}/restore", response_model=None)
def restore_asset(
    asset_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not user.is_admin:
        raise ApiError(403, "forbidden", "只有管理员可以恢复资产")
    asset = service.get_asset(db, asset_id, user, include_deleted=True)
    service.restore_asset(db, asset)
    return ok({"asset_id": asset_id, "restored": True}, message="图像资产已恢复")
