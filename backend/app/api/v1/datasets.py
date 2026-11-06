"""v1 数据集与版本接口。"""
from __future__ import annotations

import json
import uuid
from typing import Literal
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user, require_reviewer, settings_dep
from app.core.config import Settings
from app.core.errors import ApiError, accepted, ok
from app.core.storage import remove_tree, resolve_under
from app.db.models import DatasetAsset, DatasetResource, DatasetVersion
from app.db.session import get_db
from app.domains import job_service
from app.domains.datasets import import_service, service
from app.schemas.common import MAX_PAGE_SIZE, DEFAULT_PAGE_SIZE

router = APIRouter(tags=["datasets"])


@router.get("/assets/{asset_id}/augmentation-preview", response_model=None)
def augmentation_preview(
    asset_id: int, operation: Literal["rotate", "flip", "brightness", "contrast", "grayscale"],
    value: float = 1, user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    import io
    from PIL import Image, ImageDraw
    from fastapi.responses import Response
    from app.domains.datasets.derivative_service import transform
    if operation == "rotate" and value not in (90, 180, 270):
        raise ApiError(422, "invalid_rotation", "旋转角度支持 90、180、270")
    if operation in {"brightness", "contrast"} and not 0 < value <= 3:
        raise ApiError(422, "invalid_factor", "倍率须大于 0 且不超过 3")
    asset = service.get_asset(db, asset_id, user)
    version = service.get_version(db, asset.dataset_version_id, user)
    root = Path(version.root_path).resolve()
    if root == settings.data_root.resolve() or not root.is_relative_to(settings.data_root.resolve()):
        raise ApiError(422, "invalid_path", "版本路径越界")
    source = resolve_under(root / "images", asset.relative_path, must_exist=True)
    with Image.open(source) as original:
        image = original.convert("RGB")
        image.thumbnail((512, 512))
        image, boxes = transform(image, asset.annotations_json or [], operation, value)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        if box.get("polygon"):
            draw.polygon([(x * image.width, y * image.height) for x, y in box["polygon"]], outline="#21e0d0", width=2)
        else:
            x, y, w, h = (box[key] for key in ("cx", "cy", "w", "h"))
            draw.rectangle(((x - w / 2) * image.width, (y - h / 2) * image.height,
                            (x + w / 2) * image.width, (y + h / 2) * image.height), outline="#21e0d0", width=2)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(buffer.getvalue(), media_type="image/png", headers={"Cache-Control": "private, no-store"})


class DerivativeRequest(BaseModel):
    operation: Literal["merge", "augment", "preannotate"]
    version_ids: list[str] = Field(min_length=1, max_length=8)
    name: str = Field(default="合并数据集", min_length=1, max_length=255)
    rotations: list[Literal[90, 180, 270]] = Field(default_factory=list, max_length=3)
    brightness: list[float] = Field(default_factory=list, max_length=3)
    contrast: list[float] = Field(default_factory=list, max_length=3)
    flip: bool = False
    grayscale: bool = False
    model_id: str | None = None
    asset_ids: list[int] = Field(default_factory=list, max_length=100)
    confidence: float = Field(default=.5, ge=0, le=1)
    class_mapping: dict[str, int] = Field(default_factory=dict)
    annotation_mode: Literal["append", "replace"] = "append"


@router.post("/dataset-derivatives", response_model=None)
def create_derivative(
    payload: DerivativeRequest, idempotency_key: str = Header(alias="Idempotency-Key", min_length=1, max_length=128),
    user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    from app.domains.models.service import get_model
    if payload.operation == "preannotate" and not user.is_admin and "annotations" not in user.modules:
        raise ApiError(403, "module_forbidden", "需要数据标注模块权限")
    versions = [service.get_version(db, value, user) for value in payload.version_ids]
    if len(set(payload.version_ids)) != len(versions):
        raise ApiError(422, "duplicate_version", "来源版本不能重复")
    if payload.operation == "merge" and len(versions) < 2:
        raise ApiError(422, "merge_sources", "合并至少需要两个版本")
    if payload.operation != "merge" and len(versions) != 1:
        raise ApiError(422, "single_source_required", "此操作需要单个来源版本")
    for version in versions:
        service.assert_detect_task(db, version)
        if version.state != "frozen":
            raise ApiError(409, "parent_not_frozen", "请先在版本管理中冻结来源版本")
        if version.class_map_json != versions[0].class_map_json:
            raise ApiError(422, "class_map_mismatch", "类别映射不同的版本不能合并")
        if version.split_config_json.get("group_level") != versions[0].split_config_json.get("group_level"):
            raise ApiError(422, "group_level_mismatch", "分组级别不同的版本不能合并")
    factors = payload.brightness + payload.contrast
    if any(not (0 < value <= 3) for value in factors):
        raise ApiError(422, "invalid_factor", "亮度和对比度倍率须大于 0 且不超过 3")
    if payload.operation == "augment" and not (payload.rotations or factors or payload.flip or payload.grayscale):
        raise ApiError(422, "empty_augmentation", "请选择至少一种增强")
    if payload.operation == "preannotate":
        model = get_model(db, payload.model_id or "", user)
        if model.model_task != "detect":
            raise ApiError(422, "task_mismatch", "预标注需要目标检测模型")
        target_map = versions[0].class_map_json
        if any(key not in model.class_map_json or str(value) not in target_map for key, value in payload.class_mapping.items()):
            raise ApiError(422, "invalid_mapping", "类别映射包含未知源类别或目标类别")
        if not payload.class_mapping and not set(model.class_map_json.values()).intersection(target_map.values()):
            raise ApiError(422, "mapping_required", "模型与数据集类别名称无交集，请提供类别映射")
        if payload.asset_ids:
            count = db.query(DatasetAsset).filter(DatasetAsset.id.in_(payload.asset_ids),
                DatasetAsset.dataset_version_id == versions[0].id, DatasetAsset.deleted_at.is_(None)).count()
            if count != len(set(payload.asset_ids)):
                raise ApiError(404, "not_found", "所选图片不属于来源版本")
    data = payload.model_dump()
    data["dataset_version_id"] = versions[0].id
    job_type = "annotation_preannotate" if payload.operation == "preannotate" else "dataset_derive"
    job, created = job_service.create_job(db, job_type=job_type, owner_id=user.id,
        owner_username=user.username, payload=data, idempotency_key=idempotency_key)
    if created and settings.inline_jobs:
        from app.workers.dispatcher import execute_job_inline
        execute_job_inline(job.id)
        db.refresh(job)
    return accepted(job.to_public(), message="派生版本任务已入队")


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
        "requires_annotation_review": bool((version.metadata_json or {}).get("requires_annotation_review")),
        "preserve_parent_splits": bool((version.metadata_json or {}).get("preserve_parent_splits")),
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
    source_groups: str | None = Form(default=None),
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

    parsed_groups = None
    if source_groups is not None:
        try:
            if len(source_groups.encode('utf-8')) > 1024 * 1024:
                raise ValueError('source_groups too large')
            parsed_groups = json.loads(source_groups)
            if not isinstance(parsed_groups, dict):
                raise ValueError('source_groups must be an object')
        except (ValueError, TypeError) as exc:
            remove_tree(staging, allowed_root=settings.data_root / "staging")
            raise ApiError(422, "invalid_source_groups", "source_groups 必须是至多 1 MiB 的 JSON 对象") from exc

    payload = {
        "staging_token": token,
        "name": name or "导入数据集",
        "source_url": source_url,
        "source_version": source_version,
        "license": license_text,
        "modality": modality,
        "group_level": group_level,
        "source_groups": parsed_groups,
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
    search: str = Query(default="", max_length=255),
    owner: str = Query(default="", max_length=64),
    task: str | None = Query(default=None, description="按 task_type 过滤，例如 detect / cxr_multilabel"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    items, total = service.list_datasets(db, user, page=page, page_size=page_size, task_type=task, search=search, owner=owner)
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
    service.assert_detect_task(db, parent)
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
    service.assert_detect_task(db, version)
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


class AnnotationVersionRequest(BaseModel):
    base_version_id: str


class ClassMapRequest(BaseModel):
    expected_class_map: dict[str, str]
    class_map: dict[str, str]


@router.put("/dataset-versions/{version_id}/classes", response_model=None)
def update_class_map(
    version_id: str, payload: ClassMapRequest,
    user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db),
):
    from app.db.models import AnnotationDraft, User
    version = service.get_version(db, version_id, user)
    service.assert_detect_task(db, version)
    version = db.query(DatasetVersion).filter_by(id=version.id).populate_existing().with_for_update().one()
    if version.state not in {"draft", "failed"}:
        raise ApiError(409, "version_frozen", "冻结版本的类别只读，请先创建标注副本")
    if payload.expected_class_map != version.class_map_json:
        raise ApiError(409, "class_map_conflict", "类别已被其他会话修改，请刷新")
    classes = {key: value.strip() for key, value in payload.class_map.items()}
    if not 1 <= len(classes) <= 256 or any(
        not key.isdigit() or str(int(key)) != key or not value or len(value) > 64 for key, value in classes.items()
    ) or len(set(classes.values())) != len(classes):
        raise ApiError(422, "invalid_class_map", "类别需使用非负整数 ID 和不重复的非空名称，最多 256 类")
    if sorted(map(int, classes)) != list(range(len(classes))):
        raise ApiError(422, "invalid_class_map", "检测类别 ID 必须从 0 连续编号")
    assets = db.query(DatasetAsset).filter_by(dataset_version_id=version.id).yield_per(200)
    used = {str(box["class_id"]) for asset in assets for box in (asset.annotations_json or [])}
    if not used.issubset(classes):
        raise ApiError(409, "class_referenced", "待删除类别仍被标注引用，请先修改对应标注")
    from app.domains.annotation.service import advance_revision
    revision = advance_revision(db, version)
    version.class_map_json = classes
    resource = db.get(DatasetResource, version.dataset_id)
    owners = db.query(User).filter((User.role == "admin") | (User.id == resource.owner_user_id)).all()
    drafts = db.query(AnnotationDraft).filter_by(source_version_id=version.id).all()
    for draft in drafts:
        draft.revision = revision
        draft.class_names_json = list(classes.values())
    existing = {draft.owner_user_id for draft in drafts if draft.annotation_geometry == "rect"}
    for owner in owners:
        if owner.id not in existing:
            db.add(AnnotationDraft(owner_user_id=owner.id, owner_username=owner.username,
                dataset_id=version.dataset_id, source_version_id=version.id, annotation_geometry="rect",
                class_names_json=list(classes.values()), revision=revision))
    db.commit()
    return ok(version_dict(version), message="类别已更新")


@router.post("/dataset-versions/{version_id}/annotation-review", response_model=None)
def confirm_annotation_review(
    version_id: str, user: CurrentUser = Depends(require_reviewer), db: Session = Depends(get_db),
):
    from app.db.base import utcnow
    version = service.get_version(db, version_id, user)
    version = db.query(DatasetVersion).filter_by(id=version.id).with_for_update().one()
    if version.state not in {"draft", "failed"}:
        raise ApiError(409, "version_frozen", "只可确认草稿版本")
    version.metadata_json = {**(version.metadata_json or {}), "requires_annotation_review": False,
                             "annotation_review": {"user_id": user.id, "at": utcnow().isoformat()}}
    db.commit()
    return ok(version_dict(version), message="人工检查确认已记录")


@router.post('/datasets/{dataset_id}/annotation-versions', response_model=None)
def create_annotation_version(
    dataset_id: str, payload: AnnotationVersionRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db), settings: Settings = Depends(settings_dep),
):
    service.get_dataset(db, dataset_id, user)
    child = service.create_annotation_version(db, settings, dataset_id, payload.base_version_id, user)
    return ok(version_dict(child), message='可编辑标注版本已创建')
