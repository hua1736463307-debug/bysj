"""v1 模型注册与受控下载。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user, require_admin, settings_dep
from app.core.config import Settings
from app.core.errors import ApiError, created, ok
from app.db.session import get_db
from app.domains.models import service as model_service
from app.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/upload", response_model=None)
def upload_model(
    file: UploadFile = File(...), name: str = Form(..., min_length=1, max_length=255),
    sha256: str = Form(..., min_length=64, max_length=64), class_map: str = Form(...),
    user: CurrentUser = Depends(require_admin), db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    import hashlib
    import json
    import uuid
    from pathlib import Path
    from app.core.storage import ensure_dir, remove_tree
    from app.db.models import ModelRegistry
    if Path(file.filename or "").suffix.lower() != ".pt":
        raise ApiError(422, "unsupported_model", "此检测模型入口支持 .pt 权重")
    if not name.strip():
        raise ApiError(422, "invalid_name", "模型名称不能为空")
    try:
        classes = json.loads(class_map)
        if not isinstance(classes, dict) or not 1 <= len(classes) <= 256:
            raise ValueError()
        if any(not key.isdigit() or str(int(key)) != key or not isinstance(value, str) or not value.strip() or len(value) > 64 for key, value in classes.items()):
            raise ValueError()
    except (ValueError, TypeError):
        raise ApiError(422, "invalid_class_map", "请提供有效的类别 ID 到名称映射")
    identifier = str(uuid.uuid4())
    directory = ensure_dir(settings.model_root / "registry" / identifier)
    target = directory / "model.pt"
    digest, size = hashlib.sha256(), 0
    try:
        with target.open("xb") as sink:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise ApiError(413, "model_too_large", "模型文件超过上传上限")
                digest.update(chunk)
                sink.write(chunk)
        if size == 0 or digest.hexdigest() != sha256.lower():
            raise ApiError(422, "hash_mismatch", "模型文件为空或 SHA256 与声明不一致")
        model = ModelRegistry(id=identifier, name=name.strip(), path=str(target),
            sha256=digest.hexdigest(), size=size, source="upload", trusted_source="admin_upload",
            owner_id=user.id, created_by=user.username, model_task="detect", class_map_json=classes,
            metadata_json={"uploaded_by": user.id, "original_filename": Path(file.filename).name})
        db.add(model)
        db.commit()
        return created(model.to_public(), message="检测模型已上传并校验")
    except Exception:
        db.rollback()
        remove_tree(directory, allowed_root=settings.model_root / "registry")
        raise


class RegisterModelRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    relative_path: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    class_map: dict[str, str] = Field(default_factory=dict)


@router.get("", response_model=None)
def list_models(
    search: str = Query(default="", max_length=255),
    file_format: str = Query(default="", max_length=32),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    task: str | None = Query(default=None),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    items, total = model_service.list_models(db, user, page=page, page_size=page_size, task=task, search=search, file_format=file_format)
    return ok(
        {
            "items": [item.to_public() for item in items],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    )


@router.post("/register", response_model=None)
def register_model(
    payload: RegisterModelRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    model = model_service.register_base_model(
        db,
        settings,
        user,
        name=payload.name,
        relative_path=payload.relative_path,
        expected_sha256=payload.sha256,
        class_map=payload.class_map,
    )
    return created(model.to_public(), message="模型已登记")


@router.get("/{model_id}", response_model=None)
def get_model(
    model_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    model = model_service.get_model(db, model_id, user)
    return ok(model.to_public())


@router.get("/{model_id}/download", response_model=None)
def download_model(
    model_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    model = model_service.get_model(db, model_id, user)
    path = model_service.verify_model_file(model, settings)
    return FileResponse(path, media_type="application/octet-stream", filename=f"{model.name}.pt")


@router.delete("/{model_id}", response_model=None)
def delete_model(
    model_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    model = model_service.get_model(db, model_id, user)
    model_service.delete_model(db, model, user)
    return ok({"model_id": model_id, "deleted": True}, message="模型已删除")
