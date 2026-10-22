"""v1 模型注册与受控下载。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user, settings_dep
from app.core.config import Settings
from app.core.errors import ApiError, created, ok
from app.db.session import get_db
from app.domains.models import service as model_service
from app.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

router = APIRouter(prefix="/models", tags=["models"])


class RegisterModelRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    relative_path: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    class_map: dict[str, str] = Field(default_factory=dict)


@router.get("", response_model=None)
def list_models(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    task: str | None = Query(default=None),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    items, total = model_service.list_models(db, user, page=page, page_size=page_size, task=task)
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
