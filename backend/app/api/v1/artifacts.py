"""受控产物下载。"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user, settings_dep
from app.core.config import Settings
from app.core.errors import ApiError
from app.db.models import ArtifactRecord
from app.db.session import get_db

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


@router.get("/{artifact_id}", response_model=None)
def download_artifact(
    artifact_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    record = db.get(ArtifactRecord, artifact_id)
    if record is None or (not user.is_admin and record.owner_id != user.id):
        raise ApiError(404, "not_found", "产物不存在")
    root = Path(settings.artifact_root).resolve()
    path = (root / record.relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ApiError(409, "invalid_path", "产物路径越界") from exc
    if not path.exists():
        raise ApiError(404, "not_found", "产物文件缺失")
    return FileResponse(path, media_type=record.mime or "application/octet-stream", filename=path.name)
