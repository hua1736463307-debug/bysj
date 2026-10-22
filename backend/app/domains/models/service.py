"""模型注册、可信加载与下载边界。"""
from __future__ import annotations

import shutil
from pathlib import Path

from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.core.config import Settings
from app.core.errors import ApiError
from app.core.storage import ensure_dir, normalize_relative, resolve_under, sha256_file
from app.db.base import utcnow
from app.db.models import EvaluationRun, ModelRegistry, ReviewRecord


def _public(model: ModelRegistry) -> dict:
    return model.to_public()


def list_models(
    db: Session, user: CurrentUser, *, page: int, page_size: int, task: str | None = None
) -> tuple[list[ModelRegistry], int]:
    query = db.query(ModelRegistry).filter(ModelRegistry.status == "active")
    if not user.is_admin:
        query = query.filter(
            (ModelRegistry.owner_id == user.id) | (ModelRegistry.trusted_source == "base")
        )
    if task:
        query = query.filter(ModelRegistry.model_task == task)
    total = query.count()
    items = (
        query.order_by(ModelRegistry.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def get_model(db: Session, model_id: str, user: CurrentUser) -> ModelRegistry:
    model = db.get(ModelRegistry, model_id)
    if model is None or model.status != "active":
        raise ApiError(404, "not_found", "模型不存在")
    if not user.is_admin and model.owner_id != user.id and model.trusted_source != "base":
        raise ApiError(404, "not_found", "模型不存在")
    return model


def register_base_model(
    db: Session,
    settings: Settings,
    admin: CurrentUser,
    *,
    name: str,
    relative_path: str,
    expected_sha256: str,
    class_map: dict,
) -> ModelRegistry:
    if not admin.is_admin:
        raise ApiError(403, "forbidden", "只有管理员可以登记基础权重")
    source = resolve_under(settings.model_root / "base", relative_path, must_exist=True)
    digest = sha256_file(source)
    if digest != expected_sha256:
        raise ApiError(422, "hash_mismatch", "文件 hash 与声明不一致")
    model_id = __import__("uuid").uuid4().hex
    target_dir = ensure_dir(settings.model_root / "registry" / model_id)
    target = target_dir / f"{normalize_relative(relative_path).name}"
    shutil.copy2(source, target)
    model = ModelRegistry(
        id=model_id,
        name=name,
        path=str(target),
        sha256=digest,
        size=target.stat().st_size,
        source="base",
        status="active",
        format="pt",
        created_by=admin.username,
        owner_id=admin.id,
        model_task="detect",
        trusted_source="base",
        available=True,
        class_map_json={str(k): str(v) for k, v in (class_map or {}).items()},
        metadata_json={"registered_from": relative_path},
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def register_trained_model(
    db: Session,
    settings: Settings,
    *,
    job_id: str,
    version_id: str,
    owner_id: int,
    owner_username: str,
    name: str,
    best_path: Path,
    class_map: dict,
    metrics: dict,
    backend: str,
) -> ModelRegistry:
    digest = sha256_file(best_path)
    target_dir = ensure_dir(settings.model_root / "training" / job_id)
    target = target_dir / "best.pt"
    shutil.copy2(best_path, target)
    model = ModelRegistry(
        id=__import__("uuid").uuid4().hex,
        name=name,
        path=str(target),
        sha256=digest,
        size=target.stat().st_size,
        source="training",
        status="active",
        format="pt",
        task_id=job_id,
        training_job_id=job_id,
        dataset_version_id=version_id,
        created_by=owner_username,
        owner_id=owner_id,
        model_task="detect",
        trusted_source="platform",
        available=True,
        class_map_json={str(k): str(v) for k, v in class_map.items()},
        metadata_json={"metrics": metrics, "backend": backend},
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def verify_model_file(model: ModelRegistry, settings: Settings) -> Path:
    if not model.path:
        raise ApiError(409, "model_unavailable", "模型没有可用文件")
    path = Path(model.path).resolve()
    try:
        path.relative_to(settings.model_root.resolve())
    except ValueError as exc:
        raise ApiError(409, "model_unavailable", "模型路径超出受控目录") from exc
    if not path.exists():
        model.available = False
        raise ApiError(409, "model_missing", "模型文件缺失")
    digest = sha256_file(path)
    if digest != model.sha256:
        model.available = False
        raise ApiError(409, "model_tampered", "模型文件 hash 与登记不一致")
    return path


def delete_model(db: Session, model: ModelRegistry, user: CurrentUser) -> None:
    if not user.is_admin and model.owner_id != user.id:
        raise ApiError(404, "not_found", "模型不存在")
    evaluations = db.query(EvaluationRun).filter(EvaluationRun.model_id == model.id).count()
    if evaluations:
        raise ApiError(409, "resource_referenced", "模型已被评估记录引用，不能删除")
    reviews = db.query(ReviewRecord).count()
    model.status = "deleted"
    model.available = False
    model.updated_at = utcnow()
    db.commit()


def serialize(model: ModelRegistry) -> dict:
    return _public(model)
