"""Persistent model registry and one-shot filesystem reconciliation."""

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

import yaml
from sqlalchemy.orm import Session

from database import ModelRegistry, SessionLocal
from model_capabilities import model_supports_text_prompts

SUPPORTED_MODEL_SUFFIXES = {".pt", ".onnx"}


def canonical_model_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    data_root = Path(os.getenv("DEEPVISION_DATA_DIR", "/app/data")).resolve()
    for legacy_root, managed_root in ((Path("/app/models"), data_root / "models"), (Path("/app/runs"), data_root / "runs")):
        try:
            relative = resolved.relative_to(legacy_root)
        except ValueError:
            continue
        managed = (managed_root / relative).resolve()
        if managed.is_file():
            return managed
    return resolved


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def register_model_file(
    db: Session,
    path: Path,
    *,
    source: str,
    status: str = "active",
    task_id: Optional[str] = None,
    dataset_version_id: Optional[str] = None,
    created_by: str = "system",
    metadata: Optional[Dict[str, Any]] = None,
) -> ModelRegistry:
    model_path = canonical_model_path(path)
    if not model_path.is_file() or model_path.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
        raise ValueError(f"invalid model file: {model_path}")
    stat = model_path.stat()
    path_text = str(model_path).replace("\\", "/")
    item = db.query(ModelRegistry).filter(ModelRegistry.path == path_text).first()
    if not item:
        item = ModelRegistry(id=str(uuid4()), path=path_text, created_at=datetime.now())
        db.add(item)
    item.name = model_path.name
    item.sha256 = sha256_file(model_path)
    item.size = stat.st_size
    item.source = source
    item.status = status
    item.format = model_path.suffix.lower().lstrip(".")
    item.task_id = task_id
    item.dataset_version_id = dataset_version_id
    item.created_by = created_by or "system"
    item.metadata_json = metadata or {}
    item.updated_at = datetime.fromtimestamp(stat.st_mtime)
    db.commit()
    db.refresh(item)
    return item


def register_model_file_external(path: Path, **kwargs) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        return register_model_file(db, path, **kwargs).to_dict()
    finally:
        db.close()


def filter_available_model_records(db: Session, records: Iterable[ModelRegistry]) -> list[ModelRegistry]:
    """Hide missing model files and persist their registry lifecycle state."""
    available = []
    changed = False
    for item in records:
        try:
            exists = Path(item.path).is_file()
        except OSError:
            exists = False
        if exists:
            available.append(item)
            continue
        item.status = "missing"
        item.updated_at = datetime.now()
        changed = True
    if changed:
        db.commit()
    return available


def build_model_lineage(
    parent_model: Any,
    parent_metadata: Optional[Dict[str, Any]],
    training_profile: Any,
    *,
    parent_registry_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build durable model-family metadata for the next trained generation."""
    metadata = dict(parent_metadata or {})
    is_yoloe = (
        str(training_profile or "").strip().lower() == "yoloe_pe_segment"
        or model_supports_text_prompts(metadata, parent_model)
    )
    try:
        parent_generation = max(0, int(metadata.get("lineage_generation") or 0))
    except (TypeError, ValueError):
        parent_generation = 0
    root_model = (
        metadata.get("lineage_root_model")
        or metadata.get("base_model")
        or parent_model
    )
    return {
        "model_family": "yoloe" if is_yoloe else "yolo",
        "supports_text_prompts": is_yoloe,
        "lineage_root_model": str(root_model or ""),
        "parent_model": str(parent_model or ""),
        "parent_model_registry_id": parent_registry_id,
        "lineage_generation": parent_generation + 1,
    }


def merge_preserved_model_metadata(
    discovered: Dict[str, Any],
    existing: Optional[Dict[str, Any]],
    *,
    preserve_all: bool = False,
) -> Dict[str, Any]:
    """Keep durable lineage when filesystem reconciliation rediscovers a model."""
    merged = dict(discovered or {})
    current = dict(existing or {})
    keys = current.keys() if preserve_all else (
        "model_family",
        "supports_text_prompts",
        "training_profile",
        "base_model",
        "lineage_root_model",
        "parent_model",
        "parent_model_registry_id",
        "lineage_generation",
    )
    for key in keys:
        if key in current and current[key] is not None:
            merged[key] = current[key]
    return merged


def normalize_visible_model_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload or {})
    if normalized.get("status") == "candidate":
        normalized["status"] = "active"
        normalized["activation_status"] = "active"
    return normalized


def hydrate_trained_model_payload(
    payload: Dict[str, Any],
    discovered: Dict[str, Any],
) -> Dict[str, Any]:
    """Fill missing trained-model metadata from the experiment directory."""
    hydrated = dict(payload or {})
    metadata = dict(discovered or {})
    for key in (
        "display_name",
        "experiment_name",
        "weight_type",
        "dataset",
        "epochs",
        "base_model",
    ):
        if hydrated.get(key) in (None, "") and metadata.get(key) not in (None, ""):
            hydrated[key] = metadata[key]
    hydrated["supports_text_prompts"] = model_supports_text_prompts(hydrated, metadata)
    return hydrated


def trained_model_metadata(model_path: Path) -> Dict[str, Any]:
    experiment = model_path.parent.parent
    metadata: Dict[str, Any] = {
        "display_name": f"{experiment.name} - {model_path.stem}",
        "experiment_name": experiment.name,
        "weight_type": model_path.stem,
        "dataset": "未知数据集",
        "epochs": "",
    }
    args_file = experiment / "args.yaml"
    if args_file.is_file():
        try:
            args = yaml.safe_load(args_file.read_text(encoding="utf-8")) or {}
            base_model = str(args.get("model") or "").strip()
            if base_model:
                metadata["base_model"] = base_model
                metadata["supports_text_prompts"] = model_supports_text_prompts(base_model)
            if isinstance(args.get("data"), str):
                metadata["dataset"] = Path(args["data"]).stem
            if args.get("epochs") is not None:
                metadata["epochs"] = f" ({args['epochs']}轮)"
        except Exception:
            pass
    return metadata


def reconcile_models(
    db: Session,
    *,
    model_root: Path,
    runs_root: Path,
    pretrained_names: Iterable[str] = (),
    created_by: str = "system",
) -> Dict[str, Any]:
    model_root = Path(model_root).resolve()
    runs_root = Path(runs_root).resolve()
    pretrained_set = {str(name).lower() for name in pretrained_names}
    discovered = []
    if model_root.is_dir():
        discovered.extend(
            path
            for path in model_root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_MODEL_SUFFIXES
        )
    if runs_root.is_dir():
        discovered.extend(
            path
            for path in runs_root.rglob("weights/*")
            if path.is_file()
            and path.name in {"best.pt", "best.onnx"}
            and path.suffix.lower() in SUPPORTED_MODEL_SUFFIXES
        )

    seen_paths = set()
    registered = []
    skipped_deleted = []
    failed = []
    for model_path in sorted(discovered, key=lambda item: str(item).lower()):
        resolved = model_path.resolve()
        path_text = str(resolved).replace("\\", "/")
        if path_text in seen_paths:
            continue
        seen_paths.add(path_text)
        lower_parts = {part.lower() for part in resolved.parts}
        if str(resolved).startswith(str(runs_root)):
            source = "trained"
            metadata = trained_model_metadata(resolved)
        elif "behavior" in lower_parts:
            source = "behavior"
            metadata = {"display_name": resolved.stem.replace("_", " ")}
        elif "pretrained" in lower_parts or resolved.name.lower() in pretrained_set:
            source = "pretrained"
            metadata = {
                "display_name": resolved.stem.upper(),
                "supports_text_prompts": model_supports_text_prompts(resolved.name),
            }
        else:
            source = "uploaded"
            metadata = {"display_name": resolved.stem.replace("_", " ")}
        try:
            existing_item = db.query(ModelRegistry).filter(ModelRegistry.path == path_text).first()
            existing_metadata = (
                dict(existing_item.metadata_json or {})
                if existing_item is not None
                else {}
            )
            # A deleted registry row is a tombstone.  Reconciliation must not
            # silently turn it back into an active model merely because a
            # stale file (or a legacy bind-mount alias) is still present.
            if existing_item is not None and existing_item.status == "deleted":
                skipped_deleted.append(path_text)
                continue
            registered.append(
                register_model_file(
                    db,
                    resolved,
                    source=source,
                    status="active",
                    created_by=created_by,
                    metadata=merge_preserved_model_metadata(
                        metadata,
                        existing_metadata,
                        preserve_all=bool(
                            existing_item is not None
                            and existing_item.status == "candidate"
                        ),
                    ),
                ).to_dict()
            )
        except Exception as exc:
            db.rollback()
            failed.append({"path": path_text, "error": str(exc)})

    existing = db.query(ModelRegistry).filter(ModelRegistry.status == "active").all()
    missing = []
    for item in existing:
        if item.path not in seen_paths and (item.path.startswith(str(model_root)) or item.path.startswith(str(runs_root))):
            item.status = "missing"
            item.updated_at = datetime.now()
            missing.append(item.path)
    db.commit()
    return {
        "registered": registered,
        "missing": missing,
        "skipped_deleted": skipped_deleted,
        "failed": failed,
    }
