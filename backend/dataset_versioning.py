"""Immutable dataset manifests, validation, and asset indexing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml
from PIL import Image
from sqlalchemy.orm import Session

from database import DatasetAsset, DatasetVersion, ModelRegistry
from annotation_class_aliases import annotation_class_key, class_name_ids
from dataset_config_utils import read_dataset_class_names
from image_size_processing import (
    tiling_policy,
    resolve_tiling_profile,
    plan_annotation_tiles,
    annotation_pixel_bounds,
    MOBILEPHONE_CONTEXT_POLICY_VERSION,
    PHONE_BADGE_FOCUS_POLICY_VERSION,
    MAX_TILES_PER_IMAGE,
    SUPPORTED_IMAGE_SIZES,
    TILING_POLICY_VERSION,
    adaptive_tiling_policy,
    plan_image_annotation_tiles,
    render_annotation_tile,
    select_adaptive_tile_size,
    summarize_image_size_results,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class PreparedTrainingWorkspace:
    id: str
    dataset_id: str
    parent_version_id: str
    root_path: str
    content_hash: str
    class_map_json: Dict[str, str]


def delete_dataset_version_records(db: Session, dataset_id: str, dataset_role: str) -> int:
    """Delete one dataset role's versions and dependent database records."""
    version_ids = [
        version_id
        for (version_id,) in (
            db.query(DatasetVersion.id)
            .filter(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.dataset_role == dataset_role,
            )
            .all()
        )
    ]
    if not version_ids:
        return 0

    db.query(ModelRegistry).filter(
        ModelRegistry.dataset_version_id.in_(version_ids)
    ).update(
        {ModelRegistry.dataset_version_id: None},
        synchronize_session=False,
    )
    db.query(DatasetAsset).filter(
        DatasetAsset.dataset_version_id.in_(version_ids)
    ).delete(synchronize_session=False)
    return int(
        db.query(DatasetVersion)
        .filter(DatasetVersion.id.in_(version_ids))
        .delete(synchronize_session=False)
        or 0
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _image_split_and_label(root: Path, image_path: Path) -> Tuple[str, Optional[Path]]:
    relative = image_path.relative_to(root)
    parts = relative.parts
    if len(parts) >= 3 and parts[0] in SPLITS and parts[1] == "images":
        label_relative = Path(parts[0]) / "labels" / Path(*parts[2:]).with_suffix(".txt")
        return parts[0], root / label_relative
    return "unsplit", image_path.with_suffix(".txt")


def _label_split_and_image_key(root: Path, label_path: Path) -> Tuple[str, str]:
    relative = label_path.relative_to(root)
    parts = relative.parts
    if len(parts) >= 3 and parts[0] in SPLITS and parts[1] == "labels":
        key = str(Path(*parts[2:]).with_suffix("")).replace("\\", "/")
        return parts[0], key
    return "unsplit", str(relative.with_suffix("")).replace("\\", "/")


def _read_image_dimensions(path: Path) -> Tuple[int, int, Optional[str]]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return int(image.width), int(image.height), None
    except Exception as exc:
        return 0, 0, str(exc)


def _validate_label(path: Path, class_count: int) -> Tuple[List[Dict[str, Any]], Counter, List[str]]:
    annotations: List[Dict[str, Any]] = []
    counts: Counter = Counter()
    errors: List[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return annotations, counts, [f"无法读取 label: {exc}"]

    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            errors.append(f"第 {line_number} 行字段数不足")
            continue
        try:
            class_value = float(parts[0])
            class_id = int(class_value)
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"第 {line_number} 行包含非数字字段")
            continue
        if class_value != class_id or class_id < 0:
            errors.append(f"第 {line_number} 行 class ID 必须为非负整数")
            continue
        if class_count and class_id >= class_count:
            errors.append(f"第 {line_number} 行 class ID {class_id} 超出 0..{class_count - 1}")
        if not all(math.isfinite(value) for value in coords):
            errors.append(f"第 {line_number} 行坐标包含 NaN/Inf")
            continue

        if len(coords) == 4:
            x, y, width, height = coords
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                errors.append(f"第 {line_number} 行 bbox 坐标超出 0..1 或宽高非正")
                continue
            annotations.append({"class_id": class_id, "type": "bbox", "x": x, "y": y, "width": width, "height": height})
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            if not all(0 <= value <= 1 for value in coords):
                errors.append(f"第 {line_number} 行 polygon 坐标超出 0..1")
                continue
            annotations.append({"class_id": class_id, "type": "polygon", "points": coords})
        else:
            errors.append(f"第 {line_number} 行坐标格式不是 bbox 或 polygon")
            continue
        counts[class_id] += 1
    return annotations, counts, errors


def validate_dataset(root_path: Path, dataset_role: str = "training") -> Dict[str, Any]:
    root = Path(root_path).resolve()
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    class_names: List[str] = []
    yaml_path = next((root / name for name in ("dataset.yaml", "data.yaml") if (root / name).is_file()), None)

    if dataset_role == "training" and not yaml_path:
        errors.append({"code": "missing_yaml", "message": "训练数据集缺少 dataset.yaml/data.yaml"})
    if yaml_path:
        try:
            payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            class_names = read_dataset_class_names(yaml_path)
            nc = payload.get("nc")
            if not class_names:
                errors.append({"code": "missing_names", "message": "YAML 缺少有效 names"})
            if nc is not None and int(nc) != len(class_names):
                errors.append({"code": "nc_mismatch", "message": f"nc={nc} 与 names 数量 {len(class_names)} 不一致"})
        except Exception as exc:
            errors.append({"code": "invalid_yaml", "message": str(exc)})

    image_paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    label_paths = sorted(
        path for path in root.rglob("*.txt")
        if path.name.lower() != "classes.txt" and not path.name.endswith(".cache")
    )
    split_counts = {split: {"images": 0, "labels": 0} for split in (*SPLITS, "unsplit")}
    class_counts: Counter = Counter()
    assets: List[Dict[str, Any]] = []
    matched_label_paths = set()
    content_hash_counts: Counter = Counter()

    for image_path in image_paths:
        split, label_path = _image_split_and_label(root, image_path)
        split_counts[split]["images"] += 1
        width, height, image_error = _read_image_dimensions(image_path)
        relative_path = str(image_path.relative_to(root)).replace("\\", "/")
        if image_error:
            errors.append({"code": "invalid_image", "message": f"{relative_path}: {image_error}"})

        annotations: List[Dict[str, Any]] = []
        label_relative = None
        if label_path and label_path.is_file():
            matched_label_paths.add(label_path.resolve())
            split_counts[split]["labels"] += 1
            label_relative = str(label_path.relative_to(root)).replace("\\", "/")
            annotations, counts, label_errors = _validate_label(label_path, len(class_names))
            class_counts.update(counts)
            for message in label_errors:
                errors.append({"code": "invalid_label", "message": f"{label_relative}: {message}"})
        elif dataset_role == "training":
            warnings.append({"code": "missing_label", "message": f"{relative_path} 没有对应 label"})

        image_hash = sha256_file(image_path)
        content_hash_counts[image_hash] += 1
        assets.append({
            "relative_path": relative_path,
            "basename": image_path.name,
            "size": image_path.stat().st_size,
            "width": width,
            "height": height,
            "label_path": label_relative,
            "split": split,
            "source_group": None,
            "content_hash": image_hash,
            "annotations": annotations,
        })

    for label_path in label_paths:
        if label_path.resolve() in matched_label_paths:
            continue
        relative = str(label_path.relative_to(root)).replace("\\", "/")
        warnings.append({"code": "orphan_label", "message": f"{relative} 没有对应图片"})

    if dataset_role == "training":
        for split in ("train", "val"):
            if split_counts[split]["images"] == 0:
                errors.append({"code": "empty_split", "message": f"{split} split 没有图片"})
    if not image_paths:
        errors.append({"code": "empty_dataset", "message": "数据集中没有可识别图片"})
    duplicate_hashes = sorted(value for value, count in content_hash_counts.items() if count > 1)
    if duplicate_hashes:
        warnings.append({"code": "duplicate_content", "message": f"发现 {len(duplicate_hashes)} 组重复图片内容"})

    return {
        "valid": not errors,
        "root_path": str(root),
        "dataset_role": dataset_role,
        "class_names": class_names,
        "errors": errors,
        "warnings": warnings,
        "statistics": {
            "images": len(image_paths),
            "labels": len(matched_label_paths),
            "splits": split_counts,
            "classes": {str(key): value for key, value in sorted(class_counts.items())},
            "duplicate_content_groups": len(duplicate_hashes),
        },
        "assets": assets,
    }


def manifest_content_hash(manifest: Dict[str, Any]) -> str:
    stable = {
        "dataset_id": manifest["dataset_id"],
        "dataset_role": manifest["dataset_role"],
        "parent_version_id": manifest.get("parent_version_id"),
        "class_names": manifest.get("class_names", []),
        "split_config": manifest.get("split_config", {}),
        "assets": [
            {
                "relative_path": item["relative_path"],
                "content_hash": item["content_hash"],
                "label_path": item.get("label_path"),
                "annotations": item.get("annotations", []),
            }
            for item in manifest.get("assets", [])
        ],
    }
    encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_class_names(class_names: Iterable[Any]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for index, value in enumerate(class_names or []):
        name = str(value or "").strip()
        if not name:
            raise ValueError(f"第 {index + 1} 个类别名称为空")
        if len(name) > 120 or any(ord(char) < 32 for char in name):
            raise ValueError(f"类别名称不合法: {name[:30]}")
        if name in seen:
            raise ValueError(f"类别名称重复: {name}")
        seen.add(name)
        normalized.append(name)
    if not normalized:
        raise ValueError("至少需要一个类别")
    return normalized


def _ordered_class_names(version: DatasetVersion) -> List[str]:
    values = version.class_map_json or {}
    if isinstance(values, list):
        return validate_class_names(values)
    try:
        keys = sorted(values, key=lambda value: int(value))
    except (TypeError, ValueError):
        keys = sorted(values, key=str)
    return validate_class_names([values[key] for key in keys])


def _annotation_class_id(annotation: Dict[str, Any]) -> int:
    if "class_id" in annotation:
        return int(annotation["class_id"])
    line = str(annotation.get("yolo_line") or "").strip()
    if not line:
        raise ValueError("标注缺少 class_id/yolo_line")
    value = float(line.split()[0])
    class_id = int(value)
    if value != class_id or class_id < 0:
        raise ValueError("标注 class ID 非法")
    return class_id


def annotation_to_yolo_line(annotation: Dict[str, Any], class_id: Optional[int] = None) -> str:
    target_class = _annotation_class_id(annotation) if class_id is None else int(class_id)
    raw_line = str(annotation.get("yolo_line") or "").strip()
    if raw_line:
        parts = raw_line.split()
        return " ".join([str(target_class), *parts[1:]])
    annotation_type = annotation.get("type") or "bbox"
    if annotation_type == "polygon":
        points = [float(value) for value in annotation.get("points", [])]
        if len(points) < 6 or len(points) % 2:
            raise ValueError("polygon 点数非法")
        return " ".join([str(target_class), *[f"{value:.6f}" for value in points]])
    values = [float(annotation[key]) for key in ("x", "y", "width", "height")]
    return " ".join([str(target_class), *[f"{value:.6f}" for value in values]])


def persist_manifest_version(
    db: Session,
    *,
    dataset_id: str,
    dataset_role: str,
    root_path: Path,
    manifest_root: Path,
    class_names: Iterable[str],
    assets: List[Dict[str, Any]],
    created_by: str,
    parent_version_id: Optional[str] = None,
    split_config: Optional[Dict[str, Any]] = None,
    status: str = "ready",
    validation: Optional[Dict[str, Any]] = None,
    version_id: Optional[str] = None,
) -> Tuple[DatasetVersion, bool]:
    names = validate_class_names(class_names)
    split_config = split_config or {}
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_role": dataset_role,
        "parent_version_id": parent_version_id,
        "root_path": str(Path(root_path).resolve()),
        "class_names": names,
        "split_config": split_config,
        "validation": validation or {},
        "assets": assets,
    }
    content_hash = manifest_content_hash(manifest)
    existing = db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == dataset_id,
        DatasetVersion.dataset_role == dataset_role,
        DatasetVersion.content_hash == content_hash,
    ).first()
    if existing:
        return existing, False

    version_id = version_id or str(uuid.uuid4())
    manifest_dir = Path(manifest_root) / dataset_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{version_id}.json"
    manifest.update({"version_id": version_id, "content_hash": content_hash, "created_at": datetime.now().isoformat()})
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, manifest_path)

    version = DatasetVersion(
        id=version_id,
        dataset_id=dataset_id,
        parent_version_id=parent_version_id,
        dataset_role=dataset_role,
        status=status,
        root_path=str(Path(root_path).resolve()),
        manifest_path=str(manifest_path.resolve()),
        class_map_json={str(index): name for index, name in enumerate(names)},
        split_config_json=split_config,
        content_hash=content_hash,
        created_by=created_by,
    )
    db.add(version)
    db.flush()
    sqlite_asset_id = None
    if db.get_bind().dialect.name == "sqlite":
        latest_asset_id = db.query(DatasetAsset.id).order_by(DatasetAsset.id.desc()).first()
        sqlite_asset_id = int(latest_asset_id[0] if latest_asset_id else 0) + 1
    for item in assets:
        asset_values = {
            "id": sqlite_asset_id,
            "dataset_version_id": version_id,
            "relative_path": item["relative_path"],
            "basename": item.get("basename") or Path(item["relative_path"]).name,
            "size": int(item.get("size") or 0),
            "width": int(item.get("width") or 0),
            "height": int(item.get("height") or 0),
            "label_path": item.get("label_path"),
            "source_group": item.get("source_group"),
            "content_hash": item["content_hash"],
            "annotations_json": item.get("annotations", []),
        }
        if sqlite_asset_id is None:
            asset_values.pop("id")
        else:
            sqlite_asset_id += 1
        db.add(DatasetAsset(**asset_values))
    db.commit()
    db.refresh(version)
    return version, True


def create_annotation_version(
    db: Session,
    source_version: DatasetVersion,
    updates: Dict[str, Dict[str, Any]],
    class_names: Iterable[str],
    manifest_root: Path,
    created_by: str,
    required_geometry: Optional[str] = None,
    drop_unannotated: bool = False,
) -> Tuple[DatasetVersion, bool]:
    names = validate_class_names(class_names)
    source_assets = (
        db.query(DatasetAsset)
        .filter(DatasetAsset.dataset_version_id == source_version.id)
        .order_by(DatasetAsset.relative_path.asc())
        .all()
    )
    known_paths = {asset.relative_path for asset in source_assets}
    unknown_paths = sorted(set(updates) - known_paths)
    if unknown_paths:
        raise ValueError(f"标注包含未知资产: {unknown_paths[:3]}")

    assets: List[Dict[str, Any]] = []
    for asset in source_assets:
        update = updates.get(asset.relative_path, {})
        annotations = filter_annotations_by_geometry(
            update.get("annotations", asset.annotations_json or []),
            required_geometry=required_geometry,
        )
        if drop_unannotated and not annotations:
            continue
        for annotation in annotations:
            class_id = _annotation_class_id(annotation)
            if class_id >= len(names):
                raise ValueError(f"{asset.relative_path} 的类别 {class_id} 超出范围")
            annotation_to_yolo_line(annotation)
        assets.append({
            "relative_path": asset.relative_path,
            "basename": asset.basename,
            "size": asset.size,
            "width": int(update.get("width") or asset.width or 0),
            "height": int(update.get("height") or asset.height or 0),
            "label_path": asset.label_path,
            "source_group": update.get("source_group") or asset.source_group,
            "content_hash": asset.content_hash,
            "annotations": annotations,
        })
    if drop_unannotated and not assets:
        raise ValueError("没有包含有效标注的图片")
    return persist_manifest_version(
        db,
        dataset_id=source_version.dataset_id,
        dataset_role="annotation",
        root_path=Path(source_version.root_path),
        manifest_root=manifest_root,
        class_names=names,
        assets=assets,
        created_by=created_by,
        parent_version_id=source_version.id,
        status="ready",
        validation={"valid": True, "asset_count": len(assets)},
    )


def class_subset_plan(db: Session, source_version: DatasetVersion, selected_names: Iterable[str]) -> Dict[str, Any]:
    source_names = _ordered_class_names(source_version)
    selected = validate_class_names(selected_names)
    missing = [name for name in selected if name not in source_names]
    if missing:
        raise ValueError(f"数据集不存在类别: {missing}")
    old_to_new = {source_names.index(name): index for index, name in enumerate(selected)}
    kept = filtered = 0
    assets = db.query(DatasetAsset).filter(DatasetAsset.dataset_version_id == source_version.id).all()
    for asset in assets:
        for annotation in asset.annotations_json or []:
            if _annotation_class_id(annotation) in old_to_new:
                kept += 1
            else:
                filtered += 1
    return {"selected_names": selected, "old_to_new": old_to_new, "kept_annotations": kept, "filtered_annotations": filtered, "asset_count": len(assets)}


def _copy_raw_image_without_exif(source: Path, target: Path) -> None:
    """Write the uploaded pixel order without an EXIF auto-rotation marker."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        output_format = str(image.format or "").upper()
        if output_format not in {"JPEG", "PNG", "WEBP", "BMP"}:
            output_format = "JPEG"
        if output_format == "JPEG" and image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        save_kwargs = {"quality": 95} if output_format == "JPEG" else {}
        image.save(target, format=output_format, **save_kwargs)


def _annotation_geometry(annotation: Dict[str, Any]) -> str:
    annotation_type = str(annotation.get("type") or "").strip().lower()
    if annotation_type in {"polygon", "bbox"}:
        return annotation_type
    raw_line = str(annotation.get("yolo_line") or "").strip()
    if raw_line:
        field_count = len(raw_line.split())
        if field_count == 5:
            return "bbox"
        if field_count >= 7 and field_count % 2 == 1:
            return "polygon"
    return "invalid"


def filter_annotations_by_geometry(
    annotations: Iterable[Dict[str, Any]],
    required_geometry: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retain only one compatible YOLO annotation geometry when requested."""
    if required_geometry is None:
        return list(annotations or [])
    geometry = str(required_geometry).strip().lower()
    if geometry not in {"bbox", "polygon"}:
        raise ValueError("required_geometry 仅支持 bbox 或 polygon")
    return [
        annotation
        for annotation in (annotations or [])
        if isinstance(annotation, dict) and _annotation_geometry(annotation) == geometry
    ]


def summarize_annotation_assets(
    source_assets: Iterable[DatasetAsset],
    updates: Mapping[str, Dict[str, Any]],
    required_geometry: Optional[str] = None,
) -> Dict[str, Any]:
    """Count assets that remain exportable after UI/AI updates and geometry filtering."""
    total_count = 0
    annotated_count = 0
    unlabeled_paths: List[str] = []
    for asset in source_assets:
        total_count += 1
        update = updates.get(asset.relative_path, {})
        annotations = filter_annotations_by_geometry(
            update.get("annotations", asset.annotations_json or []),
            required_geometry=required_geometry,
        )
        if annotations:
            annotated_count += 1
        elif len(unlabeled_paths) < 20:
            unlabeled_paths.append(asset.relative_path)
    return {
        "total_count": total_count,
        "annotated_count": annotated_count,
        "unlabeled_count": total_count - annotated_count,
        "unlabeled_sample_paths": unlabeled_paths,
        "required_geometry": required_geometry,
    }


def _annotation_coordinate_issue(annotation: Dict[str, Any], geometry: str) -> Optional[str]:
    try:
        raw_line = str(annotation.get("yolo_line") or "").strip()
        if raw_line:
            values = [float(value) for value in raw_line.split()[1:]]
        elif geometry == "polygon":
            values = [float(value) for value in annotation.get("points", [])]
        else:
            values = [float(annotation[key]) for key in ("x", "y", "width", "height")]
    except (TypeError, ValueError, KeyError):
        return "coordinates_not_numeric"
    if any(not math.isfinite(value) for value in values):
        return "coordinates_not_finite"
    if geometry == "polygon":
        if len(values) < 6 or len(values) % 2:
            return "polygon_point_count_invalid"
        if any(value < 0 or value > 1 for value in values):
            return "coordinates_out_of_range"
        return None
    if len(values) != 4:
        return "bbox_field_count_invalid"
    x, y, width, height = values
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
        return "coordinates_out_of_range"
    if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6:
        return "bbox_outside_image"
    if y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
        return "bbox_outside_image"
    return None


def _asset_source_group(asset: DatasetAsset) -> str:
    if asset.source_group:
        return str(asset.source_group)
    relative = Path(asset.relative_path)
    parts = list(relative.parts)
    if len(parts) >= 3 and parts[0] in SPLITS and parts[1] == "images":
        parts = parts[2:]
    parent = Path(*parts[:-1]) if len(parts) > 1 else Path()
    if str(parent) not in {"", "."}:
        return f"folder:{parent.as_posix()}"
    stem = Path(parts[-1] if parts else asset.basename).stem
    series = re.sub(
        r"(?i)(?:[_-](?:frame|img|image)?[_-]?\d{3,}|[_-]\d{5,})$",
        "",
        stem,
    ).strip("_-")
    if series and series != stem:
        return f"frame-series:{series}"
    return f"asset:{asset.relative_path}"


def analyze_training_version(
    db: Session,
    source_version: DatasetVersion,
    *,
    required_geometry: Optional[str] = None,
    major_class_threshold: int = 20,
) -> Dict[str, Any]:
    """Return model-facing annotation statistics without mutating the dataset."""
    class_names = _ordered_class_names(source_version)
    assets = (
        db.query(DatasetAsset)
        .filter(DatasetAsset.dataset_version_id == source_version.id)
        .order_by(DatasetAsset.relative_path)
        .all()
    )
    class_counts: Counter = Counter()
    geometry_counts: Counter = Counter()
    invalid_annotations: List[Dict[str, Any]] = []
    split_counts: Counter = Counter()
    group_classes: Dict[str, set] = {}
    empty_label_count = 0

    for asset in assets:
        relative_parts = Path(asset.relative_path).parts
        split = relative_parts[0] if relative_parts and relative_parts[0] in SPLITS else "unsplit"
        split_counts[split] += 1
        group_key = _asset_source_group(asset)
        group_class_ids = group_classes.setdefault(group_key, set())
        if not (asset.annotations_json or []):
            empty_label_count += 1
        for annotation_index, annotation in enumerate(asset.annotations_json or []):
            try:
                class_id = _annotation_class_id(annotation)
            except (TypeError, ValueError):
                invalid_annotations.append(
                    {"asset": asset.relative_path, "annotation_index": annotation_index, "reason": "invalid_class_id"}
                )
                geometry_counts["invalid"] += 1
                continue
            geometry = _annotation_geometry(annotation)
            geometry_counts[geometry] += 1
            if class_id < 0 or class_id >= len(class_names):
                invalid_annotations.append(
                    {
                        "asset": asset.relative_path,
                        "annotation_index": annotation_index,
                        "reason": "class_id_out_of_range",
                        "class_id": class_id,
                    }
                )
                continue
            if geometry == "invalid":
                invalid_annotations.append(
                    {"asset": asset.relative_path, "annotation_index": annotation_index, "reason": "invalid_geometry"}
                )
                continue
            coordinate_issue = _annotation_coordinate_issue(annotation, geometry)
            if coordinate_issue:
                invalid_annotations.append(
                    {
                        "asset": asset.relative_path,
                        "annotation_index": annotation_index,
                        "reason": coordinate_issue,
                    }
                )
                continue
            class_counts[class_id] += 1
            group_class_ids.add(class_id)

    nonzero_ids = [class_id for class_id in range(len(class_names)) if class_counts[class_id] > 0]
    zero_ids = [class_id for class_id in range(len(class_names)) if class_counts[class_id] == 0]
    class_mapping = {class_id: new_id for new_id, class_id in enumerate(nonzero_ids)}
    class_stats = []
    for class_id, class_name in enumerate(class_names):
        count = int(class_counts[class_id])
        class_stats.append(
            {
                "class_id": class_id,
                "name": class_name,
                "instances": count,
                "new_class_id": class_mapping.get(class_id),
                "is_zero": count == 0,
                "is_sparse": 0 < count <= int(major_class_threshold),
                "is_major": count > int(major_class_threshold),
                "group_count": sum(1 for ids in group_classes.values() if class_id in ids),
            }
        )

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    if not assets:
        errors.append({"code": "no_assets", "message": "训练数据集没有图片资产"})
    if not nonzero_ids:
        errors.append({"code": "no_annotations", "message": "训练数据集没有有效标注实例"})
    if invalid_annotations:
        errors.append(
            {
                "code": "invalid_annotations",
                "message": f"发现 {len(invalid_annotations)} 条非法标注",
                "samples": invalid_annotations[:10],
            }
        )
    if required_geometry:
        mismatch_count = sum(
            count
            for geometry, count in geometry_counts.items()
            if geometry not in {required_geometry, "invalid"}
        )
        if mismatch_count:
            errors.append(
                {
                    "code": "geometry_mismatch",
                    "message": f"{required_geometry} 训练发现 {mismatch_count} 条不兼容标注",
                    "required_geometry": required_geometry,
                    "geometry_counts": dict(geometry_counts),
                }
            )
    if zero_ids:
        warnings.append(
            {
                "code": "zero_sample_classes_removed",
                "message": f"{len(zero_ids)} 个零样本类别将在派生版本中移除",
                "class_ids": zero_ids,
            }
        )
    sparse_ids = [item["class_id"] for item in class_stats if item["is_sparse"]]
    if sparse_ids:
        warnings.append(
            {
                "code": "sparse_classes_included",
                "message": f"{len(sparse_ids)} 个稀疏类别仍会正常训练，但不参与主要验收",
                "class_ids": sparse_ids,
            }
        )

    return {
        "dataset_version_id": source_version.id,
        "dataset_id": source_version.dataset_id,
        "asset_count": len(assets),
        "annotation_count": int(sum(class_counts.values())),
        "empty_label_count": int(empty_label_count),
        "geometry_counts": dict(geometry_counts),
        "split_counts": {split: int(split_counts[split]) for split in (*SPLITS, "unsplit")},
        "class_stats": class_stats,
        "class_mapping": {str(old): new for old, new in class_mapping.items()},
        "effective_class_names": [class_names[class_id] for class_id in nonzero_ids],
        "major_class_threshold": int(major_class_threshold),
        "major_class_ids": [item["class_id"] for item in class_stats if item["is_major"]],
        "sparse_class_ids": sparse_ids,
        "zero_class_ids": zero_ids,
        "errors": errors,
        "warnings": warnings,
    }


def _grouped_three_way_split(
    assets: List[DatasetAsset],
    *,
    train_ratio: float,
    test_ratio: float,
    seed: int,
    major_class_ids: Iterable[int],
) -> Dict[str, str]:
    """Assign source groups without leakage and reserve validation/test coverage for major classes."""
    group_assets: Dict[str, List[DatasetAsset]] = {}
    group_classes: Dict[str, set] = {}
    for asset in assets:
        group_key = _asset_source_group(asset)
        group_assets.setdefault(group_key, []).append(asset)
        ids = group_classes.setdefault(group_key, set())
        for annotation in asset.annotations_json or []:
            try:
                ids.add(_annotation_class_id(annotation))
            except (TypeError, ValueError):
                continue

    groups = sorted(group_assets)
    if len(groups) < 3:
        raise ValueError("80/10/10 划分至少需要 3 个独立来源组")
    random.Random(int(seed)).shuffle(groups)
    val_ratio = 1.0 - float(train_ratio) - float(test_ratio)
    if val_ratio <= 0:
        raise ValueError("train_ratio + test_ratio 必须小于 1")
    targets = {
        "test": max(1, int(round(len(groups) * float(test_ratio)))),
        "val": max(1, int(round(len(groups) * val_ratio))),
    }
    assignments: Dict[str, str] = {}

    # Cover rare major classes first in both evaluation splits.
    major_ids = sorted(
        set(int(value) for value in major_class_ids),
        key=lambda class_id: sum(1 for ids in group_classes.values() if class_id in ids),
    )
    for split in ("test", "val"):
        for class_id in major_ids:
            if any(assignments.get(group) == split and class_id in group_classes[group] for group in groups):
                continue
            candidates = [
                group
                for group in groups
                if group not in assignments and class_id in group_classes[group]
            ]
            if not candidates:
                raise ValueError(f"大类 {class_id} 无法同时覆盖 train/val/test，请增加独立来源样本")
            candidates.sort(
                key=lambda group: (
                    -sum(1 for candidate_id in major_ids if candidate_id in group_classes[group]),
                    group,
                )
            )
            assignments[candidates[0]] = split

    for split in ("test", "val"):
        for group in groups:
            if sum(1 for value in assignments.values() if value == split) >= targets[split]:
                break
            if group not in assignments:
                assignments[group] = split
    for group in groups:
        assignments.setdefault(group, "train")

    for class_id in major_ids:
        covered = {
            assignments[group]
            for group in groups
            if class_id in group_classes[group]
        }
        if covered != {"train", "val", "test"}:
            raise ValueError(f"大类 {class_id} 未覆盖 train/val/test，请增加独立来源样本")
    return assignments


def derive_class_subset(
    db: Session,
    source_version: DatasetVersion,
    selected_names: Iterable[str],
    derived_root: Path,
    staging_root: Path,
    manifest_root: Path,
    created_by: str,
) -> Tuple[DatasetVersion, Dict[str, Any], bool]:
    if source_version.dataset_role != "training":
        raise ValueError("只有训练数据集版本可派生类别子集")
    plan = class_subset_plan(db, source_version, selected_names)
    version_id = str(uuid.uuid4())
    stage = Path(staging_root) / version_id
    final_root = Path(derived_root) / source_version.dataset_id / version_id
    source_root = Path(source_version.root_path).resolve()
    stage.mkdir(parents=True, exist_ok=False)
    try:
        assets = db.query(DatasetAsset).filter(DatasetAsset.dataset_version_id == source_version.id).order_by(DatasetAsset.relative_path).all()
        for asset in assets:
            relative = Path(asset.relative_path)
            if len(relative.parts) < 3 or relative.parts[0] not in SPLITS or relative.parts[1] != "images":
                raise ValueError(f"训练资产路径不符合 split/images 结构: {asset.relative_path}")
            source_image = (source_root / relative).resolve()
            source_image.relative_to(source_root)
            if not source_image.is_file():
                raise ValueError(f"源图片不存在: {asset.relative_path}")
            target_image = stage / relative
            _copy_raw_image_without_exif(source_image, target_image)
            label_relative = Path(relative.parts[0]) / "labels" / Path(*relative.parts[2:]).with_suffix(".txt")
            lines = []
            for annotation in asset.annotations_json or []:
                old_id = _annotation_class_id(annotation)
                if old_id in plan["old_to_new"]:
                    lines.append(annotation_to_yolo_line(annotation, plan["old_to_new"][old_id]))
            target_label = stage / label_relative
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        yaml_payload = {
            "path": str(final_root.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(plan["selected_names"]),
            "names": plan["selected_names"],
        }
        (stage / "dataset.yaml").write_text(yaml.safe_dump(yaml_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        final_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, final_root)
        report = validate_dataset(final_root, "training")
        if not report["valid"]:
            raise ValueError(f"派生数据集校验失败: {report['errors'][:3]}")
        version, created = persist_manifest_version(
            db,
            dataset_id=source_version.dataset_id,
            dataset_role="training",
            root_path=final_root,
            manifest_root=manifest_root,
            class_names=plan["selected_names"],
            assets=report["assets"],
            created_by=created_by,
            parent_version_id=source_version.id,
            split_config={"type": "class_subset", "old_to_new": plan["old_to_new"]},
            validation={"valid": True, "statistics": report["statistics"]},
            version_id=version_id,
        )
        if not created and Path(version.root_path).resolve() != final_root.resolve():
            shutil.rmtree(final_root, ignore_errors=True)
        return version, plan, created
    except Exception:
        db.rollback()
        shutil.rmtree(stage, ignore_errors=True)
        if final_root.exists() and not db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first():
            shutil.rmtree(final_root, ignore_errors=True)
        raise


def _safe_dataset_name(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", str(value or "training_dataset").strip())
    return normalized.strip("._")[:100] or "training_dataset"


def _assign_tile_splits(
    tile_names: Iterable[str],
    *,
    train_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    names = sorted({str(name) for name in tile_names})
    minimum_count = 3 if float(test_ratio) > 0 else 2
    if len(names) < minimum_count:
        raise ValueError(
            "有效切片不足 3 张，无法形成 train/val/test"
            if minimum_count == 3
            else "有效切片不足 2 张，无法形成 train/val"
        )
    random.Random(int(seed)).shuffle(names)
    if float(test_ratio) > 0:
        val_ratio = 1.0 - float(train_ratio) - float(test_ratio)
        test_count = max(1, min(int(round(len(names) * float(test_ratio))), len(names) - 2))
        val_count = max(1, min(int(round(len(names) * val_ratio)), len(names) - test_count - 1))
        train_count = len(names) - val_count - test_count
    else:
        train_count = max(1, min(int(len(names) * float(train_ratio)), len(names) - 1))
        val_count = len(names) - train_count
        test_count = 0
    assignments: Dict[str, str] = {}
    for index, name in enumerate(names):
        if index < train_count:
            assignments[name] = "train"
        elif index < train_count + val_count:
            assignments[name] = "val"
        else:
            assignments[name] = "test"
    return assignments, {"train": train_count, "val": val_count, "test": test_count}


def resolve_fixed_training_class_map(
    original_names: List[str], fixed_names: Iterable[str], class_stats: List[Dict[str, Any]],
    inactive_ids: Iterable[int] = (),
) -> Tuple[List[str], Dict[int, int]]:
    """Map source labels into the checkpoint's immutable output-head order."""
    names = validate_class_names(fixed_names)
    inactive_ids = set(inactive_ids or [])
    target_ids = class_name_ids(names)
    class_name_ids(original_names, inactive_ids)
    missing = [item["name"] for item in class_stats
               if item["instances"] > 0 and annotation_class_key(item["name"]) not in target_ids]
    if missing:
        raise ValueError(f"旧模型类别表缺少仍有标注的类别: {missing}")
    known_keys = {annotation_class_key(name) for name in original_names}
    unknown = [name for name in names if annotation_class_key(name) not in known_keys]
    if unknown:
        raise ValueError(f"旧模型类别表包含当前数据集没有的类别: {unknown}")
    return names, {old_id: target_ids[annotation_class_key(name)] for old_id, name in enumerate(original_names)
                   if old_id not in inactive_ids and annotation_class_key(name) in target_ids}


def export_training_version(
    db: Session,
    source_version: DatasetVersion,
    output_name: str,
    training_root: Path,
    staging_root: Path,
    manifest_root: Path,
    created_by: str,
    seed: int = 42,
    train_ratio: float = 0.8,
    test_ratio: float = 0.0,
    drop_zero_classes: bool = False,
    fixed_class_names: Optional[Iterable[str]] = None,
    required_geometry: Optional[str] = None,
    major_class_threshold: int = 20,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    persist_version: bool = True,
    profile: str = TILING_POLICY_VERSION,
    quality_confirmation: Optional[Mapping[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any], bool]:
    if not 0 < float(train_ratio) < 1 or not 0 <= float(test_ratio) < 1:
        raise ValueError("train_ratio/test_ratio 非法")
    if float(train_ratio) + float(test_ratio) >= 1:
        raise ValueError("train_ratio + test_ratio 必须小于 1")
    policy = resolve_tiling_profile(profile)
    original_class_names = _ordered_class_names(source_version)
    source_assets = db.query(DatasetAsset).filter(DatasetAsset.dataset_version_id == source_version.id).order_by(DatasetAsset.relative_path).all()
    if not source_assets:
        raise ValueError("标注版本没有资产")

    analysis = analyze_training_version(
        db,
        source_version,
        required_geometry=required_geometry,
        major_class_threshold=major_class_threshold,
    )
    if analysis["errors"]:
        raise ValueError(f"训练数据预检失败: {analysis['errors'][:3]}")
    class_mapping = {
        int(old_id): int(new_id)
        for old_id, new_id in analysis["class_mapping"].items()
    }
    if fixed_class_names is not None:
        class_names, class_mapping = resolve_fixed_training_class_map(
            original_class_names, fixed_class_names, analysis["class_stats"],
            (source_version.metadata_json or {}).get("annotation_inactive_class_ids") or [],
        )
    elif drop_zero_classes:
        class_names = list(analysis["effective_class_names"])
    else:
        class_names = original_class_names
        class_mapping = {class_id: class_id for class_id in range(len(class_names))}
    tiling_policy_config = tiling_policy(policy["version"])
    supplied_confirmation = quality_confirmation if isinstance(quality_confirmation, Mapping) else {}
    quality_confirmation_record = {
        "confirmed": bool(supplied_confirmation.get("confirmed", False)),
        "warning_codes": sorted({
            str(code)
            for code in (supplied_confirmation.get("warning_codes") or [])
            if str(code).strip()
        }),
        "thresholds": dict(supplied_confirmation.get("thresholds") or {})
        if isinstance(supplied_confirmation.get("thresholds") or {}, Mapping)
        else {},
    }
    target_class_ids = {
        index
        for index, name in enumerate(original_class_names)
        if annotation_class_key(name) in {
            annotation_class_key(value) for value in policy.get("target_class_names", [])
        }
    }
    context_class_ids = {
        index for index, name in enumerate(original_class_names)
        if str(name).strip().casefold() in {"person", "人"}
    } if policy["version"] == PHONE_BADGE_FOCUS_POLICY_VERSION else set()
    split_config = {
        "type": "grouped_source_random_seed_v2",
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "test_ratio": float(test_ratio),
        "val_ratio": 1.0 - float(train_ratio) - float(test_ratio),
        "tiling_before_split": False,
        "split_unit": "source_group",
        "profile": policy["version"],
        "required_geometry": required_geometry,
        "drop_zero_classes": bool(drop_zero_classes),
        "major_class_threshold": int(major_class_threshold),
        "class_names": list(class_names),
        "class_mapping": {str(old): new for old, new in class_mapping.items()},
        "image_size_policy": "dataset_adaptive_annotation_pixels",
        "image_size_candidates": list(policy["target_size_candidates"]),
        "annotation_retention": "100_percent",
        "tiling_policy": tiling_policy_config,
        "quality_confirmation": quality_confirmation_record,
    }
    fingerprint_payload = json.dumps(
        {"source": source_version.id, **split_config, "required_geometry": required_geometry},
        ensure_ascii=False,
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(fingerprint_payload.encode()).hexdigest()[:10]
    output_id = f"{_safe_dataset_name(output_name)}_{fingerprint}"
    if persist_version:
        existing = db.query(DatasetVersion).filter(
            DatasetVersion.dataset_id == output_id,
            DatasetVersion.parent_version_id == source_version.id,
        ).first()
        if existing:
            return existing, {"output_id": output_id, "split_config": existing.split_config_json or split_config}, False

    version_id = str(uuid.uuid4())
    stage = Path(staging_root) / version_id
    final_root = Path(training_root) / output_id
    source_root = Path(source_version.root_path).resolve()
    stage.mkdir(parents=True, exist_ok=False)
    try:
        used_names = set()
        tile_records: List[Dict[str, Any]] = []
        annotation_state_records: List[Dict[str, Any]] = []
        unresolved_records: List[Dict[str, Any]] = []
        neutral_root = stage / ".tiles"
        neutral_image_root = neutral_root / "images"
        neutral_label_root = neutral_root / "labels"
        neutral_image_root.mkdir(parents=True, exist_ok=False)
        neutral_label_root.mkdir(parents=True, exist_ok=False)
        source_annotation_count = 0
        annotated_source_asset_count = 0
        exportable_assets = [asset for asset in source_assets if asset.annotations_json]
        source_split_assignments, source_group_counts = _assign_source_splits(
            exportable_assets,
            train_ratio=float(train_ratio),
            test_ratio=float(test_ratio),
            seed=int(seed),
            required_class_ids=analysis["major_class_ids"],
        )
        for asset in source_assets:
            source_image = (source_root / asset.relative_path).resolve()
            source_image.relative_to(source_root)
            if not source_image.is_file():
                raise ValueError(f"源图片不存在: {asset.relative_path}")
            source_annotations = list(asset.annotations_json or [])
            if not source_annotations:
                continue
            annotated_source_asset_count += 1
            source_annotation_count += len(source_annotations)
            planned = plan_image_annotation_tiles(
                source_image,
                source_annotations,
                profile=policy["version"],
                target_class_ids=target_class_ids,
                context_class_ids=context_class_ids,
            )
            if policy["version"] == PHONE_BADGE_FOCUS_POLICY_VERSION:
                annotation_state_records.extend({"source_relative_path": asset.relative_path, **state} for state in planned.get("annotation_states", []))
                unresolved_records.extend({"source_relative_path": asset.relative_path, **state} for state in planned.get("unresolved", []))
            prefix = hashlib.sha1(asset.relative_path.encode("utf-8")).hexdigest()[:12]
            suffix = Path(asset.basename or asset.relative_path).suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                suffix = ".jpg"
            for tile_plan in planned["tiles"]:
                target_name = f"{prefix}_tile_{int(tile_plan['tile_index']):03d}{suffix}"
                if target_name in used_names:
                    raise ValueError(f"导出目标名称冲突: {target_name}")
                used_names.add(target_name)
                tile_records.append({
                    "target_name": target_name,
                    "source_image": source_image,
                    "source_relative_path": asset.relative_path,
                    "source_group": _export_source_group(asset),
                    "source_split": source_split_assignments[_export_source_group(asset)],
                    "source_annotations": source_annotations,
                    "source_size": list(planned["source_size"]),
                    "plan": tile_plan,
                })

        if not tile_records:
            raise ValueError("没有包含有效标注的图片")
        target_size = select_adaptive_tile_size(
            (record["plan"] for record in tile_records),
            candidates=policy["target_size_candidates"],
            minimum_annotation_pixels=policy["minimum_annotation_pixels"],
            target_annotation_pixel_coverage=policy["target_annotation_pixel_coverage"],
            ratio_field=(
                "target_annotation_short_side_ratios"
                if target_class_ids
                else "annotation_short_side_ratios"
            ),
        )
        image_size_results: List[Dict[str, Any]] = []
        total_tiles = len(tile_records)
        for tile_index, record in enumerate(tile_records, start=1):
            target_name = record["target_name"]
            target_image = neutral_image_root / target_name
            processed_image = render_annotation_tile(
                record["source_image"],
                target_image,
                record["source_annotations"],
                record["plan"],
                target_size=target_size,
                profile=policy["version"],
            )
            image_size_results.append(processed_image)
            lines = []
            for annotation in processed_image["annotations"]:
                old_class_id = _annotation_class_id(annotation)
                if old_class_id not in class_mapping:
                    continue
                lines.append(annotation_to_yolo_line(annotation, class_mapping[old_class_id]))
            label_path = neutral_label_root / f"{Path(target_name).stem}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            record["processed"] = processed_image
            record["neutral_image"] = target_image
            record["neutral_label"] = label_path
            if progress_callback and (tile_index == total_tiles or tile_index % 100 == 0):
                progress_callback(tile_index, total_tiles)

        actual_split_counts = {split: 0 for split in SPLITS}
        output_splits = ("train", "val", "test") if test_ratio else ("train", "val")
        for split in output_splits:
            (stage / split / "images").mkdir(parents=True, exist_ok=True)
            (stage / split / "labels").mkdir(parents=True, exist_ok=True)
        tile_manifest_entries: List[Dict[str, Any]] = []
        for record in sorted(tile_records, key=lambda item: item["target_name"]):
            target_name = record["target_name"]
            split = record["source_split"]
            actual_split_counts[split] += 1
            relative_image = f"{split}/images/{target_name}"
            relative_label = f"{split}/labels/{Path(target_name).stem}.txt"
            os.replace(record["neutral_image"], stage / relative_image)
            os.replace(record["neutral_label"], stage / relative_label)
            processed = record["processed"]
            tile_manifest_entries.append({
                "relative_path": relative_image,
                "label_path": relative_label,
                "source_relative_path": record["source_relative_path"],
                "source_group": record["source_group"],
                "source_size": record["source_size"],
                "crop_box": processed["crop_box"],
                "crop_size": processed.get("crop_size"),
                "upscale_ratio": processed.get("upscale_ratio"),
                "tile_type": processed.get("tile_type"),
                "anchor_index": processed.get("anchor_index"),
                "anchor_projected_size": processed.get("anchor_projected_size"),
                "anchor_desired_projected_size": processed.get("anchor_desired_projected_size"),
                "clipped_context_indexes": processed.get("clipped_context_indexes", []),
                "container_weak_indexes": processed.get("container_weak_indexes", []),
                "invalid_context_indexes": processed.get("invalid_context_indexes", []),
                "padding_ratio": processed.get("padding_ratio", 0.0),
                "target_size": int(target_size),
                "covered_annotation_indexes": processed["covered_annotation_indexes"],
                "fallback": bool(processed["fallback"]),
                "split": split,
            })
        shutil.rmtree(neutral_root)
        fallback_count = sum(1 for entry in tile_manifest_entries if entry["fallback"])
        tiling_summary = {
            "policy_version": policy["version"],
            "source_asset_count": len(source_assets),
            "annotated_source_asset_count": annotated_source_asset_count,
            "source_annotation_count": source_annotation_count,
            "tile_count": len(tile_manifest_entries),
            "fallback_count": fallback_count,
            "annotation_coverage": (
                sum(1 for state in annotation_state_records if state.get("seen")) / max(1, source_annotation_count)
                if policy["version"] == PHONE_BADGE_FOCUS_POLICY_VERSION
                else 1.0
            ),
            "target_size": int(target_size),
            "maximum_tiles_per_image": int(policy["maximum_tiles_per_image"]),
            "split_counts": actual_split_counts,
            "source_group_counts": source_group_counts,
            "source_group_count": len(source_split_assignments),
            "profile": policy["version"],
        }
        (stage / "tile_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "policy": tiling_policy_config,
                    "split_config": split_config,
                    "summary": tiling_summary,
                    "tiles": tile_manifest_entries,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if policy["version"] == PHONE_BADGE_FOCUS_POLICY_VERSION:
            metadata_dir = stage / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            (metadata_dir / "annotation_states.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in annotation_state_records), encoding="utf-8"
            )
            (metadata_dir / "unresolved_annotations.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in unresolved_records), encoding="utf-8"
            )
            (metadata_dir / "slice_statistics.json").write_text(json.dumps({
                "profile": policy["version"], "focused_tiles": sum(entry.get("tile_type") == "focused" for entry in tile_manifest_entries),
                "scene_tiles": sum(entry.get("tile_type") == "scene" for entry in tile_manifest_entries),
                "fallback_tiles": fallback_count, "unresolved_count": len(unresolved_records),
                "padding_tiles": sum(float(entry.get("padding_ratio") or 0) > 0 for entry in tile_manifest_entries),
                "container_weak_count": sum(len(entry.get("container_weak_indexes") or []) for entry in tile_manifest_entries),
                "invalid_context_count": sum(len(entry.get("invalid_context_indexes") or []) for entry in tile_manifest_entries),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        split_config.update({
            "actual_split_counts": actual_split_counts,
            "actual_source_group_counts": source_group_counts,
            "source_group_count": len(source_split_assignments),
            "tile_count": len(tile_manifest_entries),
            "target_size": int(target_size),
        })
        yaml_payload = {
            "path": str(final_root.resolve()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(class_names),
            "names": class_names,
        }
        if test_ratio:
            yaml_payload["test"] = "test/images"
        (stage / "dataset.yaml").write_text(yaml.safe_dump(yaml_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        if final_root.exists():
            raise ValueError(f"导出目录已存在但没有版本记录: {output_id}")
        final_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, final_root)
        report = validate_dataset(final_root, "training")
        if not report["valid"]:
            raise ValueError(f"导出数据集校验失败: {report['errors'][:3]}")
        lineage_by_path = {
            entry["relative_path"]: entry["source_group"]
            for entry in tile_manifest_entries
        }
        for item in report["assets"]:
            item["source_group"] = lineage_by_path.get(item["relative_path"])
        if persist_version:
            version, created = persist_manifest_version(
                db,
                dataset_id=output_id,
                dataset_role="training",
                root_path=final_root,
                manifest_root=manifest_root,
                class_names=class_names,
                assets=report["assets"],
                created_by=created_by,
                parent_version_id=source_version.id,
                split_config=split_config,
                validation={
                    "valid": True,
                    "statistics": report["statistics"],
                    "preflight": analysis,
                    "tiling": tiling_summary,
                },
                version_id=version_id,
            )
        else:
            workspace_manifest = {
                "dataset_id": output_id,
                "dataset_role": "training",
                "parent_version_id": source_version.id,
                "class_names": class_names,
                "split_config": split_config,
                "assets": report["assets"],
            }
            version = PreparedTrainingWorkspace(
                id=version_id,
                dataset_id=output_id,
                parent_version_id=source_version.id,
                root_path=str(final_root.resolve()),
                content_hash=manifest_content_hash(workspace_manifest),
                class_map_json={str(index): name for index, name in enumerate(class_names)},
            )
            created = True
        summary = {
            "output_id": output_id,
            "split_config": split_config,
            "statistics": report["statistics"],
            "preflight": analysis,
            "image_size": summarize_image_size_results(image_size_results),
            "tiling": tiling_summary,
        }
        return version, summary, created
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        if final_root.exists() and not db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first():
            shutil.rmtree(final_root, ignore_errors=True)
        raise


def create_initial_version(
    db: Session,
    dataset_id: str,
    root_path: Path,
    dataset_role: str,
    manifest_root: Path,
    created_by: str = "system",
) -> Tuple[DatasetVersion, Dict[str, Any], bool]:
    report = validate_dataset(root_path, dataset_role)
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_role": dataset_role,
        "root_path": str(Path(root_path).resolve()),
        "class_names": report["class_names"],
        "statistics": report["statistics"],
        "validation": {"valid": report["valid"], "errors": report["errors"], "warnings": report["warnings"]},
        "assets": report["assets"],
    }
    content_hash = manifest_content_hash(manifest)
    existing = db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == dataset_id,
        DatasetVersion.dataset_role == dataset_role,
        DatasetVersion.content_hash == content_hash,
    ).first()
    if existing:
        return existing, report, False

    version_id = str(uuid.uuid4())
    manifest_dir = Path(manifest_root) / dataset_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{version_id}.json"
    manifest.update({"version_id": version_id, "content_hash": content_hash, "created_at": datetime.now().isoformat()})
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, manifest_path)

    version = DatasetVersion(
        id=version_id,
        dataset_id=dataset_id,
        dataset_role=dataset_role,
        status="ready" if report["valid"] or dataset_role == "original" else "invalid",
        root_path=str(Path(root_path).resolve()),
        manifest_path=str(manifest_path.resolve()),
        class_map_json={str(index): name for index, name in enumerate(report["class_names"])},
        split_config_json={},
        content_hash=content_hash,
        created_by=created_by,
    )
    db.add(version)
    db.flush()
    for item in report["assets"]:
        db.add(DatasetAsset(
            dataset_version_id=version_id,
            relative_path=item["relative_path"],
            basename=item["basename"],
            size=item["size"],
            width=item["width"],
            height=item["height"],
            label_path=item.get("label_path"),
            source_group=item.get("source_group"),
            content_hash=item["content_hash"],
            annotations_json=item.get("annotations", []),
        ))
    db.commit()
    db.refresh(version)
    return version, report, True


def _export_source_group(asset: DatasetAsset) -> str:
    """Use persisted lineage for export isolation, with a per-asset fallback."""
    return str(asset.source_group or asset.relative_path)


def _percentile(values: Iterable[float], percentile: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return round(ordered[0], 4)
    position = (len(ordered) - 1) * float(percentile)
    lower = int(math.floor(position))
    upper = min(len(ordered) - 1, lower + 1)
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(value, 4)


def summarize_mobilephone_export_quality(
    source_assets: Iterable[DatasetAsset],
    class_names: Iterable[str],
    *,
    annotations_by_path: Optional[Mapping[str, Dict[str, Any]]] = None,
    profile: str = TILING_POLICY_VERSION,
    train_ratio: float = 0.8,
    test_ratio: float = 0.0,
    seed: int = 42,
    required_geometry: Optional[str] = None,
) -> Dict[str, Any]:
    """Build deterministic source/tile statistics without mutating assets or files."""
    policy = resolve_tiling_profile(profile)
    names = [str(value) for value in class_names]
    target_keys = {annotation_class_key(name) for name in policy.get("target_class_names", ["mobilephone"])}
    mobile_ids = {
        index for index, name in enumerate(names)
        if annotation_class_key(name) in target_keys
    }
    updates = annotations_by_path or {}
    assets = sorted(list(source_assets), key=lambda asset: str(asset.relative_path))
    class_counts: Counter = Counter()
    object_counts: List[int] = []
    source_groups: Dict[str, set] = {}
    split_counts: Counter = Counter()
    group_splits: Dict[str, set] = {}
    mobile_source_paths: set = set()
    mobile_groups: set = set()
    mobile_sample_assets: Dict[str, Dict[str, Any]] = {}
    annotated_source_paths: set = set()
    annotated_source_groups: set = set()
    source_mobile_short_sides: List[float] = []
    source_mobile_widths: List[float] = []
    source_mobile_heights: List[float] = []
    source_mobile_area_ratios: List[float] = []
    source_mobile_edge_count = 0
    tile_count_by_source: Counter = Counter()
    tile_short_sides: List[float] = []
    tile_gray_ratios: List[float] = []
    output_sizes: Counter = Counter()
    invalid_samples: List[Dict[str, Any]] = []
    tile_plans: List[Dict[str, Any]] = []
    mobile_plan_items: List[Tuple[Dict[str, Any], int, Tuple[float, float, float, float], int, int]] = []

    for asset in assets:
        path = str(asset.relative_path)
        update = updates.get(path) or {}
        annotations = filter_annotations_by_geometry(
            update.get("annotations", asset.annotations_json or []),
            required_geometry=required_geometry,
        )
        width = int(update.get("width") or asset.width or 0)
        height = int(update.get("height") or asset.height or 0)
        group = _export_source_group(asset)
        source_groups.setdefault(group, set()).add(path)
        parts = Path(path).parts
        split = parts[0] if parts and parts[0] in SPLITS else "unsplit"
        split_counts[split] += 1
        group_splits.setdefault(group, set()).add(split)
        object_counts.append(len(annotations))
        mobile_in_asset = False
        valid_annotation_count = 0
        for annotation_index, annotation in enumerate(annotations):
            try:
                class_id = _annotation_class_id(annotation)
                bounds = annotation_pixel_bounds(annotation, width, height) if width > 0 and height > 0 else None
            except (TypeError, ValueError, KeyError) as exc:
                if len(invalid_samples) < 20:
                    invalid_samples.append({"asset": path, "annotation_index": annotation_index, "reason": str(exc)})
                continue
            valid_annotation_count += 1
            class_counts[class_id] += 1
            if class_id not in mobile_ids or bounds is None:
                continue
            mobile_in_asset = True
            mobile_source_paths.add(path)
            mobile_groups.add(group)
            mobile_sample_assets.setdefault(path, {
                "asset_id": getattr(asset, "id", None),
                "relative_path": path,
            })
            left, top, right, bottom = bounds
            box_width = max(0.0, right - left)
            box_height = max(0.0, bottom - top)
            short_side = min(box_width, box_height)
            source_mobile_widths.append(box_width)
            source_mobile_heights.append(box_height)
            source_mobile_short_sides.append(short_side)
            source_mobile_area_ratios.append((box_width * box_height) / max(1.0, width * height))
            if min(left, top, width - right, height - bottom) <= max(16.0, min(width, height) * 0.02):
                source_mobile_edge_count += 1
        if valid_annotation_count and width > 0 and height > 0:
            annotated_source_paths.add(path)
            annotated_source_groups.add(group)
            try:
                plans = {
                    "source_size": [width, height],
                    "tiles": plan_annotation_tiles(
                        width,
                        height,
                        annotations,
                        profile=policy["version"],
                        target_class_ids=mobile_ids,
                    ),
                }
            except (OSError, ValueError):
                plans = {"source_size": [width, height], "tiles": []}
            for plan in plans["tiles"]:
                tile_plans.append(plan)
                tile_count_by_source[group] += 1
                operation = str(plan.get("operation") or "crop")
                crop_box = plan.get("crop_box")
                if operation == "crop" and crop_box:
                    side = max(1, int(crop_box[2]) - int(crop_box[0]))
                    gray_ratio = 0.0
                else:
                    side = max(width, height)
                    gray_ratio = 1.0 - ((width * height) / max(1.0, float(side * side)))
                tile_gray_ratios.append(round(max(0.0, gray_ratio), 6))
                for index in plan.get("covered_annotation_indexes") or []:
                    if int(index) >= len(annotations):
                        continue
                    try:
                        class_id = _annotation_class_id(annotations[int(index)])
                        bounds = annotation_pixel_bounds(annotations[int(index)], width, height)
                    except (TypeError, ValueError, KeyError):
                        continue
                    if class_id in mobile_ids:
                        left, top, right, bottom = bounds
                        mobile_plan_items.append((plan, int(index), bounds, width, height))
                        tile_short_sides.append(min(right - left, bottom - top) / max(1.0, float(side)))

    target_size = select_adaptive_tile_size(
        tile_plans,
        candidates=policy["target_size_candidates"],
        minimum_annotation_pixels=policy["minimum_annotation_pixels"],
        target_annotation_pixel_coverage=policy["target_annotation_pixel_coverage"],
        ratio_field=(
            "target_annotation_short_side_ratios"
            if mobile_ids
            else "annotation_short_side_ratios"
        ),
    )
    output_sizes[str(target_size)] = len(tile_plans)
    output_mobile_short_sides: List[float] = []
    for plan, _index, bounds, width, height in mobile_plan_items:
        left, top, right, bottom = bounds
        if plan.get("operation") == "crop" and plan.get("crop_box"):
            side = max(1.0, float(plan["crop_box"][2]) - float(plan["crop_box"][0]))
            output_mobile_short_sides.append(min(right - left, bottom - top) / side * target_size)
        else:
            scale = target_size / max(width, height)
            output_mobile_short_sides.append(min(right - left, bottom - top) * scale)

    cross_split_groups = sorted(group for group, splits in group_splits.items() if len(splits - {"unsplit"}) > 1)
    observed_split_source_group_counts: Counter = Counter()
    for splits in group_splits.values():
        for split in splits:
            observed_split_source_group_counts[split] += 1
    warnings: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    minimum_groups = 3 if float(test_ratio) > 0 else 2
    planned_source_group_counts: Counter = Counter()
    planned_split_error: Optional[str] = None
    try:
        exportable_assets = [asset for asset in assets if str(asset.relative_path) in annotated_source_paths]
        annotation_overrides = {
            path: list((updates.get(path) or {}).get("annotations") or [])
            for path in annotated_source_paths
            if path in updates
        }
        major_class_ids = [
            class_id
            for class_id, count in class_counts.items()
            if int(count) > 20
        ]
        _, planned_counts = _assign_source_splits(
            exportable_assets,
            train_ratio=float(train_ratio),
            test_ratio=float(test_ratio),
            seed=int(seed),
            required_class_ids=major_class_ids,
            annotation_overrides=annotation_overrides,
        )
        planned_source_group_counts.update(planned_counts)
    except ValueError as exc:
        planned_split_error = str(exc)
    if cross_split_groups:
        errors.append({"code": "source_group_split_leak", "count": len(cross_split_groups), "samples": cross_split_groups[:10]})
    if len(annotated_source_groups) < minimum_groups:
        errors.append({"code": "insufficient_source_groups", "required": minimum_groups, "actual": len(annotated_source_groups)})
    if planned_split_error and len(annotated_source_groups) >= minimum_groups:
        errors.append({"code": "source_group_split_unavailable", "message": planned_split_error})
    if policy["version"] == MOBILEPHONE_CONTEXT_POLICY_VERSION and not mobile_ids:
        warnings.append({"code": "mobilephone_class_not_found", "message": "类别列表中没有 mobilephone"})
    if mobile_ids and len(mobile_groups) < minimum_groups:
        errors.append({"code": "insufficient_mobilephone_groups", "required": minimum_groups, "actual": len(mobile_groups)})
    if invalid_samples:
        errors.append({"code": "invalid_annotations", "count": len(invalid_samples), "samples": invalid_samples[:10]})
    if source_mobile_short_sides and (
        (_percentile(source_mobile_short_sides, 0.5) or 0.0) < 12.0
        or (_percentile(output_mobile_short_sides, 0.05) or 0.0) < float(policy["minimum_annotation_pixels"])
    ):
        warnings.append({
            "code": "mobilephone_bbox_too_small",
            "minimum_output_pixels": float(policy["minimum_annotation_pixels"]),
            "source_short_side_p50": _percentile(source_mobile_short_sides, 0.5),
            "output_short_side_p05": _percentile(output_mobile_short_sides, 0.05),
        })
    gray_p95 = _percentile(tile_gray_ratios, 0.95) or 0.0
    if gray_p95 > 0.2:
        warnings.append({"code": "gray_border_high", "tile_gray_ratio_p95": gray_p95, "threshold": 0.2})
    if mobile_ids and sum(class_counts[class_id] for class_id in mobile_ids) < 20:
        warnings.append({"code": "class_below_minimum", "class": "mobilephone", "minimum": 20})

    def stats(values: Iterable[float]) -> Dict[str, Any]:
        values = list(values)
        return {"count": len(values), "p05": _percentile(values, 0.05), "p50": _percentile(values, 0.5), "p95": _percentile(values, 0.95)}

    return {
        "profile": policy["version"],
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "test_ratio": float(test_ratio),
        "source_image_count": len(assets),
        "annotated_source_image_count": len(annotated_source_paths),
        "empty_source_image_count": len(assets) - len(annotated_source_paths),
        "empty_source_image_rate": round((len(assets) - len(annotated_source_paths)) / max(1, len(assets)), 6),
        "mobilephone_source_image_count": len(mobile_source_paths),
        "annotation_count": int(sum(class_counts.values())),
        "mobilephone_annotation_count": int(sum(class_counts[class_id] for class_id in mobile_ids)),
        "class_counts": {str(class_id): int(count) for class_id, count in sorted(class_counts.items())},
        "class_names": names,
        "object_count_per_source_image": stats(object_counts),
        "mobilephone_source_bbox_width_pixels": stats(source_mobile_widths),
        "mobilephone_source_bbox_height_pixels": stats(source_mobile_heights),
        "mobilephone_source_bbox_short_side_pixels": stats(source_mobile_short_sides),
        "mobilephone_source_bbox_area_ratio": stats(source_mobile_area_ratios),
        "mobilephone_source_near_edge_count": int(source_mobile_edge_count),
        "mobilephone_source_group_count": len(mobile_groups),
        "source_group_count": len(source_groups),
        "annotated_source_group_count": len(annotated_source_groups),
        "split_counts": {split: int(split_counts[split]) for split in (*SPLITS, "unsplit")},
        "split_source_group_counts": {split: int(planned_source_group_counts[split]) for split in SPLITS},
        "observed_split_source_group_counts": {split: int(observed_split_source_group_counts[split]) for split in (*SPLITS, "unsplit")},
        "cross_split_source_group_count": len(cross_split_groups),
        "cross_split_source_group_samples": cross_split_groups[:10],
        "tile_count": len(tile_plans),
        "tiles_per_source_group": stats(tile_count_by_source.values()),
        "tile_gray_ratio": stats(tile_gray_ratios),
        "mobilephone_tile_short_side_ratio": stats(tile_short_sides),
        "mobilephone_output_short_side_pixels": stats(output_mobile_short_sides),
        "target_size": int(target_size),
        "target_size_counts": dict(sorted(output_sizes.items())),
        "profile_target_size_candidates": list(policy["target_size_candidates"]),
        "mobilephone_sample_paths": sorted(mobile_source_paths)[:100],
        "mobilephone_sample_previews": [
            mobile_sample_assets[path]
            for path in sorted(mobile_sample_assets)[:100]
            if mobile_sample_assets[path].get("asset_id") is not None
        ],
        "thresholds": {
            "minimum_annotation_pixels": float(policy["minimum_annotation_pixels"]),
            "source_bbox_short_side_warning_pixels": 12.0,
            "gray_border_p95_warning_ratio": 0.2,
            "minimum_mobilephone_annotations": 20,
            "minimum_annotated_source_groups": minimum_groups,
        },
        "warnings": warnings,
        "errors": errors,
        "can_export": not errors,
        "confirmation_required": bool(warnings),
    }


def _assign_source_splits(
    assets: Iterable[DatasetAsset],
    *,
    train_ratio: float,
    test_ratio: float,
    seed: int,
    required_class_ids: Iterable[int] = (),
    annotation_overrides: Optional[Mapping[str, Sequence[Dict[str, Any]]]] = None,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Assign complete source groups before any derived tile is created."""
    group_classes: Dict[str, set] = {}
    overrides = annotation_overrides or {}
    for asset in assets:
        group = _export_source_group(asset)
        ids = group_classes.setdefault(group, set())
        annotations = overrides.get(str(asset.relative_path), asset.annotations_json or [])
        for annotation in annotations:
            try:
                ids.add(_annotation_class_id(annotation))
            except (TypeError, ValueError):
                continue

    groups = sorted(group_classes)
    minimum_count = 3 if float(test_ratio) > 0 else 2
    if len(groups) < minimum_count:
        raise ValueError(
            f"独立来源组不足 {minimum_count} 个，无法形成独立 "
            + ("train/val/test" if minimum_count == 3 else "train/val")
        )
    val_ratio = 1.0 - float(train_ratio) - float(test_ratio)
    if val_ratio <= 0:
        raise ValueError("train_ratio + test_ratio 必须小于 1")
    if float(test_ratio) > 0:
        test_count = max(1, int(round(len(groups) * float(test_ratio))))
        val_count = max(1, int(round(len(groups) * val_ratio)))
        train_count = len(groups) - val_count - test_count
    else:
        test_count = 0
        train_count = max(1, min(int(len(groups) * float(train_ratio)), len(groups) - 1))
        val_count = len(groups) - train_count
    if train_count < 1:
        raise ValueError("独立来源组不足以保留 train 集")
    targets = {"train": train_count, "val": val_count, "test": test_count}
    assignments: Dict[str, str] = {}
    required_ids = sorted({int(value) for value in required_class_ids})
    rng_groups = list(groups)
    random.Random(int(seed)).shuffle(rng_groups)

    def reserve_required(split: str) -> None:
        if targets[split] <= 0:
            return
        for class_id in required_ids:
            if any(
                assigned_split == split and class_id in group_classes[group]
                for group, assigned_split in assignments.items()
            ):
                continue
            candidates = [
                group for group in rng_groups
                if group not in assignments and class_id in group_classes[group]
            ]
            if len(assignments_for_split(split)) >= targets[split] or not candidates:
                raise ValueError(
                    f"类别 {class_id} 的独立来源组不足以覆盖 {split}，请增加来源组样本"
                )
            missing = set(required_ids)
            candidates.sort(
                key=lambda group: (
                    -len(missing.intersection(group_classes[group])),
                    rng_groups.index(group),
                    group,
                )
            )
            assignments[candidates[0]] = split

    def assignments_for_split(split: str) -> List[str]:
        return [group for group, assigned_split in assignments.items() if assigned_split == split]

    # Reserve train first so rare classes cannot be consumed entirely by eval splits.
    reserve_required("train")
    reserve_required("val")
    reserve_required("test")
    for split in ("val", "test", "train"):
        if targets[split] <= 0:
            continue
        for group in rng_groups:
            if len(assignments_for_split(split)) >= targets[split]:
                break
            if group not in assignments:
                assignments[group] = split
    for group in rng_groups:
        assignments.setdefault(group, "train")

    for class_id in required_ids:
        covered = {assignments[group] for group in groups if class_id in group_classes[group]}
        required_splits = {"train", "val", "test"} if test_count else {"train", "val"}
        if not required_splits.issubset(covered):
            raise ValueError(
                f"类别 {class_id} 未覆盖独立 {','.join(sorted(required_splits))}，请增加来源组样本"
            )
    return assignments, {
        split: sum(1 for group in groups if assignments[group] == split)
        for split in SPLITS
    }
