"""Helpers for exporting large annotation jobs without a frontend-sized payload."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from annotation_class_aliases import annotation_class_key, class_name_ids


def normalize_class_names(values: Iterable[Any]) -> List[str]:
    names: List[str] = []
    seen = set()
    for index, value in enumerate(values or []):
        name = str(value or "").strip()
        if not name:
            raise ValueError(f"第 {index + 1} 个类别名称为空")
        if len(name) > 120 or any(ord(char) < 32 for char in name):
            raise ValueError(f"类别名称不合法: {name[:30]}")
        if name in seen:
            raise ValueError(f"类别名称重复: {name}")
        names.append(name)
        seen.add(name)
    if not names:
        raise ValueError("至少需要一个类别")
    return names


def ordered_class_names(value: Any) -> List[str]:
    """Normalize a YOLO names list or numeric mapping while preserving class IDs."""
    if isinstance(value, dict):
        try:
            keys = sorted(value, key=lambda item: int(item))
        except (TypeError, ValueError):
            keys = sorted(value, key=str)
        value = [value[key] for key in keys]
    if not isinstance(value, (list, tuple)):
        return []
    try:
        return normalize_class_names(value)
    except ValueError:
        return []


def resolve_ai_class_id(
    annotation: Mapping[str, Any],
    class_names: List[str],
    *,
    class_mode: str = "auto",
    target_class: Any = None,
    inactive_ids: Iterable[int] = (),
) -> int:
    if class_mode == "manual":
        if isinstance(target_class, str):
            name = target_class.strip()
            if not name:
                raise ValueError("手动标注类别不能为空")
            matched = class_name_ids(class_names, inactive_ids).get(annotation_class_key(name))
            if matched is not None:
                return matched
            if any(annotation_class_key(name) == annotation_class_key(old) for old in class_names):
                raise ValueError("目标类别已停用，不能重新写入标注")
            if name not in class_names:
                class_names.append(name)
            return class_names.index(name)
        try:
            class_id = int(target_class)
        except (TypeError, ValueError):
            raise ValueError("手动标注类别无效")
        if class_id < 0 or class_id >= len(class_names):
            raise ValueError("手动标注类别超出范围")
        return class_id

    class_name = str(annotation.get("class_name") or "").strip()
    if not class_name:
        raw_class = annotation.get("class", annotation.get("class_id", "unknown"))
        class_name = f"class_{raw_class}"
    matched = class_name_ids(class_names, inactive_ids).get(annotation_class_key(class_name))
    if matched is not None:
        return matched
    if any(annotation_class_key(class_name) == annotation_class_key(old) for old in class_names):
        raise ValueError("目标类别已停用，不能重新写入标注")
    if class_name not in class_names:
        class_names.append(class_name)
    return class_names.index(class_name)


def frontend_annotation_to_manifest(
    annotation: Mapping[str, Any],
    *,
    class_id: Optional[int] = None,
    image_width: int = 0,
    image_height: int = 0,
) -> Dict[str, Any]:
    """Convert the annotation-page shape to the manifest's stable YOLO line."""
    try:
        target_class = int(annotation.get("class", annotation.get("class_id", 0)) if class_id is None else class_id)
        metadata = {
            key: annotation[key]
            for key in (
                "coordinate_space",
                "image_width",
                "image_height",
                "image_orientation",
            )
            if key in annotation and annotation[key] is not None
        }
        if image_width > 0 and image_height > 0:
            metadata["coordinate_space"] = "raw"
            metadata["image_width"] = int(image_width)
            metadata["image_height"] = int(image_height)
            metadata["image_orientation"] = 1
        raw_points = annotation.get("points") or annotation.get("polygon") or annotation.get("segmentation") or []
        if raw_points:
            points = list(raw_points)
            if points and not isinstance(points[0], (dict, list, tuple)):
                if len(points) < 6 or len(points) % 2:
                    raise ValueError("多边形点数无效")
                coordinates = [float(value) for value in points]
            else:
                coordinates = []
                for point in points:
                    x = float(point.get("x") if isinstance(point, dict) else point[0])
                    y = float(point.get("y") if isinstance(point, dict) else point[1])
                    if max(abs(x), abs(y)) > 1 and image_width and image_height:
                        x /= image_width
                        y /= image_height
                    coordinates.extend([min(1.0, max(0.0, x)), min(1.0, max(0.0, y))])
            if len(coordinates) < 6:
                raise ValueError("多边形点数无效")
            return {
                "yolo_line": f"{target_class} " + " ".join(f"{value:.6f}" for value in coordinates),
                **metadata,
            }

        x = float(annotation.get("x", 0))
        y = float(annotation.get("y", 0))
        width = float(annotation.get("width", 0))
        height = float(annotation.get("height", 0))
        if max(abs(x), abs(y), abs(width), abs(height)) > 1 and image_width and image_height:
            center_x = (x + width / 2) / image_width
            center_y = (y + height / 2) / image_height
            rel_width = width / image_width
            rel_height = height / image_height
        else:
            center_x = x + width / 2
            center_y = y + height / 2
            rel_width = width
            rel_height = height
        values = [min(1.0, max(0.0, value)) for value in (center_x, center_y, rel_width, rel_height)]
        if values[2] <= 0 or values[3] <= 0:
            raise ValueError("标注框尺寸无效")
        return {
            "yolo_line": (
                f"{target_class} {values[0]:.6f} {values[1]:.6f} "
                f"{values[2]:.6f} {values[3]:.6f}"
            ),
            **metadata,
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"标注格式无效: {exc}") from exc


def build_asset_path_indexes(
    source_root: Path,
    relative_paths: Iterable[str],
) -> Tuple[Dict[str, str], Dict[str, Optional[str]]]:
    absolute_paths: Dict[str, str] = {}
    basenames: Dict[str, Optional[str]] = {}
    root = Path(source_root).resolve()
    for raw_relative in relative_paths:
        relative = str(raw_relative).replace("\\", "/")
        absolute_paths[str((root / relative).resolve())] = relative
        basename = Path(relative).name
        basenames[basename] = relative if basename not in basenames else None
    return absolute_paths, basenames


def resolve_result_relative_path(
    item: Mapping[str, Any],
    source_root: Path,
    known_relative_paths: Iterable[str],
    absolute_paths: Mapping[str, str],
    basenames: Mapping[str, Optional[str]],
) -> str:
    root = Path(source_root).resolve()
    known = set(known_relative_paths)
    for key in ("relative_path", "asset_key", "path", "saved_path"):
        raw_value = str(item.get(key) or "").strip()
        if not raw_value:
            continue
        normalized = raw_value.replace("\\", "/").lstrip("./")
        if normalized in known:
            return normalized
        candidate = Path(raw_value)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = absolute_paths.get(str(candidate.resolve()))
        if resolved:
            return resolved
    basename = Path(str(item.get("filename") or item.get("name") or "")).name
    relative = basenames.get(basename)
    if relative:
        return relative
    raise ValueError(f"无法将 AI 标注结果匹配到数据集资产: {basename or 'unknown'}")
