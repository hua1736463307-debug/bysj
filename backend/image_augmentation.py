"""Server-side image augmentation primitives shared by dataset jobs."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image, ImageEnhance, ImageOps


ROTATION_OPTIONS = (30, 60, 90, 135, 180, 225, 270, 315)
MIRROR_OPTIONS = {"none", "horizontal", "vertical"}
BRIGHTNESS_OPTIONS = {0.8, 0.6, 0.4}
GRAYSCALE_OPTIONS = {0.3, 0.6, 1.0}
CONTRAST_OPTIONS = {0.6, 1.0, 1.4}


def _number_set(values: Iterable[Any], allowed: set[float], label: str) -> List[float]:
    result: List[float] = []
    for raw in values or []:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{label}选项无效")
        if value not in allowed:
            raise ValueError(f"{label}选项无效: {raw}")
        if value not in result:
            result.append(value)
    return result


def normalize_augmentation_operations(config: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    """Validate the UI configuration and return one operation per selection."""
    config = config or {}
    operations: List[Dict[str, Any]] = []
    rotations = []
    for raw in config.get("rotations") or []:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError("旋转角度选项无效")
        if value not in ROTATION_OPTIONS:
            raise ValueError(f"旋转角度选项无效: {raw}")
        if value not in rotations:
            rotations.append(value)

    operations.extend({"type": "rotation", "value": value} for value in rotations)

    mirror = str(config.get("mirror") or "none").strip().lower()
    if mirror not in MIRROR_OPTIONS:
        raise ValueError("镜像选项无效")
    if mirror != "none":
        operations.append({"type": "mirror", "value": mirror})

    operations.extend({"type": "brightness", "value": value} for value in _number_set(
        config.get("brightnessLevels"), BRIGHTNESS_OPTIONS, "亮度"
    ))
    operations.extend({"type": "grayscale", "value": value} for value in _number_set(
        config.get("grayscaleLevels"), GRAYSCALE_OPTIONS, "灰度"
    ))
    operations.extend({"type": "contrast", "value": value} for value in _number_set(
        config.get("contrastLevels"), CONTRAST_OPTIONS, "对比度"
    ))
    if not operations:
        raise ValueError("至少需要选择一种增强操作")
    return operations


def operation_label(operation: Dict[str, Any]) -> str:
    operation_type = str(operation.get("type") or "transform")
    value = operation.get("value")
    if operation_type == "rotation":
        return f"rotate-{int(value)}"
    if operation_type == "mirror":
        return f"mirror-{value}"
    return f"{operation_type}-{round(float(value) * 100)}"


def augmented_dimensions(width: int, height: int, rotation: float = 0) -> Tuple[int, int]:
    radians = math.radians(float(rotation or 0))
    return (
        max(1, math.ceil(abs(math.cos(radians)) * width + abs(math.sin(radians)) * height - 1e-9)),
        max(1, math.ceil(abs(math.sin(radians)) * width + abs(math.cos(radians)) * height - 1e-9)),
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _transform_point(point: Tuple[float, float], width: int, height: int, operation: Dict[str, Any]) -> Tuple[float, float]:
    rotation = float(operation.get("value") or 0) if operation.get("type") == "rotation" else 0.0
    radians = math.radians(rotation)
    mirror_x = -1.0 if operation.get("type") == "mirror" and operation.get("value") == "horizontal" else 1.0
    mirror_y = -1.0 if operation.get("type") == "mirror" and operation.get("value") == "vertical" else 1.0
    local_x = (float(point[0]) - 0.5) * width * mirror_x
    local_y = (float(point[1]) - 0.5) * height * mirror_y
    rotated_x = local_x * math.cos(radians) - local_y * math.sin(radians)
    rotated_y = local_x * math.sin(radians) + local_y * math.cos(radians)
    target_width, target_height = augmented_dimensions(width, height, rotation)
    return (
        _clamp((target_width / 2 + rotated_x) / target_width),
        _clamp((target_height / 2 + rotated_y) / target_height),
    )


def _bbox(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float] | None:
    if not points:
        return None
    min_x = _clamp(min(point[0] for point in points))
    min_y = _clamp(min(point[1] for point in points))
    max_x = _clamp(max(point[0] for point in points))
    max_y = _clamp(max(point[1] for point in points))
    if max_x <= min_x or max_y <= min_y:
        return None
    return min_x, min_y, max_x - min_x, max_y - min_y


def transform_annotations(annotations: Iterable[Dict[str, Any]], width: int, height: int, operation: Dict[str, Any]) -> List[Dict[str, Any]]:
    transformed: List[Dict[str, Any]] = []
    for annotation in annotations or []:
        if annotation.get("type") == "polygon" and isinstance(annotation.get("points"), list):
            raw_points = annotation.get("points") or []
            points = [
                _transform_point((raw_points[index], raw_points[index + 1]), width, height, operation)
                for index in range(0, len(raw_points) - 1, 2)
            ]
            bounds = _bbox(points)
            if not bounds or len(points) < 3:
                continue
            transformed.append({**annotation, "x": bounds[0], "y": bounds[1], "width": bounds[2], "height": bounds[3], "points": [value for point in points for value in point]})
            continue

        x = float(annotation.get("x") or 0)
        y = float(annotation.get("y") or 0)
        box_width = float(annotation.get("width") or 0)
        box_height = float(annotation.get("height") or 0)
        points = [
            _transform_point((x, y), width, height, operation),
            _transform_point((x + box_width, y), width, height, operation),
            _transform_point((x + box_width, y + box_height), width, height, operation),
            _transform_point((x, y + box_height), width, height, operation),
        ]
        bounds = _bbox(points)
        if bounds:
            transformed.append({**annotation, "x": bounds[0], "y": bounds[1], "width": bounds[2], "height": bounds[3]})
    return transformed


def apply_image_operation(image: Image.Image, operation: Dict[str, Any]) -> Image.Image:
    operation_type = operation.get("type")
    value = operation.get("value")
    if operation_type == "rotation":
        return image.rotate(float(value), expand=True, resample=Image.Resampling.BICUBIC)
    if operation_type == "mirror":
        if value == "horizontal":
            return ImageOps.mirror(image)
        if value == "vertical":
            return ImageOps.flip(image)
    if operation_type == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(value))
    if operation_type == "grayscale":
        grayscale = ImageOps.grayscale(image).convert(image.mode if image.mode in {"RGB", "RGBA"} else "RGB")
        base = image.convert(grayscale.mode)
        return Image.blend(base, grayscale, float(value))
    if operation_type == "contrast":
        return ImageEnhance.Contrast(image).enhance(float(value))
    raise ValueError(f"未知增强操作: {operation_type}")


def augmented_filename(source_name: str, operation: Dict[str, Any]) -> str:
    stem = Path(source_name).stem
    safe_label = re.sub(r"[^0-9A-Za-z_-]+", "_", operation_label(operation)).strip("_")
    return f"{stem}__aug_{safe_label}.png"
