"""Annotation-safe square image sizing for training dataset exports.

The original image and its persisted annotations remain untouched.  Exported
training images are either cropped to a square when every annotation fits in
that crop, or letterboxed when a square crop would cut any annotation.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageOps

from phone_badge_geometry import clip_context_annotation
from phone_badge_slicing import (
    ALLOWED_CROP_SIZES, MAX_UPSCALE_RATIO, MIN_CROP_SIZE, OUTPUT_SIZE,
    plan_phone_badge_tiles, validate_tile_before_export,
)


SUPPORTED_IMAGE_SIZES: Tuple[int, ...] = (320, 416, 512, 640, 1280)
ANNOTATION_EPSILON = 1e-6
TILING_POLICY_VERSION = "adaptive_annotation_tiles_v1"
MOBILEPHONE_CONTEXT_POLICY_VERSION = "mobilephone_context_v2"
PHONE_BADGE_FOCUS_POLICY_VERSION = "phone_badge_focus_v1"
MIN_CROP_SHORT_SIDE_RATIO = 0.15
TILE_WINDOW_IOU_THRESHOLD = 0.7
MAX_TILES_PER_IMAGE = 32
MAX_TILE_CANDIDATES = 256
TARGET_ANNOTATION_PIXEL_COVERAGE = 0.9
MIN_ANNOTATION_PIXELS = 24.0


def select_image_size(width: int, height: int) -> int:
    """Select the largest supported square not larger than the short side.

    This avoids unnecessary upscaling for normal images while still returning
    320 for very small uploads, which keeps every exported image on a known
    training-size tier.
    """
    short_side = min(int(width), int(height))
    if short_side <= 0:
        raise ValueError("图片尺寸无效")
    eligible = [size for size in SUPPORTED_IMAGE_SIZES if size <= short_side]
    return max(eligible) if eligible else SUPPORTED_IMAGE_SIZES[0]


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"标注字段 {field} 不是数字") from exc
    if not math.isfinite(result):
        raise ValueError(f"标注字段 {field} 不是有限数")
    return result


def _parse_annotation(annotation: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize persisted YOLO annotations to pixel-transformable geometry."""
    if not isinstance(annotation, dict):
        raise ValueError("标注必须是对象")
    raw_line = str(annotation.get("yolo_line") or "").strip()
    raw_class_id = annotation.get("class_id", annotation.get("class"))
    if raw_class_id is None:
        raw_class_id = raw_line.split()[0] if raw_line else 0
    class_id = int(raw_class_id)
    if raw_line:
        values = [_finite_float(value, "coordinate") for value in raw_line.split()[1:]]
        if len(values) == 4:
            cx, cy, width, height = values
            if width <= 0 or height <= 0:
                raise ValueError("bbox 宽高必须为正")
            points = [
                (cx - width / 2, cy - height / 2),
                (cx + width / 2, cy + height / 2),
            ]
            return {"class_id": class_id, "type": "bbox", "points": points}
        if len(values) >= 6 and len(values) % 2 == 0:
            points = list(zip(values[::2], values[1::2]))
            return {"class_id": class_id, "type": "polygon", "points": points}
        raise ValueError("YOLO 标注坐标格式无效")

    geometry = str(annotation.get("type") or "bbox").strip().lower()
    if geometry == "polygon":
        raw_points = annotation.get("points") or []
        if len(raw_points) < 6 or len(raw_points) % 2:
            raise ValueError("polygon 点数无效")
        values = [_finite_float(value, "polygon") for value in raw_points]
        return {
            "class_id": class_id,
            "type": "polygon",
            "points": list(zip(values[::2], values[1::2])),
        }
    if geometry == "bbox":
        cx = _finite_float(annotation.get("x"), "x")
        cy = _finite_float(annotation.get("y"), "y")
        width = _finite_float(annotation.get("width"), "width")
        height = _finite_float(annotation.get("height"), "height")
        if width <= 0 or height <= 0:
            raise ValueError("bbox 宽高必须为正")
        return {
            "class_id": class_id,
            "type": "bbox",
            "points": [(cx - width / 2, cy - height / 2), (cx + width / 2, cy + height / 2)],
        }
    raise ValueError(f"不支持的标注类型: {geometry}")


def _validate_normalized_points(points: Sequence[Tuple[float, float]]) -> None:
    if not points:
        raise ValueError("标注没有几何点")
    if any(
        point_x < -ANNOTATION_EPSILON
        or point_x > 1 + ANNOTATION_EPSILON
        or point_y < -ANNOTATION_EPSILON
        or point_y > 1 + ANNOTATION_EPSILON
        for point_x, point_y in points
    ):
        raise ValueError("标注超出原始图片范围，拒绝裁剪")


def _annotation_pixel_points(annotations: Iterable[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for annotation in annotations or []:
        item = _parse_annotation(annotation)
        _validate_normalized_points(item["points"])
        item["points"] = [(x * width, y * height) for x, y in item["points"]]
        normalized.append(item)
    return normalized


def _crop_origin(width: int, height: int, side: int, annotations: Sequence[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    """Return a centered feasible square crop origin, or None if impossible."""
    if side <= 0 or side > width or side > height:
        return None
    if not annotations:
        return ((width - side) // 2, (height - side) // 2)
    min_x = min(point[0] for item in annotations for point in item["points"])
    max_x = max(point[0] for item in annotations for point in item["points"])
    min_y = min(point[1] for item in annotations for point in item["points"])
    max_y = max(point[1] for item in annotations for point in item["points"])
    left_min = max(0.0, max_x - side)
    left_max = min(float(min_x), float(width - side))
    top_min = max(0.0, max_y - side)
    top_max = min(float(min_y), float(height - side))
    if left_min > left_max + ANNOTATION_EPSILON or top_min > top_max + ANNOTATION_EPSILON:
        return None
    integer_left_min = math.ceil(left_min - ANNOTATION_EPSILON)
    integer_left_max = math.floor(left_max + ANNOTATION_EPSILON)
    integer_top_min = math.ceil(top_min - ANNOTATION_EPSILON)
    integer_top_max = math.floor(top_max + ANNOTATION_EPSILON)
    if integer_left_min > integer_left_max or integer_top_min > integer_top_max:
        return None
    left = (integer_left_min + integer_left_max) // 2
    top = (integer_top_min + integer_top_max) // 2
    return max(0, min(left, width - side)), max(0, min(top, height - side))


def _transform_point(
    point: Tuple[float, float],
    *,
    operation: str,
    crop_left: int,
    crop_top: int,
    crop_side: int,
    source_width: int,
    source_height: int,
    target_size: int,
) -> Tuple[float, float]:
    x, y = point
    if operation == "crop":
        return ((x - crop_left) / crop_side, (y - crop_top) / crop_side)
    scale = target_size / max(source_width, source_height)
    pad_x = (target_size - source_width * scale) / 2
    pad_y = (target_size - source_height * scale) / 2
    return ((x * scale + pad_x) / target_size, (y * scale + pad_y) / target_size)


def _transform_annotations(
    annotations: Sequence[Dict[str, Any]],
    *,
    operation: str,
    crop_left: int,
    crop_top: int,
    crop_side: int,
    source_width: int,
    source_height: int,
    target_size: int,
) -> List[Dict[str, Any]]:
    transformed: List[Dict[str, Any]] = []
    for item in annotations:
        points = [
            _transform_point(
                point,
                operation=operation,
                crop_left=crop_left,
                crop_top=crop_top,
                crop_side=crop_side,
                source_width=source_width,
                source_height=source_height,
                target_size=target_size,
            )
            for point in item["points"]
        ]
        _validate_normalized_points(points)
        if item["type"] == "bbox":
            left, top = points[0]
            right, bottom = points[1]
            if right - left <= ANNOTATION_EPSILON or bottom - top <= ANNOTATION_EPSILON:
                raise ValueError("变换后的 bbox 尺寸无效")
            transformed.append(
                {
                    "class_id": item["class_id"],
                    "type": "bbox",
                    "x": (left + right) / 2,
                    "y": (top + bottom) / 2,
                    "width": right - left,
                    "height": bottom - top,
                }
            )
        else:
            transformed.append(
                {
                    "class_id": item["class_id"],
                    "type": "polygon",
                    "points": [value for point in points for value in point],
                }
            )
    return transformed


def _save_square_image(image: Image.Image, target_path: Path, target_size: int, operation: str, crop_box: Optional[Tuple[int, int, int, int]]) -> None:
    if operation == "crop":
        assert crop_box is not None
        output = image.crop(crop_box).resize((target_size, target_size), Image.Resampling.LANCZOS)
    else:
        scale = target_size / max(image.width, image.height)
        resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
        output = Image.new("RGB", (target_size, target_size), (114, 114, 114))
        output.paste(resized.convert("RGB"), ((target_size - resized.width) // 2, (target_size - resized.height) // 2))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = target_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        output.convert("RGB").save(target_path, quality=95, optimize=True)
    else:
        output.save(target_path)
    output.close()


def _annotation_bounds(annotation: Dict[str, Any]) -> Tuple[float, float, float, float]:
    points = annotation.get("points") or []
    if not points:
        raise ValueError("标注没有几何点")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _padding_ratio(bounds: Tuple[float, float, float, float], image_area: float) -> float:
    left, top, right, bottom = bounds
    area_ratio = max(0.0, right - left) * max(0.0, bottom - top) / max(1.0, image_area)
    if area_ratio <= 0.01:
        return 0.25
    if area_ratio <= 0.10:
        return 0.50
    return 0.80


def _square_window_for_bounds(
    bounds: Tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float,
    minimum_side: Optional[float] = None,
    minimum_side_ratio: Optional[float] = None,
    target_short_side_ratio_max: Optional[float] = None,
) -> Optional[Tuple[int, int, int, int]]:
    left, top, right, bottom = bounds
    box_width = max(0.0, right - left)
    box_height = max(0.0, bottom - top)
    short_side = min(int(image_width), int(image_height))
    if box_width <= 0 or box_height <= 0 or short_side <= 0:
        return None
    side = max(
        max(box_width, box_height) * (1.0 + 2.0 * float(padding_ratio)),
        float(short_side) * float(
            MIN_CROP_SHORT_SIDE_RATIO if minimum_side_ratio is None else minimum_side_ratio
        ),
        float(minimum_side or 0.0),
    )
    if target_short_side_ratio_max is not None:
        ratio_max = float(target_short_side_ratio_max)
        if ratio_max <= 0 or ratio_max > 1:
            raise ValueError("目标标注短边比例上限无效")
        side = max(side, max(box_width, box_height) / ratio_max)
    if side > short_side + ANNOTATION_EPSILON:
        return None
    side_int = max(1, min(short_side, int(math.ceil(side - ANNOTATION_EPSILON))))
    left_min = max(0, int(math.ceil(right - side_int - ANNOTATION_EPSILON)))
    left_max = min(image_width - side_int, int(math.floor(left + ANNOTATION_EPSILON)))
    top_min = max(0, int(math.ceil(bottom - side_int - ANNOTATION_EPSILON)))
    top_max = min(image_height - side_int, int(math.floor(top + ANNOTATION_EPSILON)))
    if left_min > left_max or top_min > top_max:
        return None
    crop_left = (left_min + left_max) // 2
    crop_top = (top_min + top_max) // 2
    window = (crop_left, crop_top, crop_left + side_int, crop_top + side_int)
    return window if _window_contains_bounds(window, bounds) else None


def _window_contains_bounds(
    window: Tuple[int, int, int, int],
    bounds: Tuple[float, float, float, float],
) -> bool:
    left, top, right, bottom = window
    box_left, box_top, box_right, box_bottom = bounds
    return (
        box_left >= left - ANNOTATION_EPSILON
        and box_top >= top - ANNOTATION_EPSILON
        and box_right <= right + ANNOTATION_EPSILON
        and box_bottom <= bottom + ANNOTATION_EPSILON
    )


def _window_intersects_bounds(
    window: Tuple[int, int, int, int],
    bounds: Tuple[float, float, float, float],
) -> bool:
    left, top, right, bottom = window
    box_left, box_top, box_right, box_bottom = bounds
    return not (
        box_right <= left + ANNOTATION_EPSILON
        or box_left >= right - ANNOTATION_EPSILON
        or box_bottom <= top + ANNOTATION_EPSILON
        or box_top >= bottom - ANNOTATION_EPSILON
    )


def _resolve_partial_intersections(
    initial_window: Tuple[int, int, int, int],
    annotation_bounds: Sequence[Tuple[float, float, float, float]],
    *,
    image_width: int,
    image_height: int,
    max_iterations: int = MAX_TILES_PER_IMAGE,
) -> Optional[Tuple[int, int, int, int]]:
    window = initial_window
    for _ in range(min(len(annotation_bounds) + 1, int(max_iterations))):
        partial_indexes = [
            index
            for index, bounds in enumerate(annotation_bounds)
            if _window_intersects_bounds(window, bounds) and not _window_contains_bounds(window, bounds)
        ]
        if not partial_indexes:
            return window
        included_bounds = [
            bounds
            for bounds in annotation_bounds
            if _window_contains_bounds(window, bounds)
        ]
        included_bounds.extend(annotation_bounds[index] for index in partial_indexes)
        union = (
            min(bounds[0] for bounds in included_bounds),
            min(bounds[1] for bounds in included_bounds),
            max(bounds[2] for bounds in included_bounds),
            max(bounds[3] for bounds in included_bounds),
        )
        expanded = _square_window_for_bounds(
            union,
            image_width=image_width,
            image_height=image_height,
            padding_ratio=0.0,
            minimum_side=float(window[2] - window[0]),
        )
        if expanded is None or expanded == window:
            return None
        window = expanded
    return None


def _window_iou(
    first: Tuple[int, int, int, int],
    second: Tuple[int, int, int, int],
) -> float:
    intersection_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return intersection / max(1.0, first_area + second_area - intersection)


def _candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
    crop_box = candidate["crop_box"]
    area = (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])
    return (
        -len(candidate["covered_annotation_indexes"]),
        area,
        crop_box[1],
        crop_box[0],
        tuple(candidate["covered_annotation_indexes"]),
    )


def _deduplicate_tile_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduplicated: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=_candidate_sort_key):
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(deduplicated)
                if existing["covered_annotation_indexes"] == candidate["covered_annotation_indexes"]
                or _window_iou(tuple(existing["crop_box"]), tuple(candidate["crop_box"])) >= TILE_WINDOW_IOU_THRESHOLD
            ),
            None,
        )
        if duplicate_index is None:
            deduplicated.append(candidate)
            continue
        existing = deduplicated[duplicate_index]
        if _candidate_sort_key(candidate) < _candidate_sort_key(existing):
            deduplicated[duplicate_index] = candidate
    return sorted(deduplicated, key=_candidate_sort_key)


def _fallback_tile_plan(
    annotations: Sequence[Dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    target_class_ids: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    max_dimension = max(image_width, image_height)
    target_ids = {int(value) for value in (target_class_ids or [])}
    short_side_ratios = []
    target_short_side_ratios = []
    for annotation in annotations:
        left, top, right, bottom = _annotation_bounds(annotation)
        ratio = min(right - left, bottom - top) / max(1.0, max_dimension)
        short_side_ratios.append(ratio)
        if int(annotation["class_id"]) in target_ids:
            target_short_side_ratios.append(ratio)
    return {
        "operation": "letterbox",
        "crop_box": None,
        "covered_annotation_indexes": list(range(len(annotations))),
        "fallback": True,
        "annotation_short_side_ratios": short_side_ratios,
        "target_annotation_short_side_ratios": target_short_side_ratios,
    }


def plan_annotation_tiles(
    image_width: int,
    image_height: int,
    annotations: Iterable[Dict[str, Any]],
    *,
    profile: str = TILING_POLICY_VERSION,
    target_class_ids: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    """Plan bounded square crops that fully cover every source annotation."""
    policy = resolve_tiling_profile(profile)
    profile_version = policy["version"]
    max_tiles = int(policy["maximum_tiles_per_image"])
    max_candidates = int(policy["maximum_tile_candidates"])
    width = int(image_width)
    height = int(image_height)
    if width <= 0 or height <= 0:
        raise ValueError("图片尺寸无效")
    pixel_annotations = _annotation_pixel_points(annotations, width, height)
    if not pixel_annotations:
        raise ValueError("无法为无标注图片生成切片")
    bounds = [_annotation_bounds(annotation) for annotation in pixel_annotations]
    image_area = float(width * height)
    candidates: List[Dict[str, Any]] = []
    target_ids = {int(value) for value in (target_class_ids or [])}
    preferred_indexes = [
        index for index, annotation in enumerate(pixel_annotations)
        if profile_version == MOBILEPHONE_CONTEXT_POLICY_VERSION
        and target_ids
        and int(annotation["class_id"]) in target_ids
    ]
    base_indexes = preferred_indexes or list(range(len(bounds)))
    if len(base_indexes) <= max_candidates:
        candidate_indexes = base_indexes
    else:
        stride = len(base_indexes) / float(max_candidates)
        candidate_indexes = sorted({
            base_indexes[min(len(base_indexes) - 1, int(index * stride))]
            for index in range(max_candidates)
        })
    for candidate_index in candidate_indexes:
        annotation_bounds = bounds[candidate_index]
        window = _square_window_for_bounds(
            annotation_bounds,
            image_width=width,
            image_height=height,
            padding_ratio=(
                float(policy.get("context_padding_ratio", _padding_ratio(annotation_bounds, image_area)))
                if profile_version == MOBILEPHONE_CONTEXT_POLICY_VERSION
                else _padding_ratio(annotation_bounds, image_area)
            ),
            minimum_side_ratio=float(policy["minimum_crop_short_side_ratio"]),
            target_short_side_ratio_max=policy.get("target_short_side_ratio_max"),
        )
        if window is None:
            continue
        window = _resolve_partial_intersections(
            window,
            bounds,
            image_width=width,
            image_height=height,
            max_iterations=max_tiles,
        )
        if window is None:
            continue
        covered = [index for index, item_bounds in enumerate(bounds) if _window_contains_bounds(window, item_bounds)]
        if not covered:
            continue
        side = float(window[2] - window[0])
        short_side_ratios = [
            min(bounds[index][2] - bounds[index][0], bounds[index][3] - bounds[index][1]) / max(1.0, side)
            for index in covered
        ]
        target_short_side_ratios = [
            ratio
            for index, ratio in zip(covered, short_side_ratios)
            if int(pixel_annotations[index]["class_id"]) in target_ids
        ]
        candidates.append({
            "operation": "crop",
            "crop_box": list(window),
            "covered_annotation_indexes": covered,
            "fallback": False,
            "annotation_short_side_ratios": short_side_ratios,
            "target_annotation_short_side_ratios": target_short_side_ratios,
        })

    candidates = _deduplicate_tile_candidates(candidates)
    annotation_indexes = set(range(len(pixel_annotations)))
    uncovered = set(annotation_indexes)
    selected: List[Dict[str, Any]] = []
    soft_limit = min(max_tiles, max(4, int(math.ceil(len(pixel_annotations) / 3.0))))
    remaining = list(candidates)
    while uncovered and remaining and len(selected) < soft_limit:
        remaining.sort(
            key=lambda candidate: (
                -len(uncovered.intersection(candidate["covered_annotation_indexes"])),
                *_candidate_sort_key(candidate)[1:],
            )
        )
        candidate = remaining.pop(0)
        newly_covered = uncovered.intersection(candidate["covered_annotation_indexes"])
        if not newly_covered:
            break
        selected.append(candidate)
        uncovered.difference_update(newly_covered)

    if uncovered and len(selected) < max_tiles - 1:
        uncovered_bounds = [bounds[index] for index in sorted(uncovered)]
        union = (
            min(item[0] for item in uncovered_bounds),
            min(item[1] for item in uncovered_bounds),
            max(item[2] for item in uncovered_bounds),
            max(item[3] for item in uncovered_bounds),
        )
        merged_window = _square_window_for_bounds(
            union,
            image_width=width,
            image_height=height,
            padding_ratio=0.0,
            minimum_side_ratio=float(policy["minimum_crop_short_side_ratio"]),
            target_short_side_ratio_max=policy.get("target_short_side_ratio_max"),
        )
        if merged_window is not None:
            merged_window = _resolve_partial_intersections(
                merged_window,
                bounds,
                image_width=width,
                image_height=height,
                max_iterations=max_tiles,
            )
        if merged_window is not None:
            covered = [
                index
                for index, item_bounds in enumerate(bounds)
                if _window_contains_bounds(merged_window, item_bounds)
            ]
            side = float(merged_window[2] - merged_window[0])
            selected.append({
                "operation": "crop",
                "crop_box": list(merged_window),
                "covered_annotation_indexes": covered,
                "fallback": False,
                "annotation_short_side_ratios": [
                    min(bounds[index][2] - bounds[index][0], bounds[index][3] - bounds[index][1]) / max(1.0, side)
                    for index in covered
                ],
                "target_annotation_short_side_ratios": [
                    min(bounds[index][2] - bounds[index][0], bounds[index][3] - bounds[index][1]) / max(1.0, side)
                    for index in covered
                    if int(pixel_annotations[index]["class_id"]) in target_ids
                ],
            })
            uncovered.difference_update(covered)

    if uncovered and len(selected) < max_tiles - 1:
        remaining = [
            candidate
            for candidate in candidates
            if candidate not in selected and uncovered.intersection(candidate["covered_annotation_indexes"])
        ]
        while uncovered and remaining and len(selected) < max_tiles - 1:
            remaining.sort(
                key=lambda candidate: (
                    -len(uncovered.intersection(candidate["covered_annotation_indexes"])),
                    *_candidate_sort_key(candidate)[1:],
                )
            )
            candidate = remaining.pop(0)
            selected.append(candidate)
            uncovered.difference_update(candidate["covered_annotation_indexes"])

    if uncovered:
        selected = selected[: max_tiles - 1]
        selected.append(
            _fallback_tile_plan(
                pixel_annotations,
                image_width=width,
                image_height=height,
                target_class_ids=target_ids,
            )
        )
        uncovered.clear()

    if not selected:
        selected = [
            _fallback_tile_plan(
                pixel_annotations,
                image_width=width,
                image_height=height,
                target_class_ids=target_ids,
            )
        ]

    covered_union = {
        index
        for tile in selected
        for index in tile["covered_annotation_indexes"]
    }
    if covered_union != annotation_indexes:
        raise ValueError("切片计划未完整覆盖所有标注")
    if len(selected) > max_tiles:
        raise ValueError("单张原图切片数量超过限制")
    return [
        {**tile, "tile_index": index, "policy_version": profile_version}
        for index, tile in enumerate(selected)
    ]


def select_adaptive_tile_size(
    tile_plans: Iterable[Dict[str, Any]],
    *,
    candidates: Optional[Iterable[int]] = None,
    minimum_annotation_pixels: Optional[float] = None,
    target_annotation_pixel_coverage: Optional[float] = None,
    ratio_field: str = "annotation_short_side_ratios",
) -> int:
    candidate_sizes = tuple(int(value) for value in (candidates or SUPPORTED_IMAGE_SIZES))
    if not candidate_sizes or any(value <= 0 for value in candidate_sizes):
        raise ValueError("目标尺寸候选无效")
    minimum_pixels = float(
        MIN_ANNOTATION_PIXELS if minimum_annotation_pixels is None else minimum_annotation_pixels
    )
    coverage = float(
        TARGET_ANNOTATION_PIXEL_COVERAGE
        if target_annotation_pixel_coverage is None
        else target_annotation_pixel_coverage
    )
    ratios = [
        float(ratio)
        for tile in tile_plans
        for ratio in (tile.get(ratio_field) or [])
        if math.isfinite(float(ratio)) and float(ratio) > 0
    ]
    if not ratios:
        return candidate_sizes[0]
    for candidate in candidate_sizes:
        passing = sum(1 for ratio in ratios if ratio * candidate >= minimum_pixels)
        if passing / len(ratios) >= coverage:
            return candidate
    return candidate_sizes[-1]


def adaptive_tiling_policy() -> Dict[str, Any]:
    return resolve_tiling_profile(TILING_POLICY_VERSION)


def plan_image_annotation_tiles(
    source_path: Path,
    annotations: Iterable[Dict[str, Any]],
    *,
    profile: str = TILING_POLICY_VERSION,
    target_class_ids: Optional[Iterable[int]] = None,
    context_class_ids: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened)
        width, height = int(image.width), int(image.height)
    if profile == PHONE_BADGE_FOCUS_POLICY_VERSION:
        source_annotations = list(annotations)
        class_roles = {int(value): "focused" for value in (target_class_ids or [])}
        class_roles.update({int(value): "context" for value in (context_class_ids or [])})
        # target_class_ids is intentionally supplied by dataset_versioning from class names.
        for class_id in range(max([int(item.get("class_id", -1)) for item in source_annotations] + [-1]) + 1):
            class_roles.setdefault(class_id, "other")
        pixel_annotations = _annotation_pixel_points(source_annotations, width, height)
        return {"source_size": [width, height], **plan_phone_badge_tiles(pixel_annotations, width, height, class_roles=class_roles, source_key=str(source_path))}
    return {
        "source_size": [width, height],
        "tiles": plan_annotation_tiles(
            width,
            height,
            annotations,
            profile=profile,
            target_class_ids=target_class_ids,
        ),
    }


def render_annotation_tile(
    source_path: Path,
    target_path: Path,
    annotations: Iterable[Dict[str, Any]],
    tile_plan: Dict[str, Any],
    *,
    target_size: int,
    profile: str = TILING_POLICY_VERSION,
) -> Dict[str, Any]:
    policy = resolve_tiling_profile(profile)
    if profile == PHONE_BADGE_FOCUS_POLICY_VERSION:
        return _render_phone_badge_tile(source_path, target_path, annotations, tile_plan, target_size=int(target_size), policy=policy)
    allowed_sizes = tuple(int(value) for value in policy["target_size_candidates"])
    if int(target_size) not in allowed_sizes:
        raise ValueError(f"图片尺寸必须为 {allowed_sizes}")
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        source_width, source_height = image.width, image.height
        pixel_annotations = _annotation_pixel_points(annotations, source_width, source_height)
        covered_indexes = [int(index) for index in tile_plan.get("covered_annotation_indexes") or []]
        if not covered_indexes or any(index < 0 or index >= len(pixel_annotations) for index in covered_indexes):
            raise ValueError("切片没有有效的完整标注")
        selected_annotations = [pixel_annotations[index] for index in covered_indexes]
        operation = str(tile_plan.get("operation") or "crop")
        crop_box_value = tile_plan.get("crop_box")
        if operation == "crop":
            if not isinstance(crop_box_value, (list, tuple)) or len(crop_box_value) != 4:
                raise ValueError("切片裁剪范围无效")
            crop_box = tuple(int(value) for value in crop_box_value)
            crop_side = crop_box[2] - crop_box[0]
            if crop_side <= 0 or crop_box[3] - crop_box[1] != crop_side:
                raise ValueError("切片必须是有效方形")
            transformed = _transform_annotations(
                selected_annotations,
                operation="crop",
                crop_left=crop_box[0],
                crop_top=crop_box[1],
                crop_side=crop_side,
                source_width=source_width,
                source_height=source_height,
                target_size=int(target_size),
            )
        elif operation == "letterbox":
            crop_box = None
            transformed = _transform_annotations(
                selected_annotations,
                operation="letterbox",
                crop_left=0,
                crop_top=0,
                crop_side=min(source_width, source_height),
                source_width=source_width,
                source_height=source_height,
                target_size=int(target_size),
            )
        else:
            raise ValueError(f"不支持的切片操作: {operation}")
        _save_square_image(image, Path(target_path), int(target_size), operation, crop_box)
    return {
        "source_size": [source_width, source_height],
        "target_size": int(target_size),
        "operation": operation,
        "crop_box": list(crop_box) if crop_box else None,
        "annotations": transformed,
        "annotation_count": len(transformed),
        "covered_annotation_indexes": covered_indexes,
        "fallback": bool(tile_plan.get("fallback", False)),
        "all_annotations_preserved": True,
        "policy_version": policy["version"],
    }


def _render_phone_badge_tile(source_path: Path, target_path: Path, annotations: Iterable[Dict[str, Any]], tile_plan: Dict[str, Any], *, target_size: int, policy: Mapping[str, Any]) -> Dict[str, Any]:
    if target_size != OUTPUT_SIZE:
        raise ValueError("phone_badge_focus_v1 输出必须为 1280")
    validate_tile_before_export(tile_plan)
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        source_width, source_height = image.size
        pixel_annotations = _annotation_pixel_points(annotations, source_width, source_height)
        crop = tuple(int(value) for value in tile_plan["crop_box"])
        side = crop[2] - crop[0]
        canvas = Image.new("RGB", (side, side), (114, 114, 114))
        source_box = (max(0, crop[0]), max(0, crop[1]), min(source_width, crop[2]), min(source_height, crop[3]))
        if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
            raise ValueError("切片不包含原图像素")
        canvas.paste(image.crop(source_box), (source_box[0] - crop[0], source_box[1] - crop[1]))
        if OUTPUT_SIZE / side > MAX_UPSCALE_RATIO:
            raise ValueError("INVALID_RESIZE_RATIO")
        canvas.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS).save(target_path)
        full_indexes = [int(index) for index in tile_plan.get("covered_annotation_indexes") or []]
        clipped_indexes = [int(index) for index in tile_plan.get("clipped_context_indexes") or []]
        transformed = _transform_annotations([pixel_annotations[index] for index in full_indexes], operation="crop", crop_left=crop[0], crop_top=crop[1], crop_side=side, source_width=source_width, source_height=source_height, target_size=OUTPUT_SIZE)
        invalid_context_indexes = []
        for index in clipped_indexes:
            try:
                transformed.extend(clip_context_annotation(pixel_annotations[index], crop, OUTPUT_SIZE))
            except ValueError as exc:
                if str(exc) != "INVALID_GEOMETRY":
                    raise
                invalid_context_indexes.append(index)
    padding_area = 1.0 - (max(0, source_box[2]-source_box[0]) * max(0, source_box[3]-source_box[1])) / float(side * side)
    return {"source_size": [source_width, source_height], "target_size": OUTPUT_SIZE, "operation": "crop", "crop_box": list(crop),
            "annotations": transformed, "annotation_count": len(transformed), "covered_annotation_indexes": full_indexes,
            "clipped_context_indexes": clipped_indexes, "invalid_context_indexes": invalid_context_indexes,
            "fallback": False, "padding_ratio": padding_area,
            "all_annotations_preserved": len(full_indexes) == len(pixel_annotations), "policy_version": policy["version"],
            "tile_type": tile_plan.get("tile_type"), "anchor_index": tile_plan.get("anchor_index"),
            "crop_size": side, "upscale_ratio": OUTPUT_SIZE / side,
            "anchor_projected_size": tile_plan.get("anchor_projected_size"), "anchor_desired_projected_size": tile_plan.get("anchor_desired_projected_size")}


def process_image_for_training(
    source_path: Path,
    target_path: Path,
    annotations: Iterable[Dict[str, Any]],
    *,
    target_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Write one square training image and transform every annotation.

    A crop is used only when all annotation geometry fits inside one square.
    Otherwise the complete source image is letterboxed, so no annotation is
    partially retained.
    """
    requested_size = int(target_size) if target_size is not None else None
    if requested_size is not None and requested_size not in SUPPORTED_IMAGE_SIZES:
        raise ValueError(f"图片尺寸必须为 {SUPPORTED_IMAGE_SIZES}")
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        source_width, source_height = image.width, image.height
        selected_size = requested_size or select_image_size(source_width, source_height)
        pixel_annotations = _annotation_pixel_points(annotations, source_width, source_height)
        side = min(source_width, source_height)
        origin = _crop_origin(source_width, source_height, side, pixel_annotations)
        operation = "crop" if origin is not None else "letterbox"
        crop_left, crop_top = origin or (0, 0)
        transformed = _transform_annotations(
            pixel_annotations,
            operation=operation,
            crop_left=crop_left,
            crop_top=crop_top,
            crop_side=side,
            source_width=source_width,
            source_height=source_height,
            target_size=selected_size,
        )
        crop_box = (crop_left, crop_top, crop_left + side, crop_top + side) if operation == "crop" else None
        _save_square_image(image, Path(target_path), selected_size, operation, crop_box)
    return {
        "source_size": [source_width, source_height],
        "target_size": selected_size,
        "operation": operation,
        "crop_box": list(crop_box) if crop_box else None,
        "annotations": transformed,
        "annotation_count": len(transformed),
        "all_annotations_preserved": True,
    }


def optimal_image_size_from_counts(target_size_counts: Mapping[Any, Any]) -> Optional[int]:
    """Choose one deterministic training tier for a dataset.

    Exported images can use different tiers because source dimensions differ,
    while Ultralytics accepts one global ``imgsz`` per training run.  Use the
    most frequent tier and prefer the larger tier on ties to retain more
    detail for small objects.  Unknown or non-positive entries are ignored so
    older metadata remains safe to consume.
    """
    candidates: List[Tuple[int, int]] = []
    for raw_size, raw_count in (target_size_counts or {}).items():
        try:
            size = int(raw_size)
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if size in SUPPORTED_IMAGE_SIZES and count > 0:
            candidates.append((count, size))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[1]


def summarize_image_size_results(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    target_sizes: Counter = Counter()
    operations: Counter = Counter()
    total = 0
    all_preserved = True
    adaptive_tiles = False
    profile_version = TILING_POLICY_VERSION
    for result in results:
        total += 1
        target_sizes[str(int(result["target_size"]))] += 1
        operations[str(result["operation"])] += 1
        all_preserved = all_preserved and bool(result.get("all_annotations_preserved", False))
        result_policy = result.get("policy_version")
        adaptive_tiles = adaptive_tiles or result_policy in {TILING_POLICY_VERSION, MOBILEPHONE_CONTEXT_POLICY_VERSION}
        if result_policy == MOBILEPHONE_CONTEXT_POLICY_VERSION:
            profile_version = result_policy
    optimal_size = optimal_image_size_from_counts(target_sizes)
    return {
        "policy": profile_version if profile_version == MOBILEPHONE_CONTEXT_POLICY_VERSION else ("dataset_adaptive_annotation_pixels" if adaptive_tiles else "auto_annotation_safe"),
        "candidates": list(resolve_tiling_profile(profile_version)["target_size_candidates"]),
        "selection": (
            "smallest_candidate_meeting_90_percent_at_profile_minimum_pixels" if profile_version == MOBILEPHONE_CONTEXT_POLICY_VERSION else "smallest_candidate_meeting_90_percent_at_24px"
            if adaptive_tiles
            else "largest_supported_not_exceeding_short_side"
        ),
        "optimal_size": optimal_size,
        "optimal_size_selection": (
            "single_adaptive_target_size"
            if adaptive_tiles
            else "most_frequent_target_size_larger_tier_on_tie"
        ),
        "annotation_retention": "100_percent",
        "all_annotations_preserved": all_preserved,
        "asset_count": total,
        "target_size_counts": dict(sorted(target_sizes.items(), key=lambda item: int(item[0]))),
        "operation_counts": dict(sorted(operations.items())),
    }


def resolve_tiling_profile(profile: Optional[str] = None) -> Dict[str, Any]:
    """Return an immutable, JSON-safe tiling profile or reject unknown values."""
    version = str(profile or TILING_POLICY_VERSION).strip()
    if version == TILING_POLICY_VERSION:
        return {
            "version": TILING_POLICY_VERSION,
            "padding_by_annotation_area": [
                {"max_ratio": 0.01, "per_side": 0.25},
                {"max_ratio": 0.10, "per_side": 0.50},
                {"max_ratio": 1.0, "per_side": 0.80},
            ],
            "minimum_crop_short_side_ratio": MIN_CROP_SHORT_SIDE_RATIO,
            "candidate_iou_deduplication": TILE_WINDOW_IOU_THRESHOLD,
            "maximum_tiles_per_image": MAX_TILES_PER_IMAGE,
            "maximum_tile_candidates": MAX_TILE_CANDIDATES,
            "minimum_annotation_pixels": MIN_ANNOTATION_PIXELS,
            "target_annotation_pixel_coverage": TARGET_ANNOTATION_PIXEL_COVERAGE,
            "target_size_candidates": list(SUPPORTED_IMAGE_SIZES),
            "target_class_names": [],
        }
    if version == MOBILEPHONE_CONTEXT_POLICY_VERSION:
        return {
            "version": MOBILEPHONE_CONTEXT_POLICY_VERSION,
            "context_padding_ratio": 0.0,
            "minimum_crop_short_side_ratio": 0.30,
            "target_short_side_ratio_range": [0.15, 0.35],
            "target_short_side_ratio_max": 0.35,
            "candidate_iou_deduplication": TILE_WINDOW_IOU_THRESHOLD,
            "maximum_tiles_per_image": 8,
            "maximum_tile_candidates": 128,
            "minimum_annotation_pixels": 32.0,
            "target_annotation_pixel_coverage": TARGET_ANNOTATION_PIXEL_COVERAGE,
            "target_size_candidates": [640, 1280],
            "target_class_names": ["mobilephone"],
        }
    if version == PHONE_BADGE_FOCUS_POLICY_VERSION:
        return {
            "version": PHONE_BADGE_FOCUS_POLICY_VERSION,
            "target_size_candidates": [OUTPUT_SIZE], "minimum_annotation_pixels": 24.0,
            "target_annotation_pixel_coverage": 0.9, "maximum_tiles_per_image": 32,
            "minimum_crop_short_side_ratio": MIN_CROP_SHORT_SIDE_RATIO, # Geometry-only preflight summary uses the generic planner; export uses plan_phone_badge_tiles.
            "maximum_tile_candidates": 256, "target_class_names": ["phone", "mobilephone", "手机", "badge", "工牌"],
            "allowed_crop_sizes": list(ALLOWED_CROP_SIZES), "minimum_crop_size": MIN_CROP_SIZE,
            "maximum_upscale_ratio": MAX_UPSCALE_RATIO, "full_frame_fallback": False,
        }
    raise ValueError(
        f"未知图像导出策略: {version}"
    )


def tiling_policy(profile: Optional[str] = None) -> Dict[str, Any]:
    return resolve_tiling_profile(profile)


def annotation_pixel_bounds(
    annotation: Dict[str, Any],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """Return validated pixel bounds for one normalized bbox or polygon."""
    parsed = _annotation_pixel_points([annotation], int(width), int(height))[0]
    return _annotation_bounds(parsed)
