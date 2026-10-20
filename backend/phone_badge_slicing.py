"""Focused phone/badge tiles for training-data export only."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

OUTPUT_SIZE = 1280
MAX_UPSCALE_RATIO = 2.0
MIN_CROP_SIZE = 640
MAX_CROP_SIZE = 2048
ALLOWED_CROP_SIZES = (640, 768, 896, 1024, 1280, 1536, 1792, 2048)
TARGET_PROJECTED_SIZE = {"small": 64.0, "medium": 96.0, "large": 160.0}
SATISFIED_RATIO = 0.90
CONTAINMENT_RATIO_THRESHOLD = 0.95
PARTIAL_IGNORE_RATIO = 0.05
MAX_CONTAINER_TILE_COVERAGE = 0.90
MIN_CONTAINER_VISIBLE_RATIO = 0.15
MAX_TILES_PER_IMAGE = 32

PHONE_NAMES = {"phone", "mobilephone", "手机"}
BADGE_NAMES = {"badge", "工牌"}
PERSON_NAMES = {"person", "人"}


@dataclass
class AnnotationSliceState:
    annotation_id: str
    class_id: int
    class_role: str
    seen: bool = False
    satisfied: bool = False
    best_projected_size: float = 0.0
    desired_projected_size: float = 0.0
    scale_type: str = ""
    contained_by: Optional[str] = None
    anchor_tile_id: Optional[str] = None
    unresolved_reason: Optional[str] = None


def resolve_class_roles(class_names: Iterable[Any]) -> Dict[int, str]:
    roles: Dict[int, str] = {}
    for index, name in enumerate(class_names):
        normalized = str(name).strip().casefold()
        roles[index] = "focused" if normalized in PHONE_NAMES | BADGE_NAMES else "context" if normalized in PERSON_NAMES else "other"
    return roles


def normalize_crop_size(requested_size: float, image_width: int, image_height: int) -> int:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("图片尺寸无效")
    requested = min(MAX_CROP_SIZE, max(MIN_CROP_SIZE, float(requested_size)))
    return next((size for size in ALLOWED_CROP_SIZES if size >= requested), ALLOWED_CROP_SIZES[-1])


def classify_annotation_scale(bounds: Tuple[float, float, float, float], image_width: int, image_height: int) -> str:
    scale = OUTPUT_SIZE / max(image_width, image_height)
    effective = min(bounds[2] - bounds[0], bounds[3] - bounds[1]) * scale
    return "small" if effective < 32 else "medium" if effective < 96 else "large"


def calculate_desired_projected_size(bounds: Tuple[float, float, float, float], scale_type: str) -> float:
    short = min(bounds[2] - bounds[0], bounds[3] - bounds[1])
    return min(TARGET_PROJECTED_SIZE[scale_type], max(0.0, short * MAX_UPSCALE_RATIO))


def calculate_projected_short_side(bounds: Tuple[float, float, float, float], crop_size: float) -> float:
    return min(bounds[2] - bounds[0], bounds[3] - bounds[1]) * OUTPUT_SIZE / crop_size


def _area(rect: Tuple[float, float, float, float]) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _intersection(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def visible_ratio(bounds: Tuple[float, float, float, float], crop: Tuple[float, float, float, float]) -> float:
    return _area(_intersection(bounds, crop)) / max(_area(bounds), 1.0)


def relation(bounds: Tuple[float, float, float, float], crop: Tuple[float, float, float, float]) -> str:
    visible = visible_ratio(bounds, crop)
    if visible <= 0:
        return "OUTSIDE"
    if visible >= 1.0 - 1e-9:
        return "FULL"
    return "PARTIAL"


def _window(center_x: float, center_y: float, size: int, width: int, height: int) -> Tuple[int, int, int, int]:
    # Keep original-pixel canvas when possible; otherwise allow symmetric padding.
    left = int(round(center_x - size / 2))
    top = int(round(center_y - size / 2))
    if width >= size:
        left = min(max(left, 0), width - size)
    if height >= size:
        top = min(max(top, 0), height - size)
    return left, top, left + size, top + size


def _bounds(item: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    points = item["points"]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def plan_phone_badge_tiles(annotations: Sequence[Mapping[str, Any]], image_width: int, image_height: int, *, class_roles: Mapping[int, str], source_key: str = "source") -> Dict[str, Any]:
    items = [{**dict(item), "bounds": _bounds(item), "index": index} for index, item in enumerate(annotations)]
    states: Dict[int, AnnotationSliceState] = {}
    for item in items:
        role = class_roles.get(int(item["class_id"]), "other")
        scale_type = classify_annotation_scale(item["bounds"], image_width, image_height)
        states[item["index"]] = AnnotationSliceState(
            annotation_id=f"{source_key}#{item['index']}", class_id=int(item["class_id"]), class_role=role,
            scale_type=scale_type, desired_projected_size=calculate_desired_projected_size(item["bounds"], scale_type),
        )
    for small in items:
        state = states[small["index"]]
        if state.class_role != "focused":
            continue
        candidates = []
        for person in items:
            if class_roles.get(int(person["class_id"]), "other") != "context":
                continue
            ratio = _area(_intersection(small["bounds"], person["bounds"])) / max(_area(small["bounds"]), 1.0)
            if ratio >= CONTAINMENT_RATIO_THRESHOLD:
                candidates.append((-ratio, _area(person["bounds"]), person["index"]))
        if candidates:
            state.contained_by = states[min(candidates)[2]].annotation_id

    tiles: List[Dict[str, Any]] = []
    # Scene tiles provide context but never use full-frame fallback.
    for item in items:
        if class_roles.get(int(item["class_id"]), "other") != "context":
            continue
        size = normalize_crop_size(max(item["bounds"][2]-item["bounds"][0], item["bounds"][3]-item["bounds"][1]), image_width, image_height)
        crop = _window((item["bounds"][0]+item["bounds"][2])/2, (item["bounds"][1]+item["bounds"][3])/2, size, image_width, image_height)
        tiles.append({"tile_type": "scene", "anchor_index": None, "crop_box": list(crop), "crop_size": size})

    for index, item in sorted(enumerate(items), key=lambda pair: (states[pair[0]].best_projected_size, pair[0])):
        state = states[index]
        if state.class_role != "focused" or state.satisfied:
            continue
        short = min(item["bounds"][2]-item["bounds"][0], item["bounds"][3]-item["bounds"][1])
        long = max(item["bounds"][2]-item["bounds"][0], item["bounds"][3]-item["bounds"][1])
        desired = state.desired_projected_size
        requested = max(MIN_CROP_SIZE, long * 6.0, short * OUTPUT_SIZE / desired if desired > 0 else MIN_CROP_SIZE)
        size = normalize_crop_size(requested, image_width, image_height)
        center_x, center_y = (item["bounds"][0]+item["bounds"][2])/2, (item["bounds"][1]+item["bounds"][3])/2
        valid: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
        for dx, dy in ((0,0),(-.1,0),(.1,0),(0,-.1),(0,.1)):
            crop = _window(center_x + dx * size, center_y + dy * size, size, image_width, image_height)
            if relation(item["bounds"], crop) != "FULL":
                continue
            ignored, invalid, full, clipped, weak_context = [], False, [], [], []
            for other in items:
                rel = relation(other["bounds"], crop)
                role = class_roles.get(int(other["class_id"]), "other")
                if rel == "FULL": full.append(other["index"])
                elif rel == "PARTIAL" and other["index"] != item["index"]:
                    if role == "focused":
                        if visible_ratio(other["bounds"], crop) <= PARTIAL_IGNORE_RATIO: ignored.append(other["index"])
                        else: invalid = True; break
                    elif role == "context":
                        visible = visible_ratio(other["bounds"], crop)
                        tile_coverage = _area(_intersection(other["bounds"], crop)) / max(size * size, 1.0)
                        if tile_coverage <= 0.90 and visible >= 0.15:
                            clipped.append(other["index"])
                        else:
                            weak_context.append(other["index"])
                    else: invalid = True; break
            if invalid: continue
            projected = calculate_projected_short_side(item["bounds"], size)
            valid.append(((abs(projected-desired), abs(dx)+abs(dy), -len(full), crop[1], crop[0]), {
                "tile_type":"focused", "anchor_index":item["index"], "crop_box":list(crop), "crop_size":size,
                "full_indexes":full, "clipped_context_indexes":clipped, "container_weak_indexes":weak_context,
                "partial_small_ignored":ignored,
                "anchor_projected_size":projected, "anchor_desired_projected_size":desired,
            }))
        if not valid:
            state.unresolved_reason = "NO_VALID_FOCUSED_CANDIDATE"; continue
        _, best = min(valid, key=lambda entry: entry[0])
        state.seen = True; state.best_projected_size = best["anchor_projected_size"]
        state.satisfied = state.best_projected_size >= state.desired_projected_size * SATISFIED_RATIO
        state.anchor_tile_id = f"focused-{len(tiles):03d}"
        tiles.append(best)
    if len(tiles) > MAX_TILES_PER_IMAGE:
        tiles = tiles[:MAX_TILES_PER_IMAGE]
    for tile_index, tile in enumerate(tiles):
        crop = tuple(tile["crop_box"])
        full = tile.get("full_indexes") or [item["index"] for item in items if relation(item["bounds"], crop) == "FULL"]
        tile.update(tile_index=tile_index, operation="crop", fallback=False, covered_annotation_indexes=full,
                    annotation_short_side_ratios=[min(items[i]["bounds"][2]-items[i]["bounds"][0], items[i]["bounds"][3]-items[i]["bounds"][1])/tile["crop_size"] for i in full],
                    target_annotation_short_side_ratios=[])
    return {"tiles": tiles, "annotation_states": [asdict(state) for state in states.values()],
            "unresolved": [asdict(state) for state in states.values() if state.unresolved_reason], "fallback_count": 0}


def validate_tile_before_export(tile: Mapping[str, Any]) -> None:
    size = int(tile.get("crop_size") or 0)
    if size not in ALLOWED_CROP_SIZES or size < MIN_CROP_SIZE or OUTPUT_SIZE / size > MAX_UPSCALE_RATIO:
        raise ValueError("INVALID_RESIZE_RATIO")
    if tile.get("tile_type") == "focused" and tile.get("anchor_index") is None:
        raise ValueError("focused tile missing anchor")
