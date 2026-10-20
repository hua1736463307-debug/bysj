"""Geometry helpers for clipped context labels in phone_badge_focus_v1."""
from __future__ import annotations
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _bounds(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    return min(x for x, _ in points), min(y for _, y in points), max(x for x, _ in points), max(y for _, y in points)


def _finite(points: Sequence[Tuple[float, float]]) -> bool:
    return len(points) >= 3 and all(math.isfinite(x) and math.isfinite(y) for x, y in points)


def _polygon_components(geometry: Any) -> List[Any]:
    """Return only area-bearing polygon parts from a Shapely result."""
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        return [part for child in geometry.geoms for part in _polygon_components(child)]
    return []


def clip_context_annotation(item: Mapping[str, Any], crop: Tuple[int, int, int, int], output_size: int) -> List[Dict[str, Any]]:
    """Clip a bbox/polygon to crop. Raises for invalid source polygon instead of repairing it."""
    left, top, right, bottom = crop
    size = right - left
    if size <= 0 or bottom - top != size:
        raise ValueError("INVALID_GEOMETRY")
    points = [(float(x), float(y)) for x, y in item["points"]]
    if item["type"] == "bbox":
        x1, y1, x2, y2 = _bounds(points)
        x1, y1, x2, y2 = max(left, x1), max(top, y1), min(right, x2), min(bottom, y2)
        if x2 <= x1 or y2 <= y1: return []
        return [{"class_id": item["class_id"], "type": "bbox", "x": ((x1+x2)/2-left)/size,
                 "y": ((y1+y2)/2-top)/size, "width": (x2-x1)/size, "height": (y2-y1)/size}]
    if not _finite(points): raise ValueError("INVALID_GEOMETRY")
    try:
        from shapely.geometry import Polygon, box
    except ModuleNotFoundError as exc:
        raise RuntimeError("Shapely is required for polygon context clipping") from exc
    poly = Polygon(points)
    if not poly.is_valid or poly.area <= 0: raise ValueError("INVALID_GEOMETRY")
    result = poly.intersection(box(left, top, right, bottom))
    output = []
    for geometry in _polygon_components(result):
        if geometry.is_empty or geometry.area <= 0: continue
        coords = list(geometry.exterior.coords)[:-1]
        if len(coords) < 3: continue
        output.append({"class_id": item["class_id"], "type": "polygon",
                       "points": [value for point in coords for value in ((point[0]-left)/size, (point[1]-top)/size)]})
    return output
