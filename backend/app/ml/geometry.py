"""通用几何变换（无业务耦合，可单元测试）。

约定：
- 训练/标注内部统一使用原图归一化中心点 BBox：``(cx, cy, w, h) ∈ [0, 1]``。
- 推理输出为原图像素 ``xyxy``。
- 切片坐标只在实验路径使用，必须经 ``tiling`` 映射回原图后再评估。
"""
from __future__ import annotations

from dataclasses import dataclass
import math


class GeometryError(ValueError):
    """几何参数非法（用于 422 映射）。"""


def normalize_polygon(points):
    if not isinstance(points, (list, tuple)) or not 3 <= len(points) <= 256:
        raise GeometryError("多边形需要 3 到 256 个顶点")
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise GeometryError("顶点须为两个归一化坐标")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
            raise GeometryError("多边形坐标须为 0 到 1 的有限数值")
        normalized.append([round(x, 6), round(y, 6)])
    area = abs(sum(normalized[index][0] * normalized[(index + 1) % len(normalized)][1] -
                   normalized[(index + 1) % len(normalized)][0] * normalized[index][1]
                   for index in range(len(normalized)))) / 2
    if area < 1e-8:
        raise GeometryError("多边形面积必须大于零")
    xs, ys = zip(*normalized)
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    return normalized, {"cx": (left + right) / 2, "cy": (top + bottom) / 2,
                        "w": right - left, "h": bottom - top}


def clip_polygon(points, bounds):
    """Clip normalized vertices against an axis-aligned rectangle."""
    result = [list(point) for point in points]
    left, top, right, bottom = bounds
    for axis, boundary, keep_greater in ((0, left, True), (0, right, False), (1, top, True), (1, bottom, False)):
        source, result = result, []
        if not source:
            break
        previous = source[-1]
        previous_inside = previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
        for current in source:
            inside = current[axis] >= boundary if keep_greater else current[axis] <= boundary
            if inside != previous_inside:
                ratio = (boundary - previous[axis]) / (current[axis] - previous[axis])
                result.append([previous[index] + ratio * (current[index] - previous[index]) for index in (0, 1)])
            if inside:
                result.append(current)
            previous, previous_inside = current, inside
    return result


def _positive(value: float, name: str) -> float:
    number = float(value)
    if number <= 0 or number != number:  # NaN
        raise GeometryError(f"{name} 必须为正数")
    return number


def xyxy_to_cxcywh_norm(box: list[float], width: int, height: int) -> list[float]:
    w = _positive(width, "width")
    h = _positive(height, "height")
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 <= x1 or y2 <= y1:
        raise GeometryError("bbox 面积必须大于 0")
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return [cx, cy, bw, bh]


def cxcywh_norm_to_xyxy(box: list[float], width: int, height: int) -> list[float]:
    w = _positive(width, "width")
    h = _positive(height, "height")
    cx, cy, bw, bh = (float(v) for v in box)
    if bw <= 0 or bh <= 0:
        raise GeometryError("bbox 宽高必须大于 0")
    return [
        (cx - bw / 2) * w,
        (cy - bh / 2) * h,
        (cx + bw / 2) * w,
        (cy + bh / 2) * h,
    ]


def clamp_xyxy(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return [max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)]


def is_inside(box: list[float], width: int, height: int, tolerance: float = 1e-6) -> bool:
    x1, y1, x2, y2 = (float(v) for v in box)
    return (
        x1 >= -tolerance
        and y1 >= -tolerance
        and x2 <= float(width) + tolerance
        and y2 <= float(height) + tolerance
    )


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = box_area([inter_x1, inter_y1, inter_x2, inter_y2])
    union = box_area(a) + box_area(b) - inter
    return 0.0 if union <= 0 else inter / union


def scale_xyxy(box: list[float], scale_x: float, scale_y: float) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


@dataclass(frozen=True)
class Tile:
    index: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1
