"""通用几何变换（无业务耦合，可单元测试）。

约定：
- 训练/标注内部统一使用原图归一化中心点 BBox：``(cx, cy, w, h) ∈ [0, 1]``。
- 推理输出为原图像素 ``xyxy``。
- 切片坐标只在实验路径使用，必须经 ``tiling`` 映射回原图后再评估。
"""
from __future__ import annotations

from dataclasses import dataclass


class GeometryError(ValueError):
    """几何参数非法（用于 422 映射）。"""


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
