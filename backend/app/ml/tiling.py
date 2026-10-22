"""通用滑窗切片（医学默认不启用；E2 实验使用）。

切片只改变视图/训练样本，评估必须把预测映射回原图后合并再计分，
不能把 tile 指标与原图指标并表（CONTRACTS C03-8）。
"""
from __future__ import annotations

from app.ml.geometry import GeometryError, Tile, box_area, iou


def plan_tiles(
    width: int,
    height: int,
    tile_size: int,
    overlap: float = 0.2,
    min_tile: int = 64,
) -> list[Tile]:
    if width <= 0 or height <= 0:
        raise GeometryError("图像尺寸必须为正")
    if tile_size <= 0 or tile_size < min_tile:
        raise GeometryError(f"tile_size 不能小于 {min_tile}")
    if not 0 <= overlap < 1:
        raise GeometryError("overlap 必须在 [0, 1)")

    step = max(1, int(round(tile_size * (1 - overlap))))
    xs = list(range(0, max(1, width - tile_size + 1), step)) or [0]
    ys = list(range(0, max(1, height - tile_size + 1), step)) or [0]
    if xs[-1] + tile_size < width:
        xs.append(max(0, width - tile_size))
    if ys[-1] + tile_size < height:
        ys.append(max(0, height - tile_size))

    tiles: list[Tile] = []
    index = 0
    for y in ys:
        for x in xs:
            tiles.append(
                Tile(
                    index=index,
                    x1=x,
                    y1=y,
                    x2=min(width, x + tile_size),
                    y2=min(height, y + tile_size),
                )
            )
            index += 1
    return tiles


def map_box_to_original(box: list[float], tile: Tile) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1 + tile.x1, y1 + tile.y1, x2 + tile.x1, y2 + tile.y1]


def clip_box_to_tile(box: list[float], tile: Tile) -> list[float] | None:
    x1 = max(float(tile.x1), box[0])
    y1 = max(float(tile.y1), box[1])
    x2 = min(float(tile.x2), box[2])
    y2 = min(float(tile.y2), box[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def visibility(box: list[float], tile: Tile) -> float:
    clipped = clip_box_to_tile(box, tile)
    if clipped is None:
        return 0.0
    total = box_area(box)
    return 0.0 if total <= 0 else box_area(clipped) / total


def merge_tile_predictions(
    per_tile: list[tuple[Tile, list[dict]]],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """把各 tile 的预测映射回原图并按类别做贪心 NMS 合并。

    ``per_tile`` 中每个预测需包含 ``class_id`` 与像素 ``bbox_xyxy``。
    """
    merged: list[dict] = []
    for tile, predictions in per_tile:
        for prediction in predictions:
            item = dict(prediction)
            item["bbox_xyxy"] = map_box_to_original(prediction["bbox_xyxy"], tile)
            merged.append(item)

    merged.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    kept: list[dict] = []
    for item in merged:
        duplicate = False
        for existing in kept:
            if existing["class_id"] != item["class_id"]:
                continue
            if iou(existing["bbox_xyxy"], item["bbox_xyxy"]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept
