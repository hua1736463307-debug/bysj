"""CXR 预处理 `cxr-preprocess-v1`（无裁剪，保留完整视野）。

流程（与 torchvision 官方 DenseNet 默认中心裁剪**不同**，是有意设计选择）：
1. 读入 → RGB（灰度图复制三通道，不改变像素值）；
2. 保持长宽比缩放，使短边等于 224（双线性）；
3. 居中填充到 224×224（填充值 0，黑边）；
4. 归一化 ImageNet mean/std，通道顺序 CHW，float32。

缩放与填充参数保存为可逆映射，供 Grad-CAM 回映到原图。本实现只依赖
Pillow + numpy，因此可在没有 torch 的 API/测试进程中被验证；张量转换留给
adapter 内部完成。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.core.errors import ApiError
from app.schemas.cxr import CXR_PREPROCESS_SCHEMA

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FILL_VALUE = 0
SUPPORTED_FORMATS = {"JPEG", "PNG"}
#: 空白图提示阈值：像素标准差低于该值视为可疑空白，仅提示不拒绝。
BLANK_STD_THRESHOLD = 2.0


def _open_rgb(path: Path) -> tuple[Image.Image, dict]:
    try:
        with Image.open(path) as image:
            image.load()
            fmt = (image.format or "").upper()
            if fmt not in SUPPORTED_FORMATS:
                raise ApiError(415, "unsupported_format", f"不支持的图像格式: {fmt or 'unknown'}（首期仅 JPG/PNG）")
            if image.mode in {"I;16", "I;16B", "I", "F"}:
                raise ApiError(415, "unsupported_format", "不支持 16bit 或浮点灰度影像")
            width, height = image.size
            rgb = image.convert("RGB")
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(422, "image_unreadable", "图像无法解码") from exc
    return rgb, {"format": fmt, "original_width": int(width), "original_height": int(height)}


def preprocess(path: Path, *, size: int = IMAGE_SIZE) -> tuple[np.ndarray, dict]:
    """返回 (CHW float32 已归一化数组, meta)。meta 含可逆回映参数。"""
    image, meta = _open_rgb(Path(path))
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ApiError(422, "image_unreadable", "图像尺寸无效")
    scale = float(size) / float(min(width, height))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    # 缩放后不得超出画布；超出的极端长宽比按较小比例回退，仍保持比例。
    if resized_w > size or resized_h > size:
        scale = float(size) / float(max(width, height))
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
    resized = image.resize((resized_w, resized_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (FILL_VALUE, FILL_VALUE, FILL_VALUE))
    pad_left = (size - resized_w) // 2
    pad_top = (size - resized_h) // 2
    canvas.paste(resized, (pad_left, pad_top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    normalized = (array - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    meta.update(
        {
            "schema": CXR_PREPROCESS_SCHEMA,
            "image_size": int(size),
            "scale": round(scale, 8),
            "resized_width": int(resized_w),
            "resized_height": int(resized_h),
            "pad_left": int(pad_left),
            "pad_top": int(pad_top),
            "fill_value": FILL_VALUE,
            "interpolation": "bilinear",
            "channel_order": "RGB",
            "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
            "crop": "none",
            "flip": "none",
        }
    )
    quality = quality_flags(array, meta)
    meta["quality"] = quality
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32), meta


def quality_flags(array: np.ndarray, meta: dict) -> dict:
    """输入质量提示：只提示，不消除模型结果。"""
    gray = array.mean(axis=2)
    std = float(gray.std())
    mean = float(gray.mean())
    flags: list[str] = []
    if std < BLANK_STD_THRESHOLD:
        flags.append("suspected_blank")
    if mean < 0.02:
        flags.append("very_dark")
    if mean > 0.98:
        flags.append("very_bright")
    if meta["original_width"] < 128 or meta["original_height"] < 128:
        flags.append("low_resolution")
    aspect = meta["original_width"] / max(1, meta["original_height"])
    if aspect > 1.2 or aspect < 0.8:
        flags.append("unusual_aspect_ratio")
    return {
        "flags": flags,
        "pixel_mean": round(mean, 6),
        "pixel_std": round(std, 6),
        "definition": "工程提示，不是经临床验证的质量评估；不影响五标签评分是否生成",
    }


def restore_map(cam: np.ndarray, meta: dict) -> np.ndarray:
    """把 224×224 的关注图裁剪并缩回原图尺寸（可逆映射）。"""
    size = int(meta["image_size"])
    pad_left = int(meta["pad_left"])
    pad_top = int(meta["pad_top"])
    resized_w = int(meta["resized_width"])
    resized_h = int(meta["resized_height"])
    cropped = cam[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w]
    if cropped.size == 0:
        raise ApiError(422, "cam_map_empty", "关注区域图为空")
    image = Image.fromarray(np.asarray(cropped, dtype=np.float32), mode="F")
    restored = image.resize((int(meta["original_width"]), int(meta["original_height"])), resample=Image.BILINEAR)
    return np.asarray(restored, dtype=np.float32)
