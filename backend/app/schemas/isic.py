"""ISIC 方向参数与指标契约。

与 CXR 的区别（有意为之）：单标签七类、softmax、多类交叉熵；
参数白名单独立，不复用 `CxrTrainingParams`，避免把五标签语义带进来。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ISIC_METRICS_SCHEMA = "isic-metrics-v1"
ISIC_PREPROCESS_SCHEMA = "isic-preprocess-v1"
ISIC_MODEL_TASK = "isic_multiclass"
ISIC_BACKBONE_TASK = "isic_backbone"

#: 与 `app.directions.isic.ISIC_LABELS` 一致的固定顺序，落盘后不得重排。
ISIC_CLASS_KEYS: tuple[str, ...] = ("mel", "nv", "bcc", "akiec", "bkl", "df", "vasc")
ISIC_CLASS_COUNT = len(ISIC_CLASS_KEYS)


class IsicTrainingParams(BaseModel):
    """皮肤镜训练参数白名单；不沿用 detect 的 imgsz/mosaic 语义。"""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=20, ge=1, le=50)
    batch: int = Field(default=16, ge=1, le=128)
    image_size: int = Field(default=224, ge=128, le=512)
    lr: float = Field(default=3e-4, gt=0, le=1e-1)
    weight_decay: float = Field(default=1e-4, ge=0, le=1e-1)
    patience: int = Field(default=5, ge=1, le=20)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    #: balanced = 按 train 类频率的倒数加权；none = 不加权。
    class_weight: Literal["none", "balanced"] = "none"
    num_workers: int = Field(default=0, ge=0, le=8)
    amp: bool = False
    device: str = "0"


class IsicEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    dataset_version_id: str
    split: Literal["val", "calibration", "test"] = "val"
    #: test 只允许管理员执行，且必须登记目的；避免把留出集当调参集反复读。
    purpose: str | None = Field(default=None, max_length=1000)
