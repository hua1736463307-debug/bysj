"""Validated experiment parameters shared by API, preflight and subprocess."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Augmentation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hsv_h: float = Field(default=.015, ge=0, le=1)
    hsv_s: float = Field(default=.7, ge=0, le=1)
    hsv_v: float = Field(default=.4, ge=0, le=1)
    degrees: float = Field(default=0, ge=0, le=180)
    translate: float = Field(default=.1, ge=0, le=1)
    scale: float = Field(default=.5, ge=0, le=1)
    fliplr: float = Field(default=.5, ge=0, le=1)
    flipud: float = Field(default=0, ge=0, le=1)
    mosaic: float = Field(default=1, ge=0, le=1)
    mixup: float = Field(default=0, ge=0, le=1)
    close_mosaic: int = Field(default=10, ge=0, le=1000)


class EarlyStopping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patience: int = Field(default=20, ge=0, le=1000)
    # Verified against Ultralytics 8.4.53 Metric.fitness: [0,0,0,1].
    monitor: Literal["val/mAP50-95"] = "val/mAP50-95"


class TrainingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epochs: int = Field(default=80, ge=1, le=1000)
    batch: int = Field(default=8, ge=1, le=128)
    imgsz: int = Field(default=640, ge=32, le=2048)
    seed: int = Field(default=42, ge=0)
    deterministic: bool = True
    device: str = "0"
    # `auto` changes optimizer according to iteration count; tiled datasets must
    # not silently select a different optimizer in a controlled comparison.
    optimizer: Literal["AdamW", "SGD"] = "AdamW"
    lr0: float = Field(default=.001, gt=0, le=1)
    lrf: float = Field(default=.01, gt=0, le=1)
    momentum: float = Field(default=.937, ge=0, lt=1)
    weight_decay: float = Field(default=.0005, ge=0, le=1)
    warmup_epochs: float = Field(default=3, ge=0, le=100)
    augmentation: Augmentation = Field(default_factory=Augmentation)
    early_stopping: EarlyStopping = Field(default_factory=EarlyStopping)

    @field_validator("imgsz")
    @classmethod
    def multiple_of_32(cls, value):
        if value % 32:
            raise ValueError("imgsz must be a multiple of 32")
        return value

    @field_validator("device")
    @classmethod
    def single_device(cls, value):
        if value not in {"0", "cuda:0", "cpu"}:
            raise ValueError("single-card execution accepts only 0, cuda:0 or cpu")
        return "0" if value == "cuda:0" else value


def ultralytics_arguments(params: dict) -> dict:
    value = TrainingParams.model_validate(params).model_dump()
    augment = value.pop("augmentation")
    stopping = value.pop("early_stopping")
    return {**value, **augment, "patience": stopping["patience"], "amp": False, "workers": 0}
