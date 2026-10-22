from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvaluationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    imgsz: int = Field(default=640, ge=32, le=2048)
    device: Literal["0", "cuda:0", "cpu"] = "0"
    batch: int = Field(default=8, ge=1, le=128)

    @field_validator("imgsz")
    @classmethod
    def multiple(cls, value):
        if value % 32:
            raise ValueError("imgsz must be a multiple of 32")
        return value


class InferenceParams(EvaluationParams):
    confidence: float = Field(default=.25, ge=0, le=1)
    iou: float = Field(default=.45, gt=0, le=1)
