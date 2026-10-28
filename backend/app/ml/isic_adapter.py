"""ISIC 七类单标签适配器：DenseNet121 + 7 路 softmax。

与 CXR 适配器的边界（不可混用）：
- CXR 是 5 路独立 sigmoid，分数**不要求和为 1**，且明确禁止 softmax；
- ISIC 是 7 类互斥，输出 softmax 分布（和为 1），只有 argmax 有意义。
两者共用同一套 DenseNet 骨干与 letterbox 预处理实现，但各自声明独立 schema
（`isic-preprocess-v1`），以便将来任何一方调整时不会静默影响另一方。

checkpoint 必须是 `torch.save` 的 dict，包含 arch / label_schema / class_order /
image_size / state_dict；加载用 `weights_only=True`，不接受任意 pickle。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.core.errors import ApiError
from app.ml.device import normalize_device
from app.schemas.isic import (
    ISIC_CLASS_KEYS,
    ISIC_CLASS_COUNT,
    ISIC_METRICS_SCHEMA,
    ISIC_MODEL_TASK,
    ISIC_PREPROCESS_SCHEMA,
)
from app.directions.isic import ISIC_LABEL_SCHEMA

ARCH_DENSENET121 = "densenet121"
SUPPORTED_ARCHES = {ARCH_DENSENET121}


@dataclass
class IsicModelHandle:
    model: object
    device: str
    arch: str
    image_size: int
    class_order: tuple[str, ...]
    path: Path


class DenseNetIsicAdapter:
    backend = "densenet121"

    def _torch(self):
        try:
            import torch
            import torchvision  # noqa: F401
        except Exception as exc:  # pragma: no cover - 依赖缺失时给出明确原因
            raise ApiError(409, "isic_torch_unavailable", "运行环境未安装 torch/torchvision") from exc
        return torch

    def load(self, path: Path, metadata: dict | None = None) -> IsicModelHandle:
        torch = self._torch()
        import torchvision

        payload = torch.load(str(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ApiError(422, "isic_checkpoint_invalid", "checkpoint 缺少 state_dict")
        arch = payload.get("arch")
        if arch not in SUPPORTED_ARCHES:
            raise ApiError(422, "isic_checkpoint_arch", f"不支持的架构：{arch}")
        if payload.get("label_schema") != ISIC_LABEL_SCHEMA:
            raise ApiError(422, "isic_schema_mismatch", "checkpoint 标签 schema 与当前契约不一致")
        class_order = tuple(payload.get("class_order") or ())
        if class_order != ISIC_CLASS_KEYS:
            raise ApiError(422, "isic_class_order_mismatch", "checkpoint 类别顺序与固定契约不一致")

        model = torchvision.models.densenet121(weights=None, num_classes=ISIC_CLASS_COUNT)
        model.load_state_dict(payload["state_dict"])
        device = normalize_device((metadata or {}).get("device", "0"), torch)
        model.to(device).eval()
        return IsicModelHandle(
            model=model,
            device=device,
            arch=arch,
            image_size=int(payload.get("image_size") or 224),
            class_order=class_order,
            path=Path(path),
        )

    def predict_proba(self, handle: IsicModelHandle, image_path: Path) -> dict:
        torch = self._torch()
        from app.ml.cxr_preprocess import preprocess

        array, meta = preprocess(Path(image_path), size=handle.image_size)
        tensor = torch.from_numpy(array).unsqueeze(0).to(handle.device)
        with torch.no_grad():
            logits = handle.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        values = {key: round(float(probabilities[index]), 6) for index, key in enumerate(handle.class_order)}
        index = int(np.argmax(probabilities))
        return {
            "backend": self.backend,
            "class_order": list(handle.class_order),
            "probabilities": values,
            "predicted_index": index,
            "predicted_key": handle.class_order[index],
            "confidence": round(float(probabilities[index]), 6),
            "probability_sum": round(float(probabilities.sum()), 6),
            "device": handle.device,
            "meta": meta,
            "schema": ISIC_METRICS_SCHEMA,
            "task": ISIC_MODEL_TASK,
            "preprocess_schema": ISIC_PREPROCESS_SCHEMA,
        }

    def release(self, handle: IsicModelHandle) -> None:
        torch = self._torch()
        handle.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DummyIsicAdapter:
    """契约测试用：确定性输出，明确标记不产生研究指标。"""

    backend = "dummy"

    def load(self, path: Path, metadata: dict | None = None) -> IsicModelHandle:
        return IsicModelHandle(
            model=None,
            device="cpu",
            arch=ARCH_DENSENET121,
            image_size=224,
            class_order=ISIC_CLASS_KEYS,
            path=Path(path),
        )

    def predict_proba(self, handle: IsicModelHandle, image_path: Path) -> dict:
        digest = hashlib.sha256(Path(image_path).read_bytes()).digest()
        raw = np.frombuffer(digest[:ISIC_CLASS_COUNT], dtype=np.uint8).astype(np.float64) + 1.0
        probabilities = raw / raw.sum()
        index = int(np.argmax(probabilities))
        return {
            "backend": self.backend,
            "class_order": list(ISIC_CLASS_KEYS),
            "probabilities": {
                key: round(float(probabilities[position]), 6) for position, key in enumerate(ISIC_CLASS_KEYS)
            },
            "predicted_index": index,
            "predicted_key": ISIC_CLASS_KEYS[index],
            "confidence": round(float(probabilities[index]), 6),
            "probability_sum": round(float(probabilities.sum()), 6),
            "device": "cpu",
            "meta": {"image_size": handle.image_size, "synthetic_backend": True},
            "schema": ISIC_METRICS_SCHEMA,
            "task": ISIC_MODEL_TASK,
            "preprocess_schema": ISIC_PREPROCESS_SCHEMA,
            "research_valid": False,
            "reason": "dummy backend 不产生真实指标，仅用于工程契约测试",
        }

    def release(self, handle: IsicModelHandle) -> None:
        return None


def get_isic_adapter(settings: Settings):
    choice = (getattr(settings, "isic_ml_backend", "auto") or "auto").lower()
    if choice == "dummy":
        return DummyIsicAdapter()
    if choice in ("auto", "densenet121"):
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except Exception:
            if choice == "densenet121":
                raise ApiError(409, "isic_torch_unavailable", "配置要求 densenet121，但环境缺少 torch")
            return DummyIsicAdapter()
        return DenseNetIsicAdapter()
    raise ApiError(422, "isic_ml_backend_invalid", f"未知的 ISIC 后端：{choice}")
