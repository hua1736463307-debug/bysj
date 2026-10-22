"""检测模型适配器。

只有 ``detect`` 任务；输出原图像素 ``xyxy``。
``dummy`` 仅用于契约测试，产物与指标都显式标注 backend=dummy，不能作为
真实实验证据；真实训练/评估走 ``ultralytics`` 子进程。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.config import Settings


@dataclass
class ModelHandle:
    path: Path
    class_map: dict[str, str]
    backend: str


class DetectionAdapter(Protocol):
    backend: str

    def load(self, model_path: Path, class_map: dict[str, str]) -> ModelHandle: ...

    def predict_image(self, handle: ModelHandle, image_path: Path, params: dict) -> list[dict]: ...

    def evaluate(self, handle: ModelHandle, dataset_yaml: Path, split: str, params: dict) -> dict: ...

    def release(self, handle: ModelHandle) -> None: ...


class DummyAdapter:
    """确定性假检测器：只验证契约（坐标/类别/持久化），不产生真实精度。"""

    backend = "dummy"

    def load(self, model_path: Path, class_map: dict[str, str]) -> ModelHandle:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path.name}")
        return ModelHandle(path=Path(model_path), class_map=class_map, backend=self.backend)

    def predict_image(self, handle: ModelHandle, image_path: Path, params: dict) -> list[dict]:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
        confidence = float(params.get("confidence", 0.5))
        first_class = int(next(iter(handle.class_map.keys()), 0))
        return [
            {
                "class_id": first_class,
                "class_name": handle.class_map.get(str(first_class)),
                "confidence": round(confidence, 4),
                "bbox_xyxy": [
                    round(width * 0.25, 2),
                    round(height * 0.25, 2),
                    round(width * 0.75, 2),
                    round(height * 0.75, 2),
                ],
            }
        ]

    def evaluate(self, handle: ModelHandle, dataset_yaml: Path, split: str, params: dict) -> dict:
        return {
            "metrics": {
                "precision": None,
                "recall": None,
                "map50": None,
                "map50_95": None,
                "per_class": {},
                "confusion_matrix": None,
                "curves": None,
            },
            "reason": "dummy backend 不产生真实指标，仅用于契约测试",
            "split": split,
            "backend": self.backend,
        }

    def release(self, handle: ModelHandle) -> None:
        return None


class UltralyticsAdapter:
    backend = "ultralytics"

    def __init__(self) -> None:
        self._model = None

    def load(self, model_path: Path, class_map: dict[str, str]) -> ModelHandle:
        from ultralytics import YOLO  # 惰性导入：API 进程不加载 torch

        self._model = YOLO(str(model_path), task="detect")
        return ModelHandle(path=Path(model_path), class_map=class_map, backend=self.backend)

    def predict_image(self, handle: ModelHandle, image_path: Path, params: dict) -> list[dict]:
        if self._model is None:
            raise RuntimeError("模型未加载")
        import torch
        import time
        cuda = str(params.get("device", "0")) != "cpu" and torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        results = self._model.predict(
            str(image_path),
            imgsz=int(params.get("imgsz", 640)),
            conf=float(params.get("confidence", 0.25)),
            iou=float(params.get("iou", 0.45)),
            device=params.get("device") or None,
            verbose=False,
        )
        predictions: list[dict] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for index in range(len(boxes)):
                class_id = int(boxes.cls[index].item())
                xyxy = [float(value) for value in boxes.xyxy[index].tolist()]
                predictions.append(
                    {
                        "class_id": class_id,
                        "class_name": handle.class_map.get(str(class_id)),
                        "confidence": round(float(boxes.conf[index].item()), 6),
                        "bbox_xyxy": [round(value, 2) for value in xyxy],
                    }
                )
        if cuda:
            torch.cuda.synchronize(0)
        speeds = [result.speed for result in results if getattr(result, "speed", None)]
        self.last_timing = {
            "predict_wall_ms": (time.perf_counter() - started) * 1000,
            **{f"{key}_ms": sum(float(speed[key]) for speed in speeds) if speeds else None
               for key in ("preprocess", "inference", "postprocess")},
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0) if cuda else None,
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0) if cuda else None,
            "timing_source": "Ultralytics result.speed plus synchronized predict wall time",
        }
        return predictions

    def evaluate(self, handle: ModelHandle, dataset_yaml: Path, split: str, params: dict) -> dict:
        if self._model is None:
            raise RuntimeError("模型未加载")
        metrics = self._model.val(
            data=str(dataset_yaml),
            split=split,
            imgsz=int(params.get("imgsz", 640)),
            batch=int(params.get("batch", 8)),
            workers=0,
            device=params.get("device") or None,
            verbose=False,
            plots=True,
            project=str(params["_output_dir"]),
            name="evaluation",
            exist_ok=True,
        )
        box = getattr(metrics, "box", None)
        per_class = {}
        names = getattr(metrics, "names", {}) or handle.class_map
        if box is not None and getattr(box, "ap_class_index", None) is not None:
            for position, class_index in enumerate(box.ap_class_index):
                per_class[str(int(class_index))] = {
                    "precision": float(box.p[position]) if box.p is not None else None,
                    "recall": float(box.r[position]) if box.r is not None else None,
                    "map50": float(box.ap50[position]) if box.ap50 is not None else None,
                    "map50_95": float(box.ap[position]) if box.ap is not None else None,
                }
        matrix = getattr(getattr(metrics, "confusion_matrix", None), "matrix", None)
        curves = getattr(metrics, "curves_results", None)
        def serializable(value):
            import math
            if hasattr(value, "tolist"):
                return serializable(value.tolist())
            if isinstance(value, (list, tuple)):
                return [serializable(item) for item in value]
            if isinstance(value, dict):
                return {str(k): serializable(v) for k, v in value.items()}
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        return {
            "metrics": {
                "precision": float(box.mp) if box is not None else None,
                "recall": float(box.mr) if box is not None else None,
                "map50": float(box.map50) if box is not None else None,
                "map50_95": float(box.map) if box is not None else None,
                "per_class": per_class,
                "confusion_matrix": serializable(matrix),
                "confusion_matrix_labels": [str(names[k]) for k in sorted(names, key=lambda x:int(x))] + ["background"],
                "confusion_matrix_axes": {"rows": "predicted", "columns": "true"},
                "curves": serializable(curves),
                "missing": {key: "backend did not provide this metric" for key, value in (("confusion_matrix", matrix), ("curves", curves)) if value is None},
            },
            "reason": None,
            "split": split,
            "backend": self.backend,
            "names": names,
            "plots": [str(p) for p in Path(metrics.save_dir).glob("*.png")],
        }

    def release(self, handle: ModelHandle) -> None:
        self._model = None


def get_adapter(settings: Settings) -> DetectionAdapter:
    backend = (settings.ml_backend or "auto").lower()
    if backend == "dummy":
        return DummyAdapter()
    if backend == "ultralytics":
        return UltralyticsAdapter()
    try:  # auto
        import ultralytics  # noqa: F401

        return UltralyticsAdapter()
    except Exception:
        return DummyAdapter()


def compute_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_snapshot() -> dict:
    import platform
    import sys

    snapshot = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": None,
        "cuda": None,
        "gpu": None,
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            snapshot["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    snapshot["code_sha"] = os.environ.get("BYSJ_CODE_SHA", "unknown")
    import importlib.metadata
    for name in ("torchvision", "ultralytics", "numpy"):
        try:
            snapshot[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            snapshot[name] = None
    from app.version import build_info
    snapshot["locks"] = {key: value for key, value in build_info().items() if "lock_sha256" in key}
    return snapshot


def write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")
    return __import__("hashlib").sha256(text.encode("utf-8")).hexdigest()
