"""CXR 多标签模型适配器（DenseNet121 五输出 + dummy 契约后端）。

约束：
- 五路独立 sigmoid，**禁止** softmax，不要求分数和为 1；
- 官方 ImageNet 骨干登记为 `cxr_backbone`，不得直接用于分析；
- 只接受受控可信 state_dict，加载前校验 hash / 架构 / 标签 schema；
- `dummy` 后端只用于工程契约测试，产物带 backend=dummy，指标必须为 null。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.config import Settings
from app.core.errors import ApiError
from app.ml.cxr_preprocess import IMAGE_SIZE, preprocess, restore_map
from app.schemas.cxr import CXR_LABEL_SCHEMA, LABEL_KEYS, LABEL_COUNT

ARCH_DENSENET121 = "densenet121"
SUPPORTED_ARCHES = {ARCH_DENSENET121}


from app.ml.device import normalize_device as _normalize_device


@dataclass
class CxrModelHandle:
    path: Path
    backend: str
    model: object | None = None
    device: str = "cpu"
    metadata: dict = field(default_factory=dict)


class CxrAdapter(Protocol):
    backend: str

    def load(self, model_path: Path, metadata: dict) -> CxrModelHandle: ...

    def predict(self, handle: CxrModelHandle, image_path: Path) -> dict: ...

    def explain(self, handle: CxrModelHandle, image_path: Path, label_key: str) -> dict: ...

    def release(self, handle: CxrModelHandle) -> None: ...


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """把 Grad-CAM 归一到 [0,1]；退化图显式拒绝而不是输出空白热图。

    Grad-CAM 经 ReLU 后非负，因此按峰值归一化即可保留相对强度。
    若峰值≈0（权重全为非正）或含非有限值，说明该标签没有可用的关注区域；
    契约要求此时热力图为 null + reason，不能返回一张全黑图让人误读。
    """
    array = np.asarray(cam, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ApiError(500, "cxr_explain_failed", "关注区域图含非有限值")
    peak = float(array.max())
    if peak <= 1e-8:
        raise ApiError(
            409,
            "cxr_explain_degenerate",
            "该标签的 Grad-CAM 权重全为非正，未生成有效关注区域；评分与结构化报告不受影响",
        )
    return (array / peak).astype(np.float32)


class DummyCxrAdapter:
    """确定性假评分：只验证契约（五路分数/阈值/持久化），不产生真实精度。"""

    backend = "dummy"

    def load(self, model_path: Path, metadata: dict) -> CxrModelHandle:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {Path(model_path).name}")
        return CxrModelHandle(path=Path(model_path), backend=self.backend, metadata=metadata or {})

    def _scores(self, image_path: Path) -> dict[str, float]:
        import hashlib

        digest = hashlib.sha256(Path(image_path).read_bytes()).digest()
        scores: dict[str, float] = {}
        for index, key in enumerate(LABEL_KEYS):
            raw = int.from_bytes(digest[index * 2:index * 2 + 2], "big") / 65535.0
            scores[key] = round(0.02 + raw * 0.96, 6)
        return scores

    def predict(self, handle: CxrModelHandle, image_path: Path) -> dict:
        array, meta = preprocess(Path(image_path), size=IMAGE_SIZE)
        return {
            "raw_scores": self._scores(image_path),
            "quality": meta.get("quality") or {},
            "preprocess": {key: meta[key] for key in (
                "schema", "image_size", "scale", "resized_width", "resized_height",
                "pad_left", "pad_top", "interpolation", "channel_order",
            )},
            "backend": self.backend,
            "timing": {"preprocess_ms": 0.0, "model_ms": 0.0, "definition": "dummy backend 不计时"},
            "limitations": ["dummy backend 不产生真实评分，仅用于工程契约测试"],
            "tensor_shape": list(array.shape),
        }

    def explain(self, handle: CxrModelHandle, image_path: Path, label_key: str) -> dict:
        _, meta = preprocess(Path(image_path), size=IMAGE_SIZE)
        # 确定性伪热图：中心高斯，仅用于验证回映与产物链路。
        axis = np.linspace(-1.0, 1.0, IMAGE_SIZE, dtype=np.float32)
        grid = np.exp(-(axis[None, :] ** 2 + axis[:, None] ** 2) * 2.0)
        restored = restore_map(grid, meta)
        return {
            "backend": self.backend,
            "label_key": label_key,
            "method": "gradcam",
            "cam": restored,
            "meta": meta,
            "limitations": ["dummy backend 的关注区域没有模型依据"],
        }

    def release(self, handle: CxrModelHandle) -> None:
        return None


class DenseNetCxrAdapter:
    """torchvision DenseNet121 + 五输出分类头（sigmoid）。"""

    backend = "densenet121"

    def __init__(self) -> None:
        self._torch = None

    def _ensure_torch(self):
        if self._torch is None:
            try:
                import torch  # 惰性导入：API 进程不加载 torch
                import torchvision  # noqa: F401
            except Exception as exc:  # pragma: no cover - 依赖缺失路径
                raise ApiError(
                    409, "cxr_backend_unavailable",
                    "未安装 torch/torchvision，CXR 真实推理不可用（请使用 CXR 运行镜像）",
                ) from exc
            self._torch = torch
        return self._torch

    def load(self, model_path: Path, metadata: dict) -> CxrModelHandle:
        torch = self._ensure_torch()
        import torchvision

        metadata = metadata or {}
        label_schema = metadata.get("label_schema") or CXR_LABEL_SCHEMA
        if label_schema != CXR_LABEL_SCHEMA:
            raise ApiError(422, "cxr_schema_mismatch", f"模型标签 schema 不匹配: {label_schema}")
        arch = metadata.get("arch") or ARCH_DENSENET121
        if arch not in SUPPORTED_ARCHES:
            raise ApiError(422, "cxr_arch_unsupported", f"不支持的 CXR 架构: {arch}")
        device = _normalize_device(metadata.get("device"), torch)
        try:
            payload = torch.load(str(model_path), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ApiError(422, "cxr_checkpoint_invalid", "checkpoint 不是受控 state_dict 结构") from exc
        state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
        if state_dict is None:
            raise ApiError(422, "cxr_checkpoint_invalid", "checkpoint 缺少 state_dict")
        model = torchvision.models.densenet121(weights=None, num_classes=LABEL_COUNT)
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise ApiError(422, "cxr_checkpoint_mismatch", "checkpoint 与五输出架构不匹配") from exc
        model.eval()
        model.to(device)
        return CxrModelHandle(
            path=Path(model_path), backend=self.backend, model=model, device=device,
            metadata={**metadata, "arch": arch, "label_schema": label_schema},
        )

    def _tensor(self, handle: CxrModelHandle, image_path: Path):
        torch = self._ensure_torch()
        array, meta = preprocess(Path(image_path), size=int(handle.metadata.get("image_size") or IMAGE_SIZE))
        tensor = torch.from_numpy(array).unsqueeze(0).to(handle.device)
        return tensor, meta

    def predict(self, handle: CxrModelHandle, image_path: Path) -> dict:
        torch = self._ensure_torch()
        import time

        if handle.model is None:
            raise ApiError(409, "cxr_model_not_loaded", "模型未加载")
        tensor, meta = self._tensor(handle, image_path)
        started = time.perf_counter()
        with torch.no_grad():
            logits = handle.model(tensor)
            probabilities = torch.sigmoid(logits)[0]
        model_ms = (time.perf_counter() - started) * 1000
        values = [float(item) for item in probabilities.detach().cpu().tolist()]
        return {
            "raw_scores": {key: round(values[index], 6) for index, key in enumerate(LABEL_KEYS)},
            "logits": {key: round(float(logits[0][index]), 6) for index, key in enumerate(LABEL_KEYS)},
            "quality": meta.get("quality") or {},
            "preprocess": {key: meta[key] for key in (
                "schema", "image_size", "scale", "resized_width", "resized_height",
                "pad_left", "pad_top", "interpolation", "channel_order",
            )},
            "backend": self.backend,
            "timing": {"model_ms": round(model_ms, 3)},
            "limitations": [],
        }

    def explain(self, handle: CxrModelHandle, image_path: Path, label_key: str) -> dict:
        torch = self._ensure_torch()
        if handle.model is None:
            raise ApiError(409, "cxr_model_not_loaded", "模型未加载")
        index = LABEL_KEYS.index(label_key)
        tensor, meta = self._tensor(handle, image_path)
        activations: dict[str, object] = {}
        gradients: dict[str, object] = {}

        # 目标层用最后一个 dense block，而不是 features.norm5：
        # DenseNet.forward 在 norm5 之后执行 `F.relu(features, inplace=True)`，
        # 会对 norm5 的输出原地修改，导致 backward hook 的保存张量被覆盖
        # （实测报 "Output 0 of BackwardHookFunctionBackward is a view and is
        # being modified inplace"）。denseblock4 的输出经 BatchNorm 消费，不原地改。
        target_layer = handle.model.features.denseblock4
        target_layer_name = "features.denseblock4"

        def forward_hook(_module, _inputs, output):
            activations["value"] = output
            output.register_hook(lambda grad: gradients.__setitem__("value", grad))

        forward_handle = target_layer.register_forward_hook(forward_hook)
        try:
            handle.model.zero_grad(set_to_none=True)
            # eval 模式保留参数冻结；仅打开解释所需梯度，不更新参数。
            logits = handle.model(tensor)
            logits[0, index].backward()
        finally:
            forward_handle.remove()
        activation = activations.get("value")
        gradient = gradients.get("value")
        if activation is None or gradient is None:
            raise ApiError(500, "cxr_explain_failed", "未捕获到目标层激活/梯度")
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1))[0]
        cam = cam.detach().cpu().numpy().astype(np.float32)
        cam = normalize_cam(cam)
        # cam 是 7x7 空间，先放大到画布尺寸，再按 preprocess 的可逆映射回原图。
        import torch.nn.functional as functional

        cam_tensor = torch.from_numpy(cam)[None, None]
        cam_tensor = functional.interpolate(
            cam_tensor, size=(int(meta["image_size"]), int(meta["image_size"])),
            mode="bilinear", align_corners=False,
        )
        cam_canvas = cam_tensor[0, 0].numpy().astype(np.float32)
        restored = restore_map(cam_canvas, meta)
        return {
            "backend": self.backend,
            "label_key": label_key,
            "method": "gradcam",
            "target_layer": target_layer_name,
            "cam": restored,
            "meta": meta,
            "limitations": [
                "关注区域只说明模型依据，不是病灶定位真值；无定位标注时不计算 IoU/Dice",
            ],
        }

    def release(self, handle: CxrModelHandle) -> None:
        torch = self._torch
        handle.model = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_cxr_adapter(settings: Settings) -> CxrAdapter:
    backend = (getattr(settings, "cxr_ml_backend", None) or "auto").lower()
    if backend == "dummy":
        return DummyCxrAdapter()
    if backend == "densenet121":
        return DenseNetCxrAdapter()
    try:  # auto：装了 torch 才走真实后端
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        return DenseNetCxrAdapter()
    except Exception:
        return DummyCxrAdapter()


def assert_analyzable_model(model, *, settings: Settings) -> None:
    """任务/格式校验：骨干、detect 模型、未激活模型一律拒绝。"""
    from app.domains.models import service as model_service

    if model is None:
        raise ApiError(404, "not_found", "模型不存在")
    if model.status != "active" or not model.available:
        raise ApiError(409, "model_unavailable", "模型不可用")
    if model.model_task == "cxr_backbone":
        raise ApiError(422, "cxr_backbone_not_analyzable", "ImageNet 骨干不能用于胸片分析，请使用训练后的 cxr_multilabel 模型")
    if model.model_task != "cxr_multilabel":
        raise ApiError(422, "model_task_mismatch", f"该模型任务为 {model.model_task}，不能用于胸片五标签分析")
    metadata = model.metadata_json or {}
    label_schema = metadata.get("label_schema")
    if label_schema and label_schema != CXR_LABEL_SCHEMA:
        raise ApiError(422, "cxr_schema_mismatch", f"模型标签 schema 不匹配: {label_schema}")
    model_service.verify_model_file(model, settings)
