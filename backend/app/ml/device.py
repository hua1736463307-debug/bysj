"""把「设备」配置归一化为 torch 能接受的形式。

`device="0"` 这类裸序号在 torch 里不是合法设备字符串（会抛
`Invalid device string`），必须显式转成 `cuda:0`；无 CUDA 时统一回退 cpu。

语义与 CXR 适配器原有实现一致（含 `-1`/`gpu` 别名与越界序号回落 `cuda:0`），
CXR 与 ISIC 共用本模块，避免两套设备解析规则漂移。
"""
from __future__ import annotations


def normalize_device(raw, torch_module) -> str:
    text = str(raw or "cpu").strip().lower()
    if text in {"", "cpu", "-1"}:
        return "cpu"
    if text in {"cuda", "gpu"}:
        text = "cuda:0"
    elif text.isdigit():
        text = f"cuda:{text}"
    if not text.startswith("cuda"):
        return "cpu"
    if not torch_module.cuda.is_available():
        return "cpu"
    index = text.partition(":")[2] or "0"
    if not index.isdigit() or int(index) >= torch_module.cuda.device_count():
        return "cuda:0"
    return f"cuda:{index}"
