"""ISIC 受控子进程入口：七类单标签训练与评价。

只在 worker 子进程中运行（GPU 长任务不得在 API 线程执行）。父进程通过
`--request` 传参、`--output` 收回结构化结果；stdout 重定向到日志文件。
子进程不读取环境凭据，checkpoint 只接受 `weights_only=True` 的 dict。

取消：父进程写 `stop.flag` 并先 SIGTERM 后 SIGKILL；子进程每个 batch 检查
`stop.flag` 与 `--parent-pid`，避免孤儿进程长期占用 GPU。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from app.directions.isic import ISIC_LABEL_SCHEMA
from app.ml.isic_metrics import evaluate as evaluate_metrics
from app.schemas.isic import (
    ISIC_CLASS_KEYS,
    ISIC_CLASS_COUNT,
    ISIC_METRICS_SCHEMA,
    ISIC_PREPROCESS_SCHEMA,
)


def _write_output(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _load_items(root: Path, manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("label_schema") != ISIC_LABEL_SCHEMA:
        raise RuntimeError("manifest 标签 schema 与当前契约不一致")
    items = []
    for asset in manifest["assets"]:
        items.append(
            {
                "path": root / "images" / asset["image"],
                "class_index": int(asset["class_index"]),
                "split": asset["split"],
                "group_key": asset["group_key"],
            }
        )
    return items


def _build_model(init_state_dict: Path | None):
    import torch
    import torchvision

    model = torchvision.models.densenet121(weights=None, num_classes=ISIC_CLASS_COUNT)
    if init_state_dict is not None:
        payload = torch.load(str(init_state_dict), map_location="cpu", weights_only=True)
        state = payload.get("state_dict") if isinstance(payload, dict) else None
        if state is None:
            raise RuntimeError("初始化权重缺少 state_dict")
        # 1000 类头与 7 类头不兼容：只加载特征层，分类头随机初始化。
        filtered = {key: value for key, value in state.items() if not key.startswith("classifier.")}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if unexpected:
            raise RuntimeError(f"初始化权重含未匹配参数：{sorted(unexpected)[:3]}")
    return model


def _environment() -> dict:
    info = {"python": sys.version.split()[0], "code_sha": os.environ.get("BYSJ_CODE_SHA", "unknown")}
    try:
        import torch
        import torchvision

        info.update(
            torch=torch.__version__,
            torchvision=torchvision.__version__,
            cuda_built=torch.version.cuda,
            cuda_available=bool(torch.cuda.is_available()),
        )
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = type(exc).__name__
    try:
        import PIL

        info["pillow"] = PIL.__version__
    except Exception:  # pragma: no cover
        pass
    info["numpy"] = np.__version__
    return info


class _Dataset:
    def __init__(self, items: list[dict], image_size: int, *, train: bool):
        self.items = items
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        import torch
        from app.ml.cxr_preprocess import preprocess

        item = self.items[index]
        array, _meta = preprocess(item["path"], size=self.image_size)
        tensor = torch.from_numpy(array)
        if self.train and np.random.rand() < 0.5:
            tensor = torch.flip(tensor, dims=[2])  # 皮肤镜左右翻转不改变诊断语义
        return tensor, item["class_index"], index


def _predict(model, items: list[dict], image_size: int, device: str, batch_size: int):
    import torch
    from app.ml.cxr_preprocess import preprocess

    model.eval()
    probabilities = np.zeros((len(items), ISIC_CLASS_COUNT), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            arrays = [preprocess(item["path"], size=image_size)[0] for item in batch]
            tensor = torch.from_numpy(np.stack(arrays)).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probabilities[start : start + len(batch)] = probs
    return probabilities


def _metrics_for(items: list[dict], probabilities: np.ndarray) -> dict:
    truth = [item["class_index"] for item in items]
    pred = probabilities.argmax(axis=1).tolist()
    return evaluate_metrics(truth, pred, probabilities=probabilities)


def _train(request: dict) -> dict:
    import torch

    params = request["params"]
    seed = int(params.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    root = Path(request["dataset_root"])
    items = _load_items(root, Path(request["manifest"]))
    train_items = [item for item in items if item["split"] == "train"]
    val_items = [item for item in items if item["split"] == "val"]
    if not train_items:
        raise RuntimeError("train 集为空")
    if not val_items:
        raise RuntimeError("val 集为空")

    image_size = int(params.get("image_size", 224))
    batch_size = int(params.get("batch", 16))
    max_epochs = int(params.get("epochs", 20))
    patience = int(params.get("patience", 5))
    workdir = Path(request["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    parent_pid = request.get("parent_pid")

    def should_stop() -> bool:
        return (workdir / "stop.flag").exists() or _parent_gone(parent_pid)

    from app.ml.device import normalize_device

    device = normalize_device(params.get("device", "0"), torch)
    model = _build_model(Path(request["init_state_dict"]) if request.get("init_state_dict") else None)
    model.to(device)

    class_weight = request.get("class_weight") or params.get("class_weight") or "none"
    weights = None
    if class_weight == "balanced":
        counts = np.zeros(ISIC_CLASS_COUNT, dtype=np.float64)
        for item in train_items:
            counts[item["class_index"]] += 1
        present = counts > 0
        if not present.all():
            missing = [ISIC_CLASS_KEYS[i] for i in range(ISIC_CLASS_COUNT) if counts[i] == 0]
            raise RuntimeError(f"train 缺少类别 {missing}，balanced 权重不可定义")
        weights = torch.tensor((counts.sum() / (ISIC_CLASS_COUNT * counts)), dtype=torch.float32).to(device)
    elif class_weight != "none":
        raise RuntimeError(f"未知 class_weight: {class_weight}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(params.get("lr", 3e-4)), weight_decay=float(params.get("weight_decay", 1e-4))
    )
    use_amp = bool(params.get("amp")) and device != "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    dataset = _Dataset(train_items, image_size, train=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=int(params.get("num_workers", 0))
    )

    best_score = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    best_path = workdir / "best.pt"
    best_val: dict | None = None

    for epoch in range(1, max_epochs + 1):
        if should_stop():
            raise RuntimeError("训练被取消或父进程退出")
        model.train()
        losses: list[float] = []
        for tensors, targets, _index in loader:
            if should_stop():
                raise RuntimeError("训练被取消或父进程退出")
            tensors = tensors.to(device)
            targets = torch.as_tensor(targets, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = criterion(model(tensors), targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(tensors), targets)
                loss.backward()
                optimizer.step()
            if not torch.isfinite(loss):
                raise RuntimeError("训练损失出现 NaN/Inf")
            losses.append(float(loss.detach().cpu()))

        probabilities = _predict(model, val_items, image_size, device, batch_size)
        metrics = _metrics_for(val_items, probabilities)
        score = metrics["macro_f1"]["value"]
        if score is None:
            score = metrics["accuracy"]["value"] or 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(np.mean(losses)) if losses else float("nan"), 6),
                "val_accuracy": metrics["accuracy"]["value"],
                "val_macro_f1": metrics["macro_f1"]["value"],
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_val = metrics
            torch.save(
                {
                    "arch": "densenet121",
                    "label_schema": ISIC_LABEL_SCHEMA,
                    "class_order": list(ISIC_CLASS_KEYS),
                    "image_size": image_size,
                    "preprocess_schema": ISIC_PREPROCESS_SCHEMA,
                    "metrics_schema": ISIC_METRICS_SCHEMA,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "metadata": {
                        "task": "isic_multiclass",
                        "seed": seed,
                        "class_weight": class_weight,
                        "dataset_manifest_sha256": request.get("dataset_manifest_sha256"),
                        "code_sha": os.environ.get("BYSJ_CODE_SHA", "unknown"),
                        "epoch": epoch,
                    },
                },
                str(best_path),
            )
        else:
            stale += 1
            if stale >= patience:
                break

    environment = _environment()
    if environment.get("cuda_available"):
        environment["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
    return {
        "ok": True,
        "backend": "densenet121",
        "best_path": str(best_path),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "metrics": best_val or {"reason": "没有可用的验证结果"},
        "history": history,
        "effective_args": {
            "epochs": max_epochs,
            "batch": batch_size,
            "image_size": image_size,
            "lr": float(params.get("lr", 3e-4)),
            "weight_decay": float(params.get("weight_decay", 1e-4)),
            "class_weight": class_weight,
            "amp": use_amp,
            "device": device,
            "seed": seed,
            "num_workers": int(params.get("num_workers", 0)),
            "train_images": len(train_items),
            "val_images": len(val_items),
        },
        "environment": environment,
        "label_schema": ISIC_LABEL_SCHEMA,
        "class_order": list(ISIC_CLASS_KEYS),
    }


def _parent_gone(parent_pid: int | None) -> bool:
    if not parent_pid:
        return False
    try:
        os.kill(int(parent_pid), 0)
    except OSError:
        return True
    return False


def _evaluate(request: dict) -> dict:
    from app.ml.isic_adapter import DenseNetIsicAdapter

    root = Path(request["dataset_root"])
    items = _load_items(root, Path(request["manifest"]))
    split = request["split"]
    subset = [item for item in items if item["split"] == split]
    if not subset:
        return {"ok": True, "metrics": {"reason": f"{split} 集为空", "support": 0}, "counts": 0}

    adapter = DenseNetIsicAdapter()
    handle = adapter.load(Path(request["model_path"]), {"device": request.get("device", "0")})
    try:
        probabilities = np.zeros((len(subset), ISIC_CLASS_COUNT), dtype=np.float64)
        for index, item in enumerate(subset):
            if (Path(request["workdir"]) / "stop.flag").exists() or _parent_gone(request.get("parent_pid")):
                raise RuntimeError("评价被取消或父进程退出")
            result = adapter.predict_proba(handle, item["path"])
            probabilities[index] = [result["probabilities"][key] for key in ISIC_CLASS_KEYS]
    finally:
        adapter.release(handle)

    metrics = _metrics_for(subset, probabilities)
    return {
        "ok": True,
        "split": split,
        "counts": len(subset),
        "metrics": metrics,
        "device": handle.device,
        "label_schema": ISIC_LABEL_SCHEMA,
        "class_order": list(ISIC_CLASS_KEYS),
        "environment": _environment(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ISIC 训练/评价受控子进程")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-pid", type=int, default=None)
    args = parser.parse_args()
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        request["parent_pid"] = args.parent_pid
        result = _train(request) if request.get("task") == "train" else _evaluate(request)
    except Exception as exc:  # noqa: BLE001 - 子进程边界必须把异常转成结构化结果
        _write_output(Path(args.output), {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    _write_output(Path(args.output), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
