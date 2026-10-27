"""CXR 受控子进程入口：训练 / 批量评分 / 解释。

只在 worker 子进程中运行（GPU 长任务不得在 API 线程执行）。父进程通过
`--request` 传参、`--output` 收回结构化结果；stdout 重定向到日志文件。
子进程不读取环境凭据，也不接受任意 pickle。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

from app.ml.cxr_metrics import auroc, fit_temperature, macro, select_threshold
from app.ml.cxr_preprocess import preprocess
from app.schemas.cxr import (
    CXR_LABEL_SCHEMA,
    CXR_PREPROCESS_SCHEMA,
    LABEL_COUNT,
    LABEL_KEYS,
    RAW_POSITIVE,
    mask_vector,
    target_vector,
)


def _read_request(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_output(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _parent_watch(parent_pid: int | None, should_stop) -> bool:
    """父进程消失即停止：避免孤儿进程长期占用 GPU。"""
    if should_stop():
        return True
    if parent_pid:
        try:
            os.kill(parent_pid, 0)
        except OSError:
            return True
    return False


def _load_manifest(root: Path, manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = []
    for asset in manifest["assets"]:
        items.append(
            {
                "path": root / "images" / asset["relative_path"],
                "raw_labels": asset["raw_labels"],
                "split": asset["split"],
                "component_key": asset["component_key"],
                "relative_path": asset["relative_path"],
            }
        )
    return items


def _build_model(init_state_dict: Path | None):
    import torch
    import torchvision

    model = torchvision.models.densenet121(weights=None, num_classes=LABEL_COUNT)
    if init_state_dict is not None:
        payload = torch.load(str(init_state_dict), map_location="cpu", weights_only=True)
        state = payload.get("state_dict") if isinstance(payload, dict) else None
        if state is None:
            raise RuntimeError("初始化权重缺少 state_dict")
        # 骨干 1000 类头与五输出头不兼容：只加载特征与中间层，分类头随机初始化。
        filtered = {key: value for key, value in state.items() if not key.startswith("classifier.")}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if unexpected:
            raise RuntimeError(f"初始化权重包含未知参数: {unexpected[:5]}")
        if any(not key.startswith("classifier.") for key in missing):
            raise RuntimeError(f"初始化权重缺少必要参数: {[k for k in missing if not k.startswith('classifier.')][:5]}")
    return model


class _CxrDataset:
    def __init__(self, items: list[dict], image_size: int) -> None:
        self.items = items
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        import torch

        item = self.items[index]
        array, _meta = preprocess(item["path"], size=self.image_size)
        targets = target_vector(item["raw_labels"])
        mask = mask_vector(item["raw_labels"])
        y = np.asarray([0 if value is None else value for value in targets], dtype=np.float32)
        m = np.asarray(mask, dtype=np.float32)
        return torch.from_numpy(array), torch.from_numpy(y), torch.from_numpy(m)


def _masked_bce(logits, targets, mask, pos_weight=None):
    import torch

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    denominator = mask.sum()
    if float(denominator) <= 0:
        return None
    return (loss * mask).sum() / denominator


def _predict_arrays(model, items: list[dict], image_size: int, device: str, batch_size: int) -> dict:
    import torch

    scores: dict[str, list[float]] = {key: [] for key in LABEL_KEYS}
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            tensors = []
            for item in chunk:
                array, _meta = preprocess(item["path"], size=image_size)
                tensors.append(torch.from_numpy(array))
            batch = torch.stack(tensors).to(device)
            probabilities = torch.sigmoid(model(batch)).detach().cpu().numpy()
            for row in range(probabilities.shape[0]):
                for position, key in enumerate(LABEL_KEYS):
                    scores[key].append(float(probabilities[row][position]))
    return scores


def _evaluate_split(model, items: list[dict], image_size: int, device: str, batch_size: int) -> dict:
    if not items:
        return {"metrics": {}, "reason": "empty_split"}
    scores = _predict_arrays(model, items, image_size, device, batch_size)
    truth: dict[str, list[float]] = {key: [] for key in LABEL_KEYS}
    mask: dict[str, list[int]] = {key: [] for key in LABEL_KEYS}
    for item in items:
        targets = target_vector(item["raw_labels"])
        masks = mask_vector(item["raw_labels"])
        for position, key in enumerate(LABEL_KEYS):
            truth[key].append(0.0 if targets[position] is None else float(targets[position]))
            mask[key].append(masks[position])
    per_label: dict[str, dict] = {}
    for key in LABEL_KEYS:
        value, reason = auroc(truth[key], scores[key], mask[key])
        per_label[key] = {"auroc": value, "auroc_reason": reason}
    return {
        "metrics": {"per_label": per_label, "macro_auroc": macro(per_label, "auroc")},
        "scores": scores,
        "truth": truth,
        "mask": mask,
    }


def _train(request: dict) -> dict:
    import torch

    params = request["params"]
    seed = int(params.get("seed", request.get("seed", 42)))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    root = Path(request["dataset_root"])
    items = _load_manifest(root, Path(request["manifest"]))
    train_items = [item for item in items if item["split"] == "train"]
    val_items = [item for item in items if item["split"] == "val"]
    if not train_items:
        raise RuntimeError("train 集为空")
    if not val_items:
        raise RuntimeError("val 集为空")

    image_size = int(params.get("image_size", 224))
    batch_size = int(params.get("batch", 8))
    class_weight = request.get("class_weight") or params.get("class_weight") or "none"

    pos_weight = None
    if class_weight == "pos_weight":
        counts_pos, counts_neg = [], []
        for key in LABEL_KEYS:
            positives = sum(1 for item in train_items if item["raw_labels"].get(key) == RAW_POSITIVE)
            negatives = sum(1 for item in train_items if item["raw_labels"].get(key) == "negative")
            if positives == 0 or negatives == 0:
                raise RuntimeError(f"标签 {key} 在 train 缺少明确正例或负例，pos_weight 不可定义")
            counts_pos.append(positives)
            counts_neg.append(negatives)
        values = [min(max(negatives / positives, 1.0), 20.0) for positives, negatives in zip(counts_pos, counts_neg)]
        pos_weight = torch.tensor(values, dtype=torch.float32)
    elif class_weight != "none":
        raise RuntimeError(f"未知 class_weight: {class_weight}")

    device = "cuda:0" if torch.cuda.is_available() and str(params.get("device", "0")) != "cpu" else "cpu"
    model = _build_model(Path(request["init_state_dict"]) if request.get("init_state_dict") else None)
    model.to(device)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(params.get("lr", 1e-4)), weight_decay=float(params.get("weight_decay", 1e-4))
    )
    use_amp = bool(params.get("amp")) and device != "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    train_dataset = _CxrDataset(train_items, image_size)
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(params.get("num_workers", 0))
    )

    max_epochs = int(params.get("epochs", 20))
    patience = int(params.get("patience", 5))
    workdir = Path(request["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    best_macro = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    valid_updates = 0
    parent_pid = request.get("parent_pid")

    def should_stop() -> bool:
        stop_file = workdir / "stop.flag"
        return stop_file.exists()

    for epoch in range(1, max_epochs + 1):
        if _parent_watch(parent_pid, should_stop):
            raise RuntimeError("训练被取消或父进程退出")
        model.train()
        epoch_losses: list[float] = []
        for images, targets, masks in loader:
            if _parent_watch(parent_pid, should_stop):
                raise RuntimeError("训练被取消或父进程退出")
            images, targets, masks = images.to(device), targets.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = _masked_bce(logits, targets, masks, pos_weight)
            else:
                logits = model(images)
                loss = _masked_bce(logits, targets, masks, pos_weight)
            if loss is None:
                continue  # 全 mask=0 的 batch 跳过优化，不除 0
            if not torch.isfinite(loss):
                raise RuntimeError("损失出现非有限值")
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            valid_updates += 1
        model.eval()
        evaluation = _evaluate_split(model, val_items, image_size, device, batch_size)
        macro_value = evaluation["metrics"]["macro_auroc"]["value"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(np.mean(epoch_losses)), 6) if epoch_losses else None,
                "val_macro_auroc": macro_value,
                "val_per_label": {key: item["auroc"] for key, item in evaluation["metrics"]["per_label"].items()},
            }
        )
        if macro_value is not None and macro_value > best_macro:
            best_macro = float(macro_value)
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "arch": "densenet121",
                    "label_schema": CXR_LABEL_SCHEMA,
                    "preprocess_schema": CXR_PREPROCESS_SCHEMA,
                    "image_size": image_size,
                    "label_order": list(LABEL_KEYS),
                    "class_weight": class_weight,
                    "seed": seed,
                    "epoch": epoch,
                    "val_macro_auroc": best_macro,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                },
                workdir / "best.pt",
            )
        else:
            stale += 1
            if stale >= patience:
                break

    if valid_updates == 0:
        raise RuntimeError("整个训练没有有效更新（mask 全 0）")
    if best_macro < 0:
        raise RuntimeError("所有标签 AUROC 均不可定义，训练门禁失败")

    return {
        "backend": "densenet121",
        "best_path": str(workdir / "best.pt"),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "val_macro_auroc": best_macro,
        "val_per_label": history[best_epoch - 1]["val_per_label"] if best_epoch else {},
        "history": history,
        "effective_args": {
            "image_size": image_size,
            "batch": batch_size,
            "epochs": max_epochs,
            "patience": patience,
            "lr": float(params.get("lr", 1e-4)),
            "weight_decay": float(params.get("weight_decay", 1e-4)),
            "seed": seed,
            "class_weight": class_weight,
            "amp": use_amp,
            "device": device,
            "num_workers": int(params.get("num_workers", 0)),
            "pos_weight": None if pos_weight is None else [round(float(v), 6) for v in pos_weight.tolist()],
        },
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "environment": {
            "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() else None,
        },
    }


def _load_model_for_scoring(request: dict):
    import torch

    payload = torch.load(str(request["model_path"]), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError("checkpoint 缺少 state_dict")
    if payload.get("label_schema") and payload["label_schema"] != CXR_LABEL_SCHEMA:
        raise RuntimeError(f"checkpoint 标签 schema 不匹配: {payload['label_schema']}")
    model = _build_model(None)
    model.load_state_dict(payload["state_dict"], strict=True)
    device = "cuda:0" if torch.cuda.is_available() and str(request.get("device", "0")) != "cpu" else "cpu"
    model.to(device).eval()
    return model, device, int(request.get("image_size") or payload.get("image_size") or 224)


def _score_split(request: dict) -> dict:
    """返回某 split 的原始评分、真值、mask 与 component 分组（供 worker 计算指标）。"""
    split = request["split"]
    root = Path(request["dataset_root"])
    items = [item for item in _load_manifest(root, Path(request["manifest"])) if item["split"] == split]
    if not items:
        return {"backend": "densenet121", "split": split, "samples": 0, "reason": "empty_split"}
    model, device, image_size = _load_model_for_scoring(request)
    scores = _predict_arrays(model, items, image_size, device, int(request.get("batch", 8)))
    truth: dict[str, list[float]] = {key: [] for key in LABEL_KEYS}
    mask: dict[str, list[int]] = {key: [] for key in LABEL_KEYS}
    groups: list[str] = []
    for item in items:
        targets = target_vector(item["raw_labels"])
        masks = mask_vector(item["raw_labels"])
        groups.append(item["component_key"])
        for position, key in enumerate(LABEL_KEYS):
            truth[key].append(0.0 if targets[position] is None else float(targets[position]))
            mask[key].append(masks[position])
    return {
        "backend": "densenet121",
        "split": split,
        "samples": len(items),
        "scores": scores,
        "truth": truth,
        "mask": mask,
        "groups": groups,
        "image_size": image_size,
    }


def _calibrate(request: dict) -> dict:
    payload = _score_split({**request, "split": "calibration"})
    if payload.get("samples", 0) == 0:
        raise RuntimeError("calibration 集为空")
    scores, truth, mask = payload["scores"], payload["truth"], payload["mask"]
    temperatures: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    threshold_sources: dict[str, str] = {}
    per_label: dict[str, dict] = {}
    for key in LABEL_KEYS:
        fit = fit_temperature(truth[key], scores[key], mask[key])
        temperatures[key] = float(fit["temperature"])
        calibrated = [
            float(1.0 / (1.0 + np.exp(-(np.log(min(max(score, 1e-6), 1 - 1e-6) / (1 - min(max(score, 1e-6), 1 - 1e-6))) / temperatures[key])))
            for score in scores[key]
        ]
        selection = select_threshold(truth[key], calibrated, mask[key])
        thresholds[key] = float(selection["threshold"])
        threshold_sources[key] = str(selection["source"])
        per_label[key] = {
            "temperature": temperatures[key],
            "calibration": fit,
            "threshold": thresholds[key],
            "threshold_selection": {k: v for k, v in selection.items() if k != "candidates"},
        }
    return {
        "backend": payload["backend"],
        "split": "calibration",
        "temperatures": temperatures,
        "thresholds": thresholds,
        "threshold_sources": threshold_sources,
        "per_label": per_label,
        "groups": payload["groups"],
        "scores": scores,
        "truth": truth,
        "mask": mask,
        "samples": payload["samples"],
    }


def _predict(request: dict) -> dict:
    import torch

    model, device, image_size = _load_model_for_scoring(request)
    results = []
    for item in request["images"]:
        started = time.perf_counter()
        array, meta = preprocess(Path(item["path"]), size=image_size)
        with torch.no_grad():
            logits = model(torch.from_numpy(array).unsqueeze(0).to(device))
            probabilities = torch.sigmoid(logits)[0].detach().cpu().numpy()
        results.append(
            {
                "item_id": item["item_id"],
                "raw_scores": {key: round(float(probabilities[index]), 6) for index, key in enumerate(LABEL_KEYS)},
                "quality": meta.get("quality") or {},
                "preprocess": {k: meta[k] for k in ("image_size", "scale", "pad_left", "pad_top")},
                "model_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
    return {"backend": "densenet121", "items": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="CXR 受控子进程")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-pid", type=int, default=None)
    args = parser.parse_args()

    request = _read_request(Path(args.request))
    request["parent_pid"] = args.parent_pid or request.get("parent_pid")
    task = request.get("task")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        if task == "train":
            payload = _train(request)
        elif task == "calibrate":
            payload = _calibrate(request)
        elif task == "score":
            payload = _score_split(request)
        elif task == "predict":
            payload = _predict(request)
        else:
            raise RuntimeError(f"未知任务: {task}")
    except Exception as exc:  # 结构化失败，父进程据此判失败
        _write_output(Path(args.output), {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    payload["ok"] = True
    _write_output(Path(args.output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
