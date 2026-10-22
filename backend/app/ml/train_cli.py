"""真实 Ultralytics 训练入口（子进程）。

worker 通过 ``python -m app.ml.train_cli --request ... --output ...`` 调用；
失败时写错误结果并返回非 0，不伪造指标。
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import traceback
from pathlib import Path


def run(request: dict) -> dict:
    from ultralytics import YOLO
    import time
    import torch
    from app.schemas.training import ultralytics_arguments
    from app.ml.yolo_adapter import environment_snapshot

    params = request["params"]
    base_path = request.get("base_model_path")
    if not base_path:
        raise RuntimeError("真实训练必须提供已登记的基础权重")
    model = YOLO(str(base_path), task="detect")
    training_args = ultralytics_arguments(params)
    cuda = training_args["device"] != "cpu" and torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    results = model.train(
        data=request["dataset_yaml"],
        **training_args,
        project=request["workdir"],
        name="train",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )
    trainer = getattr(model, "trainer", None)
    if cuda:
        torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(0) if cuda else None
    peak_reserved = torch.cuda.max_memory_reserved(0) if cuda else None
    best = Path(getattr(trainer, "best", "") or "")
    if not best.exists():
        raise RuntimeError("训练完成但没有 best 权重")
    target = Path(request["workdir"]) / "best.pt"
    shutil.copy2(best, target)

    # Standalone val otherwise restores framework defaults (8 loader workers),
    # which can exhaust Docker's shared memory on the full dataset. Treat failed
    # validation as a failed attempt; never register a success with error metrics.
    validation_started = time.perf_counter()
    val = model.val(data=request["dataset_yaml"], split="val", imgsz=params["imgsz"],
                    batch=params["batch"], workers=0, device=training_args["device"], verbose=False,
                    project=request["workdir"], name="validation", exist_ok=True)
    box = getattr(val, "box", None)
    if box is None:
        raise RuntimeError("validation did not return detection metrics")
    metrics = {"precision": float(box.mp), "recall": float(box.mr),
               "map50": float(box.map50), "map50_95": float(box.map)}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("validation returned nonfinite metrics")

    return {
        "best_path": str(target),
        "metrics": metrics,
        "epochs_completed": int(getattr(trainer, "epoch", 0)) + 1,
        "backend": "ultralytics",
        "ultralytics_version": getattr(__import__("ultralytics"), "__version__", None),
        "results_dict": getattr(results, "results_dict", None),
        "effective_args": json.loads(json.dumps(vars(trainer.args) if trainer is not None else training_args, default=str)),
        "training_seconds": elapsed,
        "validation_seconds": time.perf_counter() - validation_started,
        "peak_gpu_allocated_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": peak_reserved,
        "environment": environment_snapshot(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ultralytics 训练子进程入口")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()

    if args.parent_pid and sys.platform == "linux":
        # Install before importing torch. If the worker is killed, the training
        # process also dies; the DB GPU lease remains quarantined until recovery.
        import ctypes
        import os
        import signal
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise RuntimeError("cannot install training parent-death signal")
        if os.getppid() != args.parent_pid:
            return 1

    request_path = Path(args.request)
    output_path = Path(args.output)
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = run(request)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        payload = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]}
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(payload["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
