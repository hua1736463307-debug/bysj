"""真实 Ultralytics 训练入口（子进程）。

worker 通过 ``python -m app.ml.train_cli --request ... --output ...`` 调用；
失败时写错误结果并返回非 0，不伪造指标。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path


def run(request: dict) -> dict:
    from ultralytics import YOLO

    params = request["params"]
    base_path = request.get("base_model_path")
    if not base_path:
        raise RuntimeError("真实训练必须提供已登记的基础权重")
    model = YOLO(str(base_path), task="detect")
    results = model.train(
        data=request["dataset_yaml"],
        epochs=int(params["epochs"]),
        batch=int(params["batch"]),
        imgsz=int(params["imgsz"]),
        seed=int(params["seed"]),
        deterministic=bool(params.get("deterministic", True)),
        device=params.get("device") or None,
        project=request["workdir"],
        name="train",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )
    trainer = getattr(model, "trainer", None)
    best = Path(getattr(trainer, "best", "") or "")
    if not best.exists():
        raise RuntimeError("训练完成但没有 best 权重")
    target = Path(request["workdir"]) / "best.pt"
    shutil.copy2(best, target)

    metrics = {}
    try:
        val = model.val(data=request["dataset_yaml"], split="val", verbose=False)
        box = getattr(val, "box", None)
        if box is not None:
            metrics = {
                "precision": float(box.mp),
                "recall": float(box.mr),
                "map50": float(box.map50),
                "map50_95": float(box.map),
            }
    except Exception as exc:  # 指标缺失保留 null + 原因
        metrics = {"error": f"val_failed: {type(exc).__name__}: {exc}"}

    return {
        "best_path": str(target),
        "metrics": metrics,
        "epochs_completed": int(getattr(trainer, "epoch", 0)) + 1,
        "backend": "ultralytics",
        "ultralytics_version": getattr(__import__("ultralytics"), "__version__", None),
        "results_dict": getattr(results, "results_dict", None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ultralytics 训练子进程入口")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

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
