#!/usr/bin/env python3
"""推理性能测量（E4）：冷/热启动、p50/p95、吞吐、失败率与硬件信息。

用法：
    python scripts/benchmark_medical.py --config configs/experiments/performance.json \
        --output artifacts/performance

账号从环境变量读取（BYSJ_USER / BYSJ_PASSWORD）。缺采样维度写 null 并注明，
不填 0 假装测量完成。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

TERMINAL = {"completed", "failed", "cancelled"}


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(ratio * len(ordered)) - 1))
    return ordered[index]


def collect_hardware() -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from environment_probe import collect

    return collect()


def wait_run(client: httpx.Client, api: str, token: str, run_id: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"{api}/inference/runs/{run_id}", headers={"Authorization": f"Bearer {token}"}).json()["data"]
        if run["status"] in {"completed", "partial", "failed", "cancelled"}:
            return run
        time.sleep(1)
    raise TimeoutError(run_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="推理性能测量")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not config.get("model_id") or not config.get("input_manifest"):
        print(json.dumps({"ok": False, "error": "配置必须提供 model_id 与 input_manifest"}, ensure_ascii=False))
        return 2

    username, password = os.environ.get("BYSJ_USER", ""), os.environ.get("BYSJ_PASSWORD", "")
    if not username or not password:
        print(json.dumps({"ok": False, "error": "需要 BYSJ_USER / BYSJ_PASSWORD"}, ensure_ascii=False))
        return 2

    manifest = json.loads(Path(config["input_manifest"]).read_text(encoding="utf-8"))
    images_root = Path(config["input_manifest"]).parent / "images"
    paths = [images_root / item["relative_path"] for item in manifest["assets"]][: int(config.get("count", 100))]
    paths = [path for path in paths if path.exists()]
    if not paths:
        print(json.dumps({"ok": False, "error": "没有可测量的图像"}, ensure_ascii=False))
        return 2

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    api = args.base_url.rstrip("/") + "/api/v1"
    client = httpx.Client(timeout=120, trust_env=False)
    token = client.post(f"{api}/auth/login", json={"username": username, "password": password}).json()["data"]["token"]
    headers = {"Authorization": f"Bearer {token}"}

    def run_batch(batch: list[Path]) -> tuple[dict, float]:
        files = [("files", (path.name, path.read_bytes(), "image/jpeg")) for path in batch]
        start = time.perf_counter()
        response = client.post(
            f"{api}/inference/images",
            headers=headers,
            data={"model_id": config["model_id"], "params": json.dumps({"device": config.get("device")})},
            files=files,
        )
        response.raise_for_status()
        run = wait_run(client, api, token, response.json()["data"]["run_id"], float(config.get("timeout_seconds", 120)))
        return run, (time.perf_counter() - start) * 1000

    warmup = paths[: int(config.get("warmup_count", 5))]
    if warmup:
        run_batch(warmup)

    raw = []
    for repeat in range(int(config.get("repeat_count", 1))):
        run, wall_ms = run_batch(paths)
        for item in run["items"]:
            raw.append(
                {
                    "repeat": repeat,
                    "item_id": item["item_id"],
                    "status": item["status"],
                    "total_ms": (item.get("timing") or {}).get("total_ms"),
                    "model_ms": (item.get("timing") or {}).get("model_ms"),
                    "wall_ms": round(wall_ms, 2),
                }
            )

    with (output / "raw-timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["repeat", "item_id", "status", "total_ms", "model_ms", "wall_ms"])
        writer.writeheader()
        writer.writerows(raw)

    totals = [row["total_ms"] for row in raw if isinstance(row["total_ms"], (int, float))]
    failures = [row for row in raw if row["status"] != "completed"]
    total_wall = sum({row["repeat"]: row["wall_ms"] for row in raw}.values())
    summary = {
        "config": config,
        "samples": len(raw),
        "success": len(raw) - len(failures),
        "failure_rate": (len(failures) / len(raw)) if raw else None,
        "latency_ms": {
            "mean": statistics.mean(totals) if totals else None,
            "p50": percentile(totals, 0.5),
            "p95": percentile(totals, 0.95),
            "max": max(totals) if totals else None,
        },
        "throughput_per_second": (len(raw) / (total_wall / 1000)) if total_wall else None,
        "not_measured": [
            "peak_rss_bytes",
            "peak_gpu_memory_bytes",
            "gpu_utilization",
        ],
        "notes": [
            "计时来自服务端 item.timing；GPU 计时需真实后端在 synchronize 前后取样",
            "未采样峰值内存/显存时保留 null，不填 0",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "hardware.json").write_text(
        json.dumps(collect_hardware(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"ok": True, "samples": len(raw), "failure_rate": summary["failure_rate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
