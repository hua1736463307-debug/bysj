#!/usr/bin/env python3
"""按实验配置提交训练并汇总真实指标（PR-07/PR-09）。

用法：
    python scripts/run_experiments.py --config configs/experiments/bccd-baseline.json \
        --output artifacts/experiments

配置中 dataset_version_id / approved_model_id 等为 null 时拒绝运行（退出 2）；
不手填指标，不把失败包装成 0。断点重跑通过固定幂等键复用原任务。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx

TERMINAL = {"completed", "failed", "cancelled"}
REQUIRED = ("dataset_version_id", "split_seed", "epochs", "batch", "imgsz", "seeds")


def config_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def wait_job(client: httpx.Client, api: str, token: str, job_id: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"{api}/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status()
        job = response.json()["data"]
        if job["status"] in TERMINAL:
            return job
        time.sleep(3)
    raise TimeoutError(job_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="提交训练实验并收集真实指标")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=os.environ.get("BYSJ_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED if not config.get(key)]
    if missing:
        print(json.dumps({"ok": False, "error": f"配置缺少必需字段: {missing}"}, ensure_ascii=False))
        return 2

    username = os.environ.get("BYSJ_USER", "")
    password = os.environ.get("BYSJ_PASSWORD", "")
    if not username or not password:
        print(json.dumps({"ok": False, "error": "需要 BYSJ_USER / BYSJ_PASSWORD"}, ensure_ascii=False))
        return 2

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "config.frozen.json"
    frozen.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    digest = config_hash(config)

    api = args.base_url.rstrip("/") + "/api/v1"
    client = httpx.Client(timeout=120, trust_env=False)
    token = client.post(f"{api}/auth/login", json={"username": username, "password": password}).json()["data"]["token"]
    headers = {"Authorization": f"Bearer {token}"}

    runs = []
    failures = []
    for seed in config["seeds"]:
        body = {
            "dataset_version_id": config["dataset_version_id"],
            "model_id": config.get("approved_model_id"),
            "task": "detect",
            "params": {
                "epochs": config["epochs"],
                "batch": config["batch"],
                "imgsz": config["imgsz"],
                "seed": int(seed),
                "deterministic": bool(config.get("deterministic", True)),
                "device": config.get("device", "cuda:0"),
            },
        }
        key = f"exp-{digest[:16]}-seed{seed}"
        response = client.post(f"{api}/training/jobs", headers={**headers, "Idempotency-Key": key}, json=body)
        if response.status_code >= 400:
            failures.append({"seed": seed, "error": response.json().get("message"), "status": response.status_code})
            continue
        job = wait_job(client, api, token, response.json()["data"]["job_id"], args.timeout)
        record = {"seed": seed, "job_id": job["job_id"], "status": job["status"], "result": job.get("result"), "error": job.get("error")}
        (output / f"seed-{seed}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        runs.append(record)
        if job["status"] != "completed":
            failures.append({"seed": seed, "error": job.get("error")})

    metrics: dict[str, list[float]] = {}
    for run in runs:
        for key, value in ((run.get("result") or {}).get("metrics") or {}).items():
            if isinstance(value, (int, float)):
                metrics.setdefault(key, []).append(float(value))

    def stats(values: list[float]) -> dict:
        mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            std = variance ** 0.5
        else:
            std = None
        return {"mean": mean, "std": std, "n": len(values), "values": values}

    summary = {
        "config_sha256": digest,
        "config_path": str(Path(args.config)),
        "seeds": config["seeds"],
        "runs": runs,
        "failures": failures,
        "metrics": {key: stats(values) for key, values in metrics.items()},
        "notes": [
            "指标来自真实训练任务结果；null 表示该后端不产生指标",
            "仅 1 个 seed 时 std 为 null，必须披露样本量限制",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": not failures, "runs": len(runs), "failures": failures, "metrics": list(metrics)}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
