#!/usr/bin/env python3
"""CXR 三 seed 对比实验 runner（走新 CXR API，不调用 detect runner）。

设计要点：
- 协议先冻结：配置 + 代码/锁信息 + 数据集/模型 hash 写入协议文件，独占创建；
- 每 seed 分阶段持久化 train / calibrate / test 的 job 与模型/评价 ID；
- 相同幂等键换内容一律 409；请求结果不明先查状态，不再 POST；
- 未运行的阶段标 NOT_RUN，失败保留原因，不折算为 0 分；
- 凭据只从环境变量读取（BYSJ_USER / BYSJ_PASSWORD），不落盘。

用法：
    python scripts/run_cxr_experiments.py --config configs/cxr/pilot.json \
        --base-url http://127.0.0.1:18004 --output artifacts/cxr-run --prepare-only
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

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}
REQUIRED_CONFIG_KEYS = ("name", "dataset_version_id", "base_model_id", "seeds", "arms", "params")


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode())
    os.replace(temporary, path)


def validate_config(config: dict) -> list[str]:
    problems: list[str] = []
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            problems.append(f"缺少配置项: {key}")
    if not config.get("seeds"):
        problems.append("seeds 不能为空")
    if len(set(config.get("seeds") or [])) != len(config.get("seeds") or []):
        problems.append("seeds 不能重复")
    arms = config.get("arms") or []
    if not arms:
        problems.append("arms 不能为空")
    for arm in arms:
        if arm.get("class_weight") not in {"none", "pos_weight"}:
            problems.append(f"arm {arm.get('name')} 的 class_weight 非法")
    if config.get("label_schema") not in (None, "cxr-five-v1"):
        problems.append("label_schema 必须是 cxr-five-v1")
    return problems


class Client:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self.token = ""

    def login(self, username: str, password: str) -> dict:
        response = self.http.post("/api/v1/auth/login", json={"username": username, "password": password})
        response.raise_for_status()
        payload = response.json()["data"]
        self.token = payload["token"]
        self.http.headers["Authorization"] = f"Bearer {self.token}"
        return payload.get("user") or {}

    def request(self, method: str, path: str, *, json_body=None, headers=None):
        return self.http.request(method, path, json=json_body, headers=headers or {})

    def wait_job(self, job_id: str, timeout: float, interval: float = 3.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.request("GET", f"/api/v1/jobs/{job_id}")
            if response.status_code != 200:
                raise RuntimeError(f"查询作业失败 {response.status_code}: {response.text[:200]}")
            job = response.json()["data"]
            if job["status"] in TERMINAL:
                return job
            time.sleep(interval)
        return {"job_id": job_id, "status": "timeout", "message": "等待超时，未判为成功"}


def submit_job(client: Client, path: str, body: dict, key: str) -> dict:
    response = client.request("POST", path, json_body=body, headers={"Idempotency-Key": key})
    if response.status_code == 409:
        raise RuntimeError(f"幂等键冲突（相同键不同内容）: {key}")
    if response.status_code not in (200, 202):
        raise RuntimeError(f"提交失败 {response.status_code}: {response.text[:300]}")
    return response.json()["data"]


def run_seed_arm(client: Client, *, config: dict, arm: dict, seed: int, protocol_hash: str,
                 output: Path, timeout: float) -> dict:
    params = {**config["params"], "seed": seed, "class_weight": arm["class_weight"]}
    stage: dict = {
        "arm": arm["name"],
        "class_weight": arm["class_weight"],
        "seed": seed,
        "stages": {},
    }
    key = f"{protocol_hash[:16]}:{arm['name']}:{seed}"
    try:
        job = submit_job(
            client,
            "/api/v1/cxr/training/jobs",
            {"dataset_version_id": config["dataset_version_id"], "base_model_id": config["base_model_id"],
             "params": params},
            f"{key}:train",
        )
        stage["stages"]["train"] = {"job_id": job.get("job_id"), "status": job.get("status")}
        final = client.wait_job(job["job_id"], timeout)
        stage["stages"]["train"].update({"status": final["status"], "result": final.get("result")})
        if final["status"] != "completed":
            stage["status"] = "TRAIN_FAILED"
            return stage
        model_id = (final.get("result") or {}).get("model_id")
        stage["model_id"] = model_id
        stage["model_sha256"] = (final.get("result") or {}).get("model_sha256")
        stage["backend"] = (final.get("result") or {}).get("backend")
        stage["research_valid"] = bool((final.get("result") or {}).get("research_valid", False))

        job = submit_job(
            client,
            "/api/v1/cxr/evaluations",
            {"model_id": model_id, "dataset_version_id": config["dataset_version_id"],
             "split": "calibration", "action": "calibrate", "purpose": "fit temperature and threshold"},
            f"{key}:calibrate",
        )
        stage["stages"]["calibrate"] = {"job_id": job.get("job_id"), "status": job.get("status")}
        final = client.wait_job(job["job_id"], timeout)
        stage["stages"]["calibrate"].update({"status": final["status"], "result": final.get("result")})
        if final["status"] != "completed":
            stage["status"] = "CALIBRATION_FAILED"
            return stage

        test_config = config.get("test") or {}
        if not test_config.get("enabled"):
            stage["stages"]["test"] = {"status": "NOT_RUN", "reason": "配置未启用 test 评价（需管理员 + 预登记目的）"}
            stage["status"] = "PARTIAL_NO_TEST"
            return stage
        job = submit_job(
            client,
            "/api/v1/cxr/evaluations",
            {"model_id": model_id, "dataset_version_id": config["dataset_version_id"],
             "split": "test", "action": "evaluate", "purpose": test_config.get("purpose") or "frozen protocol"},
            f"{key}:test",
        )
        stage["stages"]["test"] = {"job_id": job.get("job_id"), "status": job.get("status")}
        final = client.wait_job(job["job_id"], timeout)
        stage["stages"]["test"].update({"status": final["status"], "result": final.get("result")})
        stage["status"] = "COMPLETED" if final["status"] == "completed" else "TEST_FAILED"
        return stage
    except Exception as exc:  # 单 seed/arm 失败不影响其他组合
        stage["status"] = "FAILED"
        stage["error"] = f"{type(exc).__name__}: {exc}"
        return stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-url", default=os.environ.get("BYSJ_BASE_URL", "http://127.0.0.1:18004"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    problems = validate_config(config)
    if problems:
        print(json.dumps({"status": "CONFIG_INVALID", "problems": problems}, ensure_ascii=False, indent=2))
        return 2

    protocol = {
        "schema": "cxr-experiment-protocol-v1",
        "name": config["name"],
        "label_schema": config.get("label_schema", "cxr-five-v1"),
        "task": "cxr_multilabel",
        "dataset_version_id": config["dataset_version_id"],
        "expected_manifest_sha256": config.get("expected_manifest_sha256"),
        "base_model_id": config["base_model_id"],
        "expected_base_model_sha256": config.get("expected_base_model_sha256"),
        "seeds": config["seeds"],
        "arms": config["arms"],
        "params": config["params"],
        "test": config.get("test") or {"enabled": False},
        "code_sha": os.environ.get("BYSJ_CODE_SHA", "unknown"),
        "runner": "scripts/run_cxr_experiments.py",
        "notes": config.get("notes"),
    }
    protocol_hash = digest(protocol)
    output = Path(args.output) / protocol_hash[:16]
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("protocol_hash") != protocol_hash:
            print(json.dumps({"status": "PROTOCOL_CONFLICT", "path": str(protocol_path)}, ensure_ascii=False))
            return 2
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(protocol_path, {**protocol, "protocol_hash": protocol_hash, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")})

    if args.prepare_only:
        print(json.dumps({
            "status": "PROTOCOL_FROZEN",
            "protocol_hash": protocol_hash,
            "output": str(output),
            "combinations": len(config["seeds"]) * len(config["arms"]),
            "test_enabled": bool((config.get("test") or {}).get("enabled")),
            "note": "prepare-only 不提交任何作业",
        }, ensure_ascii=False, indent=2))
        return 0

    username = os.environ.get("BYSJ_USER", "")
    password = os.environ.get("BYSJ_PASSWORD", "")
    if not username or not password:
        print(json.dumps({"status": "CREDENTIALS_MISSING", "need": ["BYSJ_USER", "BYSJ_PASSWORD"]}, ensure_ascii=False))
        return 2

    client = Client(args.base_url, timeout=60)
    client.login(username, password)
    results = []
    for arm in config["arms"]:
        for seed in config["seeds"]:
            stage_path = output / f"{arm['name']}-seed{seed}.json"
            if args.resume and stage_path.exists():
                existing = json.loads(stage_path.read_text(encoding="utf-8"))
                if existing.get("status") in {"COMPLETED", "PARTIAL_NO_TEST"}:
                    results.append(existing)
                    continue
            stage = run_seed_arm(
                client, config=config, arm=arm, seed=seed,
                protocol_hash=protocol_hash, output=output, timeout=args.timeout,
            )
            write_json(stage_path, stage)
            results.append(stage)

    summary = {
        "protocol_hash": protocol_hash,
        "status": "COMPLETE" if all(item.get("status") == "COMPLETED" for item in results) else "PARTIAL",
        "research_valid": all(item.get("research_valid") for item in results),
        "results": [
            {
                "arm": item["arm"], "seed": item["seed"], "status": item.get("status"),
                "model_id": item.get("model_id"), "model_sha256": item.get("model_sha256"),
                "backend": item.get("backend"),
                "macro_auroc": (((item.get("stages", {}).get("test") or {}).get("result") or {})
                                .get("metrics") or {}).get("macro_auroc"),
            }
            for item in results
        ],
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
