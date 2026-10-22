#!/usr/bin/env python3
"""真 API 冒烟：登录 → 导入 → 标注 → 冻结 → 训练 → 模型 → 评估 → 推理 → 复核回流。

用法：
    python scripts/smoke_medical.py --base-url http://127.0.0.1:8000 \
        --config configs/experiments/bccd-smoke.json --evidence artifacts/smoke --images data/prepared/bccd/images

账号只从环境变量读取：SMOKE_ADMIN_USER / SMOKE_ADMIN_PASSWORD /
SMOKE_RESEARCH_USER / SMOKE_RESEARCH_PASSWORD。
任何一步未完成则退出码非 0；不把失败包装成“通过”。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


class Smoke:
    def __init__(self, base_url: str, evidence: Path, timeout: float) -> None:
        self.api = base_url.rstrip("/") + "/api/v1"
        self.evidence = evidence
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(timeout=timeout, trust_env=False)
        self.steps: list[dict] = []

    def record(self, step: str, status: str, **extra) -> None:
        item = {"step": step, "status": status, "at": now(), **extra}
        self.steps.append(item)
        (self.evidence / "steps.json").write_text(json.dumps(self.steps, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(item, ensure_ascii=False))

    def login(self, username: str, password: str) -> str:
        response = self.client.post(f"{self.api}/auth/login", json={"username": username, "password": password})
        response.raise_for_status()
        return response.json()["data"]["token"]

    def wait_job(self, token: str, job_id: str, timeout: float = 600) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            response = self.client.get(f"{self.api}/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"})
            response.raise_for_status()
            job = response.json()["data"]
            if job["status"] in TERMINAL:
                return job
            time.sleep(2)
        raise TimeoutError(f"任务 {job_id} 超时")

    def run(self, config: dict, images_dir: Path) -> int:
        admin_user = os.environ.get("SMOKE_ADMIN_USER", "admin")
        admin_password = os.environ.get("SMOKE_ADMIN_PASSWORD", "")
        research_user = os.environ.get("SMOKE_RESEARCH_USER", "smoke-researcher")
        research_password = os.environ.get("SMOKE_RESEARCH_PASSWORD", "")
        if not admin_password or not research_password:
            self.record("credentials", "failed", reason="缺少 SMOKE_ADMIN_PASSWORD / SMOKE_RESEARCH_PASSWORD")
            return 2

        admin_token = self.login(admin_user, admin_password)
        self.record("login_admin", "ok")
        base_model_id = config.get("approved_model_id")
        if config.get("base_relative_path"):
            registered = self.client.post(f"{self.api}/models/register",
                headers={"Authorization": f"Bearer {admin_token}"}, json={
                    "name": "gpu-smoke-base", "relative_path": config["base_relative_path"],
                    "sha256": config["base_weight_sha256"], "class_map": config["class_map"]})
            registered.raise_for_status()
            base_model_id = registered.json()["data"]["model_id"]
            self.record("register_base", "ok", model_id=base_model_id, sha256=config["base_weight_sha256"])
        create = self.client.post(
            f"{self.api}/auth/users",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"username": research_user, "password": research_password, "role": "researcher"},
        )
        self.record("create_researcher", "ok" if create.status_code in (200, 201, 409) else "failed", http_status=create.status_code)
        research_token = self.login(research_user, research_password)
        headers = {"Authorization": f"Bearer {research_token}"}

        image_paths = sorted(
            [path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )[: int(config.get("subset_images", 24))]
        if not image_paths:
            self.record("collect_images", "failed", reason=f"{images_dir} 没有图像")
            return 1
        files = [("files", (path.name, path.read_bytes(), "image/jpeg")) for path in image_paths]
        labels_dir = images_dir.parent / "labels"
        for path in image_paths:
            label = labels_dir / f"{path.stem}.txt"
            if label.exists():
                files.append(("files", (f"{path.stem}.txt", label.read_bytes(), "text/plain")))
        response = self.client.post(
            f"{self.api}/datasets/import",
            headers=headers,
            files=files,
            data={
                "name": "smoke-bccd",
                "source_url": config.get("source_repository", ""),
                "license": "MIT (BCCD_Dataset)",
                "class_map": json.dumps(config["class_map"]),
            },
        )
        response.raise_for_status()
        job = self.wait_job(research_token, response.json()["data"]["job_id"])
        if job["status"] != "completed":
            self.record("import", "failed", job=job["status"], error=job.get("error"))
            return 1
        dataset_id = job["result"]["dataset_id"]
        version_id = job["result"]["version_id"]
        self.record("import", "ok", dataset_id=dataset_id, version_id=version_id)

        assets = self.client.get(
            f"{self.api}/datasets/{dataset_id}/assets", headers=headers, params={"version_id": version_id}
        ).json()["data"]["items"]
        first = assets[0]
        state = self.client.get(f"{self.api}/annotations/{first['asset_id']}", headers=headers).json()["data"]
        save = self.client.put(
            f"{self.api}/annotations/{first['asset_id']}",
            headers=headers,
            json={
                "expected_revision": state["revision"],
                "boxes": [{"client_id": "smoke-1", "class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.3}],
            },
        )
        save.raise_for_status()
        self.record("annotate", "ok", revision=save.json()["data"]["revision"], asset_id=first["asset_id"])

        response = self.client.post(
            f"{self.api}/datasets/{dataset_id}/versions",
            headers=headers,
            json={"base_version_id": version_id, "split_seed": config.get("split_seed", 42)},
        )
        response.raise_for_status()
        freeze = self.wait_job(research_token, response.json()["data"]["job_id"])
        if freeze["status"] != "completed":
            self.record("freeze", "failed", error=freeze.get("error"))
            return 1
        self.record("freeze", "ok", result=freeze["result"])

        train_body = {
            "dataset_version_id": version_id,
            "model_id": base_model_id,
            "params": {
                "epochs": int(config.get("epochs", 1)),
                "batch": int(config.get("batch", 4)),
                "imgsz": int(config.get("imgsz", 320)),
                "seed": int(config.get("split_seed", 42)),
                "deterministic": True,
                "device": "cuda:0",
            },
        }
        preflight = self.client.post(f"{self.api}/training/preflight", headers=headers, json=train_body).json()["data"]
        self.record("preflight", "ok" if preflight["ok"] else "failed", checks=preflight["checks"])
        if not preflight["ok"]:
            return 1
        response = self.client.post(
            f"{self.api}/training/jobs",
            headers={**headers, "Idempotency-Key": f"smoke-{version_id}"},
            json=train_body,
        )
        response.raise_for_status()
        train = self.wait_job(research_token, response.json()["data"]["job_id"], timeout=3600)
        if train["status"] != "completed":
            self.record("train", "failed", error=train.get("error"))
            return 1
        model_id = train["result"]["model_id"]
        backend = train["result"].get("backend")
        expected = config.get("expected_backend")
        if expected and backend != expected:
            self.record("train", "failed", reason="unexpected backend", backend=backend)
            return 1
        self.record("train", "ok", job_id=train["job_id"], model_id=model_id,
            backend=backend, metrics=train["result"].get("metrics"))

        response = self.client.post(
            f"{self.api}/evaluations",
            headers=headers,
            json={"model_id": model_id, "dataset_version_id": version_id, "split": "val"},
        )
        response.raise_for_status()
        evaluation_job = self.wait_job(research_token, response.json()["data"]["job"]["job_id"])
        evaluation_id = response.json()["data"]["evaluation"]["evaluation_id"]
        evaluation = self.client.get(f"{self.api}/evaluations/{evaluation_id}", headers=headers).json()["data"]
        self.record(
            "evaluate",
            "ok" if evaluation_job["status"] == "completed" else "failed",
            evaluation_id=evaluation_id,
            metrics=evaluation.get("metrics"),
        )

        inference = self.client.post(
            f"{self.api}/inference/images",
            headers=headers,
            data={"model_id": model_id, "asset_ids": str(first["asset_id"])},
        )
        inference.raise_for_status()
        inference_job = self.wait_job(research_token, inference.json()["data"]["job"]["job_id"])
        run_id = inference.json()["data"]["run_id"]
        run = self.client.get(f"{self.api}/inference/runs/{run_id}", headers=headers).json()["data"]
        self.record("inference", "ok" if run["status"] in {"completed", "partial"} else "failed", run=run["status"])
        export = self.client.get(f"{self.api}/inference/runs/{run_id}/export", headers=headers, params={"format": "json"})
        export.raise_for_status()
        (self.evidence / "inference-export.json").write_bytes(export.content)

        item = run["items"][0]
        review = self.client.post(
            f"{self.api}/reviews",
            headers=headers,
            json={
                "inference_item_id": item["item_id"],
                "image_asset_id": item["asset_id"],
                "items": [
                    {
                        "prediction_id": item["predictions"][0]["id"] if item["predictions"] else None,
                        "verdict": "missed_object",
                        "corrected": {"class_id": 1, "cx": 0.4, "cy": 0.4, "w": 0.2, "h": 0.2},
                    }
                ],
            },
        )
        review.raise_for_status()
        review_id = review.json()["data"]["review_id"]
        self.client.post(f"{self.api}/reviews/{review_id}/submit", headers=headers).raise_for_status()
        decision = self.client.post(
            f"{self.api}/reviews/{review_id}/decision",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"approve": True},
        )
        decision.raise_for_status()
        export_version = self.client.post(
            f"{self.api}/reviews/export-version",
            headers=headers,
            json={"parent_version_id": version_id, "review_ids": [review_id]},
        )
        export_version.raise_for_status()
        feedback = self.wait_job(research_token, export_version.json()["data"]["job_id"])
        self.record(
            "review_roundtrip",
            "ok" if feedback["status"] == "completed" else "failed",
            review_id=review_id,
            child_version_id=(feedback.get("result") or {}).get("child_version_id"),
        )

        summary = {
            "generated_at": now(),
            "config": config,
            "dataset_id": dataset_id,
            "version_id": version_id,
            "model_id": model_id,
            "steps": self.steps,
            "ok": all(step["status"] in {"ok", "completed"} for step in self.steps),
        }
        payload = json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8")
        (self.evidence / "summary.json").write_bytes(payload)
        (self.evidence / "summary.sha256").write_text(hashlib.sha256(payload).hexdigest(), encoding="utf-8")
        return 0 if summary["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="医学平台端到端冒烟")
    parser.add_argument("--base-url", required=True, help="不带 /api/v1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--images", required=True, help="预处理后的 images 目录")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    runner = Smoke(args.base_url, Path(args.evidence), args.timeout)
    return runner.run(config, Path(args.images))


if __name__ == "__main__":
    sys.exit(main())
