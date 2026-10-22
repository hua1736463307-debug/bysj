#!/usr/bin/env python3
"""生成构建元数据 version.json（提交 SHA、构建时间、依赖锁 hash）。"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
    except Exception:
        return "unknown"


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="写入 deploy/version.json")
    parser.add_argument("--output", default=str(REPO_ROOT / "deploy" / "version.json"))
    args = parser.parse_args()
    payload = {
        "code_sha": git_sha(),
        "build_time": datetime.now(timezone.utc).astimezone().isoformat(),
        "requirements_lock_sha256": file_hash(REPO_ROOT / "backend" / "requirements.lock.txt"),
        "requirements_ml_lock_sha256": file_hash(REPO_ROOT / "backend" / "requirements-ml.lock.txt"),
        "schema_version": 1,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "WROTE", "path": str(target), "code_sha": payload["code_sha"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
