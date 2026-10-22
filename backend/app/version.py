"""构建元数据（commit / build time / lock hash）。

优先读取仓库根 ``version.json``（PR-00B 由构建流程生成）；缺失时退回
环境变量或 ``unknown``，不伪造 release 信息。
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _lock_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def build_info() -> dict:
    payload: dict = {
        "code_sha": os.environ.get("BYSJ_CODE_SHA", "unknown"),
        "build_time": os.environ.get("BYSJ_BUILD_TIME", "unknown"),
        "requirements_lock_sha256": _lock_hash(REPO_ROOT / "backend" / "requirements-core-linux-py312.lock.txt"),
        "ml_lock_sha256": _lock_hash(REPO_ROOT / "backend" / "requirements-ml-linux-py312-cu124.lock.txt"),
    }
    candidate = REPO_ROOT / "version.json"
    if candidate.exists():
        try:
            payload.update(json.loads(candidate.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            payload["version_file"] = "invalid"
    # Runtime source identity overrides metadata inherited from a stale image.
    if os.environ.get("BYSJ_CODE_SHA"):
        payload["code_sha"] = os.environ["BYSJ_CODE_SHA"]
    # Dependency identity belongs to the mounted/built source, not stale JSON.
    payload["requirements_lock_sha256"] = _lock_hash(REPO_ROOT / "backend" / "requirements-core-linux-py312.lock.txt")
    payload["ml_lock_sha256"] = _lock_hash(REPO_ROOT / "backend" / "requirements-ml-linux-py312-cu124.lock.txt")
    return payload
