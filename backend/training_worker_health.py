"""Shared, dependency-light training worker heartbeat helpers."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from gpu_runtime import probe_gpu


WORKER_HEARTBEAT_PATH = Path(
    os.getenv("TRAINING_WORKER_HEARTBEAT_FILE", "temp/training_worker_heartbeat.json")
)
WORKER_STALE_AFTER_SECONDS = 30.0


def write_worker_heartbeat(
    active_task_id: Optional[str],
    *,
    status: str = "running",
    path: Path = WORKER_HEARTBEAT_PATH,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Atomically publish worker state for the API container."""
    timestamp = float(now if now is not None else time.time())
    payload = {
        "status": status,
        "timestamp": timestamp,
        "pid": os.getpid(),
        "active_task_id": active_task_id,
    }
    try:
        gpu = probe_gpu()
    except Exception as exc:
        gpu = {
            "gpu_status": "unavailable",
            "gpu_reason": "cuda_init_failed",
            "gpu_error": str(exc),
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_version": None,
            "gpu_uuid": None,
            "probe_time": time.time(),
        }
    payload.update(
        {
            "gpu_status": gpu.get("gpu_status", "unavailable"),
            "gpu_reason": gpu.get("gpu_reason", "cuda_unavailable"),
            "gpu_error": gpu.get("gpu_error"),
            "cuda_available": bool(gpu.get("cuda_available", False)),
            "cuda_device_count": int(gpu.get("cuda_device_count", 0) or 0),
            "cuda_version": gpu.get("cuda_version"),
            "gpu_uuid": gpu.get("gpu_uuid"),
            "last_gpu_probe_at": gpu.get("probe_time"),
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    return payload


def read_worker_heartbeat(
    *,
    path: Path = WORKER_HEARTBEAT_PATH,
    stale_after_seconds: float = WORKER_STALE_AFTER_SECONDS,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a stable API-facing worker health snapshot."""
    current_time = float(now if now is not None else time.time())
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        age = max(0.0, current_time - float(payload.get("timestamp", 0.0)))
        available = payload.get("status") == "running" and age <= stale_after_seconds
        return {
            "status": "available" if available else "unavailable",
            "worker_status": "available" if available else "unavailable",
            "heartbeat_age_seconds": round(age, 1),
            "active_task_id": payload.get("active_task_id"),
            "gpu_status": payload.get("gpu_status", "unavailable"),
            "gpu_reason": payload.get("gpu_reason", "cuda_unavailable"),
            "gpu_error": payload.get("gpu_error"),
            "cuda_available": bool(payload.get("cuda_available", False)),
            "cuda_device_count": int(payload.get("cuda_device_count", 0) or 0),
            "cuda_version": payload.get("cuda_version"),
            "gpu_uuid": payload.get("gpu_uuid"),
            "last_gpu_probe_at": payload.get("last_gpu_probe_at"),
        }
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError, OSError):
        return {
            "status": "unavailable",
            "worker_status": "unavailable",
            "heartbeat_age_seconds": None,
            "active_task_id": None,
            "gpu_status": "unavailable",
            "gpu_reason": "cuda_unavailable",
            "gpu_error": "training worker heartbeat is missing or stale",
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_version": None,
            "gpu_uuid": None,
            "last_gpu_probe_at": None,
        }


def training_worker_is_available() -> bool:
    return read_worker_heartbeat()["status"] == "available"
