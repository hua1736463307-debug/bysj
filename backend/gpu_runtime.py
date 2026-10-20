"""Unified, dependency-light GPU runtime probing for API and workers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional


PROBE_TIMEOUT_SECONDS = 3.0
GPU_REASONS = {
    "nvidia_smi_missing",
    "nvml_init_failed",
    "cuda_unavailable",
    "cuda_init_failed",
    "gpu_memory_insufficient",
    "ok",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error_reason(output: str) -> str:
    lowered = (output or "").lower()
    if "nvml" in lowered or "unknown error" in lowered or "failed to initialize" in lowered:
        return "nvml_init_failed"
    return "cuda_unavailable"


def _run_smi(args: Iterable[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["nvidia-smi", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT_SECONDS,
    )


def _parse_gpus(stdout: str) -> list[Dict[str, Any]]:
    gpus: list[Dict[str, Any]] = []
    for line in (stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            index = int(float(parts[0]))
            memory_used = float(parts[4])
            memory_total = float(parts[5])
            utilization = float(parts[3])
            temperature = float(parts[6])
        except (TypeError, ValueError):
            continue
        gpus.append(
            {
                "index": index,
                "name": parts[1],
                "uuid": parts[2],
                "utilization": utilization,
                "memory_used": memory_used,
                "memory_total": memory_total,
                "memory_percent": (memory_used / memory_total * 100) if memory_total else 0.0,
                "temperature": temperature,
                "memory_free": max(0.0, memory_total - memory_used),
            }
        )
    return gpus


def _base_payload() -> Dict[str, Any]:
    return {
        "gpu_status": "unavailable",
        "gpu_reason": "cuda_unavailable",
        "gpu_error": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_version": None,
        "gpu_uuid": None,
        "gpus": [],
        "probe_time": _now(),
    }


def probe_gpu_raw(*, min_free_memory_mb: Optional[float] = None) -> Dict[str, Any]:
    """Probe NVML and PyTorch independently, returning stable reason codes."""
    payload = _base_payload()
    if not shutil.which("nvidia-smi"):
        payload["gpu_reason"] = "nvidia_smi_missing"
        payload["gpu_error"] = "nvidia-smi is not available in the container"
        return payload

    try:
        result = _run_smi(
            [
                "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,pci.bus_id",
                "--format=csv,noheader,nounits",
            ]
        )
    except subprocess.TimeoutExpired:
        payload["gpu_reason"] = "nvml_init_failed"
        payload["gpu_error"] = "nvidia-smi probe timed out"
        return payload
    except OSError as exc:
        payload["gpu_reason"] = "nvml_init_failed"
        payload["gpu_error"] = str(exc)
        return payload

    if result.returncode != 0:
        payload["gpu_reason"] = _error_reason(result.stdout + result.stderr)
        payload["gpu_error"] = (result.stderr or result.stdout or "nvidia-smi failed").strip()[-1000:]
        return payload

    gpus = _parse_gpus(result.stdout)
    payload["gpus"] = gpus
    payload["cuda_device_count"] = len(gpus)
    payload["gpu_uuid"] = gpus[0].get("uuid") if gpus else None
    if not gpus:
        payload["gpu_reason"] = "nvml_init_failed"
        payload["gpu_error"] = "nvidia-smi returned no GPU records"
        return payload

    try:
        import torch

        payload["cuda_version"] = getattr(torch.version, "cuda", None)
        payload["cuda_available"] = bool(torch.cuda.is_available())
        torch_count = int(torch.cuda.device_count()) if payload["cuda_available"] else 0
        if torch_count:
            # Verify an actual CUDA operation, not merely enumeration.
            for index in range(torch_count):
                sample = torch.ones(1, device=f"cuda:{index}")
                sample.add_(1)
                torch.cuda.synchronize(index)
                del sample
    except Exception as exc:  # pragma: no cover - exact exception is runtime-specific
        payload["gpu_reason"] = "cuda_init_failed"
        payload["gpu_error"] = str(exc)[-1000:]
        return payload

    if not payload["cuda_available"] or torch_count <= 0:
        payload["gpu_reason"] = "cuda_unavailable"
        payload["gpu_error"] = "PyTorch cannot access CUDA in this container"
        return payload

    required_free = min_free_memory_mb
    if required_free is None:
        required_free = float(os.getenv("GPU_RUNTIME_MIN_FREE_MB", "0") or 0)
    if required_free > 0 and any(gpu["memory_free"] < required_free for gpu in gpus):
        payload["gpu_status"] = "degraded"
        payload["gpu_reason"] = "gpu_memory_insufficient"
        payload["gpu_error"] = f"GPU free memory is below {required_free:.0f} MB"
        return payload

    payload["gpu_status"] = "ready"
    payload["gpu_reason"] = "ok"
    return payload


def probe_gpu(*, min_free_memory_mb: Optional[float] = None) -> Dict[str, Any]:
    from gpu_probe_service import MONITOR
    payload = MONITOR.snapshot()
    if min_free_memory_mb and payload.get("gpu_status") == "ready":
        if any(gpu.get("memory_free", 0) < min_free_memory_mb for gpu in payload.get("gpus", [])):
            payload.update(gpu_status="degraded", gpu_reason="gpu_memory_insufficient")
    return payload


def gpu_runtime_is_ready(probe: Optional[Dict[str, Any]] = None) -> bool:
    return (probe or probe_gpu()).get("gpu_status") == "ready"


def training_device_requires_gpu(device: Any) -> bool:
    return str(device or "auto").strip().lower() != "cpu"


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthcheck", action="store_true")
    parser.add_argument("--raw-probe", action="store_true")
    args = parser.parse_args()
    if args.raw_probe:
        payload = probe_gpu_raw()
    else:
        from gpu_probe_service import MONITOR
        payload = MONITOR.snapshot(start=False)
    print(json.dumps(payload, ensure_ascii=False))
    if args.healthcheck and not gpu_runtime_is_ready(payload):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
