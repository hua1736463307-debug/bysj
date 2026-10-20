"""Cached GPU admission; CUDA is touched only by a disposable subprocess.

Cache/lock is per container, while a fault generation is shared by containers.
Readers never wait for CUDA or for the probe lock. CPU work does not call this.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

WAIT_MESSAGE = "当前无剩余运算力，GPU 恢复后自动继续"
PROBE_DEADLINE = 8.0
FRESH_SECONDS = 35.0
READY_INTERVAL = 15.0
RETRY_INTERVAL = 3.0
FAULT_COOLDOWN = 10.0
ROOT = Path(os.getenv("GPU_RUNTIME_STATE_DIR", "temp/gpu_runtime"))
_UNREAPED_PROBE = None


def unavailable(reason="probe_pending", error=None):
    return dict(gpu_status="unavailable", gpu_reason=reason, gpu_error=error,
                cuda_available=False, cuda_device_count=0, cuda_version=None,
                gpu_uuid=None, gpus=[], probe_time=None, message=WAIT_MESSAGE)


def _read(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def bounded_probe(command=None, timeout=PROBE_DEADLINE):
    """Kill/reap the entire probe process group on timeout, including nvidia-smi."""
    global _UNREAPED_PROBE
    if _UNREAPED_PROBE is not None:
        try:
            _UNREAPED_PROBE.communicate(timeout=0.01)
        except subprocess.TimeoutExpired:
            # A driver-blocked process may remain unkillable until the GPU
            # recovers. Never pile up more CUDA processes behind it.
            return unavailable("probe_terminating", "Waiting for the previous GPU probe to exit")
        _UNREAPED_PROBE = None
    command = command or [sys.executable, str(Path(__file__).with_name("gpu_runtime.py")), "--raw-probe"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=os.name == "posix")
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            _UNREAPED_PROBE = process
            return unavailable("probe_terminating", "Waiting for the timed-out GPU probe to exit")
        return unavailable("probe_timeout", "GPU probe exceeded its hard deadline")
    if process.returncode:
        return unavailable("probe_failed", (err or out)[-1000:])
    try:
        result = json.loads(out)
        if not isinstance(result, dict) or result.get("gpu_status") not in {"ready", "unavailable", "degraded"}:
            raise ValueError("invalid probe schema")
        return result
    except (ValueError, TypeError):
        return unavailable("probe_failed", "Invalid GPU probe response")


class ProbeMonitor:
    def __init__(self, root=ROOT, probe=bounded_probe, clock=time.time):
        self.root, self.probe, self.clock = Path(root), probe, clock
        self.cache = self.root / (socket.gethostname() + ".json")
        self.fault_path = self.root / "fault.json"
        self._start_lock = threading.Lock()
        self._thread = None
        self._pid = os.getpid()

    def snapshot(self, *, start=True):
        if start:
            self.start()
        fault = _read(self.fault_path)
        cached = _read(self.cache)
        age = self.clock() - float(cached.get("checked_at", 0))
        if cached.get("fault_id") != fault.get("id"):
            result = unavailable("gpu_recovering", fault.get("error"))
        elif age < 0 or age > FRESH_SECONDS:
            result = unavailable("probe_stale" if cached else "probe_pending")
        else:
            result = copy.deepcopy(cached.get("result") or unavailable())
        result["probe_age_seconds"] = round(max(0, age), 2) if cached else None
        result["message"] = "" if result.get("gpu_status") == "ready" else WAIT_MESSAGE
        return result

    def mark_fault(self, error):
        _write(self.fault_path, {"id": uuid.uuid4().hex, "at": self.clock(), "error": str(error)[-1000:]})

    def tick(self):
        """Run by the background thread only; share one probe across processes."""
        self.root.mkdir(parents=True, exist_ok=True)
        with self.cache.with_suffix(".lock").open("a") as lease:
            if os.name == "posix":
                import fcntl
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return
            fault = _read(self.fault_path)
            previous = _read(self.cache)
            same_fault = previous.get("fault_id") == fault.get("id")
            now = self.clock()
            if same_fault and now < float(previous.get("next_at", 0)):
                return
            if fault and now < float(fault.get("at", 0)) + FAULT_COOLDOWN:
                return
            result = self.probe()
            # A fault during the probe invalidates this result.
            if _read(self.fault_path).get("id") != fault.get("id"):
                return
            successes = int(previous.get("successes", 0)) if same_fault else 0
            successes = successes + 1 if result.get("gpu_status") == "ready" else 0
            if result.get("gpu_status") == "ready" and successes < 2:
                result = dict(result, gpu_status="recovering", gpu_reason="gpu_recovering")
            interval = READY_INTERVAL if successes >= 2 else RETRY_INTERVAL
            _write(self.cache, {"checked_at": self.clock(), "next_at": self.clock() + interval,
                                "fault_id": fault.get("id"), "successes": min(successes, 2), "result": result})

    def start(self):
        with self._start_lock:
            if self._pid == os.getpid() and self._thread and self._thread.is_alive():
                return
            self._pid = os.getpid()
            self._thread = threading.Thread(target=self._loop, name="gpu-probe-supervisor", daemon=True)
            self._thread.start()

    def _loop(self):
        while True:
            try:
                self.tick()
            except Exception:
                # Keep service/heartbeat alive. An expired cache fails admission closed.
                pass
            time.sleep(1.0)


MONITOR = ProbeMonitor()


def is_gpu_fault(error):
    value = str(error).lower()
    return any(marker in value for marker in (
        "gpu_runtime_unavailable", "cuda out of memory", "cuda error:",
        "cuda initialization", "cuda driver", "cudnn_status", "cublas_status",
        "failed to initialize nvml", "cuda-capable device", "no cuda gpus are available",
    ))


class GPUUnavailable(RuntimeError):
    pass


def require_gpu():
    state = MONITOR.snapshot()
    if state.get("gpu_status") != "ready":
        raise GPUUnavailable("GPU_RUNTIME_UNAVAILABLE: " + WAIT_MESSAGE)
    return state


def report_gpu_fault(error):
    if is_gpu_fault(error) and not isinstance(error, GPUUnavailable):
        MONITOR.mark_fault(error)
        return True
    return isinstance(error, GPUUnavailable)
