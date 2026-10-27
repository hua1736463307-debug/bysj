"""CXR 受控子进程封装。

GPU 训练/校准/解释都在独立进程组里执行：取消时先写 stop.flag，再 SIGTERM，
超时后 SIGKILL 并 reap；父进程死亡时子进程通过 --parent-pid 自检退出。
只接受 workdir 内的结果文件，不把子进程返回的任意路径当产物。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from app.core.errors import ApiError
from app.core.storage import ensure_dir
from app.workers.dispatcher import JobCancelled


def _signal_child(process: subprocess.Popen, *, hard: bool) -> None:
    if os.name == "posix":
        import signal

        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL if hard else signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif hard:
        process.kill()
    else:
        process.terminate()


def run_cxr_subprocess(
    request: dict,
    workdir: Path,
    *,
    should_stop=None,
    progress=None,
    poll_seconds: float = 2.0,
    grace_seconds: int = 30,
) -> dict:
    workdir = ensure_dir(Path(workdir))
    request_file = workdir / "cxr-request.json"
    output_file = workdir / "cxr-result.json"
    stop_file = workdir / "stop.flag"
    stop_file.unlink(missing_ok=True)
    request_file.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")

    import app as app_package

    backend_dir = str(Path(app_package.__file__).resolve().parents[1])
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = backend_dir + os.pathsep + child_env.get("PYTHONPATH", "")
    log_path = workdir / "cxr-child.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        process = subprocess.Popen(
            [
                sys.executable, "-m", "app.ml.cxr_cli",
                "--request", str(request_file),
                "--output", str(output_file),
                "--parent-pid", str(os.getpid()),
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(workdir),
            env=child_env,
            start_new_session=os.name == "posix",
        )
        try:
            while process.poll() is None:
                if should_stop and should_stop():
                    stop_file.write_text("cancel", encoding="utf-8")
                    _signal_child(process, hard=False)
                    try:
                        process.wait(timeout=grace_seconds)
                    except subprocess.TimeoutExpired:
                        _signal_child(process, hard=True)
                        process.wait(timeout=10)
                    raise JobCancelled()
                if progress:
                    progress(None, "running", "CXR 子进程执行中")
                time.sleep(poll_seconds)
        finally:
            if process.poll() is None:
                _signal_child(process, hard=True)
            process.wait(timeout=10)

    if not output_file.exists():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise ApiError(500, "cxr_child_no_result", f"子进程未产出结果（退出码 {process.returncode}）: {tail}")
    try:
        payload = json.loads(output_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ApiError(500, "cxr_child_invalid_result", "子进程结果不是合法 JSON") from exc
    if not payload.get("ok"):
        raise ApiError(500, "cxr_child_failed", str(payload.get("error") or "子进程执行失败")[:1000])
    return payload


def ensure_inside(path: Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise ApiError(500, "cxr_child_path_escape", "子进程返回路径越界") from exc
    return resolved
