"""CXR 受控子进程封装。

执行/取消/回收逻辑在 `app.domains.child_runner`（方向无关），本模块只绑定
CXR 的子进程入口与错误码前缀，保持既有调用方与错误码不变。
"""
from __future__ import annotations

from pathlib import Path

from app.domains.child_runner import ensure_inside, run_child

__all__ = ["run_cxr_subprocess", "ensure_inside"]


def run_cxr_subprocess(
    request: dict,
    workdir: Path,
    *,
    should_stop=None,
    progress=None,
    poll_seconds: float = 2.0,
    grace_seconds: int = 30,
) -> dict:
    return run_child(
        request,
        workdir,
        module="app.ml.cxr_cli",
        code_prefix="cxr_",
        should_stop=should_stop,
        progress=progress,
        progress_message="CXR 子进程执行中",
        poll_seconds=poll_seconds,
        grace_seconds=grace_seconds,
    )
