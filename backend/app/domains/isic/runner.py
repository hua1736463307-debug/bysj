"""ISIC 受控子进程封装。

与 CXR 共用 `app.domains.child_runner` 的执行/取消/回收逻辑，
只绑定 ISIC 的子进程入口与错误码前缀。
"""
from __future__ import annotations

from pathlib import Path

from app.domains.child_runner import ensure_inside, run_child

__all__ = ["run_isic_subprocess", "ensure_inside"]


def run_isic_subprocess(
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
        module="app.ml.isic_cli",
        code_prefix="isic_",
        should_stop=should_stop,
        progress=progress,
        progress_message="ISIC 子进程执行中",
        poll_seconds=poll_seconds,
        grace_seconds=grace_seconds,
    )
