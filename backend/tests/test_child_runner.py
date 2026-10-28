"""受控子进程的取消语义：必须走"写 stop.flag → SIGTERM → 宽限"的优雅路径。

`ctx.should_stop()` 在收到取消时是抛 `JobCancelled` 而不是返回 True，
如果不显式接住，异常会跳过优雅终止、子进程被 finally 里的 SIGKILL 直接杀掉。
本测试用一个假进程固定这一行为。
"""
from __future__ import annotations

import json

import pytest

from app.domains import child_runner
from app.workers.dispatcher import JobCancelled


class FakeProcess:
    def __init__(self, *args, **kwargs):
        self.pid = 4242
        self.returncode = None
        self.terminated = False
        self.killed = False
        self._exited = False

    def poll(self):
        return 0 if self._exited else None

    def terminate(self):
        self.terminated = True
        self._exited = True

    def kill(self):
        self.killed = True
        self._exited = True

    def wait(self, timeout=None):
        return 0


def _run(tmp_path, should_stop, monkeypatch):
    created = {}

    def factory(*args, **kwargs):
        process = FakeProcess()
        created["process"] = process
        return process

    monkeypatch.setattr(child_runner.subprocess, "Popen", factory)
    with pytest.raises(JobCancelled):
        child_runner.run_child(
            {"task": "probe"},
            tmp_path,
            module="app.ml.cxr_cli",
            code_prefix="cxr_",
            should_stop=should_stop,
            poll_seconds=0.01,
            grace_seconds=1,
        )
    return created["process"], tmp_path


def test_should_stop_raising_jobcancelled_still_writes_stop_flag(tmp_path, monkeypatch):
    """取消信号以异常形式到达时，仍要先写 stop.flag 并 SIGTERM。"""

    def should_stop():
        raise JobCancelled()

    process, workdir = _run(tmp_path, should_stop, monkeypatch)
    assert (workdir / "stop.flag").read_text(encoding="utf-8") == "cancel"
    assert process.terminated is True
    assert process.killed is False


def test_should_stop_returning_true_uses_same_graceful_path(tmp_path, monkeypatch):
    process, workdir = _run(tmp_path, lambda: True, monkeypatch)
    assert (workdir / "stop.flag").exists()
    assert process.terminated is True
    assert process.killed is False


def test_request_file_is_written_before_spawn(tmp_path, monkeypatch):
    seen = {}

    def factory(*args, **kwargs):
        seen["request_exists"] = (tmp_path / "child-request.json").exists()
        process = FakeProcess()
        process._exited = True
        return process

    monkeypatch.setattr(child_runner.subprocess, "Popen", factory)
    with pytest.raises(Exception):
        child_runner.run_child(
            {"task": "probe", "value": 1},
            tmp_path,
            module="app.ml.cxr_cli",
            code_prefix="cxr_",
            poll_seconds=0.01,
        )
    assert seen["request_exists"] is True
    payload = json.loads((tmp_path / "child-request.json").read_text(encoding="utf-8"))
    assert payload["value"] == 1
