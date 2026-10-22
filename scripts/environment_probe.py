#!/usr/bin/env python3
"""采集执行环境的可复现信息（PR-00A）。

只读取本机状态，不安装依赖、不连接业务数据库、不加载模型权重。
输出 JSON 到 stdout 或 --output 指定的文件；缺失项标注 ``GPU_PENDING`` /
``NOT_AVAILABLE``，不以猜测值填充。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, (proc.stdout or proc.stderr).strip()
    except FileNotFoundError:
        return 127, "NOT_AVAILABLE"
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    except OSError as exc:  # pragma: no cover - 平台相关
        return 1, f"ERROR: {exc}"


def _which(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for suffix in (".cmd", ".exe", ".bat"):
        found = shutil.which(name + suffix)
        if found:
            return found
    return None


def collect_python() -> dict:
    info = {
        "executable": sys.executable,
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    try:
        import importlib.metadata as md

        packages = {}
        for name in (
            "fastapi",
            "pydantic",
            "sqlalchemy",
            "alembic",
            "psycopg",
            "pillow",
            "numpy",
            "torch",
            "ultralytics",
        ):
            try:
                packages[name] = md.version(name)
            except md.PackageNotFoundError:
                packages[name] = None
        info["packages"] = packages
    except Exception as exc:  # pragma: no cover
        info["packages"] = {"error": str(exc)}
    return info


def collect_gpu() -> dict:
    if not _which("nvidia-smi"):
        return {"status": "GPU_PENDING", "reason": "nvidia-smi 不在 PATH"}
    code, out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    if code != 0:
        return {"status": "GPU_PENDING", "reason": out}
    devices = []
    for line in out.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) >= 4:
            devices.append(
                {
                    "name": parts[0],
                    "driver": parts[1],
                    "memory_total": parts[2],
                    "compute_capability": parts[3],
                }
            )
    return {"status": "AVAILABLE" if devices else "GPU_PENDING", "devices": devices}


def collect_docker() -> dict:
    if not _which("docker"):
        return {"status": "NOT_AVAILABLE", "reason": "docker 不在 PATH"}
    _, cli = _run(["docker", "--version"])
    _, compose = _run(["docker", "compose", "version"])
    code, info = _run(["docker", "info", "--format", "{{.ServerVersion}} {{.OperatingSystem}}"])
    daemon_ok = code == 0 and bool(info.strip())
    return {
        "status": "AVAILABLE" if daemon_ok else "CLI_ONLY",
        "cli": cli,
        "compose": compose,
        "server": info if daemon_ok else None,
        "daemon_error": None if daemon_ok else (info or "docker info 退出码非 0"),
    }


def collect_node() -> dict:
    if not _which("node"):
        return {"status": "NOT_AVAILABLE"}
    node = _which("node") or "node"
    npm = _which("npm")
    _, node_version = _run([node, "--version"])
    npm_cmd = [npm, "--version"] if npm else [node, "--version"]
    if npm and os.name == "nt":
        npm_cmd = [os.environ.get("COMSPEC", "cmd.exe"), "/c", npm, "--version"]
    _, npm_version = _run(npm_cmd) if npm else ("", "NOT_AVAILABLE")
    return {
        "status": "AVAILABLE",
        "node": node_version,
        "npm": npm_version,
        "npm_path": npm,
    }


def collect_git() -> dict:
    if not _which("git"):
        return {"status": "NOT_AVAILABLE"}
    _, head = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    _, status = _run(["git", "-C", str(REPO_ROOT), "status", "--short"])
    return {
        "status": "AVAILABLE",
        "commit": head,
        "dirty_entries": [line for line in status.splitlines() if line.strip()],
    }


def collect_host() -> dict:
    total_memory = None
    try:
        import psutil  # type: ignore

        total_memory = psutil.virtual_memory().total
    except Exception:
        total_memory = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "total_memory_bytes": total_memory,
        "cwd": os.getcwd(),
        "repo_root": str(REPO_ROOT),
    }


def collect() -> dict:
    return {
        "schema_version": 1,
        "collected_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "host": collect_host(),
        "python": collect_python(),
        "gpu": collect_gpu(),
        "docker": collect_docker(),
        "node": collect_node(),
        "git": collect_git(),
        "notes": [
            "GPU_PENDING 表示本机未检测到可用 NVIDIA 设备，不代表 GPU 验收通过。",
            "CLI_ONLY 表示 Docker CLI 存在但守护进程不可用，不能据此声明构建通过。",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="采集 PR-00A 环境矩阵；只读，不安装依赖、不连接业务数据库。"
    )
    parser.add_argument("--output", help="写入 JSON 文件（默认打印到 stdout）")
    args = parser.parse_args()

    report = collect()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        target = Path(args.output)
        if not target.parent.exists():
            print(f"输出目录不存在: {target.parent}", file=sys.stderr)
            return 2
        target.write_text(text + "\n", encoding="utf-8")
        print(json.dumps({"status": "WROTE", "path": str(target)}, ensure_ascii=False))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
