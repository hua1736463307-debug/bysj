#!/usr/bin/env python3
"""由 pip 解析报告生成带哈希的锁文件。

用法：
    pip install --dry-run --ignore-installed --report report.json -r requirements.in
    python scripts/generate_lock.py --report report.json --output requirements.lock.txt \
        --header "# 由 pip 解析结果生成"

脚本只读 PyPI 元数据，不安装依赖；同一版本收集全部发行文件（含 sdist 和各平台
wheel）的 sha256，保证 Linux 容器与本地 Windows 都能校验通过。
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"


def fetch_digests(name: str, version: str, timeout: int = 30, retries: int = 4) -> list[str]:
    url = PYPI_JSON.format(name=name, version=version)
    last_error: Exception | None = None
    payload = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec - 固定 PyPI 域名
                payload = json.loads(response.read().decode("utf-8"))
            break
        except Exception as exc:  # 网络抖动重试
            last_error = exc
            import time

            time.sleep(1.5 * (attempt + 1))
    if payload is None:
        raise last_error or RuntimeError("无法读取 PyPI 元数据")
    digests = []
    for item in payload.get("urls", []):
        digest = item.get("digests", {}).get("sha256")
        if digest:
            digests.append(digest)
    return sorted(set(digests))


def main() -> int:
    parser = argparse.ArgumentParser(description="生成带哈希的 requirements 锁文件")
    parser.add_argument("--report", required=True, help="pip --report 生成的 JSON")
    parser.add_argument("--output", required=True, help="输出锁文件")
    parser.add_argument("--header", default="# generated lock file")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"报告不存在: {report_path}", file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    packages = {}
    for item in report.get("install", []):
        metadata = item.get("metadata", {})
        name = metadata.get("name", "").strip()
        version = metadata.get("version", "").strip()
        if name and version:
            packages[name] = version

    lines = [args.header, "# 安装：python -m pip install --require-hashes -r <本文件>"]
    failures = []
    for name in sorted(packages, key=str.lower):
        version = packages[name]
        try:
            digests = fetch_digests(name, version)
        except Exception as exc:  # 网络/包不存在
            failures.append(f"{name}=={version}: {exc}")
            continue
        if not digests:
            failures.append(f"{name}=={version}: 无 sha256 摘要")
            continue
        hashes = " ".join(f"--hash=sha256:{digest}" for digest in digests)
        lines.append(f"{name}=={version} {hashes}")

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output, "packages": len(packages) - len(failures), "failures": failures}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
