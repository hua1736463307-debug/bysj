#!/usr/bin/env python3
"""Target-specific hash locks from pip's actual installation report.

Resolve with --ignore-installed --only-binary=:all: in the target environment.
CUDA artifacts retain their official URL; failures preserve the last good lock.
Install with --require-hashes to verify downloads; --wheelhouse also verifies
every downloaded artifact before writing the lock.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import unquote, urlsplit

TORCH_HOSTS = {"download.pytorch.org", "download-r2.pytorch.org"}
ALLOWED_HOSTS = {"files.pythonhosted.org", *TORCH_HOSTS}


def render_lock(report: dict, *, platform: str, python_version: str,
                wheelhouse: Path | None = None, header: str = "# generated target lock",
                offline: bool = False) -> str:
    if offline and wheelhouse is None:
        raise ValueError("offline lock requires a completely verified wheelhouse")
    env = report.get("environment", {})
    if env.get("sys_platform") != platform or env.get("python_version") != python_version:
        raise ValueError("pip report target does not match requested platform/Python")
    installs = report.get("install")
    if not isinstance(installs, list) or not installs:
        raise ValueError("empty report; resolve with --ignore-installed")
    packages = {}
    for item in installs:
        meta = item.get("metadata", {})
        raw_name, version = meta.get("name", ""), meta.get("version", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", raw_name) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version):
            raise ValueError("invalid package identity")
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        info = item.get("download_info", {})
        url = info.get("url", "")
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.port not in (None, 443)):
            raise ValueError(f"{name}: unapproved artifact source")
        filename = unquote(parsed.path.rsplit("/", 1)[-1])
        if not filename.endswith(".whl") or "/" in filename or "\\" in filename:
            raise ValueError(f"{name}: a binary wheel is required")
        digest = info.get("archive_info", {}).get("hashes", {}).get("sha256", "")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"{name}: missing/invalid sha256")
        digest = digest.lower()
        wheel_platform = filename[:-4].rsplit("-", 1)[-1]
        if (platform == "linux" and ("win" in wheel_platform or "macosx" in wheel_platform)) or (platform == "win32" and ("linux" in wheel_platform or "macosx" in wheel_platform)):
            raise ValueError(f"{name}: wheel platform mismatch")
        if wheelhouse is not None:
            wheel = wheelhouse / filename
            if not wheel.is_file():
                raise ValueError(f"{name}: wheel missing from wheelhouse")
            with wheel.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != digest:
                raise ValueError(f"{name}: wheel sha256 mismatch")
        value = (version, url, digest)
        if name in packages and packages[name] != value:
            raise ValueError(f"{name}: conflicting duplicate package")
        packages[name] = value
    lines = [header, f"# Target: {platform}; Python {python_version}; selected wheels only.",
             "# Install: python -m pip install --require-hashes -r <this-file>"]
    for name, (version, url, digest) in sorted(packages.items()):
        if offline:
            lines.append(f"# Verified source: {url}")
        requirement = f"{name} @ {url}" if not offline and urlsplit(url).hostname in TORCH_HOSTS else f"{name}=={version}"
        lines.append(f"{requirement} --hash=sha256:{digest}")
    return "\n".join(lines) + "\n"


def write_lock(report_path: Path, output: Path, **kwargs) -> None:
    content = render_lock(json.loads(report_path.read_text(encoding="utf-8")), **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", default="linux")
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--offline", action="store_true", help="Require verified wheels; emit exact pins without network URL requirements")
    parser.add_argument("--header", default="# generated target lock")
    args = parser.parse_args()
    try:
        write_lock(args.report, args.output, platform=args.platform, python_version=args.python_version,
                   wheelhouse=args.wheelhouse, header=args.header, offline=args.offline)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"lock generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "target": f"{args.platform}/py{args.python_version}"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
