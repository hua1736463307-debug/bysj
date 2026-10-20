"""Bounded ZIP upload and extraction helpers."""

from __future__ import annotations

import os
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class ArchiveSafetyError(ValueError):
    """Raised when an archive violates a configured safety limit."""


@dataclass(frozen=True)
class ArchiveLimits:
    max_upload_bytes: int = 2 * 1024**3
    max_extracted_bytes: int = 10 * 1024**3
    max_entries: int = 100_000
    max_single_file_bytes: int = 1024**3
    max_compression_ratio: float = 100.0
    disk_safety_margin_bytes: int = 1024**3

    @classmethod
    def from_env(cls) -> "ArchiveLimits":
        return cls(
            max_upload_bytes=int(os.getenv("DATASET_ZIP_MAX_UPLOAD_BYTES", str(cls.max_upload_bytes))),
            max_extracted_bytes=int(os.getenv("DATASET_ZIP_MAX_EXTRACTED_BYTES", str(cls.max_extracted_bytes))),
            max_entries=int(os.getenv("DATASET_ZIP_MAX_ENTRIES", str(cls.max_entries))),
            max_single_file_bytes=int(os.getenv("DATASET_ZIP_MAX_SINGLE_FILE_BYTES", str(cls.max_single_file_bytes))),
            max_compression_ratio=float(os.getenv("DATASET_ZIP_MAX_COMPRESSION_RATIO", str(cls.max_compression_ratio))),
            disk_safety_margin_bytes=int(os.getenv("DATASET_ZIP_DISK_SAFETY_MARGIN_BYTES", str(cls.disk_safety_margin_bytes))),
        )


def save_upload_file_limited(upload_file: Any, target_path: Path, max_bytes: int) -> int:
    """Stream an UploadFile-like object to disk and stop at max_bytes."""
    target_path = Path(target_path)
    written = 0
    try:
        with target_path.open("wb") as output:
            while True:
                chunk = upload_file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise ArchiveSafetyError(f"上传文件超过限制 {max_bytes} bytes")
                output.write(chunk)
    except Exception:
        target_path.unlink(missing_ok=True)
        raise
    return written


def _safe_member_path(info: zipfile.ZipInfo) -> PurePosixPath:
    raw_name = info.filename.replace("\\", "/")
    member = PurePosixPath(raw_name)
    if not raw_name or member.is_absolute() or ".." in member.parts:
        raise ArchiveSafetyError(f"ZIP 包含不安全路径: {info.filename}")
    if member.parts and ":" in member.parts[0]:
        raise ArchiveSafetyError(f"ZIP 包含盘符路径: {info.filename}")
    if stat.S_ISLNK(info.external_attr >> 16):
        raise ArchiveSafetyError(f"ZIP 包含符号链接: {info.filename}")
    return member


def safe_extract_zip(zip_path: Path, destination: Path, limits: ArchiveLimits | None = None) -> int:
    """Validate every ZIP member, then extract within destination with hard limits."""
    limits = limits or ArchiveLimits.from_env()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        entries = archive.infolist()
        if len(entries) > limits.max_entries:
            raise ArchiveSafetyError(f"ZIP 条目数超过限制 {limits.max_entries}")

        total_size = 0
        targets = set()
        validated = []
        for info in entries:
            member = _safe_member_path(info)
            target = (destination / Path(*member.parts)).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise ArchiveSafetyError(f"ZIP 路径越界: {info.filename}") from exc
            target_key = os.path.normcase(str(target))
            if target_key in targets:
                raise ArchiveSafetyError(f"ZIP 包含重复路径: {info.filename}")
            targets.add(target_key)

            if info.file_size > limits.max_single_file_bytes:
                raise ArchiveSafetyError(f"ZIP 单文件超过限制: {info.filename}")
            total_size += info.file_size
            if total_size > limits.max_extracted_bytes:
                raise ArchiveSafetyError("ZIP 解压后总量超过限制")
            if info.file_size:
                ratio = info.file_size / max(info.compress_size, 1)
                if ratio > limits.max_compression_ratio:
                    raise ArchiveSafetyError(f"ZIP 压缩比异常: {info.filename}")
            validated.append((info, target))

        free_bytes = shutil.disk_usage(destination).free
        required_bytes = total_size + limits.disk_safety_margin_bytes
        if free_bytes < required_bytes:
            raise ArchiveSafetyError("磁盘剩余空间不足，无法安全解压")

        actual_total = 0
        for info, target in validated:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            file_written = 0
            with archive.open(info, "r") as source, target.open("xb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    file_written += len(chunk)
                    actual_total += len(chunk)
                    if file_written > limits.max_single_file_bytes or actual_total > limits.max_extracted_bytes:
                        raise ArchiveSafetyError("ZIP 实际解压量超过限制")
                    output.write(chunk)
        return actual_total
