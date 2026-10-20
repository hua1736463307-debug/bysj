"""Filesystem helpers for composing original datasets without mutating sources."""

import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


class DatasetMergeError(ValueError):
    """Raised when a merge request cannot be completed safely."""


def _source_directory_name(index: int, dataset_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(dataset_id or "")).strip("._-")
    return f"source_{index:02d}_{safe_id or 'dataset'}"


def _resolve_source_file(root: Path, relative_path: Any) -> Path:
    relative_text = str(relative_path or "").replace("\\", "/").strip().lstrip("/")
    if not relative_text:
        raise DatasetMergeError("源数据集包含空文件路径")
    source_file = (root / relative_text).resolve()
    try:
        source_file.relative_to(root)
    except ValueError as exc:
        raise DatasetMergeError(f"源数据集包含非法文件路径: {relative_text}") from exc
    if not source_file.is_file():
        raise DatasetMergeError(f"源图片不存在: {relative_text}")
    return source_file


def merge_original_dataset_assets(
    source_specs: Iterable[Dict[str, Any]],
    destination: Path,
) -> Dict[str, Any]:
    """Copy indexed images into source-scoped directories under a new dataset root."""
    sources: List[Dict[str, Any]] = list(source_specs)
    if len(sources) < 2:
        raise DatasetMergeError("至少选择两个源数据集")

    destination = Path(destination).resolve()
    if destination.exists():
        raise DatasetMergeError("目标数据集目录已存在")

    source_ids = [str(item.get("dataset_id") or "").strip() for item in sources]
    if any(not source_id for source_id in source_ids) or len(set(source_ids)) != len(source_ids):
        raise DatasetMergeError("源数据集ID不能为空或重复")

    source_reports = []
    total_images = 0
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for index, source in enumerate(sources, start=1):
            source_id = source_ids[index - 1]
            source_root = Path(source.get("root_path") or "").resolve()
            if not source_root.is_dir():
                raise DatasetMergeError(f"源数据集不存在: {source_id}")
            try:
                destination.relative_to(source_root)
            except ValueError:
                pass
            else:
                raise DatasetMergeError(f"目标目录不能位于源数据集内: {source_id}")

            relative_paths = [str(value) for value in (source.get("relative_paths") or [])]
            if not relative_paths:
                raise DatasetMergeError(f"源数据集没有可合并图片: {source_id}")

            source_directory = destination / _source_directory_name(index, source_id)
            copied_count = 0
            for relative_path in relative_paths:
                source_file = _resolve_source_file(source_root, relative_path)
                normalized_relative = Path(relative_path.replace("\\", "/"))
                target_file = source_directory.joinpath(*normalized_relative.parts)
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target_file)
                copied_count += 1

            total_images += copied_count
            source_reports.append({"dataset_id": source_id, "image_count": copied_count})

        return {"source_count": len(sources), "image_count": total_images, "sources": source_reports}
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
