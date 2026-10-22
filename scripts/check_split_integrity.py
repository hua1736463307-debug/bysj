#!/usr/bin/env python3
"""校验数据版本 manifest 的 split 完整性。

用法：
    python scripts/check_split_integrity.py --manifest data/prepared/bccd/manifest.json

退出码：0 通过；1 发现泄漏；2 输入无效。
检查来源组 / root_asset / 内容 hash 三个层面，并重新计算实际文件 hash
（不只看清单文本）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(_fail(2, f"manifest 无法解析: {exc}"))
    if not isinstance(manifest, dict) or not manifest.get("assets"):
        raise SystemExit(_fail(2, "manifest 缺少 assets"))
    return manifest


def _fail(code: int, message: str) -> int:
    print(json.dumps({"ok": False, "error": message}, ensure_ascii=False))
    return code


def check(manifest: dict, manifest_path: Path) -> dict:
    assets = manifest["assets"]
    problems: list[dict] = []

    for level in ("source_group", "root_asset", "sha256"):
        seen: dict[str, set[str]] = defaultdict(set)
        for item in assets:
            seen[str(item.get(level))].add(item.get("split"))
        leaks = [
            {"value": key, "splits": sorted(splits)}
            for key, splits in seen.items()
            if len(splits) > 1
        ]
        if leaks:
            problems.append({"level": level, "leaks": leaks[:20], "count": len(leaks)})

    missing_split = [item["relative_path"] for item in assets if item.get("split") not in SPLITS]
    if missing_split:
        problems.append({"level": "split_missing", "files": missing_split[:20], "count": len(missing_split)})

    unknown_split = [item["relative_path"] for item in assets if item.get("split") not in SPLITS]
    # 重复路径
    paths = Counter(item["relative_path"] for item in assets)
    duplicates = [path for path, count in paths.items() if count > 1]
    if duplicates:
        problems.append({"level": "duplicate_path", "files": duplicates[:20], "count": len(duplicates)})

    # 实际文件校验：manifest 同级 images/<relative_path>
    images_root = manifest_path.parent / "images"
    hash_mismatch = []
    missing_files = []
    checked = 0
    for item in assets:
        candidate = images_root / item["relative_path"]
        if not candidate.exists():
            missing_files.append(item["relative_path"])
            continue
        checked += 1
        actual = sha256_file(candidate)
        if actual != item["sha256"]:
            hash_mismatch.append(
                {"file": item["relative_path"], "expected": item["sha256"], "actual": actual}
            )
    if hash_mismatch:
        problems.append({"level": "content_hash", "items": hash_mismatch[:20], "count": len(hash_mismatch)})
    if missing_files:
        problems.append({"level": "missing_file", "files": missing_files[:20], "count": len(missing_files)})

    counts = Counter(item.get("split") for item in assets)
    return {
        "ok": not problems,
        "problems": problems,
        "counts": {split: counts.get(split, 0) for split in SPLITS},
        "asset_count": len(assets),
        "files_verified": checked,
        "manifest_sha256_declared": manifest.get("manifest_sha256"),
        "split_manifest_hash_declared": manifest.get("split_manifest_hash"),
    }


def compare_parent(child: dict, parent: dict) -> list[dict]:
    """反馈子版本必须保持 val/test 逐资产 hash 与父版本一致。"""
    problems = []
    for split in ("val", "test"):
        parent_map = {
            item["relative_path"]: item["sha256"] for item in parent["assets"] if item.get("split") == split
        }
        child_map = {
            item["relative_path"]: item["sha256"] for item in child["assets"] if item.get("split") == split
        }
        if parent_map != child_map:
            problems.append(
                {
                    "level": f"{split}_snapshot_changed",
                    "added": sorted(set(child_map) - set(parent_map))[:20],
                    "removed": sorted(set(parent_map) - set(child_map))[:20],
                    "changed": sorted(
                        path for path in set(parent_map) & set(child_map) if parent_map[path] != child_map[path]
                    )[:20],
                }
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 manifest 的三集隔离与文件 hash")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--parent", help="父版本 manifest，用于反馈子版本比较")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        return _fail(2, f"manifest 不存在: {manifest_path}")
    manifest = load_manifest(manifest_path)
    report = check(manifest, manifest_path)

    if args.parent:
        parent_path = Path(args.parent)
        if not parent_path.exists():
            return _fail(2, f"父 manifest 不存在: {parent_path}")
        parent = load_manifest(parent_path)
        parent_problems = compare_parent(manifest, parent)
        if parent_problems:
            report["ok"] = False
            report["problems"].extend(parent_problems)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
