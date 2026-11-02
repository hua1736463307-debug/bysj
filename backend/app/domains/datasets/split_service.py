"""版本划分与 manifest 生成（纯函数，便于复现测试）。

不变量（CONTRACTS C03）：
- 先按来源组分组，再整体分配 split，绝不按图片打散；
- train/val/test 在来源组、root_asset、内容 hash 三个层面都不相交；
- manifest 不含生成时间等易变字段，同 seed 同输入产出相同 hash。
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict

SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.8, 0.1, 0.1)


class SplitError(ValueError):
    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def canonical_json(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def class_map_hash(class_map: dict[str, str]) -> str:
    return sha256_text(canonical_json({str(k): str(v) for k, v in sorted(class_map.items(), key=lambda kv: int(kv[0]))}))


def _normalize_ratios(ratios) -> tuple[float, float, float]:
    values = tuple(float(v) for v in ratios)
    if len(values) != 3 or any(v <= 0 for v in values):
        raise SplitError("三集比例必须是三个正数", {"ratios": list(values)})
    total = sum(values)
    return tuple(v / total for v in values)  # type: ignore[return-value]


def assign_groups_to_splits(
    groups: dict[str, list[dict]],
    *,
    seed: int,
    ratios=DEFAULT_RATIOS,
) -> dict[str, str]:
    """返回 group -> split 的确定性分配。"""
    if len(groups) < 3:
        raise SplitError(
            "来源组少于 3 个，无法形成互不重叠的三集划分",
            {"group_count": len(groups), "required": 3},
        )
    weights = _normalize_ratios(ratios)
    total_images = sum(len(items) for items in groups.values())
    targets = {split: weights[i] * total_images for i, split in enumerate(SPLITS)}
    counts = {split: 0 for split in SPLITS}

    rng = random.Random(seed)
    group_ids = sorted(groups.keys())
    rng.shuffle(group_ids)
    # 大组优先，减少尾部不均衡；同尺寸保持 seed 决定的顺序。
    ordered = sorted(group_ids, key=lambda gid: (-len(groups[gid]), group_ids.index(gid)))

    assignment: dict[str, str] = {}
    empty = [split for split in SPLITS]
    for position, gid in enumerate(ordered):
        remaining = len(ordered) - position
        if empty and remaining <= len(empty):
            split = empty[0]
        else:
            split = max(SPLITS, key=lambda s: (targets[s] - counts[s], -SPLITS.index(s)))
        assignment[gid] = split
        counts[split] += len(groups[gid])
        if split in empty:
            empty.remove(split)
    return assignment


def _classes_of(items: list[dict]) -> set[int]:
    classes: set[int] = set()
    for item in items:
        for annotation in item.get("annotations") or []:
            classes.add(int(annotation["class_id"]))
    return classes


def build_manifest(
    entries: list[dict],
    *,
    class_map: dict[str, str],
    seed: int,
    ratios=DEFAULT_RATIOS,
    dataset_id: str,
    version_id: str,
    parent_version_id: str | None = None,
    group_level: str = "original_image",
    fixed_splits: dict[str, str] | None = None,
) -> tuple[dict, dict]:
    """生成 manifest 与统计；不可满足时抛 SplitError。"""
    if not entries:
        raise SplitError("版本中没有可用图像资产", {"asset_count": 0})

    by_group: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        group = str(entry.get("source_group") or f"asset:{entry['relative_path']}")
        by_group[group].append(entry)

    # 同内容不同文件名必须同组，否则直接拒绝（不能只检查组名交集）。
    hash_to_groups: dict[str, set[str]] = defaultdict(set)
    for group, items in by_group.items():
        for item in items:
            hash_to_groups[item["sha256"]].add(group)
    conflicts = {h: sorted(g) for h, g in hash_to_groups.items() if len(g) > 1}
    if conflicts:
        raise SplitError(
            "相同内容哈希出现在不同来源组，请先合并来源组或移除重复文件",
            {"conflicts": [{"sha256": h, "groups": g} for h, g in list(conflicts.items())[:20]]},
        )

    if fixed_splits:
        # 反馈子版本必须继承父版本的 val/test 快照，不允许重新随机划分。
        assignment = {}
        for group, items in by_group.items():
            splits = {fixed_splits.get(item["relative_path"]) for item in items}
            if None in splits:
                raise SplitError("子版本存在缺少 split 的资产", {"group": group})
            if len(splits) > 1:
                raise SplitError("同一来源组的资产被分到不同 split", {"group": group, "splits": sorted(splits)})
            assignment[group] = splits.pop()
    else:
        assignment = assign_groups_to_splits(by_group, seed=seed, ratios=ratios)

    known_classes = {int(k) for k in class_map.keys()}
    for entry in entries:
        for annotation in entry.get("annotations") or []:
            if int(annotation["class_id"]) not in known_classes:
                raise SplitError(
                    "标注类别不在冻结 class_map 中",
                    {"file": entry["relative_path"], "class_id": annotation["class_id"]},
                )

    split_classes: dict[str, set[int]] = {split: set() for split in SPLITS}
    split_counts: Counter[str] = Counter()
    ordered_entries = []
    for entry in sorted(entries, key=lambda item: item["relative_path"]):
        group = str(entry.get("source_group") or f"asset:{entry['relative_path']}")
        split = assignment[group]
        split_counts[split] += 1
        split_classes[split] |= _classes_of([entry])
        record = {
            "relative_path": entry["relative_path"],
            "sha256": entry["sha256"],
            "size": int(entry.get("size") or 0),
            "width": int(entry.get("width") or 0),
            "height": int(entry.get("height") or 0),
            "source_group": group,
            "split": split,
            "root_asset": entry.get("root_asset") or entry["relative_path"],
            "annotations": sorted(
                (
                    {
                        "class_id": int(annotation["class_id"]),
                        "cx": round(float(annotation["cx"]), 6),
                        "cy": round(float(annotation["cy"]), 6),
                        "w": round(float(annotation["w"]), 6),
                        "h": round(float(annotation["h"]), 6),
                        **({"polygon": annotation["polygon"]} if annotation.get("polygon") else {}),
                    }
                    for annotation in (entry.get("annotations") or [])
                ),
                key=lambda ann: (ann["class_id"], ann["cx"], ann["cy"], ann["w"], ann["h"]),
            ),
        }
        ordered_entries.append(record)

    all_classes = set().union(*split_classes.values()) if split_classes else set()
    missing = {split: sorted(all_classes - classes) for split, classes in split_classes.items()}
    missing = {split: classes for split, classes in missing.items() if classes}
    if missing:
        raise SplitError("存在 split 缺少某些类别，冻结被阻止", {"missing_classes": missing, "counts": dict(split_counts)})

    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "version_id": version_id,
        "parent_version_id": parent_version_id,
        "group_level": group_level,
        "split_seed": int(seed),
        "ratios": {split: round(weight, 6) for split, weight in zip(SPLITS, _normalize_ratios(ratios))},
        "class_map": {str(k): str(v) for k, v in sorted(class_map.items(), key=lambda kv: int(kv[0]))},
        "counts": {split: split_counts[split] for split in SPLITS},
        "assets": ordered_entries,
    }
    manifest_sha = sha256_text(canonical_json(manifest))
    split_hash = sha256_text(
        canonical_json(
            {
                split: sorted(item["sha256"] for item in ordered_entries if item["split"] == split)
                for split in SPLITS
            }
        )
    )
    stats = {
        "manifest_sha256": manifest_sha,
        "split_manifest_hash": split_hash,
        "class_map_hash": class_map_hash(class_map),
        "asset_count": len(ordered_entries),
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "group_count": len(by_group),
    }
    return manifest, stats


def integrity_report(manifest: dict) -> dict:
    """检查三集在来源组 / root_asset / 内容 hash 三个层面是否相交。"""
    assets = manifest.get("assets") or []
    problems = []
    for level in ("source_group", "root_asset", "sha256"):
        seen: dict[str, set[str]] = defaultdict(set)
        for item in assets:
            seen[str(item.get(level))].add(item.get("split"))
        leaks = {key: sorted(splits) for key, splits in seen.items() if len(splits) > 1}
        if leaks:
            problems.append({"level": level, "leaks": [{"value": k, "splits": v} for k, v in leaks.items()]})
    counts = Counter(item.get("split") for item in assets)
    return {
        "ok": not problems,
        "problems": problems,
        "counts": {split: counts.get(split, 0) for split in SPLITS},
        "asset_count": len(assets),
    }
