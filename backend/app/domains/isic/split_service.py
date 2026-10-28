"""ISIC 单标签方向的分组切分与冻结。

与 CXR 的差异：
- 分组级别是 lesion（官方 training 集才有 lesion_id），无分组时退化为单图分组并标注限制；
- 单标签互斥：切分后按**类别**检查支持度，而不是逐标签正负例；
- 四集用途与 CXR 相同（train/val/calibration/test），但指标与阈值语义是 softmax 概率。
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ApiError
from app.core.storage import atomic_write_bytes
from app.db.base import utcnow
from app.db.models import DatasetAsset, DatasetResource, DatasetVersion, IsicSample
from app.directions.isic import ISIC_DIRECTION_KEY, ISIC_LABEL_SCHEMA, ISIC_LABELS, ISIC_SPLITS
from app.domains.cxr.split_service import canonical_json, sha256_text


class IsicSplitError(ValueError):
    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def assign_groups(
    groups: list[str], *, namespace: str, split_seed: int, ratios: list[float]
) -> dict[str, str]:
    """按 group_key 的确定性 hash 分配四集；分组数足够时保证四集非空。"""
    ordered = sorted(set(groups), key=lambda key: (sha256_text(f"{namespace}|{split_seed}|{key}"), key))
    if not ordered:
        return {}
    cumulative: list[float] = []
    running = 0.0
    for ratio in ratios:
        running += float(ratio)
        cumulative.append(running)
    cumulative[-1] = 1.0

    assignment: dict[str, str] = {}
    for group in ordered:
        unit = int(sha256_text(f"{namespace}|{split_seed}|{group}")[:13], 16) / float(16**13)
        target = ISIC_SPLITS[-1]
        for split, boundary in zip(ISIC_SPLITS, cumulative):
            if unit < boundary:
                target = split
                break
        assignment[group] = target

    if len(ordered) >= len(ISIC_SPLITS):
        counts = {split: 0 for split in ISIC_SPLITS}
        for split in assignment.values():
            counts[split] += 1
        for split in ISIC_SPLITS:
            if counts[split]:
                continue
            donor = max(ISIC_SPLITS, key=lambda name: (counts[name], -ISIC_SPLITS.index(name)))
            if counts[donor] <= 1:
                continue
            movable = next(item for item in ordered if assignment[item] == donor)
            assignment[movable] = split
            counts[donor] -= 1
            counts[split] += 1
    return assignment


def build_manifest(
    entries: list[dict],
    *,
    namespace: str,
    dataset_code: str,
    version_id: str,
    split_seed: int,
    ratios: list[float],
    group_level: str,
    source: dict | None = None,
) -> tuple[dict, dict]:
    if not entries:
        raise IsicSplitError("没有可冻结的皮肤镜样本")
    assignment = assign_groups(
        [entry["group_key"] for entry in entries],
        namespace=namespace,
        split_seed=split_seed,
        ratios=ratios,
    )
    if len(assignment) < len(ISIC_SPLITS):
        raise IsicSplitError(
            "分组数少于四集数量，无法建立 train/val/calibration/test",
            {"groups": len(assignment), "required": len(ISIC_SPLITS)},
        )

    prepared = []
    for entry in entries:
        item = dict(entry)
        item["split"] = assignment[item["group_key"]]
        prepared.append(item)
    prepared.sort(key=lambda item: item["image"])

    group_splits: dict[str, set[str]] = {}
    for item in prepared:
        group_splits.setdefault(item["group_key"], set()).add(item["split"])
    crossing = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    if crossing:
        raise IsicSplitError("同一分组跨集", {"groups": list(crossing)[:5]})

    counts = {split: 0 for split in ISIC_SPLITS}
    class_counts = {split: {label.key: 0 for label in ISIC_LABELS} for split in ISIC_SPLITS}
    for item in prepared:
        counts[item["split"]] += 1
        class_counts[item["split"]][item["class_key"]] += 1

    manifest = {
        "schema": "isic-manifest-v1",
        "direction": ISIC_DIRECTION_KEY,
        "task": "isic_multiclass",
        "label_schema": ISIC_LABEL_SCHEMA,
        "labels": [
            {"index": label.index, "key": label.key, "display_name": label.display_name}
            for label in ISIC_LABELS
        ],
        "group_level": group_level,
        "namespace": namespace,
        "dataset_code": dataset_code,
        "version_id": version_id,
        "split_seed": int(split_seed),
        "ratios": [float(item) for item in ratios],
        "split_algorithm": "sha256(namespace|split_seed|group_key) ordered cumulative assignment",
        "source": source or {},
        "counts": counts,
        "class_counts": class_counts,
        "group_counts": {split: len({item["group_key"] for item in prepared if item["split"] == split}) for split in ISIC_SPLITS},
        "assets": prepared,
    }
    stats = {
        "manifest_sha256": sha256_text(canonical_json(manifest)),
        "split_manifest_hash": sha256_text(
            canonical_json({item["image"]: item["split"] for item in prepared})
        ),
        "label_schema_hash": sha256_text(canonical_json(manifest["labels"])),
        "asset_count": len(prepared),
        "counts": counts,
        "class_counts": class_counts,
        "group_counts": manifest["group_counts"],
    }
    return manifest, stats


def class_gate(manifest: dict) -> list[dict]:
    """单标签门禁：train 中每个类别至少 1 例；总样本 <5 的类标为稀有类警告。"""
    problems: list[dict] = []
    train_counts = (manifest.get("class_counts") or {}).get("train") or {}
    total_counts = {label.key: 0 for label in ISIC_LABELS}
    for split, counts in (manifest.get("class_counts") or {}).items():
        for key, value in counts.items():
            total_counts[key] += int(value)
    for label in ISIC_LABELS:
        if int(train_counts.get(label.key) or 0) == 0:
            problems.append(
                {"class_key": label.key, "problem": "train 中没有该类别", "total": total_counts[label.key]}
            )
    return problems


def rare_class_warnings(manifest: dict) -> list[dict]:
    total = {label.key: 0 for label in ISIC_LABELS}
    for counts in (manifest.get("class_counts") or {}).values():
        for key, value in counts.items():
            total[key] += int(value)
    return [
        {"class_key": label.key, "total": total[label.key]}
        for label in ISIC_LABELS
        if total[label.key] < 5
    ]


def freeze_version(
    db: Session,
    settings: Settings,
    *,
    version: DatasetVersion,
    resource: DatasetResource,
    split_seed: int,
    ratios: list[float],
    progress=None,
) -> dict:
    db.flush()
    version = (
        db.query(DatasetVersion).filter(DatasetVersion.id == version.id).populate_existing().with_for_update().one()
    )
    if version.state == "frozen":
        raise ApiError(409, "version_frozen", "版本已冻结，内容不可修改")
    if version.state not in {"draft", "failed"}:
        raise ApiError(409, "version_invalid_state", f"当前版本状态不可冻结: {version.state}")
    root = Path(version.root_path or "")
    if not version.root_path or not root.is_absolute():
        raise ApiError(422, "invalid_version_root", "版本目录必须是受控根下的绝对路径")
    version_dir = root.resolve()
    try:
        version_dir.relative_to(Path(settings.data_root).resolve())
    except ValueError as exc:
        raise ApiError(422, "invalid_version_root", "版本目录超出数据根目录") from exc

    version.state = "validating"
    db.commit()

    rows = (
        db.query(IsicSample, DatasetAsset)
        .join(DatasetAsset, DatasetAsset.id == IsicSample.asset_id)
        .filter(IsicSample.dataset_version_id == version.id, DatasetAsset.deleted_at.is_(None))
        .order_by(DatasetAsset.relative_path.asc())
        .all()
    )
    if not rows:
        version.state = "failed"
        db.commit()
        raise ApiError(422, "isic_no_samples", "版本没有皮肤镜样本，无法冻结")

    entries = [
        {
            "image": asset.relative_path,
            "group_key": sample.group_key,
            "lesion_key": sample.lesion_key,
            "class_key": sample.class_key,
            "class_index": int(sample.class_index),
            "sha256": asset.content_hash,
            "pixel_sha256": sample.pixel_hash or asset.content_hash,
            "width": asset.width,
            "height": asset.height,
        }
        for sample, asset in rows
    ]
    namespace = f"isic:{resource.dataset_id}"
    try:
        manifest, stats = build_manifest(
            entries,
            namespace=namespace,
            dataset_code=resource.dataset_id,
            version_id=version.id,
            split_seed=int(split_seed),
            ratios=[float(item) for item in ratios],
            group_level=resource.group_level,
            source={
                "dataset_code": resource.dataset_id,
                "source_version": resource.source_version,
                "license": resource.license,
                "source_split": (version.metadata_json or {}).get("source_split"),
            },
        )
    except IsicSplitError as exc:
        version.state = "failed"
        db.commit()
        raise ApiError(422, "isic_split_invalid", str(exc), exc.details) from exc

    manifest_path = version_dir / "isic-manifest.json"
    atomic_write_bytes(
        manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    )
    for split in ISIC_SPLITS:
        paths = [item["image"] for item in manifest["assets"] if item["split"] == split]
        atomic_write_bytes(
            version_dir / f"{split}.txt", ("\n".join(paths) + ("\n" if paths else "")).encode("utf-8")
        )

    by_path = {asset.relative_path: asset for _sample, asset in rows}
    for item in manifest["assets"]:
        asset = by_path.get(item["image"])
        if asset is not None:
            asset.split = item["split"]

    version.state = "frozen"
    version.status = "ready"
    version.frozen_at = utcnow()
    version.manifest_path = str(manifest_path)
    version.manifest_sha256 = stats["manifest_sha256"]
    version.split_manifest_hash = stats["split_manifest_hash"]
    version.class_map_hash = stats["label_schema_hash"]
    version.content_hash = stats["manifest_sha256"]
    version.asset_count = stats["asset_count"]
    version.split_counts_json = stats["counts"]
    version.split_config_json = {
        "seed": int(split_seed),
        "ratios": manifest["ratios"],
        "group_level": manifest["group_level"],
        "label_schema": ISIC_LABEL_SCHEMA,
        "namespace": namespace,
    }
    version.metadata_json = {
        **(version.metadata_json or {}),
        "direction": ISIC_DIRECTION_KEY,
        "label_schema": ISIC_LABEL_SCHEMA,
        "class_gate": class_gate(manifest),
        "rare_class_warnings": rare_class_warnings(manifest),
    }
    db.commit()
    if progress:
        progress(1.0, "frozen", "皮肤镜版本已冻结")
    return {
        "version_id": version.id,
        "state": version.state,
        "manifest_sha256": version.manifest_sha256,
        "split_manifest_hash": version.split_manifest_hash,
        "asset_count": version.asset_count,
        "split_counts": version.split_counts_json,
        "class_counts": stats["class_counts"],
        "group_counts": stats["group_counts"],
        "class_gate": class_gate(manifest),
        "rare_class_warnings": rare_class_warnings(manifest),
    }
