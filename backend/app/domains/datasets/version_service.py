"""版本冻结：分组划分、manifest 固化与只读保护。"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ApiError
from app.core.storage import atomic_write_bytes
from app.db.base import utcnow
from app.db.models import DatasetAsset, DatasetVersion
from app.domains.datasets.split_service import SplitError, build_manifest

SPLITS = ("train", "val", "test")


def freeze_version(
    db: Session,
    settings: Settings,
    *,
    version: DatasetVersion,
    seed: int,
    ratios,
    fixed_splits: dict[str, str] | None = None,
    progress=None,
) -> dict:
    if version.state == "frozen":
        raise ApiError(409, "version_frozen", "版本已冻结，内容不可修改")
    if version.state not in {"draft", "failed"}:
        raise ApiError(409, "version_invalid_state", f"当前版本状态不可冻结: {version.state}")

    assets = (
        db.query(DatasetAsset)
        .filter(DatasetAsset.dataset_version_id == version.id, DatasetAsset.deleted_at.is_(None))
        .order_by(DatasetAsset.relative_path.asc())
        .all()
    )
    entries = [
        {
            "relative_path": asset.relative_path,
            "sha256": asset.content_hash,
            "size": asset.size,
            "width": asset.width,
            "height": asset.height,
            "source_group": asset.source_group or f"asset:{asset.relative_path}",
            "root_asset": asset.relative_path,
            "annotations": asset.annotations_json or [],
        }
        for asset in assets
    ]

    version.state = "validating"
    db.commit()
    try:
        manifest, stats = build_manifest(
            entries,
            class_map=version.class_map_json or {},
            seed=seed,
            ratios=ratios,
            dataset_id=version.dataset_id,
            version_id=version.id,
            parent_version_id=version.parent_version_id,
            group_level=(version.metadata_json or {}).get("group_level", "original_image"),
            fixed_splits=fixed_splits,
        )
    except SplitError as exc:
        version.state = "failed"
        db.commit()
        raise ApiError(422, "split_invalid", str(exc), exc.details) from exc

    version_dir = Path(version.root_path)
    manifest_path = version_dir / "manifest.json"
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(manifest_path, payload)

    by_path = {asset.relative_path: asset for asset in assets}
    for item in manifest["assets"]:
        asset = by_path.get(item["relative_path"])
        if asset is not None:
            asset.split = item["split"]
            if not asset.root_asset_id:
                asset.root_asset_id = asset.id

    version.state = "frozen"
    version.status = "ready"
    version.frozen_at = utcnow()
    version.manifest_path = str(manifest_path)
    version.manifest_sha256 = stats["manifest_sha256"]
    version.split_manifest_hash = stats["split_manifest_hash"]
    version.class_map_hash = stats["class_map_hash"]
    version.content_hash = stats["manifest_sha256"]
    version.asset_count = stats["asset_count"]
    version.split_counts_json = stats["split_counts"]
    version.split_config_json = {
        "seed": int(seed),
        "ratios": manifest["ratios"],
        "group_level": manifest["group_level"],
    }
    db.commit()
    if progress:
        progress(1.0, "frozen", "版本已冻结")
    return {
        "version_id": version.id,
        "state": version.state,
        "manifest_sha256": version.manifest_sha256,
        "split_manifest_hash": version.split_manifest_hash,
        "class_map_hash": version.class_map_hash,
        "asset_count": version.asset_count,
        "split_counts": version.split_counts_json,
    }


def assert_editable(version: DatasetVersion) -> None:
    if version.state == "frozen":
        raise ApiError(409, "version_frozen", "冻结版本只读；修改标注请创建子版本")
    if version.deleted_at is not None:
        raise ApiError(404, "not_found", "版本不存在")
