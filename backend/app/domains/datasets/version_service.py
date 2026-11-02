"""版本冻结：分组划分、manifest 固化与只读保护。"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ApiError
from app.core.storage import atomic_write_bytes, resolve_under, normalize_relative
from app.db.base import utcnow
from app.db.models import DatasetAsset, DatasetVersion
from app.domains.datasets.split_service import SplitError, build_manifest

SPLITS = ("train", "val", "test")
GROUP_LEVELS = {"original_image", "image_cluster"}


def _explicit_group_level(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_group_level(metadata: dict | None) -> str:
    """Resolve the declared grouping level without silently picking a side.

    Read order: top-level ``group_level`` -> ``source.group_level`` ->
    ``original_image``. Only the two supported levels are accepted; a draft
    that declares both and they disagree is rejected instead of normalized.
    """
    metadata = metadata or {}
    top = _explicit_group_level(metadata.get("group_level"))
    source = metadata.get("source")
    nested = _explicit_group_level(source.get("group_level")) if isinstance(source, dict) else None
    if top and nested and top != nested:
        raise ApiError(
            409,
            "group_level_conflict",
            "版本元数据中的 group_level 声明冲突",
            {"top_level": top, "source": nested},
        )
    value = top or nested or "original_image"
    if value not in GROUP_LEVELS:
        raise ApiError(422, "invalid_group_level", f"未知的 group_level: {value}")
    return value



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
    # Feedback creates a child and sets its directory in this transaction.
    # Refreshing with autoflush disabled would discard that pending directory.
    db.flush()
    version = db.query(DatasetVersion).filter(DatasetVersion.id == version.id).populate_existing().with_for_update().one()
    if version.state == "frozen":
        raise ApiError(409, "version_frozen", "版本已冻结，内容不可修改")
    if version.state not in {"draft", "failed"}:
        raise ApiError(409, "version_invalid_state", f"当前版本状态不可冻结: {version.state}")
    if (version.metadata_json or {}).get("requires_annotation_review"):
        raise ApiError(409, "annotation_review_required", "AI 预标注版本须由复核员确认后再冻结")
    # Resolve before changing state so a conflicting declaration cannot leave
    # the version in "validating" or pick one of the two values silently.
    declared_group_level = resolve_group_level(version.metadata_json)
    root = Path(version.root_path or "")
    if not version.root_path or not root.is_absolute():
        raise ApiError(422, "invalid_version_root", "版本目录必须是受控根下的绝对路径")
    version_dir = root.resolve()
    try:
        relative_root = version_dir.relative_to(Path(settings.data_root).resolve())
    except ValueError as exc:
        raise ApiError(422, "invalid_version_root", "版本目录超出数据根目录") from exc
    if not relative_root.parts:
        raise ApiError(422, "invalid_version_root", "版本目录不能是数据根目录本身")

    version.state = "validating"
    db.commit()
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
    if (version.metadata_json or {}).get("preserve_parent_splits") and fixed_splits is None:
        fixed_splits = {asset.relative_path: asset.split for asset in assets}
        config = version.split_config_json or {}
        seed = int(config.get("seed", seed))
        if config.get("ratios"):
            ratios = [config["ratios"][name] for name in ("train", "val", "test")]

    try:
        manifest, stats = build_manifest(
            entries,
            class_map=version.class_map_json or {},
            seed=seed,
            ratios=ratios,
            dataset_id=version.dataset_id,
            version_id=version.id,
            parent_version_id=version.parent_version_id,
            group_level=declared_group_level,
            fixed_splits=fixed_splits,
        )
    except SplitError as exc:
        version.state = "failed"
        db.commit()
        raise ApiError(422, "split_invalid", str(exc), exc.details) from exc

    # YOLO reads sidecar files, not database JSON. Materialize the exact frozen
    # labels, including empty files for deleted boxes, before publishing frozen.
    try:
        label_names = set()
        for item in manifest["assets"]:
            relative = normalize_relative(item["relative_path"]).with_suffix(".txt").as_posix()
            if relative in label_names:
                version.state = "failed"
                db.commit()
                raise ApiError(409, "label_name_collision", "不同图像对应相同 YOLO 标签路径")
            label_names.add(relative)
            label = resolve_under(version_dir / "labels", relative)
            body = "".join(f"{a['class_id']} {a['cx']:.6f} {a['cy']:.6f} {a['w']:.6f} {a['h']:.6f}\n" for a in item["annotations"])
            atomic_write_bytes(label, body.encode("utf-8"))
        manifest_path = version_dir / "manifest.json"
        payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        atomic_write_bytes(manifest_path, payload)
    except Exception:
        db.rollback()
        version.state = "failed"
        db.commit()
        raise

    by_path = {asset.relative_path: asset for asset in assets}
    for item in manifest["assets"]:
        asset = by_path.get(item["relative_path"])
        if asset is not None:
            asset.split = item["split"]
            asset.label_path = "labels/" + normalize_relative(asset.relative_path).with_suffix(".txt").as_posix()
            if not asset.root_asset_id:
                asset.root_asset_id = asset.id

    # Freeze the declaration next to the manifest so later readers (API,
    # processing, feedback) do not fall back to original_image.
    metadata = dict(version.metadata_json or {})
    metadata["group_level"] = declared_group_level
    version.metadata_json = metadata

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
    if version.state not in {"draft", "failed"}:
        raise ApiError(409, "version_frozen", "冻结版本只读；修改标注请创建子版本")
    if version.deleted_at is not None:
        raise ApiError(404, "not_found", "版本不存在")
