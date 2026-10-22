"""复核回流：创建 train 子版本，test/val 快照逐资产不变。

规则（CONTRACTS C03-5/6、TASK_BOOK PR-08）：
- 只允许 approved、train、非派生资产回流；
- test 的原图/切片/改名副本一律拒绝；
- 一次成功导出只产生一个子版本，重复导出幂等；
- 导出后重新验证 split/类别，冻结成功才可训练。
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ApiError
from app.core.storage import ensure_dir, sha256_text
from app.db.models import DatasetAsset, DatasetVersion, FeedbackExport, ReviewRecord
from app.domains.datasets import version_service


def _parent_ratios(parent: DatasetVersion) -> tuple[float, float, float]:
    ratios = (parent.split_config_json or {}).get("ratios") or {}
    return (
        float(ratios.get("train", 0.8)),
        float(ratios.get("val", 0.1)),
        float(ratios.get("test", 0.1)),
    )


def _review_hash(review_ids: list[str]) -> str:
    return sha256_text(json.dumps(sorted(review_ids), separators=(",", ":")))


def export_feedback_version(
    db: Session,
    settings: Settings,
    payload: dict,
    progress=None,
) -> dict:
    parent = db.get(DatasetVersion, payload["parent_version_id"])
    if parent is None or parent.deleted_at is not None:
        raise ApiError(404, "not_found", "父版本不存在")
    if parent.state != "frozen":
        raise ApiError(409, "version_not_frozen", "只有冻结版本可以创建回流子版本")

    review_ids = list(payload.get("review_ids") or [])
    if not review_ids:
        raise ApiError(422, "review_required", "必须选择已批准的复核记录")
    key = payload.get("idempotency_key") or _review_hash(review_ids)

    existing = db.query(FeedbackExport).filter(FeedbackExport.idempotency_key == key).one_or_none()
    if existing and existing.child_version_id:
        return {
            "child_version_id": existing.child_version_id,
            "idempotent": True,
            "parent_version_id": parent.id,
        }

    reviews = db.query(ReviewRecord).filter(ReviewRecord.id.in_(review_ids)).all()
    if len(reviews) != len(set(review_ids)):
        raise ApiError(404, "not_found", "部分复核记录不存在")
    for review in reviews:
        if review.state != "approved":
            raise ApiError(409, "review_not_approved", "只有 approved 复核可以回流")

    parent_assets = (
        db.query(DatasetAsset)
        .filter(DatasetAsset.dataset_version_id == parent.id, DatasetAsset.deleted_at.is_(None))
        .all()
    )
    by_id = {asset.id: asset for asset in parent_assets}
    test_hashes = {asset.content_hash for asset in parent_assets if asset.split == "test"}
    test_groups = {asset.source_group for asset in parent_assets if asset.split == "test"}
    protected = {asset.relative_path for asset in parent_assets if asset.split in {"val", "test"}}

    reviewed: dict[int, list[dict]] = {}
    excluded: set[int] = set()
    for review in reviews:
        asset = by_id.get(review.image_asset_id)
        if asset is None:
            raise ApiError(422, "asset_not_in_parent", "复核资产不属于父版本")
        if asset.split != "train":
            raise ApiError(409, "test_leak_blocked", "只允许 train 资产回流，test/val 一律拒绝")
        if asset.original_asset_id or (asset.root_asset_id and asset.root_asset_id != asset.id):
            raise ApiError(409, "derived_asset_blocked", "派生资产不能作为回流来源")
        if asset.content_hash in test_hashes and asset.relative_path in protected:
            raise ApiError(409, "test_leak_blocked", "该资产与测试集内容相同")
        if asset.source_group in test_groups and asset.relative_path in protected:
            raise ApiError(409, "test_leak_blocked", "该资产与测试集同源")
        from app.db.models import ReviewItem

        verdicts = {
            item.verdict for item in db.query(ReviewItem).filter(ReviewItem.review_id == review.id).all()
        }
        if "unreviewable" in verdicts:
            excluded.add(asset.id)
            continue
        reviewed[asset.id] = review.final_annotations_json or []

    record = FeedbackExport(
        job_id=payload.get("job_id"),
        parent_version_id=parent.id,
        review_ids_hash=_review_hash(review_ids),
        idempotency_key=key,
        status="running",
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    child = DatasetVersion(
        dataset_id=parent.dataset_id,
        dataset_role=parent.dataset_role,
        parent_version_id=parent.id,
        status="draft",
        state="draft",
        root_path="",
        manifest_path="",
        class_map_json=parent.class_map_json or {},
        split_config_json={
            **(parent.split_config_json or {}),
            "feedback_parent": parent.id,
            "review_ids_hash": record.review_ids_hash,
        },
        metadata_json={**(parent.metadata_json or {}), "feedback_export_id": record.id},
        content_hash="",
        created_by=parent.created_by,
        created_by_user_id=parent.created_by_user_id,
        asset_count=len(parent_assets),
    )
    db.add(child)
    db.flush()

    child_root = ensure_dir(Path(settings.data_root) / "datasets" / parent.dataset_id / child.id)
    images_root = ensure_dir(child_root / "images")
    labels_root = ensure_dir(child_root / "labels")
    parent_root = Path(parent.root_path)

    fixed_splits: dict[str, str] = {}
    copied = 0
    for asset in parent_assets:
        if asset.id in excluded:
            continue
        annotations = reviewed.get(asset.id, asset.annotations_json or [])
        source = parent_root / "images" / asset.relative_path
        target = images_root / asset.relative_path
        ensure_dir(target.parent)
        if source.exists():
            shutil.copy2(source, target)
        label_target = None
        if annotations:
            label_target = labels_root / Path(asset.relative_path).with_suffix(".txt")
            ensure_dir(label_target.parent)
            label_target.write_text(
                "\n".join(
                    f"{int(ann['class_id'])} {float(ann['cx']):.6f} {float(ann['cy']):.6f} {float(ann['w']):.6f} {float(ann['h']):.6f}"
                    for ann in annotations
                )
                + "\n",
                encoding="utf-8",
            )
        db.add(
            DatasetAsset(
                dataset_version_id=child.id,
                relative_path=asset.relative_path,
                basename=asset.basename,
                size=asset.size,
                width=asset.width,
                height=asset.height,
                label_path=str(label_target.relative_to(child_root)).replace("\\", "/") if label_target else None,
                source_group=asset.source_group,
                group_level=asset.group_level,
                content_hash=asset.content_hash,
                annotations_json=annotations,
                root_asset_id=asset.id,
                split=asset.split,
            )
        )
        fixed_splits[asset.relative_path] = asset.split
        copied += 1
        if progress:
            progress(min(0.7, copied / max(1, len(parent_assets)) * 0.7), "copying", f"复制 {copied}/{len(parent_assets)}")
    db.commit()

    child.root_path = str(child_root)
    try:
        stats = version_service.freeze_version(
            db,
            settings,
            version=child,
            seed=int((parent.split_config_json or {}).get("seed") or 42),
            ratios=_parent_ratios(parent),
            fixed_splits=fixed_splits,
            progress=progress,
        )
    except Exception:
        child.state = "failed"
        db.commit()
        record.status = "failed"
        db.commit()
        raise

    # 冻结后再次校验 test/val 逐资产 hash 未变
    child_assets = (
        db.query(DatasetAsset)
        .filter(DatasetAsset.dataset_version_id == child.id, DatasetAsset.deleted_at.is_(None))
        .all()
    )
    for split in ("val", "test"):
        parent_map = {a.relative_path: a.content_hash for a in parent_assets if a.split == split}
        child_map = {a.relative_path: a.content_hash for a in child_assets if a.split == split}
        if parent_map != child_map:
            raise ApiError(409, "split_changed", f"{split} 快照发生变化，回流被拒绝")

    record.child_version_id = child.id
    record.status = "completed"
    db.commit()
    return {
        "child_version_id": child.id,
        "parent_version_id": parent.id,
        "idempotent": False,
        "asset_count": stats["asset_count"],
        "split_counts": stats["split_counts"],
        "manifest_sha256": stats["manifest_sha256"],
        "review_ids_hash": record.review_ids_hash,
    }
