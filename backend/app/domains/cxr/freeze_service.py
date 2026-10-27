"""CXR 版本冻结：患者 component 四集、manifest 原子发布。

与 detect 的 `freeze_version` 完全分离：不写 YOLO 标签、不产生 BBox、
不改 GROUP_LEVELS，也不接受三集比例。
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ApiError
from app.core.storage import atomic_write_bytes
from app.db.base import utcnow
from app.db.models import CxrSample, DatasetAsset, DatasetResource, DatasetVersion
from app.domains.cxr.split_service import (
    CxrSplitError,
    build_cxr_manifest,
    sha256_text,
    summarize_version,
    training_gate,
)
from app.schemas.cxr import CXR_LABEL_SCHEMA, CXR_TASK


def freeze_cxr_version(
    db: Session,
    settings: Settings,
    *,
    version: DatasetVersion,
    resource: DatasetResource,
    split_seed: int,
    ratios: list[float],
    progress=None,
) -> dict:
    if resource.task_type != CXR_TASK:
        raise ApiError(409, "task_mismatch", "该数据集不是胸片五标签数据集")
    db.flush()
    version = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == version.id)
        .populate_existing()
        .with_for_update()
        .one()
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
        db.query(CxrSample, DatasetAsset)
        .join(DatasetAsset, DatasetAsset.id == CxrSample.asset_id)
        .filter(
            CxrSample.dataset_version_id == version.id,
            DatasetAsset.deleted_at.is_(None),
        )
        .order_by(DatasetAsset.relative_path.asc())
        .all()
    )
    if not rows:
        version.state = "failed"
        db.commit()
        raise ApiError(422, "cxr_no_samples", "版本没有胸片样本，无法冻结")

    entries = [
        {
            "relative_path": asset.relative_path,
            "sha256": asset.content_hash,
            "pixel_sha256": sample.pixel_hash or asset.content_hash,
            "size": asset.size,
            "width": asset.width,
            "height": asset.height,
            "patient_key": sample.patient_key,
            "study_key": sample.study_key,
            "component_key": sample.component_key,
            "view": sample.view,
            "raw_labels": dict(sample.raw_labels_json or {}),
        }
        for sample, asset in rows
    ]

    namespace = f"cxr:{resource.dataset_id}"
    try:
        manifest, stats = build_cxr_manifest(
            entries,
            namespace=namespace,
            dataset_id=resource.dataset_id,
            version_id=version.id,
            split_seed=int(split_seed),
            ratios=[float(item) for item in ratios],
            source={
                "dataset_code": resource.dataset_id,
                "source_version": resource.source_version,
                "license": resource.license,
                "manifest_schema": "cxr-manifest-v1",
            },
        )
    except CxrSplitError as exc:
        version.state = "failed"
        db.commit()
        raise ApiError(422, "cxr_split_invalid", str(exc), exc.details) from exc

    if stats["asset_count"] < 4 or len(stats["component_counts"]) < 4:
        version.state = "failed"
        db.commit()
        raise ApiError(
            422,
            "cxr_insufficient_components",
            "患者 component 少于 4 个，无法建立 train/val/calibration/test 四集",
            {"asset_count": stats["asset_count"], "component_counts": stats["component_counts"]},
        )

    try:
        manifest_path = version_dir / "cxr-manifest.json"
        payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        atomic_write_bytes(manifest_path, payload)
        for split in ("train", "val", "calibration", "test"):
            paths = [entry["relative_path"] for entry in manifest["assets"] if entry["split"] == split]
            atomic_write_bytes(version_dir / f"{split}.txt", ("\n".join(paths) + ("\n" if paths else "")).encode("utf-8"))
        # 训练/评价只需 manifest；拆分清单是给人核对的副产物。
        sidecar = version_dir / "cxr-summary.json"
        atomic_write_bytes(sidecar, json.dumps(summarize_version(manifest), ensure_ascii=False, indent=2).encode("utf-8"))
    except Exception:
        db.rollback()
        version.state = "failed"
        db.commit()
        raise

    by_path = {asset.relative_path: asset for _sample, asset in rows}
    sample_by_asset = {sample.asset_id: sample for sample, _asset in rows}
    for item in manifest["assets"]:
        asset = by_path.get(item["relative_path"])
        if asset is not None:
            asset.split = item["split"]
            asset.group_level = "patient_component"
        sample = sample_by_asset.get(asset.id) if asset is not None else None
        if sample is not None:
            sample.component_key = item["component_key"]

    version.state = "frozen"
    version.status = "ready"
    version.frozen_at = utcnow()
    version.manifest_path = str(manifest_path)
    version.manifest_sha256 = stats["manifest_sha256"]
    version.split_manifest_hash = stats["split_manifest_hash"]
    version.class_map_hash = stats["label_schema_hash"]
    version.content_hash = stats["manifest_sha256"]
    version.asset_count = stats["asset_count"]
    version.split_counts_json = stats["split_counts"]
    version.split_config_json = {
        "seed": int(split_seed),
        "ratios": manifest["ratios"],
        "group_level": manifest["group_level"],
        "label_schema": CXR_LABEL_SCHEMA,
        "namespace": namespace,
    }
    version.metadata_json = {
        **(version.metadata_json or {}),
        "group_level": "patient_component",
        "label_schema": CXR_LABEL_SCHEMA,
        "task": CXR_TASK,
        "training_gate": training_gate(manifest),
        "summary_sha256": sha256_text(json.dumps(summarize_version(manifest), sort_keys=True)),
    }
    db.commit()
    if progress:
        progress(1.0, "frozen", "胸片版本已冻结")
    return {
        "version_id": version.id,
        "state": version.state,
        "manifest_sha256": version.manifest_sha256,
        "split_manifest_hash": version.split_manifest_hash,
        "label_schema_hash": version.class_map_hash,
        "asset_count": version.asset_count,
        "split_counts": version.split_counts_json,
        "patient_counts": stats["patient_counts"],
        "training_gate": training_gate(manifest),
    }
