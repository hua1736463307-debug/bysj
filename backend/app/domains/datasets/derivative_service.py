"""Durable merge, train-only augmentation and AI annotation derivatives."""
import copy
import json
import shutil
import uuid
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
from app.core.errors import ApiError
from app.core.storage import atomic_write_bytes, ensure_dir, resolve_under, sha256_file, remove_tree
from app.db.base import utcnow
from app.db.models import DatasetAsset, DatasetResource, DatasetVersion, ModelRegistry
from app.domains.datasets.split_service import build_manifest, integrity_report
from app.domains.models.service import verify_model_file
from app.ml.yolo_adapter import get_adapter
from app.workers.gpu import acquire, release


def transform(image, boxes, operation, value):
    boxes = copy.deepcopy(boxes)
    if operation == "rotate":
        image = image.rotate(value, expand=True)
        for box in boxes:
            x, y = box["cx"], box["cy"]
            if value == 90:
                box["cx"], box["cy"] = y, 1 - x
            elif value == 180:
                box["cx"], box["cy"] = 1 - x, 1 - y
            else:
                box["cx"], box["cy"] = 1 - y, x
            if value in (90, 270):
                box["w"], box["h"] = box["h"], box["w"]
            if box.get("polygon"):
                rotations = {90: lambda x, y: [y, 1 - x], 180: lambda x, y: [1 - x, 1 - y],
                             270: lambda x, y: [1 - y, x]}
                box["polygon"] = [rotations[value](x, y) for x, y in box["polygon"]]
    elif operation == "flip":
        image = ImageOps.mirror(image)
        for box in boxes:
            box["cx"] = 1 - box["cx"]
            if box.get("polygon"):
                box["polygon"] = [[1 - x, y] for x, y in box["polygon"]]
    elif operation == "brightness":
        image = ImageEnhance.Brightness(image).enhance(value)
    elif operation == "contrast":
        image = ImageEnhance.Contrast(image).enhance(value)
    elif operation == "grayscale":
        image = ImageOps.grayscale(image).convert("RGB")
    return image, boxes


def _predictions(adapter, handle, source, asset, payload, class_map):
    predictions = adapter.predict_image(handle, source, {"confidence": payload["confidence"], "device": "0"})
    mapping = payload.get("class_mapping") or {}
    names = {name: int(key) for key, name in class_map.items()}
    result = []
    for prediction in predictions:
        class_id = mapping.get(str(prediction["class_id"]))
        if class_id is None:
            class_id = names.get(prediction.get("class_name"))
        if class_id is None:
            continue
        left, top, right, bottom = prediction["bbox_xyxy"]
        left, right = max(0, left), min(asset.width, right)
        top, bottom = max(0, top), min(asset.height, bottom)
        if right <= left or bottom <= top:
            continue
        result.append({"class_id": int(class_id), "cx": (left + right) / (2 * asset.width),
                       "cy": (top + bottom) / (2 * asset.height), "w": (right - left) / asset.width,
                       "h": (bottom - top) / asset.height, "source": "ai"})
    return result


def process_derivative(ctx):
    db, settings, payload = ctx.db, ctx.settings, ctx.job.input_json
    operation = payload["operation"]
    child_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "bysj:derivative:" + ctx.job.id))
    existing = db.get(DatasetVersion, child_id)
    if existing:
        return {"dataset_id": existing.dataset_id, "version_id": child_id, "asset_count": existing.asset_count}
    parents = [db.get(DatasetVersion, value) for value in payload["version_ids"]]
    if any(parent is None or parent.deleted_at or parent.state != "frozen" for parent in parents):
        raise ApiError(409, "parent_not_frozen", "来源版本需要保持冻结且可用")
    first = parents[0]
    if any(parent.class_map_json != first.class_map_json for parent in parents):
        raise ApiError(422, "class_map_mismatch", "来源版本类别映射必须一致")
    if any(parent.split_config_json.get("group_level") != first.split_config_json.get("group_level") for parent in parents):
        raise ApiError(422, "group_level_mismatch", "来源版本分组级别必须一致")
    resources = [db.get(DatasetResource, parent.dataset_id) for parent in parents]
    if any(resource is None or resource.deleted_at for resource in resources):
        raise ApiError(409, "source_deleted", "来源数据集已删除")
    dataset_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "bysj:merge:" + ctx.job.id)) if operation == "merge" else first.dataset_id
    assets = db.query(DatasetAsset).filter(DatasetAsset.dataset_version_id.in_(payload["version_ids"]), DatasetAsset.deleted_at.is_(None)).order_by(DatasetAsset.id).all()
    variants = [("rotate", angle) for angle in payload["rotations"]]
    variants += [("brightness", factor) for factor in payload["brightness"]]
    variants += [("contrast", factor) for factor in payload["contrast"]]
    if payload["flip"]:
        variants.append(("flip", True))
    if payload["grayscale"]:
        variants.append(("grayscale", True))
    count = len(assets) + sum(asset.split == "train" for asset in assets) * (len(variants) if operation == "augment" else 0)
    if count > settings.max_zip_files:
        raise ApiError(422, "processing_limit", "派生版本图片数量超出上限")
    parent_map = {parent.id: parent for parent in parents}
    stage_root = ensure_dir(settings.data_root / "staging" / "derivatives" / ctx.job.id)
    stage = stage_root / str(ctx.attempt)
    stage.mkdir(exist_ok=False)
    final = resolve_under(settings.data_root, f"datasets/{dataset_id}/{child_id}")
    entries, fixed, lineage = [], {}, {}
    size = 0
    adapter = handle = ownership = None

    def append(path, relative, asset, boxes, transform_data):
        nonlocal size
        size += path.stat().st_size
        if size > settings.max_extract_bytes:
            raise ApiError(413, "processing_limit", "派生产物体积超出上限")
        with Image.open(path) as image:
            width, height = image.size
        entries.append({"relative_path": relative, "sha256": sha256_file(path), "size": path.stat().st_size,
                        "width": width, "height": height, "source_group": asset.source_group,
                        "root_asset": str(asset.root_asset_id or asset.id), "annotations": boxes})
        fixed[relative] = asset.split
        lineage[relative] = (asset, boxes, transform_data)
        label = resolve_under(stage / "labels", str(Path(relative).with_suffix(".txt")))
        body = "".join(f"{box['class_id']} {box['cx']:.6f} {box['cy']:.6f} {box['w']:.6f} {box['h']:.6f}\n" for box in boxes)
        atomic_write_bytes(label, body.encode())

    try:
        if operation == "preannotate":
            model = db.get(ModelRegistry, payload["model_id"])
            if model is None or model.status != "active" or model.model_task != "detect":
                raise ApiError(409, "model_unavailable", "检测模型不可用")
            path = verify_model_file(model, settings)
            adapter = get_adapter(settings)
            ownership = acquire(ctx)
            handle = adapter.load(path, model.class_map_json)
        for index, asset in enumerate(assets):
            ctx.check_cancelled()
            parent = parent_map[asset.dataset_version_id]
            source_root = Path(parent.root_path).resolve()
            if source_root == settings.data_root.resolve() or not source_root.is_relative_to(settings.data_root.resolve()):
                raise ApiError(422, "invalid_path", "来源版本目录越界")
            source = resolve_under(source_root / "images", asset.relative_path, must_exist=True)
            if sha256_file(source) != asset.content_hash:
                raise ApiError(409, "source_tampered", "来源图片摘要不匹配")
            relative = f"{parent.id}/{asset.id}/{Path(asset.relative_path).name}"
            target = resolve_under(stage / "images", relative)
            ensure_dir(target.parent)
            shutil.copy2(source, target)
            boxes = copy.deepcopy(asset.annotations_json or [])
            selected = not payload["asset_ids"] or asset.id in payload["asset_ids"]
            if operation == "preannotate" and selected:
                proposed = _predictions(adapter, handle, source, asset, payload, first.class_map_json)
                boxes = boxes + proposed if payload["annotation_mode"] == "append" else proposed
            append(target, relative, asset, boxes, {"kind": operation, "identity_image": True})
            if operation == "augment" and asset.split == "train":
                with Image.open(source) as original:
                    image = original.convert("RGB")
                    for variant_index, (kind, value) in enumerate(variants):
                        ctx.check_cancelled()
                        transformed, transformed_boxes = transform(image.copy(), boxes, kind, value)
                        derived_relative = f"augmented/{asset.id}/{variant_index}.png"
                        derived_path = resolve_under(stage / "images", derived_relative)
                        ensure_dir(derived_path.parent)
                        transformed.save(derived_path, format="PNG")
                        append(derived_path, derived_relative, asset, transformed_boxes, {"kind": kind, "value": value})
            ctx.progress(.85 * (index + 1) / max(1, len(assets)), "processing", f"处理 {index + 1}/{len(assets)}")
        config = first.split_config_json
        ratios = config["ratios"]
        manifest, stats = build_manifest(entries, class_map=first.class_map_json, seed=config["seed"],
            ratios=[ratios[key] for key in ("train", "val", "test")], dataset_id=dataset_id, version_id=child_id,
            parent_version_id=first.id, group_level=config.get("group_level", "original_image"), fixed_splits=fixed)
        report = integrity_report(manifest)
        if not report["ok"]:
            raise ApiError(409, "split_leakage", "合并后的来源组、原图或内容跨越不同数据集划分", data=report)
        atomic_write_bytes(stage / "manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode())
        ctx.lock_for_write()
        if final.exists():
            expected = {path.relative_to(stage).as_posix(): sha256_file(path) for path in stage.rglob("*") if path.is_file()}
            actual = {path.relative_to(final).as_posix(): sha256_file(path) for path in final.rglob("*") if path.is_file()}
            if expected != actual:
                raise ApiError(409, "artifact_conflict", "重试产物与已有文件不一致")
        else:
            ensure_dir(final.parent)
            stage.rename(final)
        if operation == "merge":
            db.add(DatasetResource(id=dataset_id, dataset_id=dataset_id, name=payload["name"],
                owner_user_id=ctx.job.owner_user_id, owner_username=ctx.job.owner_username,
                task_type="detect", group_level=resources[0].group_level,
                license="；".join(sorted({resource.license or "未填写" for resource in resources}))))
        child = DatasetVersion(id=child_id, dataset_id=dataset_id, parent_version_id=first.id,
            dataset_role=first.dataset_role, status="ready", state="draft" if operation == "preannotate" else "frozen",
            frozen_at=None if operation == "preannotate" else utcnow(), root_path=str(final),
            manifest_path=str(final / "manifest.json"), class_map_json=first.class_map_json,
            split_config_json=copy.deepcopy(config), split_counts_json=stats["split_counts"],
            content_hash=stats["manifest_sha256"], manifest_sha256=stats["manifest_sha256"],
            split_manifest_hash=stats["split_manifest_hash"], class_map_hash=stats["class_map_hash"],
            asset_count=len(entries), created_by=ctx.job.owner_username, created_by_user_id=ctx.job.owner_user_id,
            metadata_json={**copy.deepcopy(first.metadata_json or {}),
                           "job_id": ctx.job.id, "operation": operation, "parent_version_ids": payload["version_ids"],
                           "group_level": config.get("group_level", resources[0].group_level),
                           "preserve_parent_splits": True,
                           "synthetic": any((parent.metadata_json or {}).get("synthetic") for parent in parents) or bool(adapter and adapter.backend == "dummy"),
                           "research_valid": all((parent.metadata_json or {}).get("research_valid") is True for parent in parents) and not bool(adapter and adapter.backend == "dummy"),
                           "ai_backend": adapter.backend if adapter else None, "requires_annotation_review": operation == "preannotate"})
        db.add(child)
        db.flush()
        for item in manifest["assets"]:
            parent_asset, boxes, transform_data = lineage[item["relative_path"]]
            db.add(DatasetAsset(dataset_version_id=child_id, relative_path=item["relative_path"],
                basename=Path(item["relative_path"]).name, size=item["size"], width=item["width"], height=item["height"],
                content_hash=item["sha256"], annotations_json=boxes, source_group=item["source_group"],
                split=item["split"], group_level=parent_asset.group_level,
                label_path="labels/" + str(Path(item["relative_path"]).with_suffix(".txt")),
                root_asset_id=parent_asset.root_asset_id or parent_asset.id, original_asset_id=parent_asset.id,
                transform_json=transform_data))
        db.commit()
        return {"dataset_id": dataset_id, "version_id": child_id, "asset_count": child.asset_count, "state": child.state}
    except Exception:
        db.rollback()
        raise
    finally:
        if adapter and handle:
            adapter.release(handle)
        if ownership:
            release(ctx, ownership)
        if stage.exists():
            remove_tree(stage, allowed_root=stage_root)
