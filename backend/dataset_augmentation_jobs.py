"""Persistent whole-dataset augmentation job execution."""

from __future__ import annotations

import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from database import AnnotationDraft, AnnotationDraftAsset, DatasetAsset, DatasetResource, DatasetVersion, JobTask, SessionLocal
from annotation_session_classes import draft_dataset_view, isolated_draft, session_base_annotations, manifest_frontend_annotations
from annotation_export import frontend_annotation_to_manifest
from dataset_versioning import annotation_to_yolo_line, create_initial_version
from job_store import is_cancel_requested, update_job
from image_augmentation import (
    apply_image_operation,
    augmented_filename,
    normalize_augmentation_operations,
    transform_annotations,
)
from utils import constants


logger = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATA_ROOT = Path(os.getenv("DEEPVISION_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))).resolve()


def _build_unique_output_dir(name: str) -> Path:
    safe_name = re.sub(r'[\\/:*?"<>|]+', "_", str(name or "").strip()).strip(" .") or "augmented-dataset"
    root = Path(constants.DATASETS_DIR).resolve()
    candidate = root / safe_name
    index = 1
    while candidate.exists():
        candidate = root / f"{safe_name}_{index}"
        index += 1
    return candidate


class DatasetAugmentationCancelled(Exception):
    pass


def _safe_relative(root: Path, relative_path: str) -> Path:
    candidate = (root / str(relative_path).replace("\\", "/")).resolve()
    candidate.relative_to(root.resolve())
    return candidate


def _job_workspace(source_root: Path, task_id: str) -> tuple[Path, Path, Path]:
    """Return a hidden workspace on the same mount as the managed dataset."""
    datasets_root = Path(constants.DATASETS_DIR).resolve()
    resolved_source = source_root.resolve()
    resolved_source.relative_to(datasets_root)
    if resolved_source.parent != datasets_root:
        raise ValueError("原始数据集目录必须位于数据集根目录下")
    workspace = datasets_root / f".dataset-augmentation-{task_id}"
    return workspace, workspace / "stage", workspace / "backup"


def _cleanup_workspace(workspace: Path, backup: Path) -> None:
    """Remove transient files but retain any backup that still needs recovery."""
    if workspace.exists() and not backup.exists():
        shutil.rmtree(workspace, ignore_errors=True)


def _target_label_relative(image_relative: Path, source_asset: DatasetAsset, *, force: bool = False) -> Path | None:
    if not source_asset.label_path and not force:
        return None
    parts = image_relative.parts
    if len(parts) >= 3 and parts[0] in {"train", "val", "test"} and parts[1] == "images":
        return Path(parts[0]) / "labels" / Path(*parts[2:]).with_suffix(".txt")
    return image_relative.with_suffix(".txt")


def _normalize_relative_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip().lstrip("./")


def _select_source_assets(source_assets: List[DatasetAsset], payload: Dict[str, Any]) -> List[DatasetAsset]:
    requested = payload.get("source_relative_paths")
    if requested is None:
        return [asset for asset in source_assets if "__aug_" not in Path(asset.relative_path).stem]
    if not isinstance(requested, list) or not requested:
        raise ValueError("指定增强图片列表不能为空")
    normalized = list(dict.fromkeys(_normalize_relative_path(value) for value in requested))
    if any(not value for value in normalized):
        raise ValueError("指定增强图片缺少资产路径")
    assets_by_path = {_normalize_relative_path(asset.relative_path): asset for asset in source_assets}
    missing = [value for value in normalized if value not in assets_by_path]
    if missing:
        raise ValueError(f"指定增强图片不是数据集资产: {missing[0]}")
    return [assets_by_path[value] for value in normalized]


def _annotation_override_map(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    raw_overrides = payload.get("annotation_overrides") or []
    if not isinstance(raw_overrides, list):
        raise ValueError("增强标注覆盖格式不正确")
    result: Dict[str, List[Dict[str, Any]]] = {}
    for item in raw_overrides:
        if not isinstance(item, dict):
            raise ValueError("增强标注覆盖包含非对象条目")
        relative_path = _normalize_relative_path(item.get("relative_path"))
        annotations = item.get("annotations")
        if not relative_path or not isinstance(annotations, list):
            raise ValueError("增强标注覆盖缺少资产路径或标注列表")
        result[relative_path] = annotations
    return result


def _write_label(target: Path, annotations: List[Dict[str, Any]]) -> None:
    lines = [annotation_to_yolo_line(annotation) for annotation in annotations]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_transformed_label(
    target: Path,
    annotations: List[Dict[str, Any]],
    width: int,
    height: int,
    operation: Dict[str, Any],
) -> None:
    transformed = transform_annotations(annotations, width, height, operation)
    _write_label(target, transformed)


def _update_progress(task_id: str, *, stage: str, progress: float, message: str, processed: int, total: int) -> None:
    db = SessionLocal()
    try:
        current = db.query(JobTask).filter(JobTask.id == task_id).first()
        if current is None or current.status in {"completed", "failed", "cancelled"}:
            return
        update_job(
            db,
            task_id,
            stage=stage,
            progress=progress,
            message=message,
            worker_pid=os.getpid(),
            event_payload={"processed_images": processed, "total_images": total},
        )
    finally:
        db.close()


def _copy_metadata(source_version: DatasetVersion, target_version: DatasetVersion, report: Dict[str, Any]) -> None:
    metadata = dict(source_version.metadata_json or {})
    statistics = report.get("statistics") or {}
    metadata.update({
        "version_id": target_version.id,
        "path": target_version.root_path,
        "storage_path": str(Path(target_version.root_path).resolve().relative_to(Path(constants.DATASETS_DIR).resolve())).replace("\\", "/"),
        "updated_at": datetime.now().isoformat(),
        "image_count": int(statistics.get("images") or 0),
        "label_count": int(statistics.get("labels") or 0),
        "size": sum(int(asset.get("size") or 0) for asset in report.get("assets") or []),
    })
    target_version.metadata_json = metadata


def execute_dataset_augmentation_job(task_id: str) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        task = db.query(JobTask).filter(JobTask.id == task_id, JobTask.type == "dataset_augmentation").first()
        if task is None:
            raise KeyError(f"数据增强任务不存在: {task_id}")
        payload = dict(task.input_json or {})
        dataset_id = str(payload.get("dataset_id") or "").strip()
        source_version_id = str(payload.get("source_version_id") or "").strip()
        create_new_dataset = bool(payload.get("create_new_dataset"))
        temporary_dataset = bool(payload.get("temporary_dataset"))
        original_dataset_id = str(payload.get("original_dataset_id") or dataset_id).strip()
        output_name = str(payload.get("output_name") or "").strip()
        source_version = db.query(DatasetVersion).filter(
            DatasetVersion.id == source_version_id,
            DatasetVersion.dataset_id == dataset_id,
            DatasetVersion.dataset_role == "original",
        ).first()
        if not source_version:
            raise ValueError("原始数据集版本不存在")
        session_draft = None
        draft_annotations = {}
        draft_id = str(payload.get("annotation_draft_id") or "").strip()
        if draft_id:
            session_draft = db.query(AnnotationDraft).filter(
                AnnotationDraft.id == draft_id,
                AnnotationDraft.dataset_id == original_dataset_id,
                AnnotationDraft.owner_user_id == payload.get("annotation_draft_owner_id"),
            ).first()
            if session_draft is None or str(session_draft.source_version_id) != str(payload.get("draft_source_version_id")):
                raise ValueError("标注草稿不存在或来源版本已变化")
            if payload.get("annotation_draft_revision") is not None and int(session_draft.revision or 0) != int(payload['annotation_draft_revision']):
                raise ValueError("标注草稿在增强前已变化，请重新启动增强")
            if isolated_draft(session_draft):
                source_version = draft_dataset_view(source_version, session_draft)
                for row in db.query(AnnotationDraftAsset).filter(AnnotationDraftAsset.draft_id == draft_id).all():
                    draft_annotations[row.relative_path] = [frontend_annotation_to_manifest(
                        item, image_width=row.width, image_height=row.height) for item in row.annotations_json or []]
        session_isolated = isolated_draft(session_draft)
        source_metadata = dict(source_version.metadata_json or {})
        if temporary_dataset and not create_new_dataset:
            if (
                source_metadata.get("source_type") != "annotation_temporary"
                or str(source_metadata.get("source_dataset_id") or "") != original_dataset_id
            ):
                raise ValueError("草稿增强只能更新当前草稿的临时数据集")
        source_assets = db.query(DatasetAsset).filter(
            DatasetAsset.dataset_version_id == source_version.id,
        ).order_by(DatasetAsset.relative_path.asc()).all()
        if session_isolated:
            for asset in source_assets:
                draft_annotations.setdefault(asset.relative_path, session_base_annotations(source_version, asset.annotations_json))
    finally:
        db.close()

    operations = normalize_augmentation_operations(payload.get("config") or {})
    source_root = Path(source_version.root_path).resolve()
    workspace, stage, backup = _job_workspace(source_root, task_id)
    output_path: Path | None = None
    target_dataset_id = dataset_id
    target_root = source_root
    if create_new_dataset:
        if not output_name:
            raise ValueError("增强数据集名称不能为空")
        output_path = _build_unique_output_dir(output_name)
        target_dataset_id = output_path.name
        target_root = output_path
    if backup.exists():
        raise RuntimeError(f"检测到待恢复的数据集备份，已停止任务: {backup}")
    if not source_root.is_dir():
        raise ValueError("原始数据集目录不存在")

    all_source_assets = list(source_assets)
    source_assets = _select_source_assets(source_assets, payload)
    if not source_assets:
        raise ValueError("原始数据集中没有可增强的图片")
    annotation_overrides = _annotation_override_map(payload)
    selected_paths = {_normalize_relative_path(asset.relative_path) for asset in source_assets}
    unknown_override = next((path for path in annotation_overrides if path not in selected_paths), None)
    if unknown_override:
        raise ValueError(f"增强标注覆盖不是本次选中资产: {unknown_override}")

    total = len(source_assets) * len(operations)
    processed = 0
    created = 0
    copied_files = 0

    def copy_source_file(source: str, target: str):
        nonlocal copied_files
        if is_cancel_requested(task_id):
            raise DatasetAugmentationCancelled("数据增强任务已取消")
        result = shutil.copy2(source, target)
        copied_files += 1
        if copied_files % 100 == 0:
            _update_progress(
                task_id,
                stage="preparing",
                progress=1.0,
                message=f"正在准备数据集文件（已复制 {copied_files} 个）",
                processed=0,
                total=total,
            )
        return result

    try:
        shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True, exist_ok=False)
        _update_progress(task_id, stage="preparing", progress=0.0, message="正在准备数据集副本", processed=0, total=total)
        shutil.copytree(source_root, stage, copy_function=copy_source_file)
        if session_isolated:
            import yaml
            for asset in all_source_assets:
                source_label = _target_label_relative(Path(asset.relative_path), asset, force=True)
                if source_label:
                    _write_label(stage / source_label, draft_annotations.get(asset.relative_path, []))
            for name in ['data.yaml', 'dataset.yaml']:
                config_path = stage / name
                if name == 'dataset.yaml' or config_path.is_file():
                    config = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
                    config = config if isinstance(config, dict) else {}
                    config.update({'names': source_version.class_map_json, 'nc': len(source_version.class_map_json)})
                    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        _update_progress(task_id, stage="preparing", progress=5.0, message="数据集副本准备完成", processed=0, total=total)
        for asset in source_assets:
            if is_cancel_requested(task_id):
                raise DatasetAugmentationCancelled("数据增强任务已取消")
            source_image = _safe_relative(source_root, asset.relative_path)
            if source_image.suffix.lower() not in IMAGE_SUFFIXES or not source_image.is_file():
                processed += len(operations)
                continue
            with Image.open(source_image) as opened:
                source_image_data = opened.convert("RGBA" if "A" in opened.getbands() else "RGB")
            width, height = int(source_image_data.width), int(source_image_data.height)
            source_relative = Path(asset.relative_path)
            source_key = _normalize_relative_path(asset.relative_path)
            source_annotations = annotation_overrides.get(source_key,
                draft_annotations.get(source_key, []) if session_isolated else asset.annotations_json or [])
            if source_key in annotation_overrides:
                source_label = _target_label_relative(source_relative, asset, force=True)
                if source_label:
                    _write_label(stage / source_label, source_annotations)
            for operation in operations:
                if is_cancel_requested(task_id):
                    raise DatasetAugmentationCancelled("数据增强任务已取消")
                target_relative = source_relative.with_name(augmented_filename(source_relative.name, operation))
                target = stage / target_relative
                if not target.exists():
                    transformed_image = apply_image_operation(source_image_data, operation)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    transformed_image.save(target, format="PNG", optimize=True)
                    transformed_image.close()
                    target_label = _target_label_relative(
                        target_relative,
                        asset,
                        force=source_key in annotation_overrides or bool(source_annotations),
                    )
                    if target_label:
                        if session_isolated:
                            transformed = transform_annotations(manifest_frontend_annotations(source_annotations), width, height, operation)
                            _write_label(stage / target_label, [frontend_annotation_to_manifest(item) for item in transformed])
                        else:
                            _write_transformed_label(target=stage / target_label, annotations=source_annotations, width=width, height=height, operation=operation)
                    created += 1
                processed += 1
                if processed == total or processed % 20 == 0:
                    _update_progress(
                        task_id,
                        stage="augmenting",
                        progress=5.0 + processed / max(total, 1) * 90.0,
                        message=f"正在增强图片 {processed}/{total}",
                        processed=processed,
                        total=total,
                    )
            source_image_data.close()

        if is_cancel_requested(task_id):
            raise DatasetAugmentationCancelled("数据增强任务已取消")

        # Commit the staged tree only after every generated file is complete.
        if create_new_dataset:
            os.replace(stage, output_path)
        else:
            os.replace(source_root, backup)
            os.replace(stage, source_root)
        commit_complete = False
        recovery_complete = False
        created_version_id = None
        created_manifest_path = None
        created_version_record = False
        try:
            db = SessionLocal()
            try:
                annotation_draft_id = str(payload.get("annotation_draft_id") or "").strip()
                draft = None
                if annotation_draft_id:
                    draft = db.query(AnnotationDraft).filter(
                        AnnotationDraft.id == annotation_draft_id,
                        AnnotationDraft.dataset_id == original_dataset_id,
                        AnnotationDraft.owner_user_id == payload.get("annotation_draft_owner_id"),
                    ).with_for_update().first()
                    if draft is None:
                        raise ValueError("标注草稿不存在或不属于当前用户")
                    if str(draft.source_version_id or "") != str(payload.get("draft_source_version_id") or ""):
                        raise ValueError("标注草稿对应的数据集版本已变化")
                    if payload.get("annotation_draft_revision") is not None and int(draft.revision or 0) != int(payload['annotation_draft_revision']):
                        raise ValueError("标注草稿在增强过程中已变化，已取消提交")

                version, report, created_version_record = create_initial_version(
                    db,
                    dataset_id=target_dataset_id,
                    root_path=target_root,
                    dataset_role="original",
                    manifest_root=DATA_ROOT / "manifests",
                    created_by=str(source_version.created_by or "system"),
                )
                if created_version_record:
                    created_version_id = version.id
                    created_manifest_path = str(version.manifest_path or "")
                _copy_metadata(source_version, version, report)
                metadata = dict(version.metadata_json or {})
                metadata.update({
                    "id": target_dataset_id,
                    "name": target_dataset_id if create_new_dataset else metadata.get("name", dataset_id),
                    "path": str(target_root),
                    "source_type": "annotation_temporary" if temporary_dataset else "augmentation",
                    "temporary": temporary_dataset,
                    "source_dataset_id": original_dataset_id,
                    "augmentation_operations": payload.get("config") or {},
                })
                version.metadata_json = metadata
                if create_new_dataset:
                    db.add(DatasetResource(
                        dataset_id=target_dataset_id,
                        dataset_role="original",
                        owner_user_id=payload.get("owner_id"),
                        owner_username=str(payload.get("owner_username") or "system"),
                        owner_role_id=payload.get("owner_role_id"),
                    ))
                if draft is not None:
                    draft.temporary_dataset_id = target_dataset_id
                    draft.revision = int(draft.revision or 0) + 1
                    draft.updated_at = datetime.now()
                db.commit()
                commit_complete = True
                result = {
                    "dataset_id": target_dataset_id,
                    "dataset_info": {
                        "id": target_dataset_id,
                        "name": target_dataset_id,
                        "path": str(target_root),
                        "temporary": temporary_dataset,
                        "source_type": "annotation_temporary" if temporary_dataset else "augmentation",
                    },
                    "source_version_id": source_version.id,
                    "version_id": version.id,
                    "source_images": len(source_assets),
                    "operations": len(operations),
                    "created_images": created,
                    "total_images": int(report.get("statistics", {}).get("images") or 0),
                }
                if annotation_draft_id:
                    result.update({
                        "annotation_draft_id": annotation_draft_id,
                        "draft_revision": int(draft.revision or 0),
                        "temporary_dataset_id": target_dataset_id,
                    })
            finally:
                db.close()
        except Exception:
            if not commit_complete:
                if created_version_record and created_version_id:
                    cleanup_db = SessionLocal()
                    try:
                        cleanup_db.query(DatasetAsset).filter(
                            DatasetAsset.dataset_version_id == created_version_id,
                        ).delete(synchronize_session=False)
                        cleanup_db.query(DatasetVersion).filter(
                            DatasetVersion.id == created_version_id,
                        ).delete(synchronize_session=False)
                        cleanup_db.commit()
                    except Exception:
                        cleanup_db.rollback()
                        logger.exception("清理失败的增强版本索引失败: version=%s", created_version_id)
                    finally:
                        cleanup_db.close()
                    if created_manifest_path:
                        Path(created_manifest_path).unlink(missing_ok=True)
                if create_new_dataset:
                    if output_path and output_path.exists():
                        shutil.rmtree(output_path)
                else:
                    if source_root.exists():
                        shutil.rmtree(source_root)
                    if backup.exists():
                        os.replace(backup, source_root)
                recovery_complete = True
            raise
        finally:
            if commit_complete or recovery_complete:
                shutil.rmtree(backup, ignore_errors=True)

        _update_progress(
            task_id,
            stage="completed",
            progress=100.0,
            message=f"数据增强完成，新增 {created} 张图片",
            processed=total,
            total=total,
        )
        result_db = SessionLocal()
        try:
            update_job(
                result_db,
                task_id,
                status="completed",
                stage="completed",
                progress=100.0,
                message=f"数据增强完成，新增 {created} 张图片",
                result=result,
                worker_pid=os.getpid(),
                event_type="completed",
                event_payload=result,
            )
        finally:
            result_db.close()
        return result
    except DatasetAugmentationCancelled:
        shutil.rmtree(stage, ignore_errors=True)
        cancel_db = SessionLocal()
        try:
            update_job(cancel_db, task_id, status="cancelled", stage="cancelled", message="数据增强任务已取消", worker_pid=os.getpid(), event_type="cancelled")
        finally:
            cancel_db.close()
        raise
    except Exception as exc:
        logger.exception("数据增强任务失败: task_id=%s", task_id)
        shutil.rmtree(stage, ignore_errors=True)
        failed_db = SessionLocal()
        try:
            update_job(
                failed_db,
                task_id,
                status="failed",
                stage="failed",
                message=f"数据增强失败: {exc}",
                error_code="dataset_augmentation_failed",
                error_detail=str(exc),
                worker_pid=os.getpid(),
                event_type="failed",
            )
        finally:
            failed_db.close()
        raise
    finally:
        _cleanup_workspace(workspace, backup)
