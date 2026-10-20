"""Bounded, transactional transfer of durable AI results into an owned draft."""

from copy import deepcopy
from pathlib import Path

from database import AnnotationDraftAsset, DatasetAsset, JobEvent, JobTask
from annotation_drafts import sanitize_draft_asset

BATCH_SIZE = 200
TERMINAL_STATUSES = {"completed", "cancelled", "failed"}


def validate_batch_task(task, draft, version, working_version):
    data = task.input_json or {}
    if (task.type != "annotation" or str(task.owner_id) != str(draft.owner_user_id)
            or data.get("annotation_mode") not in {"append", "replace"}
            or str(data.get("annotation_draft_id") or "") != str(draft.id)
            or str(data.get("dataset_id") or "") != str(draft.dataset_id)
            or str(data.get("annotation_draft_source_version_id", data.get("source_version_id")) or "") != str(draft.source_version_id)
            or str(data.get("source_version_id") or "") != str(working_version.id)
            or str(data.get("annotation_draft_model_path", data.get("model_path")) or "") != str(draft.model_path)
            or list(data.get("annotation_classes") or []) != list(draft.class_names_json or [])
            or list(data.get("annotation_draft_class_bindings", data.get("annotation_class_bindings")) or []) != list(draft.class_bindings_json or [])
            or str(version.id) != str(draft.source_version_id)):
        raise ValueError("AI 任务与当前草稿、用户、模型或类别版本不匹配")


def latest_draft_batch(db, draft):
    task = (db.query(JobTask).filter(
        JobTask.type == "annotation", JobTask.owner_id == str(draft.owner_user_id),
        JobTask.input_json["annotation_draft_id"].as_string() == str(draft.id),
        JobTask.input_json["annotation_mode"].as_string().in_(["append", "replace"]),
    ).order_by(JobTask.created_at.desc(), JobTask.id.desc()).first())
    if task is None:
        return None
    return {"task_id": task.id, "status": task.status,
            "annotation_mode": (task.input_json or {}).get("annotation_mode", "replace"),
            "synced": bool((task.result_json or {}).get("draft_sync", {}).get("complete"))}


def _result_asset(db, version, item):
    root = Path(version.root_path).resolve()
    relative = str(item.get("relative_path") or "").replace("\\", "/")
    if not relative:
        for key in ("asset_key", "path"):
            try:
                relative = Path(str(item.get(key) or "")).resolve().relative_to(root).as_posix()
                break
            except ValueError:
                continue
    if relative:
        asset = db.query(DatasetAsset).filter(
            DatasetAsset.dataset_version_id == version.id,
            DatasetAsset.relative_path == relative,
        ).first()
        if asset is not None:
            return asset
    # Never guess when two subdirectories contain the same filename.
    matches = db.query(DatasetAsset).filter(
        DatasetAsset.dataset_version_id == version.id,
        DatasetAsset.basename == str(item.get("filename") or item.get("name") or ""),
    ).limit(2).all()
    if len(matches) != 1:
        raise ValueError("AI 结果图片不存在或文件名不唯一")
    return matches[0]


def _generated_shape(annotation, draft):
    result = deepcopy(annotation)
    class_id = int(result.get("page_class_id", result.get("class", -1)))
    if class_id < 0 or class_id >= len(draft.class_names_json or []):
        raise ValueError("AI 结果类别不属于当前草稿")
    result["class"] = class_id
    points = result.get("points") or []
    if points and isinstance(points[0], (list, tuple)):
        points = [{"x": p[0], "y": p[1]} for p in points]
    if draft.annotation_geometry == "polygon":
        if not points:
            x, y, w, h = (float(result.get(key, 0)) for key in ("x", "y", "width", "height"))
            points = [{"x": x, "y": y}, {"x": x + w, "y": y},
                      {"x": x + w, "y": y + h}, {"x": x, "y": y + h}]
        result.update(type="polygon", points=points)
    else:
        if points:
            xs, ys = [p["x"] for p in points], [p["y"] for p in points]
            result.update(x=min(xs), y=min(ys), width=max(xs)-min(xs), height=max(ys)-min(ys))
        result.pop("points", None)
        result.pop("polygon", None)
        result.pop("segmentation", None)
        result["type"] = "rect"
    return result


def _source_shape(annotation):
    result = deepcopy(annotation)
    line = str(result.pop("yolo_line", "") or "").split()
    if line:
        result["class"] = int(line[0])
        values = [float(value) for value in line[1:]]
        if len(values) == 4:
            x, y, w, h = values
            result.update(type="rect", x=x-w/2, y=y-h/2, width=w, height=h)
        elif len(values) >= 6 and len(values) % 2 == 0:
            result.update(type="polygon", points=[{"x": x, "y": y} for x, y in zip(values[::2], values[1::2])])
        else:
            raise ValueError("原始标注坐标无效")
    return result


def import_draft_batch(db, draft, source_version, working_version, task_id, *,
                       inactive_ids=(), override_paths=()):
    """Caller holds the draft lock; checkpoint and asset writes commit together."""
    task = db.query(JobTask).filter(JobTask.id == str(task_id)).with_for_update().first()
    if task is None:
        raise ValueError("AI 标注任务不存在")
    validate_batch_task(task, draft, source_version, working_version)
    context = getattr(draft, "class_context_json", None) or {}
    isolated = context.get("scope") == "model_session"
    if isolated:
        inactive_ids = context.get("inactive_class_ids") or []
    if task.status not in TERMINAL_STATUSES:
        raise ValueError("AI 标注任务尚未结束")
    state = dict((task.result_json or {}).get("draft_sync") or {})
    if state and state.get("draft_id") != str(draft.id):
        raise ValueError("AI 结果已用于另一草稿")
    after = int(state.get("after_sequence") or 0)
    events = (db.query(JobEvent).filter(
        JobEvent.task_id == task.id, JobEvent.event_type == "annotation_result",
        JobEvent.sequence > after,
    ).order_by(JobEvent.sequence.asc()).limit(BATCH_SIZE + 1).all())
    has_more = len(events) > BATCH_SIZE
    for event in events[:BATCH_SIZE]:
        item = (event.payload_json or {}).get("item") or {}
        after = event.sequence
        if item.get("status") == "failed":
            continue
        asset = _result_asset(db, working_version, item)
        if asset.relative_path in override_paths:
            continue
        row = db.query(AnnotationDraftAsset).filter(
            AnnotationDraftAsset.draft_id == draft.id,
            AnnotationDraftAsset.relative_path == asset.relative_path,
        ).first()
        if row is not None and task.finished_at and row.updated_at and row.updated_at > task.finished_at:
            continue  # Explicit edits saved after inference take precedence.
        generated = [_generated_shape(ann, draft) for ann in item.get("annotations") or []
                     if int(ann.get("page_class_id", ann.get("class", -1))) not in inactive_ids]
        if (task.input_json or {}).get("annotation_mode") == "append":
            baseline = [_source_shape(ann) for ann in
                        (row.annotations_json if row is not None else ([] if isolated else asset.annotations_json or []))]
            baseline = [ann for ann in baseline
                        if int(ann.get("class", ann.get("class_id", -1))) not in inactive_ids]
            generated = [*baseline, *generated]
        checked = sanitize_draft_asset({"relative_path": asset.relative_path,
            "width": asset.width, "height": asset.height, "annotations": generated})
        if row is None:
            row = AnnotationDraftAsset(draft_id=draft.id, relative_path=asset.relative_path)
            db.add(row)
        row.width, row.height = checked["width"], checked["height"]
        row.annotations_json = checked["annotations"]
    state = {"draft_id": str(draft.id), "after_sequence": after, "complete": not has_more}
    task.result_json = {**(task.result_json or {}), "draft_sync": state}
    db.flush()
    return {"has_more": has_more, "after_sequence": after}
