"""Resolve the one model that a completed training task may test.

The UI must never select a training result by project name or by the latest
model.  This module keeps that rule in one place for list/status responses and
for inference requests.
"""

from pathlib import Path

from fastapi import HTTPException

from database import ModelRegistry


_UNAVAILABLE = {
    "available": False,
    "model_id": None,
    "display_name": None,
    "reason_code": "model_unavailable",
    "reason": "训练模型当前不可用",
}


def _unavailable(code, reason):
    return {**_UNAVAILABLE, "reason_code": code, "reason": reason}


def _result(task):
    value = getattr(task, "result_json", None) or {}
    return value if isinstance(value, dict) else {}


def _eligible(record, task_id):
    return bool(
        record
        and str(record.source or "").lower() == "trained"
        and str(record.status or "").lower() == "active"
        and str(record.task_id or "") == str(task_id)
        and Path(str(record.path or "")).suffix.lower() == ".pt"
    )


def _describe(record, resolve_path):
    try:
        resolved = resolve_path(record.path, record, require_file=True)
        if resolved.stat().st_size <= 0:
            return _unavailable("invalid_artifact", "训练模型文件为空")
    except HTTPException:
        return _unavailable("model_missing", "训练模型文件不存在或不可访问")
    except OSError:
        return _unavailable("model_missing", "训练模型文件不存在或不可访问")
    metadata = record.metadata_json or {}
    return {
        "available": True,
        "model_id": record.id,
        "display_name": metadata.get("display_name") or record.name,
        "reason_code": None,
        "reason": None,
    }


def _describe_unregistered_path(model_path, resolve_path):
    try:
        resolved = resolve_path(model_path, None, require_file=True)
        if resolved.suffix.lower() != ".pt" or resolved.stat().st_size <= 0:
            return _unavailable("invalid_artifact", "训练模型文件无效")
    except HTTPException:
        return _unavailable("model_missing", "训练模型文件不存在或不可访问")
    except OSError:
        return _unavailable("model_missing", "训练模型文件不存在或不可访问")
    return {
        "available": True,
        "model_id": None,
        "display_name": resolved.name,
        "reason_code": None,
        "reason": None,
    }


def describe_training_test_model(task, db, resolve_path, prefetched_records=None):
    """Return a safe test-model descriptor for one visible training task."""
    if str(task.status or "") != "completed":
        return _unavailable("task_not_completed", "训练尚未成功完成")

    result = _result(task)
    model_id = str(result.get("model_registry_id") or "").strip()
    records = list(prefetched_records) if prefetched_records is not None else None
    if model_id:
        record = (
            next((item for item in records if str(item.id) == model_id), None)
            if records is not None
            else db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
        )
        if not _eligible(record, task.id):
            return _unavailable("model_unavailable", "训练模型已失效或不属于当前任务")
        return _describe(record, resolve_path)

    if records is None:
        records = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.task_id == task.id)
            .filter(ModelRegistry.source == "trained")
            .filter(ModelRegistry.status == "active")
            .all()
        )
    task_records = [item for item in records if _eligible(item, task.id)]
    if len(task_records) > 1:
        return _unavailable("ambiguous_model", "当前训练任务存在多个可用模型")
    if len(task_records) == 1:
        return _describe(task_records[0], resolve_path)

    model_path = str(result.get("model_path") or "").strip()
    if not model_path:
        return _unavailable("model_missing", "训练未生成可测试模型")
    owners = (
        [item for item in records if str(item.path or "") == model_path]
        if prefetched_records is not None
        else db.query(ModelRegistry).filter(ModelRegistry.path == model_path).all()
    )
    if any(str(owner.task_id or "") != str(task.id) for owner in owners):
        return _unavailable("model_unavailable", "训练模型归属不匹配")
    if not owners:
        return _describe_unregistered_path(model_path, resolve_path)
    if len(owners) != 1 or not _eligible(owners[0], task.id):
        return _unavailable("model_unavailable", "训练模型已失效或不属于当前任务")
    return _describe(owners[0], resolve_path)


def resolve_training_test_model(task, db, resolve_path):
    """Return the managed model path or raise a safe client-facing error."""
    descriptor = describe_training_test_model(task, db, resolve_path)
    if not descriptor["available"]:
        raise HTTPException(status_code=409, detail=descriptor["reason"])
    if descriptor["model_id"] is None:
        return str(resolve_path(_result(task).get("model_path"), None, require_file=True)), descriptor
    record = db.query(ModelRegistry).filter(ModelRegistry.id == descriptor["model_id"]).first()
    if not _eligible(record, task.id):
        raise HTTPException(status_code=409, detail="训练模型已失效或不属于当前任务")
    return str(resolve_path(record.path, record, require_file=True)), descriptor
