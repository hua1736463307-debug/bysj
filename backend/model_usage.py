"""Detect whether a managed model is referenced by an active runtime workload."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from database import ContinuousMonitor, JobTask, ModelRegistry


ACTIVE_MONITOR_STATES = {"running", "stopping"}
ACTIVE_JOB_STATES = {"pending", "running", "cancelling", "interrupted"}


def _normalized(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def _string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield _normalized(value)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _string_values(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _string_values(child)


def model_reference_candidates(record: ModelRegistry, resolved_path: Path) -> set[str]:
    values = {
        str(record.id),
        record.name,
        record.path,
        str(resolved_path),
        Path(str(record.path or "")).name,
        Path(resolved_path).name,
    }
    return {_normalized(value) for value in values if _normalized(value)}


def payload_references_model(payload: Any, candidates: set[str]) -> bool:
    if not candidates:
        return False
    return any(value in candidates for value in _string_values(payload))


def is_model_in_active_use(db, record: ModelRegistry, resolved_path: Path) -> bool:
    candidates = model_reference_candidates(record, resolved_path)

    monitors = db.query(ContinuousMonitor).filter(
        (ContinuousMonitor.desired_state == "running")
        | (ContinuousMonitor.status == "stopping")
    ).all()
    if any(payload_references_model(row.config_json or {}, candidates) for row in monitors):
        return True

    jobs = db.query(JobTask).filter(JobTask.status.in_(ACTIVE_JOB_STATES)).all()
    for row in jobs:
        if payload_references_model(row.input_json or {}, candidates):
            return True
        if payload_references_model(row.result_json or {}, candidates):
            return True
    return False
