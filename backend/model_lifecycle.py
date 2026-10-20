"""Lifecycle operations that keep trained models, files and job history in sync."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from database import JobTask, ModelRegistry
from dataset_deletion import purge_staged_path, restore_staged_path, stage_path_for_deletion
from job_store import TERMINAL_STATUSES, delete_terminal_job


StagedModel = tuple[ModelRegistry, Path, Path | None]


class ModelDeletionRecoveryError(RuntimeError):
    """A delete operation could not restore every staged file safely."""

    def __init__(self, message: str, failures: Optional[list[dict]] = None):
        super().__init__(message)
        self.failures = failures or []


def _journal_root(create: bool = True) -> Path:
    configured = os.getenv("MODEL_DELETE_JOURNAL_DIR", "").strip()
    root = (
        Path(configured).expanduser()
        if configured
        else Path(os.getenv("DEEPVISION_DATA_DIR", "/app/data")) / "temp" / "model-delete-operations"
    )
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class ModelDeletionJournal:
    """Durable file journal for crash diagnosis/recovery of model deletion."""

    def __init__(self, resource_type: str, resource_id: str, operation_id: Optional[str] = None):
        self.operation_id = operation_id or uuid.uuid4().hex
        self.path = _journal_root() / f"{self.operation_id}.json"
        now = datetime.now().isoformat()
        self.data = {
            "operation_id": self.operation_id,
            "resource_type": resource_type,
            "resource_id": str(resource_id),
            "state": "created",
            "entries": [],
            "created_at": now,
            "updated_at": now,
        }
        self.write()

    def write(self) -> None:
        self.data["updated_at"] = datetime.now().isoformat()
        _atomic_json_write(self.path, self.data)

    def plan(self, entries: list[dict]) -> None:
        self.data["entries"] = entries
        self.data["state"] = "staging"
        self.write()

    def mark_staged(self, index: int, staged: Optional[Path]) -> None:
        entry = self.data["entries"][index]
        entry["staged_path"] = str(staged) if staged else None
        entry["file_state"] = "staged" if staged else "missing"
        self.write()

    def set_state(self, state: str, **details) -> None:
        self.data["state"] = state
        for key, value in details.items():
            self.data[key] = value
        self.write()


def read_model_delete_journals(states: Optional[set[str]] = None) -> list[dict]:
    root = _journal_root(create=False)
    if not root.exists():
        return []
    rows = []
    for path in root.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        payload["_journal_path"] = str(path)
        if states is None or payload.get("state") in states:
            rows.append(payload)
    rows.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return rows


def _previous_result(resource_type: str, resource_id: str) -> Optional[dict]:
    for payload in read_model_delete_journals({"completed", "cleanup_pending"}):
        if payload.get("resource_type") == resource_type and str(payload.get("resource_id")) == str(resource_id):
            result = payload.get("result")
            if isinstance(result, dict):
                return {**result, "already_deleted": True}
    return None


def get_linked_model_records(db: Session, task_id: str, *, lock: bool = False) -> List[ModelRegistry]:
    query = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.task_id == task_id, ModelRegistry.status != "deleted")
        .order_by(ModelRegistry.created_at.asc())
    )
    if lock:
        query = query.with_for_update()
    return query.all()


def _failure(path: Path, staged: Optional[Path], exc: Exception) -> dict:
    return {
        "path": str(path),
        "staged_path": str(staged) if staged else None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def restore_staged_model_files(
    staged_models: Iterable[StagedModel],
    *,
    raise_on_error: bool = True,
) -> list[dict]:
    failures: list[dict] = []
    for _record, path, staged in reversed(list(staged_models)):
        try:
            restore_staged_path(staged, path)
        except Exception as exc:
            failures.append(_failure(path, staged, exc))
    if failures and raise_on_error:
        raise ModelDeletionRecoveryError("one or more staged model files could not be restored", failures)
    return failures


def purge_staged_model_files(staged_models: Iterable[StagedModel]) -> list[dict]:
    failures: list[dict] = []
    for _record, path, staged in staged_models:
        try:
            purge_staged_path(staged)
        except Exception as exc:
            failures.append(_failure(path, staged, exc))
    return failures


def stage_registered_model_files(
    records: Iterable[ModelRegistry],
    resolve_path: Callable[[ModelRegistry], Path],
    is_protected: Callable[[ModelRegistry, Path], bool],
    *,
    journal: Optional[ModelDeletionJournal] = None,
) -> List[StagedModel]:
    """Stage approved model files, rolling back earlier moves if any later move fails."""
    targets = []
    for record in records:
        path = Path(resolve_path(record))
        if is_protected(record, path):
            raise ValueError(f"protected model cannot be deleted: {record.name}")
        if path.exists() and not path.is_file():
            raise FileNotFoundError(f"managed model path is not a file: {path}")
        targets.append((record, path, record.status, record.updated_at))

    staged_hints: list[Path | None] = []
    if journal:
        entries = []
        for index, (record, path, previous_status, _previous_updated_at) in enumerate(targets):
            staged_hint = (
                path.parent
                / ".delete-quarantine"
                / f"{path.name}-{journal.operation_id}-{index:04d}"
                if path.is_file()
                else None
            )
            staged_hints.append(staged_hint)
            entries.append({
                "record_id": str(record.id),
                "record_name": record.name,
                "original_path": str(path),
                "staged_path": str(staged_hint) if staged_hint else None,
                "file_state": "planned" if staged_hint else "missing",
                "status_before": previous_status,
            })
        journal.plan(entries)
    else:
        staged_hints = [None] * len(targets)

    staged_models: list[StagedModel] = []
    try:
        for index, ((record, path, _previous_status, _previous_updated_at), staged_hint) in enumerate(zip(targets, staged_hints)):
            staged = stage_path_for_deletion(path, staged_hint) if path.is_file() else None
            staged_models.append((record, path, staged))
            record.status = "deleted"
            record.updated_at = datetime.now()
            if journal:
                journal.mark_staged(index, staged)
        if journal:
            journal.set_state("staged")
        return staged_models
    except Exception as exc:
        restore_failures = restore_staged_model_files(staged_models, raise_on_error=False)
        for record, _path, previous_status, previous_updated_at in targets:
            record.status = previous_status
            record.updated_at = previous_updated_at
        if journal:
            journal.set_state(
                "recovery_required" if restore_failures else "rolled_back",
                error=f"{type(exc).__name__}: {exc}",
                restore_failures=restore_failures,
            )
        if restore_failures:
            raise ModelDeletionRecoveryError(
                "model staging failed and one or more earlier files could not be restored",
                restore_failures,
            ) from exc
        raise


def _rollback_after_database_failure(
    db: Session,
    journal: ModelDeletionJournal,
    staged_models: list[StagedModel],
    exc: Exception,
) -> None:
    db.rollback()
    if not staged_models and journal.data.get("state") in {"rolled_back", "recovery_required"}:
        return
    failures = restore_staged_model_files(staged_models, raise_on_error=False)
    journal.set_state(
        "recovery_required" if failures else "rolled_back",
        error=f"{type(exc).__name__}: {exc}",
        restore_failures=failures,
    )
    if failures:
        raise ModelDeletionRecoveryError(
            "database delete failed and one or more staged files could not be restored",
            failures,
        ) from exc


def _finish_cleanup(
    journal: ModelDeletionJournal,
    staged_models: list[StagedModel],
    result: dict,
    *,
    journal_error: Optional[str] = None,
) -> dict:
    failures = purge_staged_model_files(staged_models)
    pending = bool(failures or journal_error)
    enriched = {
        **result,
        "operation_id": journal.operation_id,
        "cleanup_pending": pending,
        "cleanup_failures": failures,
        "journal_pending": bool(journal_error),
    }
    try:
        journal.set_state(
            "cleanup_pending" if pending else "completed",
            cleanup_failures=failures,
            journal_error=journal_error,
            result=enriched,
        )
    except Exception as exc:
        # The database commit is already durable at this point. Never restore
        # files because the recovery journal itself could not be updated.
        enriched["cleanup_pending"] = True
        enriched["journal_pending"] = True
        enriched["journal_error"] = f"{type(exc).__name__}: {exc}"
    return enriched


def delete_training_task_and_models(
    db: Session,
    task_id: str,
    *,
    resolve_path: Callable[[ModelRegistry], Path],
    is_protected: Callable[[ModelRegistry, Path], bool],
) -> Dict[str, object]:
    """Delete a completed training job and linked models using a durable staged operation."""
    task = db.query(JobTask).filter(JobTask.id == task_id).with_for_update().first()
    if not task:
        previous = _previous_result("training_task", task_id)
        if previous is not None:
            return previous
        raise KeyError("training job not found")
    if task.type != "training":
        raise KeyError("training job not found")
    if task.status not in TERMINAL_STATUSES:
        raise ValueError("training job is not terminal")

    journal = ModelDeletionJournal("training_task", task_id)
    staged_models: list[StagedModel] = []
    try:
        staged_models = stage_registered_model_files(
            get_linked_model_records(db, task_id, lock=True),
            resolve_path,
            is_protected,
            journal=journal,
        )
        deleted_job = delete_terminal_job(db, task_id, job_type="training", commit=False)
        db.commit()
    except Exception as exc:
        _rollback_after_database_failure(db, journal, staged_models, exc)
        raise

    journal_error = None
    try:
        journal.set_state("db_committed")
    except Exception as exc:
        journal_error = f"{type(exc).__name__}: {exc}"

    result = {
        **deleted_job,
        "deleted_model_count": len(staged_models),
        "deleted_model_paths": [str(path) for _record, path, _staged in staged_models],
    }
    return _finish_cleanup(journal, staged_models, result, journal_error=journal_error)


def delete_model_and_linked_training_task(
    db: Session,
    record: ModelRegistry,
    *,
    resolve_path: Callable[[ModelRegistry], Path],
    is_protected: Callable[[ModelRegistry, Path], bool],
) -> Dict[str, object]:
    """Delete a model and, when it came from training, its linked job history."""
    # Read first without taking a model lock. Training-backed deletion always
    # locks task -> linked models in that order, avoiding a model/task deadlock
    # against deletion initiated from the training-task endpoint.
    current = db.query(ModelRegistry).filter(ModelRegistry.id == record.id).first()
    if current is None:
        previous = _previous_result("model", str(record.id))
        if previous is not None:
            return previous
        raise KeyError("model not found")

    linked_task = (
        db.query(JobTask).filter(JobTask.id == current.task_id).first()
        if current.task_id
        else None
    )
    if linked_task and linked_task.type == "training" and str(current.format or "pt").lower() == "pt":
        deleted = delete_training_task_and_models(
            db,
            linked_task.id,
            resolve_path=resolve_path,
            is_protected=is_protected,
        )
        return {
            "deleted_model_count": deleted["deleted_model_count"],
            "deleted_model_paths": deleted["deleted_model_paths"],
            "deleted_training_task_id": deleted["task_id"],
            "operation_id": deleted.get("operation_id"),
            "cleanup_pending": deleted.get("cleanup_pending", False),
            "cleanup_failures": deleted.get("cleanup_failures", []),
            "already_deleted": deleted.get("already_deleted", False),
        }

    current = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.id == record.id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if current is None:
        previous = _previous_result("model", str(record.id))
        if previous is not None:
            return previous
        raise KeyError("model not found")
    if current.status == "deleted":
        previous = _previous_result("model", str(current.id))
        return previous or {
            "deleted_model_count": 0,
            "deleted_model_paths": [],
            "deleted_training_task_id": current.task_id,
            "cleanup_pending": False,
            "already_deleted": True,
        }

    journal = ModelDeletionJournal("model", str(current.id))
    staged_models: list[StagedModel] = []
    try:
        staged_models = stage_registered_model_files(
            [current],
            resolve_path,
            is_protected,
            journal=journal,
        )
        db.commit()
    except Exception as exc:
        _rollback_after_database_failure(db, journal, staged_models, exc)
        raise

    journal_error = None
    try:
        journal.set_state("db_committed")
    except Exception as exc:
        journal_error = f"{type(exc).__name__}: {exc}"

    result = {
        "deleted_model_count": len(staged_models),
        "deleted_model_paths": [str(path) for _record, path, _staged in staged_models],
        "deleted_training_task_id": None,
    }
    return _finish_cleanup(journal, staged_models, result, journal_error=journal_error)
