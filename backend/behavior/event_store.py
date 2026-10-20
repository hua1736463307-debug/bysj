"""Persistence helpers for classifier-free behavior events and rule profiles."""

from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from database import BehaviorEvent, BehaviorRuleProfile, SessionLocal

from .realtime_engine import normalize_behavior_config, normalize_behavior_override


_SPOOL_LOCK = threading.Lock()
BEHAVIOR_EVENT_SPOOL_PATH = Path(
    os.getenv("BEHAVIOR_EVENT_SPOOL_PATH", "/app/data/spool/behavior_events.jsonl")
)


def _as_datetime(value: Any, default: Optional[datetime] = None) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (float, int)):
        return datetime.fromtimestamp(float(value))
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
    return default or datetime.now()


def persist_behavior_transitions(
    transitions: Iterable[Dict[str, Any]],
    *,
    owner_id: Optional[int],
    owner_username: str,
    mode: str,
    source_id: str,
    session_id: Optional[str] = None,
    grid_id: Optional[str] = None,
    grid_position: Optional[str] = None,
) -> List[BehaviorEvent]:
    db = SessionLocal()
    changed: List[BehaviorEvent] = []
    try:
        for transition in transitions or []:
            event_id = str(transition.get("event_id") or "")
            if not event_id:
                continue
            row = db.query(BehaviorEvent).filter(BehaviorEvent.event_id == event_id).first()
            state = str(transition.get("state") or "confirmed")
            if row is None and state == "confirmed":
                object_payload = {
                    key: transition.get(key)
                    for key in ("object_id", "object_class")
                    if transition.get(key) is not None
                }
                row = BehaviorEvent(
                    event_id=event_id,
                    owner_id=owner_id,
                    owner_username=str(owner_username or "system"),
                    mode=str(mode or "realtime"),
                    source_id=str(source_id or transition.get("source_id") or ""),
                    session_id=session_id,
                    grid_id=grid_id,
                    grid_position=None if grid_position is None else str(grid_position),
                    event_type=str(transition.get("event_type") or "unknown"),
                    state="confirmed",
                    severity=str(transition.get("severity") or "low"),
                    rule_score=float(transition.get("rule_score") or 0.0),
                    should_alert=bool(transition.get("should_alert")),
                    alerted_at=(
                        _as_datetime(transition.get("alerted_at") or transition.get("confirmed_at"))
                        if transition.get("should_alert")
                        else None
                    ),
                    decision_model_id=transition.get("decision_model_id"),
                    decision_model_version=transition.get("decision_model_version"),
                    decision_json=dict(transition.get("decision") or {}),
                    persistence_status="persisted",
                    track_ids_json=list(transition.get("track_ids") or transition.get("persons") or []),
                    object_json=object_payload,
                    reason_json=dict(transition.get("reason") or {}),
                    started_at=_as_datetime(transition.get("started_at")),
                    confirmed_at=_as_datetime(transition.get("confirmed_at")),
                    snapshot_path=transition.get("snapshot_path"),
                    clip_path=transition.get("clip_path"),
                    evidence_status=str(transition.get("evidence_status") or "collecting"),
                    evidence_error=transition.get("evidence_error"),
                )
                db.add(row)
            elif row is not None:
                row.state = state
                row.rule_score = max(float(row.rule_score or 0.0), float(transition.get("rule_score") or 0.0))
                row.reason_json = dict(transition.get("reason") or row.reason_json or {})
                if transition.get("ended_at") is not None:
                    row.ended_at = _as_datetime(transition.get("ended_at"))
                if transition.get("should_alert"):
                    row.should_alert = True
                    row.alerted_at = row.alerted_at or _as_datetime(
                        transition.get("alerted_at") or transition.get("confirmed_at")
                    )
                if transition.get("decision_model_id") is not None:
                    row.decision_model_id = str(transition["decision_model_id"])
                if transition.get("decision_model_version") is not None:
                    row.decision_model_version = str(transition["decision_model_version"])
                if transition.get("decision") is not None:
                    row.decision_json = dict(transition.get("decision") or {})
                row.persistence_status = "persisted"
            if row is not None:
                changed.append(row)
        from alarm_push import enqueue_alarm_pushes

        enqueue_alarm_pushes(db, changed)
        db.commit()
        for row in changed:
            db.refresh(row)
        return changed
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _append_spool_record(record: Dict[str, Any]) -> None:
    BEHAVIOR_EVENT_SPOOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":"))
    with _SPOOL_LOCK:
        with BEHAVIOR_EVENT_SPOOL_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def persist_behavior_transitions_resilient(
    transitions: Iterable[Dict[str, Any]],
    **context: Any,
) -> Dict[str, Any]:
    items = [dict(item) for item in (transitions or []) if item]
    if not items:
        return {"status": "persisted", "rows": []}
    try:
        rows = persist_behavior_transitions(items, **context)
        return {"status": "persisted", "rows": rows}
    except Exception as exc:
        try:
            _append_spool_record({
                "transitions": items,
                "context": context,
                "spooled_at": datetime.now().isoformat(),
                "error": str(exc),
            })
            return {"status": "spooled", "rows": [], "error": str(exc)}
        except Exception as spool_exc:
            return {
                "status": "unsynced",
                "rows": [],
                "error": str(exc),
                "spool_error": str(spool_exc),
            }


def replay_spooled_behavior_events(max_records: int = 1000) -> Dict[str, int]:
    if not BEHAVIOR_EVENT_SPOOL_PATH.is_file():
        return {"replayed": 0, "remaining": 0, "invalid": 0}
    with _SPOOL_LOCK:
        lines = BEHAVIOR_EVENT_SPOOL_PATH.read_text(encoding="utf-8").splitlines()
        replayed = 0
        invalid = 0
        remaining: List[str] = []
        for index, line in enumerate(lines):
            if index >= max(1, int(max_records)):
                remaining.append(line)
                continue
            try:
                record = json.loads(line)
                if record.get("kind") == "evidence_updates":
                    expected = len({str(item.get("event_id")) for item in (record.get("updates") or []) if item.get("event_id")})
                    updated = update_behavior_evidence(record.get("updates") or [])
                    if updated < expected:
                        raise RuntimeError("告警主记录尚未恢复")
                else:
                    persist_behavior_transitions(
                        record.get("transitions") or [],
                        **dict(record.get("context") or {}),
                    )
                replayed += 1
            except (TypeError, ValueError, json.JSONDecodeError):
                invalid += 1
            except Exception:
                remaining.append(line)
        tmp_path = BEHAVIOR_EVENT_SPOOL_PATH.with_suffix(".jsonl.tmp")
        if remaining:
            tmp_path.write_text("\n".join(remaining) + "\n", encoding="utf-8")
            tmp_path.replace(BEHAVIOR_EVENT_SPOOL_PATH)
        else:
            BEHAVIOR_EVENT_SPOOL_PATH.unlink(missing_ok=True)
            tmp_path.unlink(missing_ok=True)
        return {"replayed": replayed, "remaining": len(remaining), "invalid": invalid}


def update_behavior_evidence(updates: Iterable[Dict[str, Any]]) -> int:
    db = SessionLocal()
    updated = 0
    try:
        for update in updates or []:
            event_id = str(update.get("event_id") or "")
            if not event_id:
                continue
            row = db.query(BehaviorEvent).filter(BehaviorEvent.event_id == event_id).first()
            if row is None:
                continue
            if update.get("snapshot_path") is not None:
                row.snapshot_path = str(update["snapshot_path"])
            if update.get("clip_path") is not None:
                row.clip_path = str(update["clip_path"])
            row.evidence_status = str(update.get("evidence_status") or row.evidence_status)
            row.evidence_error = update.get("evidence_error")
            updated += 1
        db.commit()
        return updated
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def update_behavior_evidence_resilient(updates: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = [dict(item) for item in (updates or []) if item and item.get("event_id")]
    if not items:
        return {"status": "persisted", "updated": 0}
    expected = len({str(item["event_id"]) for item in items})
    try:
        updated = update_behavior_evidence(items)
        if updated < expected:
            raise RuntimeError("告警主记录尚未持久化")
        return {"status": "persisted", "updated": updated}
    except Exception as exc:
        try:
            _append_spool_record({
                "kind": "evidence_updates",
                "updates": items,
                "spooled_at": datetime.now().isoformat(),
                "error": str(exc),
            })
            return {"status": "spooled", "updated": 0, "error": str(exc)}
        except Exception as spool_exc:
            return {
                "status": "unsynced",
                "updated": 0,
                "error": str(exc),
                "spool_error": str(spool_exc),
            }


def get_behavior_profile(
    db: Session,
    owner_id: int,
    *,
    source_id: Optional[str] = None,
) -> Optional[BehaviorRuleProfile]:
    scope = "source" if source_id else "default"
    return (
        db.query(BehaviorRuleProfile)
        .filter(
            BehaviorRuleProfile.owner_id == owner_id,
            BehaviorRuleProfile.scope == scope,
            BehaviorRuleProfile.source_id == (str(source_id) if source_id else ""),
        )
        .first()
    )


def save_behavior_profile(
    db: Session,
    owner_id: int,
    owner_username: str,
    config: Dict[str, Any],
    *,
    source_id: Optional[str] = None,
) -> BehaviorRuleProfile:
    normalized = (
        normalize_behavior_override(config)
        if source_id
        else normalize_behavior_config(config)
    )
    row = get_behavior_profile(db, owner_id, source_id=source_id)
    if row is None:
        row = BehaviorRuleProfile(
            owner_id=owner_id,
            owner_username=owner_username,
            scope="source" if source_id else "default",
            source_id=str(source_id) if source_id else "",
        )
        db.add(row)
    row.config_json = normalized
    row.owner_username = owner_username
    db.commit()
    db.refresh(row)
    return row


def resolve_behavior_profile(
    owner_id: Optional[int],
    source_id: Optional[str],
    request_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if owner_id is None:
        return normalize_behavior_config(request_config)
    db = SessionLocal()
    try:
        default_row = get_behavior_profile(db, int(owner_id))
        source_row = get_behavior_profile(db, int(owner_id), source_id=source_id) if source_id else None
        return normalize_behavior_config(
            default_row.config_json if default_row else None,
            source_row.config_json if source_row else None,
            request_config,
        )
    finally:
        db.close()


def cleanup_expired_behavior_events(
    *,
    retention_days: int = 90,
    evidence_root: Optional[str] = None,
) -> Dict[str, int]:
    cutoff = datetime.now() - timedelta(days=max(1, int(retention_days)))
    root = Path(evidence_root or "/app/static/behavior_events").resolve()
    batch_root = Path("/app/static/behavior_results").resolve()
    db = SessionLocal()
    deleted_rows = 0
    deleted_paths = 0
    try:
        rows = db.query(BehaviorEvent).filter(BehaviorEvent.confirmed_at < cutoff).all()
        for row in rows:
            candidate_dirs = set()
            for value in (row.snapshot_path, row.clip_path):
                if not value:
                    continue
                path = Path(value).resolve()
                try:
                    path.relative_to(root)
                    candidate_dirs.add(path.parent)
                    continue
                except ValueError:
                    pass
                try:
                    path.relative_to(batch_root)
                except ValueError:
                    continue
                if path.is_file():
                    path.unlink()
                    deleted_paths += 1
                    parent = path.parent
                    while parent != batch_root:
                        try:
                            parent.rmdir()
                        except OSError:
                            break
                        parent = parent.parent
            for directory in candidate_dirs:
                if directory.exists():
                    shutil.rmtree(directory)
                    deleted_paths += 1
            db.delete(row)
            deleted_rows += 1
        db.commit()
        return {"deleted_rows": deleted_rows, "deleted_paths": deleted_paths}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
