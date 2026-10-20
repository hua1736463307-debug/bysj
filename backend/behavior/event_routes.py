"""Authenticated behavior event history and rule-profile API."""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from jose import JWTError, jwt
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth_utils import ALGORITHM, SECRET_KEY, get_current_user, require_any_menu
from database import BehaviorEvent, BehaviorEventAction, BehaviorReviewSample, BehaviorRuleProfile, RecorderConfig, get_db

from .event_store import get_behavior_profile, save_behavior_profile
from .realtime_queue import resolve_behavior_queue_process_interval_ms
from .realtime_engine import normalize_behavior_config
from .feedback_mapping import (
    SUPPORTED_BEHAVIOR_EVENTS,
    feedback_training_mapping,
    normalize_behavior_label,
)


router = APIRouter(tags=["behavior-events"])

ALARM_LABELS = set(SUPPORTED_BEHAVIOR_EVENTS)
FEEDBACK_VERDICTS = {"correct", "false_positive", "wrong_type", "uncertain", "not_evaluable"}
HANDLING_STATUSES = {"pending", "processing", "resolved"}
SAMPLE_ROOT = Path(os.getenv("BEHAVIOR_SAMPLE_ROOT", "/app/data/action_samples/feedback"))
MISSED_UPLOAD_MAX_BYTES = max(1, int(os.getenv("BEHAVIOR_MISSED_UPLOAD_MAX_BYTES", str(500 * 1024 * 1024))))
MISSED_UPLOAD_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
require_alarm_menu = require_any_menu("/alarm-center", "/behavior-review")
require_optimization_menu = require_any_menu("/algorithm-training", "/action-training")


class BehaviorRuleProfilePayload(BaseModel):
    config: Dict[str, Any]


class BehaviorHandlingPayload(BaseModel):
    status: str
    note: str = ""


class BehaviorFeedbackPayload(BaseModel):
    verdict: str
    actual_label: Optional[str] = None
    note: str = ""


class BehaviorEventBatchDeletePayload(BaseModel):
    event_ids: List[str]


def _is_admin(user: Dict[str, Any]) -> bool:
    return user.get("role") == "admin" or "admin" in (user.get("roleKeys") or [])


def _profile_owner_id(user: Dict[str, Any], requested_owner_id: Optional[int]) -> int:
    current_owner_id = int(user["id"])
    if requested_owner_id is None or int(requested_owner_id) == current_owner_id:
        return current_owner_id
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="无权查看其他用户的行为规则")
    return int(requested_owner_id)


def _visible_event(db: Session, event_id: str, user: Dict[str, Any]) -> BehaviorEvent:
    query = db.query(BehaviorEvent).filter(BehaviorEvent.event_id == event_id)
    if not _is_admin(user):
        query = query.filter(BehaviorEvent.owner_id == int(user["id"]))
    row = query.first()
    if row is None:
        raise HTTPException(status_code=404, detail="行为事件不存在")
    return row


def _media_token(row: BehaviorEvent, kind: str) -> str:
    payload = {
        "typ": "behavior_event_media",
        "event_id": row.event_id,
        "owner_id": row.owner_id,
        "kind": kind,
        "exp": datetime.utcnow() + timedelta(minutes=5),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


_RECORDER_SOURCE_ID = re.compile(r"^recorder:([^:]+):channel:([^:]+)$")


def _camera_names(db: Session, rows: List[BehaviorEvent]) -> Dict[str, str]:
    references = []
    for row in rows:
        match = _RECORDER_SOURCE_ID.fullmatch(str(row.source_id or ""))
        if match:
            references.append((row, match.group(1), match.group(2)))
    if not references:
        return {}
    recorders = {
        (str(item.id), str(item.owner_id)): item
        for item in db.query(RecorderConfig).filter(
            RecorderConfig.id.in_({recorder_id for _, recorder_id, _ in references})
        ).all()
    }
    names = {}
    for row, recorder_id, channel_number in references:
        recorder = recorders.get((recorder_id, str(row.owner_id)))
        if not recorder:
            continue
        channel = next(
            (item for item in (recorder.config_json or {}).get("channels", [])
             if str(item.get("number")) == channel_number),
            None,
        )
        if channel and str(channel.get("name") or "").strip():
            names[row.event_id] = str(channel["name"]).strip()
    return names


def _serialize_event(row: BehaviorEvent, camera_name: Optional[str] = None) -> Dict[str, Any]:
    payload = row.to_dict()
    payload["camera_name"] = camera_name or None
    decision = dict(payload.get("decision") or {})
    decision.pop("training_feature_vector", None)
    payload["decision"] = decision
    if row.snapshot_path:
        token = quote(_media_token(row, "snapshot"), safe="")
        payload["snapshot_url"] = f"/api/behavior/event-media/{row.event_id}/snapshot?token={token}"
    else:
        payload["snapshot_url"] = None
    face_box = ((row.reason_json or {}).get('face_capture') or {}).get('face_crop_bbox') or ((row.reason_json or {}).get("face_crop_bbox") if row.event_type == "missing_badge" else None)
    if row.snapshot_path and isinstance(face_box, list) and len(face_box) == 4:
        token = quote(_media_token(row, "face"), safe="")
        payload["face_url"] = f"/api/behavior/event-media/{row.event_id}/face?token={token}"
    else:
        payload["face_url"] = None
    if row.clip_path and not str(row.source_id or '').startswith('recorder:'):
        token = quote(_media_token(row, "clip"), safe="")
        payload["clip_url"] = f"/api/behavior/event-media/{row.event_id}/clip?token={token}"
    else:
        payload["clip_url"] = None
    if str(row.source_id or '').startswith('recorder:'):
        payload['video_evidence_source'] = 'nvr'
        payload['clip_path'] = None
        # Keep full provenance internally for review; the monitoring product exposes categories only.
        for key in ('decision_model_id', 'decision_model_version', 'decision', 'reason', 'evidence_error'):
            payload.pop(key, None)
    return payload


@router.get('/behavior/events/{event_id}/recording')
@router.get('/api/behavior/events/{event_id}/recording')
def get_event_recording(event_id: str, db=Depends(get_db), user=Depends(require_alarm_menu)):
    from recorder_playback import recording_source, window
    row = _visible_event(db, event_id, user)
    recording_source(db, row)
    start, end = window(row)
    from datetime import timezone
    if end > datetime.now(timezone.utc):
        raise HTTPException(409, '告警后 30 秒录像尚未形成，请稍后重试')
    token = quote(_media_token(row, 'recording'), safe='')
    return {'url':f'/api/behavior/event-recording/{event_id}?token={token}',
            'start_time':start.isoformat(), 'end_time':end.isoformat(), 'duration_seconds':60, 'source':'nvr'}


@router.get('/behavior/event-recording/{event_id}')
@router.get('/api/behavior/event-recording/{event_id}')
async def stream_event_recording(event_id: str, token: str, db=Depends(get_db)):
    from recorder_playback import playback_url, stream_recording
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(401, '录像链接已失效，请重新查询') from None
    if payload.get('typ') != 'behavior_event_media' or payload.get('kind') != 'recording' or payload.get('event_id') != event_id:
        raise HTTPException(401, '录像链接无效')
    row = db.query(BehaviorEvent).filter(BehaviorEvent.event_id == event_id).first()
    if row is None or row.owner_id != payload.get('owner_id'):
        raise HTTPException(404, '行为事件不存在')
    import asyncio
    url = await asyncio.to_thread(playback_url, db, row)
    return await stream_recording(url)


def _model_family(event_type: str) -> str:
    if event_type == "missing_badge":
        return "badge_image"
    if event_type == "phone_use":
        return "pose_object"
    return "pose"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sample_media_path(content_hash: str, suffix: str) -> Path:
    target_dir = SAMPLE_ROOT / "media" / content_hash[:2]
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = next(target_dir.glob(f"{content_hash}.*"), None)
    return existing or (target_dir / f"{content_hash}{suffix.lower()}")


def _copy_event_evidence(row: BehaviorEvent) -> Dict[str, Optional[str]]:
    source_value = row.clip_path or row.snapshot_path
    if not source_value:
        return {"sample_type": "clip", "clip_path": None, "image_path": None, "content_hash": None}
    source = Path(str(source_value))
    if not source.is_file():
        return {"sample_type": "clip", "clip_path": None, "image_path": None, "content_hash": None}
    digest = _sha256_file(source)
    target = _canonical_sample_media_path(digest, source.suffix)
    if not target.is_file():
        shutil.copy2(source, target)
    is_image = source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return {
        "sample_type": "image" if is_image else "clip",
        "clip_path": None if is_image else str(target),
        "image_path": str(target) if is_image else None,
        "content_hash": digest,
    }


def _record_action(
    db: Session,
    row: BehaviorEvent,
    action_type: str,
    before: Dict[str, Any],
    after: Dict[str, Any],
    note: str,
    user: Dict[str, Any],
) -> None:
    db.add(BehaviorEventAction(
        event_id=row.event_id,
        owner_id=row.owner_id,
        action_type=action_type,
        from_value_json=before,
        to_value_json=after,
        note=str(note or ""),
        actor_username=str(user.get("username") or "unknown"),
    ))


def _sync_feedback_sample(db: Session, row: BehaviorEvent, user: Dict[str, Any]) -> None:
    db.query(BehaviorReviewSample).filter(
        BehaviorReviewSample.behavior_event_id == row.event_id,
    ).delete(synchronize_session=False)
    evaluable = row.feedback_verdict not in {"uncertain", "not_evaluable"}
    mapping = feedback_training_mapping(
        row.event_type,
        row.feedback_verdict,
        row.feedback_label,
        evaluable=evaluable,
    )
    if not mapping["trainable"]:
        return
    label = str(mapping["label"] or "")
    media = _copy_event_evidence(row)
    evidence = {
        "feedback_verdict": row.feedback_verdict,
        "original_event_type": row.event_type,
        "negative_for": mapping["negative_for"],
        "evaluable": evaluable,
        "review_status": "not_evaluable" if not evaluable else "reviewed",
        "decision": row.decision_json or {},
    }
    db.add(BehaviorReviewSample(
        owner_id=row.owner_id,
        owner_username=row.owner_username,
        source_task_id=f"realtime:{row.source_id}"[:128],
        source_event_id=row.event_id,
        source_event_type=row.event_type,
        source_kind="realtime_alarm",
        behavior_event_id=row.event_id,
        source_id=row.source_id,
        model_family=_model_family(label if label != "normal" else row.event_type),
        content_hash=media["content_hash"],
        review_label=label,
        review_status="reviewed",
        sample_type=str(media["sample_type"]),
        clip_path=media["clip_path"],
        image_path=media["image_path"],
        evidence_json=evidence,
        note=row.feedback_note or "",
        reviewed_by=str(user.get("username") or "unknown"),
        reviewed_at=row.feedback_at or datetime.now(),
    ))


def _unlink_file_in_root(path_value: Optional[str], root: Path) -> int:
    if not path_value:
        return 0
    try:
        path = Path(path_value).resolve()
        path.relative_to(root.resolve())
    except (OSError, ValueError):
        return 0
    if not path.is_file():
        return 0
    path.unlink()
    return 1


def _remove_event_evidence(
    event_id: str,
    snapshot_path: Optional[str],
    clip_path: Optional[str],
) -> int:
    root = Path(os.getenv("DEEPVISION_BEHAVIOR_EVENT_DIR", "/app/static/behavior_events")).resolve()
    deleted_files = 0
    event_dirs = set()
    for path_value in (snapshot_path, clip_path):
        if not path_value:
            continue
        try:
            path = Path(path_value).resolve()
            path.relative_to(root)
        except (OSError, ValueError):
            continue
        if path.parent.name == event_id:
            event_dirs.add(path.parent)
        else:
            deleted_files += _unlink_file_in_root(path_value, root)
    for event_dir in event_dirs:
        if event_dir.is_dir():
            shutil.rmtree(event_dir)
            deleted_files += 1
    return deleted_files


def _remove_unreferenced_sample_media(db: Session, sample_paths: List[str]) -> int:
    deleted_files = 0
    for path_value in set(path for path in sample_paths if path):
        remaining = db.query(BehaviorReviewSample.id).filter(
            (BehaviorReviewSample.clip_path == path_value)
            | (BehaviorReviewSample.image_path == path_value)
        ).first()
        if remaining is None:
            deleted_files += _unlink_file_in_root(path_value, SAMPLE_ROOT)
    return deleted_files


def _delete_behavior_events(
    event_ids: List[str],
    db: Session,
    current_user: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_ids = list(dict.fromkeys(str(event_id).strip() for event_id in event_ids if str(event_id).strip()))
    if not normalized_ids:
        raise HTTPException(status_code=422, detail="select at least one event")
    if len(normalized_ids) > 200:
        raise HTTPException(status_code=422, detail="a maximum of 200 events can be deleted at once")

    prepared = []
    for event_id in normalized_ids:
        row = _visible_event(db, event_id, current_user)
        if row.evidence_status == "collecting":
            raise HTTPException(status_code=409, detail="evidence is still being generated; try again shortly")
        sample_rows = db.query(BehaviorReviewSample).filter(
            BehaviorReviewSample.behavior_event_id == row.event_id,
        ).all()
        prepared.append({
            "row": row,
            "samples": sample_rows,
            "evidence": (row.event_id, row.snapshot_path, row.clip_path),
        })

    deleted_actions = 0
    deleted_samples = 0
    sample_paths = []
    for item in prepared:
        row = item["row"]
        deleted_actions += db.query(BehaviorEventAction).filter(
            BehaviorEventAction.event_id == row.event_id,
        ).delete(synchronize_session=False)
        deleted_samples += len(item["samples"])
        for sample in item["samples"]:
            sample_paths.extend(value for value in (sample.clip_path, sample.image_path) if value)
            db.delete(sample)
        db.delete(row)
    db.commit()

    deleted_files = 0
    cleanup_warnings = []
    for item in prepared:
        try:
            deleted_files += _remove_event_evidence(*item["evidence"])
        except OSError as exc:
            cleanup_warnings.append(str(exc))
    try:
        deleted_files += _remove_unreferenced_sample_media(db, sample_paths)
    except OSError as exc:
        cleanup_warnings.append(str(exc))

    return {
        "success": True,
        "event_ids": normalized_ids,
        "deleted_events": len(prepared),
        "deleted_actions": deleted_actions,
        "deleted_samples": deleted_samples,
        "deleted_files": deleted_files,
        "cleanup_warning": (
            "records deleted but some evidence could not be removed: " + "; ".join(cleanup_warnings)
            if cleanup_warnings else None
        ),
    }


@router.get("/behavior/events")
@router.get("/api/behavior/events")
def list_behavior_events(
    mode: Optional[str] = None,
    source_id: Optional[str] = None,
    event_type: Optional[str] = None,
    should_alert: Optional[bool] = None,
    handling_status: Optional[str] = None,
    feedback_verdict: Optional[str] = None,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    started_after: Optional[datetime] = None,
    started_before: Optional[datetime] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_alarm_menu),
):
    query = db.query(BehaviorEvent)
    if not _is_admin(current_user):
        query = query.filter(BehaviorEvent.owner_id == int(current_user["id"]))
    if mode:
        query = query.filter(BehaviorEvent.mode == mode)
    if source_id:
        query = query.filter(BehaviorEvent.source_id == source_id)
    if event_type:
        query = query.filter(BehaviorEvent.event_type == event_type)
    if should_alert is not None:
        query = query.filter(BehaviorEvent.should_alert == bool(should_alert))
    if handling_status:
        query = query.filter(BehaviorEvent.handling_status == handling_status)
    if feedback_verdict:
        query = query.filter(BehaviorEvent.feedback_verdict == feedback_verdict)
    if model_id:
        query = query.filter(BehaviorEvent.decision_model_id == model_id)
    if model_version:
        query = query.filter(BehaviorEvent.decision_model_version.ilike(f"%{model_version}%"))
    if started_after:
        query = query.filter(BehaviorEvent.confirmed_at >= started_after)
    if started_before:
        query = query.filter(BehaviorEvent.confirmed_at <= started_before)
    total = query.count()
    rows = (
        query.order_by(BehaviorEvent.confirmed_at.desc(), BehaviorEvent.event_id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    camera_names = _camera_names(db, rows)
    return {
        "items": [_serialize_event(row, camera_names.get(row.event_id)) for row in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/behavior/events/refresh-config")
@router.get("/api/behavior/events/refresh-config")
def get_behavior_event_refresh_config(
    current_user: Dict[str, Any] = Depends(require_alarm_menu),
):
    return {"refresh_interval_ms": resolve_behavior_queue_process_interval_ms()}


@router.get("/behavior/events/{event_id}")
@router.get("/api/behavior/events/{event_id}")
def get_behavior_event(
    event_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_alarm_menu),
):
    row = _visible_event(db, event_id, current_user)
    payload = _serialize_event(row, _camera_names(db, [row]).get(row.event_id))
    payload["actions"] = [
        item.to_dict()
        for item in db.query(BehaviorEventAction)
        .filter(BehaviorEventAction.event_id == row.event_id)
        .order_by(BehaviorEventAction.created_at.asc())
        .all()
    ]
    return payload


@router.delete("/behavior/events/{event_id}")
@router.delete("/api/behavior/events/{event_id}")
def delete_behavior_event(
    event_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_alarm_menu),
):
    result = _delete_behavior_events([event_id], db, current_user)
    result["event_id"] = event_id
    return result


@router.delete("/behavior/events")
@router.delete("/api/behavior/events")
def delete_behavior_events(
    payload: BehaviorEventBatchDeletePayload,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_alarm_menu),
):
    return _delete_behavior_events(payload.event_ids, db, current_user)


@router.patch("/behavior/events/{event_id}/handling")
@router.patch("/api/behavior/events/{event_id}/handling")
def update_behavior_event_handling(
    event_id: str,
    payload: BehaviorHandlingPayload,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_alarm_menu),
):
    status = str(payload.status or "").strip().lower()
    if status not in HANDLING_STATUSES:
        raise HTTPException(status_code=400, detail="不支持的处置状态")
    row = _visible_event(db, event_id, current_user)
    before = {"status": row.handling_status or "pending", "note": row.handling_note or ""}
    now = datetime.now()
    row.handling_status = status
    row.handling_note = str(payload.note or "")
    row.handled_by = str(current_user.get("username") or "unknown")
    row.handled_at = now
    after = {"status": status, "note": row.handling_note}
    _record_action(db, row, "handling", before, after, row.handling_note, current_user)
    db.commit()
    db.refresh(row)
    return _serialize_event(row, _camera_names(db, [row]).get(row.event_id))


@router.put("/behavior/events/{event_id}/feedback")
@router.put("/api/behavior/events/{event_id}/feedback")
def update_behavior_event_feedback(
    event_id: str,
    payload: BehaviorFeedbackPayload,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_alarm_menu),
):
    verdict = str(payload.verdict or "").strip().lower()
    if verdict not in FEEDBACK_VERDICTS:
        raise HTTPException(status_code=400, detail="不支持的算法反馈")
    actual_label = str(payload.actual_label or "").strip()
    if verdict == "wrong_type" and (actual_label not in ALARM_LABELS or actual_label == "normal"):
        raise HTTPException(status_code=400, detail="类型错误时必须选择真实告警类型")
    row = _visible_event(db, event_id, current_user)
    before = {
        "verdict": row.feedback_verdict,
        "label": row.feedback_label,
        "note": row.feedback_note or "",
    }
    row.feedback_verdict = verdict
    row.feedback_label = (
        row.event_type if verdict == "correct"
        else "normal" if verdict == "false_positive"
        else actual_label if verdict == "wrong_type"
        else None
    )
    row.feedback_note = str(payload.note or "")
    row.feedback_by = str(current_user.get("username") or "unknown")
    row.feedback_at = datetime.now()
    after = {"verdict": verdict, "label": row.feedback_label, "note": row.feedback_note}
    _record_action(db, row, "feedback", before, after, row.feedback_note, current_user)
    _sync_feedback_sample(db, row, current_user)
    db.commit()
    db.refresh(row)
    return _serialize_event(row, _camera_names(db, [row]).get(row.event_id))


@router.post("/behavior/feedback/missed")
@router.post("/api/behavior/feedback/missed")
async def upload_missed_behavior_feedback(
    file: UploadFile = File(...),
    label: str = Form(...),
    source_id: str = Form("manual-upload"),
    note: str = Form(""),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_optimization_menu),
):
    normalized_label = str(label or "").strip()
    if normalized_label not in ALARM_LABELS:
        raise HTTPException(status_code=400, detail="不支持的漏报告警类型")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in MISSED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=400, detail="仅支持 mp4、mov、avi、mkv、webm 视频")
    sample_id = str(uuid.uuid4())
    target_dir = SAMPLE_ROOT / "missed" / sample_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"clip{suffix}"
    size = 0
    digest = hashlib.sha256()
    try:
        with target.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MISSED_UPLOAD_MAX_BYTES:
                    raise HTTPException(status_code=413, detail="漏报视频超过允许大小")
                digest.update(chunk)
                handle.write(chunk)
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    finally:
        await file.close()
    content_hash = digest.hexdigest()
    media_target = _canonical_sample_media_path(content_hash, suffix)
    if media_target.is_file():
        target.unlink(missing_ok=True)
    else:
        shutil.move(str(target), str(media_target))
    shutil.rmtree(target_dir, ignore_errors=True)
    row = BehaviorReviewSample(
        id=sample_id,
        owner_id=current_user.get("id"),
        owner_username=str(current_user.get("username") or "unknown"),
        source_task_id=f"missed:{str(source_id or 'manual-upload')}"[:128],
        source_event_id=sample_id,
        source_event_type=normalized_label,
        source_kind="missed_upload",
        source_id=str(source_id or "manual-upload"),
        model_family=_model_family(normalized_label),
        content_hash=content_hash,
        review_label=normalized_label,
        review_status="reviewed",
        sample_type="clip",
        clip_path=str(media_target),
        evidence_json={"missed_alarm": True, "original_filename": file.filename or ""},
        note=str(note or ""),
        reviewed_by=str(current_user.get("username") or "unknown"),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"success": True, "sample": row.to_dict()}


@router.get("/behavior/optimization-samples/stats")
@router.get("/api/behavior/optimization-samples/stats")
def get_optimization_sample_stats(
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_optimization_menu),
):
    query = db.query(BehaviorReviewSample).filter(BehaviorReviewSample.review_status == "reviewed")
    if not _is_admin(current_user):
        query = query.filter(BehaviorReviewSample.owner_id == int(current_user["id"]))
    rows: List[BehaviorReviewSample] = query.all()
    labels: Dict[str, int] = {}
    families: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    content_hash_counts: Dict[str, int] = {}
    content_hash_groups: Dict[str, set] = {}
    by_behavior_type: Dict[str, Dict[str, Any]] = {
        event_type: {
            "positive": 0,
            "negative": 0,
            "hard_negative": 0,
            "source_groups": [],
            "camera_sources": [],
            "failed_media": 0,
            "temporal_failed_media": 0,
            "duplicate_samples": 0,
        }
        for event_type in sorted(ALARM_LABELS)
    }
    for row in rows:
        if row.content_hash:
            content_hash_counts[row.content_hash] = content_hash_counts.get(row.content_hash, 0) + 1
            content_hash_groups.setdefault(row.content_hash, set()).add(row.source_task_id or row.source_id or row.id)
    for row in rows:
        labels[row.review_label] = labels.get(row.review_label, 0) + 1
        family = row.model_family or _model_family(row.source_event_type or row.review_label)
        families[family] = families.get(family, 0) + 1
        kind = row.source_kind or "batch_review"
        sources[kind] = sources.get(kind, 0) + 1
        evidence = row.evidence_json if isinstance(row.evidence_json, dict) else {}
        if evidence.get("evaluable") is False or str(evidence.get("review_status") or "").lower() == "not_evaluable":
            continue
        for event_type, item in by_behavior_type.items():
            normalized_label = normalize_behavior_label(row.review_label)
            is_positive = normalized_label == event_type
            is_negative = event_type in (evidence.get("negative_for") or [])
            if is_positive:
                item["positive"] += 1
            elif is_negative:
                item["negative"] += 1
                item["hard_negative"] += 1
            else:
                continue
            item["source_groups"].append(row.source_task_id or row.source_id or row.id)
            if row.source_id:
                item["camera_sources"].append(row.source_id)
            if row.content_hash and content_hash_counts.get(row.content_hash, 0) > 1:
                item["duplicate_samples"] += 1
            media_path = row.clip_path or row.image_path
            decision = evidence.get("decision") if isinstance(evidence.get("decision"), dict) else {}
            expected_family = _model_family(event_type)
            missing_specialized_feature = (
                expected_family in {"pose_object", "badge_image"}
                and (
                    decision.get("model_family") != expected_family
                    or not decision.get("training_feature_vector")
                )
            )
            if not media_path or not Path(str(media_path)).is_file() or missing_specialized_feature:
                item["failed_media"] += 1
            if not row.clip_path or not Path(str(row.clip_path)).is_file():
                item["temporal_failed_media"] += 1
    for item in by_behavior_type.values():
        item["source_group_count"] = len(set(item.pop("source_groups")))
        item["camera_coverage"] = len(set(item.pop("camera_sources")))
        item["leakage_check"] = {
            "status": "warning" if item["duplicate_samples"] else "passed",
            "duplicate_group_collisions": sum(
                1 for groups in content_hash_groups.values() if len(groups) > 1
            ),
        }
        item["ready"] = (
            item["positive"] >= 20
            and item["negative"] >= 20
            and item["source_group_count"] >= 3
            and item["failed_media"] == 0
        )
    return {
        "total": len(rows),
        "trainable": sum(
            1 for row in rows
            if row.review_label not in {"uncertain", "delete"}
            and not (
                isinstance(row.evidence_json, dict)
                and (
                    row.evidence_json.get("evaluable") is False
                    or str(row.evidence_json.get("review_status") or "").lower() == "not_evaluable"
                )
            )
        ),
        "labels": labels,
        "model_families": families,
        "source_kinds": sources,
        "by_behavior_type": by_behavior_type,
    }


@router.get("/behavior/rule-profiles/default")
@router.get("/api/behavior/rule-profiles/default")
def get_default_behavior_profile(
    owner_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    target_owner_id = _profile_owner_id(current_user, owner_id)
    row = get_behavior_profile(db, target_owner_id)
    return row.to_dict() if row else {
        "scope": "default",
        "source_id": None,
        "config": normalize_behavior_config(),
    }


@router.put("/behavior/rule-profiles/default")
@router.put("/api/behavior/rule-profiles/default")
def put_default_behavior_profile(
    payload: BehaviorRuleProfilePayload,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return save_behavior_profile(
        db,
        int(current_user["id"]),
        str(current_user.get("username") or ""),
        payload.config,
    ).to_dict()


@router.get("/behavior/rule-profiles/sources/{source_id}")
@router.get("/api/behavior/rule-profiles/sources/{source_id}")
def get_source_behavior_profile(
    source_id: str,
    owner_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    target_owner_id = _profile_owner_id(current_user, owner_id)
    row = get_behavior_profile(db, target_owner_id, source_id=source_id)
    return row.to_dict() if row else {
        "scope": "source",
        "source_id": source_id,
        "config": {},
    }


@router.put("/behavior/rule-profiles/sources/{source_id}")
@router.put("/api/behavior/rule-profiles/sources/{source_id}")
def put_source_behavior_profile(
    source_id: str,
    payload: BehaviorRuleProfilePayload,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return save_behavior_profile(
        db,
        int(current_user["id"]),
        str(current_user.get("username") or ""),
        payload.config,
        source_id=source_id,
    ).to_dict()


@router.get("/behavior/event-media/{event_id}/{kind}")
@router.get("/api/behavior/event-media/{event_id}/{kind}")
def get_behavior_event_media(
    event_id: str,
    kind: str,
    token: str,
    db: Session = Depends(get_db),
):
    if kind not in {"snapshot", "clip", "face"}:
        raise HTTPException(status_code=404, detail="证据类型不存在")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="证据链接已失效")
    if (
        payload.get("typ") != "behavior_event_media"
        or payload.get("event_id") != event_id
        or payload.get("kind") != kind
    ):
        raise HTTPException(status_code=401, detail="证据链接无效")
    row = db.query(BehaviorEvent).filter(BehaviorEvent.event_id == event_id).first()
    if row is None or row.owner_id != payload.get("owner_id"):
        raise HTTPException(status_code=404, detail="行为事件不存在")
    if kind == 'clip' and str(row.source_id or '').startswith('recorder:'):
        raise HTTPException(404, '录像机告警请通过录像回放查询视频证据')
    path_value = row.snapshot_path if kind in {"snapshot", "face"} else row.clip_path
    if not path_value or not Path(path_value).is_file():
        raise HTTPException(status_code=404, detail="事件证据不存在")
    if kind == "face":
        box = ((row.reason_json or {}).get('face_capture') or {}).get('face_crop_bbox') or ((row.reason_json or {}).get("face_crop_bbox") if row.event_type == "missing_badge" else None)
        if not isinstance(box, list) or len(box) != 4:
            raise HTTPException(status_code=404, detail="同图人脸证据不存在")
        try:
            coordinates = [float(value) for value in box]
        except (TypeError, ValueError):
            raise HTTPException(status_code=404, detail="同图人脸证据无效") from None
        if not all(math.isfinite(value) for value in coordinates):
            raise HTTPException(status_code=404, detail="同图人脸证据无效")
        with Image.open(path_value) as image:
            left, top, right, bottom = coordinates
            crop = (max(0, int(left)), max(0, int(top)),
                    min(image.width, int(math.ceil(right))), min(image.height, int(math.ceil(bottom))))
            if crop[2] - crop[0] < 10 or crop[3] - crop[1] < 10:
                raise HTTPException(status_code=404, detail="同图人脸证据无效")
            output = io.BytesIO()
            image.convert("RGB").crop(crop).save(output, format="JPEG", quality=90)
        return Response(output.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})
    media_type = "image/jpeg" if kind == "snapshot" else "video/mp4"
    return FileResponse(path_value, media_type=media_type, filename=Path(path_value).name)
