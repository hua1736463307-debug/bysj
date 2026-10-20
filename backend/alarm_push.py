"""Durable outbound alarm delivery for the 图像训推实验平台."""

from __future__ import annotations

import json
import base64
import io
import math
import mimetypes
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from jose import JWTError, jwt
from PIL import Image
from sqlalchemy import text
from sqlalchemy.orm import Session

from behavior.rules import EVENT_NAME_MAP
from auth_utils import require_admin
from database import AlarmPushDelivery, BehaviorEvent, SessionLocal, get_db


router = APIRouter(tags=["alarm-push"])
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
MEDIA_TOKEN_ALGORITHM = "HS256"
PENDING_STATUSES = {"pending", "retrying", "waiting_evidence"}
MAX_ERROR_LENGTH = 800


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def push_target_url() -> Optional[str]:
    value = os.getenv("TRAIN_INFERENCE_ALARM_PUSH_URL", "").strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        return None
    return value


def _media_signing_key() -> str:
    return os.getenv("TRAIN_INFERENCE_ALARM_MEDIA_SIGNING_KEY", "").strip()


def _public_base_url() -> Optional[str]:
    value = os.getenv("TRAIN_INFERENCE_ALARM_PUSH_PUBLIC_BASE_URL", "").strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return value


def _media_token_minutes() -> int:
    return _bounded_int("TRAIN_INFERENCE_ALARM_MEDIA_TOKEN_MINUTES", 30, 1, 60)


def _media_url(row: BehaviorEvent, kind: str) -> Optional[str]:
    base_url = _public_base_url()
    signing_key = _media_signing_key()
    if not base_url or len(signing_key) < 32:
        return None
    token = jwt.encode(
        {
            "typ": "train_inference_alarm_push_media",
            "alarm_id": row.event_id,
            "kind": kind,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=_media_token_minutes()),
        },
        signing_key,
        algorithm=MEDIA_TOKEN_ALGORITHM,
    )
    return f"/api/integration/alarm-media/{row.event_id}/{kind}?token={quote(token, safe='')}"


def _format_time(value: Optional[datetime]) -> str:
    return value.strftime(TIME_FORMAT) if value else datetime.now().strftime(TIME_FORMAT)


def _model_info(row: BehaviorEvent) -> Tuple[str, str, Optional[str], str]:
    decision = row.decision_json if isinstance(row.decision_json, dict) else {}
    final = decision.get("final") if isinstance(decision.get("final"), dict) else {}
    temporal = decision.get("temporal") if isinstance(decision.get("temporal"), dict) else {}
    model_id = str(row.decision_model_id or temporal.get("model_id") or "rule-engine")
    model_version = str(row.decision_model_version or temporal.get("model_version") or "").strip() or None
    source = str(final.get("source") or ("model" if row.decision_model_id else "rule"))
    model_name = model_version or ("图像训推实验平台行为规则引擎" if model_id == "rule-engine" else model_id)
    return model_id, model_name, model_version, source


def _confidence(row: BehaviorEvent) -> float:
    decision = row.decision_json if isinstance(row.decision_json, dict) else {}
    final = decision.get("final") if isinstance(decision.get("final"), dict) else {}
    temporal = decision.get("temporal") if isinstance(decision.get("temporal"), dict) else {}
    value = temporal.get("confidence") if final.get("source") == "temporal_fusion" else row.rule_score
    try:
        return round(float(value or 0.0), 6)
    except (TypeError, ValueError):
        return 0.0


def _box_from_reason(row: BehaviorEvent) -> list[Dict[str, Any]]:
    reason = row.reason_json if isinstance(row.reason_json, dict) else {}
    candidate = reason.get("person_bbox") or reason.get("object_bbox")
    if not isinstance(candidate, (list, tuple)) or len(candidate) < 4:
        return []
    try:
        x1, y1, x2, y2 = [float(value) for value in candidate[:4]]
    except (TypeError, ValueError):
        return []
    width = max(0, round(x2 - x1))
    height = max(0, round(y2 - y1))
    if width == 0 or height == 0:
        return []
    return [{
        "x": max(0, round(x1)),
        "y": max(0, round(y1)),
        "width": width,
        "height": height,
        "confidence": _confidence(row),
        "classId": row.event_type,
        "className": (row.reason_json or {}).get("algorithm_name") or EVENT_NAME_MAP.get(row.event_type, row.event_type),
    }]


def _image_info(row: BehaviorEvent) -> Optional[Dict[str, Any]]:
    if not row.snapshot_path:
        return None
    image_url = _media_url(row, "image")
    if not image_url:
        return None
    info: Dict[str, Any] = {
        "imageId": f"{row.event_id}:image",
        "imageUrl": image_url,
        "receiveTime": _format_time(row.alerted_at or row.confirmed_at),
    }
    try:
        with Image.open(row.snapshot_path) as image:
            width, height = image.size
        info["width"] = int(width)
        info["height"] = int(height)
    except (OSError, ValueError):
        pass
    boxes = _box_from_reason(row)
    if boxes:
        info["objectNum"] = len(boxes)
        info["boxs"] = boxes
    return info


CAMERA_DEVICE_SERIALS = {
    "127.0.0.1": "DS-2CD3346P1-I20231108AACHAW1333891",
    "127.0.0.1": "DS-2CD2345CV4-I20220625AACHK20849228",
}


def receiver_device_id(source_id: str) -> str:
    try:
        host = urlparse(source_id).hostname
    except ValueError:
        host = None
    return CAMERA_DEVICE_SERIALS.get(host or source_id, source_id)


def build_alarm_push_payload(row: BehaviorEvent) -> Dict[str, Any]:
    """Map one confirmed behavior event to the documented `items` contract."""
    model_id, model_name, model_version, decision_source = _model_info(row)
    source_id = str(row.source_id or "")
    task_id = str(row.session_id or row.grid_id or source_id or row.event_id)
    task_name = f"{row.mode or 'realtime'}:{source_id or task_id}"
    item: Dict[str, Any] = {
        "heId": row.event_id,
        "taskId": task_id,
        "taskName": task_name,
        "algoId": row.event_type,
        "algoName": (row.reason_json or {}).get("algorithm_name") or EVENT_NAME_MAP.get(row.event_type, row.event_type),
        "eventTime": _format_time(row.alerted_at or row.confirmed_at),
        "reportTime": _format_time(row.created_at),
        "dsInfo": {
            "dsId": receiver_device_id(source_id),
            "dsType": "camera" if row.mode in {"realtime", "grid"} else "xds",
            "dsRaw": {"deviceId": receiver_device_id(source_id)},
        },
        "ctx": {
            "alarmType": row.event_type,
            "alarmName": (row.reason_json or {}).get("algorithm_name") or EVENT_NAME_MAP.get(row.event_type, row.event_type),
            "severity": row.severity or "low",
            "confidence": _confidence(row),
            "model": {
                "modelId": model_id,
                "modelName": model_name,
                "modelVersion": model_version,
                "decisionSource": decision_source,
            },
        },
    }
    capture = (row.reason_json or {}).get('face_capture')
    if isinstance(capture, dict):
        item['ctx']['faceCapture'] = face_capture_payload(row, capture)
    image_info = _image_info(row)
    if image_info:
        item["imageInfo"] = image_info
    video_url = _media_url(row, "video") if row.clip_path else None
    if video_url:
        item["mps"] = video_url
    return {"items": [item]}


def face_capture_payload(row: BehaviorEvent, capture: Dict[str, Any]) -> Dict[str, Any]:
    """Same-frame face evidence; identification and review belong to the receiver."""
    payload = {
        'status': capture['status'],
        'alarmPolicy': capture.get('alarm_policy', 'manual_review' if capture['affect_alarm'] else 'ignore'),
        'affectsAlarm': bool(capture['affect_alarm']),
        'requiresManualConfirmation': bool(capture['requires_manual_confirmation']),
        'comparisonBy': 'business_system',
        'reason': capture.get('reason'),
    }
    if capture['status'] != 'detected':
        return payload
    bounds = capture.get('face_crop_bbox')
    if not isinstance(bounds, list) or len(bounds) != 4 or not all(type(v) in (int, float) and math.isfinite(v) for v in bounds):
        raise ValueError('face_evidence_invalid')
    root = Path(os.getenv('DEEPVISION_BEHAVIOR_EVENT_DIR', '/app/static/behavior_events')).resolve()
    path = Path(row.snapshot_path or '').resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('face_evidence_unavailable')
    with Image.open(path) as image:
        left, top, right, bottom = bounds
        crop = (math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom))
        if not (0 <= crop[0] < crop[2] <= image.width and 0 <= crop[1] < crop[3] <= image.height):
            raise ValueError('face_evidence_invalid')
        output = io.BytesIO()
        image.convert('RGB').crop(crop).save(output, format='JPEG', quality=95)
        data = output.getvalue()
        if len(data) > 2 * 1024 ** 2:
            raise ValueError('face_evidence_too_large')
        payload.update(imageMime='image/jpeg', imageBase64=base64.b64encode(data).decode('ascii'),
                       width=crop[2]-crop[0], height=crop[3]-crop[1],
                       faceBbox=capture.get('face_bbox'), confidence=capture.get('face_confidence'))
    return payload


def enqueue_alarm_pushes(db: Session, rows: Iterable[BehaviorEvent]) -> int:
    """Create one delivery per new alert in the caller's transaction."""
    target_url = push_target_url()
    if not target_url:
        return 0
    created = 0
    for row in rows:
        if not row.should_alert or not row.event_id:
            continue
        if db.bind and db.bind.dialect.name == "postgresql":
            result = db.execute(
                text(
                    "INSERT INTO alarm_push_deliveries "
                    "(id, event_id, destination_url, status, attempt_count, next_attempt_at, created_at, updated_at) "
                    "VALUES (:id, :event_id, :destination_url, 'pending', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                    "ON CONFLICT (event_id) DO NOTHING"
                ),
                {
                    "id": str(uuid.uuid4()),
                    "event_id": row.event_id,
                    "destination_url": target_url,
                },
            )
            created += int(result.rowcount or 0)
            continue
        exists = db.query(AlarmPushDelivery.id).filter(AlarmPushDelivery.event_id == row.event_id).first()
        if exists:
            continue
        db.add(AlarmPushDelivery(event_id=row.event_id, destination_url=target_url, status="pending"))
        created += 1
    return created


def _evidence_wait_seconds() -> int:
    return _bounded_int("TRAIN_INFERENCE_ALARM_PUSH_EVIDENCE_WAIT_SECONDS", 90, 0, 3600)


def _retry_delay_seconds(attempt_count: int) -> int:
    base = _bounded_int("TRAIN_INFERENCE_ALARM_PUSH_RETRY_BASE_SECONDS", 5, 1, 300)
    maximum = _bounded_int("TRAIN_INFERENCE_ALARM_PUSH_RETRY_MAX_SECONDS", 300, base, 3600)
    return min(maximum, base * (2 ** max(0, attempt_count - 1)))


def _max_attempts() -> int:
    return _bounded_int("TRAIN_INFERENCE_ALARM_PUSH_MAX_ATTEMPTS", 8, 1, 100)


def configure_alarm_push_db_session(db: Session) -> None:
    """Bound worker-side PostgreSQL waits without changing global settings."""
    if not db.bind or db.bind.dialect.name != "postgresql":
        return
    lock_timeout_ms = _bounded_int("ALARM_PUSH_WORKER_LOCK_TIMEOUT_MS", 5000, 100, 60000)
    statement_timeout_ms = _bounded_int("ALARM_PUSH_WORKER_STATEMENT_TIMEOUT_MS", 15000, 1000, 120000)
    db.execute(
        text("select set_config('lock_timeout', :value, true)"),
        {"value": f"{lock_timeout_ms}ms"},
    )
    db.execute(
        text("select set_config('statement_timeout', :value, true)"),
        {"value": f"{statement_timeout_ms}ms"},
    )


def recover_stale_alarm_pushes(db: Session, now: Optional[datetime] = None) -> int:
    now = now or datetime.now()
    configure_alarm_push_db_session(db)
    stale_before = now - timedelta(seconds=_bounded_int("TRAIN_INFERENCE_ALARM_PUSH_REQUEST_TIMEOUT_SECONDS", 10, 1, 60) * 2)
    rows = db.query(AlarmPushDelivery).filter(
        AlarmPushDelivery.status == "delivering",
        AlarmPushDelivery.last_attempt_at < stale_before,
    ).with_for_update(skip_locked=True).all()
    for row in rows:
        row.status = "retrying"
        row.next_attempt_at = now
        row.last_error = "投递工作器在请求完成前退出，已重新排队"
    db.commit()
    return len(rows)


def claim_next_alarm_push(db: Session, now: Optional[datetime] = None) -> Optional[str]:
    now = now or datetime.now()
    configure_alarm_push_db_session(db)
    row = (
        db.query(AlarmPushDelivery)
        .filter(AlarmPushDelivery.status.in_(PENDING_STATUSES), AlarmPushDelivery.next_attempt_at <= now)
        .order_by(AlarmPushDelivery.next_attempt_at.asc(), AlarmPushDelivery.created_at.asc())
        .with_for_update(skip_locked=True)
        .first()
    )
    if row is None:
        db.rollback()
        return None
    row.status = "delivering"
    row.last_attempt_at = now
    db.commit()
    return row.id


def _retryable(status_code: Optional[int]) -> bool:
    return status_code is None or status_code == 408 or status_code == 429 or status_code >= 500


def _safe_error(value: Any) -> str:
    return str(value or "未知错误").replace("\n", " ").replace("\r", " ")[:MAX_ERROR_LENGTH]


def _defer_for_evidence(db: Session, delivery: AlarmPushDelivery, now: datetime) -> bool:
    event = db.query(BehaviorEvent).filter(
        BehaviorEvent.event_id == delivery.event_id,
        BehaviorEvent.should_alert.is_(True),
    ).first()
    if event is None:
        delivery.status = "dead"
        delivery.last_error = "告警记录不存在或已不再是告警"
        db.commit()
        return True
    wait_until = delivery.created_at + timedelta(seconds=_evidence_wait_seconds())
    if event.evidence_status == "collecting" and now < wait_until:
        delivery.status = "waiting_evidence"
        delivery.next_attempt_at = min(wait_until, now + timedelta(seconds=5))
        delivery.last_error = None
        db.commit()
        return True
    return False


def _post_json(url: str, payload: Dict[str, Any]) -> Tuple[int, str]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json;charset=UTF-8", "Accept": "application/json"},
        method="POST",
    )
    timeout = _bounded_int("TRAIN_INFERENCE_ALARM_PUSH_REQUEST_TIMEOUT_SECONDS", 10, 1, 60)
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read(4096).decode("utf-8", errors="replace")
    except HTTPError as exc:
        return int(exc.code), exc.read(4096).decode("utf-8", errors="replace")
    except (URLError, OSError) as exc:
        raise RuntimeError(_safe_error(exc)) from exc


def _is_success_response(status_code: int, body: str) -> bool:
    if not 200 <= status_code < 300:
        return False
    try:
        response = json.loads(body)
    except json.JSONDecodeError:
        return False
    return isinstance(response, dict) and response.get("code") == 200


def process_alarm_push(
    session_factory: Callable[[], Session] = SessionLocal,
    sender: Callable[[str, Dict[str, Any]], Tuple[int, str]] = _post_json,
    now: Optional[datetime] = None,
) -> str:
    """Process one record. The HTTP request runs outside all DB transactions."""
    now = now or datetime.now()
    claim_db = session_factory()
    try:
        delivery_id = claim_next_alarm_push(claim_db, now)
    finally:
        claim_db.close()
    if not delivery_id:
        return "idle"

    prepare_db = session_factory()
    try:
        configure_alarm_push_db_session(prepare_db)
        delivery = prepare_db.query(AlarmPushDelivery).filter(AlarmPushDelivery.id == delivery_id).with_for_update().first()
        if delivery is None or delivery.status != "delivering":
            prepare_db.rollback()
            return "skipped"
        if _defer_for_evidence(prepare_db, delivery, now):
            return "deferred"
        event = prepare_db.query(BehaviorEvent).filter(
            BehaviorEvent.event_id == delivery.event_id,
            BehaviorEvent.should_alert.is_(True),
        ).first()
        if event is None:
            delivery.status = "dead"
            delivery.last_error = "告警记录不存在或已不再是告警"
            prepare_db.commit()
            return "dead"
        destination_url = delivery.destination_url
        # Build/media errors follow the same durable retry path as HTTP failures.
        payload_error = None
        try:
            payload = build_alarm_push_payload(event)
        except Exception:
            payload, payload_error = None, 'alarm_payload_evidence_unavailable'
        prepare_db.rollback()
    finally:
        prepare_db.close()

    status_code: Optional[int] = None
    body = ""
    error: Optional[str] = None
    try:
        if payload_error:
            raise ValueError(payload_error)
        status_code, body = sender(destination_url, payload)
        if not _is_success_response(status_code, body):
            error = f"接收方未确认: HTTP {status_code}, body={_safe_error(body)}"
    except Exception as exc:
        error = _safe_error(exc)

    finish_db = session_factory()
    try:
        configure_alarm_push_db_session(finish_db)
        delivery = finish_db.query(AlarmPushDelivery).filter(AlarmPushDelivery.id == delivery_id).with_for_update().first()
        if delivery is None or delivery.status != "delivering":
            finish_db.rollback()
            return "skipped"
        delivery.last_response_code = status_code
        if error is None:
            delivery.status = "succeeded"
            delivery.delivered_at = now
            delivery.last_error = None
            delivery.attempt_count += 1
            finish_db.commit()
            return "succeeded"
        delivery.attempt_count += 1
        delivery.last_error = error
        if (
            (status_code is not None and 200 <= status_code < 300) or _retryable(status_code)
        ) and delivery.attempt_count < _max_attempts():
            delivery.status = "retrying"
            delivery.next_attempt_at = now + timedelta(seconds=_retry_delay_seconds(delivery.attempt_count))
            result = "retrying"
        else:
            delivery.status = "dead"
            result = "dead"
        finish_db.commit()
        return result
    finally:
        finish_db.close()


def _validate_media_token(alarm_id: str, kind: str, token: str) -> None:
    key = _media_signing_key()
    if len(key) < 32:
        raise HTTPException(status_code=503, detail="告警推送媒体签名密钥未配置")
    try:
        payload = jwt.decode(token, key, algorithms=[MEDIA_TOKEN_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="告警媒体链接无效或已过期") from exc
    if (
        payload.get("typ") != "train_inference_alarm_push_media"
        or payload.get("alarm_id") != alarm_id
        or payload.get("kind") != kind
    ):
        raise HTTPException(status_code=401, detail="告警媒体链接无效")


@router.post("/api/admin/alarm-push/{alarm_id}/retry")
def retry_dead_alarm_push(
    alarm_id: str,
    db: Session = Depends(get_db),
    _admin: Dict[str, Any] = Depends(require_admin),
) -> Dict[str, Any]:
    delivery = db.query(AlarmPushDelivery).filter(AlarmPushDelivery.event_id == alarm_id).with_for_update().first()
    if delivery is None:
        raise HTTPException(status_code=404, detail="告警投递记录不存在")
    if delivery.status == "succeeded":
        raise HTTPException(status_code=409, detail="告警已成功推送，不能重复投递")
    delivery.status = "retrying"
    delivery.next_attempt_at = datetime.now()
    delivery.last_error = "管理员手动重试"
    db.commit()
    return {"code": 200, "msg": "已重新加入告警推送队列"}


@router.get("/integration/alarm-media/{alarm_id}/{kind}", include_in_schema=False)
@router.get("/api/integration/alarm-media/{alarm_id}/{kind}")
def get_alarm_push_media(alarm_id: str, kind: str, token: str, db: Session = Depends(get_db)):
    if kind not in {"image", "video"}:
        raise HTTPException(status_code=404, detail="媒体类型不存在")
    _validate_media_token(alarm_id, kind, token)
    row = db.query(BehaviorEvent).filter(
        BehaviorEvent.event_id == alarm_id,
        BehaviorEvent.should_alert.is_(True),
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="告警不存在")
    value = row.snapshot_path if kind == "image" else row.clip_path
    root = Path(os.getenv("TRAIN_INFERENCE_BEHAVIOR_EVENT_DIR", "/app/static/behavior_events")).resolve()
    path = Path(value or "").resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="告警媒体不存在") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="告警媒体不存在")
    media_type = mimetypes.guess_type(path.name)[0] or ("image/jpeg" if kind == "image" else "video/mp4")
    return FileResponse(path, media_type=media_type)
