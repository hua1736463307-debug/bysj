import hashlib
import io
import os
import threading
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from auth_utils import require_any_menu
from database import SessionLocal, engine, get_db, RecorderConfig
from .models import Base, Capture, Source
from .detector import detect, model_status
from .service import describe, image_path, save_faces
from .worker import Workers
from .sampling import cancel_source_frames

router = APIRouter(prefix='/monitoring/faces', tags=['face-capture'])
access = require_any_menu('/monitoring/config')
test_access = require_any_menu('/monitoring/config', '/alarm-center')
slots = threading.BoundedSemaphore(1)


class ManualTestRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    request_id: uuid.UUID


@router.post('/sources/{recorder_id}/{channel_number}/manual-test')
def manual_test(recorder_id: str, channel_number: str, body: ManualTestRequest,
                db=Depends(get_db), user=Depends(access)):
    if not slots.acquire(blocking=False):
        raise HTTPException(429, '人脸测试繁忙，请等待当前测试完成')
    try:
        from .manual_test import snapshot_test
        return snapshot_test(db, str(user['id']), recorder_id, channel_number, body.request_id)
    finally:
        slots.release()


@router.get('/manual-tests/{test_id}/image')
def manual_test_image(test_id: uuid.UUID, user=Depends(access)):
    from .manual_test import results
    return Response(content=results.image(str(user['id']), test_id), media_type='image/png',
                    headers={'Cache-Control': 'private, no-store'})


def source_dict(row):
    values = {name: getattr(row, name) for name in ('id', 'recorder_id', 'channel_number', 'enabled', 'push_enabled',
            'interval_seconds', 'threshold', 'min_size', 'last_error', 'last_at', 'last_count')}
    return {**values, 'sampling_mode': 'alarm_frame_person_rois'}


@router.get('/status')
def status(user=Depends(test_access)):
    return {**model_status(), 'sampling_mode': 'alarm_frame_person_rois', 'push_configured': True, 'push_destination': 'external' if os.getenv('FACE_PUSH_URL') else 'test'}


@router.get('/sources')
def sources(db=Depends(get_db), user=Depends(access)):
    return {'items': [source_dict(row) for row in db.query(Source).filter(Source.owner_id == str(user['id'])).all()]}


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    enabled: bool = False
    push_enabled: bool = False
    interval_seconds: float = Field(default=.5, ge=.1, le=86400, allow_inf_nan=False)
    threshold: float = Field(default=.6, ge=.1, le=.99, allow_inf_nan=False)
    min_size: int = Field(default=64, ge=4, le=1000)


@router.put('/sources/{recorder_id}/{channel_number}')
def configure(recorder_id: str, channel_number: str, body: Settings, db=Depends(get_db), user=Depends(access)):
    recorder = db.query(RecorderConfig).filter(RecorderConfig.id == recorder_id, RecorderConfig.owner_id == str(user['id'])).first()
    channel = next((c for c in (recorder.config_json if recorder else {}).get('channels', []) if c['number'] == channel_number), None)
    if not channel:
        raise HTTPException(404, '摄像头不存在')
    if body.enabled and (not recorder.config_json.get('enabled') or not channel.get('enabled') or not channel.get('rtsp_path')):
        raise HTTPException(422, '摄像头未启用或未配置视频路径')
    if body.enabled and not model_status()['available']:
        raise HTTPException(503, '人脸模型不可用')
    key = hashlib.sha256(f"{user['id']}\0{recorder_id}\0{channel_number}".encode()).hexdigest()
    row = db.query(Source).filter(Source.id == key).with_for_update().first()
    if row is None:
        row = Source(id=key, owner_id=str(user['id']), recorder_id=recorder_id, channel_number=channel_number)
        db.add(row)
    for name, value in body.model_dump().items():
        setattr(row, name, value)
    # Invalidate in-flight output when settings or enabled state change.
    row.lease = None
    row.lease_until = 0
    row.next_at = 0
    row.last_error = None
    cancel_source_frames(db, key)
    db.commit()
    from .signals import capture_ready
    from .streams import close_camera
    if not body.enabled:
        close_camera((str(user['id']), recorder_id, channel_number))
    capture_ready.set()
    return source_dict(row)


@router.post('/detect')
def detect_upload(file: UploadFile = File(...), push_enabled: bool = Form(False),
                  test_receiver: bool = Form(False),
                  request_id: str = Form('', max_length=100), threshold: float = Form(.8, ge=.1, le=.99),
                  db=Depends(get_db), user=Depends(test_access)):
    if not slots.acquire(blocking=False):
        raise HTTPException(429, '人脸检测繁忙，请稍后重试')
    try:
        content = file.file.read(12 * 1024 * 1024 + 1)
        if len(content) > 12 * 1024 * 1024:
            raise HTTPException(413, '图片不能超过 12 MB')
        try:
            with Image.open(io.BytesIO(content)) as image:
                if image.width * image.height > 20_000_000:
                    raise HTTPException(413, '图片不能超过 2000 万像素')
                image.verify()
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(422, '请上传有效图片') from None
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        faces = detect(image, threshold)
        # Include image content in retry identity to prevent accidental reuse with another image.
        frame_id = hashlib.sha256((request_id or uuid.uuid4().hex).encode() + content).hexdigest()
        items = save_faces(db, str(user['id']), 'upload', '图片测试', frame_id, faces, push_enabled, test_receiver=test_receiver)
        from .signals import delivery_ready
        delivery_ready.set()
        return {'items': items, 'count': len(faces)}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from None
    finally:
        slots.release()


@router.get('/captures')
def captures(page: int = Query(1, ge=1), page_size: int = Query(12, ge=1, le=100),
             db=Depends(get_db), user=Depends(test_access)):
    query = db.query(Capture).filter(Capture.owner_id == str(user['id']))
    return {'total': query.count(), 'items': [describe(row) for row in query.order_by(Capture.captured_at.desc(), Capture.id).offset((page-1)*page_size).limit(page_size).all()]}


def owned_capture(db, user, capture_id):
    row = db.query(Capture).filter(Capture.id == capture_id, Capture.owner_id == str(user['id'])).first()
    if row is None:
        raise HTTPException(404, '抓拍记录不存在')
    return row


@router.get('/captures/{capture_id}/image')
def image(capture_id: str, db=Depends(get_db), user=Depends(test_access)):
    owned_capture(db, user, capture_id)
    path = image_path(capture_id)
    if not path.is_file():
        raise HTTPException(404, '图片文件不存在')
    return FileResponse(path, media_type='image/jpeg', headers={'Cache-Control': 'private, no-store'})


@router.post('/captures/{capture_id}/retry')
def retry(capture_id: str, db=Depends(get_db), user=Depends(test_access)):
    row = owned_capture(db, user, capture_id)
    if row.destination == 'external' and not os.getenv('FACE_PUSH_URL'):
        raise HTTPException(422, '尚未配置人脸推送接收端')
    from sqlalchemy import update
    changed = db.execute(update(Capture).where(Capture.id == row.id, Capture.status.in_(['failed', 'stored'])).values(
        status='pending', attempts=0, next_at=0, last_error=None)).rowcount
    db.commit()
    if not changed:
        raise HTTPException(409, '仅未推送或失败记录可重新入队')
    db.refresh(row)
    from .signals import delivery_ready
    delivery_ready.set()
    return describe(row)


def register(app, live_context=None):
    from .streams import bind_live_context
    bind_live_context(live_context)
    workers = Workers(SessionLocal)
    def startup():
        Base.metadata.create_all(engine)
        if model_status()['available']:
            detect(np.zeros((320, 320, 3), dtype=np.uint8))
        workers.start()
    app.include_router(router)
    from .receiver import build_router
    app.include_router(build_router(get_db, test_access))
    app.on_event('startup')(startup)
    app.on_event('shutdown')(workers.stop)
    from .manual_test import results
    app.on_event('shutdown')(results.clear)
