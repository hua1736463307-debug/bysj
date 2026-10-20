"""Authenticated internal HTTP test receiver with independently persisted receipts."""
import base64
import hashlib
import hmac
import io
import json
import os
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool
from sqlalchemy.exc import IntegrityError

from .models import Capture, Receipt
from .service import root, utcnow

RECEIVER_PATH = '/integration/face-test-receiver'


def signature(body, timestamp):
    secret = os.getenv('JWT_SECRET_KEY', '')
    if len(secret) < 32:
        raise RuntimeError('Test receiver signing key unavailable')
    key = hmac.new(secret.encode(), b'face-test-receiver-v1', hashlib.sha256).digest()
    return hmac.new(key, timestamp.encode() + b'\n' + body, hashlib.sha256).hexdigest()


def receipt_path(capture_id):
    if len(capture_id) != 64 or any(c not in '0123456789abcdef' for c in capture_id):
        raise ValueError('Invalid capture id')
    directory = root() / 'received'
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (capture_id + '.jpg')


def accept(db, body, timestamp, signed, idempotency_key):
    try:
        if abs(time.time() - int(timestamp)) > 120 or not hmac.compare_digest(signature(body, timestamp), signed):
            raise ValueError()
    except (TypeError, ValueError, RuntimeError):
        raise HTTPException(401, '接收端签名无效') from None
    try:
        payload = json.loads(body)
        capture_id = payload['capture_id']
        if payload['type'] != 'face_capture' or capture_id != idempotency_key:
            raise ValueError()
        image = base64.b64decode(payload['image_base64'], validate=True)
        digest = hashlib.sha256(image).hexdigest()
        if digest != payload['image_sha256']:
            raise ValueError()
        with Image.open(io.BytesIO(image)) as photo:
            if photo.format != 'JPEG' or photo.width * photo.height > 20_000_000:
                raise ValueError()
            photo.verify()
    except Exception:
        raise HTTPException(422, '接收数据或图片无效') from None
    capture = db.get(Capture, capture_id)
    if capture is None or capture.image_sha256 != digest or capture.destination != 'test':
        raise HTTPException(409, '抓拍记录与接收图片不匹配')
    old = db.get(Receipt, capture_id)
    if old and old.image_sha256 != digest:
        raise HTTPException(409, '重复抓拍 ID 的图片不同')
    path = receipt_path(capture_id)
    if not path.exists():
        used = sum(p.stat().st_size for p in path.parent.glob('*.jpg'))
        if used + len(image) > int(os.getenv('FACE_RECEIVER_LIMIT_BYTES', str(1024**3))):
            raise HTTPException(507, '测试接收图片容量已满')
        temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        try:
            temporary.write_bytes(image)
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise HTTPException(409, '接收图片校验失败')
    if not old:
        db.add(Receipt(id=capture_id, owner_id=capture.owner_id, camera_name=capture.camera_name,
                       camera_id=capture.camera_id, captured_at=capture.captured_at, received_at=utcnow(),
                       confidence=capture.confidence, image_sha256=digest))
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            old = db.get(Receipt, capture_id)
            if old is None or old.image_sha256 != digest:
                raise HTTPException(409, '接收记录冲突') from None
    return {'accepted': True, 'capture_id': capture_id}


def build_router(get_db, access):
    router = APIRouter(tags=['face-test-receiver'])

    @router.post(RECEIVER_PATH)
    async def ingest(request: Request, db=Depends(get_db)):
        timestamp = request.headers.get('X-Face-Timestamp', '')
        signed = request.headers.get('X-Face-Signature', '')
        if not timestamp or not signed:
            raise HTTPException(401, '需要接收端签名')
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 16 * 1024 * 1024:
                raise HTTPException(413, '接收图片过大')
        return await run_in_threadpool(accept, db, bytes(body), timestamp, signed, request.headers.get('Idempotency-Key', ''))

    @router.get('/monitoring/faces/received')
    def received(page: int = Query(1, ge=1), page_size: int = Query(12, ge=1, le=100), db=Depends(get_db), user=Depends(access)):
        query = db.query(Receipt).filter(Receipt.owner_id == str(user['id']))
        fields = ('id', 'camera_name', 'camera_id', 'captured_at', 'received_at', 'confidence', 'image_sha256')
        return {'total': query.count(), 'items': [{name: getattr(row, name) for name in fields} for row in
                query.order_by(Receipt.received_at.desc(), Receipt.id).offset((page-1)*page_size).limit(page_size).all()]}

    @router.get('/monitoring/faces/received/{capture_id}/image')
    def received_image(capture_id: str, db=Depends(get_db), user=Depends(access)):
        row = db.query(Receipt).filter(Receipt.id == capture_id, Receipt.owner_id == str(user['id'])).first()
        if row is None or not receipt_path(capture_id).is_file():
            raise HTTPException(404, '接收图片不存在')
        return FileResponse(receipt_path(capture_id), media_type='image/jpeg', headers={'Cache-Control': 'private, no-store'})
    return router
