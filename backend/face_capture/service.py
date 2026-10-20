import hashlib
import os
from pathlib import Path
import time
import uuid
from datetime import datetime, timezone

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError

from .models import Capture


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def root():
    path = Path(os.getenv('FACE_STORAGE_DIR', str(Path(__file__).resolve().parents[1] / 'data' / 'face-captures')))
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_path(capture_id):
    if len(capture_id) != 64 or any(c not in '0123456789abcdef' for c in capture_id):
        raise ValueError('Invalid capture id')
    return root() / (capture_id + '.jpg')


def describe(row):
    return {name: getattr(row, name) for name in ('id', 'camera_id', 'camera_name', 'frame_id', 'captured_at',
            'bbox', 'confidence', 'model_sha256', 'status', 'attempts', 'last_error', 'delivered_at')}


def save_faces(db, owner_id, camera_id, camera_name, frame_id, faces, push_enabled=False, captured_at=None, _retry=False, test_receiver=False):
    # Stable identity for a particular source frame and detection box, independent of retries.
    records = []
    pending = db.query(Capture).filter(Capture.status.in_(['pending', 'delivering'])).count()
    total = db.query(Capture).count()
    storage_bytes = sum(p.stat().st_size for p in root().glob('*.jpg'))
    added = 0
    for face in faces:
        key = '\0'.join((str(owner_id), camera_id, frame_id, ','.join(map(str, face['bbox']))))
        capture_id = hashlib.sha256(key.encode()).hexdigest()
        existing = db.get(Capture, capture_id)
        if existing:
            records.append(existing)
            continue
        if push_enabled and pending + added >= int(os.getenv('FACE_QUEUE_LIMIT', '1000')):
            db.rollback()
            raise ValueError('人脸推送队列已满，请等待投递恢复')
        if total + added >= int(os.getenv('FACE_RECORD_LIMIT', '10000')) or storage_bytes + len(face['jpeg']) > int(os.getenv('FACE_STORAGE_LIMIT_BYTES', str(1024**3))):
            db.rollback()
            raise ValueError('人脸存储达到容量上限，请归档后继续')
        path = image_path(capture_id)
        temp = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        try:
            temp.write_bytes(face['jpeg'])
            # Use an exclusive destination: concurrent retries cannot replace a committed crop.
            try:
                os.link(temp, path)
            except FileExistsError:
                pass
        finally:
            temp.unlink(missing_ok=True)
        row = Capture(id=capture_id, owner_id=str(owner_id), camera_id=camera_id, camera_name=camera_name,
                      frame_id=frame_id, captured_at=captured_at or utcnow(), bbox=face['bbox'],
                      confidence=face['confidence'], model_sha256=face['model_sha256'],
                      image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      destination='test' if test_receiver or not os.getenv('FACE_PUSH_URL') else 'external',
                      status='pending' if push_enabled else 'stored', attempts=0, next_at=0)
        db.add(row)
        records.append(row)
        added += 1
        storage_bytes += len(face['jpeg'])
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        if _retry:
            raise
        return save_faces(db, owner_id, camera_id, camera_name, frame_id, faces, push_enabled, captured_at, True, test_receiver)
    return [describe(row) for row in records]


def claim_delivery(db):
    now = time.time()
    row = db.query(Capture).filter(Capture.status.in_(['pending', 'delivering']), Capture.next_at <= now).order_by(Capture.next_at, Capture.id).first()
    if row is None:
        return None
    capture_id = row.id
    token = uuid.uuid4().hex
    changed = db.execute(update(Capture).where(Capture.id == capture_id, Capture.status.in_(['pending', 'delivering']),
                        Capture.next_at <= now).values(status='delivering', lease=token, next_at=now + 60,
                                                     attempts=Capture.attempts + 1)).rowcount
    db.commit()
    db.expire_all()
    return (db.get(Capture, capture_id), token) if changed else None
