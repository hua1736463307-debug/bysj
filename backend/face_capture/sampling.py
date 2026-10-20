"""Durable, bounded native-frame handoff from alarm monitoring to CPU face work."""
import hashlib
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
from sqlalchemy import and_, or_, text, update

from database import ContinuousMonitor, SessionLocal
from .models import Capture, FrameSample, Source
from .service import root, save_faces
from .signals import capture_ready

ACTIVE = ('pending', 'processing')
TERMINAL = ('completed', 'cancelled', 'failed')
QUEUE_LOCK = 731846250987612034
PERSON_NAMES = {'person', 'people', 'human', '人员', '人', '行人', '人体'}


def limits():
    return {
        'count': max(1, min(128, int(os.getenv('FACE_FRAME_QUEUE_LIMIT', '16')))),
        'camera': max(1, min(8, int(os.getenv('FACE_FRAME_CAMERA_LIMIT', '2')))),
        'bytes': max(1024, int(os.getenv('FACE_FRAME_DISK_LIMIT_BYTES', str(256 * 1024**2)))),
        'frame_bytes': max(1024, int(os.getenv('FACE_FRAME_MAX_BYTES', str(64 * 1024**2)))),
        'pixels': max(1, int(os.getenv('FACE_FRAME_MAX_PIXELS', '32000000'))),
        'ttl': max(1., min(600., float(os.getenv('FACE_FRAME_TTL_SECONDS', '180')))),
    }


def frame_path(frame_id):
    if not re.fullmatch(r'[0-9a-f]{64}', frame_id):
        raise ValueError('invalid_frame_id')
    directory = root() / 'sample-frames'
    directory.mkdir(exist_ok=True)
    path = directory / (frame_id + '.png')
    if path.is_symlink():
        raise ValueError('invalid_frame_path')
    return path


def _bounded_transaction(db, queue=False):
    if db.bind.dialect.name == 'postgresql':
        db.execute(text("SET LOCAL statement_timeout='3000ms'"))
        db.execute(text("SET LOCAL lock_timeout='250ms'"))
        if queue:
            db.execute(text('SELECT pg_advisory_xact_lock(:key)'), {'key': QUEUE_LOCK})


def _settings(source):
    return {k: getattr(source, k) for k in ('enabled', 'push_enabled', 'threshold', 'min_size')}


def _owns_source(source, owner, camera):
    return source is not None and str(source.owner_id) == str(owner) and source.recorder_id + ':' + source.channel_number == camera


def _monitor_source(db, monitor_id, owner, camera_id, run_id, lock=False, require_running=False):
    query = db.query(ContinuousMonitor).filter(ContinuousMonitor.id == monitor_id)
    monitor = (query.with_for_update() if lock else query).first()
    if not monitor or str(monitor.owner_id) != str(owner) or monitor.desired_state != 'running' or (require_running and monitor.status != 'running'):
        return None
    for source in (monitor.config_json or {}).get('video_sources', []):
        if str(source.get('recorder_id', '')) + ':' + str(source.get('channel_number', '')) != camera_id:
            continue
        if source.get('desired_state', 'running') != 'running':
            return None
        if source.get('run_id') and str(source['run_id']) != run_id:
            return None
        return source
    return None


def person_boxes(detections, width, height):
    boxes = []
    for detection in detections or []:
        if not isinstance(detection, dict):
            continue
        label = detection.get('class_name') or detection.get('class') or detection.get('label')
        if str(label).strip().lower() not in PERSON_NAMES:
            continue
        box = detection.get('bbox')
        try:
            if isinstance(box, dict):
                box = ([box[k] for k in ('x1', 'y1', 'x2', 'y2')] if 'x1' in box else
                       [box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']])
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            if not all(math.isfinite(float(v)) and not isinstance(v, bool) for v in box):
                continue
            left, top, right, bottom = map(float, box)
            left, top = max(0, math.floor(left)), max(0, math.floor(top))
            right, bottom = min(width, math.ceil(right)), min(height, math.ceil(bottom))
            if right <= left or bottom <= top:
                continue
            candidate = [left, top, right, bottom]
            if not any(_iou(candidate, other) > .85 for other in boxes):
                boxes.append(candidate)
        except (TypeError, ValueError, KeyError, OverflowError):
            continue
    if len(boxes) > 32:
        raise ValueError('too_many_person_regions')
    return sorted(boxes)


def _iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / area if area > 0 else 0.


def submit_alarm_frame(image_rgb, detections, context, *, session_key, factory=SessionLocal, now=None):
    """Called once AFTER this alarm frame's person results, BEFORE preview/filtering."""
    now = time.time() if now is None else now
    policy = limits()
    match = re.fullmatch(r'recorder:([^:]{1,64}):channel:([^:]{1,64})', str(context.get('source_id', '')))
    grid = str(context.get('grid_id', ''))
    owner = str(context.get('owner_id', ''))
    run_id = str(context.get('run_id', ''))
    if 'monitoring_algorithms' not in context or not match or not grid.startswith('monitor_') or not owner or not run_id or len(run_id) > 64:
        return {'status': 'not_alarm_source'}
    monitor_id = grid[len('monitor_'):]
    try:
        uuid.UUID(monitor_id)
        captured = float(context['capture_timestamp'])
        if not math.isfinite(captured) or captured > now + 5 or now - captured >= policy['ttl']:
            return {'status': 'expired_frame'}
        if not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return {'status': 'invalid_frame'}
        height, width = image_rgb.shape[:2]
        if not height or not width or height * width > policy['pixels']:
            return {'status': 'frame_too_large'}
        same_frame = [d for d in detections or [] if isinstance(d, dict) and
                      (d.get('frameId') is None or str(d['frameId']) == str(context.get('_monitor_source_sequence')))]
        boxes = person_boxes(same_frame, width, height)
    except (ValueError, TypeError, KeyError):
        return {'status': 'invalid_signal'}
    recorder, channel = match.groups()
    camera = recorder + ':' + channel
    source_id = hashlib.sha256(f'{owner}\0{recorder}\0{channel}'.encode()).hexdigest()
    sequence = str(context.get('_monitor_capture_sequence', ''))[:64]
    frame_key = f'{session_key}:{run_id}:{sequence}:{captured:.9f}'
    if len(frame_key) > 256:
        return {'status': 'invalid_signal'}
    frame_id = hashlib.sha256((owner + '\0' + camera + '\0' + frame_key).encode()).hexdigest()
    captured_at = datetime.fromtimestamp(captured, timezone.utc).isoformat()
    with factory() as db:
        _bounded_transaction(db)
        source = db.get(Source, source_id)
        reference = _monitor_source(db, monitor_id, owner, camera, run_id, require_running=True)
        if not _owns_source(source, owner, camera) or not source.enabled or not reference:
            return {'status': 'disabled_or_stopped'}
        settings = _settings(source)
        if db.get(FrameSample, frame_id):
            return {'status': 'duplicate', 'frame_id': frame_id}
        if not boxes:
            source.last_at, source.last_count, source.last_error = captured_at, 0, None
            db.commit()
            return {'status': 'no_person', 'frame_id': frame_id}
    # PNG is lossless; no resize, annotation, JPEG preview, or independent RTSP read.
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ValueError('frame_encoding_failed')
    data = encoded.tobytes()
    if len(data) > policy['frame_bytes']:
        return {'status': 'frame_too_large'}
    digest = hashlib.sha256(data).hexdigest()
    with factory() as db:
        _bounded_transaction(db, queue=True)
        source = db.query(Source).filter(Source.id == source_id).with_for_update().first()
        reference = _monitor_source(db, monitor_id, owner, camera, run_id, lock=True, require_running=True)
        if not _owns_source(source, owner, camera) or _settings(source) != settings or not source.enabled or not reference:
            return {'status': 'disabled_or_stopped'}
        if db.get(FrameSample, frame_id):
            return {'status': 'duplicate', 'frame_id': frame_id}
        active = db.query(FrameSample).filter(FrameSample.status.in_(ACTIVE))
        directory = frame_path(frame_id).parent
        used = sum(f.stat().st_size for pattern in ('*.png', '*.tmp') for f in directory.glob(pattern) if not f.is_symlink())
        addition = 0 if frame_path(frame_id).is_file() else len(data)
        if active.count() >= policy['count'] or active.filter(FrameSample.source_id == source_id).count() >= policy['camera'] or used + addition > policy['bytes']:
            source.last_at, source.last_count = captured_at, 0
            source.last_error = '人脸采样队列繁忙，本帧未采样，等待下一次告警取帧'
            db.commit()
            return {'status': 'queue_full'}
        path = frame_path(frame_id)
        temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        try:
            temporary.write_bytes(data)
            os.chmod(temporary, 0o600)
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.stat().st_size > policy['frame_bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError('frame_identity_conflict')
        finally:
            temporary.unlink(missing_ok=True)
        db.add(FrameSample(id=frame_id, owner_id=owner, source_id=source_id, monitor_id=monitor_id,
                           run_id=run_id, source_sequence=sequence, alarm_frame_index=str(context.get('_monitor_source_sequence', ''))[:64],
                           frame_key=frame_key, camera_id=camera, camera_name=str(reference.get('name') or f'摄像头 {recorder[:8]} / 通道 {channel}')[:200],
                           captured_at=captured_at, width=width, height=height, person_boxes=boxes, settings_json=settings,
                           image_sha256=digest, byte_count=len(data), status='pending', attempts=0,
                           created_at=now, expires_at=captured + policy['ttl'], lease_until=0, face_count=0))
        db.commit()
    capture_ready.set()
    return {'status': 'queued', 'frame_id': frame_id, 'person_count': len(boxes), 'width': width, 'height': height}


def cancel_source_frames(db, source_id):
    """Use within the settings transaction; invalidate in-flight worker ownership."""
    db.execute(update(FrameSample).where(FrameSample.source_id == source_id, FrameSample.status.in_(ACTIVE)).values(
        status='cancelled', lease=None, lease_until=0, completed_at=time.time(), last_error='source_settings_changed'))


def claim_frame(factory, now=None):
    now = time.time() if now is None else now
    with factory() as db:
        _bounded_transaction(db)
        eligible = or_(FrameSample.status == 'pending', and_(FrameSample.status == 'processing', FrameSample.lease_until <= now))
        row = db.query(FrameSample).filter(eligible, FrameSample.expires_at > now).order_by(FrameSample.created_at).with_for_update(skip_locked=True).first()
        if row is None:
            return None
        source = db.get(Source, row.source_id)
        if row.attempts >= 3 or not _owns_source(source, row.owner_id, row.camera_id) or not source.enabled or _settings(source) != row.settings_json or not _monitor_source(db, row.monitor_id, row.owner_id, row.camera_id, row.run_id):
            row.status, row.completed_at, row.last_error = 'cancelled', now, 'inactive_or_attempts_exhausted'
            row.lease, row.lease_until = None, 0
            db.commit()
            return None
        token = uuid.uuid4().hex
        changed = db.execute(update(FrameSample).where(FrameSample.id == row.id, eligible, FrameSample.expires_at > now).values(
            status='processing', lease=token, lease_until=now + 120, attempts=FrameSample.attempts + 1)).rowcount
        db.commit()
        if not changed:
            return None
        db.refresh(row)
        return ({name: getattr(row, name) for name in ('id', 'owner_id', 'source_id', 'monitor_id', 'run_id', 'camera_id',
                 'camera_name', 'captured_at', 'width', 'height', 'person_boxes', 'settings_json', 'image_sha256', 'byte_count', 'expires_at')}, token)


def detect_person_faces(image_bgr, boxes, detector, settings, stop_event=None):
    """Detect only within same-frame person boxes; restore original image coordinates."""
    faces = []
    for left, top, right, bottom in boxes:
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError('worker_stopped')
        roi = image_bgr[top:bottom, left:right]
        for face in detector(roi, settings['threshold'], settings['min_size']):
            item = dict(face)
            for name in ('bbox', 'detected_bbox'):
                if name in item:
                    x1, y1, x2, y2 = item[name]
                    if not (0 <= x1 < x2 <= right - left and 0 <= y1 < y2 <= bottom - top):
                        raise ValueError('invalid_face_coordinates')
                    item[name] = [x1 + left, y1 + top, x2 + left, y2 + top]
            item['person_bbox'] = [left, top, right, bottom]
            faces.append(item)
    faces.sort(key=lambda f: f['confidence'], reverse=True)
    unique = []
    for face in faces:
        if not any(_iou(face.get('detected_bbox', face['bbox']), other.get('detected_bbox', other['bbox'])) > .65 for other in unique):
            unique.append(face)
    if len(unique) > 64:
        raise ValueError('too_many_faces')
    return sorted(unique, key=lambda f: tuple(f['bbox']))


def process_frame(factory, claimed, detector, stop_event=None):
    sample, token = claimed
    path = frame_path(sample['id'])
    policy = limits()
    if path.stat().st_size > policy['frame_bytes'] or path.stat().st_size != sample['byte_count']:
        raise ValueError('frame_integrity')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sample['image_sha256']:
        raise ValueError('frame_integrity')
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (sample['height'], sample['width']):
        raise ValueError('frame_integrity')
    faces = detect_person_faces(image, sample['person_boxes'], detector, sample['settings_json'], stop_event)
    with factory() as db:
        _bounded_transaction(db)
        # Consistent lock order with configure(): source first, then monitor/frame.
        source = db.query(Source).filter(Source.id == sample['source_id']).with_for_update().first()
        reference = _monitor_source(db, sample['monitor_id'], sample['owner_id'], sample['camera_id'], sample['run_id'], lock=True)
        row = db.query(FrameSample).filter(FrameSample.id == sample['id']).with_for_update().first()
        if row is None or row.status != 'processing' or row.lease != token:
            return 0
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError('worker_stopped')
        if not _owns_source(source, sample['owner_id'], sample['camera_id']) or not source.enabled or _settings(source) != sample['settings_json'] or not reference or row.expires_at <= time.time():
            row.status, row.completed_at, row.last_error = 'cancelled', time.time(), 'source_stopped_or_expired'
            row.lease, row.lease_until = None, 0
            db.commit()
            return 0
        row.status, row.completed_at, row.face_count = 'completed', time.time(), len(faces)
        row.lease, row.lease_until, row.last_error = None, 0, None
        source.last_at, source.last_count, source.last_error = sample['captured_at'], len(faces), None
        # Existing save_faces commits captures, delivery jobs and this sample atomically.
        records = save_faces(db, sample['owner_id'], sample['camera_id'], sample['camera_name'], sample['id'],
                             faces, sample['settings_json']['push_enabled'], sample['captured_at'])
        return len(records)


def fail_frame(factory, sample_id, token, error):
    with factory() as db:
        _bounded_transaction(db)
        sample = db.get(FrameSample, sample_id)
        if sample:
            source = db.query(Source).filter(Source.id == sample.source_id).with_for_update().first()
            changed = db.execute(update(FrameSample).where(FrameSample.id == sample_id,
                FrameSample.status == 'processing', FrameSample.lease == token).values(
                status='failed', completed_at=time.time(), last_error=error, lease=None, lease_until=0)).rowcount
            if not changed:
                return
            if source and source.enabled:
                source.last_error = '人脸采样失败，请检查原始帧、模型和存储'
            db.commit()


def release_frame(factory, sample_id, token):
    """A graceful CPU worker stop may resume the already captured frame after restart."""
    with factory() as db:
        _bounded_transaction(db)
        db.execute(update(FrameSample).where(FrameSample.id == sample_id, FrameSample.status == 'processing', FrameSample.lease == token).values(
            status='pending', lease=None, lease_until=0))
        db.commit()


def cleanup_frame(factory, sample_id):
    with factory() as db:
        _bounded_transaction(db)
        row = db.query(FrameSample).filter(FrameSample.id == sample_id).with_for_update().first()
        if row and row.status in TERMINAL:
            frame_path(sample_id).unlink(missing_ok=True)
            row.cache_released = True
            db.commit()


def reconcile_frames(factory, now=None):
    now = time.time() if now is None else now
    with factory() as db:
        _bounded_transaction(db, queue=True)
        expired = db.query(FrameSample).filter(FrameSample.status.in_(ACTIVE), FrameSample.expires_at <= now).with_for_update(skip_locked=True).limit(64).all()
        for row in expired:
            row.status, row.completed_at, row.last_error = 'cancelled', now, 'expired_frame'
            row.lease, row.lease_until = None, 0
        db.commit()
        finished = db.query(FrameSample).filter(FrameSample.status.in_(TERMINAL), FrameSample.cache_released.is_(False)).with_for_update(skip_locked=True).limit(64).all()
        for row in finished:
            frame_path(row.id).unlink(missing_ok=True)
            row.cache_released = True
        db.commit()
        # Keep metadata while a face capture refers to it; never delete business captures.
        db.query(FrameSample).filter(FrameSample.status.in_(TERMINAL), FrameSample.created_at < now - 3600,
            ~FrameSample.id.in_(db.query(Capture.frame_id))).delete(synchronize_session=False)
        db.commit()
        # Uncertain producer commits keep their PNG; clean only old, unreferenced cache files.
        directory = frame_path('0' * 64).parent
        for path in list(directory.glob('*.png'))[:256]:
            if path.is_symlink() or not re.fullmatch(r'[0-9a-f]{64}\.png', path.name):
                continue
            if path.stat().st_mtime < now - max(600, limits()['ttl'] + 120) and db.get(FrameSample, path.stem) is None:
                path.unlink(missing_ok=True)
        for path in list(directory.glob('*.tmp'))[:256]:
            if not path.is_symlink() and re.fullmatch(r'[0-9a-f]{64}\.[0-9a-f]{32}\.tmp', path.name) and path.stat().st_mtime < now - 600:
                path.unlink(missing_ok=True)
