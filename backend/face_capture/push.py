import base64
import hashlib
import json
import os
import time
import urllib.request
from sqlalchemy import update

from .models import Capture
from .service import claim_delivery, image_path, utcnow


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def deliver_once(session_factory):
    with session_factory() as db:
        claimed = claim_delivery(db)
        if not claimed:
            return False
        row, token = claimed
        capture_id, attempts = row.id, row.attempts
        destination = row.destination
        payload = {'type': 'face_capture', 'capture_id': row.id, 'captured_at': row.captured_at,
                   'camera_id': row.camera_id, 'camera_name': row.camera_name,
                   'face_bbox': row.bbox, 'confidence': row.confidence, 'model_sha256': row.model_sha256,
                   'image_sha256': row.image_sha256, 'image_mime': 'image/jpeg'}
        expected_digest = row.image_sha256
    try:
        image = image_path(capture_id).read_bytes()
        if hashlib.sha256(image).hexdigest() != expected_digest:
            raise ValueError('media_integrity')
        payload['image_base64'] = base64.b64encode(image).decode('ascii')
        headers = {'Content-Type': 'application/json', 'Idempotency-Key': capture_id}
        body = json.dumps(payload).encode()
        if destination == 'test':
            from .receiver import signature, RECEIVER_PATH
            timestamp = str(int(time.time()))
            headers.update({'X-Face-Timestamp': timestamp, 'X-Face-Signature': signature(body, timestamp)})
            url = 'http://127.0.0.1:8000' + RECEIVER_PATH
        else:
            url = os.environ['FACE_PUSH_URL']
        if destination != 'test' and os.getenv('FACE_PUSH_TOKEN'):
            headers['Authorization'] = 'Bearer ' + os.environ['FACE_PUSH_TOKEN']
        request = urllib.request.Request(url, body, headers, method='POST')
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=10) as response:
            body = response.read(65537)
            if len(body) > 65536:
                raise ValueError('response_too_large')
            answer = json.loads(body)
            if answer.get('accepted') is not True or answer.get('capture_id') != capture_id:
                raise ValueError('business_ack_missing')
        values = dict(status='succeeded', delivered_at=utcnow(), last_error=None, lease=None)
    except Exception as exc:
        # Never expose credentials or receiver response contents in UI/logs.
        values = dict(status='failed' if attempts >= 5 else 'pending', lease=None,
                      next_at=time.time() + min(300, 2 ** attempts), last_error=type(exc).__name__)
    with session_factory() as db:
        db.execute(update(Capture).where(Capture.id == capture_id, Capture.lease == token).values(**values))
        db.commit()
    return True
