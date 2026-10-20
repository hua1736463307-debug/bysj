"""Owner-scoped raw camera previews, independent of alarm snapshot jobs."""
from __future__ import annotations

import base64
import math
import threading
import time
import uuid

import cv2

from utils.video_utils import SupervisedCapture

MAX_CAPTURES = 18
IDLE_SECONDS = 30


def _open_capture(_kind, url):
    try:
        capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
        ])
        if capture.isOpened():
            return capture
        capture.release()
    except Exception:
        return None
    return None


class LiveFramePool:
    def __init__(self, factory=None):
        self.factory = factory or self._capture
        self._lock = threading.RLock()
        self._items = {}
        self._sweeper = None
        self._closed = threading.Event()

    @staticmethod
    def _capture(key, url):
        capture = SupervisedCapture(
            'display-' + uuid.uuid4().hex, 'rtsp', url,
            _open_capture, lambda cap, _: cap.read()[1], lambda cap: cap.release(),
            reconnect_initial_seconds=.2, reconnect_max_seconds=3,
        )
        return capture.start()

    def _sweep(self):
        while not self._closed.wait(10):
            now = time.monotonic()
            with self._lock:
                stale = [key for key, item in self._items.items()
                         if now - item['seen_at'] > IDLE_SECONDS]
                captures = [self._items.pop(key)['capture'] for key in stale]
            for capture in captures:
                capture.stop(join_timeout=.1)

    def _borrow(self, owner_id, recorder_id, channel_number, url):
        if self._closed.is_set():
            raise RuntimeError('实时画面服务已停止')
        key = (str(owner_id), str(recorder_id), str(channel_number))
        with self._lock:
            if self._sweeper is None:
                self._sweeper = threading.Thread(target=self._sweep, name='display-capture-sweeper', daemon=True)
                self._sweeper.start()
            item = self._items.get(key)
            if item and not item['capture'].matches('rtsp', url):
                item['capture'].stop(join_timeout=.1)
                self._items.pop(key, None)
                item = None
            if item is None:
                if len(self._items) >= MAX_CAPTURES:
                    raise RuntimeError('实时画面连接已达上限')
                item = {'capture': self.factory(key, url), 'generation': uuid.uuid4().hex,
                        'seen_at': time.monotonic(), 'signature': None, 'signature_sequence': 0}
                self._items[key] = item
            item['seen_at'] = time.monotonic()
            capture, current_generation = item['capture'], item['generation']
        return capture, current_generation, item

    def native_after(self, owner_id, recorder_id, channel_number, url, not_before, timeout=8.):
        """Return one fresh native frame decoded AFTER this command was accepted.

Do not pass a resized JPEG/annotated preview or a pre-command cache to inference.
The display decoder is reused without changing automatic alarm monitoring.
        """
        if not math.isfinite(not_before):
            raise ValueError('invalid capture boundary')
        capture, generation, _ = self._borrow(owner_id, recorder_id, channel_number, url)
        deadline = time.monotonic() + min(8., max(0., timeout))
        after = 0
        while time.monotonic() < deadline:
            image, sequence, health = capture.wait_for_latest(after, min(.1, max(0., deadline - time.monotonic())))
            health = health or {}
            after = max(after, sequence)
            stamp, age = health.get('last_frame_at'), health.get('frame_age_ms')
            if (image is not None and health.get('status') == 'live' and stamp is not None and
                    math.isfinite(stamp) and stamp >= not_before and age is not None and 0 <= age <= 500):
                return image.copy(), {'capture_timestamp': stamp, 'frame_sequence': sequence, 'run_id': generation}
        raise RuntimeError('未取得本次点击后的新画面，请确认摄像头在线后重试')

    def frame(self, owner_id, recorder_id, channel_number, url, generation=None, sequence=0,
              probe_only=False):
        capture, current_generation, item = self._borrow(owner_id, recorder_id, channel_number, url)
        after = max(0, int(sequence)) if generation == current_generation else 0
        image, current_sequence, health = capture.wait_for_latest(after, 0)
        status = (health or {}).get('status', 'connecting')
        fresh = ((health or {}).get('frame_age_ms') is not None
                 and health['frame_age_ms'] <= 2000 and status == 'live')
        if not fresh:
            return {'status': status, 'run_id': current_generation, 'frame_sequence': current_sequence,
                    'available': False, 'unchanged': True}
        if image is not None and item['signature_sequence'] != current_sequence:
            small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 8),
                               interpolation=cv2.INTER_AREA)
            bits = (small[:, 1:] > small[:, :-1]).ravel()
            signature = 0
            for bit in bits:
                signature = (signature << 1) | int(bit)
            item['signature'] = {'fingerprint': f'{signature:016x}',
                                 'brightness': round(float(small.mean()), 1),
                                 'contrast': round(float(small.std()), 1)}
            item['signature_sequence'] = current_sequence
        details = {'status': status, 'run_id': current_generation,
                   'frame_sequence': current_sequence, 'available': True,
                   **(item['signature'] or {})}
        if probe_only or image is None:
            return {**details, 'unchanged': True}
        if current_sequence <= after:
            return {**details, 'unchanged': True}
        height, width = image.shape[:2]
        if width > 960:
            image = cv2.resize(image, (960, round(height * 960 / width)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if not ok:
            return {**details, 'status': 'error', 'unchanged': True}
        return {**details, 'annotated_image': base64.b64encode(encoded).decode('ascii'),
                'capture_timestamp': health.get('last_frame_at')}

    def close(self):
        self._closed.set()
        with self._lock:
            captures = [item['capture'] for item in self._items.values()]
            self._items.clear()
        for capture in captures:
            capture.stop(join_timeout=.1)


live_frame_pool = LiveFramePool()
