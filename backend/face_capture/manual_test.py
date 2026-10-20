"""Bounded private test previews. No alarm/capture/delivery records are created."""
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import math
import threading
import time

import cv2
import numpy as np
from fastapi import HTTPException

from .camera import resolve_camera
from .detector import detect, model_status
from .models import Source
from monitoring_live_frames import live_frame_pool

TTL_SECONDS = 600
MAX_RESULTS = 128
MAX_IMAGES = 16
OWNER_IMAGES = 6
MAX_BYTES = 128 * 1024 * 1024
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 32_000_000


class TestResults:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.items = OrderedDict()

    def _prune(self):
        now = self.clock()
        for key in list(self.items):
            if now - self.items[key]['created'] >= TTL_SECONDS:
                del self.items[key]

    def find(self, owner, request_id, camera):
        with self.lock:
            self._prune()
            item = self.items.get((str(owner), str(request_id)))
            if item is None:
                return None
            if item['camera'] != camera:
                raise HTTPException(409, '本次测试编号已用于另一摄像头，请重新点击')
            return deepcopy(item['result'])

    def add(self, owner, request_id, camera, result, png):
        with self.lock:
            self._prune()
            if len(self.items) >= MAX_RESULTS:
                raise HTTPException(429, '临时测试次数已达缓存上限，请稍后再试')
            key = (str(owner), str(request_id))
            self.items[key] = {'camera': camera, 'result': deepcopy(result), 'png': png, 'created': self.clock()}
            own = [item for k, item in self.items.items() if k[0] == str(owner) and item['png'] is not None]
            for item in own[:-OWNER_IMAGES]:
                item['png'] = None
            images = [item for item in self.items.values() if item['png'] is not None]
            total = sum(len(item['png']) for item in images)
            while len(images) > MAX_IMAGES or total > MAX_BYTES:
                item = images.pop(0)
                total -= len(item['png'])
                item['png'] = None

    def image(self, owner, request_id):
        with self.lock:
            self._prune()
            item = self.items.get((str(owner), str(request_id)))
            if item is None:
                raise HTTPException(404, '临时测试不存在或已过期')
            if item['png'] is None:
                raise HTTPException(410, '临时测试原图已释放，请重新点击测试')
            return item['png']

    def clear(self):
        with self.lock:
            self.items.clear()


results = TestResults()


def snapshot_test(db, owner, recorder_id, channel_number, request_id):
    requested_at = time.time()
    camera = (str(recorder_id), str(channel_number))
    try:
        url, name = resolve_camera(db, owner, *camera)
    except ValueError:
        raise HTTPException(404, '摄像头不存在、未启用或无权访问') from None
    cached = results.find(owner, request_id, camera)
    if cached is not None:
        return cached
    if not model_status()['available']:
        raise HTTPException(503, '人脸模型不可用')
    source = db.query(Source).filter(Source.owner_id == str(owner), Source.recorder_id == camera[0], Source.channel_number == camera[1]).first()
    threshold = float(source.threshold) if source else .6
    min_size = int(source.min_size) if source else 64
    db.rollback()
    if not math.isfinite(threshold) or not .1 <= threshold <= .99 or not 4 <= min_size <= 1000:
        raise HTTPException(422, '摄像头人脸参数无效，请先保存有效设置')
    try:
        image, frame = live_frame_pool.native_after(str(owner), *camera, url, requested_at)
    except (ValueError, RuntimeError):
        raise HTTPException(503, '未取得本次点击后的新原帧，请确认摄像头在线后重试') from None
    stamp = frame.get('capture_timestamp')
    if stamp is None or not math.isfinite(stamp) or stamp < requested_at:
        raise HTTPException(503, '取得的画面早于本次请求，请重新测试')
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or
            not 0 < image.shape[0] * image.shape[1] <= MAX_PIXELS):
        raise HTTPException(413, '测试原图尺寸超过限制或格式无效')
    # Freeze the native source before inference. Never use a preview JPEG.
    ok, encoded = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise HTTPException(422, '测试原图编码失败')
    png = encoded.tobytes()
    if len(png) > MAX_IMAGE_BYTES:
        raise HTTPException(413, '测试原图超过16 MiB限制')
    try:
        faces = detect(image, threshold, min_size)
    except (ValueError, RuntimeError, cv2.error):
        raise HTTPException(503, '本帧人脸判断失败，请重试') from None
    if len(faces) > 64:
        raise HTTPException(422, '单帧检测结果过多，请缩小测试范围')
    rows = [{key: face[key] for key in ('bbox', 'detected_bbox', 'confidence')} for face in faces]
    result = {'test_id': str(request_id), 'recorder_id': camera[0], 'channel_number': camera[1], 'camera_name': name,
              'requested_at': datetime.fromtimestamp(requested_at, timezone.utc).isoformat(),
              'captured_at': datetime.fromtimestamp(frame['capture_timestamp'], timezone.utc).isoformat(),
              'frame_sequence': frame['frame_sequence'], 'frame_run_id': frame['run_id'],
              'width': image.shape[1], 'height': image.shape[0], 'image_sha256': hashlib.sha256(png).hexdigest(),
              'threshold': threshold, 'min_size': min_size, 'count': len(rows), 'faces': rows,
              'image_url': '/monitoring/faces/manual-tests/' + str(request_id) + '/image'}
    results.add(owner, request_id, camera, result, png)
    return result
