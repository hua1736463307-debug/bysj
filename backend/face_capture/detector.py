import hashlib
import math
import os
from pathlib import Path
import threading

import cv2
import numpy as np

from .structure import has_face_structure
from . import pose

MODEL = Path(os.getenv('FACE_MODEL_PATH', str(Path(__file__).parent / 'assets' / 'face_detection_yunet_2023mar.onnx')))
_lock = threading.Lock()
_detector = None
_verifier = None
_digest = None


def _verify_face_crop(crop, expected_box, threshold):
    """Re-detect the same native pixels and match this face, not a neighbour.

Requiring a second prediction on the isolated face context reduces false heads
supported by the surrounding person ROI. It is not an identity/pose classifier.
The configured score threshold is unchanged and both predictions must pass it.
    """
    global _verifier
    height, width = crop.shape[:2]
    scale = min(1., 1280. / max(height, width))
    small = crop if scale == 1. else cv2.resize(crop, (max(1, round(width * scale)), max(1, round(height * scale))))
    sx, sy = width / small.shape[1], height / small.shape[0]
    with _lock:
        if _verifier is None:
            _verifier = cv2.FaceDetectorYN.create(str(MODEL), '', (320, 320), .8, .3, 5000,
                                                cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
        _verifier.setInputSize((small.shape[1], small.shape[0]))
        _verifier.setScoreThreshold(float(threshold))
        _, rows = _verifier.detect(small)
    ex1, ey1, ex2, ey2 = expected_box
    expected_area = (ex2 - ex1) * (ey2 - ey1)
    for row in ([] if rows is None else rows):
        if not has_face_structure(row) or not float(threshold) <= float(row[-1]) <= 1.:
            continue
        x, y, w, h = map(float, row[:4])
        x1, y1, x2, y2 = x * sx, y * sy, (x + w) * sx, (y + h) * sy
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            continue
        intersection = max(0., min(ex2, x2) - max(ex1, x1)) * max(0., min(ey2, y2) - max(ey1, y1))
        union = expected_area + (x2 - x1) * (y2 - y1) - intersection
        if union > 0 and intersection / union >= .4:
            return True
    return False


def model_status():
    view = pose.model_status()
    return {'available': MODEL.is_file() and MODEL.stat().st_size > 100000 and hasattr(cv2, 'FaceDetectorYN') and view['available'],
            'name': 'YuNet 2023mar', 'backend': 'OpenCV CPU', 'pose_filter': view}


def detect(image, threshold=.8, min_size=20):
    """Map resized model boxes back to the original, unannotated image."""
    global _detector, _digest
    if image is None or image.ndim != 3 or image.size == 0:
        raise ValueError('图片无效')
    if not model_status()['available']:
        raise RuntimeError('人脸模型不可用，请安装指定权重及兼容 OpenCV')
    height, width = image.shape[:2]
    scale = min(1., 1280. / max(height, width))
    small = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    sx, sy = width / small.shape[1], height / small.shape[0]
    with _lock:
        if _detector is None:
            _digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
            _detector = cv2.FaceDetectorYN.create(str(MODEL), '', (320, 320), .8, .3, 5000,
                                                cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
        _detector.setInputSize((small.shape[1], small.shape[0]))
        _detector.setScoreThreshold(float(threshold))
        _, faces = _detector.detect(small)
    result = []
    for face in ([] if faces is None else faces):
        if not has_face_structure(face):
            continue
        if not float(threshold) <= float(face[-1]) <= 1.:
            continue
        x, y, w, h = map(float, face[:4])
        face_left, face_top = math.floor(x * sx), math.floor(y * sy)
        face_right, face_bottom = math.ceil((x + w) * sx), math.ceil((y + h) * sy)
        if min(face_right - face_left, face_bottom - face_top) < min_size:
            continue
        # Do not publish a partial face crossing the original frame boundary.
        if face_left < 0 or face_top < 0 or face_right > width or face_bottom > height:
            continue
        # Preserve profile outline, forehead and chin, without requiring both eyes
        # or symmetric landmarks (which would exclude genuine profile faces).
        left = max(0, math.floor((x - .4 * w) * sx))
        right = min(width, math.ceil((x + 1.4 * w) * sx))
        top = max(0, math.floor((y - .35 * h) * sy))
        bottom = min(height, math.ceil((y + 1.2 * h) * sy))
        crop = image[top:bottom, left:right]
        expected_box = [face_left - left, face_top - top, face_right - left, face_bottom - top]
        if not _verify_face_crop(crop, expected_box, threshold):
            continue
        if not pose.is_face_view(crop):
            continue
        ok, jpeg = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError('人脸裁剪编码失败')
        result.append({'bbox': [left, top, right, bottom],
                       'detected_bbox': [face_left, face_top, face_right, face_bottom],
                       'confidence': float(face[-1]),
                       'jpeg': jpeg.tobytes(), 'model_sha256': _digest})
    return sorted(result, key=lambda item: tuple(item['bbox']))
