"""CPU-only, fixed-weight head view gate; never an identity/confidence score.

Use the predicted forward normal, not folded Euler yaw (which aliases a rear
view into [-90, 90]). A true profile may have only one visible eye. This gate
therefore imposes no landmark symmetry or skin-colour rule.
"""
import hashlib
import math
from pathlib import Path
import threading

import cv2
import numpy as np
from PIL import Image

MODEL = Path(__file__).parent / 'assets' / 'head_pose_6drepnet360.onnx'
MODEL_SHA256 = '68b8ba2adef2ca08ed4eeb41067120f457660b84b3f33ed1ca55b88e5257047d'
POLICY = 'head_forward_normal_v1'
MAX_TURN = 95.  # five-degree model tolerance around a genuine 90-degree profile
MAX_TILT = 35.
_lock = threading.Lock()
_net = None


def model_status():
    return {'available': MODEL.is_file() and MODEL.stat().st_size > 1000000,
            'name': '6DRepNet360 Full Rotation', 'backend': 'OpenCV CPU',
            'policy': POLICY, 'max_turn_degrees': MAX_TURN,
            'max_tilt_degrees': MAX_TILT, 'model_sha256': MODEL_SHA256}


def forward_angles(output):
    """Recover the full-range head normal from the two 6D basis vectors."""
    value = np.asarray(output, dtype=np.float64).reshape(-1)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError('Invalid head pose output')
    x = value[:3]
    length = float(np.linalg.norm(x))
    if length < 1e-8:
        raise ValueError('Degenerate head pose basis')
    x = x / length
    normal = np.cross(x, value[3:])
    length = float(np.linalg.norm(normal))
    if length < 1e-8:
        raise ValueError('Degenerate head pose normal')
    normal /= length
    turn = math.degrees(math.atan2(float(normal[0]), float(normal[2])))
    tilt = math.degrees(math.atan2(float(-normal[1]), math.hypot(float(normal[0]), float(normal[2]))))
    return turn, tilt


def within_view(turn, tilt):
    return math.isfinite(turn) and math.isfinite(tilt) and abs(turn) <= MAX_TURN and abs(tilt) <= MAX_TILT


def preprocess(crop):
    """Match the author's PIL Resize(256), CenterCrop(224), RGB normalization."""
    if crop is None or crop.ndim != 3 or crop.shape[2] != 3 or crop.dtype != np.uint8 or min(crop.shape[:2]) < 1:
        raise ValueError('Invalid head pose crop')
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    width, height = image.size
    if width <= height:
        resized = (256, int(256 * height / width))
    else:
        resized = (int(256 * width / height), 256)
    image = image.resize(resized, Image.Resampling.BILINEAR)
    left, top = round((resized[0] - 224) / 2), round((resized[1] - 224) / 2)
    pixels = np.asarray(image.crop((left, top, left + 224, top + 224)), dtype=np.float32) / np.float32(255.)
    pixels = (pixels - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


def is_face_view(crop):
    """Reject unsuitable views; missing/corrupt models never disable the gate."""
    global _net
    blob = preprocess(crop)
    with _lock:
        if _net is None:
            if not model_status()['available'] or hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
                raise RuntimeError('人脸姿态模型不可用或校验失败')
            net = cv2.dnn.readNetFromONNX(str(MODEL))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            _net = net
        # Mirror augmentation protects genuine left/right profiles against the
        # model's directional bias. Both views must be outside the permitted
        # range to reject; this is not a vote on identity or image sharpness.
        for pixels in (blob, np.ascontiguousarray(blob[:, :, :, ::-1])):
            _net.setInput(pixels)
            output = _net.forward().copy()
            try:
                if within_view(*forward_angles(output)):
                    return True
            except ValueError:
                return False
    return False
