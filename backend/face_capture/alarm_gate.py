"""Same-frame person face presence, before object/violation alarm decisions."""
import math
import time

import numpy as np

from .sampling import detect_person_faces, person_boxes, _iou

POLICY = 'same_frame_person_face_v1'
REQUIRED_CATEGORIES = {'object', 'violation'}
LEGACY_CATEGORIES = {'phone_use': 'object', 'missing_badge': 'violation', 'fall_down': 'behavior'}
THRESHOLD = .6
MIN_SIZE = 20


def requires_face(item):
    return (item.get('category') or LEGACY_CATEGORIES.get(item.get('code'))) in REQUIRED_CATEGORIES


class FacePresenceGate:
    """Instance/cache lives for one original frame only; never reads a camera."""
    def __init__(self, image_rgb, frame_id, detector=None, clock=time.monotonic, budget_seconds=2., max_regions=32, threshold=THRESHOLD, min_size=MIN_SIZE):
        self.image = image_rgb
        self.frame_id = str(frame_id)
        self.detector = detector
        self.clock = clock
        self.budget_seconds = budget_seconds
        self.spent_seconds = 0.
        self.max_regions = max_regions
        self.people = []
        self.cache = {}
        self.calls = 0
        self.blocked = 0
        self.threshold = threshold
        self.min_size = min_size
        self.valid = isinstance(image_rgb, np.ndarray) and image_rgb.dtype == np.uint8 and image_rgb.ndim == 3 and image_rgb.shape[2] == 3 and 0 < image_rgb.shape[0] * image_rgb.shape[1] <= 32000000

    def _box(self, value):
        if not self.valid:
            return None
        boxes = person_boxes([{'class': 'person', 'bbox': value}], self.image.shape[1], self.image.shape[0])
        return boxes[0] if boxes else None

    def observe_people(self, values):
        for value in values:
            box = self._box(value)
            if box and not any(_iou(box, other) > .85 for other in self.people):
                self.people.append(box)

    def check(self, person_bbox):
        box = self._box(person_bbox)
        result = {'policy': POLICY, 'passed': False, 'frame_id': self.frame_id,
                  'threshold': self.threshold, 'min_size': self.min_size, 'person_bbox': box}
        if box is None:
            return self._blocked(result, 'no_person_lineage')
        if len(self.people) > self.max_regions:
            return self._blocked(result, 'too_many_person_regions')
        key = tuple(box)
        if key not in self.cache:
            if self.calls >= self.max_regions or self.spent_seconds >= self.budget_seconds:
                return self._blocked(result, 'face_budget_exhausted')
            self.calls += 1
            started = self.clock()
            try:
                if self.detector is None:
                    from .detector import detect
                    detector = detect
                else:
                    detector = self.detector
                left, top, right, bottom = box
                rgb_roi = self.image[top:bottom, left:right]
                # Native ROI, RGB->BGR; never scale or decode another frame.
                faces = detect_person_faces(rgb_roi[:, :, ::-1].copy(), [[0, 0, right-left, bottom-top]],
                                            detector, {'threshold': self.threshold, 'min_size': self.min_size})
                mapped = []
                for face in faces:
                    item = dict(face)
                    for name in ('bbox', 'detected_bbox'):
                        if name in item:
                            x1,y1,x2,y2 = item[name]
                            item[name] = [x1+left,y1+top,x2+left,y2+top]
                    mapped.append(item)
                self.spent_seconds += max(0.,self.clock()-started)
                if self.spent_seconds > self.budget_seconds:
                    self.cache[key] = ('face_budget_exhausted', [])
                else:
                    self.cache[key] = (None, mapped)
            except Exception:
                self.spent_seconds += max(0.,self.clock()-started)
                self.cache[key] = ('face_detection_failed', [])
        error, faces = self.cache[key]
        if error:
            return self._blocked(result, error)
        qualified = []
        for face in faces:
            try:
                confidence = float(face['confidence'])
                detected = face.get('detected_bbox') or face['bbox']
                if not math.isfinite(confidence) or not self.threshold <= confidence <= 1:
                    continue
                if min(detected[2]-detected[0], detected[3]-detected[1]) < self.min_size:
                    continue
                cx,cy = (detected[0]+detected[2])/2,(detected[1]+detected[3])/2
                # Overlapping people cannot borrow one another's face.
                owners = [p for p in self.people if p[0]<=cx<=p[2] and p[1]<=cy<=p[3]]
                if not owners or any(_iou(box,p) <= .85 for p in owners):
                    continue
                qualified.append(face)
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        if not qualified:
            return self._blocked(result, 'no_unambiguous_face')
        face = max(qualified, key=lambda f: float(f['confidence']))
        return {**result, 'passed': True, 'reason': 'face_detected',
                'face_bbox': list(face.get('detected_bbox') or face['bbox']),
                'face_crop_bbox': list(face['bbox']), 'face_confidence': float(face['confidence']),
                'face_model_sha256': face.get('model_sha256')}

    def _blocked(self, result, reason):
        self.blocked += 1
        return {**result, 'reason': reason}

    def stats(self):
        return {'policy': POLICY, 'calls': self.calls, 'blocked': self.blocked,
                'threshold': self.threshold, 'min_size': self.min_size}
