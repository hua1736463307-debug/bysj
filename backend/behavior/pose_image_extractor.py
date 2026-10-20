"""Image pose feature extraction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .pose_feature_encoder import encode_image_pose


@lru_cache(maxsize=4)
def _get_pose_model(model_path: str):
    return YOLO(model_path)


class ImagePoseExtractor:
    def __init__(self, model_path: str = "yolov8n-pose.pt", conf: float = 0.30):
        self.model_path = model_path
        self.conf = float(conf)

    @property
    def model(self):
        return _get_pose_model(self.model_path)

    def _load_image(self, image: Any) -> Optional[np.ndarray]:
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            return img
        if isinstance(image, np.ndarray):
            return image
        return None

    def extract_keypoints(self, image: Any) -> Optional[np.ndarray]:
        img = self._load_image(image)
        if img is None:
            return None

        results = self.model(img, conf=self.conf, verbose=False)
        if not results:
            return None

        result = results[0]
        if getattr(result, "keypoints", None) is None:
            return None

        xy = result.keypoints.xy
        if xy is None or len(xy) == 0:
            return None

        # Select the person with the largest keypoint spread. This is more stable
        # than taking the first result when multiple people are present.
        candidates = xy.cpu().numpy()
        best = None
        best_area = -1.0

        for kp in candidates:
            if kp.shape[0] < 17:
                continue
            valid = np.isfinite(kp).all(axis=1) & (np.abs(kp).sum(axis=1) > 1e-6)
            if not valid.any():
                continue
            mins = kp[valid].min(axis=0)
            maxs = kp[valid].max(axis=0)
            area = float(np.prod(np.maximum(maxs - mins, 1.0)))
            if area > best_area:
                best_area = area
                best = kp

        return best

    def extract(self, image: Any) -> Optional[np.ndarray]:
        keypoints = self.extract_keypoints(image)
        if keypoints is None:
            return None
        return encode_image_pose(keypoints)
