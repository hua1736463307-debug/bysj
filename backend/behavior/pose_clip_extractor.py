"""Video clip pose feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .pose_feature_encoder import aggregate_pose_features, encode_single_pose
from .pose_image_extractor import ImagePoseExtractor


class ClipPoseExtractor:
    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        conf: float = 0.30,
        sample_stride: int = 5,
    ):
        self.extractor = ImagePoseExtractor(model_path=model_path, conf=conf)
        self.sample_stride = max(1, int(sample_stride))

    def extract(self, video_path: str | Path, max_frames: int = 96) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        features = []
        frame_id = 0
        sampled = 0

        try:
            while sampled < int(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % self.sample_stride != 0:
                    frame_id += 1
                    continue

                keypoints = self.extractor.extract_keypoints(frame)
                if keypoints is not None:
                    features.append(encode_single_pose(keypoints))

                sampled += 1
                frame_id += 1
        finally:
            cap.release()

        if not features:
            return None

        return aggregate_pose_features(features)
