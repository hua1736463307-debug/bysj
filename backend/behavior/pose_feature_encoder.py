"""Pose feature encoder for action sample classification.

The encoder converts COCO-17 keypoints into a fixed-length feature vector.
For compatibility between image samples and video clip samples, we use the same
temporal aggregate layout for both:
- mean pose features
- std pose features
- delta pose features

For a single image, std and delta are zero vectors.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

COCO_KEYPOINT_COUNT = 17

COCO_PAIRS = [
    (5, 6),    # shoulders
    (5, 11),   # left shoulder-hip
    (6, 12),   # right shoulder-hip
    (11, 12),  # hips
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

BASE_FEATURE_NAMES: List[str] = (
    [f"kp_{i}_x" for i in range(COCO_KEYPOINT_COUNT)]
    + [f"kp_{i}_y" for i in range(COCO_KEYPOINT_COUNT)]
    + [f"dist_{i}_{j}" for i, j in COCO_PAIRS]
)

POSE_FEATURE_NAMES: List[str] = (
    [f"mean_{name}" for name in BASE_FEATURE_NAMES]
    + [f"std_{name}" for name in BASE_FEATURE_NAMES]
    + [f"delta_{name}" for name in BASE_FEATURE_NAMES]
)


def _as_keypoint_array(keypoints: Iterable) -> np.ndarray:
    arr = np.asarray(keypoints, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)

    if arr.shape[0] < COCO_KEYPOINT_COUNT:
        padded = np.zeros((COCO_KEYPOINT_COUNT, max(2, arr.shape[1] if arr.ndim == 2 else 2)), dtype=np.float32)
        padded[: arr.shape[0], : arr.shape[1]] = arr
        arr = padded

    if arr.shape[1] < 2:
        padded = np.zeros((COCO_KEYPOINT_COUNT, 2), dtype=np.float32)
        padded[:, : arr.shape[1]] = arr[:COCO_KEYPOINT_COUNT]
        arr = padded

    return arr[:COCO_KEYPOINT_COUNT]


def normalize_pose(keypoints: Iterable) -> np.ndarray:
    arr = _as_keypoint_array(keypoints)
    xy = arr[:, :2].astype(np.float32)

    valid = np.isfinite(xy).all(axis=1) & (np.abs(xy).sum(axis=1) > 1e-6)
    if valid.any():
        center = xy[valid].mean(axis=0)
    else:
        center = xy.mean(axis=0)

    xy = xy - center

    # Use torso/box scale first, then fallback to norm.
    if valid.any():
        mins = xy[valid].min(axis=0)
        maxs = xy[valid].max(axis=0)
    else:
        mins = xy.min(axis=0)
        maxs = xy.max(axis=0)

    scale = float(np.linalg.norm(maxs - mins))
    if scale <= 1e-6:
        scale = float(np.linalg.norm(xy))

    if scale > 1e-6:
        xy = xy / scale

    xy[~np.isfinite(xy)] = 0.0
    return xy.astype(np.float32)


def encode_single_pose(keypoints: Iterable) -> np.ndarray:
    kp = normalize_pose(keypoints)

    features: List[float] = []
    features.extend(kp[:, 0].tolist())
    features.extend(kp[:, 1].tolist())

    for i, j in COCO_PAIRS:
        features.append(float(np.linalg.norm(kp[i] - kp[j])))

    return np.asarray(features, dtype=np.float32)


def aggregate_pose_features(pose_features: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(list(pose_features), dtype=np.float32)

    if arr.size == 0:
        raise ValueError("没有可聚合的 pose 特征")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    if arr.shape[0] >= 2:
        delta = arr[-1] - arr[0]
    else:
        delta = np.zeros_like(mean)

    output = np.concatenate([mean, std, delta]).astype(np.float32)
    output[~np.isfinite(output)] = 0.0
    return output


def encode_image_pose(keypoints: Iterable) -> np.ndarray:
    return aggregate_pose_features([encode_single_pose(keypoints)])
