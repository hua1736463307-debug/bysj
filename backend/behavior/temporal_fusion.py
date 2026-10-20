"""Video temporal-fusion behavior model and bounded training utilities.

The model deliberately sits above the existing person/Pose/object detectors.
It consumes short person-track clips plus normalized detector evidence, so the
rule engine remains available as an operational fallback while this model is
being evaluated in shadow mode.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import uuid
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional in lightweight API/test images
    torch = None
    Tensor = Any  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore

from .feedback_mapping import SUPPORTED_BEHAVIOR_EVENTS

logger = logging.getLogger(__name__)

MODEL_FAMILY = "temporal_fusion"
MODEL_TYPE = "behavior_temporal_fusion_v1"
TEMPORAL_EVENT_TYPES = tuple(SUPPORTED_BEHAVIOR_EVENTS)
FEATURE_SIZE = 96
DEFAULT_CLIP_LENGTH = 12
DEFAULT_IMAGE_SIZE = 64
DEFAULT_SAMPLE_FPS = 6.0
DEFAULT_THRESHOLDS = {event_type: 0.35 for event_type in TEMPORAL_EVENT_TYPES}
LABELS = (*TEMPORAL_EVENT_TYPES, "normal", "uncertain")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _clip_box(box: Iterable[Any], width: int, height: int) -> Tuple[int, int, int, int]:
    values = list(box or [])
    if len(values) < 4:
        return 0, 0, max(1, width), max(1, height)
    x1, y1, x2, y2 = [_safe_float(item) for item in values[:4]]
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(x1 + 1, min(width, round(x2))))
    y2 = int(max(y1 + 1, min(height, round(y2))))
    return x1, y1, x2, y2


def crop_track_frame(
    frame: np.ndarray,
    bbox: Iterable[Any],
    image_size: int = DEFAULT_IMAGE_SIZE,
    *,
    input_is_rgb: bool = False,
) -> np.ndarray:
    """Return a fixed RGB person crop with no unbounded image allocation."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 2:
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(bbox, width, height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame
    crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    return crop if input_is_rgb else cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _nearest_phone(person: Dict[str, Any], objects: Dict[Any, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    track_id = person.get("track_id")
    owned = [item for item in (objects or {}).values() if item.get("owner_id") == track_id]
    if owned:
        return max(owned, key=lambda item: _safe_float(item.get("confidence")))
    return None


def build_temporal_feature_vector(
    person: Dict[str, Any],
    objects: Optional[Dict[Any, Dict[str, Any]]] = None,
    frame_shape: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Encode detector evidence into a stable, bounded feature vector."""
    height = max(1.0, _safe_float((frame_shape or [1, 1])[:1][0], 1.0))
    width = max(1.0, _safe_float((frame_shape or [1, 1])[:2][-1], 1.0))
    bbox = list(person.get("bbox") or [0, 0, width, height])
    x1, y1, x2, y2 = [_safe_float(item) for item in (bbox + [0, 0, 0, 0])[:4]]
    person_width = max(1.0, x2 - x1)
    person_height = max(1.0, y2 - y1)
    vector: List[float] = [
        x1 / width, y1 / height, x2 / width, y2 / height,
        person_width / width, person_height / height,
        _safe_float(person.get("center", [0, 0])[0] if person.get("center") else (x1 + x2) / 2) / width,
        _safe_float(person.get("center", [0, 0])[1] if person.get("center") else (y1 + y2) / 2) / height,
    ]

    keypoints = list(person.get("keypoints") or [])[:17]
    for index in range(17):
        point = keypoints[index] if index < len(keypoints) else []
        px = _safe_float(point[0]) if len(point) >= 1 else 0.0
        py = _safe_float(point[1]) if len(point) >= 2 else 0.0
        confidence = _safe_float(point[2], 0.0) if len(point) >= 3 else 0.0
        vector.extend(((px - x1) / person_width, (py - y1) / person_height, confidence))

    phone = _nearest_phone(person, objects or {})
    vector.extend([
        1.0 if phone else 0.0,
        _safe_float(phone.get("confidence") if phone else 0.0),
        _safe_float(phone.get("owner_distance_ratio") if phone else 1.0, 1.0),
        1.0 if phone and phone.get("owner_id") is not None else 0.0,
        1.0 if phone and phone.get("owner_history") else 0.0,
        _safe_float(phone.get("bbox", [0, 0, 0, 0])[0] if phone else 0.0) / width,
        _safe_float(phone.get("bbox", [0, 0, 0, 0])[1] if phone else 0.0) / height,
    ])
    vector.extend([
        1.0 if person.get("badge_evaluable") else 0.0,
        1.0 if person.get("badge_present") else 0.0,
        _safe_float(person.get("badge_visibility_score"), 0.0),
        1.0 if person.get("chest_roi") else 0.0,
        _safe_float(person.get("badge_confidence"), 0.0),
    ])
    vector.extend([
        _safe_float(person.get("pose_confidence"), 0.0),
        _safe_float(person.get("keypoint_confidence"), 0.0),
        _safe_float(person.get("track_age"), 0.0) / 120.0,
    ])
    array = np.asarray(vector[:FEATURE_SIZE], dtype=np.float32)
    if array.size < FEATURE_SIZE:
        array = np.pad(array, (0, FEATURE_SIZE - array.size))
    return np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("端到端行为模型需要 PyTorch 运行时")


if nn is not None:

    class TemporalFusionNet(nn.Module):
        """Small video encoder with a detector-evidence fusion head."""

        def __init__(self, feature_size: int = FEATURE_SIZE, event_count: int = len(TEMPORAL_EVENT_TYPES)):
            super().__init__()
            self.video_encoder = nn.Sequential(
                nn.Conv3d(3, 16, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2)),
                nn.Conv3d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(2, 2, 2)),
                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d(1),
            )
            self.feature_encoder = nn.Sequential(
                nn.Linear(feature_size, 64),
                nn.LayerNorm(64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Sequential(
                nn.Linear(64 + 32, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(64, event_count),
            )

        def forward(self, video: Tensor, features: Tensor) -> Tensor:
            if video.ndim != 5:
                raise ValueError("video tensor 必须为 [B,T,C,H,W] 或 [B,C,T,H,W]")
            if video.shape[1] == 3:
                channels_first = video
            else:
                channels_first = video.permute(0, 2, 1, 3, 4)
            encoded_video = self.video_encoder(channels_first).flatten(1)
            encoded_features = self.feature_encoder(features)
            return self.head(torch.cat((encoded_video, encoded_features), dim=1))

else:

    class TemporalFusionNet:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            _require_torch()


def _frame_tensor(frame: np.ndarray) -> np.ndarray:
    return np.asarray(frame, dtype=np.float32).transpose(2, 0, 1) / 255.0


def _read_clip(
    path: str,
    clip_length: int,
    image_size: int,
    bbox: Optional[Iterable[Any]] = None,
) -> Optional[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    frames: List[np.ndarray] = []
    bbox_values = list(bbox or [])
    crop_box = bbox_values if len(bbox_values) >= 4 else None
    try:
        max_read = max(clip_length * 8, 96)
        while len(frames) < max_read:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            crop_box = crop_box or [0, 0, frame.shape[1], frame.shape[0]]
            frames.append(crop_track_frame(frame, crop_box, image_size))
    finally:
        capture.release()
    if not frames:
        return None
    indices = np.linspace(0, len(frames) - 1, num=clip_length).round().astype(int)
    return np.stack([_frame_tensor(frames[index]) for index in indices], axis=0).astype(np.float32)


def _sample_feature_vector(sample: Dict[str, Any]) -> np.ndarray:
    evidence = sample.get("evidence") if isinstance(sample.get("evidence"), dict) else {}
    decision = evidence.get("decision") if isinstance(evidence.get("decision"), dict) else {}
    raw = decision.get("training_feature_vector")
    if isinstance(raw, list) and raw:
        values = np.asarray(raw, dtype=np.float32).reshape(-1)
        values = values[:FEATURE_SIZE]
        if values.size < FEATURE_SIZE:
            values = np.pad(values, (0, FEATURE_SIZE - values.size))
        return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.zeros(FEATURE_SIZE, dtype=np.float32)


def _sample_person_bbox(sample: Dict[str, Any]) -> Optional[List[float]]:
    evidence = sample.get("evidence") if isinstance(sample.get("evidence"), dict) else {}
    reason = evidence.get("reason") if isinstance(evidence.get("reason"), dict) else {}
    candidate = reason.get("person_bbox") or evidence.get("person_bbox")
    if isinstance(candidate, (list, tuple)) and len(candidate) >= 4:
        try:
            return [float(value) for value in candidate[:4]]
        except (TypeError, ValueError):
            return None
    persons = evidence.get("persons")
    if isinstance(persons, list):
        for person in persons:
            if isinstance(person, dict) and isinstance(person.get("bbox"), (list, tuple)):
                try:
                    return [float(value) for value in person["bbox"][:4]]
                except (TypeError, ValueError):
                    continue
    return None


def _group_for_sample(sample: Dict[str, Any], index: int) -> str:
    return str(
        sample.get("source_task_id")
        or sample.get("source_id")
        or sample.get("source_video")
        or sample.get("id")
        or f"sample-{index}"
    )


def _sample_label(sample: Dict[str, Any]) -> str:
    label = str(sample.get("label") or "").strip()
    if label in TEMPORAL_EVENT_TYPES or label == "normal":
        return label
    return "uncertain"


def _sample_is_evaluable(sample: Dict[str, Any]) -> bool:
    evidence = sample.get("evidence") if isinstance(sample.get("evidence"), dict) else {}
    return not (
        sample.get("evaluable") is False
        or evidence.get("evaluable") is False
        or str(sample.get("status") or evidence.get("review_status") or "").strip().lower()
        in {"not_evaluable", "not-evaluable"}
    )


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, thresholds: Dict[str, float]) -> Dict[str, Any]:
    per_event: Dict[str, Dict[str, Any]] = {}
    for index, event_type in enumerate(TEMPORAL_EVENT_TYPES):
        expected = y_true[:, index] > 0.5
        predicted = probabilities[:, index] >= float(thresholds.get(event_type, 0.35))
        tp = int(np.sum(expected & predicted))
        fp = int(np.sum(~expected & predicted))
        fn = int(np.sum(expected & ~predicted))
        tn = int(np.sum(~expected & ~predicted))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        order = np.argsort(-probabilities[:, index])
        ranked_truth = expected[order].astype(np.float32)
        cumulative_true = np.cumsum(ranked_truth)
        precision_at_k = cumulative_true / np.arange(1, len(ranked_truth) + 1)
        pr_auc = float(np.sum(precision_at_k * ranked_truth) / max(1.0, float(np.sum(ranked_truth))))
        per_event[event_type] = {
            "support": int(np.sum(expected)),
            "predicted_positive": int(np.sum(predicted)),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "pr_auc": round(pr_auc, 4),
            "false_positive_rate": round(float(fp / max(1, fp + tn)), 4),
            "miss_rate": round(float(fn / max(1, fn + tp)), 4),
            "threshold": round(float(thresholds.get(event_type, 0.35)), 4),
            "confusion_matrix": [[tn, fp], [fn, tp]],
        }
    return {"sample_count": int(len(y_true)), "per_event": per_event}


def _build_training_arrays(
    samples: Sequence[Dict[str, Any]],
    *,
    clip_length: int,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
    videos: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    features: List[np.ndarray] = []
    groups: List[str] = []
    failures: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        label = _sample_label(sample)
        if label == "uncertain":
            continue
        clip_path = str(sample.get("clip_path") or sample.get("video_path") or "").strip()
        if not clip_path or not Path(clip_path).is_file():
            failures.append({"id": sample.get("id"), "label": label, "path": clip_path, "error": "clip 不存在"})
            continue
        video = _read_clip(clip_path, clip_length, image_size, _sample_person_bbox(sample))
        if video is None:
            failures.append({"id": sample.get("id"), "label": label, "path": clip_path, "error": "无法解码 clip"})
            continue
        target = np.zeros(len(TEMPORAL_EVENT_TYPES), dtype=np.float32)
        if label in TEMPORAL_EVENT_TYPES:
            target[TEMPORAL_EVENT_TYPES.index(label)] = 1.0
        videos.append(video)
        features.append(_sample_feature_vector(sample))
        labels.append(target)
        groups.append(_group_for_sample(sample, index))
    if not videos:
        raise ValueError("没有可解码的行为视频样本")
    return np.stack(videos), np.stack(features), np.stack(labels), groups, failures


def train_temporal_fusion_from_samples(
    samples: Sequence[Dict[str, Any]],
    model_path: str | Path,
    *,
    clip_length: int = DEFAULT_CLIP_LENGTH,
    image_size: int = DEFAULT_IMAGE_SIZE,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 0.0005,
    device: str = "auto",
    thresholds: Optional[Dict[str, float]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Train a candidate model with deterministic source-group holdout."""
    _require_torch()
    clip_length = max(4, min(int(clip_length), 48))
    image_size = max(32, min(int(image_size), 128))
    epochs = max(1, min(int(epochs), 50))
    batch_size = max(1, min(int(batch_size), 16))
    thresholds = {
        event_type: max(0.05, min(0.95, _safe_float((thresholds or {}).get(event_type), 0.35)))
        for event_type in TEMPORAL_EVENT_TYPES
    }
    usable = [
        sample for sample in samples
        if _sample_label(sample) != "uncertain" and _sample_is_evaluable(sample)
    ]
    label_counts = Counter(_sample_label(sample) for sample in usable)
    hash_groups: Dict[str, set] = defaultdict(set)
    for index, sample in enumerate(usable):
        content_hash = str(sample.get("content_hash") or "").strip()
        if content_hash:
            hash_groups[content_hash].add(_group_for_sample(sample, index))
    duplicate_group_collisions = sum(1 for groups in hash_groups.values() if len(groups) > 1)
    if duplicate_group_collisions:
        raise ValueError(f"检测到 {duplicate_group_collisions} 个跨来源重复样本，已阻止数据泄漏训练")
    missing = {
        event_type: int(label_counts.get(event_type, 0))
        for event_type in TEMPORAL_EVENT_TYPES
        if int(label_counts.get(event_type, 0)) < 2
    }
    if missing:
        raise ValueError(f"三类行为都至少需要 2 个正样本才能启动端到端训练: {missing}")
    videos, features, targets, groups, failures = _build_training_arrays(
        usable,
        clip_length=clip_length,
        image_size=image_size,
    )
    if len(set(groups)) < 3:
        raise ValueError(f"至少需要 3 个来源分组才能进行独立验证，当前仅 {len(set(groups))} 个")
    # Split by whole source groups.  The previous fallback selected one sample
    # when the hash bucket was empty/full, which leaked the same video/camera
    # into both train and validation.
    unique_groups = sorted(set(groups))
    validation_groups = {
        group for group in unique_groups
        if int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 5 == 0
    }
    if not validation_groups or validation_groups == set(unique_groups):
        validation_groups = {unique_groups[0]}
    valid_mask = np.asarray([group in validation_groups for group in groups], dtype=bool)
    train_mask = ~valid_mask
    if not train_mask.any():
        raise ValueError("来源分组划分后没有训练样本")

    selected_device = "cuda" if device in {"auto", "cuda", "0"} and torch.cuda.is_available() else "cpu"
    model = TemporalFusionNet()
    model.to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max(1e-6, float(learning_rate)), weight_decay=1e-4)
    positive = targets[train_mask].sum(axis=0)
    negative = max(1.0, float(train_mask.sum())) - positive
    pos_weight = torch.tensor(np.maximum(1.0, negative / np.maximum(positive, 1.0)), dtype=torch.float32, device=selected_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_dataset = TensorDataset(
        torch.from_numpy(videos[train_mask]),
        torch.from_numpy(features[train_mask]),
        torch.from_numpy(targets[train_mask]),
    )
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.train()
    for epoch in range(epochs):
        if cancel_callback and cancel_callback():
            raise RuntimeError("端到端行为训练已取消")
        epoch_loss = 0.0
        batches = 0
        for video_batch, feature_batch, target_batch in loader:
            if cancel_callback and cancel_callback():
                raise RuntimeError("端到端行为训练已取消")
            video_batch = video_batch.to(selected_device)
            feature_batch = feature_batch.to(selected_device)
            target_batch = target_batch.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(video_batch, feature_batch), target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        if progress_callback:
            progress_callback(25.0 + (epoch + 1) / epochs * 55.0, f"端到端模型训练 {epoch + 1}/{epochs}")

    model.eval()
    with torch.no_grad():
        val_video = torch.from_numpy(videos[valid_mask]).to(selected_device)
        val_features = torch.from_numpy(features[valid_mask]).to(selected_device)
        probabilities = torch.sigmoid(model(val_video, val_features)).cpu().numpy() if len(val_video) else np.zeros((0, len(TEMPORAL_EVENT_TYPES)))
    validation = _metrics(targets[valid_mask], probabilities, thresholds) if len(val_video) else {"sample_count": 0, "per_event": {}}
    snapshot = [
        {"id": sample.get("id"), "label": _sample_label(sample), "content_hash": sample.get("content_hash"), "group": _group_for_sample(sample, index)}
        for index, sample in enumerate(usable)
    ]
    snapshot_hash = hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    metadata = {
        "model_type": MODEL_TYPE,
        "model_family": MODEL_FAMILY,
        "supported_event_types": list(TEMPORAL_EVENT_TYPES),
        "clip_length": clip_length,
        "image_size": image_size,
        "sample_fps": DEFAULT_SAMPLE_FPS,
        "feature_size": FEATURE_SIZE,
        "thresholds": thresholds,
        "counts": dict(label_counts),
        "sample_count": int(len(videos)),
        "source_group_count": int(len(set(groups))),
        "train_group_count": int(len(set(group for group, use in zip(groups, train_mask) if use))),
        "validation_group_count": int(len(set(group for group, use in zip(groups, valid_mask) if use))),
        "validation_groups": sorted(validation_groups),
        "sample_snapshot_hash": snapshot_hash,
        "leakage_check": {
            "status": "passed",
            "duplicate_group_collisions": duplicate_group_collisions,
        },
        "failed_samples": failures[:100],
        "grouped_validation": {
            "available": bool(len(val_video) and len(set(groups)) >= 3),
            "group_count": int(len(set(groups))),
            **validation,
        },
        "train_loss": round(float(epoch_loss / max(1, batches)), 6),
        "device": selected_device,
    }
    target = Path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save({"model_type": MODEL_TYPE, "metadata": metadata, "state_dict": model.state_dict()}, temporary)
    temporary.replace(target)
    if progress_callback:
        progress_callback(95.0, "端到端候选模型已保存")
    return {
        "success": True,
        "message": "端到端行为模型训练完成",
        "model_path": str(target),
        "metadata": metadata,
        "labels": list(TEMPORAL_EVENT_TYPES),
        "counts": dict(label_counts),
        "sample_count": int(len(videos)),
        "failed_count": len(failures),
        "failed_samples": failures[:50],
    }


def load_temporal_model(model_path: str | Path, device: str = "cpu") -> Tuple[Any, Dict[str, Any]]:
    _require_torch()
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"端到端行为模型不存在: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.6 has no weights_only argument
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError("不是受支持的端到端行为模型文件")
    model = TemporalFusionNet()
    model.load_state_dict(checkpoint.get("state_dict") or {})
    model.to(device)
    model.eval()
    return model, dict(checkpoint.get("metadata") or {})


class TemporalFusionStream:
    """Per-source bounded track buffers for batch and realtime inference."""

    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        clip_length: int = DEFAULT_CLIP_LENGTH,
        sample_fps: float = DEFAULT_SAMPLE_FPS,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.model = model
        self.model_id = str(model_id or "")
        self.metadata = dict(metadata or {})
        self.clip_length = max(4, min(int(clip_length), 48))
        self.sample_fps = max(1.0, min(float(sample_fps), 15.0))
        configured = self.metadata.get("thresholds") if isinstance(self.metadata.get("thresholds"), dict) else {}
        self.thresholds = {
            event_type: max(0.05, min(0.95, _safe_float((thresholds or {}).get(event_type, configured.get(event_type, 0.35)), 0.35)))
            for event_type in TEMPORAL_EVENT_TYPES
        }
        self.frames: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.clip_length))
        self.features: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.clip_length))
        self.timestamps: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=self.clip_length))
        self.last_seen: Dict[int, float] = {}
        self.last_inference: Dict[int, float] = {}
        self.last_emitted: Dict[Tuple[int, str], float] = {}
        self.inference_stride = max(1, self.clip_length // 4)
        self._update_count = 0
        self._device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    def update(
        self,
        frame: np.ndarray,
        persons: Dict[int, Dict[str, Any]],
        objects: Optional[Dict[Any, Dict[str, Any]]],
        timestamp: float,
        *,
        enabled_events: Optional[Iterable[str]] = None,
        input_is_rgb: bool = False,
    ) -> Dict[str, Any]:
        events: List[Dict[str, Any]] = []
        score_snapshots: List[Dict[str, Any]] = []
        enabled = set(enabled_events or TEMPORAL_EVENT_TYPES) & set(TEMPORAL_EVENT_TYPES)
        self._update_count += 1
        frame_shape = frame.shape[:2] if isinstance(frame, np.ndarray) and frame.ndim >= 2 else (1, 1)
        for raw_track_id, person in (persons or {}).items():
            track_id = int(person.get("track_id", raw_track_id))
            person_enabled = set(enabled)
            if "missing_badge" in person_enabled and not person.get("badge_evaluable", True):
                # An unobservable chest is unknown, but phone/fall remain evaluable.
                person_enabled.discard("missing_badge")
            if not person_enabled:
                continue
            self.last_seen[track_id] = float(timestamp)
            self._prune_tracks(float(timestamp))
            self.frames[track_id].append(
                crop_track_frame(frame, person.get("bbox", []), input_is_rgb=input_is_rgb)
            )
            self.features[track_id].append(build_temporal_feature_vector(person, objects, frame_shape))
            self.timestamps[track_id].append(float(timestamp))
            if len(self.frames[track_id]) < self.clip_length:
                continue
            if self._update_count % self.inference_stride != 0:
                continue
            if timestamp - self.last_inference.get(track_id, -1e9) < 0.05:
                continue
            self.last_inference[track_id] = float(timestamp)
            scores = self._predict(track_id)
            start_time = float(self.timestamps[track_id][0])
            for event_type in person_enabled:
                score = float(scores.get(event_type, 0.0))
                last = self.last_emitted.get((track_id, event_type), -1e9)
                threshold = self.thresholds[event_type]
                cooldown_remaining = max(0.0, self._cooldown(event_type) - (float(timestamp) - last))
                above_threshold = score >= threshold
                score_snapshots.append({
                    "track_id": track_id,
                    "persons": [track_id],
                    "event_type": event_type,
                    "score": round(score, 4),
                    "confidence": round(score, 4),
                    "threshold": round(threshold, 4),
                    "above_threshold": above_threshold,
                    "cooldown_remaining_sec": round(cooldown_remaining, 3),
                    "would_emit": bool(above_threshold and cooldown_remaining <= 0.0),
                    "start_time": start_time,
                    "timestamp": float(timestamp),
                    "model_id": self.model_id,
                    "model_version": self.metadata.get("display_name") or self.model_id,
                    "input_quality": {
                        "track_id": track_id,
                        "frames": len(self.frames[track_id]),
                        "badge_evaluable": person.get("badge_evaluable"),
                    },
                })
                if not above_threshold or cooldown_remaining > 0.0:
                    continue
                self.last_emitted[(track_id, event_type)] = float(timestamp)
                events.append({
                    "event_id": f"temporal-{uuid.uuid4()}",
                    "event_type": event_type,
                    "state": "confirmed",
                    "persons": [track_id],
                    "start_time": start_time,
                    "confirmed_at": float(timestamp),
                    "end_time": float(timestamp),
                    "confidence": round(score, 4),
                    "rule_score": 0.0,
                    "reason": {
                        "model_family": MODEL_FAMILY,
                        "model_id": self.model_id,
                        "model_version": self.metadata.get("display_name") or self.model_id,
                        "temporal_span": {"start": start_time, "end": float(timestamp)},
                        "input_quality": {
                            "track_id": track_id,
                            "frames": len(self.frames[track_id]),
                            "badge_evaluable": person.get("badge_evaluable"),
                        },
                        "scores": scores,
                    },
                })
        return {
            "events": events,
            "score_snapshots": score_snapshots,
            "model_id": self.model_id,
            "model_family": MODEL_FAMILY,
            "tracked_persons": len(self.frames),
            "buffered_frames": sum(len(items) for items in self.frames.values()),
        }

    def _prune_tracks(self, timestamp: float) -> None:
        stale = [track_id for track_id, seen_at in self.last_seen.items() if timestamp - seen_at > 60.0]
        if len(self.last_seen) > 512:
            stale.extend(sorted(self.last_seen, key=self.last_seen.get)[: len(self.last_seen) - 512])
        for track_id in set(stale):
            self.last_seen.pop(track_id, None)
            self.frames.pop(track_id, None)
            self.features.pop(track_id, None)
            self.timestamps.pop(track_id, None)
            self.last_inference.pop(track_id, None)
            for key in [item for item in self.last_emitted if item[0] == track_id]:
                self.last_emitted.pop(key, None)

    def _cooldown(self, event_type: str) -> float:
        return {"phone_use": 3.0, "missing_badge": 8.0, "fall_down": 12.0}.get(event_type, 5.0)

    def _predict(self, track_id: int) -> Dict[str, float]:
        video = torch.from_numpy(np.stack(list(self.frames[track_id]), axis=0)[None, ...]).to(self._device)
        features = torch.from_numpy(np.stack(list(self.features[track_id]), axis=0).mean(axis=0, keepdims=True)).to(self._device)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(video, features))[0].detach().cpu().numpy()
        return {event_type: round(float(probabilities[index]), 4) for index, event_type in enumerate(TEMPORAL_EVENT_TYPES)}
