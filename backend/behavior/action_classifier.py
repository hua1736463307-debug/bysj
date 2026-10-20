"""Lightweight behavior event classifier with Pose-Feature support.

This version supports both reviewed video clips and reviewed image samples.

Model type:
- standardized nearest-centroid classifier

Training samples:
- clip sample: {"sample_type": "clip", "clip_path": "...mp4", "label": "..."}
- image sample: {"sample_type": "image", "image_path": "...jpg", "label": "..."}

Feature:
- YOLOv8-Pose COCO-17 skeleton feature
- image and clip share the same fixed-length feature vector
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .pose_clip_extractor import ClipPoseExtractor
from .pose_feature_encoder import POSE_FEATURE_NAMES
from .pose_image_extractor import ImagePoseExtractor

DEFAULT_MODEL_PATH = "/app/models/action_event_classifier.pkl"
DEFAULT_EXCLUDED_LABELS = {"delete", "uncertain"}

FEATURE_NAMES = list(POSE_FEATURE_NAMES)

LABEL_NAME_MAP = {
    "phone_use": "\u4f7f\u7528\u624b\u673a",
    "fall_down": "\u4eba\u5458\u6454\u5012",
    "missing_badge": "\u672a\u4f69\u6234\u5de5\u724c",
    "normal": "普通行为",
}


def _now() -> str:
    return datetime.now().isoformat()


def _softmax_negative_distance(distances: np.ndarray) -> np.ndarray:
    if distances.size == 0:
        return distances
    scale = float(np.std(distances))
    if scale <= 1e-9:
        scale = max(1.0, float(np.mean(distances)) + 1e-6)
    logits = -distances / scale
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    total = np.sum(exp)
    if total <= 1e-12:
        return np.ones_like(distances) / len(distances)
    return exp / total


class CentroidActionClassifier:
    def __init__(self):
        self.labels: List[str] = []
        self.label_names: Dict[str, str] = dict(LABEL_NAME_MAP)
        self.feature_names: List[str] = list(FEATURE_NAMES)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.centroids: Dict[str, np.ndarray] = {}
        self.counts: Dict[str, int] = {}
        self.metadata: Dict[str, Any] = {}

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("分类器尚未训练")
        return (x - self.mean) / self.std

    def fit(self, x: np.ndarray, y: List[str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if x.ndim != 2:
            raise ValueError("训练特征必须是二维矩阵")
        if len(y) != x.shape[0]:
            raise ValueError("样本数量和标签数量不一致")
        unique_labels = sorted(set(y))
        if len(unique_labels) < 2:
            raise ValueError("至少需要 2 个类别才能训练分类器")

        self.labels = unique_labels
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-9] = 1.0
        z = self._standardize(x)

        self.centroids = {}
        self.counts = {}
        for label in self.labels:
            indices = [i for i, item in enumerate(y) if item == label]
            self.centroids[label] = z[indices].mean(axis=0)
            self.counts[label] = len(indices)

        preds = [self.predict_vector(row)["label"] for row in x]
        train_acc = float(np.mean([p == t for p, t in zip(preds, y)])) if y else 0.0
        loo_acc = self._leave_one_out_accuracy(x, y)

        self.metadata = {
            **(metadata or {}),
            "model_type": "standardized_nearest_centroid_pose_feature",
            "created_at": _now(),
            "feature_names": list(self.feature_names),
            "feature_count": int(x.shape[1]),
            "labels": list(self.labels),
            "label_names": self.label_names,
            "counts": dict(self.counts),
            "train_accuracy": round(train_acc, 4),
            "leave_one_out_accuracy": None if loo_acc is None else round(float(loo_acc), 4),
            "sample_count": int(len(y)),
        }
        return dict(self.metadata)

    def _leave_one_out_accuracy(self, x: np.ndarray, y: List[str]) -> Optional[float]:
        if len(y) < 4 or len(set(y)) < 2:
            return None
        correct = 0
        evaluated = 0
        for i in range(len(y)):
            train_indices = [j for j in range(len(y)) if j != i]
            y_train = [y[j] for j in train_indices]
            if y[i] not in y_train:
                continue
            if len(set(y_train)) < 2:
                continue
            tmp = CentroidActionClassifier()
            tmp.fit_without_metrics(x[train_indices], y_train)
            pred = tmp.predict_vector(x[i])["label"]
            correct += int(pred == y[i])
            evaluated += 1
        if evaluated == 0:
            return None
        return correct / evaluated

    def fit_without_metrics(self, x: np.ndarray, y: List[str]) -> None:
        self.labels = sorted(set(y))
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-9] = 1.0
        z = self._standardize(x)
        self.centroids = {}
        self.counts = {}
        for label in self.labels:
            indices = [i for i, item in enumerate(y) if item == label]
            self.centroids[label] = z[indices].mean(axis=0)
            self.counts[label] = len(indices)

    def predict_vector(self, vector: Iterable[float]) -> Dict[str, Any]:
        if not self.centroids:
            raise RuntimeError("分类器尚未训练")
        x = np.asarray(list(vector), dtype=np.float32)
        z = self._standardize(x)
        labels = list(self.labels)
        distances = np.asarray([np.linalg.norm(z - self.centroids[label]) for label in labels], dtype=np.float32)
        probs = _softmax_negative_distance(distances)
        best_idx = int(np.argmax(probs))
        label = labels[best_idx]
        return {
            "label": label,
            "label_name": self.label_names.get(label, label),
            "confidence": float(probs[best_idx]),
            "scores": {
                labels[i]: float(probs[i])
                for i in range(len(labels))
            },
            "distances": {
                labels[i]: float(distances[i])
                for i in range(len(labels))
            },
        }

    def predict_clip(self, clip_path: str | Path, max_frames: int = 96) -> Dict[str, Any]:
        extractor = ClipPoseExtractor()
        features = extractor.extract(clip_path, max_frames=max_frames)
        if features is None:
            raise ValueError(f"clip 中未提取到可用姿态特征: {clip_path}")
        pred = self.predict_vector(features)
        pred["clip_path"] = str(clip_path)
        pred["feature_map"] = {
            name: float(features[i])
            for i, name in enumerate(self.feature_names[: len(features)])
        }
        return pred

    def predict_image(self, image_path: str | Path) -> Dict[str, Any]:
        extractor = ImagePoseExtractor()
        features = extractor.extract(str(image_path))
        if features is None:
            raise ValueError(f"图片中未提取到可用姿态特征: {image_path}")
        pred = self.predict_vector(features)
        pred["image_path"] = str(image_path)
        pred["feature_map"] = {
            name: float(features[i])
            for i, name in enumerate(self.feature_names[: len(features)])
        }
        return pred

    def save(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> str:
        target = Path(model_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "labels": self.labels,
            "label_names": self.label_names,
            "feature_names": self.feature_names,
            "mean": self.mean,
            "std": self.std,
            "centroids": self.centroids,
            "counts": self.counts,
            "metadata": self.metadata,
        }
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        tmp.replace(target)
        return str(target)

    @classmethod
    def load(cls, model_path: str | Path = DEFAULT_MODEL_PATH) -> "CentroidActionClassifier":
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"动作复核模型不存在: {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        clf = cls()
        clf.labels = list(payload.get("labels") or [])
        clf.label_names = dict(payload.get("label_names") or LABEL_NAME_MAP)
        clf.feature_names = list(payload.get("feature_names") or FEATURE_NAMES)
        clf.mean = payload.get("mean")
        clf.std = payload.get("std")
        clf.centroids = dict(payload.get("centroids") or {})
        clf.counts = dict(payload.get("counts") or {})
        clf.metadata = dict(payload.get("metadata") or {})
        return clf


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _classification_metrics(expected: List[str], predicted: List[str], labels: List[str]) -> Dict[str, Any]:
    total = len(expected)
    correct = sum(1 for truth, guess in zip(expected, predicted) if truth == guess)
    per_class: Dict[str, Dict[str, Any]] = {}
    confusion_matrix = {
        truth: {guess: 0 for guess in labels}
        for truth in labels
    }
    for truth, guess in zip(expected, predicted):
        if truth in confusion_matrix and guess in confusion_matrix[truth]:
            confusion_matrix[truth][guess] += 1
    for label in labels:
        tp = sum(1 for truth, guess in zip(expected, predicted) if truth == label and guess == label)
        fp = sum(1 for truth, guess in zip(expected, predicted) if truth != label and guess == label)
        fn = sum(1 for truth, guess in zip(expected, predicted) if truth == label and guess != label)
        tn = sum(1 for truth, guess in zip(expected, predicted) if truth != label and guess != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": sum(1 for item in expected if item == label),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "false_positive_rate": round(float(fp / (fp + tn)), 4) if fp + tn else 0.0,
            "miss_rate": round(float(fn / (fn + tp)), 4) if fn + tp else 0.0,
        }
    macro_f1 = float(np.mean([item["f1"] for item in per_class.values()])) if per_class else 0.0
    return {
        "sample_count": total,
        "accuracy": round(correct / total, 4) if total else None,
        "macro_f1": round(macro_f1, 4) if total else None,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }


class PoseEventVerifier(CentroidActionClassifier):
    """One-vs-rest logistic verifier trained on human-reviewed Pose features."""

    MODEL_TYPE = "one_vs_rest_logistic_pose_verifier_v1"

    def __init__(self):
        super().__init__()
        self.weights: Dict[str, np.ndarray] = {}
        self.biases: Dict[str, float] = {}

    def fit_without_metrics(self, x: np.ndarray, y: List[str]) -> None:
        self.labels = sorted(set(y))
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-9] = 1.0
        z = self._standardize(x)
        self.weights = {}
        self.biases = {}
        self.counts = dict(Counter(y))
        sample_count = max(1, len(y))

        for label in self.labels:
            targets = np.asarray([1.0 if item == label else 0.0 for item in y], dtype=np.float32)
            positive_count = max(1, int(np.sum(targets)))
            negative_count = max(1, sample_count - positive_count)
            sample_weights = np.where(
                targets > 0.5,
                sample_count / (2.0 * positive_count),
                sample_count / (2.0 * negative_count),
            ).astype(np.float32)
            weights = np.zeros(z.shape[1], dtype=np.float32)
            bias = 0.0
            learning_rate = 0.08
            regularization = 0.002
            normalizer = max(float(np.sum(sample_weights)), 1.0)
            for _ in range(500):
                probabilities = _sigmoid(z @ weights + bias)
                errors = (probabilities - targets) * sample_weights
                gradient = (z.T @ errors) / normalizer + regularization * weights
                bias_gradient = float(np.sum(errors) / normalizer)
                weights -= learning_rate * gradient.astype(np.float32)
                bias -= learning_rate * bias_gradient
            self.weights[label] = weights
            self.biases[label] = float(bias)

    def fit(
        self,
        x: np.ndarray,
        y: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        groups: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if x.ndim != 2 or len(y) != x.shape[0]:
            raise ValueError("训练特征和标签数量不一致")
        if len(set(y)) < 2:
            raise ValueError("至少需要 2 个类别才能训练验证器")

        grouped_validation = self._grouped_validation(x, y, groups or [])
        self.fit_without_metrics(x, y)
        train_predictions = [self.predict_vector(row)["label"] for row in x]
        train_metrics = _classification_metrics(y, train_predictions, self.labels)
        self.metadata = {
            **(metadata or {}),
            "model_type": self.MODEL_TYPE,
            "created_at": _now(),
            "feature_names": list(self.feature_names),
            "feature_count": int(x.shape[1]),
            "labels": list(self.labels),
            "label_names": self.label_names,
            "counts": dict(self.counts),
            "sample_count": len(y),
            "train_metrics": train_metrics,
            "grouped_validation": grouped_validation,
        }
        return dict(self.metadata)

    def _grouped_validation(self, x: np.ndarray, y: List[str], groups: List[str]) -> Dict[str, Any]:
        normalized_groups = [str(group or f"sample-{index}") for index, group in enumerate(groups)]
        unique_groups = sorted(set(normalized_groups))
        if len(normalized_groups) != len(y) or len(unique_groups) < 3:
            return {
                "available": False,
                "reason": "at least three source tasks/videos are required",
                "group_count": len(unique_groups),
            }

        expected: List[str] = []
        predicted: List[str] = []
        evaluated_groups = set()
        for fold in range(3):
            test_indices = [
                index for index, group in enumerate(normalized_groups)
                if int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 3 == fold
            ]
            train_indices = [index for index in range(len(y)) if index not in set(test_indices)]
            if not test_indices or len(set(y[index] for index in train_indices)) < 2:
                continue
            verifier = PoseEventVerifier()
            verifier.fit_without_metrics(x[train_indices], [y[index] for index in train_indices])
            for index in test_indices:
                expected.append(y[index])
                predicted.append(verifier.predict_vector(x[index])["label"])
                evaluated_groups.add(normalized_groups[index])

        if not expected:
            return {
                "available": False,
                "reason": "grouped folds did not contain trainable class coverage",
                "group_count": len(unique_groups),
            }
        return {
            "available": True,
            "group_count": len(unique_groups),
            "evaluated_group_count": len(evaluated_groups),
            **_classification_metrics(expected, predicted, sorted(set(y))),
        }

    def predict_vector(self, vector: Iterable[float]) -> Dict[str, Any]:
        if not self.weights:
            raise RuntimeError("验证器尚未训练")
        x = np.asarray(list(vector), dtype=np.float32)
        z = self._standardize(x)
        scores = {
            label: float(_sigmoid(np.asarray([z @ self.weights[label] + self.biases[label]]))[0])
            for label in self.labels
        }
        label = max(scores, key=scores.get)
        return {
            "label": label,
            "label_name": self.label_names.get(label, label),
            "confidence": scores[label],
            "scores": scores,
        }

    def save(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> str:
        target = Path(model_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.MODEL_TYPE,
            "labels": self.labels,
            "label_names": self.label_names,
            "feature_names": self.feature_names,
            "mean": self.mean,
            "std": self.std,
            "weights": self.weights,
            "biases": self.biases,
            "counts": self.counts,
            "metadata": self.metadata,
        }
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "wb") as handle:
            pickle.dump(payload, handle)
        tmp.replace(target)
        return str(target)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "PoseEventVerifier":
        verifier = cls()
        verifier.labels = list(payload.get("labels") or [])
        verifier.label_names = dict(payload.get("label_names") or LABEL_NAME_MAP)
        verifier.feature_names = list(payload.get("feature_names") or FEATURE_NAMES)
        verifier.mean = payload.get("mean")
        verifier.std = payload.get("std")
        verifier.weights = dict(payload.get("weights") or {})
        verifier.biases = {key: float(value) for key, value in dict(payload.get("biases") or {}).items()}
        verifier.counts = dict(payload.get("counts") or {})
        verifier.metadata = dict(payload.get("metadata") or {})
        return verifier


def load_action_verifier(model_path: str | Path = DEFAULT_MODEL_PATH):
    path = Path(model_path)
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if payload.get("model_type") == PoseEventVerifier.MODEL_TYPE or payload.get("weights"):
        return PoseEventVerifier.from_payload(payload)
    classifier = CentroidActionClassifier()
    classifier.labels = list(payload.get("labels") or [])
    classifier.label_names = dict(payload.get("label_names") or LABEL_NAME_MAP)
    classifier.feature_names = list(payload.get("feature_names") or FEATURE_NAMES)
    classifier.mean = payload.get("mean")
    classifier.std = payload.get("std")
    classifier.centroids = dict(payload.get("centroids") or {})
    classifier.counts = dict(payload.get("counts") or {})
    classifier.metadata = dict(payload.get("metadata") or {})
    return classifier


def _sample_path_exists(sample: Dict[str, Any]) -> bool:
    sample_type = str(sample.get("sample_type") or "clip")
    if sample_type == "image":
        image_path = sample.get("image_path") or sample.get("snapshot")
        return bool(image_path and Path(str(image_path)).exists())
    clip_path = sample.get("clip_path")
    return bool(clip_path and Path(str(clip_path)).exists())


def _filter_samples(
    samples: List[Dict[str, Any]],
    labels: Optional[List[str]] = None,
    include_uncertain: bool = False,
) -> List[Dict[str, Any]]:
    allowed = set(labels or [])
    excluded = set(DEFAULT_EXCLUDED_LABELS)
    if include_uncertain:
        excluded.discard("uncertain")

    filtered: List[Dict[str, Any]] = []
    for sample in samples:
        evidence = sample.get("evidence") if isinstance(sample.get("evidence"), dict) else {}
        if (
            sample.get("evaluable") is False
            or evidence.get("evaluable") is False
            or str(sample.get("status") or evidence.get("review_status") or "").strip().lower()
            in {"not_evaluable", "not-evaluable"}
        ):
            continue
        label = str(sample.get("label") or "").strip()
        if not label:
            continue
        if label in excluded:
            continue
        if allowed and label not in allowed:
            continue
        if not _sample_path_exists(sample):
            continue
        filtered.append(sample)
    return filtered


def _extract_sample_features(
    sample: Dict[str, Any],
    clip_extractor: ClipPoseExtractor,
    image_extractor: ImagePoseExtractor,
    max_frames_per_clip: int,
) -> np.ndarray:
    sample_type = str(sample.get("sample_type") or "clip")
    if sample_type == "image":
        image_path = sample.get("image_path") or sample.get("snapshot")
        if not image_path:
            raise ValueError("图片样本缺少 image_path")
        features = image_extractor.extract(str(image_path))
        if features is None:
            raise ValueError(f"图片中未提取到可用姿态: {image_path}")
        return features

    clip_path = sample.get("clip_path")
    if not clip_path:
        raise ValueError("视频样本缺少 clip_path")
    features = clip_extractor.extract(str(clip_path), max_frames=max_frames_per_clip)
    if features is None:
        raise ValueError(f"clip 中未提取到可用姿态: {clip_path}")
    return features


def train_classifier_from_samples(
    samples: List[Dict[str, Any]],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    labels: Optional[List[str]] = None,
    include_uncertain: bool = False,
    min_samples_per_class: int = 1,
    max_frames_per_clip: int = 96,
) -> Dict[str, Any]:
    filtered = _filter_samples(samples, labels=labels, include_uncertain=include_uncertain)
    if not filtered:
        raise ValueError("没有可用于训练的复核样本，请先在样本复核页面完成标注")

    counts = Counter(str(item.get("label")) for item in filtered)
    valid_labels = sorted([label for label, count in counts.items() if count >= int(min_samples_per_class)])
    if len(valid_labels) < 2:
        raise ValueError(
            f"至少需要 2 个类别，且每类不少于 {min_samples_per_class} 个样本；当前统计: {dict(counts)}"
        )

    used = [item for item in filtered if item.get("label") in valid_labels]
    x_rows: List[List[float]] = []
    y: List[str] = []
    groups: List[str] = []
    failures: List[Dict[str, str]] = []

    clip_extractor = ClipPoseExtractor()
    image_extractor = ImagePoseExtractor()

    for item in used:
        try:
            features = _extract_sample_features(
                item,
                clip_extractor=clip_extractor,
                image_extractor=image_extractor,
                max_frames_per_clip=max_frames_per_clip,
            )
            x_rows.append(np.asarray(features, dtype=np.float32).tolist())
            y.append(str(item.get("label")))
            groups.append(str(item.get("source_task_id") or item.get("source_video") or item.get("id") or len(groups)))
        except Exception as exc:
            failures.append({
                "sample_type": str(item.get("sample_type") or "clip"),
                "clip_path": str(item.get("clip_path") or ""),
                "image_path": str(item.get("image_path") or ""),
                "label": str(item.get("label")),
                "error": str(exc),
            })

    if len(x_rows) < 2 or len(set(y)) < 2:
        raise ValueError(
            f"有效训练样本不足。成功读取 {len(x_rows)} 个，类别 {sorted(set(y))}，失败 {len(failures)} 个"
        )
    extracted_counts = Counter(y)
    insufficient = {
        label: int(extracted_counts.get(label, 0))
        for label in valid_labels
        if extracted_counts.get(label, 0) < int(min_samples_per_class)
    }
    if insufficient:
        raise ValueError(
            f"特征提取后有效样本不足，每类至少需要 {min_samples_per_class} 个；当前不足类别: {insufficient}"
        )

    x = np.asarray(x_rows, dtype=np.float32)
    sample_snapshot = [
        {
            "id": item.get("id"),
            "source_task_id": item.get("source_task_id"),
            "source_event_id": item.get("source_event_id"),
            "label": item.get("label"),
            "clip_path": item.get("clip_path"),
            "image_path": item.get("image_path"),
        }
        for item in used
    ]
    sample_snapshot_hash = hashlib.sha256(
        json.dumps(sample_snapshot, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    clf = PoseEventVerifier()
    metadata = clf.fit(
        x,
        y,
        groups=groups,
        metadata={
            "requested_labels": labels or [],
            "include_uncertain": bool(include_uncertain),
            "min_samples_per_class": int(min_samples_per_class),
            "max_frames_per_clip": int(max_frames_per_clip),
            "failed_samples": failures,
            "source": "behavior_review_samples",
            "feature_source": "yolov8_pose",
            "sample_snapshot_hash": sample_snapshot_hash,
            "group_field": "source_task_id",
            "sample_type_counts": dict(Counter(str(item.get("sample_type") or "clip") for item in used)),
        },
    )
    saved_path = clf.save(model_path)

    return {
        "success": True,
        "message": "动作复核分类器训练完成",
        "model_path": saved_path,
        "metadata": metadata,
        "labels": clf.labels,
        "counts": clf.counts,
        "sample_count": len(y),
        "failed_count": len(failures),
        "failed_samples": failures[:50],
    }


def train_classifier_from_embedded_features(
    samples: List[Dict[str, Any]],
    model_path: str | Path,
    labels: List[str],
    *,
    feature_family: str,
    min_samples_per_class: int = 20,
) -> Dict[str, Any]:
    filtered = _filter_samples(samples, labels=labels, include_uncertain=False)
    x_rows: List[List[float]] = []
    y: List[str] = []
    groups: List[str] = []
    failures: List[Dict[str, str]] = []
    feature_count: Optional[int] = None
    for item in filtered:
        try:
            evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            decision = evidence.get("decision") if isinstance(evidence.get("decision"), dict) else {}
            if str(decision.get("model_family") or "") != feature_family:
                raise ValueError("样本专用特征类型与目标模型不一致")
            vector = decision.get("training_feature_vector")
            if not isinstance(vector, list) or not vector:
                raise ValueError("样本缺少实时确认阶段的专用特征")
            row = np.asarray(vector, dtype=np.float32)
            if row.ndim != 1 or not np.all(np.isfinite(row)):
                raise ValueError("专用特征格式无效")
            if feature_count is None:
                feature_count = int(row.size)
            if row.size != feature_count:
                raise ValueError(f"专用特征维度不一致: {row.size} != {feature_count}")
            x_rows.append(row.tolist())
            y.append(str(item.get("label") or ""))
            groups.append(str(item.get("source_task_id") or item.get("source_video") or item.get("id")))
        except Exception as exc:
            failures.append({
                "id": str(item.get("id") or ""),
                "label": str(item.get("label") or ""),
                "error": str(exc),
            })

    counts = Counter(y)
    insufficient = {
        label: int(counts.get(label, 0))
        for label in labels
        if counts.get(label, 0) < int(min_samples_per_class)
    }
    if insufficient:
        raise ValueError(
            f"有效专用特征样本不足，每类至少需要 {min_samples_per_class} 个；当前不足类别: {insufficient}"
        )
    if len(set(groups)) < 3:
        raise ValueError(f"至少需要 3 个独立设备/视频来源组；当前仅 {len(set(groups))} 个")

    x = np.asarray(x_rows, dtype=np.float32)
    snapshot = [
        {"id": item.get("id"), "content_hash": item.get("content_hash"), "label": item.get("label")}
        for item in filtered
    ]
    clf = PoseEventVerifier()
    clf.feature_names = [f"{feature_family}_{index}" for index in range(int(feature_count or 0))]
    metadata = clf.fit(
        x,
        y,
        groups=groups,
        metadata={
            "requested_labels": labels,
            "min_samples_per_class": int(min_samples_per_class),
            "failed_samples": failures,
            "source": "behavior_review_samples",
            "feature_source": feature_family,
            "sample_snapshot_hash": hashlib.sha256(
                json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
            "group_field": "source_task_id",
        },
    )
    saved_path = clf.save(model_path)
    return {
        "success": True,
        "message": "专用行为验证器训练完成",
        "model_path": saved_path,
        "metadata": metadata,
        "labels": clf.labels,
        "counts": clf.counts,
        "sample_count": len(y),
        "failed_count": len(failures),
        "failed_samples": failures[:50],
    }


def get_model_info(model_path: str | Path = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        return {
            "exists": False,
            "model_path": str(path),
        }
    clf = load_action_verifier(path)
    stat = path.stat()
    return {
        "exists": True,
        "model_path": str(path),
        "size": stat.st_size,
        "modified_time": stat.st_mtime,
        "metadata": clf.metadata,
        "labels": clf.labels,
        "counts": clf.counts,
    }
