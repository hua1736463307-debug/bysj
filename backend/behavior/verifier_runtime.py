"""Realtime behavior verifier selection and fail-open decision logic."""

from __future__ import annotations

import os
import hashlib
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .action_classifier import load_action_verifier


MODEL_FAMILIES = {
    "phone_use": "pose_object",
    "missing_badge": "badge_image",
}
DEFAULT_THRESHOLD = float(os.getenv("BEHAVIOR_VERIFIER_THRESHOLD", "0.65"))

_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: Dict[Tuple[str, float], Any] = {}


def _badge_image_features(image: Any, chest_roi: Any) -> Any:
    if not isinstance(image, np.ndarray) or image.ndim < 3 or not chest_roi or len(chest_roi) < 4:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in chest_roi[:4]]
    x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size < 16:
        return None
    features = []
    for channel in range(min(3, crop.shape[2])):
        values = crop[:, :, channel].astype(np.float32)
        hist, _ = np.histogram(values, bins=8, range=(0, 256))
        hist = hist.astype(np.float32) / max(1.0, float(hist.sum()))
        features.extend(hist.tolist())
        features.extend([float(values.mean() / 255.0), float(values.std() / 255.0)])
    features.extend([
        float((x2 - x1) / max(1, width)),
        float((y2 - y1) / max(1, height)),
    ])
    return [round(float(value), 6) for value in features]


def prepare_realtime_feature_vectors(events: Iterable[Dict[str, Any]], image: Any) -> None:
    for event in events or []:
        if event.get("state") != "confirmed":
            continue
        event_type = str(event.get("event_type") or "")
        pose_vector = event.get("pose_feature_vector")
        reason = event.get("reason") if isinstance(event.get("reason"), dict) else {}
        if event_type == "missing_badge":
            vector = _badge_image_features(image, reason.get("chest_roi"))
        elif event_type == "phone_use" and pose_vector:
            class_name = str(reason.get("object_class") or "unknown")
            class_bins = [0.0] * 8
            class_index = int(hashlib.sha256(class_name.encode("utf-8")).hexdigest()[:8], 16) % len(class_bins)
            class_bins[class_index] = 1.0
            # The rule engine's canonical field is owner_distance_ratio.  Keep
            # the legacy person_distance_ratio fallback for old persisted events.
            owner_distance_ratio = reason.get("owner_distance_ratio")
            if owner_distance_ratio is None:
                owner_distance_ratio = reason.get("person_distance_ratio")
            vector = [float(value) for value in pose_vector] + [
                float(owner_distance_ratio or 0.0),
                float(reason.get("old_owner_distance") or 0.0),
                float(reason.get("new_owner_distance") or 0.0),
                *class_bins,
            ]
        else:
            vector = pose_vector
        if vector:
            event["training_feature_vector"] = [round(float(value), 6) for value in vector]
            event["feature_schema_version"] = "phone_pose_object_v2" if event_type == "phone_use" else "badge_image_v1"


def clear_behavior_verifier_cache() -> None:
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()


def preload_behavior_verifier(record: Any) -> None:
    _load(record)


def _load(record: Any):
    path = Path(record.path)
    modified = path.stat().st_mtime
    key = (str(path.resolve()), float(modified))
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        model = load_action_verifier(path)
        _MODEL_CACHE[key] = model
        while len(_MODEL_CACHE) > 12:
            _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
        return model


def _records() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from database import ModelRegistry, SessionLocal

    db = SessionLocal()
    try:
        rows = (
            db.query(ModelRegistry)
            .filter(
                ModelRegistry.source == "behavior_verifier",
                ModelRegistry.status.in_(("active", "candidate")),
            )
            .order_by(ModelRegistry.updated_at.desc(), ModelRegistry.created_at.desc())
            .all()
        )
        active: Dict[str, Any] = {}
        candidate: Dict[str, Any] = {}
        for item in rows:
            behavior_type = str((item.metadata_json or {}).get("behavior_type") or "*")
            target = active if item.status == "active" else candidate
            if behavior_type not in target:
                db.expunge(item)
                target[behavior_type] = item
        return active, candidate
    finally:
        db.close()


def _predict(record: Any, vector: Iterable[float], expected_label: str) -> Dict[str, Any]:
    model = _load(record)
    prediction = model.predict_vector(vector)
    confidence = float(prediction.get("confidence") or 0.0)
    label = str(prediction.get("label") or "")
    metadata = dict(record.metadata_json or {})
    threshold = float(metadata.get("verifier_threshold", DEFAULT_THRESHOLD))
    accepted_labels = {expected_label}
    return {
        "model_id": record.id,
        "model_version": record.name,
        "model_sha256": record.sha256,
        "label": label,
        "confidence": round(confidence, 6),
        "threshold": threshold,
        "accepted": bool(label in accepted_labels and confidence >= threshold),
        "scores": prediction.get("scores") or {},
    }


def apply_realtime_verifiers(events: Iterable[Dict[str, Any]]) -> None:
    items = [item for item in (events or []) if item.get("state") == "confirmed"]
    if not items:
        return
    try:
        active_records, candidate_records = _records()
    except Exception as exc:
        active_records, candidate_records = {}, {}
        records_error = str(exc)
    else:
        records_error = None

    for event in items:
        event_type = str(event.get("event_type") or "")
        expected_label = event_type
        active = active_records.get(event_type) or active_records.get("*")
        candidate = candidate_records.get(event_type) or candidate_records.get("*")
        previous_decision = event.get("decision") if isinstance(event.get("decision"), dict) else {}
        pose_vector = event.pop("pose_feature_vector", None)
        vector = event.pop("training_feature_vector", None) or pose_vector
        decision: Dict[str, Any] = {
            "rule": {
                "event_type": event_type,
                "score": float(event.get("rule_score") or 0.0),
                "accepted": True,
            },
            "model_family": MODEL_FAMILIES.get(event_type, "pose"),
            "fail_open": False,
        }
        if vector:
            decision["training_feature_vector"] = vector
        event["should_alert"] = True

        if records_error:
            decision.update({"fail_open": True, "error": records_error})
        elif not vector:
            decision.update({"fail_open": True, "active": {"mode": "rule_fallback", "reason": "model feature unavailable"}})
        elif active is None:
            decision["active"] = {"mode": "rule_fallback", "reason": "no active verifier"}
        else:
            try:
                active_result = _predict(active, vector, expected_label)
                decision["active"] = active_result
                event["should_alert"] = bool(active_result["accepted"])
                event["decision_model_id"] = active.id
                event["decision_model_version"] = active.name
            except Exception as exc:
                decision.update({
                    "fail_open": True,
                    "active": {"mode": "rule_fallback", "model_id": active.id, "error": str(exc)},
                })
                event["should_alert"] = True

        if candidate is not None and vector:
            try:
                decision["shadow"] = _predict(candidate, vector, expected_label)
            except Exception as exc:
                decision["shadow"] = {"model_id": candidate.id, "error": str(exc)}
        if isinstance(previous_decision.get("temporal"), dict):
            decision["temporal"] = previous_decision["temporal"]
        if isinstance(previous_decision.get("final"), dict):
            decision["final"] = previous_decision["final"]
        event["decision"] = decision
        if event.get("should_alert"):
            event["alerted_at"] = event.get("confirmed_at")
