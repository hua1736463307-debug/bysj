"""Pure feedback-to-training mapping shared by APIs, training, and tests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


SUPPORTED_BEHAVIOR_EVENTS = ("phone_use", "missing_badge", "fall_down")
SUPPORTED_BEHAVIOR_LABELS = set(SUPPORTED_BEHAVIOR_EVENTS) | {"normal"}


def normalize_behavior_events(values: Optional[List[Any]]) -> List[str]:
    """Keep the event taxonomy bounded in the system's canonical order."""
    requested = {str(value).strip() for value in (values or []) if str(value).strip()}
    return [event_type for event_type in SUPPORTED_BEHAVIOR_EVENTS if event_type in requested]


def normalize_behavior_label(label: str) -> str:
    normalized = str(label or "").strip()
    return normalized if normalized in SUPPORTED_BEHAVIOR_LABELS else ""


def feedback_training_mapping(
    event_type: str,
    verdict: str,
    actual_label: Optional[str] = None,
    *,
    evaluable: Optional[bool] = None,
) -> Dict[str, Any]:
    # A reviewer may explicitly mark a sample as not evaluable.  Keep the
    # record for audit, but never turn an unobservable frame into a label.
    if verdict == "not_evaluable" or evaluable is False:
        return {"trainable": False, "label": None, "negative_for": []}
    if verdict == "correct":
        return {"trainable": True, "label": event_type, "negative_for": []}
    if verdict == "false_positive":
        return {"trainable": True, "label": "normal", "negative_for": [event_type]}
    if verdict == "wrong_type":
        return {"trainable": True, "label": str(actual_label or ""), "negative_for": [event_type]}
    return {"trainable": False, "label": None, "negative_for": []}


def select_binary_behavior_samples(
    samples: List[Dict[str, Any]],
    behavior_type: str,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for sample in samples:
        # Evidence carries the authoritative evaluability flag.  Accept the
        # top-level form as well for legacy/imported manifests.
        evidence = sample.get("evidence") if isinstance(sample.get("evidence"), dict) else {}
        if (
            sample.get("evaluable") is False
            or evidence.get("evaluable") is False
            or str(sample.get("status") or evidence.get("review_status") or "").strip().lower()
            in {"not_evaluable", "not-evaluable"}
        ):
            continue
        label = normalize_behavior_label(sample.get("label"))
        negative_for = {str(item) for item in (evidence.get("negative_for") or [])}
        if label == behavior_type:
            selected.append({**sample, "label": behavior_type})
        elif behavior_type in negative_for:
            selected.append({**sample, "label": "normal"})
    return selected
