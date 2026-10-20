"""Model-aware training profiles and metric grouping."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Iterable, List, Optional
from model_capabilities import model_supports_text_prompts


TRAINING_PROFILES = {"auto", "standard_yolo", "yoloe_pe_segment"}
MAJOR_CLASS_THRESHOLD = 20


def model_supports_yoloe(*references: Any) -> bool:
    return model_supports_text_prompts(*references)


def resolve_training_profile(requested: Any, *model_references: Any) -> str:
    profile = str(requested or "auto").strip().lower()
    if profile not in TRAINING_PROFILES:
        raise ValueError(f"不支持的训练模式: {profile}")
    is_yoloe = model_supports_yoloe(*model_references)
    if profile == "auto":
        return "yoloe_pe_segment" if is_yoloe else "standard_yolo"
    if profile == "yoloe_pe_segment" and not is_yoloe:
        raise ValueError("yoloe_pe_segment 只能用于 YOLOE 模型")
    return profile


def recommended_training_config(profile: str) -> Dict[str, Any]:
    if profile == "yoloe_pe_segment":
        return {
            "epochs": 150,
            "batch_size": 16,
            "img_size": 640,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "patience": 30,
            "major_class_threshold": MAJOR_CLASS_THRESHOLD,
        }
    return {
        "epochs": 100,
        "batch_size": 16,
        "img_size": 640,
        "learning_rate": 0.01,
        "optimizer": "AdamW",
        "patience": 20,
        "major_class_threshold": MAJOR_CLASS_THRESHOLD,
    }


def normalize_training_config(config: Dict[str, Any], profile: str) -> Dict[str, Any]:
    normalized = dict(config or {})
    defaults = recommended_training_config(profile)
    for key in ("epochs", "batch_size", "img_size", "learning_rate"):
        if normalized.get(key) is None:
            normalized[key] = defaults[key]
    normalized["training_profile"] = profile
    normalized["major_class_threshold"] = int(
        normalized.get("major_class_threshold") or MAJOR_CLASS_THRESHOLD
    )
    normalized["patience"] = int(normalized.get("patience") or defaults["patience"])
    normalized["optimizer"] = str(normalized.get("optimizer") or defaults["optimizer"])
    return normalized


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _component_per_class(component: Any, class_names: List[str]) -> List[Dict[str, Any]]:
    if component is None:
        return []
    precision = _to_list(getattr(component, "p", None))
    recall = _to_list(getattr(component, "r", None))
    ap50 = _to_list(getattr(component, "ap50", None))
    maps = _to_list(getattr(component, "maps", None))
    class_indices = _to_list(getattr(component, "ap_class_index", None))
    if not class_indices:
        class_indices = list(range(max(len(precision), len(recall), len(ap50), len(maps))))
    rows: List[Dict[str, Any]] = []
    for metric_index, raw_class_id in enumerate(class_indices):
        try:
            class_id = int(raw_class_id)
        except (TypeError, ValueError):
            continue
        if class_id < 0 or class_id >= len(class_names):
            continue
        row = {"class_id": class_id, "name": class_names[class_id]}
        for key, values in (
            ("precision", precision),
            ("recall", recall),
            ("map50", ap50),
            ("map50_95", maps),
        ):
            if metric_index < len(values):
                converted = _to_float(values[metric_index])
                if converted is not None:
                    row[key] = converted
        rows.append(row)
    return rows


def _macro(rows: Iterable[Dict[str, Any]], keys: Iterable[str]) -> Dict[str, float]:
    materialized = list(rows)
    result: Dict[str, float] = {}
    for key in keys:
        values = [
            float(row[key])
            for row in materialized
            if row.get(key) is not None
        ]
        if values:
            result[key] = float(mean(values))
    return result


def extract_grouped_metrics(
    validator_metrics: Any,
    preflight: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract box/mask per-class metrics and macro averages for nonzero/major classes."""
    class_stats = list((preflight or {}).get("class_stats") or [])
    mapped_stats = [
        item
        for item in class_stats
        if item.get("new_class_id") is not None
    ]
    mapped_stats.sort(key=lambda item: int(item["new_class_id"]))
    class_names = [str(item.get("name") or f"class_{index}") for index, item in enumerate(mapped_stats)]
    major_ids = {
        int(item["new_class_id"])
        for item in mapped_stats
        if bool(item.get("is_major"))
    }
    output: Dict[str, Any] = {
        "major_class_threshold": int(
            (preflight or {}).get("major_class_threshold") or MAJOR_CLASS_THRESHOLD
        ),
        "major_class_ids": sorted(major_ids),
        "all_classes": {},
        "major_classes": {},
        "per_class": {},
    }
    for label, attribute in (("box", "box"), ("mask", "seg")):
        component = getattr(validator_metrics, attribute, None) if validator_metrics is not None else None
        rows = _component_per_class(component, class_names)
        major_rows = [row for row in rows if int(row["class_id"]) in major_ids]
        output["per_class"][label] = rows
        output["all_classes"][label] = _macro(rows, ("precision", "recall", "map50", "map50_95"))
        output["major_classes"][label] = _macro(
            major_rows,
            ("precision", "recall", "map50", "map50_95"),
        )
    return output
