"""Deterministic YOLOE candidate activation gates."""

from typing import Any, Dict, List


def _number(mapping: Dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"缺少验收指标: {key}")
    return float(value)


def evaluate_yoloe_candidate(
    current_best: Dict[str, Any],
    candidate: Dict[str, Any],
    smoke_tests: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate only major-class mask metrics; sparse/all-class metrics never block."""
    failures: List[Dict[str, Any]] = []
    current_major = current_best.get("major_classes", {}).get("mask", {})
    candidate_major = candidate.get("major_classes", {}).get("mask", {})
    current_map = _number(current_major, "map50_95")
    candidate_map = _number(candidate_major, "map50_95")
    current_map50 = _number(current_major, "map50")
    candidate_map50 = _number(candidate_major, "map50")
    if candidate_map + 1e-12 < current_map + 0.02:
        failures.append(
            {
                "code": "major_map50_95_gain",
                "required": current_map + 0.02,
                "actual": candidate_map,
            }
        )
    if candidate_map50 + 1e-12 < current_map50:
        failures.append(
            {
                "code": "major_map50_regression",
                "required": current_map50,
                "actual": candidate_map50,
            }
        )

    current_rows = {
        str(item.get("name")): item
        for item in current_best.get("per_class", {}).get("mask", [])
        if item.get("name")
    }
    candidate_rows = {
        str(item.get("name")): item
        for item in candidate.get("per_class", {}).get("mask", [])
        if item.get("name")
    }
    major_names = [
        str(item.get("name"))
        for item in candidate.get("major_class_rows", [])
        if item.get("name")
    ]
    if not major_names:
        major_ids = {int(value) for value in candidate.get("major_class_ids", [])}
        major_names = [
            str(item.get("name"))
            for item in candidate.get("per_class", {}).get("mask", [])
            if int(item.get("class_id", -1)) in major_ids
        ]
    for name in major_names:
        baseline_row = current_rows.get(name)
        candidate_row = candidate_rows.get(name)
        if not baseline_row or not candidate_row:
            failures.append({"code": "major_class_metric_missing", "class_name": name})
            continue
        baseline_recall = _number(baseline_row, "recall")
        candidate_recall = _number(candidate_row, "recall")
        if candidate_recall + 1e-12 < baseline_recall - 0.10:
            failures.append(
                {
                    "code": "major_recall_regression",
                    "class_name": name,
                    "required": baseline_recall - 0.10,
                    "actual": candidate_recall,
                }
            )

    for key in ("keyword_switch", "untrained_prompt", "mask_output", "text_pe"):
        if smoke_tests.get(key) is not True:
            failures.append({"code": f"smoke_{key}", "actual": smoke_tests.get(key)})
    return {
        "passed": not failures,
        "failures": failures,
        "thresholds": {
            "major_map50_95_absolute_gain": 0.02,
            "major_map50_minimum": current_map50,
            "major_recall_max_drop": 0.10,
            "major_class_threshold": 20,
        },
    }
