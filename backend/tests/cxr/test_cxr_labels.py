"""CXR 标签契约：四态解析、固定顺序、mask 与输出结构。"""
from __future__ import annotations

import pytest

from app.schemas.cxr import (
    BORDERLINE_MARGIN,
    CXR_LABEL_SCHEMA,
    CXR_LABELS,
    CXR_TASK,
    LABEL_KEYS,
    RAW_NEGATIVE,
    RAW_POSITIVE,
    RAW_UNCERTAIN,
    RAW_UNMENTIONED,
    CxrLabelError,
    build_label_results,
    mask_vector,
    normalize_raw_labels,
    parse_raw_label,
    raw_state_counts,
    schema_payload,
    target_vector,
)


def row(**overrides) -> dict:
    base = {
        "Atelectasis": 1,
        "Cardiomegaly": 0,
        "Consolidation": -1,
        "Edema": "",
        "Pleural Effusion": 1,
    }
    base.update(overrides)
    return base


def test_raw_four_states_are_not_collapsed():
    assert parse_raw_label(1) == RAW_POSITIVE
    assert parse_raw_label(0) == RAW_NEGATIVE
    assert parse_raw_label(-1) == RAW_UNCERTAIN
    assert parse_raw_label("") == RAW_UNMENTIONED
    assert parse_raw_label(None) == RAW_UNMENTIONED
    assert parse_raw_label("1") == RAW_POSITIVE
    assert parse_raw_label("-1") == RAW_UNCERTAIN
    # -1 绝不能被 bool() 当成 True
    assert parse_raw_label(-1) != RAW_POSITIVE


def test_bool_and_unknown_values_are_rejected():
    with pytest.raises(CxrLabelError):
        parse_raw_label(True)
    with pytest.raises(CxrLabelError):
        parse_raw_label(2)
    with pytest.raises(CxrLabelError):
        parse_raw_label("positive-ish")


def test_missing_label_column_is_rejected_not_guessed():
    broken = row()
    broken.pop("Edema")
    with pytest.raises(CxrLabelError) as excinfo:
        normalize_raw_labels(broken)
    assert "Edema" in excinfo.value.details["missing_columns"]


def test_label_order_is_fixed_by_contract():
    assert LABEL_KEYS == ("atelectasis", "cardiomegaly", "consolidation", "edema", "pleural_effusion")
    assert [label.index for label in CXR_LABELS] == [0, 1, 2, 3, 4]


def test_mask_excludes_uncertain_and_unmentioned_but_keeps_state():
    labels = normalize_raw_labels(row())
    assert labels["consolidation"] == RAW_UNCERTAIN
    assert labels["edema"] == RAW_UNMENTIONED
    assert mask_vector(labels) == [1, 1, 0, 0, 1]
    assert target_vector(labels) == [1, 0, None, None, 1]


def test_multi_label_coexistence_and_no_sum_to_one():
    results = build_label_results(
        {"atelectasis": 0.9, "cardiomegaly": 0.8, "consolidation": 0.7, "edema": 0.6, "pleural_effusion": 0.55},
        None,
        {key: 0.5 for key in LABEL_KEYS},
        calibration_status="not_calibrated",
        threshold_source="default_no_calibration",
    )
    assert len(results) == 5
    assert all(item["flag"] == "above_threshold" for item in results)
    assert all(item["calibrated_score"] is None for item in results)


def test_borderline_margin_and_not_available():
    results = build_label_results(
        {"atelectasis": 0.5 + BORDERLINE_MARGIN / 2, "cardiomegaly": 0.5 + BORDERLINE_MARGIN * 3},
        None,
        {"atelectasis": 0.5, "cardiomegaly": 0.5},
        calibration_status="not_calibrated",
        threshold_source="default_no_calibration",
    )
    by_key = {item["label_key"]: item for item in results}
    assert by_key["atelectasis"]["flag"] == "borderline"
    assert by_key["cardiomegaly"]["flag"] == "above_threshold"
    # 没有阈值的标签不得被判成 below（不能暗示"已排除"）
    assert by_key["edema"]["flag"] == "not_available"


def test_raw_state_counts_cover_all_states():
    counts = raw_state_counts([normalize_raw_labels(row()), normalize_raw_labels(row(Atelectasis=0))])
    assert counts["atelectasis"][RAW_POSITIVE] == 1
    assert counts["atelectasis"][RAW_NEGATIVE] == 1
    assert counts["consolidation"][RAW_UNCERTAIN] == 2
    assert set(counts["edema"]) == {"positive", "negative", "uncertain", "unmentioned"}


def test_schema_payload_is_stable_and_public():
    payload = schema_payload()
    assert payload["task"] == CXR_TASK
    assert payload["label_schema"] == CXR_LABEL_SCHEMA
    assert [item["key"] for item in payload["labels"]] == list(LABEL_KEYS)
    assert payload["splits"] == ["train", "val", "calibration", "test"]
    # 不得泄露来源配置或路径
    assert "source" not in payload
    assert "root" not in payload
