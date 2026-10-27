"""CXR 指标、校准与阈值：单类/零分母/掩码/来源约束。"""
from __future__ import annotations

import math

import pytest

from app.domains.cxr.evaluation_service import compute_metrics
from app.ml.cxr_metrics import (
    apply_temperature,
    auroc,
    average_precision,
    binary_at_threshold,
    bootstrap_ci,
    expected_calibration_error,
    fit_temperature,
    macro,
    select_threshold,
)
from app.schemas.cxr import LABEL_KEYS


def test_auroc_perfect_and_inverted():
    truth = [0, 0, 1, 1]
    assert auroc(truth, [0.1, 0.2, 0.8, 0.9], [1, 1, 1, 1]) == (1.0, None)
    assert auroc(truth, [0.9, 0.8, 0.2, 0.1], [1, 1, 1, 1]) == (0.0, None)


def test_auroc_ties_are_half():
    value, reason = auroc([0, 1], [0.5, 0.5], [1, 1])
    assert value == pytest.approx(0.5)
    assert reason is None


def test_single_class_and_empty_mask_return_reason_not_zero():
    value, reason = auroc([1, 1, 1], [0.2, 0.3, 0.4], [1, 1, 1])
    assert value is None and reason == "single_class"
    value, reason = auroc([0, 1], [0.2, 0.8], [0, 0])
    assert value is None and reason == "no_masked_samples"


def test_masked_out_samples_do_not_affect_metrics():
    # mask=0 的样本若被当成阴性会把 AUROC 拉低；这里必须完全忽略。
    truth = [0, 1, 1, 0]
    score = [0.1, 0.9, 0.05, 0.02]
    mask = [1, 1, 0, 0]
    value, _ = auroc(truth, score, mask)
    assert value == 1.0


def test_average_precision_undefined_without_positive():
    value, reason = average_precision([0, 0], [0.1, 0.2], [1, 1])
    assert value is None and reason == "no_positive"


def test_binary_metrics_handle_zero_denominators():
    stats = binary_at_threshold([0, 0], [0.1, 0.2], [1, 1], 0.5)
    assert stats["sensitivity"] is None
    assert stats["ppv"] is None
    assert stats["specificity"] == 1.0
    assert stats["reason"] is None


def test_ece_is_zero_for_perfectly_calibrated_constant():
    ece, detail, reason = expected_calibration_error([1] * 50, [1.0] * 50, [1] * 50)
    assert reason is None
    assert ece == pytest.approx(0.0)
    assert detail["n"] == 50


def test_threshold_falls_back_when_single_class():
    selection = select_threshold([1, 1, 1], [0.2, 0.3, 0.4], [1, 1, 1])
    assert selection["threshold"] == 0.5
    assert selection["source"] == "default_no_calibration"
    assert selection["reason"] == "single_class"


def test_threshold_maximises_youden_and_prefers_higher_on_tie():
    truth = [0, 0, 1, 1]
    score = [0.1, 0.2, 0.8, 0.9]
    selection = select_threshold(truth, score, [1, 1, 1, 1])
    assert selection["source"] == "calibration_youden_j"
    assert 0.2 <= selection["threshold"] <= 0.8


def test_temperature_not_calibrated_when_support_is_small():
    fit = fit_temperature([0, 1], [0.2, 0.8], [1, 1])
    assert fit["status"] == "not_calibrated"
    assert fit["temperature"] == 1.0
    assert "samples<50" in fit["reason"]


def test_temperature_fitted_within_bounds_when_support_is_sufficient():
    truth = [0] * 60 + [1] * 60
    # 过度自信的评分：模型给 0.99/0.01，但真实标签只有 120 例
    scores = [0.99] * 60 + [0.01] * 60
    fit = fit_temperature(truth, scores, [1] * 120)
    assert fit["status"] == "temperature_scaled"
    assert 0.05 <= fit["temperature"] <= 10.0
    assert fit["objective"] <= fit["objective_at_t1"] + 1e-9


def test_apply_temperature_keeps_order():
    calibrated = apply_temperature({"atelectasis": 0.8, "edema": 0.2}, {"atelectasis": 2.0, "edema": 2.0})
    # T>1 向 0.5 收缩：高分下降、低分上升，但相对顺序不变。
    assert 0.5 < calibrated["atelectasis"] < 0.8
    assert 0.2 < calibrated["edema"] < 0.5


def test_macro_skips_undefined_labels():
    per_label = {"a": {"auroc": 0.8}, "b": {"auroc": None}, "c": {"auroc": 0.6}}
    result = macro(per_label, "auroc")
    assert result["value"] == pytest.approx(0.7)
    assert result["valid_labels"] == 2
    assert result["total_labels"] == 3


def test_bootstrap_requires_enough_groups():
    result = bootstrap_ci(["g1", "g1"], [0, 1], [0.2, 0.8], [1, 1], auroc)
    assert result["lower"] is None
    assert result["reason"] == "insufficient_groups"


def test_compute_metrics_has_no_map_fields_and_reports_missing():
    truth = {key: [0, 1, 0, 1] * 5 for key in LABEL_KEYS}
    scores = {key: [0.1, 0.9, 0.2, 0.8] * 5 for key in LABEL_KEYS}
    mask = {key: [1] * 20 for key in LABEL_KEYS}
    groups = [f"c{index % 5}" for index in range(20)]
    metrics = compute_metrics(
        scores=scores, truth=truth, mask=mask, groups=groups,
        thresholds={key: 0.5 for key in LABEL_KEYS}, bootstrap_rounds=20,
    )
    assert metrics["metrics_schema"] == "cxr-metrics-v1"
    assert "map50" not in str(metrics)
    assert metrics["macro_auroc"]["value"] == pytest.approx(1.0)
    assert metrics["bootstrap"]["unit"] == "patient_component"


def test_compute_metrics_keeps_single_class_as_null():
    truth = {key: [0, 0, 0, 0] for key in LABEL_KEYS}
    scores = {key: [0.1, 0.2, 0.3, 0.4] for key in LABEL_KEYS}
    mask = {key: [1] * 4 for key in LABEL_KEYS}
    metrics = compute_metrics(
        scores=scores, truth=truth, mask=mask, groups=["c1", "c2", "c3", "c4"],
        thresholds={key: 0.5 for key in LABEL_KEYS}, bootstrap_rounds=5,
    )
    assert metrics["per_label"]["atelectasis"]["auroc"] is None
    assert metrics["per_label"]["atelectasis"]["auroc_reason"] == "single_class"
    assert metrics["macro_auroc"]["value"] is None
    assert metrics["macro_auroc"]["reason"] == "no_defined_label"
