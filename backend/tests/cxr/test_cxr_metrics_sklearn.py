"""与 scikit-learn 的独立对照测试（`cxr_metrics` 为纯 numpy 实现）。

本文件只在安装了 scikit-learn 的环境中运行：普通 CI（无 sklearn）会整模块 skip，
避免给 API/worker 镜像引入未锁定的依赖。取证时使用隔离 venv：

    python -m venv <env> && <env>/Scripts/pip install "scikit-learn==1.5.2" pytest
    PYTHONPATH=backend <env>/Scripts/python -m pytest backend/tests/cxr/test_cxr_metrics_sklearn.py -q

对照范围与**已知差异**：
- AUROC：与 `roc_auc_score` 数值一致（含并列评分）。
- AUPRC：与 `average_precision_score` 一致；两者都是"阶梯积分"，不是梯形积分
  （`auc(recall, precision)` 会偏高，本项目不使用）。
- Brier：与手算 `mean((p-y)^2)` 一致。
- ECE：sklearn 无内置 ECE，只对照 `calibration_curve` 的等宽分箱与箱内均值。
- 单类：sklearn 抛 `ValueError`，本项目返回 `None + reason`（契约要求不填 0、不返回 NaN）。
- 掩码：本项目先按 mask 过滤再计算；对照时把过滤后的数组交给 sklearn，验证二者等价。
"""
from __future__ import annotations

import numpy as np
import pytest

sklearn_metrics = pytest.importorskip("sklearn.metrics")
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from app.ml.cxr_metrics import (  # noqa: E402
    auroc,
    average_precision,
    binary_at_threshold,
    brier,
    expected_calibration_error,
    select_threshold,
)

CASES = [
    (0, 200, False),
    (1, 200, False),
    (2, 137, False),
    (3, 512, False),
    (4, 64, True),   # 大量并列评分
    (5, 31, True),
]


def _dataset(seed: int, size: int, ties: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    truth = (rng.random(size) < 0.3).astype(float)
    if truth.sum() == 0:
        truth[0] = 1.0
    if truth.sum() == size:
        truth[0] = 0.0
    scores = np.clip(truth * 0.35 + rng.normal(0.35, 0.28, size), 0.0, 1.0)
    if ties:
        scores = np.round(scores, 1)
    return truth, scores


@pytest.mark.parametrize("seed,size,ties", CASES)
def test_auroc_matches_sklearn(seed, size, ties):
    truth, scores = _dataset(seed, size, ties)
    ours, reason = auroc(truth, scores, [1] * size)
    assert reason is None
    expected = roc_auc_score(truth, scores)
    assert ours == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("seed,size,ties", CASES)
def test_average_precision_matches_sklearn(seed, size, ties):
    truth, scores = _dataset(seed, size, ties)
    ours, reason = average_precision(truth, scores, [1] * size)
    assert reason is None
    expected = average_precision_score(truth, scores)
    assert ours == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_average_precision_is_step_integral_not_trapezoidal_area():
    """记录定义：AP 为阶梯积分（sklearn `average_precision_score` 语义）。

    与"PR 曲线梯形面积"不是同一个量；二者可能偏高也可能偏低（本用例梯形面积更低），
    因此本项目不提供 `auc(recall, precision)` 形式的 PR-AUC，避免与论文口径混淆。
    """
    truth = np.array([0, 0, 1, 1, 0, 1], dtype=float)
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.9])
    ours, _ = average_precision(truth, scores, [1] * 6)
    expected = average_precision_score(truth, scores)
    assert ours == pytest.approx(expected, rel=1e-12)

    precision, recall, _ = sklearn_metrics.precision_recall_curve(truth, scores)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    trapezoid = float(trapz(precision[::-1], recall[::-1]))
    assert trapezoid != pytest.approx(ours, rel=1e-6), "AP 与梯形面积不应被当作同一指标"


@pytest.mark.parametrize("seed,size,ties", CASES)
def test_brier_matches_manual_mean_squared_error(seed, size, ties):
    truth, scores = _dataset(seed, size, ties)
    ours, reason = brier(truth, scores, [1] * size)
    assert reason is None
    assert ours == pytest.approx(float(np.mean((scores - truth) ** 2)), rel=1e-12, abs=1e-12)


def test_ece_bins_match_sklearn_for_continuous_scores():
    """连续评分（无恰好落在箱边界的值）时，分箱与 `calibration_curve` 一致。"""
    rng = np.random.default_rng(21)
    truth = (rng.random(400) < 0.35).astype(float)
    scores = np.round(np.clip(truth * 0.3 + rng.normal(0.4, 0.25, 400), 0, 1), 4)
    # 前置条件：没有恰好等于内部边界 0.1…0.9 的评分（0 与 1 两端两种约定一致）。
    # 舍入可能命中边界，这里把命中值推开 5e-5，保持"连续评分"语义。
    interior_edges = np.arange(1, 10) / 10.0
    on_edge = np.isclose(scores[:, None], interior_edges[None, :], atol=1e-12).any(axis=1)
    scores[on_edge] = np.clip(scores[on_edge] + 5e-5, 0.0, 1.0)
    assert not np.isclose(scores[:, None], interior_edges[None, :], atol=1e-12).any()

    ece, detail, reason = expected_calibration_error(truth, scores, [1] * 400)
    assert reason is None
    assert 0.0 <= ece <= 1.0
    fraction_positive, mean_predicted = calibration_curve(truth, scores, n_bins=10, strategy="uniform")
    ours_non_empty = [item for item in detail["table"] if item["count"] > 0]
    assert len(ours_non_empty) == len(mean_predicted)
    assert sorted(item["confidence"] for item in ours_non_empty) == pytest.approx(
        sorted(float(value) for value in mean_predicted), rel=1e-6, abs=1e-6
    )
    assert sorted(item["accuracy"] for item in ours_non_empty) == pytest.approx(
        sorted(float(value) for value in fraction_positive), rel=1e-6, abs=1e-6
    )
    assert sum(item["count"] for item in detail["table"]) == 400


def test_ece_boundary_convention_differs_from_sklearn():
    """记录差异：本项目箱区间为左闭右开 [lower, upper)，sklearn 把边界值归入下箱。

    评分恰好等于 0.1 时：
    - 本项目 → bin 1（区间 [0.1, 0.2)）
    - `calibration_curve(strategy='uniform')` → bin 0（`searchsorted(..., side='left')`）
    两者都是"等宽 10 bins"，但边界归属不同；本项目采用左闭右开并在文档中固定。
    """
    truth = np.array([1.0, 0.0, 1.0, 0.0])
    scores = np.array([0.1, 0.1, 0.5, 0.5])
    _ece, detail, _reason = expected_calibration_error(truth, scores, [1] * 4)
    counts = {item["bin"]: item["count"] for item in detail["table"]}
    assert counts[1] == 2, "0.1 应落在 [0.1,0.2)"
    assert counts[0] == 0

    fraction_positive, mean_predicted = calibration_curve(truth, scores, n_bins=10, strategy="uniform")
    # sklearn 侧只有两个非空箱，且 0.1 与 0.5 分处不同箱
    assert len(mean_predicted) == 2
    assert min(mean_predicted) == pytest.approx(0.1)


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_binary_confusion_matches_sklearn(threshold):
    truth, scores = _dataset(7, 300, False)
    stats = binary_at_threshold(truth, scores, [1] * 300, threshold)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth.astype(int), predicted, labels=[0, 1]).ravel()
    assert (stats["tp"], stats["fp"], stats["tn"], stats["fn"]) == (int(tp), int(fp), int(tn), int(fn))
    precision, recall, f1, _support = precision_recall_fscore_support(
        truth.astype(int), predicted, average="binary", zero_division=0
    )
    # 对外指标按 6 位小数落库，故按 1e-6 对照。
    assert stats["ppv"] == pytest.approx(float(precision), rel=1e-6, abs=1e-6)
    assert stats["sensitivity"] == pytest.approx(float(recall), rel=1e-6, abs=1e-6)
    assert stats["f1"] == pytest.approx(float(f1), rel=1e-6, abs=1e-6)


def test_masked_computation_equals_prefiltered_sklearn():
    truth, scores = _dataset(11, 240, False)
    mask = np.ones(240, dtype=int)
    mask[::3] = 0  # 模拟 uncertain/unmentioned
    ours, reason = auroc(truth, scores, mask)
    assert reason is None
    selected = mask.astype(bool)
    expected = roc_auc_score(truth[selected], scores[selected])
    assert ours == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_single_class_divergence_is_documented():
    """单类：sklearn 抛错，本项目返回 None + reason（不填 0、不返回 NaN）。"""
    truth = np.array([1.0, 1.0, 1.0])
    scores = np.array([0.2, 0.5, 0.8])
    with pytest.raises(ValueError):
        roc_auc_score(truth, scores)
    ours, reason = auroc(truth, scores, [1, 1, 1])
    assert ours is None
    assert reason == "single_class"

    ap_ours, ap_reason = average_precision(truth, scores, [1, 1, 1])
    assert ap_ours == pytest.approx(1.0)  # 全正例时 AP 有定义
    assert ap_reason is None

    selection = select_threshold(truth, scores, [1, 1, 1])
    assert selection["threshold"] == 0.5
    assert selection["source"] == "default_no_calibration"
    assert selection["reason"] == "single_class"


def test_all_ties_match_sklearn():
    truth = np.array([0, 1, 0, 1, 1, 0], dtype=float)
    scores = np.full(6, 0.42)
    ours, _ = auroc(truth, scores, [1] * 6)
    assert ours == pytest.approx(roc_auc_score(truth, scores), rel=1e-12)
    assert ours == pytest.approx(0.5)
