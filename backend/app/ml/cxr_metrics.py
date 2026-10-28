"""CXR 逐标签指标、温度缩放与阈值选择（纯 numpy 实现）。

设计说明（有意偏离 02 契约 G 的"scikit-learn 等库"）：
本机与 API 运行环境未安装 scikit-learn，而按 03 文件要求新增依赖必须先做
Linux/Py3.12 完整 hash 锁并跑 pip check，本轮无法在该环境验证。因此指标改为
纯 numpy 实现，语义与 sklearn 对齐（Mann-Whitney AUROC、average precision、
固定等宽 10 bins ECE），并**显式处理 mask**——绝不把 -1/空值矩阵直接塞进
多标签 ROC 函数。该偏离记录在 docs/cxr/execution/M06.md。

与 scikit-learn 的对照由 `backend/tests/cxr/test_cxr_metrics_sklearn.py` 覆盖
（该模块在无 sklearn 的环境整体 skip）。三处**已记录的口径差异**：

1. average precision 为**阶梯积分**（等同 `sklearn.metrics.average_precision_score`）：
   ``AP = Σ (R_n - R_{n-1}) · P_n``，阈值按相同评分分组。它**不是** PR 曲线梯形
   面积 ``auc(recall, precision)``；两者可能偏高也可能偏低，本项目不提供后者。
2. 单类/空 mask：sklearn 抛 ``ValueError``，本项目返回 ``(None, reason)``。
   契约要求"不填 0、不返回 NaN、不伪造宏平均"。
3. ECE 分箱为**左闭右开** ``[lower, upper)``（最后一箱含 1.0）；
   ``sklearn.calibration_curve(strategy='uniform')`` 把恰好落在边界上的值归入
   下箱。连续评分下两者一致，边界命中时归属不同，本项目按左闭右开固定。

所有函数在指标不可定义时返回 ``(None, reason)``，不填 0、不返回 NaN。
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np

ECE_BINS = 10
DEFAULT_BOOTSTRAP_ROUNDS = 1000


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _masked(truth: Sequence[float], score: Sequence[float], mask: Sequence[int]):
    y = _as_float_array(truth)
    s = _as_float_array(score)
    m = np.asarray(list(mask), dtype=bool)
    if y.shape != s.shape or y.shape != m.shape:
        raise ValueError("truth/score/mask 长度必须一致")
    if m.any() and not np.all(np.isfinite(s[m])):
        raise ValueError("评分含非有限值")
    return y[m], s[m]


def auroc(truth: Sequence[float], score: Sequence[float], mask: Sequence[int]) -> tuple[float | None, str | None]:
    """秩和法 AUROC，含并列评分处理；单类返回 None + reason。"""
    y, s = _masked(truth, score, mask)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return None, "single_class" if positives or negatives else "no_masked_samples"
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=float)
    index = 0
    while index < len(sorted_scores):
        end = index
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        ranks[order[index:end + 1]] = average_rank
        index = end + 1
    rank_sum = ranks[y == 1].sum()
    value = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(value), None


def average_precision(truth: Sequence[float], score: Sequence[float], mask: Sequence[int]) -> tuple[float | None, str | None]:
    """Average precision（阶梯积分），并列评分按同一阈值分组。

    定义：``AP = Σ_n (R_n - R_{n-1}) · P_n``，n 遍历按评分降序的唯一阈值，
    P_n/R_n 为该阈值下的精确率/召回率。等价于
    ``sklearn.metrics.average_precision_score``，**不是** PR 曲线梯形面积。
    无正例时不可定义，返回 ``(None, "no_positive")``。
    """
    y, s = _masked(truth, score, mask)
    positives = int((y == 1).sum())
    if positives == 0:
        return None, "no_positive"
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    total = len(y_sorted)
    previous_recall = 0.0
    area = 0.0
    tp = 0
    index = 0
    while index < total:
        end = index
        while end + 1 < total and s_sorted[end + 1] == s_sorted[index]:
            end += 1
        tp += int(y_sorted[index:end + 1].sum())
        predicted = end + 1
        precision = tp / predicted if predicted else 0.0
        recall = tp / positives
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = end + 1
    return float(area), None


def brier(truth: Sequence[float], probability: Sequence[float], mask: Sequence[int]) -> tuple[float | None, str | None]:
    y, p = _masked(truth, probability, mask)
    if len(y) == 0:
        return None, "no_masked_samples"
    return float(np.mean((p - y) ** 2)), None


def expected_calibration_error(
    truth: Sequence[float], probability: Sequence[float], mask: Sequence[int], bins: int = ECE_BINS
) -> tuple[float | None, dict | None, str | None]:
    """固定等宽 bins 的 ECE；返回 (ece, bin_table, reason)。

    箱区间为左闭右开 ``[lower, upper)``，最后一箱含 1.0。``bin_table`` 的
    confidence/accuracy/gap 按 6 位小数落库，ECE 本身用未舍入值累加。
    """
    y, p = _masked(truth, probability, mask)
    if len(y) == 0:
        return None, None, "no_masked_samples"
    edges = np.linspace(0.0, 1.0, bins + 1)
    table: list[dict] = []
    total = len(y)
    ece = 0.0
    for position in range(bins):
        lower, upper = edges[position], edges[position + 1]
        if position == bins - 1:
            selected = (p >= lower) & (p <= upper)
        else:
            selected = (p >= lower) & (p < upper)
        count = int(selected.sum())
        if count == 0:
            table.append({"bin": position, "lower": float(lower), "upper": float(upper), "count": 0,
                          "confidence": None, "accuracy": None, "gap": None})
            continue
        confidence = float(p[selected].mean())
        accuracy = float(y[selected].mean())
        gap = abs(accuracy - confidence)
        ece += gap * count / total
        table.append({"bin": position, "lower": float(lower), "upper": float(upper), "count": count,
                      "confidence": round(confidence, 6), "accuracy": round(accuracy, 6), "gap": round(gap, 6)})
    return float(ece), {"bins": bins, "table": table, "n": total}, None


def binary_at_threshold(
    truth: Sequence[float], score: Sequence[float], mask: Sequence[int], threshold: float
) -> dict:
    """冻结阈值下的混淆计数与派生指标；分母为 0 时各自为 None。"""
    y, s = _masked(truth, score, mask)
    if len(y) == 0:
        return {"threshold": threshold, "support": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                "sensitivity": None, "specificity": None, "ppv": None, "npv": None, "f1": None,
                "reason": "no_masked_samples"}
    predicted = (s >= threshold).astype(int)
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    ppv = tp / (tp + fp) if (tp + fp) else None
    npv = tn / (tn + fn) if (tn + fn) else None
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
    return {
        "threshold": float(threshold),
        "support": int(len(y)),
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": None if sensitivity is None else round(sensitivity, 6),
        "specificity": None if specificity is None else round(specificity, 6),
        "ppv": None if ppv is None else round(ppv, 6),
        "npv": None if npv is None else round(npv, 6),
        "f1": None if f1 is None else round(f1, 6),
        "reason": None,
    }


def youden_j(truth: Sequence[float], score: Sequence[float], mask: Sequence[int], threshold: float) -> float | None:
    stats = binary_at_threshold(truth, score, mask, threshold)
    if stats["sensitivity"] is None or stats["specificity"] is None:
        return None
    return float(stats["sensitivity"]) + float(stats["specificity"]) - 1.0


def select_threshold(
    truth: Sequence[float],
    score: Sequence[float],
    mask: Sequence[int],
    *,
    step: float = 0.05,
    lower: float = 0.05,
    upper: float = 0.95,
) -> dict:
    """Youden J 最大；并列取更高阈值；两类不足回退 0.5 + default_no_calibration。"""
    y, s = _masked(truth, score, mask)
    if int((y == 1).sum()) == 0 or int((y == 0).sum()) == 0:
        return {
            "threshold": 0.5,
            "source": "default_no_calibration",
            "reason": "single_class",
            "candidates": [],
        }
    grid: list[float] = []
    value = lower
    while value <= upper + 1e-9:
        grid.append(round(value, 4))
        value += step
    scored = [(threshold, youden_j(y, s, np.ones(len(y), dtype=int), threshold)) for threshold in grid]
    scored = [(threshold, j) for threshold, j in scored if j is not None]
    if not scored:
        return {"threshold": 0.5, "source": "default_no_calibration", "reason": "undefined_j", "candidates": []}
    best_j = max(item[1] for item in scored)
    best_threshold = max(item[0] for item in scored if abs(item[1] - best_j) <= 1e-12)
    return {
        "threshold": float(best_threshold),
        "source": "calibration_youden_j",
        "reason": None,
        "youden_j": round(float(best_j), 6),
        "candidates": [{"threshold": threshold, "youden_j": round(float(j), 6)} for threshold, j in scored],
    }


def sigmoid(values):
    array = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-array))


def logit(probabilities):
    array = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(array / (1.0 - array))


def _bce_sum(logits: np.ndarray, y: np.ndarray) -> float:
    # 数值稳定形式：max(x,0) - x*y + log(1+exp(-|x|))
    return float(np.sum(np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))))


def fit_temperature(
    truth: Sequence[float],
    probability: Sequence[float],
    mask: Sequence[int],
    *,
    min_samples: int = 50,
    min_per_class: int = 5,
    bounds: tuple[float, float] = (0.05, 10.0),
) -> dict:
    """逐标签温度缩放；数据不足时 T=1 + not_calibrated + 明确 reason。"""
    y, p = _masked(truth, probability, mask)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if len(y) < min_samples or positives < min_per_class or negatives < min_per_class:
        reasons = []
        if len(y) < min_samples:
            reasons.append(f"samples<{min_samples}")
        if positives < min_per_class:
            reasons.append(f"positives<{min_per_class}")
        if negatives < min_per_class:
            reasons.append(f"negatives<{min_per_class}")
        return {
            "temperature": 1.0,
            "status": "not_calibrated",
            "reason": ";".join(reasons),
            "samples": int(len(y)),
            "positives": positives,
            "negatives": negatives,
            "objective": None,
        }

    base_logits = logit(p)
    low, high = math.log(bounds[0]), math.log(bounds[1])

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        return _bce_sum(base_logits / temperature, y)

    # 粗网格避免非单峰时落入局部极小，再做黄金分割细化。
    grid = np.linspace(low, high, 61)
    values = [objective(float(item)) for item in grid]
    best_index = int(np.argmin(values))
    left = float(grid[max(0, best_index - 1)])
    right = float(grid[min(len(grid) - 1, best_index + 1)])
    phi = (math.sqrt(5) - 1) / 2
    for _ in range(60):
        mid_left = right - phi * (right - left)
        mid_right = left + phi * (right - left)
        if objective(mid_left) <= objective(mid_right):
            right = mid_right
        else:
            left = mid_left
    temperature = float(math.exp((left + right) / 2))
    temperature = min(max(temperature, bounds[0]), bounds[1])
    return {
        "temperature": round(temperature, 6),
        "status": "temperature_scaled",
        "reason": None,
        "samples": int(len(y)),
        "positives": positives,
        "negatives": negatives,
        "objective": round(objective(math.log(temperature)), 6),
        "objective_at_t1": round(objective(0.0), 6),
    }


def apply_temperature(probabilities: dict[str, float], temperatures: dict[str, float]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in probabilities.items():
        temperature = float(temperatures.get(key) or 1.0)
        output[key] = float(sigmoid(logit([value]) / temperature)[0])
    return output


def macro(per_label: dict[str, dict], field: str) -> dict:
    """只对有定义标签求宏平均，并报告有效标签数；缺类不填 0。"""
    values = [item[field] for item in per_label.values() if item.get(field) is not None]
    return {
        "value": round(float(sum(values) / len(values)), 6) if values else None,
        "valid_labels": len(values),
        "total_labels": len(per_label),
        "reason": None if values else "no_defined_label",
    }


def bootstrap_ci(
    groups: Iterable[str],
    truth: Sequence[float],
    score: Sequence[float],
    mask: Sequence[int],
    metric: Callable[[Sequence[float], Sequence[float], Sequence[int]], tuple[float | None, str | None]],
    *,
    rounds: int = DEFAULT_BOOTSTRAP_ROUNDS,
    seed: int = 42,
) -> dict:
    """以 patient/component 为重采样单位的 bootstrap 区间。"""
    group_list = list(groups)
    if len(group_list) != len(truth):
        raise ValueError("groups 与样本数不一致")
    unique_groups = sorted(set(group_list))
    if len(unique_groups) < 2:
        return {"lower": None, "upper": None, "rounds": 0, "reason": "insufficient_groups"}
    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(group_list):
        by_group.setdefault(group, []).append(index)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    y = _as_float_array(truth)
    s = _as_float_array(score)
    m = np.asarray(list(mask), dtype=bool)
    for _ in range(rounds):
        picked = rng.choice(len(unique_groups), size=len(unique_groups), replace=True)
        indices: list[int] = []
        for position in picked:
            indices.extend(by_group[unique_groups[int(position)]])
        value, _reason = metric(y[indices], s[indices], m[indices])
        if value is not None:
            samples.append(value)
    if len(samples) < max(10, rounds // 10):
        return {"lower": None, "upper": None, "rounds": len(samples), "reason": "too_few_defined_resamples"}
    array = np.asarray(samples)
    return {
        "lower": round(float(np.percentile(array, 2.5)), 6),
        "upper": round(float(np.percentile(array, 97.5)), 6),
        "rounds": len(samples),
        "reason": None,
    }


def evaluate_labels(
    truth_by_label: dict[str, list[float]],
    scores_by_label: dict[str, list[float]],
    mask_by_label: dict[str, list[int]],
    thresholds: dict[str, float],
) -> dict[str, dict]:
    """逐标签完整指标；只对 mask=1 样本计算。"""
    output: dict[str, dict] = {}
    for key, truth in truth_by_label.items():
        score = scores_by_label[key]
        mask = mask_by_label[key]
        auc, auc_reason = auroc(truth, score, mask)
        ap, ap_reason = average_precision(truth, score, mask)
        brier_value, brier_reason = brier(truth, score, mask)
        ece, ece_table, ece_reason = expected_calibration_error(truth, score, mask)
        threshold = float(thresholds.get(key, 0.5))
        output[key] = {
            "auroc": auc,
            "auroc_reason": auc_reason,
            "auprc": ap,
            "auprc_reason": ap_reason,
            "brier": None if brier_value is None else round(brier_value, 6),
            "brier_reason": brier_reason,
            "ece": None if ece is None else round(ece, 6),
            "ece_reason": ece_reason,
            "ece_detail": ece_table,
            "binary": binary_at_threshold(truth, score, mask, threshold),
            "masked_samples": int(np.asarray(list(mask), dtype=bool).sum()),
        }
    return output
