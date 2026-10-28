"""ISIC 七类单标签指标：混淆矩阵、逐类 P/R/F1、macro-F1、准确率与 top-1 ECE。

设计原则与 CXR 指标一致：**指标不可定义时返回 `None` 并给出原因，不填 0**。
唯一例外是「该类有真值样本、但一次都没被预测到」——此时 recall=0、
precision 无定义、F1 按定义为 0，这是真实值而不是补零，原因仍会记录。

纯 numpy 实现，不引入 scikit-learn 运行时依赖（与 CXR 一致）。
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.schemas.isic import ISIC_CLASS_KEYS, ISIC_METRICS_SCHEMA


def _metric(value, reason=None) -> dict:
    return {"value": value, "reason": reason}


def confusion_matrix(truth: Sequence[int], pred: Sequence[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, predicted in zip(truth, pred):
        if not (0 <= int(actual) < num_classes) or not (0 <= int(predicted) < num_classes):
            raise ValueError("class index out of range")
        matrix[int(actual), int(predicted)] += 1
    return matrix


def _per_class(matrix: np.ndarray, labels: Sequence[str]) -> list[dict]:
    rows: list[dict] = []
    for index, key in enumerate(labels):
        tp = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted = int(matrix[:, index].sum())

        if predicted > 0:
            precision, precision_reason = tp / predicted, None
        else:
            precision, precision_reason = None, "no_predicted_positive"

        if support > 0:
            recall, recall_reason = tp / support, None
        else:
            recall, recall_reason = None, "no_support_in_split"

        if support == 0:
            f1, f1_reason = None, "no_support_in_split"
        elif precision is None:
            # 有真值但从未被预测：recall 必然为 0，F1 按定义为 0（不是补零）。
            f1, f1_reason = 0.0, "no_predicted_positive"
        elif precision + recall == 0:
            f1, f1_reason = 0.0, "zero_precision_and_recall"
        else:
            f1, f1_reason = 2 * precision * recall / (precision + recall), None

        rows.append(
            {
                "label_key": key,
                "index": index,
                "support": support,
                "predicted": predicted,
                "true_positive": tp,
                "precision": _metric(None if precision is None else round(precision, 6), precision_reason),
                "recall": _metric(None if recall is None else round(recall, 6), recall_reason),
                "f1": _metric(None if f1 is None else round(f1, 6), f1_reason),
            }
        )
    return rows


def _top1_ece(truth: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict:
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == truth).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(truth)
    ece = 0.0
    table: list[dict] = []
    for position in range(bins):
        lower, upper = float(edges[position]), float(edges[position + 1])
        # 左闭右开，最后一箱包含 1.0；与 CXR ECE 的箱约定保持一致。
        if position == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        if count == 0:
            table.append({"bin": position, "lower": round(lower, 6), "upper": round(upper, 6),
                          "count": 0, "confidence": None, "accuracy": None})
            continue
        bin_confidence = float(confidence[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        ece += count / total * abs(bin_accuracy - bin_confidence)
        table.append(
            {
                "bin": position,
                "lower": round(lower, 6),
                "upper": round(upper, 6),
                "count": count,
                "confidence": round(bin_confidence, 6),
                "accuracy": round(bin_accuracy, 6),
            }
        )
    return {"value": round(float(ece), 6), "reason": None, "bins": bins, "table": table}


def evaluate(
    truth: Sequence[int],
    pred: Sequence[int],
    *,
    probabilities: np.ndarray | None = None,
    labels: Sequence[str] = ISIC_CLASS_KEYS,
) -> dict:
    """返回完整指标；不可定义项带 reason。"""
    truth_array = np.asarray(list(truth), dtype=np.int64)
    pred_array = np.asarray(list(pred), dtype=np.int64)
    if truth_array.shape != pred_array.shape:
        raise ValueError("truth and pred length differ")
    num_classes = len(labels)
    report: dict = {
        "schema": ISIC_METRICS_SCHEMA,
        "labels": list(labels),
        "support": int(truth_array.size),
        "note": "单标签七类；总体准确率会被多数类主导，必须同时看逐类指标与 macro-F1。",
    }
    if truth_array.size == 0:
        report.update(
            confusion_matrix=[[0] * num_classes for _ in range(num_classes)],
            per_class=[],
            accuracy=_metric(None, "no_samples"),
            balanced_accuracy=_metric(None, "no_samples"),
            macro_precision=_metric(None, "no_samples"),
            macro_recall=_metric(None, "no_samples"),
            macro_f1=_metric(None, "no_samples"),
            ece=_metric(None, "no_samples"),
        )
        return report

    matrix = confusion_matrix(truth_array, pred_array, num_classes)
    per_class = _per_class(matrix, labels)
    report["confusion_matrix"] = matrix.tolist()
    report["per_class"] = per_class
    report["accuracy"] = _metric(round(float((truth_array == pred_array).mean()), 6), None)

    with_support = [row for row in per_class if row["support"] > 0]
    with_prediction = [row for row in per_class if row["predicted"] > 0]

    recalls = [row["recall"]["value"] for row in with_support]
    report["balanced_accuracy"] = _metric(
        round(float(np.mean(recalls)), 6) if recalls else None,
        None if recalls else "no_support_in_split",
    )
    precisions = [row["precision"]["value"] for row in with_prediction]
    report["macro_precision"] = _metric(
        round(float(np.mean(precisions)), 6) if precisions else None,
        None if precisions else "no_predicted_positive",
    )
    report["macro_recall"] = _metric(
        round(float(np.mean(recalls)), 6) if recalls else None,
        None if recalls else "no_support_in_split",
    )
    f1_values = [row["f1"]["value"] for row in with_support if row["f1"]["value"] is not None]
    report["macro_f1"] = _metric(
        round(float(np.mean(f1_values)), 6) if f1_values else None,
        None if f1_values else "no_support_in_split",
    )
    report["macro_f1_scope"] = "仅对 split 内真实出现过的类别取平均"

    if probabilities is None:
        report["ece"] = _metric(None, "probabilities_unavailable")
    else:
        array = np.asarray(probabilities, dtype=np.float64)
        if array.shape != (truth_array.size, num_classes):
            raise ValueError("probabilities shape mismatch")
        report["ece"] = _top1_ece(truth_array, array)
    return report
