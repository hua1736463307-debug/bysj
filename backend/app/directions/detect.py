"""检测方向：BCCD 血细胞检测（历史研究模块）。

语义与两个分类方向不同：输出 bbox + 置信度，不产生"疾病概率列表"。
保留该方向是为了让注册表覆盖三类任务语义，并防止把检测结果包装成分类输出。
"""
from __future__ import annotations

from app.directions.base import DirectionSpec, LabelSpec
from app.directions.registry import register

DETECT_DIRECTION_KEY = "bccd_detect"
DETECT_LABEL_SCHEMA = "bccd-detect-v1"
DETECT_TASK = "detect"
DETECT_SPLITS = ("train", "val", "test")

DETECT_LABELS: tuple[LabelSpec, ...] = (
    LabelSpec(0, "rbc", "红细胞", None),
    LabelSpec(1, "wbc", "白细胞", None),
    LabelSpec(2, "platelets", "血小板", None),
)

DETECT_DIRECTION = DirectionSpec(
    key=DETECT_DIRECTION_KEY,
    display_name="血细胞检测（BCCD，历史研究模块）",
    modality="Microscopy",
    task_kind="object_detection",
    label_schema=DETECT_LABEL_SCHEMA,
    labels=DETECT_LABELS,
    group_level="image_cluster",
    splits=DETECT_SPLITS,
    dataset_adapter="app.domains.datasets.import_service",
    analysis_output="bounding_boxes",
    model_defaults={
        "arch": "yolo",
        "image_size": 640,
        "preprocess_schema": "yolo-default",
        "activation": "sigmoid_per_anchor",
        "loss": "yolo_loss",
        "metrics_schema": "detect-metrics-v1",
    },
    report_limitations=(
        "检测输出为 bbox 与置信度，不是疾病概率，不参与五标签/七类评分。",
        "BCCD 为教学用血涂片数据，与临床诊断无关。",
    ),
    status="implemented",
    data_status="downloaded",
    source_ref="docs/experiments/2026-10-22/DATA-QUALITY-REVIEW.md",
    notes="旧 BCCD 链路保留为历史研究模块；旧跨集近重复风险仍记为 BLOCKED，不因改题而消失。",
)

register(DETECT_DIRECTION)
