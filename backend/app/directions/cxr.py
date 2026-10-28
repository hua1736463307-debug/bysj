"""胸部 X 光方向（`cxr_multilabel` / `cxr-five-v1`）。

标签与四态定义仍以 `app/schemas/cxr.py` 为唯一来源；本模块只做**声明**，
不复制标签常量，避免两处漂移。
"""
from __future__ import annotations

from app.directions.base import DirectionSpec, LabelSpec
from app.directions.registry import register
from app.schemas.cxr import CXR_LABEL_SCHEMA, CXR_LABELS, CXR_SPLITS, CXR_TASK

CXR_DIRECTION_KEY = "cxr"

CXR_DIRECTION = DirectionSpec(
    key=CXR_DIRECTION_KEY,
    display_name="胸部X光五标签辅助分析",
    modality="X-ray",
    task_kind="multilabel_classification",
    label_schema=CXR_LABEL_SCHEMA,
    labels=tuple(
        LabelSpec(index=label.index, key=label.key, display_name=label.display_name, source_column=label.source_column)
        for label in CXR_LABELS
    ),
    group_level="patient_component",
    splits=tuple(CXR_SPLITS),
    dataset_adapter="app.domains.cxr.import_service",
    analysis_output="per_label_scores",
    model_defaults={
        "arch": "densenet121",
        "image_size": 224,
        "preprocess_schema": "cxr-preprocess-v1",
        "activation": "sigmoid",
        "loss": "masked_bce_with_logits",
        "metrics_schema": "cxr-metrics-v1",
    },
    report_limitations=(
        "五项结果为模型观察评分（含影像征象），不等同于疾病确诊。",
        "未超过阈值不代表排除该征象，也不代表排除其他疾病。",
        "系统未做外部机构验证，未取得临床有效性证据。",
        "模型关注区域只说明模型依据，不是病灶定位真值。",
    ),
    status="implemented",
    data_status="blocked_access",
    source_ref="docs/datasets/CXR_DATASETS.md",
    notes=(
        "四态 positive/negative/uncertain/unmentioned 只适用于 CheXpert 系标签表；"
        "换用其他胸片数据源时必须按该源官方文档另建映射，不得复用本四态语义。"
    ),
)

register(CXR_DIRECTION)
