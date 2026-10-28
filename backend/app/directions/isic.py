"""皮肤镜方向：ISIC 2018 Task 3（7 类单标签分类）。

标签顺序与官方 GroundTruth CSV 的列顺序一致（`image,MEL,NV,BCC,AKIEC,BKL,DF,VASC`），
不按中文或字典序重排。本方向是**单标签互斥**语义：一张图恰好一个诊断类别，
输出 softmax 分布（和为 1），与胸片的多标签独立 sigmoid 不是同一种任务。
"""
from __future__ import annotations

from app.directions.base import DirectionSpec, LabelSpec
from app.directions.registry import register

ISIC_DIRECTION_KEY = "isic2018"
ISIC_LABEL_SCHEMA = "isic2018-task3-v1"
ISIC_TASK = "isic_multiclass"
ISIC_SPLITS = ("train", "val", "calibration", "test")

#: 与官方 CSV 列顺序一致；index 即 softmax 输出维度。
ISIC_LABELS: tuple[LabelSpec, ...] = (
    LabelSpec(0, "mel", "黑色素瘤", "MEL"),
    LabelSpec(1, "nv", "黑色素细胞痣", "NV"),
    LabelSpec(2, "bcc", "基底细胞癌", "BCC"),
    LabelSpec(3, "akiec", "光化性角化病/鲍温病", "AKIEC"),
    LabelSpec(4, "bkl", "良性角化病", "BKL"),
    LabelSpec(5, "df", "皮肤纤维瘤", "DF"),
    LabelSpec(6, "vasc", "血管性病变", "VASC"),
)

ISIC_DIRECTION = DirectionSpec(
    key=ISIC_DIRECTION_KEY,
    display_name="皮肤镜七类皮损辅助分类",
    modality="Dermoscopy",
    task_kind="single_label_classification",
    label_schema=ISIC_LABEL_SCHEMA,
    labels=ISIC_LABELS,
    group_level="lesion",
    splits=ISIC_SPLITS,
    dataset_adapter="app.domains.isic.import_service",
    analysis_output="single_label_probabilities",
    model_defaults={
        "arch": "densenet121",
        "image_size": 224,
        "preprocess_schema": "isic-preprocess-v1",
        "activation": "softmax",
        "loss": "cross_entropy",
        "metrics_schema": "isic-metrics-v1",
    },
    report_limitations=(
        "七类输出为模型分类概率，不是病理确诊；确诊需组织病理。",
        "数据为皮肤镜图像，不适用于临床照片、全身照或其他成像方式。",
        "类别不平衡显著（NV 占多数），总体准确率会被多数类主导，须同时看逐类指标。",
        "官方 validation/test 与训练集是否同一患者未公开保证，不能称为患者级外部验证。",
        "系统未取得临床有效性证据，仅用于研究辅助分析。",
    ),
    status="implemented",
    data_status="partial",
    source_ref="docs/datasets/DERMOSCOPY_DATASETS.md",
    notes=(
        "四集 train/val/calibration/test 由官方 training 集按 lesion 分组切分；"
        "官方 validation(193) 与 test(1512) 作为外部留出集单独报告，不参与拟合与阈值选择。"
        "官方仅对 training 集提供 lesion 分组表，因此患者级隔离只在 training 派生四集内可证明。"
    ),
)

register(ISIC_DIRECTION)
