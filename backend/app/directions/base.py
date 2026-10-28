"""方向规格的数据结构与校验。

约束（与 02 技术契约一致）：
- 标签顺序由声明固定，禁止运行时按字典/中文排序重建；
- 任务语义（多标签 / 单标签 / 检测）必须显式声明，不能靠"有几个标签"推断；
- 每个方向独立声明数据适配器、分组级别、四集用途与报告限制；
- 注册表不得包含数据集路径、账号、密钥或权重。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TASK_KINDS = {
    "multilabel_classification",
    "single_label_classification",
    "object_detection",
}

OUTPUT_KINDS = {
    "per_label_scores",          # 多标签：逐标签独立评分
    "single_label_probabilities",  # 单标签：softmax 分布，和为 1
    "bounding_boxes",            # 检测：bbox + 置信度
}

DIRECTION_STATUSES = {"implemented", "data_blocked", "planned"}
DATA_STATUSES = {"downloaded", "partial", "blocked_access", "not_started"}


class DirectionError(ValueError):
    """方向声明不合法。"""


@dataclass(frozen=True)
class LabelSpec:
    index: int
    key: str
    display_name: str
    source_column: str | None = None


@dataclass(frozen=True)
class DirectionSpec:
    key: str
    display_name: str
    task_kind: str
    label_schema: str
    labels: tuple[LabelSpec, ...]
    group_level: str
    splits: tuple[str, ...]
    dataset_adapter: str
    analysis_output: str
    model_defaults: dict[str, Any] = field(default_factory=dict)
    report_limitations: tuple[str, ...] = ()
    status: str = "planned"
    data_status: str = "not_started"
    modality: str = ""
    notes: str = ""
    #: 官方来源与许可见 docs/datasets/，注册表不复制长文本。
    source_ref: str = ""

    def label_keys(self) -> tuple[str, ...]:
        return tuple(label.key for label in self.labels)

    def label_index(self) -> dict[str, int]:
        return {label.key: label.index for label in self.labels}

    def display_map(self) -> dict[str, str]:
        return {label.key: label.display_name for label in self.labels}

    def validate(self) -> None:
        if not self.key or not self.key.replace("_", "").isalnum():
            raise DirectionError(f"方向 key 必须是字母数字下划线: {self.key!r}")
        if self.task_kind not in TASK_KINDS:
            raise DirectionError(f"{self.key}: 未知 task_kind {self.task_kind}")
        if self.analysis_output not in OUTPUT_KINDS:
            raise DirectionError(f"{self.key}: 未知 analysis_output {self.analysis_output}")
        if self.status not in DIRECTION_STATUSES:
            raise DirectionError(f"{self.key}: 未知 status {self.status}")
        if self.data_status not in DATA_STATUSES:
            raise DirectionError(f"{self.key}: 未知 data_status {self.data_status}")
        if not self.labels:
            raise DirectionError(f"{self.key}: 至少需要一个标签")
        if [label.index for label in self.labels] != list(range(len(self.labels))):
            raise DirectionError(f"{self.key}: 标签 index 必须从 0 连续递增")
        keys = self.label_keys()
        if len(set(keys)) != len(keys):
            raise DirectionError(f"{self.key}: 标签 key 重复")
        if len(set(label.display_name for label in self.labels)) != len(self.labels):
            raise DirectionError(f"{self.key}: 展示名重复")
        if not self.splits:
            raise DirectionError(f"{self.key}: 必须声明数据切分")
        if len(set(self.splits)) != len(self.splits):
            raise DirectionError(f"{self.key}: 切分名重复")
        if self.task_kind == "single_label_classification" and self.analysis_output != "single_label_probabilities":
            raise DirectionError(f"{self.key}: 单标签任务的输出必须是 single_label_probabilities")
        if self.task_kind == "multilabel_classification" and self.analysis_output != "per_label_scores":
            raise DirectionError(f"{self.key}: 多标签任务的输出必须是 per_label_scores")
        if self.task_kind == "object_detection" and self.analysis_output != "bounding_boxes":
            raise DirectionError(f"{self.key}: 检测任务的输出必须是 bounding_boxes")

    def to_public(self) -> dict:
        return {
            "direction_key": self.key,
            "display_name": self.display_name,
            "modality": self.modality,
            "task_kind": self.task_kind,
            "label_schema": self.label_schema,
            "labels": [
                {
                    "index": label.index,
                    "key": label.key,
                    "display_name": label.display_name,
                    "source_column": label.source_column,
                }
                for label in self.labels
            ],
            "group_level": self.group_level,
            "splits": list(self.splits),
            "dataset_adapter": self.dataset_adapter,
            "analysis_output": self.analysis_output,
            "model_defaults": dict(self.model_defaults),
            "report_limitations": list(self.report_limitations),
            "status": self.status,
            "data_status": self.data_status,
            "source_ref": self.source_ref,
            "notes": self.notes,
        }
