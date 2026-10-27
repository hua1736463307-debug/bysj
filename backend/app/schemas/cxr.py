"""CXR 五标签技术契约（`cxr_multilabel` / `cxr-five-v1`）。

本模块是标签语义、四态、mask、阈值提示与请求模型的**唯一来源**。标签顺序
按固定常量声明，任何地方都不得按字典遍历、中文排序或列名排序重建顺序。

原始四态：
    positive=1 / negative=0 / uncertain=-1 / unmentioned=空

监督只使用明确 positive/negative（mask=1）；uncertain/unmentioned 置 mask=0，
但**保留原始状态**，不 fillna(0)、不使用 bool(-1)、不把空白当阴性。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CXR_TASK = "cxr_multilabel"
CXR_LABEL_SCHEMA = "cxr-five-v1"
CXR_MANIFEST_SCHEMA = "cxr-manifest-v1"
CXR_PREPROCESS_SCHEMA = "cxr-preprocess-v1"
CXR_METRICS_SCHEMA = "cxr-metrics-v1"
CXR_GROUP_LEVEL = "patient_component"
CXR_SPLITS: tuple[str, ...] = ("train", "val", "calibration", "test")
#: 首期只接正位胸片。CheXpert 的 Frontal/Lateral 列只区分正/侧位，
#: AP/PA 若文件名未标注则记为 unknown，不猜测。
CXR_VIEWS = ("Frontal",)
CXR_PROJECTIONS = ("AP", "PA", "unknown")

RAW_POSITIVE = "positive"
RAW_NEGATIVE = "negative"
RAW_UNCERTAIN = "uncertain"
RAW_UNMENTIONED = "unmentioned"
RAW_STATES = (RAW_POSITIVE, RAW_NEGATIVE, RAW_UNCERTAIN, RAW_UNMENTIONED)
MASKED_RAW_STATES = (RAW_POSITIVE, RAW_NEGATIVE)

FLAG_ABOVE = "above_threshold"
FLAG_BELOW = "below_threshold"
FLAG_BORDERLINE = "borderline"
FLAG_NOT_AVAILABLE = "not_available"

#: 工程提示：|score-threshold| <= 0.05 记为 borderline。这是产品提示，
#: 不是经过验证的医学不确定性估计，不并入"置信度低"的单一字段。
BORDERLINE_MARGIN = 0.05
DEFAULT_THRESHOLD = 0.5

CALIBRATION_NOT_CALIBRATED = "not_calibrated"
CALIBRATION_OK = "temperature_scaled"


@dataclass(frozen=True)
class CxrLabel:
    index: int
    key: str
    source_column: str
    display_name: str


#: 顺序即契约；禁止在运行时重排。
CXR_LABELS: tuple[CxrLabel, ...] = (
    CxrLabel(0, "atelectasis", "Atelectasis", "肺不张"),
    CxrLabel(1, "cardiomegaly", "Cardiomegaly", "心影增大"),
    CxrLabel(2, "consolidation", "Consolidation", "肺实变"),
    CxrLabel(3, "edema", "Edema", "肺水肿"),
    CxrLabel(4, "pleural_effusion", "Pleural Effusion", "胸腔积液"),
)
LABEL_KEYS: tuple[str, ...] = tuple(label.key for label in CXR_LABELS)
LABEL_INDEX: dict[str, int] = {label.key: label.index for label in CXR_LABELS}
LABEL_DISPLAY: dict[str, str] = {label.key: label.display_name for label in CXR_LABELS}
SOURCE_COLUMNS: dict[str, str] = {label.key: label.source_column for label in CXR_LABELS}
LABEL_COUNT = len(CXR_LABELS)


class CxrLabelError(ValueError):
    """原始标签值/列结构不符合契约。"""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def parse_raw_label(value: Any) -> str:
    """把 CheXpert 原始取值映射为四态，未知值明确拒绝。"""
    if value is None:
        return RAW_UNMENTIONED
    if isinstance(value, bool):  # bool 是 int 子类，必须先拦
        raise CxrLabelError("布尔值不是合法标签取值", {"value": value})
    if isinstance(value, (int, float)):
        number = float(value)
        if number == 1.0:
            return RAW_POSITIVE
        if number == 0.0:
            return RAW_NEGATIVE
        if number == -1.0:
            return RAW_UNCERTAIN
        raise CxrLabelError("标签取值只能是 1/0/-1/空", {"value": value})
    text = str(value).strip()
    if text == "":
        return RAW_UNMENTIONED
    if text in {"1", "1.0"}:
        return RAW_POSITIVE
    if text in {"0", "0.0"}:
        return RAW_NEGATIVE
    if text in {"-1", "-1.0"}:
        return RAW_UNCERTAIN
    lowered = text.lower()
    if lowered in RAW_STATES:
        return lowered
    raise CxrLabelError("标签取值只能是 1/0/-1/空", {"value": value})


def normalize_raw_labels(row: dict) -> dict[str, str]:
    """按固定顺序抽取五列；缺列直接拒绝，不猜列、不补默认值。"""
    missing = [column for column in SOURCE_COLUMNS.values() if column not in row]
    if missing:
        raise CxrLabelError("缺少标签列", {"missing_columns": missing})
    return {
        label.key: parse_raw_label(row[label.source_column]) for label in CXR_LABELS
    }


def mask_vector(raw_labels: dict[str, str]) -> list[int]:
    """mask=1 表示该标签参与监督；uncertain/unmentioned 为 0。"""
    return [1 if raw_labels.get(key) in MASKED_RAW_STATES else 0 for key in LABEL_KEYS]


def target_vector(raw_labels: dict[str, str]) -> list[int | None]:
    """目标值；mask=0 的位置为 None，调用方不得把它当 0。"""
    values: list[int | None] = []
    for key in LABEL_KEYS:
        state = raw_labels.get(key)
        if state == RAW_POSITIVE:
            values.append(1)
        elif state == RAW_NEGATIVE:
            values.append(0)
        else:
            values.append(None)
    return values


def raw_state_counts(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, int]]:
    """逐标签统计四态数量，供数据卡与冻结审计使用。

    只接受四态字符串；出现其他取值说明存储层被污染，直接拒绝而不是静默计数。
    """
    counts = {key: {state: 0 for state in RAW_STATES} for key in LABEL_KEYS}
    for raw_labels in rows:
        for key in LABEL_KEYS:
            state = raw_labels.get(key, RAW_UNMENTIONED)
            if state not in counts[key]:
                raise CxrLabelError(
                    "raw_labels 只允许四态字符串",
                    {"label_key": key, "value": state, "allowed": list(RAW_STATES)},
                )
            counts[key][state] += 1
    return counts


def classify_score(score: float | None, threshold: float | None) -> str:
    """生成 flag。永远不返回 confirmed_disease / healthy 之类结论。"""
    if score is None or threshold is None:
        return FLAG_NOT_AVAILABLE
    if abs(score - threshold) <= BORDERLINE_MARGIN:
        return FLAG_BORDERLINE
    return FLAG_ABOVE if score > threshold else FLAG_BELOW


def build_label_results(
    raw_scores: dict[str, float] | None,
    calibrated_scores: dict[str, float] | None,
    thresholds: dict[str, float] | None,
    *,
    calibration_status: str,
    threshold_source: str,
) -> list[dict]:
    """五路完整输出，不是只返回 top1；分数不要求和为 1。"""
    results: list[dict] = []
    for label in CXR_LABELS:
        raw = None if raw_scores is None else raw_scores.get(label.key)
        calibrated = None if calibrated_scores is None else calibrated_scores.get(label.key)
        threshold = None if thresholds is None else thresholds.get(label.key)
        effective = calibrated if calibrated is not None else raw
        results.append(
            {
                "label_key": label.key,
                "display_name": label.display_name,
                "index": label.index,
                "raw_score": None if raw is None else round(float(raw), 6),
                "calibrated_score": None if calibrated is None else round(float(calibrated), 6),
                "threshold": None if threshold is None else round(float(threshold), 6),
                "threshold_source": threshold_source,
                "flag": classify_score(effective, threshold),
                "calibration_status": calibration_status,
            }
        )
    return results


def schema_payload() -> dict:
    return {
        "task": CXR_TASK,
        "label_schema": CXR_LABEL_SCHEMA,
        "labels": [
            {
                "index": label.index,
                "key": label.key,
                "display_name": label.display_name,
                "source_column": label.source_column,
            }
            for label in CXR_LABELS
        ],
        "raw_states": list(RAW_STATES),
        "masked_raw_states": list(MASKED_RAW_STATES),
        "splits": list(CXR_SPLITS),
        "group_level": CXR_GROUP_LEVEL,
        "views": list(CXR_VIEWS),
        "supported_formats": ["JPG", "PNG"],
        "borderline_margin": BORDERLINE_MARGIN,
        "default_threshold": DEFAULT_THRESHOLD,
        "preprocess_schema": CXR_PREPROCESS_SCHEMA,
        "manifest_schema": CXR_MANIFEST_SCHEMA,
        "metrics_schema": CXR_METRICS_SCHEMA,
    }


class CxrTrainingParams(BaseModel):
    """CXR 专用参数；不沿用 detect 的 imgsz/mosaic 语义。"""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=20, ge=1, le=20)
    batch: int = Field(default=8, ge=1, le=64)
    image_size: int = Field(default=224, ge=128, le=512)
    lr: float = Field(default=1e-4, gt=0, le=1e-1)
    weight_decay: float = Field(default=1e-4, ge=0, le=1e-1)
    patience: int = Field(default=5, ge=1, le=20)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    class_weight: Literal["none", "pos_weight"] = "none"
    num_workers: int = Field(default=0, ge=0, le=8)
    amp: bool = False
    device: str = "0"


class CxrImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    source_csv: str = Field(default="train.csv", min_length=1, max_length=255)
    name: str = Field(default="", max_length=255)
    license_note: str | None = Field(default=None, max_length=2000)
    max_patients: int = Field(default=10000, ge=1, le=10000)
    dry_run: bool = False


class CxrFreezeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split_seed: int = Field(default=42, ge=0, le=2**31 - 1)
    ratios: list[float] = Field(default_factory=lambda: [0.7, 0.1, 0.1, 0.1])

    @field_validator("ratios")
    @classmethod
    def _check_ratios(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("四集比例必须正好四项")
        if any(item <= 0 for item in value):
            raise ValueError("四集比例必须全为正数")
        total = sum(value)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("四集比例之和必须为 1")
        return value


class CxrTrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version_id: str
    base_model_id: str | None = None
    params: CxrTrainingParams = Field(default_factory=CxrTrainingParams)


class CxrEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    dataset_version_id: str
    split: Literal["val", "calibration", "test"] = "val"
    action: Literal["evaluate", "calibrate"] = "evaluate"
    purpose: str | None = Field(default=None, max_length=1000)


class CxrAnalysisBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    asset_ids: list[int] = Field(default_factory=list, max_length=100)


class CxrExplanationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_key: str
    method: Literal["gradcam"] = "gradcam"

    @field_validator("label_key")
    @classmethod
    def _known_label(cls, value: str) -> str:
        if value not in LABEL_INDEX:
            raise ValueError(f"未知标签: {value}")
        return value


class CxrLabelOpinion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_key: str
    opinion: Literal["supported", "not_supported", "indeterminate"]
    reason: str | None = Field(default=None, max_length=2000)

    @field_validator("label_key")
    @classmethod
    def _known_label(cls, value: str) -> str:
        if value not in LABEL_INDEX:
            raise ValueError(f"未知标签: {value}")
        return value


class CxrReviewCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    reason: str | None = Field(default=None, max_length=2000)
    items: list[CxrLabelOpinion] = Field(default_factory=list)


class CxrReviewUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=0)
    reason: str | None = Field(default=None, max_length=2000)
    items: list[CxrLabelOpinion] = Field(default_factory=list)


class CxrDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approve: bool
    reason: str | None = Field(default=None, max_length=2000)


class CxrReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    review_id: str
