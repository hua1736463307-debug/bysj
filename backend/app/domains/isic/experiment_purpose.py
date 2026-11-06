"""ISIC 实验用途与论文资格门禁（P0）。

为什么需要这个模块：`research_valid` 只回答"产物是不是 dummy"，
它回答不了"能不能进论文"。真实数据 + 真实后端仍可能是一次 1 epoch 的
工程 smoke，或没有绑定冻结协议的临时试验。因此把三件事拆开：

1. `experiment_purpose`：`smoke` / `formal`，提交训练时显式声明；
2. `research_valid`：产物是否来自真实后端的布尔标记（历史字段，语义保持不动）；
3. `paper_eligible`：能否进入论文汇总 / 导出，由本模块计算，不接受人工直填。

历史记录没有 `experiment_purpose` 一律按"未分类"处理（默认拒绝）：
仍可查询，但不会进入论文汇总，因此**不需要改写历史元数据**。
dummy / synthetic 产物永远不可能获得论文资格，即使被标成 formal。

`FORMAL_PROTOCOL_SHA256` 绑定 `docs/datasets/ISIC-FORMAL-PROTOCOL-v1.md` 的字节内容：
协议文本一改，哈希即不匹配，formal 产物自动失去资格；需要修改协议时必须新建
`isic-formal-v2` 并同步更新本文件常量，不得静默改文档。

本模块只依赖标准库，便于在不加载 FastAPI/torch 的环境里单独验证。
"""
from __future__ import annotations

EXPERIMENT_PURPOSE_SMOKE = "smoke"
EXPERIMENT_PURPOSE_FORMAL = "formal"
EXPERIMENT_PURPOSES: tuple[str, ...] = (EXPERIMENT_PURPOSE_SMOKE, EXPERIMENT_PURPOSE_FORMAL)

#: 冻结协议标识与文档哈希（`sha256sum docs/datasets/ISIC-FORMAL-PROTOCOL-v1.md`）。
FORMAL_PROTOCOL_ID = "isic-formal-v1"
FORMAL_PROTOCOL_SHA256 = "40064e0639e7e66727e94c2c90133253da991be55dc963615dd0f8f5b0266c60"
#: 协议第 2 节冻结的数据版本（`dataset_versions.manifest_sha256`）。
#: 换数据版本必须新建 `isic-formal-v2`，不能只改这个常量。
FORMAL_DATASET_MANIFEST_SHA256 = "8f5b609c6197792adf7cc71f79236338fd11c47b6fab85bc3fe51d20b2c563ae"
#: 预登记 seed；其他 seed 的正式产物不具备论文资格。
FORMAL_SEEDS: tuple[int, ...] = (42, 43, 44)
FORMAL_PARAMS = {
    "epochs": 30, "batch": 16, "image_size": 224, "lr": 0.0003,
    "weight_decay": 0.0001, "patience": 8, "num_workers": 0, "amp": True,
}

_DUMMY_BACKENDS = frozenset({"dummy", "none", ""})


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalize_purpose(value: object) -> str:
    """把请求里的 purpose 规范化为受限取值；无法识别一律返回空串（未分类）。"""
    text = _text(value).lower()
    return text if text in EXPERIMENT_PURPOSES else ""


def validate_formal_protocol(protocol_id: object, protocol_sha256: object) -> list[str]:
    """校验 formal 训练绑定的冻结协议；返回不满足的原因（空列表 = 通过）。"""
    identifier = _text(protocol_id)
    digest = _text(protocol_sha256).lower()
    if not identifier or not digest:
        return ["frozen_protocol_missing"]
    reasons: list[str] = []
    if identifier != FORMAL_PROTOCOL_ID:
        reasons.append("frozen_protocol_id_mismatch")
    if digest != FORMAL_PROTOCOL_SHA256:
        reasons.append("frozen_protocol_sha256_mismatch")
    return reasons


def validate_formal_dataset(manifest_sha256: object) -> list[str]:
    """校验 formal 训练使用的数据版本是否为协议第 2 节冻结的那一个。"""
    digest = _text(manifest_sha256).lower()
    if not digest:
        return ["dataset_manifest_sha256_missing"]
    if digest != FORMAL_DATASET_MANIFEST_SHA256:
        return ["dataset_manifest_sha256_not_frozen_version"]
    return []


def is_paper_eligible(metadata: dict | None) -> bool:
    """重新计算（不读取已存的布尔字段），防止历史 true 标志被当成放行凭证。"""
    return bool(paper_eligibility(metadata)["paper_eligible"])


def validate_formal_params(params: object) -> list[str]:
    """Check submitted or actually executed parameters against frozen protocol v1."""
    if not isinstance(params, dict):
        return ["formal_params_missing"]
    reasons = []
    for name, expected in FORMAL_PARAMS.items():
        value = params.get(name)
        if value != expected or (isinstance(value, bool) != isinstance(expected, bool)):
            reasons.append(f"formal_param_{name}_mismatch")
    if params.get("class_weight") not in ("none", "balanced"):
        reasons.append("formal_param_class_weight_mismatch")
    seed = params.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed not in FORMAL_SEEDS:
        reasons.append("seed_not_preregistered")
    return reasons


def paper_eligibility(metadata: dict | None) -> dict:
    """计算论文资格。返回 `paper_eligible` 与逐条 `reasons`（原因码，可用于展示）。"""
    meta = metadata or {}
    reasons: list[str] = []

    purpose = normalize_purpose(meta.get("experiment_purpose"))
    if not purpose:
        reasons.append("experiment_purpose_missing")
    elif purpose != EXPERIMENT_PURPOSE_FORMAL:
        reasons.append(f"experiment_purpose_{purpose}")

    if meta.get("synthetic") is True:
        reasons.append("synthetic_artifact")
    if meta.get("research_valid") is not True:
        reasons.append("research_valid_not_true")

    backend = _text(meta.get("backend")).lower()
    if backend in _DUMMY_BACKENDS:
        reasons.append("backend_not_real")

    if not _text(meta.get("dataset_manifest_sha256")):
        reasons.append("dataset_manifest_sha256_missing")

    if purpose == EXPERIMENT_PURPOSE_FORMAL:
        reasons.extend(validate_formal_protocol(meta.get("protocol_id"), meta.get("protocol_sha256")))
        reasons.extend(validate_formal_dataset(meta.get("dataset_manifest_sha256")))
        if not _text(meta.get("init_weights_sha256")):
            reasons.append("init_weights_sha256_missing")
        seed = meta.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed not in FORMAL_SEEDS:
            reasons.append("seed_not_preregistered")
        effective = meta.get("effective_args")
        if not isinstance(effective, dict):
            reasons.append("effective_args_missing")
        else:
            reasons.extend(validate_formal_params(effective))
            if effective.get("seed") != seed:
                reasons.append("effective_seed_mismatch")
            if effective.get("class_weight") != meta.get("class_weight"):
                reasons.append("effective_class_weight_mismatch")

    return {
        "paper_eligible": not reasons,
        "experiment_purpose": purpose or None,
        "protocol_id": FORMAL_PROTOCOL_ID,
        "reasons": list(dict.fromkeys(reasons)),
    }


def eligibility_metadata(metadata: dict | None) -> dict:
    """把资格判定结果摊平成可直接写入 `metadata_json` 的字段。"""
    report = paper_eligibility(metadata)
    return {
        "paper_eligible": report["paper_eligible"],
        "paper_ineligible_reasons": report["reasons"],
    }
