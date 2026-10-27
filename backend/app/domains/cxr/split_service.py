"""CXR 患者/component 隔离与四集划分。

规则（02 技术契约 B3）：
- 以**完整 patient-component** 做确定性 hash 分配，比例 70/10/10/10，split_seed=42；
- 同患者不跨集；同源/已识别重复 component 不跨集；
- 每张图保留 patient_key / study_key / component_key / root / hash；
- 划分在取样后、训练前完成，不在看到结果后重分。
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

from app.schemas.cxr import (
    CXR_GROUP_LEVEL,
    CXR_LABEL_SCHEMA,
    CXR_LABELS,
    CXR_MANIFEST_SCHEMA,
    CXR_PREPROCESS_SCHEMA,
    CXR_SPLITS,
    CXR_TASK,
    LABEL_KEYS,
    MASKED_RAW_STATES,
    RAW_POSITIVE,
    RAW_STATES,
    RAW_UNCERTAIN,
    RAW_UNMENTIONED,
    mask_vector,
    raw_state_counts,
)


class CxrSplitError(ValueError):
    """冻结前置条件不满足；details 直接进 422 data。"""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def canonical_json(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pseudo_key(namespace: str, value: str, *, length: int = 32) -> str:
    """稳定伪标识：namespace + 源 ID 的 SHA256 前缀。

    这不是"完全不可重识别匿名化"，只保证服务与报告不直接暴露真实身份。
    """
    return sha256_text(f"{namespace}:{value}")[:length]


def component_hash(namespace: str, split_seed: int, component_key: str) -> str:
    return sha256_text(f"{namespace}|{split_seed}|{component_key}")


def _unit(hash_hex: str) -> float:
    # 52 bit 精度，避免浮点边界抖动；同一输入永远同一结果。
    return int(hash_hex[:13], 16) / float(16**13)


def assign_components(entries: list[dict], *, namespace: str) -> tuple[list[dict], list[dict]]:
    """按患者 + 已识别重复合并 component，返回 (entries, 重复候选)。

    同像素（或同文件）hash 出现在不同患者时，合并为一个 component，避免
    "独立患者 ID"掩盖图像不独立。合并不删除任何样本。
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            # 取字典序更小者作根，保证结果与输入顺序无关。
            low, high = sorted((left_root, right_root))
            parent[high] = low

    for entry in entries:
        find(entry["patient_key"])

    by_hash: dict[str, list[str]] = {}
    for entry in entries:
        for key_name in ("pixel_sha256", "sha256"):
            digest = entry.get(key_name)
            if digest:
                by_hash.setdefault(f"{key_name}:{digest}", []).append(entry["patient_key"])

    duplicates: list[dict] = []
    for digest_key, patients in sorted(by_hash.items()):
        unique_patients = sorted(set(patients))
        if len(patients) > 1:
            duplicates.append(
                {
                    "kind": digest_key.split(":", 1)[0],
                    "hash": digest_key.split(":", 1)[1],
                    "occurrences": len(patients),
                    "patients": unique_patients,
                    "action": "merged_into_one_component" if len(unique_patients) > 1 else "same_patient",
                }
            )
        for patient in unique_patients[1:]:
            union(unique_patients[0], patient)

    for entry in entries:
        root = find(entry["patient_key"])
        entry["component_key"] = pseudo_key(namespace, f"component:{root}")
    return entries, duplicates


def assign_splits(
    components: Iterable[str],
    *,
    namespace: str,
    split_seed: int,
    ratios: list[float],
) -> dict[str, str]:
    """确定性 hash 分配；component 数量足够时保证四集非空。"""
    ordered = sorted(set(components), key=lambda key: (component_hash(namespace, split_seed, key), key))
    if not ordered:
        return {}
    cumulative: list[float] = []
    running = 0.0
    for ratio in ratios:
        running += float(ratio)
        cumulative.append(running)
    cumulative[-1] = 1.0

    assignment: dict[str, str] = {}
    for component in ordered:
        unit = _unit(component_hash(namespace, split_seed, component))
        target = CXR_SPLITS[-1]
        for split, boundary in zip(CXR_SPLITS, cumulative):
            if unit < boundary:
                target = split
                break
        assignment[component] = target

    if len(ordered) >= len(CXR_SPLITS):
        counts = {split: 0 for split in CXR_SPLITS}
        for split in assignment.values():
            counts[split] += 1
        for split in CXR_SPLITS:
            if counts[split]:
                continue
            donor = max(CXR_SPLITS, key=lambda name: (counts[name], -CXR_SPLITS.index(name)))
            if counts[donor] <= 1:
                continue
            movable = next(item for item in ordered if assignment[item] == donor)
            assignment[movable] = split
            counts[donor] -= 1
            counts[split] += 1
    return assignment


def build_cxr_manifest(
    entries: list[dict],
    *,
    namespace: str,
    dataset_id: str,
    version_id: str,
    split_seed: int,
    ratios: list[float],
    source: dict | None = None,
) -> tuple[dict, dict]:
    """构建 cxr-manifest-v1，返回 (manifest, stats)。

    entries 必须已包含 patient_key / study_key / relative_path / sha256 /
    pixel_sha256 / width / height / view / raw_labels。
    """
    if not entries:
        raise CxrSplitError("没有可冻结的胸片样本")
    if len(ratios) != len(CXR_SPLITS) or any(item <= 0 for item in ratios):
        raise CxrSplitError("四集比例必须是四个正数", {"ratios": ratios})

    prepared = [dict(entry) for entry in entries]
    prepared, duplicates = assign_components(prepared, namespace=namespace)
    assignment = assign_splits(
        (entry["component_key"] for entry in prepared),
        namespace=namespace,
        split_seed=split_seed,
        ratios=ratios,
    )
    if not assignment:
        raise CxrSplitError("没有可分配的 component")
    if len(assignment) < len(CXR_SPLITS):
        raise CxrSplitError(
            "患者 component 少于四集数量，无法建立 train/val/calibration/test",
            {"components": len(assignment), "required": len(CXR_SPLITS)},
        )

    # 同 component 必须同集；构造保证，但显式校验以防未来改动破坏不变量。
    seen: dict[str, str] = {}
    for entry in prepared:
        component = entry["component_key"]
        split = assignment[component]
        if component in seen and seen[component] != split:
            raise CxrSplitError("同一 component 跨集", {"component_key": component})
        seen[component] = split
        entry["split"] = split
        entry["mask"] = mask_vector(entry["raw_labels"])
        entry["label_schema"] = CXR_LABEL_SCHEMA

    prepared.sort(key=lambda item: item["relative_path"])
    split_counts = {split: 0 for split in CXR_SPLITS}
    patient_sets = {split: set() for split in CXR_SPLITS}
    component_sets = {split: set() for split in CXR_SPLITS}
    for entry in prepared:
        split_counts[entry["split"]] += 1
        patient_sets[entry["split"]].add(entry["patient_key"])
        component_sets[entry["split"]].add(entry["component_key"])

    overlap = _overlap_report(patient_sets, component_sets, prepared)
    if overlap["patients_cross_split"] or overlap["components_cross_split"]:
        raise CxrSplitError("患者或 component 跨集", overlap)

    label_coverage = {
        split: raw_state_counts(entry["raw_labels"] for entry in prepared if entry["split"] == split)
        for split in CXR_SPLITS
    }
    masked_support = {
        split: {
            key: {
                "positive": sum(
                    1 for entry in prepared
                    if entry["split"] == split and entry["raw_labels"].get(key) == RAW_POSITIVE
                ),
                "negative": sum(
                    1 for entry in prepared
                    if entry["split"] == split and entry["raw_labels"].get(key) == "negative"
                ),
            }
            for key in LABEL_KEYS
        }
        for split in CXR_SPLITS
    }

    manifest = {
        "schema": CXR_MANIFEST_SCHEMA,
        "task": CXR_TASK,
        "label_schema": CXR_LABEL_SCHEMA,
        "labels": [
            {"index": label.index, "key": label.key, "source_column": label.source_column}
            for label in CXR_LABELS
        ],
        "raw_states": list(RAW_STATES),
        "masked_raw_states": list(MASKED_RAW_STATES),
        "mask_policy": "positive_negative_only",
        "group_level": CXR_GROUP_LEVEL,
        "preprocess_schema": CXR_PREPROCESS_SCHEMA,
        "namespace": namespace,
        "dataset_id": dataset_id,
        "version_id": version_id,
        "split_seed": int(split_seed),
        "ratios": [float(item) for item in ratios],
        "split_algorithm": "sha256(namespace|split_seed|component_key) ordered cumulative assignment",
        "duplicate_policy": "exact file sha256 + decoded pixel sha256 merged into one component",
        "source": source or {},
        "counts": split_counts,
        "patient_counts": {split: len(patient_sets[split]) for split in CXR_SPLITS},
        "component_counts": {split: len(component_sets[split]) for split in CXR_SPLITS},
        "label_coverage": label_coverage,
        "masked_support": masked_support,
        "duplicate_candidates": duplicates,
        "assets": prepared,
    }
    manifest_sha256 = sha256_text(canonical_json(manifest))
    split_payload = {
        "split_seed": int(split_seed),
        "namespace": namespace,
        "assignment": {entry["relative_path"]: entry["split"] for entry in prepared},
    }
    stats = {
        "manifest_sha256": manifest_sha256,
        "split_manifest_hash": sha256_text(canonical_json(split_payload)),
        "label_schema_hash": sha256_text(canonical_json(manifest["labels"])),
        "asset_count": len(prepared),
        "split_counts": split_counts,
        "patient_counts": manifest["patient_counts"],
        "component_counts": manifest["component_counts"],
        "masked_support": masked_support,
        "label_coverage": label_coverage,
        "duplicate_candidates": duplicates,
    }
    return manifest, stats


def _overlap_report(patient_sets: dict[str, set], component_sets: dict[str, set], entries: list[dict]) -> dict:
    patients_cross: dict[str, list[str]] = {}
    all_patients: dict[str, set[str]] = {}
    for entry in entries:
        all_patients.setdefault(entry["patient_key"], set()).add(entry["split"])
    for patient, splits in all_patients.items():
        if len(splits) > 1:
            patients_cross[patient] = sorted(splits)
    components_cross: dict[str, list[str]] = {}
    all_components: dict[str, set[str]] = {}
    for entry in entries:
        all_components.setdefault(entry["component_key"], set()).add(entry["split"])
    for component, splits in all_components.items():
        if len(splits) > 1:
            components_cross[component] = sorted(splits)
    return {
        "patients_cross_split": patients_cross,
        "components_cross_split": components_cross,
        "patient_counts": {split: len(patient_sets[split]) for split in CXR_SPLITS},
        "component_counts": {split: len(component_sets[split]) for split in CXR_SPLITS},
    }


def integrity_report(manifest: dict) -> dict:
    """独立复核：患者/component/重复 hash 都不跨集。"""
    problems: list[dict] = []
    patients: dict[str, set[str]] = {}
    components: dict[str, set[str]] = {}
    hashes: dict[str, set[str]] = {}
    for entry in manifest.get("assets", []):
        patients.setdefault(entry["patient_key"], set()).add(entry["split"])
        components.setdefault(entry["component_key"], set()).add(entry["split"])
        for key_name in ("pixel_sha256", "sha256"):
            digest = entry.get(key_name)
            if digest:
                hashes.setdefault(digest, set()).add(entry["split"])
    for patient, splits in patients.items():
        if len(splits) > 1:
            problems.append({"kind": "patient_cross_split", "key": patient, "splits": sorted(splits)})
    for component, splits in components.items():
        if len(splits) > 1:
            problems.append({"kind": "component_cross_split", "key": component, "splits": sorted(splits)})
    for digest, splits in hashes.items():
        if len(splits) > 1:
            problems.append({"kind": "duplicate_cross_split", "key": digest[:12], "splits": sorted(splits)})
    return {
        "ok": not problems,
        "problems": problems,
        "counts": manifest.get("counts", {}),
        "patient_counts": manifest.get("patient_counts", {}),
    }


def training_gate(manifest: dict) -> list[dict]:
    """真实训练门禁：每个标签在 train 必须有明确正例与负例。"""
    problems: list[dict] = []
    support = (manifest.get("masked_support") or {}).get("train") or {}
    for key in LABEL_KEYS:
        item = support.get(key) or {}
        positive = int(item.get("positive") or 0)
        negative = int(item.get("negative") or 0)
        if positive == 0:
            problems.append({"label_key": key, "problem": "train 没有明确正例", "positive": positive})
        if negative == 0:
            problems.append({"label_key": key, "problem": "train 没有明确负例", "negative": negative})
    return problems


def summarize_version(manifest: dict) -> dict:
    """版本摘要：四集患者/图数、四态覆盖、hash。"""
    return {
        "schema": manifest.get("schema"),
        "task": manifest.get("task"),
        "label_schema": manifest.get("label_schema"),
        "group_level": manifest.get("group_level"),
        "split_seed": manifest.get("split_seed"),
        "counts": manifest.get("counts"),
        "patient_counts": manifest.get("patient_counts"),
        "component_counts": manifest.get("component_counts"),
        "label_coverage": manifest.get("label_coverage"),
        "masked_support": manifest.get("masked_support"),
        "duplicate_candidates": manifest.get("duplicate_candidates"),
        "source": manifest.get("source"),
        "training_gate": training_gate(manifest),
    }
