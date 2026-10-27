"""CXR 患者/component 隔离、四集划分与重复候选。"""
from __future__ import annotations

import pytest

from app.domains.cxr.split_service import (
    CxrSplitError,
    assign_splits,
    build_cxr_manifest,
    integrity_report,
    pseudo_key,
    training_gate,
)
from app.schemas.cxr import LABEL_KEYS, RAW_NEGATIVE, RAW_POSITIVE, RAW_UNCERTAIN


def state_labels(positive_index: int = 0) -> dict[str, str]:
    return {
        key: (RAW_POSITIVE if position == positive_index else RAW_NEGATIVE)
        for position, key in enumerate(LABEL_KEYS)
    }


def entry(index: int, *, patient: str | None = None, study: str = "study1",
          pixel: str | None = None, labels: dict | None = None) -> dict:
    patient = patient or f"patient{index:05d}"
    return {
        "relative_path": f"{patient}/{study}/{index}.jpg",
        "sha256": pixel or (f"{index:064d}"),
        "pixel_sha256": pixel or (f"{index:064d}"),
        "size": 100,
        "width": 64,
        "height": 64,
        "patient_key": pseudo_key("ns", f"patient:{patient}"),
        "study_key": pseudo_key("ns", f"study:{patient}:{study}"),
        "view": "Frontal",
        "raw_labels": labels or state_labels(0),
    }


def build(entries, *, seed: int = 42, ratios=None):
    return build_cxr_manifest(
        entries,
        namespace="ns",
        dataset_id="d1",
        version_id="v1",
        split_seed=seed,
        ratios=ratios or [0.7, 0.1, 0.1, 0.1],
    )


def test_four_splits_non_empty_and_no_patient_crosses():
    entries = [entry(index) for index in range(12)]
    manifest, stats = build(entries)
    report = integrity_report(manifest)
    assert report["ok"], report["problems"]
    assert all(manifest["counts"][split] >= 1 for split in ("train", "val", "calibration", "test"))
    assert stats["asset_count"] == 12
    assert manifest["group_level"] == "patient_component"


def test_same_patient_multiple_studies_stay_together():
    entries = [entry(1, patient="patient00001", study="study1"), entry(2, patient="patient00001", study="study2")]
    entries += [entry(index) for index in range(3, 12)]
    manifest, _stats = build(entries)
    splits = {item["relative_path"]: item["split"] for item in manifest["assets"]}
    assert splits["patient00001/study1/1.jpg"] == splits["patient00001/study2/2.jpg"]


def test_same_pixel_hash_across_patients_merges_into_one_component():
    shared = "f" * 64
    entries = [entry(1, patient="patient00001", pixel=shared), entry(2, patient="patient00002", pixel=shared)]
    entries += [entry(index) for index in range(3, 12)]
    manifest, _stats = build(entries)
    components = {item["patient_key"]: item["component_key"] for item in manifest["assets"]}
    assert components[pseudo_key("ns", "patient:patient00001")] == components[pseudo_key("ns", "patient:patient00002")]
    assert manifest["duplicate_candidates"], "重复候选必须被记录"
    assert integrity_report(manifest)["ok"]


def test_manifest_is_deterministic_regardless_of_input_order():
    entries = [entry(index) for index in range(12)]
    first, stats_a = build(entries)
    second, stats_b = build(list(reversed(entries)))
    assert first == second
    assert stats_a["manifest_sha256"] == stats_b["manifest_sha256"]
    assert stats_a["split_manifest_hash"] == stats_b["split_manifest_hash"]


def test_fewer_than_four_components_is_rejected():
    entries = [entry(index) for index in range(3)]
    with pytest.raises(CxrSplitError):
        build(entries)


def test_integrity_report_detects_forced_cross_split():
    entries = [entry(index) for index in range(8)]
    manifest, _stats = build(entries)
    first = manifest["assets"][0]
    other = next(item for item in manifest["assets"] if item["split"] != first["split"])
    other["patient_key"] = first["patient_key"]
    report = integrity_report(manifest)
    assert not report["ok"]
    assert any(item["kind"] == "patient_cross_split" for item in report["problems"])


def test_training_gate_flags_missing_positive_or_negative():
    all_negative = {key: RAW_NEGATIVE for key in LABEL_KEYS}
    entries = [entry(index, labels=all_negative) for index in range(12)]
    manifest, _stats = build(entries)
    problems = training_gate(manifest)
    assert {item["label_key"] for item in problems} == set(LABEL_KEYS)
    assert all("没有明确正例" in item["problem"] for item in problems)


def test_uncertain_and_unmentioned_do_not_count_as_supervision():
    labels = {key: RAW_UNCERTAIN for key in LABEL_KEYS}
    entries = [entry(index, labels=labels) for index in range(12)]
    manifest, _stats = build(entries)
    train_support = manifest["masked_support"]["train"]
    assert all(item["positive"] == 0 and item["negative"] == 0 for item in train_support.values())
    coverage = manifest["label_coverage"]["train"]
    assert all(item["uncertain"] == manifest["counts"]["train"] for item in coverage.values())


def test_split_assignment_ratio_shape():
    components = [pseudo_key("ns", f"c{index}") for index in range(40)]
    assignment = assign_splits(components, namespace="ns", split_seed=7, ratios=[0.7, 0.1, 0.1, 0.1])
    counts = {split: 0 for split in ("train", "val", "calibration", "test")}
    for split in assignment.values():
        counts[split] += 1
    assert counts["train"] >= counts["test"]
    assert sum(counts.values()) == len(components)
