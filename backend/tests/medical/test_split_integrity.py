"""PR-02：来源组三集划分与 manifest 不变量（纯函数级）。"""
from __future__ import annotations

import pytest

from app.domains.datasets.split_service import (
    SplitError,
    build_manifest,
    integrity_report,
)


def entry(path: str, group: str, classes=(0, 1, 2), sha: str | None = None) -> dict:
    return {
        "relative_path": path,
        "sha256": sha or (path * 8)[:64].ljust(64, "a"),
        "size": 100,
        "width": 64,
        "height": 64,
        "source_group": group,
        "annotations": [{"class_id": c, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2} for c in classes],
    }


def test_split_has_no_overlap_on_three_levels():
    entries = [entry(f"img{i:02d}.jpg", f"g{i // 2}") for i in range(12)]
    manifest, stats = build_manifest(
        entries,
        class_map={"0": "RBC", "1": "WBC", "2": "Platelets"},
        seed=42,
        dataset_id="d1",
        version_id="v1",
    )
    report = integrity_report(manifest)
    assert report["ok"], report["problems"]
    assert stats["asset_count"] == 12
    assert all(manifest["counts"][split] >= 1 for split in ("train", "val", "test"))


def test_manifest_is_deterministic_for_same_seed():
    entries = [entry(f"img{i:02d}.jpg", f"g{i}", classes=(0,)) for i in range(9)]
    manifest_a, stats_a = build_manifest(
        entries, class_map={"0": "RBC"}, seed=7, dataset_id="d1", version_id="v1"
    )
    # 输入顺序变化不影响结果
    manifest_b, stats_b = build_manifest(
        list(reversed(entries)), class_map={"0": "RBC"}, seed=7, dataset_id="d1", version_id="v1"
    )
    assert manifest_a == manifest_b
    assert stats_a["manifest_sha256"] == stats_b["manifest_sha256"]
    assert stats_a["split_manifest_hash"] == stats_b["split_manifest_hash"]


def test_group_count_below_three_is_rejected():
    entries = [entry("a.jpg", "g1"), entry("b.jpg", "g2")]
    with pytest.raises(SplitError) as excinfo:
        build_manifest(entries, class_map={"0": "RBC"}, seed=1, dataset_id="d", version_id="v")
    assert excinfo.value.details["required"] == 3


def test_class_coverage_is_enforced():
    entries = [
        entry("a.jpg", "g1", classes=(0,)),
        entry("b.jpg", "g2", classes=(0,)),
        entry("c.jpg", "g3", classes=(1,)),
    ]
    with pytest.raises(SplitError) as excinfo:
        build_manifest(
            entries, class_map={"0": "RBC", "1": "WBC"}, seed=3, dataset_id="d", version_id="v"
        )
    assert "missing_classes" in excinfo.value.details


def test_same_hash_in_different_groups_is_rejected():
    same = "b" * 64
    entries = [
        entry("a.jpg", "g1", sha=same),
        entry("copy.jpg", "g2", sha=same),
        entry("c.jpg", "g3"),
        entry("d.jpg", "g4"),
    ]
    with pytest.raises(SplitError) as excinfo:
        build_manifest(entries, class_map={"0": "RBC"}, seed=1, dataset_id="d", version_id="v")
    assert excinfo.value.details["conflicts"][0]["sha256"] == same


def test_unknown_class_is_rejected():
    entries = [entry(f"img{i}.jpg", f"g{i}", classes=(5,)) for i in range(3)]
    with pytest.raises(SplitError) as excinfo:
        build_manifest(entries, class_map={"0": "RBC"}, seed=1, dataset_id="d", version_id="v")
    assert excinfo.value.details["class_id"] == 5
