"""P0：实验用途与论文资格门禁。

这些测试只用标准库 + API 契约，不需要 torch/GPU，也不需要真实数据。
它们要保证的不是"分数好不好"，而是**smoke 不可能混进论文汇总**：
`research_valid=true` 只说明产物不是 dummy，不能自动获得论文资格。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.domains.isic.experiment_purpose import (
    FORMAL_DATASET_MANIFEST_SHA256,
    FORMAL_PROTOCOL_ID,
    FORMAL_PROTOCOL_SHA256,
    eligibility_metadata,
    is_paper_eligible,
    normalize_purpose,
    paper_eligibility,
    validate_formal_dataset,
    validate_formal_protocol,
)

FORMAL_META = {
    "experiment_purpose": "formal",
    "research_valid": True,
    "backend": "torch",
    "synthetic": False,
    "dataset_manifest_sha256": FORMAL_DATASET_MANIFEST_SHA256,
    "protocol_id": FORMAL_PROTOCOL_ID,
    "protocol_sha256": FORMAL_PROTOCOL_SHA256,
    "init_weights_sha256": "a" * 64,
    "seed": 42,
}


def test_normalize_purpose_accepts_only_two_values():
    assert normalize_purpose("smoke") == "smoke"
    assert normalize_purpose(" FORMAL ") == "formal"
    assert normalize_purpose("production") == ""
    assert normalize_purpose(None) == ""
    assert normalize_purpose(1) == ""


def test_formal_fully_bound_metadata_is_eligible():
    report = paper_eligibility(dict(FORMAL_META))
    assert report["paper_eligible"] is True, report["reasons"]
    assert report["reasons"] == []


def test_smoke_is_never_paper_eligible_even_when_research_valid():
    meta = {**FORMAL_META, "experiment_purpose": "smoke"}
    report = paper_eligibility(meta)
    assert report["paper_eligible"] is False
    assert "experiment_purpose_smoke" in report["reasons"]


def test_missing_purpose_is_default_deny():
    meta = {key: value for key, value in FORMAL_META.items() if key != "experiment_purpose"}
    report = paper_eligibility(meta)
    assert report["paper_eligible"] is False
    assert "experiment_purpose_missing" in report["reasons"]


def test_synthetic_and_dummy_never_eligible():
    assert "synthetic_artifact" in paper_eligibility({**FORMAL_META, "synthetic": True})["reasons"]
    assert "backend_not_real" in paper_eligibility({**FORMAL_META, "backend": "dummy"})["reasons"]
    assert "research_valid_not_true" in paper_eligibility({**FORMAL_META, "research_valid": False})["reasons"]


def test_formal_requires_frozen_protocol_data_weights_and_seed():
    assert "frozen_protocol_sha256_mismatch" in paper_eligibility(
        {**FORMAL_META, "protocol_sha256": "0" * 64}
    )["reasons"]
    assert "frozen_protocol_id_mismatch" in paper_eligibility(
        {**FORMAL_META, "protocol_id": "isic-formal-v2"}
    )["reasons"]
    assert "dataset_manifest_sha256_not_frozen_version" in paper_eligibility(
        {**FORMAL_META, "dataset_manifest_sha256": "1" * 64}
    )["reasons"]
    assert "init_weights_sha256_missing" in paper_eligibility(
        {**FORMAL_META, "init_weights_sha256": None}
    )["reasons"]
    assert "seed_not_preregistered" in paper_eligibility({**FORMAL_META, "seed": 7})["reasons"]


def test_validate_helpers_require_both_fields():
    assert validate_formal_protocol(None, FORMAL_PROTOCOL_SHA256) == ["frozen_protocol_missing"]
    assert validate_formal_dataset(None) == ["dataset_manifest_sha256_missing"]


def test_stored_boolean_is_not_trusted():
    """即使历史元数据里被写进 paper_eligible=true，也要按字段重算。"""
    meta = {**FORMAL_META, "experiment_purpose": "smoke", "paper_eligible": True}
    assert is_paper_eligible(meta) is False
    flattened = eligibility_metadata(meta)
    assert flattened["paper_eligible"] is False
    assert flattened["paper_ineligible_reasons"]


def test_protocol_document_hash_matches_constant():
    """协议文本改动后必须重新计算常量，否则门禁会静默放行旧哈希。"""
    doc = Path(__file__).resolve().parents[3] / "docs" / "datasets" / "ISIC-FORMAL-PROTOCOL-v1.md"
    if not doc.exists():  # 容器只挂载 backend/ 时可以跳过
        pytest.skip("protocol document not mounted")
    digest = hashlib.sha256(doc.read_bytes()).hexdigest()
    assert digest == FORMAL_PROTOCOL_SHA256


# ---------------------------------------------------------------- API 契约


def test_preflight_marks_missing_purpose_as_warning(client, admin_headers):
    response = client.post(
        "/api/v1/isic/training/preflight",
        json={"dataset_version_id": "not-a-real-version", "params": {"epochs": 1}},
        headers=admin_headers,
    )
    assert response.status_code == 200
    report = response.json()["data"]
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["experiment_purpose"]["status"] == "warn"
    assert report["experiment_purpose"] is None


def test_preflight_formal_without_protocol_fails(client, admin_headers):
    response = client.post(
        "/api/v1/isic/training/preflight",
        json={"dataset_version_id": "not-a-real-version", "experiment_purpose": "formal"},
        headers=admin_headers,
    )
    assert response.status_code == 200
    report = response.json()["data"]
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["formal_protocol"]["status"] == "fail"
    assert "frozen_protocol_missing" in checks["formal_protocol"]["detail"]
    assert report["ok"] is False


def test_preflight_rejects_unknown_purpose(client, admin_headers):
    response = client.post(
        "/api/v1/isic/training/preflight",
        json={"dataset_version_id": "not-a-real-version", "experiment_purpose": "production"},
        headers=admin_headers,
    )
    assert response.status_code == 422


def test_paper_summary_endpoint_excludes_ineligible(client, admin_headers):
    response = client.get("/api/v1/isic/results/paper-summary", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["protocol_id"] == FORMAL_PROTOCOL_ID
    assert data["included"] == []
    assert data["excluded"] == []


# ------------------------------------------------- 汇总排除（真实落库数据）


def _seed_model(db, metadata: dict, *, name: str):
    from app.db.models import ModelRegistry

    model = ModelRegistry(name=name, path="/tmp/does-not-exist.pt", sha256="b" * 64, metadata_json=metadata)
    db.add(model)
    db.commit()
    return model


def _seed_test_evaluation(db, model, *, metrics: dict, purpose: str | None):
    from app.db.models import EvaluationRun

    run = EvaluationRun(
        model_id=model.id,
        dataset_version_id="87e0eb4f-f112-4992-be69-7a1c57812aa6",
        split="test",
        status="completed",
        metrics_json=metrics,
        purpose=purpose,
    )
    db.add(run)
    db.commit()
    return run


def test_paper_summary_excludes_smoke_run_but_keeps_it_queryable(client, admin_headers, db):
    """真实落库的 smoke test 记录：可查，但汇总里只出现原因、不出现指标。"""
    smoke_meta = {
        "experiment_purpose": "smoke",
        "research_valid": True,
        "backend": "torch",
        "synthetic": False,
    }
    model = _seed_model(db, smoke_meta, name="isic-smoke")
    run = _seed_test_evaluation(db, model, metrics={"macro_f1": 0.91}, purpose="smoke-1-epoch")

    summary = client.get("/api/v1/isic/results/paper-summary", headers=admin_headers).json()["data"]
    assert [item["evaluation_id"] for item in summary["included"]] == []
    assert [item["evaluation_id"] for item in summary["excluded"]] == [run.id]
    excluded = summary["excluded"][0]
    assert "metrics" not in excluded
    assert excluded["paper_eligibility"]["paper_eligible"] is False
    assert "experiment_purpose_smoke" in excluded["paper_eligibility"]["reasons"]

    listed = client.get("/api/v1/isic/evaluations", headers=admin_headers).json()["data"]
    assert [item["evaluation_id"] for item in listed["items"]] == [run.id]
    assert listed["items"][0]["paper_eligibility"]["paper_eligible"] is False


def test_paper_summary_includes_run_that_passes_all_gates(client, admin_headers, db):
    """同一接口在门禁全部满足时确实会收录，排除逻辑不是"永远为空"。"""
    model = _seed_model(db, dict(FORMAL_META), name="isic-formal")
    run = _seed_test_evaluation(db, model, metrics={"macro_f1": 0.71}, purpose=f"{FORMAL_PROTOCOL_ID}-seed42-A")

    summary = client.get("/api/v1/isic/results/paper-summary", headers=admin_headers).json()["data"]
    assert [item["evaluation_id"] for item in summary["included"]] == [run.id]
    assert summary["excluded"] == []
    assert summary["included"][0]["metrics"] == {"macro_f1": 0.71}
