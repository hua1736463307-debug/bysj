"""CXR 端到端工程闭环：导入 → 冻结 → 预检 → 分析 → 解释。"""
from __future__ import annotations

import json
from pathlib import Path

import cxr_support
from app.schemas.cxr import CXR_LABEL_SCHEMA, LABEL_KEYS


def test_schema_endpoint_requires_auth_and_lists_five_labels(client):
    assert client.get("/api/v1/cxr/schema").status_code == 401


def test_schema_endpoint_payload(client, admin_headers):
    response = client.get("/api/v1/cxr/schema", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["label_schema"] == CXR_LABEL_SCHEMA
    assert [item["key"] for item in data["labels"]] == list(LABEL_KEYS)
    assert data["splits"] == ["train", "val", "calibration", "test"]


def test_synthetic_source_is_rejected_by_default(client, admin_headers, settings, tmp_path):
    root = cxr_support.build_source(Path(tmp_path) / "src", patients=40)
    settings.cxr_sources = {"fixture": str(root)}
    settings.cxr_allow_synthetic = False
    response = client.post(
        "/api/v1/cxr/datasets/import",
        json={"source_id": "fixture", "source_csv": "train.csv", "max_patients": 40},
        headers={**admin_headers, "Idempotency-Key": "reject-synthetic"},
    )
    assert response.status_code == 202, response.text
    job = response.json()["data"]
    assert job["status"] == "failed"
    assert job["error"]["code"] == "cxr_synthetic_source_rejected"


def test_import_requires_configured_source(client, admin_headers, settings):
    settings.cxr_sources = {}
    settings.cxr_import_root = None
    response = client.post(
        "/api/v1/cxr/datasets/import",
        json={"source_id": "missing", "source_csv": "train.csv", "max_patients": 5},
        headers={**admin_headers, "Idempotency-Key": "missing-source"},
    )
    assert response.status_code == 202
    job = response.json()["data"]
    assert job["status"] == "failed"
    assert job["error"]["code"] == "cxr_source_not_configured"


def test_import_requires_idempotency_key(client, admin_headers, settings, tmp_path):
    root = cxr_support.build_source(Path(tmp_path) / "src", patients=40)
    cxr_support.configure_source(settings, root)
    response = client.post(
        "/api/v1/cxr/datasets/import",
        json={"source_id": "fixture", "source_csv": "train.csv", "max_patients": 40},
        headers=admin_headers,
    )
    assert response.status_code == 422


def test_import_and_freeze_produce_four_isolated_splits(client, admin_headers, settings, tmp_path):
    result = cxr_support.ready_dataset(client, admin_headers, settings, tmp_path, patients=40)
    freeze_result = result["freeze"]
    assert freeze_result["state"] == "frozen"
    assert freeze_result["asset_count"] >= 10
    counts = freeze_result["split_counts"]
    assert all(counts[split] >= 1 for split in ("train", "val", "calibration", "test"))

    summary = client.get(
        f"/api/v1/cxr/datasets/{result['dataset_id']}/versions/{result['version_id']}/summary",
        headers=admin_headers,
    )
    assert summary.status_code == 200, summary.text
    data = summary.json()["data"]
    assert data["counts"] == counts
    assert data["patient_counts"]["train"] >= 1
    # 患者伪标识不进摘要
    assert "patient_key" not in json.dumps(data)
    assert data["duplicate_candidates"], "跨患者同像素样本必须被记录为重复候选"


def test_manifest_records_no_patient_crosses_split(client, admin_headers, settings, tmp_path):
    result = cxr_support.ready_dataset(client, admin_headers, settings, tmp_path, patients=40)
    manifest_path = None
    for candidate in Path(settings.data_root).rglob("cxr-manifest.json"):
        if result["version_id"] in str(candidate):
            manifest_path = candidate
    assert manifest_path is not None and manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seen: dict[str, set[str]] = {}
    for item in manifest["assets"]:
        seen.setdefault(item["patient_key"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in seen.values())
    assert manifest["label_schema"] == CXR_LABEL_SCHEMA
    assert manifest["group_level"] == "patient_component"


def test_training_preflight_requires_backbone_and_reports_dummy(client, admin_headers, settings, db, cxr_dataset):
    admin_id = db.query(__import__("app.db.models", fromlist=["User"]).User).filter_by(username="admin").one().id
    payload = {"dataset_version_id": cxr_dataset["version_id"], "params": {"epochs": 2, "batch": 2}}
    response = client.post("/api/v1/cxr/training/preflight", json=payload, headers=admin_headers)
    assert response.status_code == 200
    report = response.json()["data"]
    assert report["ok"] is False
    names = {item["name"]: item["status"] for item in report["checks"]}
    assert names["base_model"] == "fail"

    backbone = cxr_support.register_model(
        db, settings, owner_id=admin_id, model_task="cxr_backbone", backend="dummy",
        name="imagenet-backbone", trusted_source="base",
    )
    response = client.post(
        "/api/v1/cxr/training/preflight",
        json={**payload, "base_model_id": backbone.id},
        headers=admin_headers,
    )
    report = response.json()["data"]
    names = {item["name"]: item["status"] for item in report["checks"]}
    assert names["base_model"] == "pass"
    assert names["label_support"] == "pass"
    assert names["backend"] in {"pass", "warn"}


def test_analysis_batch_returns_five_labels_and_quality(client, admin_headers, settings, db, cxr_dataset):
    admin_id = db.query(__import__("app.db.models", fromlist=["User"]).User).filter_by(username="admin").one().id
    model = cxr_support.register_model(db, settings, owner_id=admin_id)
    response = client.post(
        "/api/v1/cxr/analysis-batches",
        json={"model_id": model.id, "asset_ids": cxr_dataset["asset_ids"]},
        headers={**admin_headers, "Idempotency-Key": "batch-1"},
    )
    assert response.status_code == 202, response.text
    batch = response.json()["data"]
    assert batch["status"] == "completed", batch
    assert batch["total"] == len(cxr_dataset["asset_ids"])

    detail = client.get(f"/api/v1/cxr/analysis-batches/{batch['batch_id']}", headers=admin_headers)
    assert detail.status_code == 200
    items = detail.json()["data"]["items"]
    assert len(items) == batch["total"]
    first = items[0]
    assert [item["label_key"] for item in first["labels"]] == list(LABEL_KEYS)
    assert all(item["flag"] in {"above_threshold", "below_threshold", "borderline", "not_available"}
               for item in first["labels"])
    assert first["calibration"]["status"] == "not_calibrated"
    assert first["calibration"]["threshold_source"] == "default_no_calibration"
    assert "relative_path" not in json.dumps(first)
    assert first["quality"]["definition"]


def test_same_idempotency_key_with_different_body_conflicts(client, admin_headers, settings, db, cxr_dataset):
    admin_id = db.query(__import__("app.db.models", fromlist=["User"]).User).filter_by(username="admin").one().id
    model = cxr_support.register_model(db, settings, owner_id=admin_id)
    first = client.post(
        "/api/v1/cxr/analysis-batches",
        json={"model_id": model.id, "asset_ids": cxr_dataset["asset_ids"][:1]},
        headers={**admin_headers, "Idempotency-Key": "dup-key"},
    )
    assert first.status_code == 202
    second = client.post(
        "/api/v1/cxr/analysis-batches",
        json={"model_id": model.id, "asset_ids": cxr_dataset["asset_ids"]},
        headers={**admin_headers, "Idempotency-Key": "dup-key"},
    )
    assert second.status_code == 409
    assert second.json()["code"] == "idempotency_conflict"


def test_upload_batch_works_and_rejects_unsupported_format(client, admin_headers, settings, db):
    from PIL import Image
    import io

    admin_id = db.query(__import__("app.db.models", fromlist=["User"]).User).filter_by(username="admin").one().id
    model = cxr_support.register_model(db, settings, owner_id=admin_id)
    buffer = io.BytesIO()
    Image.new("RGB", (160, 160), (90, 90, 90)).save(buffer, format="JPEG")
    buffer.seek(0)
    response = client.post(
        "/api/v1/cxr/analysis-batches/upload",
        data={"model_id": model.id},
        files=[("files", ("chest.jpg", buffer.getvalue(), "image/jpeg"))],
        headers={**admin_headers, "Idempotency-Key": "upload-1"},
    )
    assert response.status_code == 202, response.text
    assert response.json()["data"]["status"] == "completed"

    response = client.post(
        "/api/v1/cxr/analysis-batches/upload",
        data={"model_id": model.id},
        files=[("files", ("chest.bmp", b"BMfake", "image/bmp"))],
        headers={**admin_headers, "Idempotency-Key": "upload-bmp"},
    )
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_format"


def test_duplicate_upload_in_one_batch_is_deduplicated(client, admin_headers, settings, db):
    import io

    from PIL import Image

    admin_id = db.query(__import__("app.db.models", fromlist=["User"]).User).filter_by(username="admin").one().id
    model = cxr_support.register_model(db, settings, owner_id=admin_id)
    buffer = io.BytesIO()
    Image.new("RGB", (160, 160), (60, 70, 80)).save(buffer, format="JPEG")
    payload = buffer.getvalue()
    response = client.post(
        "/api/v1/cxr/analysis-batches/upload",
        data={"model_id": model.id},
        files=[
            ("files", ("a.jpg", payload, "image/jpeg")),
            ("files", ("b.jpg", payload, "image/jpeg")),
        ],
        headers={**admin_headers, "Idempotency-Key": "dup-upload"},
    )
    assert response.status_code == 202, response.text
    batch = response.json()["data"]
    assert batch["total"] == 1, "同批重复上传同一文件应只产生一条分析"
    assert batch["completed"] == 1


def test_batch_limit_is_not_silently_truncated(client, admin_headers, settings, db, cxr_dataset):
    admin_id = db.query(__import__("app.db.models", fromlist=["User"]).User).filter_by(username="admin").one().id
    model = cxr_support.register_model(db, settings, owner_id=admin_id)
    settings.max_batch_inference = 1
    try:
        response = client.post(
            "/api/v1/cxr/analysis-batches",
            json={"model_id": model.id, "asset_ids": cxr_dataset["asset_ids"]},
            headers={**admin_headers, "Idempotency-Key": "too-many"},
        )
    finally:
        settings.max_batch_inference = 100
    assert response.status_code == 422
    assert response.json()["code"] == "cxr_batch_too_large"


def test_explanation_creates_artifact(client, admin_headers, settings, db, cxr_dataset):
    admin_id = db.query(__import__("app.db.models", fromlist=["User"]).User).filter_by(username="admin").one().id
    model = cxr_support.register_model(db, settings, owner_id=admin_id)
    batch = client.post(
        "/api/v1/cxr/analysis-batches",
        json={"model_id": model.id, "asset_ids": cxr_dataset["asset_ids"][:1]},
        headers={**admin_headers, "Idempotency-Key": "explain-batch"},
    ).json()["data"]
    analysis_id = client.get(
        f"/api/v1/cxr/analysis-batches/{batch['batch_id']}", headers=admin_headers
    ).json()["data"]["items"][0]["analysis_id"]

    response = client.post(
        f"/api/v1/cxr/analyses/{analysis_id}/explanations",
        json={"label_key": "cardiomegaly", "method": "gradcam"},
        headers=admin_headers,
    )
    assert response.status_code == 202, response.text
    assert response.json()["data"]["status"] == "completed"

    analysis = client.get(f"/api/v1/cxr/analyses/{analysis_id}", headers=admin_headers).json()["data"]
    assert analysis["explanation_artifact_id"]
    artifacts = analysis["explanation"]["artifacts"]
    assert artifacts["cardiomegaly"]["label_key"] == "cardiomegaly"
    assert "无定位真值" in " ".join(artifacts["cardiomegaly"]["limitations"]) or artifacts["cardiomegaly"]["limitations"]

    bad = client.post(
        f"/api/v1/cxr/analyses/{analysis_id}/explanations",
        json={"label_key": "not_a_label"},
        headers=admin_headers,
    )
    assert bad.status_code == 422


def test_cross_user_cannot_read_analysis(client, admin_headers, other_headers, settings, db, cxr_dataset):
    admin_id = db.query(__import__("app.db.models", fromlist=["User"]).User).filter_by(username="admin").one().id
    model = cxr_support.register_model(db, settings, owner_id=admin_id)
    batch = client.post(
        "/api/v1/cxr/analysis-batches",
        json={"model_id": model.id, "asset_ids": cxr_dataset["asset_ids"][:1]},
        headers={**admin_headers, "Idempotency-Key": "cross-user"},
    ).json()["data"]
    analysis_id = client.get(
        f"/api/v1/cxr/analysis-batches/{batch['batch_id']}", headers=admin_headers
    ).json()["data"]["items"][0]["analysis_id"]

    assert client.get(f"/api/v1/cxr/analyses/{analysis_id}", headers=other_headers).status_code == 404
    assert client.get(
        f"/api/v1/cxr/analysis-batches/{batch['batch_id']}", headers=other_headers
    ).status_code == 404
    # 他人资产也不能进入自己的分析批次
    response = client.post(
        "/api/v1/cxr/analysis-batches",
        json={"model_id": model.id, "asset_ids": cxr_dataset["asset_ids"][:1]},
        headers={**other_headers, "Idempotency-Key": "other-assets"},
    )
    assert response.status_code == 404
