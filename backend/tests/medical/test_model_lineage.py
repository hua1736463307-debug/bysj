"""PR-05：模型血缘、可信加载与评估契约。"""
from __future__ import annotations

import support


def test_evaluation_idempotency_reuses_run_and_rejects_changed_params(client, researcher_headers):
    version_id, model_id, _ = _train(client, researcher_headers)
    body = {"model_id":model_id,"dataset_version_id":version_id,"split":"val","idempotency_key":"stable-eval"}
    first = client.post("/api/v1/evaluations",headers=researcher_headers,json=body).json()["data"]
    second = client.post("/api/v1/evaluations",headers=researcher_headers,json=body).json()["data"]
    assert first["evaluation"]["evaluation_id"] == second["evaluation"]["evaluation_id"]
    assert first["job"]["job_id"] == second["job"]["job_id"]
    changed = client.post("/api/v1/evaluations",headers=researcher_headers,json={**body,"params":{"imgsz":320}})
    assert changed.status_code == 409


def test_pending_evaluation_cancellation_updates_domain_state(client, researcher_headers, settings, monkeypatch):
    version_id, model_id, _ = _train(client, researcher_headers)
    monkeypatch.setattr(settings,"inline_jobs",False)
    result = client.post("/api/v1/evaluations",headers=researcher_headers,json={"model_id":model_id,"dataset_version_id":version_id}).json()["data"]
    job_id=result["job"]["job_id"]
    assert client.post(f"/api/v1/jobs/{job_id}/cancel",headers=researcher_headers).json()["data"]["status"] == "cancelled"
    run_id=result["evaluation"]["evaluation_id"]
    assert client.get(f"/api/v1/evaluations/{run_id}",headers=researcher_headers).json()["data"]["status"] == "cancelled"


def _train(client, headers):
    dataset_id, version_id = support.make_dataset(client, headers, count=9)
    job = support.freeze(client, headers, dataset_id, version_id=version_id).json()["data"]
    assert job["status"] == "completed"
    request = {
        "dataset_version_id": version_id,
        "params": {"epochs": 1, "batch": 2, "imgsz": 64, "seed": 42, "deterministic": True, "device": "cpu"},
    }
    trained = client.post("/api/v1/training/jobs", headers=headers, json=request).json()["data"]
    assert trained["status"] == "completed", trained
    return version_id, trained["result"]["model_id"], trained["job_id"]


def test_model_lineage_points_to_job_and_version(client, researcher_headers):
    version_id, model_id, job_id = _train(client, researcher_headers)
    model = client.get(f"/api/v1/models/{model_id}", headers=researcher_headers).json()["data"]
    assert model["lineage"]["training_job_id"] == job_id
    assert model["lineage"]["dataset_version_id"] == version_id
    assert model["trusted_source"] == "platform"
    assert model["class_map"]


def test_evaluation_val_reports_null_metrics_with_reason(client, researcher_headers):
    version_id, model_id, _ = _train(client, researcher_headers)
    response = client.post(
        "/api/v1/evaluations",
        headers=researcher_headers,
        json={"model_id": model_id, "dataset_version_id": version_id, "split": "val"},
    )
    assert response.status_code == 202, response.text
    data = response.json()["data"]
    assert data["evaluation"]["status"] == "completed"
    metrics = data["evaluation"]["metrics"]
    assert metrics["map50"] is None
    assert data["evaluation"]["artifacts"]["reason"]
    fetched = client.get(f"/api/v1/evaluations/{data['evaluation']['evaluation_id']}", headers=researcher_headers)
    assert fetched.status_code == 200
    assert fetched.json()["data"]["metrics"]["map50"] is None


def test_test_split_requires_admin_and_purpose(client, researcher_headers, admin_headers):
    version_id, model_id, _ = _train(client, researcher_headers)
    denied = client.post(
        "/api/v1/evaluations",
        headers=researcher_headers,
        json={"model_id": model_id, "dataset_version_id": version_id, "split": "test", "purpose": "final"},
    )
    assert denied.status_code == 403
    assert denied.json()["code"] == "approval_required"

    no_purpose = client.post(
        "/api/v1/evaluations",
        headers=admin_headers,
        json={"model_id": model_id, "dataset_version_id": version_id, "split": "test"},
    )
    assert no_purpose.status_code == 422
    assert no_purpose.json()["code"] == "purpose_required"

    approved = client.post(
        "/api/v1/evaluations",
        headers=admin_headers,
        json={"model_id": model_id, "dataset_version_id": version_id, "split": "test", "purpose": "final E1"},
    )
    assert approved.status_code == 202
    assert approved.json()["data"]["evaluation"]["approved_by"] == "admin"


def test_class_mismatch_is_422(client, researcher_headers, db):
    version_id, model_id, _ = _train(client, researcher_headers)
    from app.db.models import ModelRegistry

    model = db.get(ModelRegistry, model_id)
    model.class_map_json = {"9": "Other"}
    db.commit()
    response = client.post(
        "/api/v1/evaluations",
        headers=researcher_headers,
        json={"model_id": model_id, "dataset_version_id": version_id, "split": "val"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "class_map_mismatch"


def test_tampered_model_file_is_blocked(client, researcher_headers):
    from pathlib import Path

    _, model_id, _ = _train(client, researcher_headers)
    model = client.get(f"/api/v1/models/{model_id}", headers=researcher_headers).json()["data"]
    assert model["available"] is True
    # 通过数据库路径定位受控模型文件并篡改
    from app.db.models import ModelRegistry
    from app.db.session import get_session_factory

    with get_session_factory()() as session:
        record = session.get(ModelRegistry, model_id)
        path = Path(record.path)
    path.write_bytes(b"tampered")
    response = client.get(f"/api/v1/models/{model_id}/download", headers=researcher_headers)
    assert response.status_code == 409
    assert response.json()["code"] == "model_tampered"


def test_non_admin_cannot_register_weights(client, researcher_headers):
    response = client.post(
        "/api/v1/models/register",
        headers=researcher_headers,
        json={"name": "evil", "relative_path": "evil.pt", "sha256": "a" * 64, "class_map": {}},
    )
    assert response.status_code == 403


def test_cross_user_cannot_download_model(client, researcher_headers, db):
    from app.domains.auth_service import create_user

    _, model_id, _ = _train(client, researcher_headers)
    create_user(db, username="thief", password="thief-password", role="researcher")
    token = client.post("/api/v1/auth/login", json={"username": "thief", "password": "thief-password"}).json()["data"]["token"]
    response = client.get(f"/api/v1/models/{model_id}/download", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 404
