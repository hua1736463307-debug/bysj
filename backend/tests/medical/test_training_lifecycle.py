"""PR-04：训练排队、幂等、取消与可复现记录。"""
from __future__ import annotations

import support


def _frozen_dataset(client, headers, count=9):
    dataset_id, version_id = support.make_dataset(client, headers, count=count)
    job = support.freeze(client, headers, dataset_id, version_id=version_id).json()["data"]
    assert job["status"] == "completed", job
    return dataset_id, version_id


def _request(version_id: str, **overrides) -> dict:
    body = {
        "dataset_version_id": version_id,
        "task": "detect",
        "params": {"epochs": 1, "batch": 2, "imgsz": 64, "seed": 42, "deterministic": True, "device": "cpu"},
    }
    body.update(overrides)
    return body


def test_preflight_and_training_job_complete(client, researcher_headers):
    _, version_id = _frozen_dataset(client, researcher_headers)
    preflight = client.post(
        "/api/v1/training/preflight", headers=researcher_headers, json=_request(version_id)
    )
    assert preflight.status_code == 200
    report = preflight.json()["data"]
    assert report["ok"] is True, report["checks"]

    response = client.post("/api/v1/training/jobs", headers=researcher_headers, json=_request(version_id))
    assert response.status_code == 202, response.text
    job = response.json()["data"]
    assert job["type"] == "training"
    assert job["status"] == "completed", job
    result = job["result"]
    assert result["backend"] == "dummy"
    assert len(result["model_sha256"]) == 64
    assert result["experiment_id"]

    model = client.get(f"/api/v1/models/{result['model_id']}", headers=researcher_headers)
    assert model.status_code == 200
    assert model.json()["data"]["task"] == "detect"
    download = client.get(f"/api/v1/models/{result['model_id']}/download", headers=researcher_headers)
    assert download.status_code == 200
    assert len(download.content) > 0


def test_training_job_is_marked_dummy_and_metrics_not_faked(client, researcher_headers):
    _, version_id = _frozen_dataset(client, researcher_headers)
    job = client.post("/api/v1/training/jobs", headers=researcher_headers, json=_request(version_id)).json()["data"]
    metrics = job["result"]["metrics"]
    assert metrics["map50"] is None
    assert metrics["reason"]


def test_idempotency_key_returns_same_job(client, researcher_headers):
    _, version_id = _frozen_dataset(client, researcher_headers)
    body = _request(version_id)
    first = client.post(
        "/api/v1/training/jobs", headers={**researcher_headers, "Idempotency-Key": "k-1"}, json=body
    ).json()["data"]
    second = client.post(
        "/api/v1/training/jobs", headers={**researcher_headers, "Idempotency-Key": "k-1"}, json=body
    ).json()["data"]
    assert first["job_id"] == second["job_id"]

    changed = _request(version_id, params={"epochs": 2, "batch": 2, "imgsz": 64, "seed": 42})
    conflict = client.post(
        "/api/v1/training/jobs", headers={**researcher_headers, "Idempotency-Key": "k-1"}, json=changed
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "idempotency_conflict"


def test_queue_limit_returns_429(client, researcher_headers, settings, monkeypatch):
    _, version_id = _frozen_dataset(client, researcher_headers)
    monkeypatch.setattr(settings, "inline_jobs", False)
    monkeypatch.setattr(settings, "max_pending_training_per_user", 1)
    first = client.post("/api/v1/training/jobs", headers=researcher_headers, json=_request(version_id))
    assert first.status_code == 202
    assert first.json()["data"]["status"] == "pending"
    second = client.post("/api/v1/training/jobs", headers=researcher_headers, json=_request(version_id))
    assert second.status_code == 429
    assert second.json()["code"] == "queue_limit"


def test_pending_job_can_be_cancelled(client, researcher_headers, settings, monkeypatch):
    _, version_id = _frozen_dataset(client, researcher_headers)
    monkeypatch.setattr(settings, "inline_jobs", False)
    job = client.post("/api/v1/training/jobs", headers=researcher_headers, json=_request(version_id)).json()["data"]
    cancelled = client.post(f"/api/v1/jobs/{job['job_id']}/cancel", headers=researcher_headers)
    assert cancelled.status_code == 202
    assert cancelled.json()["data"]["status"] == "cancelled"

    again = client.post(f"/api/v1/jobs/{job['job_id']}/cancel", headers=researcher_headers)
    assert again.status_code == 409
    assert again.json()["code"] == "job_finished"


def test_draft_version_cannot_train(client, researcher_headers):
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=3)
    response = client.post("/api/v1/training/preflight", headers=researcher_headers, json=_request(version_id))
    assert response.status_code == 200
    report = response.json()["data"]
    assert report["ok"] is False
    failed = [item["name"] for item in report["checks"] if item["status"] == "fail"]
    assert "dataset_version" in failed or "frozen" in failed

    job = client.post("/api/v1/training/jobs", headers=researcher_headers, json=_request(version_id))
    assert job.status_code == 422
    assert job.json()["code"] == "preflight_failed"


def test_cross_user_cannot_read_training_job(client, researcher_headers, db):
    from app.domains.auth_service import create_user

    _, version_id = _frozen_dataset(client, researcher_headers)
    job = client.post("/api/v1/training/jobs", headers=researcher_headers, json=_request(version_id)).json()["data"]
    create_user(db, username="other", password="other-password", role="researcher")
    token = client.post("/api/v1/auth/login", json={"username": "other", "password": "other-password"}).json()["data"]["token"]
    response = client.get(f"/api/v1/jobs/{job['job_id']}", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 404
