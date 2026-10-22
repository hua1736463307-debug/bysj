"""PR-06：推理持久化、部分失败、导出与越权。"""
from __future__ import annotations

import json

import support


def _trained_model(client, headers):
    dataset_id, version_id = support.make_dataset(client, headers, count=9)
    assert support.freeze(client, headers, dataset_id, version_id=version_id).json()["data"]["status"] == "completed"
    job = client.post(
        "/api/v1/training/jobs",
        headers=headers,
        json={
            "dataset_version_id": version_id,
            "params": {"epochs": 1, "batch": 2, "imgsz": 64, "seed": 42, "deterministic": True, "device": "cpu"},
        },
    ).json()["data"]
    assert job["status"] == "completed", job
    return job["result"]["model_id"], dataset_id, version_id


def test_single_image_inference_persists_predictions(client, researcher_headers):
    model_id, _, _ = _trained_model(client, researcher_headers)
    response = client.post(
        "/api/v1/inference/images",
        headers=researcher_headers,
        data={"model_id": model_id, "params": '{"confidence": 0.4}'},
        files=[("files", ("scan.jpg", support.image_bytes((12, 34, 56), size=(80, 60)), "image/jpeg"))],
    )
    assert response.status_code == 202, response.text
    data = response.json()["data"]
    run_id = data["run_id"]
    run = client.get(f"/api/v1/inference/runs/{run_id}", headers=researcher_headers).json()["data"]
    assert run["status"] == "completed"
    assert run["completed"] == 1
    item = run["items"][0]
    assert item["status"] == "completed"
    assert item["width"] == 80 and item["height"] == 60
    prediction = item["predictions"][0]
    assert prediction["class_name"] == "RBC"
    assert 0 <= prediction["bbox_xyxy"][0] < 80
    assert item["artifact_id"]
    artifact = client.get(f"/api/v1/artifacts/{item['artifact_id']}", headers=researcher_headers)
    assert artifact.status_code == 200
    assert artifact.headers["content-type"].startswith("image/")


def test_batch_with_one_broken_image_keeps_successes(client, researcher_headers):
    model_id, _, _ = _trained_model(client, researcher_headers)
    response = client.post(
        "/api/v1/inference/images",
        headers=researcher_headers,
        data={"model_id": model_id},
        files=[
            ("files", ("a.jpg", support.image_bytes((1, 1, 1)), "image/jpeg")),
            ("files", ("b.jpg", b"broken-bytes", "image/jpeg")),
            ("files", ("c.jpg", support.image_bytes((2, 2, 2)), "image/jpeg")),
        ],
    )
    run_id = response.json()["data"]["run_id"]
    run = client.get(f"/api/v1/inference/runs/{run_id}", headers=researcher_headers).json()["data"]
    assert run["status"] == "partial"
    assert run["completed"] == 2
    assert run["failed"] == 1
    statuses = {item["status"] for item in run["items"]}
    assert statuses == {"completed", "failed"}
    # JobTask 仍是 completed，不新增终态枚举
    assert client.get(f"/api/v1/jobs/{response.json()['data']['job']['job_id']}", headers=researcher_headers).json()["data"]["status"] == "completed"


def test_inference_by_asset_id(client, researcher_headers):
    model_id, dataset_id, version_id = _trained_model(client, researcher_headers)
    assets = client.get(
        f"/api/v1/datasets/{dataset_id}/assets?version_id={version_id}", headers=researcher_headers
    ).json()["data"]["items"][:2]
    ids = ",".join(str(item["asset_id"]) for item in assets)
    response = client.post(
        "/api/v1/inference/images",
        headers=researcher_headers,
        data={"model_id": model_id, "asset_ids": ids},
    )
    run_id = response.json()["data"]["run_id"]
    run = client.get(f"/api/v1/inference/runs/{run_id}", headers=researcher_headers).json()["data"]
    assert run["completed"] == 2
    assert all(item["asset_id"] for item in run["items"])


def test_export_json_and_csv(client, researcher_headers):
    model_id, _, _ = _trained_model(client, researcher_headers)
    response = client.post(
        "/api/v1/inference/images",
        headers=researcher_headers,
        data={"model_id": model_id},
        files=[("files", ("a.jpg", support.image_bytes((3, 4, 5)), "image/jpeg"))],
    )
    run_id = response.json()["data"]["run_id"]
    exported = client.get(f"/api/v1/inference/runs/{run_id}/export?format=json", headers=researcher_headers)
    payload = json.loads(exported.content)
    assert payload["schema_version"] == 1
    assert payload["run"]["model_sha256"]
    assert payload["items"][0]["predictions"]

    csv_response = client.get(f"/api/v1/inference/runs/{run_id}/export?format=csv", headers=researcher_headers)
    assert csv_response.status_code == 200
    header = csv_response.content.decode().splitlines()[0]
    assert "class_id" in header and "x1" in header


def test_batch_limit_and_missing_model(client, researcher_headers, settings, monkeypatch):
    model_id, _, _ = _trained_model(client, researcher_headers)
    monkeypatch.setattr(settings, "max_batch_inference", 1)
    response = client.post(
        "/api/v1/inference/images",
        headers=researcher_headers,
        data={"model_id": model_id},
        files=[
            ("files", ("a.jpg", support.image_bytes((1, 1, 1)), "image/jpeg")),
            ("files", ("b.jpg", support.image_bytes((2, 2, 2)), "image/jpeg")),
        ],
    )
    assert response.status_code == 422
    assert response.json()["code"] == "batch_too_large"

    missing = client.post(
        "/api/v1/inference/images",
        headers=researcher_headers,
        data={"model_id": "00000000-0000-0000-0000-000000000000"},
        files=[("files", ("a.jpg", support.image_bytes((1, 1, 1)), "image/jpeg"))],
    )
    assert missing.status_code == 404


def test_cross_user_cannot_read_run_or_artifact(client, researcher_headers, db):
    from app.domains.auth_service import create_user

    model_id, _, _ = _trained_model(client, researcher_headers)
    response = client.post(
        "/api/v1/inference/images",
        headers=researcher_headers,
        data={"model_id": model_id},
        files=[("files", ("a.jpg", support.image_bytes((6, 6, 6)), "image/jpeg"))],
    )
    run_id = response.json()["data"]["run_id"]
    item = client.get(f"/api/v1/inference/runs/{run_id}", headers=researcher_headers).json()["data"]["items"][0]

    create_user(db, username="spy", password="spy-password", role="researcher")
    token = client.post("/api/v1/auth/login", json={"username": "spy", "password": "spy-password"}).json()["data"]["token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get(f"/api/v1/inference/runs/{run_id}", headers=headers).status_code == 404
    assert client.get(f"/api/v1/artifacts/{item['artifact_id']}", headers=headers).status_code == 404
