"""PR-02：导入、不可变版本、分页与引用保护。"""
from __future__ import annotations

import support


def test_zero_split_seed_reaches_frozen_manifest(client, researcher_headers):
    dataset, version = support.make_dataset(client, researcher_headers, count=12)
    response = support.freeze(client, researcher_headers, dataset, version_id=version, seed=0)
    assert response.status_code == 202, response.text
    assert response.json()['data']['status'] == 'completed'
    frozen = client.get('/api/v1/dataset-versions/'+version, headers=researcher_headers).json()['data']
    assert frozen['split_config']['seed'] == 0


def test_admin_can_restore_deleted_draft_asset(client, researcher_headers, admin_headers):
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=3)
    asset = client.get(f"/api/v1/datasets/{dataset_id}/assets?version_id={version_id}", headers=researcher_headers).json()["data"]["items"][0]
    route = f"/api/v1/assets/{asset['asset_id']}"
    assert client.delete(route, headers=researcher_headers).status_code == 200
    assert client.get(route + "/image", headers=researcher_headers).status_code == 404
    assert client.post(route + "/restore", headers=researcher_headers).status_code == 403
    assert client.post(route + "/restore", headers=admin_headers).status_code == 200
    assert client.get(route + "/image", headers=researcher_headers).status_code == 200


def test_freeze_rejects_missing_root_without_writing_cwd(client, researcher_headers, db):
    from app.db.models import DatasetVersion
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=3)
    version = db.get(DatasetVersion, version_id)
    version.root_path = ""
    db.commit()
    result = support.freeze(client, researcher_headers, dataset_id, version_id=version_id).json()["data"]
    assert result["status"] == "failed"
    assert result["error"]["code"] == "invalid_version_root"


def test_freeze_io_failure_is_retryable(client, researcher_headers, db, monkeypatch):
    from app.db.models import DatasetVersion
    from app.domains.datasets import version_service
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=6)
    original_write = version_service.atomic_write_bytes
    def disk_failure(*args, **kwargs):
        raise OSError("simulated disk full")
    monkeypatch.setattr(version_service, "atomic_write_bytes", disk_failure)
    result = support.freeze(client, researcher_headers, dataset_id, version_id=version_id).json()["data"]
    assert result["status"] == "failed"
    db.expire_all()
    assert db.get(DatasetVersion, version_id).state == "failed"
    monkeypatch.setattr(version_service, "atomic_write_bytes", original_write)
    retried = support.freeze(client, researcher_headers, dataset_id, version_id=version_id).json()["data"]
    assert retried["status"] == "completed", retried


def test_import_creates_draft_version_with_annotations(client, researcher_headers):
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=3)
    detail = client.get(f"/api/v1/datasets/{dataset_id}", headers=researcher_headers).json()["data"]
    assert detail["latest_version"]["state"] == "draft"
    assert detail["latest_version"]["asset_count"] == 3
    assert detail["owner"] == "researcher1"

    assets = client.get(
        f"/api/v1/datasets/{dataset_id}/assets?version_id={version_id}", headers=researcher_headers
    ).json()["data"]
    assert assets["total"] == 3
    assert assets["page_size"] == 50
    first = assets["items"][0]
    assert len(first["annotations"]) == 3
    assert 0 < first["annotations"][0]["cx"] < 1


def test_duplicate_content_is_deduplicated(client, researcher_headers):
    payload = support.image_bytes((10, 20, 30))
    response = support.import_dataset(
        client,
        researcher_headers,
        files=[
            ("a.jpg", payload),
            ("copy-of-a.jpg", payload),
            ("b.jpg", support.image_bytes((30, 20, 10))),
            ("a.txt", support.yolo_txt([(0, 0.5, 0.5, 0.2, 0.2)]).encode()),
        ],
    )
    job = response.json()["data"]
    assert job["status"] == "completed", job
    report = job["result"]["report"]
    assert report["counts"]["images_imported"] == 2
    assert len(report["duplicates"]) == 1
    assert report["duplicates"][0]["duplicate_of"] == "a.jpg"


def test_freeze_produces_manifest_and_split_stats(client, researcher_headers):
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=9)
    response = support.freeze(client, researcher_headers, dataset_id, version_id=version_id)
    job = response.json()["data"]
    assert job["status"] == "completed", job
    result = job["result"]
    assert result["state"] == "frozen"
    assert result["asset_count"] == 9
    assert sum(result["split_counts"].values()) == 9
    assert all(result["split_counts"][split] >= 1 for split in ("train", "val", "test"))
    assert len(result["manifest_sha256"]) == 64

    version = client.get(f"/api/v1/dataset-versions/{version_id}", headers=researcher_headers).json()["data"]
    assert version["state"] == "frozen"
    assert version["manifest_sha256"] == result["manifest_sha256"]
    assert version["class_map"] == {"0": "RBC", "1": "WBC", "2": "Platelets"}


def test_same_seed_same_split_hash(client, researcher_headers):
    hashes = []
    for _ in range(2):
        dataset_id, version_id = support.make_dataset(client, researcher_headers, count=9)
        job = support.freeze(client, researcher_headers, dataset_id, version_id=version_id, seed=42).json()["data"]
        assert job["status"] == "completed", job
        hashes.append(job["result"]["split_manifest_hash"])
    assert hashes[0] == hashes[1]


def test_frozen_version_rejects_mutation(client, researcher_headers):
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=6)
    assert support.freeze(client, researcher_headers, dataset_id, version_id=version_id).json()["data"]["status"] == "completed"

    refreeze = support.freeze(client, researcher_headers, dataset_id, version_id=version_id)
    assert refreeze.status_code == 202
    assert refreeze.json()["data"]["status"] == "failed"
    assert refreeze.json()["data"]["error"]["code"] == "version_frozen"

    assets = client.get(
        f"/api/v1/datasets/{dataset_id}/assets?version_id={version_id}", headers=researcher_headers
    ).json()["data"]
    delete = client.delete(f"/api/v1/assets/{assets['items'][0]['asset_id']}", headers=researcher_headers)
    assert delete.status_code == 409
    assert delete.json()["code"] == "version_frozen"


def test_delete_referenced_dataset_returns_409(client, researcher_headers, db):
    from app.db.models import ModelRegistry

    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=3)
    db.add(
        ModelRegistry(
            name="referenced.pt",
            path="/tmp/referenced.pt",
            sha256="0" * 64,
            source="training",
            dataset_version_id=version_id,
            class_map_json={"0": "RBC"},
        )
    )
    db.commit()
    response = client.delete(f"/api/v1/datasets/{dataset_id}", headers=researcher_headers)
    assert response.status_code == 409
    assert response.json()["code"] == "resource_referenced"


def test_asset_pagination_bounds(client, researcher_headers):
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=3)
    ok = client.get(
        f"/api/v1/datasets/{dataset_id}/assets?version_id={version_id}&page_size=200",
        headers=researcher_headers,
    )
    assert ok.status_code == 200
    over = client.get(
        f"/api/v1/datasets/{dataset_id}/assets?version_id={version_id}&page_size=201",
        headers=researcher_headers,
    )
    assert over.status_code == 422


def test_cross_user_access_is_404(client, researcher_headers, admin_headers, db):
    from app.domains.auth_service import create_user

    dataset_id, _ = support.make_dataset(client, researcher_headers, count=3)
    create_user(db, username="researcher2", password="researcher-password", role="researcher")
    login = client.post(
        "/api/v1/auth/login", json={"username": "researcher2", "password": "researcher-password"}
    ).json()["data"]["token"]
    other = {"Authorization": f"Bearer {login}"}
    assert client.get(f"/api/v1/datasets/{dataset_id}", headers=other).status_code == 404
    assert client.get(f"/api/v1/datasets/{dataset_id}", headers=admin_headers).status_code == 200


def test_annotation_fork_preserves_frozen_parent(client, researcher_headers):
    dataset, parent = support.make_dataset(client, researcher_headers, count=3)
    assert support.freeze(client, researcher_headers, dataset, version_id=parent).json()['data']['status'] == 'completed'
    url = f'/api/v1/datasets/{dataset}/assets'
    original = client.get(url, params={'version_id': parent}, headers=researcher_headers).json()['data']['items'][0]
    response = client.post(f'/api/v1/datasets/{dataset}/annotation-versions', json={'base_version_id': parent}, headers=researcher_headers)
    assert response.status_code == 200, response.text
    child = response.json()['data']
    assert child['state'] == 'draft' and child['parent_version_id'] == parent
    asset = client.get(url, params={'version_id': child['version_id']}, headers=researcher_headers).json()['data']['items'][0]
    assert asset['asset_id'] != original['asset_id']
    assert asset['annotations'] == original['annotations']
    assert asset['source_group'] == original['source_group']
    assert client.get(f"/api/v1/assets/{asset['asset_id']}/image", headers=researcher_headers).status_code == 200
    result = client.put(f"/api/v1/annotations/{asset['asset_id']}", json={'expected_revision': 0, 'boxes': []}, headers=researcher_headers)
    assert result.status_code == 200, result.text
    before = client.get(f"/api/v1/annotations/{original['asset_id']}", headers=researcher_headers).json()['data']
    assert before['annotations'] == original['annotations'] and before['editable'] is False
    assert client.put(f"/api/v1/annotations/{original['asset_id']}", json={'expected_revision': 0, 'boxes': []}, headers=researcher_headers).status_code == 409


def test_annotation_fork_rejects_wrong_dataset_and_draft(client, researcher_headers):
    dataset, parent = support.make_dataset(client, researcher_headers, count=3)
    route = f'/api/v1/datasets/{dataset}/annotation-versions'
    assert client.post(route, json={'base_version_id': parent}, headers=researcher_headers).status_code == 409
    other, _ = support.make_dataset(client, researcher_headers, count=2)
    assert client.post(f'/api/v1/datasets/{other}/annotation-versions', json={'base_version_id': parent}, headers=researcher_headers).status_code == 404
