"""PR-03：标注修订 CAS、分页草稿与只读保护。"""
from __future__ import annotations

import support


def _first_asset(client, headers, dataset_id, version_id) -> dict:
    data = client.get(
        f"/api/v1/datasets/{dataset_id}/assets?version_id={version_id}", headers=headers
    ).json()["data"]
    return data["items"][0]


def _box(class_id: int = 1, cx: float = 0.4) -> dict:
    return {"client_id": f"c-{class_id}", "class_id": class_id, "cx": cx, "cy": 0.5, "w": 0.2, "h": 0.2}


def test_get_returns_imported_annotations_and_zero_revision(client, researcher_headers):
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=3)
    asset = _first_asset(client, researcher_headers, dataset_id, version_id)
    state = client.get(f"/api/v1/annotations/{asset['asset_id']}", headers=researcher_headers).json()["data"]
    assert state["revision"] == 0
    assert len(state["annotations"]) == 3
    assert state["editable"] is True
    assert state["class_map"]["0"] == "RBC"


def test_save_increments_revision_and_persists(client, researcher_headers):
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=3)
    asset = _first_asset(client, researcher_headers, dataset_id, version_id)
    response = client.put(
        f"/api/v1/annotations/{asset['asset_id']}",
        headers=researcher_headers,
        json={"expected_revision": 0, "boxes": [_box(1)]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["data"]["revision"] == 1

    state = client.get(f"/api/v1/annotations/{asset['asset_id']}", headers=researcher_headers).json()["data"]
    assert state["revision"] == 1
    assert len(state["annotations"]) == 1
    assert state["annotations"][0]["class_id"] == 1


def test_stale_revision_returns_409_with_server_revision(client, researcher_headers):
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=3)
    asset = _first_asset(client, researcher_headers, dataset_id, version_id)
    first = client.put(
        f"/api/v1/annotations/{asset['asset_id']}",
        headers=researcher_headers,
        json={"expected_revision": 0, "boxes": [_box(0)]},
    )
    assert first.status_code == 200
    stale = client.put(
        f"/api/v1/annotations/{asset['asset_id']}",
        headers=researcher_headers,
        json={"expected_revision": 0, "boxes": [_box(2)]},
    )
    assert stale.status_code == 409
    body = stale.json()
    assert body["code"] == "revision_conflict"
    assert body["data"]["server_revision"] == 1

    state = client.get(f"/api/v1/annotations/{asset['asset_id']}", headers=researcher_headers).json()["data"]
    assert state["annotations"][0]["class_id"] == 0


def test_empty_boxes_and_tombstones(client, researcher_headers):
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=3)
    asset = _first_asset(client, researcher_headers, dataset_id, version_id)
    deleted = client.put(
        f"/api/v1/annotations/{asset['asset_id']}",
        headers=researcher_headers,
        json={"expected_revision": 0, "boxes": [], "deleted_ids": ["c-0", "c-1"]},
    )
    assert deleted.status_code == 200
    state = client.get(f"/api/v1/annotations/{asset['asset_id']}", headers=researcher_headers).json()["data"]
    assert state["annotations"] == []
    assert state["deleted_ids"] == ["c-0", "c-1"]

    restored = client.put(
        f"/api/v1/annotations/{asset['asset_id']}",
        headers=researcher_headers,
        json={"expected_revision": 1, "boxes": [_box(0)], "deleted_ids": []},
    )
    assert restored.status_code == 200
    assert restored.json()["data"]["revision"] == 2


def test_saving_one_page_keeps_other_pages(client, researcher_headers):
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=3)
    assets = client.get(
        f"/api/v1/datasets/{dataset_id}/assets?version_id={version_id}", headers=researcher_headers
    ).json()["data"]["items"]
    first, second = assets[0], assets[1]
    first_save = client.put(
        f"/api/v1/annotations/{first['asset_id']}",
        headers=researcher_headers,
        json={"expected_revision": 0, "boxes": [_box(0)], "page": 1},
    )
    assert first_save.status_code == 200
    revision = first_save.json()["data"]["revision"]
    second_save = client.put(
        f"/api/v1/annotations/{second['asset_id']}",
        headers=researcher_headers,
        json={"expected_revision": revision, "boxes": [_box(2)], "page": 2},
    )
    assert second_save.status_code == 200

    first_state = client.get(f"/api/v1/annotations/{first['asset_id']}", headers=researcher_headers).json()["data"]
    second_state = client.get(f"/api/v1/annotations/{second['asset_id']}", headers=researcher_headers).json()["data"]
    assert first_state["annotations"][0]["class_id"] == 0
    assert second_state["annotations"][0]["class_id"] == 2


def test_frozen_version_is_read_only(client, researcher_headers):
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=6)
    assert support.freeze(client, researcher_headers, dataset_id, version_id=version_id).json()["data"]["status"] == "completed"
    asset = _first_asset(client, researcher_headers, dataset_id, version_id)
    state = client.get(f"/api/v1/annotations/{asset['asset_id']}", headers=researcher_headers).json()["data"]
    assert state["editable"] is False
    response = client.put(
        f"/api/v1/annotations/{asset['asset_id']}",
        headers=researcher_headers,
        json={"expected_revision": state["revision"], "boxes": [_box(0)]},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "version_frozen"


def test_invalid_box_is_422(client, researcher_headers):
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=3)
    asset = _first_asset(client, researcher_headers, dataset_id, version_id)
    bad = {"class_id": 0, "cx": 0.95, "cy": 0.5, "w": 0.5, "h": 0.5}
    response = client.put(
        f"/api/v1/annotations/{asset['asset_id']}",
        headers=researcher_headers,
        json={"expected_revision": 0, "boxes": [bad]},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_box"

    unknown = {"class_id": 99, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
    response = client.put(
        f"/api/v1/annotations/{asset['asset_id']}",
        headers=researcher_headers,
        json={"expected_revision": 0, "boxes": [unknown]},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "unknown_class"


def test_cross_user_cannot_read_annotations(client, researcher_headers, db):
    from app.domains.auth_service import create_user

    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=3)
    asset = _first_asset(client, researcher_headers, dataset_id, version_id)
    create_user(db, username="intruder", password="intruder-password", role="researcher")
    token = client.post(
        "/api/v1/auth/login", json={"username": "intruder", "password": "intruder-password"}
    ).json()["data"]["token"]
    response = client.get(
        f"/api/v1/annotations/{asset['asset_id']}", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 404


def test_export_returns_job(client, researcher_headers):
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=3)
    response = client.post(
        "/api/v1/annotations/export",
        headers=researcher_headers,
        json={"dataset_id": dataset_id, "version_id": version_id, "format": "yolo-txt"},
    )
    job = response.json()["data"]
    assert response.status_code == 202
    assert job["status"] == "completed"
    assert job["result"]["annotation_count"] == 9
