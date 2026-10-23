"""PR-08：复核回流、审批与测试集隔离。"""
from __future__ import annotations

import support


def _trained_and_inferred(client, headers):
    dataset_id, version_id = support.make_dataset(client, headers, count=12)
    frozen = support.freeze(client, headers, dataset_id, version_id=version_id).json()["data"]
    assert frozen["status"] == "completed", frozen
    job = client.post(
        "/api/v1/training/jobs",
        headers=headers,
        json={
            "dataset_version_id": version_id,
            "params": {"epochs": 1, "batch": 2, "imgsz": 64, "seed": 42, "deterministic": True, "device": "cpu"},
        },
    ).json()["data"]
    model_id = job["result"]["model_id"]

    assets = client.get(
        f"/api/v1/datasets/{dataset_id}/assets?version_id={version_id}&page_size=200", headers=headers
    ).json()["data"]["items"]
    train_asset = next(item for item in assets if item["split"] == "train")
    test_asset = next(item for item in assets if item["split"] == "test")

    response = client.post(
        "/api/v1/inference/images",
        headers=headers,
        data={"model_id": model_id, "asset_ids": str(train_asset["asset_id"])},
    )
    run_id = response.json()["data"]["run_id"]
    run = client.get(f"/api/v1/inference/runs/{run_id}", headers=headers).json()["data"]
    item = run["items"][0]
    return {
        "dataset_id": dataset_id,
        "version_id": version_id,
        "model_id": model_id,
        "train_asset": train_asset,
        "test_asset": test_asset,
        "item": item,
        "assets": assets,
    }


def _create_review(client, headers, item, verdict="missed_object", corrected=None):
    body = {
        "inference_item_id": item["item_id"],
        "image_asset_id": item["asset_id"],
        "reason": "人工复核",
        "items": [
            {
                "prediction_id": item["predictions"][0]["id"] if item["predictions"] else None,
                "verdict": verdict,
                "corrected": corrected or {"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
            }
        ],
    }
    return client.post("/api/v1/reviews", headers=headers, json=body)


def _approve(client, headers, review_id):
    assert client.post(f"/api/v1/reviews/{review_id}/submit", headers=headers).status_code == 200
    decision = client.post(
        f"/api/v1/reviews/{review_id}/decision", headers=headers, json={"approve": True}
    )
    assert decision.status_code == 200, decision.text
    return decision.json()["data"]


def test_review_lifecycle_and_export(client, researcher_headers, admin_headers, db, settings):
    ctx = _trained_and_inferred(client, researcher_headers)
    created = _create_review(client, researcher_headers, ctx["item"], verdict="false_positive")
    assert created.status_code == 201, created.text
    review = created.json()["data"]
    assert review["state"] == "draft"
    assert review["final_annotations"] == []

    submitted = client.post(f"/api/v1/reviews/{review['review_id']}/submit", headers=researcher_headers)
    assert submitted.status_code == 200
    assert submitted.json()["data"]["state"] == "submitted"

    approved = client.post(
        f"/api/v1/reviews/{review['review_id']}/decision", headers=admin_headers, json={"approve": True}
    )
    assert approved.status_code == 200
    assert approved.json()["data"]["state"] == "approved"

    export = client.post(
        "/api/v1/reviews/export-version",
        headers=researcher_headers,
        json={"parent_version_id": ctx["version_id"], "review_ids": [review["review_id"]]},
    )
    assert export.status_code == 202, export.text
    job = export.json()["data"]
    assert job["status"] == "completed", job
    child_id = job["result"]["child_version_id"]
    child = client.get(f"/api/v1/dataset-versions/{child_id}", headers=researcher_headers).json()["data"]
    assert child["state"] == "frozen"
    assert child["parent_version_id"] == ctx["version_id"]
    from pathlib import Path
    from app.db.models import DatasetVersion
    stored = db.get(DatasetVersion, child_id)
    parent_row = db.get(DatasetVersion, ctx["version_id"])
    assert stored.metadata_json["group_level"] == parent_row.metadata_json["group_level"]
    assert stored.split_config_json["group_level"] == parent_row.split_config_json["group_level"]
    expected_root = Path(settings.data_root) / "datasets" / ctx["dataset_id"] / child_id
    assert Path(stored.root_path).resolve() == expected_root.resolve()
    assert Path(stored.manifest_path).resolve() == (expected_root / "manifest.json").resolve()
    assert (expected_root / "manifest.json").is_file()
    # False-positive removal must also clear the actual YOLO sidecar.
    label = expected_root / "labels" / Path(ctx["train_asset"]["relative_path"]).with_suffix(".txt")
    assert label.read_text(encoding="utf-8") == ""

    # test/val 逐资产 hash 未变
    parent_assets = ctx["assets"]
    child_assets = client.get(
        f"/api/v1/datasets/{ctx['dataset_id']}/assets?version_id={child_id}&page_size=200",
        headers=researcher_headers,
    ).json()["data"]["items"]
    for split in ("val", "test"):
        before = {a["relative_path"]: a["sha256"] for a in parent_assets if a["split"] == split}
        after = {a["relative_path"]: a["sha256"] for a in child_assets if a["split"] == split}
        assert before == after

    train_before = next(a for a in parent_assets if a["asset_id"] == ctx["train_asset"]["asset_id"])
    train_after = next(a for a in child_assets if a["relative_path"] == train_before["relative_path"])
    assert train_after["annotations"] == []


def test_duplicate_export_is_idempotent(client, researcher_headers, admin_headers):
    ctx = _trained_and_inferred(client, researcher_headers)
    review = _create_review(client, researcher_headers, ctx["item"]).json()["data"]
    _approve(client, admin_headers, review["review_id"])
    body = {"parent_version_id": ctx["version_id"], "review_ids": [review["review_id"]]}
    first = client.post("/api/v1/reviews/export-version", headers=researcher_headers, json=body).json()["data"]
    second = client.post("/api/v1/reviews/export-version", headers=researcher_headers, json=body).json()["data"]
    assert first["result"]["child_version_id"] == second["result"]["child_version_id"]


def test_test_asset_cannot_flow_back(client, researcher_headers):
    ctx = _trained_and_inferred(client, researcher_headers)
    test_item = client.post(
        "/api/v1/inference/images",
        headers=researcher_headers,
        data={"model_id": ctx["model_id"], "asset_ids": str(ctx["test_asset"]["asset_id"])},
    ).json()["data"]["run_id"]
    run = client.get(f"/api/v1/inference/runs/{test_item}", headers=researcher_headers).json()["data"]
    review = _create_review(client, researcher_headers, run["items"][0]).json()["data"]
    assert review["state"] == "draft"
    export = client.post(
        "/api/v1/reviews/export-version",
        headers=researcher_headers,
        json={"parent_version_id": ctx["version_id"], "review_ids": [review["review_id"]]},
    )
    job = export.json()["data"]
    assert job["status"] == "failed"
    assert job["error"]["code"] in {"review_not_approved", "test_leak_blocked"}


def test_decision_requires_reviewer(client, researcher_headers):
    ctx = _trained_and_inferred(client, researcher_headers)
    review = _create_review(client, researcher_headers, ctx["item"]).json()["data"]
    client.post(f"/api/v1/reviews/{review['review_id']}/submit", headers=researcher_headers)
    response = client.post(
        f"/api/v1/reviews/{review['review_id']}/decision", headers=researcher_headers, json={"approve": True}
    )
    assert response.status_code == 403


def test_patch_conflict_returns_server_revision(client, researcher_headers):
    ctx = _trained_and_inferred(client, researcher_headers)
    review = _create_review(client, researcher_headers, ctx["item"]).json()["data"]
    body = {
        "expected_revision": review["revision"],
        "reason": "更新",
        "items": [{"verdict": "unreviewable"}],
    }
    first = client.patch(f"/api/v1/reviews/{review['review_id']}", headers=researcher_headers, json=body)
    assert first.status_code == 200, first.text
    stale = client.patch(f"/api/v1/reviews/{review['review_id']}", headers=researcher_headers, json=body)
    assert stale.status_code == 409
    assert stale.json()["data"]["server_revision"] == first.json()["data"]["revision"]


def test_unreviewable_asset_is_excluded_from_child(client, researcher_headers, admin_headers):
    ctx = _trained_and_inferred(client, researcher_headers)
    review = _create_review(
        client, researcher_headers, ctx["item"], verdict="unreviewable"
    ).json()["data"]
    _approve(client, admin_headers, review["review_id"])
    job = client.post(
        "/api/v1/reviews/export-version",
        headers=researcher_headers,
        json={"parent_version_id": ctx["version_id"], "review_ids": [review["review_id"]]},
    ).json()["data"]
    assert job["status"] == "completed", job
    child_assets = client.get(
        f"/api/v1/datasets/{ctx['dataset_id']}/assets?version_id={job['result']['child_version_id']}&page_size=200",
        headers=researcher_headers,
    ).json()["data"]["items"]
    assert ctx["train_asset"]["relative_path"] not in {item["relative_path"] for item in child_assets}


def test_feedback_child_inherits_image_cluster_declaration(client, researcher_headers, admin_headers, db):
    """A grouped parent must stay grouped through review -> child version."""
    import json

    from app.db.models import DatasetVersion

    files = []
    groups = {}
    for index in range(12):
        name = f"cluster-{index:02d}.jpg"
        files.append((name, support.image_bytes((index * 13, index * 7, index * 3))))
        files.append(
            (f"cluster-{index:02d}.txt", support.yolo_txt([(c, 0.5, 0.5, 0.3, 0.3) for c in (0, 1, 2)]).encode())
        )
        groups[name] = "cluster-pair" if index < 2 else f"cluster-{index}"
    job = support.import_dataset(
        client,
        researcher_headers,
        files=files,
        form={"group_level": "image_cluster", "source_groups": json.dumps(groups)},
    ).json()["data"]
    assert job["status"] == "completed", job
    dataset_id, version_id = job["result"]["dataset_id"], job["result"]["version_id"]
    assert support.freeze(client, researcher_headers, dataset_id, version_id=version_id).json()["data"]["status"] == "completed"

    trained = client.post(
        "/api/v1/training/jobs",
        headers=researcher_headers,
        json={
            "dataset_version_id": version_id,
            "params": {"epochs": 1, "batch": 2, "imgsz": 64, "seed": 42, "deterministic": True, "device": "cpu"},
        },
    ).json()["data"]
    assert trained["status"] == "completed", trained
    model_id = trained["result"]["model_id"]

    assets = client.get(
        f"/api/v1/datasets/{dataset_id}/assets?version_id={version_id}&page_size=200",
        headers=researcher_headers,
    ).json()["data"]["items"]
    train_asset = next(item for item in assets if item["split"] == "train")
    run_id = client.post(
        "/api/v1/inference/images",
        headers=researcher_headers,
        data={"model_id": model_id, "asset_ids": str(train_asset["asset_id"])},
    ).json()["data"]["run_id"]
    item = client.get(f"/api/v1/inference/runs/{run_id}", headers=researcher_headers).json()["data"]["items"][0]
    review = _create_review(client, researcher_headers, item, verdict="missed_object").json()["data"]
    _approve(client, admin_headers, review["review_id"])

    exported = client.post(
        "/api/v1/reviews/export-version",
        headers=researcher_headers,
        json={"parent_version_id": version_id, "review_ids": [review["review_id"]]},
    ).json()["data"]
    assert exported["status"] == "completed", exported
    child = db.get(DatasetVersion, exported["result"]["child_version_id"])
    assert child.metadata_json["group_level"] == "image_cluster"
    assert child.split_config_json["group_level"] == "image_cluster"
    manifest = json.loads((__import__("pathlib").Path(child.root_path) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["group_level"] == "image_cluster"
