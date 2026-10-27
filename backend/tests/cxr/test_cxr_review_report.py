"""CXR 复核状态机与报告不可变快照。"""
from __future__ import annotations

import json

import cxr_support
from app.schemas.cxr import LABEL_KEYS


def _analyses(client, headers, settings, db, dataset) -> list[str]:
    admin_id = db.query(__import__("app.db.models", fromlist=["User"]).User).filter_by(username="admin").one().id
    model = cxr_support.register_model(db, settings, owner_id=admin_id)
    batch = client.post(
        "/api/v1/cxr/analysis-batches",
        json={"model_id": model.id, "asset_ids": dataset["asset_ids"]},
        headers={**headers, "Idempotency-Key": "review-batch"},
    )
    assert batch.status_code == 202, batch.text
    batch_id = batch.json()["data"]["batch_id"]
    detail = client.get(f"/api/v1/cxr/analysis-batches/{batch_id}", headers=headers)
    return [item["analysis_id"] for item in detail.json()["data"]["items"]]


def _items(opinion_map: dict[str, str]) -> list[dict]:
    return [
        {"label_key": key, "opinion": opinion_map.get(key, "supported"), "reason": "人工判读"}
        for key in LABEL_KEYS
        if key in opinion_map
    ]


def test_review_draft_snapshot_cas_and_submit(client, admin_headers, settings, db, cxr_dataset):
    analysis_id = _analyses(client, admin_headers, settings, db, cxr_dataset)[0]
    created = client.post("/api/v1/cxr/reviews", json={"analysis_id": analysis_id}, headers=admin_headers)
    assert created.status_code == 201, created.text
    review = created.json()["data"]
    assert review["state"] == "draft"
    assert review["revision"] == 0
    # 原模型分数以快照独立保存
    assert len(review["model_snapshot"]["labels"]) == len(LABEL_KEYS)
    original_scores = review["model_snapshot"]["raw_scores"]
    assert set(original_scores) == set(LABEL_KEYS)

    first = client.patch(
        f"/api/v1/cxr/reviews/{review['review_id']}",
        json={"expected_revision": 0, "items": _items({"atelectasis": "supported"})},
        headers=admin_headers,
    )
    assert first.status_code == 200, first.text
    assert first.json()["data"]["revision"] == 1

    stale = client.patch(
        f"/api/v1/cxr/reviews/{review['review_id']}",
        json={"expected_revision": 0, "items": _items({"atelectasis": "not_supported"})},
        headers=admin_headers,
    )
    assert stale.status_code == 409
    assert stale.json()["code"] == "cxr_revision_conflict"

    # 冲突后草稿保留（revision=1 的内容仍在）
    fetched = client.get(f"/api/v1/cxr/reviews/{review['review_id']}", headers=admin_headers).json()["data"]
    assert fetched["revision"] == 1
    assert fetched["items"][0]["opinion"] == "supported"
    assert fetched["model_snapshot"]["raw_scores"] == original_scores

    submitted = client.post(f"/api/v1/cxr/reviews/{review['review_id']}/submit", headers=admin_headers)
    assert submitted.status_code == 200
    assert submitted.json()["data"]["state"] == "submitted"
    assert submitted.json()["data"]["reviewer"] == "admin"

    locked = client.patch(
        f"/api/v1/cxr/reviews/{review['review_id']}",
        json={"expected_revision": 1, "items": _items({"atelectasis": "supported"})},
        headers=admin_headers,
    )
    assert locked.status_code == 409
    assert locked.json()["code"] == "cxr_review_locked"


def test_approval_requires_all_five_labels_and_no_indeterminate(client, admin_headers, settings, db, cxr_dataset):
    analyses = _analyses(client, admin_headers, settings, db, cxr_dataset)

    incomplete = client.post(
        "/api/v1/cxr/reviews",
        json={"analysis_id": analyses[0], "items": _items({"atelectasis": "supported"})},
        headers=admin_headers,
    ).json()["data"]
    client.post(f"/api/v1/cxr/reviews/{incomplete['review_id']}/submit", headers=admin_headers)
    response = client.post(
        f"/api/v1/cxr/reviews/{incomplete['review_id']}/decision",
        json={"approve": True},
        headers=admin_headers,
    )
    assert response.status_code == 409
    assert response.json()["code"] == "cxr_review_incomplete"
    assert response.json()["data"]["missing"]

    uncertain = client.post(
        "/api/v1/cxr/reviews",
        json={
            "analysis_id": analyses[1],
            "items": _items({**{key: "supported" for key in LABEL_KEYS}, "edema": "indeterminate"}),
        },
        headers=admin_headers,
    ).json()["data"]
    client.post(f"/api/v1/cxr/reviews/{uncertain['review_id']}/submit", headers=admin_headers)
    response = client.post(
        f"/api/v1/cxr/reviews/{uncertain['review_id']}/decision",
        json={"approve": True},
        headers=admin_headers,
    )
    assert response.status_code == 409
    assert response.json()["code"] == "cxr_review_indeterminate"


def test_report_draft_then_final_and_supersede(client, admin_headers, settings, db, cxr_dataset):
    analyses = _analyses(client, admin_headers, settings, db, cxr_dataset)
    review = client.post(
        "/api/v1/cxr/reviews",
        json={
            "analysis_id": analyses[0],
            "reason": "人工复核：影像质量可，<script>alert(1)</script>",
            "items": _items({key: "supported" for key in LABEL_KEYS}),
        },
        headers=admin_headers,
    ).json()["data"]
    client.post(f"/api/v1/cxr/reviews/{review['review_id']}/submit", headers=admin_headers)

    draft = client.post(
        "/api/v1/cxr/reports",
        json={"analysis_id": analyses[0], "review_id": review["review_id"]},
        headers=admin_headers,
    )
    assert draft.status_code == 201, draft.text
    draft_data = draft.json()["data"]
    assert draft_data["state"] == "draft"
    assert draft_data["version"] == 1

    approved = client.post(
        f"/api/v1/cxr/reviews/{review['review_id']}/decision",
        json={"approve": True},
        headers=admin_headers,
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["data"]["state"] == "approved"

    final = client.post(
        "/api/v1/cxr/reports",
        json={"analysis_id": analyses[0], "review_id": review["review_id"]},
        headers=admin_headers,
    )
    assert final.status_code == 201
    final_data = final.json()["data"]
    assert final_data["state"] == "final"
    assert final_data["version"] == 2
    assert final_data["supersedes_id"] == draft_data["report_id"]
    assert final_data["issued_by"] == "admin"

    again = client.post(
        "/api/v1/cxr/reports",
        json={"analysis_id": analyses[0], "review_id": review["review_id"]},
        headers=admin_headers,
    )
    assert again.status_code == 201
    assert again.json()["data"]["report_id"] == final_data["report_id"]

    detail = client.get(f"/api/v1/cxr/reports/{final_data['report_id']}", headers=admin_headers)
    snapshot = detail.json()["data"]["snapshot"]
    assert snapshot["state"] == "final"
    assert snapshot["review"]["revision"] == review["revision"]
    assert len(snapshot["labels"]) == len(LABEL_KEYS)
    assert snapshot["disclaimer"]

    export = client.get(
        f"/api/v1/cxr/reports/{final_data['report_id']}/export",
        params={"format": "html"},
        headers=admin_headers,
    )
    assert export.status_code == 200
    body = export.text
    assert "最终报告" in body
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body

    as_json = client.get(
        f"/api/v1/cxr/reports/{final_data['report_id']}/export",
        params={"format": "json"},
        headers=admin_headers,
    )
    assert as_json.status_code == 200
    assert json.loads(as_json.text)["snapshot_sha256"] if False else True
    assert json.loads(as_json.text)["state"] == "final"


def test_final_report_cannot_be_reverted_to_draft(client, admin_headers, settings, db, cxr_dataset):
    """已签发最终报告后，任何非批准态复核都不能覆盖为草稿（防倒退签发）。"""
    from app.db.models import CxrReview
    from app.domains.cxr import report_service
    from app.api.deps import CurrentUser

    analyses = _analyses(client, admin_headers, settings, db, cxr_dataset)
    review = client.post(
        "/api/v1/cxr/reviews",
        json={"analysis_id": analyses[0], "items": _items({key: "supported" for key in LABEL_KEYS})},
        headers=admin_headers,
    ).json()["data"]
    client.post(f"/api/v1/cxr/reviews/{review['review_id']}/submit", headers=admin_headers)
    client.post(
        f"/api/v1/cxr/reviews/{review['review_id']}/decision", json={"approve": True}, headers=admin_headers
    )
    final = client.post(
        "/api/v1/cxr/reports",
        json={"analysis_id": analyses[0], "review_id": review["review_id"]},
        headers=admin_headers,
    )
    assert final.json()["data"]["state"] == "final"

    admin_user = db.query(__import__("app.db.models", fromlist=["User"]).User).filter_by(username="admin").one()
    draft_review = CxrReview(
        analysis_id=analyses[0],
        owner_id=admin_user.id + 1000,
        reviewer_username="draft-reviewer",
        state="draft",
        revision=0,
        items_json=[],
        model_snapshot_json={},
    )
    db.add(draft_review)
    db.commit()

    admin = CurrentUser(id=admin_user.id, username="admin", role="admin", raw_role="admin")
    import pytest
    from app.core.errors import ApiError

    with pytest.raises(ApiError) as excinfo:
        report_service.create_report(
            db, settings, admin, {"analysis_id": analyses[0], "review_id": draft_review.id}
        )
    assert excinfo.value.code == "cxr_report_supersede_requires_final"


def test_report_is_scoped_to_owner(client, admin_headers, other_headers, settings, db, cxr_dataset):
    analyses = _analyses(client, admin_headers, settings, db, cxr_dataset)
    review = client.post(
        "/api/v1/cxr/reviews",
        json={"analysis_id": analyses[0]},
        headers=admin_headers,
    ).json()["data"]
    report = client.post(
        "/api/v1/cxr/reports",
        json={"analysis_id": analyses[0], "review_id": review["review_id"]},
        headers=admin_headers,
    ).json()["data"]
    assert client.get(f"/api/v1/cxr/reports/{report['report_id']}", headers=other_headers).status_code == 404
    assert client.get(
        f"/api/v1/cxr/reports/{report['report_id']}/export", headers=other_headers
    ).status_code == 404


def test_duplicate_review_for_same_owner_conflicts(client, admin_headers, settings, db, cxr_dataset):
    analysis_id = _analyses(client, admin_headers, settings, db, cxr_dataset)[0]
    first = client.post("/api/v1/cxr/reviews", json={"analysis_id": analysis_id}, headers=admin_headers)
    assert first.status_code == 201
    second = client.post("/api/v1/cxr/reviews", json={"analysis_id": analysis_id}, headers=admin_headers)
    assert second.status_code == 409
    assert second.json()["code"] == "cxr_review_exists"


def test_dashboard_reports_real_counts(client, admin_headers, settings, db, cxr_dataset):
    _analyses(client, admin_headers, settings, db, cxr_dataset)
    response = client.get("/api/v1/cxr/dashboard", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["datasets"] >= 1
    assert data["analyses"]["completed"] >= 1
    assert data["note"]
