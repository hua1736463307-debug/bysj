"""CXR 任务类型守卫：CXR 与 detect 互不串用，骨干不能直接分析。"""
from __future__ import annotations

import cxr_support
from app.db.models import DatasetResource, DatasetVersion


def _detect_dataset(db) -> DatasetResource:
    resource = DatasetResource(
        dataset_id="bccd-legacy",
        dataset_role="medical",
        owner_username="admin",
        name="BCCD detect",
        task_type="detect",
        group_level="original_image",
    )
    db.add(resource)
    db.flush()
    db.add(
        DatasetVersion(
            dataset_id=resource.id,
            dataset_role="medical",
            status="ready",
            state="draft",
            class_map_json={"0": "RBC"},
            root_path="/tmp/does-not-matter",
        )
    )
    db.commit()
    return resource


def test_cxr_endpoints_reject_detect_dataset(client, admin_headers, db):
    resource = _detect_dataset(db)
    response = client.get(f"/api/v1/cxr/datasets/{resource.id}", headers=admin_headers)
    assert response.status_code == 409
    assert response.json()["code"] == "task_mismatch"


def test_detect_freeze_endpoint_rejects_cxr_dataset(client, admin_headers, cxr_dataset):
    response = client.post(
        f"/api/v1/datasets/{cxr_dataset['dataset_id']}/versions",
        json={"split_seed": 42, "ratios": [0.8, 0.1, 0.1]},
        headers=admin_headers,
    )
    assert response.status_code == 409
    assert response.json()["code"] == "task_mismatch"


def test_detect_train_tiles_rejects_cxr_version(client, admin_headers, cxr_dataset):
    response = client.post(
        f"/api/v1/datasets/{cxr_dataset['dataset_id']}/train-tile-versions",
        json={"parent_version_id": cxr_dataset["version_id"], "tile_size": 320},
        headers={**admin_headers, "Idempotency-Key": "tiles-on-cxr"},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "task_mismatch"


def test_backbone_model_cannot_be_used_for_analysis(client, admin_headers, settings, db, cxr_dataset):
    admin_id = db.query(__import__("app.db.models", fromlist=["User"]).User).filter_by(username="admin").one().id
    backbone = cxr_support.register_model(
        db, settings, owner_id=admin_id, model_task="cxr_backbone", name="imagenet"
    )
    response = client.post(
        "/api/v1/cxr/analysis-batches",
        json={"model_id": backbone.id, "asset_ids": cxr_dataset["asset_ids"]},
        headers={**admin_headers, "Idempotency-Key": "backbone-analysis"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "cxr_backbone_not_analyzable"


def test_detect_model_cannot_be_used_for_cxr_analysis(client, admin_headers, settings, db, cxr_dataset):
    admin_id = db.query(__import__("app.db.models", fromlist=["User"]).User).filter_by(username="admin").one().id
    detect_model = cxr_support.register_model(
        db, settings, owner_id=admin_id, model_task="detect", label_schema="", name="yolo"
    )
    response = client.post(
        "/api/v1/cxr/analysis-batches",
        json={"model_id": detect_model.id, "asset_ids": cxr_dataset["asset_ids"]},
        headers={**admin_headers, "Idempotency-Key": "detect-on-cxr"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "model_task_mismatch"


def test_schema_mismatch_is_rejected(client, admin_headers, settings, db, cxr_dataset):
    admin_id = db.query(__import__("app.db.models", fromlist=["User"]).User).filter_by(username="admin").one().id
    model = cxr_support.register_model(db, settings, owner_id=admin_id, label_schema="cxr-other-v9")
    response = client.post(
        "/api/v1/cxr/analysis-batches",
        json={"model_id": model.id, "asset_ids": cxr_dataset["asset_ids"]},
        headers={**admin_headers, "Idempotency-Key": "schema-mismatch"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "cxr_schema_mismatch"


def test_model_file_tampering_is_rejected(client, admin_headers, settings, db, cxr_dataset):
    admin_id = db.query(__import__("app.db.models", fromlist=["User"]).User).filter_by(username="admin").one().id
    model = cxr_support.register_model(db, settings, owner_id=admin_id)
    with open(model.path, "ab") as handle:
        handle.write(b"tampered")
    response = client.post(
        "/api/v1/cxr/analysis-batches",
        json={"model_id": model.id, "asset_ids": cxr_dataset["asset_ids"]},
        headers={**admin_headers, "Idempotency-Key": "tampered-model"},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "model_tampered"


def test_cxr_import_requires_admin(client, researcher_headers, settings, tmp_path):
    from pathlib import Path

    root = cxr_support.build_source(Path(tmp_path) / "src", patients=8)
    cxr_support.configure_source(settings, root)
    response = client.post(
        "/api/v1/cxr/datasets/import",
        json={"source_id": "fixture", "source_csv": "train.csv", "max_patients": 8},
        headers={**researcher_headers, "Idempotency-Key": "researcher-import"},
    )
    assert response.status_code == 403


def test_test_split_evaluation_requires_admin_and_purpose(client, admin_headers, settings, db, cxr_dataset):
    admin_id = db.query(__import__("app.db.models", fromlist=["User"]).User).filter_by(username="admin").one().id
    model = cxr_support.register_model(db, settings, owner_id=admin_id)
    response = client.post(
        "/api/v1/cxr/evaluations",
        json={
            "model_id": model.id,
            "dataset_version_id": cxr_dataset["version_id"],
            "split": "test",
            "action": "evaluate",
        },
        headers={**admin_headers, "Idempotency-Key": "test-no-purpose"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "cxr_test_purpose_required"


def test_test_split_evaluation_forbidden_for_non_admin(client, researcher_headers, settings, db, researcher):
    """非管理员即使拥有数据集也不能读取 test 集。"""
    from app.db.models import DatasetResource, DatasetVersion

    resource = DatasetResource(
        dataset_id="cxr-researcher-owned",
        dataset_role="medical",
        owner_user_id=researcher.id,
        owner_username=researcher.username,
        name="researcher cxr",
        task_type="cxr_multilabel",
        group_level="patient_component",
    )
    db.add(resource)
    db.flush()
    version = DatasetVersion(
        dataset_id=resource.id,
        dataset_role="medical",
        status="ready",
        state="frozen",
        manifest_sha256="0" * 64,
        root_path=str(settings.data_root / "datasets" / resource.dataset_id / "v1"),
    )
    db.add(version)
    db.commit()
    model = cxr_support.register_model(db, settings, owner_id=researcher.id)
    response = client.post(
        "/api/v1/cxr/evaluations",
        json={
            "model_id": model.id,
            "dataset_version_id": version.id,
            "split": "test",
            "action": "evaluate",
            "purpose": "researcher trying test",
        },
        headers={**researcher_headers, "Idempotency-Key": "test-non-admin"},
    )
    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"


def test_calibration_must_use_calibration_split(client, admin_headers, settings, db, cxr_dataset):
    admin_id = db.query(__import__("app.db.models", fromlist=["User"]).User).filter_by(username="admin").one().id
    model = cxr_support.register_model(db, settings, owner_id=admin_id)
    response = client.post(
        "/api/v1/cxr/evaluations",
        json={
            "model_id": model.id,
            "dataset_version_id": cxr_dataset["version_id"],
            "split": "val",
            "action": "calibrate",
        },
        headers={**admin_headers, "Idempotency-Key": "calibrate-on-val"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "cxr_calibration_split"


def test_degenerate_cam_is_rejected_not_blank():
    """Grad-CAM 全为非正时必须报错，不能输出全黑热图让人误读。"""
    import numpy as np
    import pytest

    from app.core.errors import ApiError
    from app.ml.cxr_adapter import normalize_cam

    with pytest.raises(ApiError) as excinfo:
        normalize_cam(np.zeros((7, 7), dtype=np.float32))
    assert excinfo.value.code == "cxr_explain_degenerate"

    with pytest.raises(ApiError) as excinfo:
        normalize_cam(np.array([[np.nan, 1.0]], dtype=np.float32))
    assert excinfo.value.code == "cxr_explain_failed"

    normalized = normalize_cam(np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32))
    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == 1.0


def test_device_string_is_normalized_for_torch():
    """训练参数默认 device="0"，torch 只接受 "cuda:0"/"cpu"。"""

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

    class _FakeTorch:
        cuda = _FakeCuda()

    from app.ml.cxr_adapter import _normalize_device

    assert _normalize_device("0", _FakeTorch) == "cuda:0"
    assert _normalize_device("cuda", _FakeTorch) == "cuda:0"
    assert _normalize_device("cuda:1", _FakeTorch) == "cuda:0"  # 越界回退 0
    assert _normalize_device("cpu", _FakeTorch) == "cpu"

    class _NoCuda:
        @staticmethod
        def is_available():
            return False

    class _NoTorch:
        cuda = _NoCuda()

    assert _normalize_device("0", _NoTorch) == "cpu"
