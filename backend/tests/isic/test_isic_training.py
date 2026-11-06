"""ISIC 七类训练与评价链路：指标、预检、作业、模型登记与策略。

全部使用 dummy 后端：只验证契约与指标计算，不产生任何研究指标。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.ml.isic_metrics import evaluate as evaluate_metrics
from app.schemas.isic import ISIC_CLASS_KEYS

COLUMNS = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")
#: 每类至少 4 张，保证四集派生后 val/calibration/test 仍有该类样本。
CLASS_PLAN = {"MEL": 24, "NV": 40, "BCC": 12, "AKIEC": 10, "BKL": 8, "DF": 4, "VASC": 4}


def build_training_source(root: Path, plan: dict | None = None) -> Path:
    plan = plan or CLASS_PLAN
    label_dir = root / "ISIC2018_Task3_Training_GroundTruth"
    image_dir = root / "ISIC2018_Task3_Training_Input"
    label_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    rows, groupings = [], []
    index = 0
    for column, count in plan.items():
        for position in range(count):
            index += 1
            name = f"ISIC_{index:07d}"
            rng = np.random.default_rng(index)
            array = (rng.random((48, 48, 3)) * 255).astype("uint8")
            Image.fromarray(array, mode="RGB").save(image_dir / f"{name}.jpg", format="JPEG", quality=90)
            rows.append({"image": name, **{key: (1 if key == column else 0) for key in COLUMNS}})
            # 每 4 张同 lesion，用来验证分组切分不会把同组拆开
            groupings.append({"image": name, "lesion_id": f"HAM_{column}_{position // 4:04d}",
                              "diagnosis_confirm_type": "histopathology"})
    csv_path = label_dir / "ISIC2018_Task3_Training_GroundTruth.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", *COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    with (root / "ISIC2018_Task3_Training_LesionGroupings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", "lesion_id", "diagnosis_confirm_type"))
        writer.writeheader()
        writer.writerows(groupings)
    return root


def import_and_freeze(client, admin_headers, settings, root, *, key: str = "isic") -> dict:
    settings.dataset_sources = {**(settings.dataset_sources or {}), "isic-train": str(root)}
    response = client.post(
        "/api/v1/isic/datasets/import",
        json={"source_id": "isic-train", "split": "train", "name": "ISIC training fixture"},
        headers={**admin_headers, "Idempotency-Key": f"{key}-import"},
    )
    assert response.status_code == 202, response.text
    imported = response.json()["data"]
    assert imported["status"] == "completed", imported
    result = imported["result"]
    freeze = client.post(
        f"/api/v1/isic/datasets/{result['dataset_id']}/versions",
        json={"split_seed": 42, "ratios": [0.7, 0.1, 0.1, 0.1]},
        headers={**admin_headers, "Idempotency-Key": f"{key}-freeze"},
    )
    assert freeze.status_code == 202, freeze.text
    job = freeze.json()["data"]
    assert job["status"] == "completed", job
    return {
        "dataset_id": result["dataset_id"],
        "version_id": job["result"]["version_id"],
        "manifest": job["result"],
    }


@pytest.fixture()
def isic_version(client, admin_headers, settings, tmp_path):
    settings.isic_ml_backend = "dummy"
    root = build_training_source(tmp_path / "isic-train")
    return import_and_freeze(client, admin_headers, settings, root)


# ---------------------------------------------------------------- 指标


def test_confusion_matrix_and_per_class_metrics():
    truth = [0, 0, 1, 1, 2, 2]
    pred = [0, 1, 1, 1, 2, 0]
    report = evaluate_metrics(truth, pred)
    matrix = report["confusion_matrix"]
    assert len(matrix) == 7 and len(matrix[0]) == 7
    assert matrix[0] == [1, 1, 0, 0, 0, 0, 0]
    assert matrix[1] == [0, 2, 0, 0, 0, 0, 0]
    assert report["accuracy"]["value"] == pytest.approx(4 / 6, rel=1e-6)

    mel = report["per_class"][0]
    assert mel["support"] == 2 and mel["predicted"] == 2 and mel["true_positive"] == 1
    assert mel["precision"]["value"] == pytest.approx(0.5)
    assert mel["recall"]["value"] == pytest.approx(0.5)
    assert mel["f1"]["value"] == pytest.approx(0.5)

    # 未出现的类别：support=0 → recall/f1 为 None 且带原因，不填 0
    absent = report["per_class"][3]
    assert absent["support"] == 0
    assert absent["recall"] == {"value": None, "reason": "no_support_in_split"}
    assert absent["f1"] == {"value": None, "reason": "no_support_in_split"}
    assert absent["precision"] == {"value": None, "reason": "no_predicted_positive"}

    # 逐类 F1：mel 0.5、nv 0.8（P=2/3, R=1）、bcc 2/3（P=1, R=0.5）
    assert report["per_class"][1]["f1"]["value"] == pytest.approx(0.8, rel=1e-6)
    assert report["per_class"][2]["f1"]["value"] == pytest.approx(2 / 3, rel=1e-6)
    # macro 只对 split 内出现过的类别取平均
    assert report["macro_f1"]["value"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3, rel=1e-6)
    assert report["macro_f1_scope"].startswith("仅对")


def test_class_never_predicted_has_zero_f1_with_reason():
    truth = [0, 0, 1, 1]
    pred = [0, 0, 0, 0]
    report = evaluate_metrics(truth, pred)
    nv = report["per_class"][1]
    assert nv["support"] == 2 and nv["predicted"] == 0
    assert nv["precision"] == {"value": None, "reason": "no_predicted_positive"}
    assert nv["recall"]["value"] == 0.0
    assert nv["f1"] == {"value": 0.0, "reason": "no_predicted_positive"}
    # mel: P=0.5（4 次预测里 2 次正确）、R=1 → F1=2/3；nv: F1=0 → macro=1/3
    assert report["per_class"][0]["f1"]["value"] == pytest.approx(2 / 3, rel=1e-6)
    # 指标对外保留 6 位小数，比较用绝对容差
    assert report["macro_f1"]["value"] == pytest.approx((2 / 3 + 0.0) / 2, abs=1e-6)


def test_empty_split_returns_reasons_not_zeros():
    report = evaluate_metrics([], [])
    assert report["support"] == 0
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "ece"):
        assert report[key]["value"] is None
        assert report[key]["reason"] == "no_samples"
    assert report["confusion_matrix"] == [[0] * 7 for _ in range(7)]


def test_ece_present_when_probabilities_given():
    truth = [0, 1, 2]
    probabilities = np.array([[0.9, 0.05, 0.05, 0, 0, 0, 0],
                              [0.1, 0.8, 0.1, 0, 0, 0, 0],
                              [0.1, 0.1, 0.8, 0, 0, 0, 0]], dtype=float)
    report = evaluate_metrics(truth, [0, 1, 2], probabilities=probabilities)
    assert report["ece"]["value"] is not None
    assert report["ece"]["reason"] is None
    assert len(report["ece"]["table"]) == 10
    without = evaluate_metrics(truth, [0, 1, 2])
    assert without["ece"] == {"value": None, "reason": "probabilities_unavailable"}


# ---------------------------------------------------------------- 预检与作业


def test_training_preflight_requires_frozen_version(client, admin_headers, settings, tmp_path):
    settings.isic_ml_backend = "dummy"
    root = build_training_source(tmp_path / "isic-unfrozen")
    settings.dataset_sources = {**(settings.dataset_sources or {}), "isic-unfrozen": str(root)}
    imported = client.post(
        "/api/v1/isic/datasets/import",
        json={"source_id": "isic-unfrozen", "split": "train", "name": "unfrozen"},
        headers={**admin_headers, "Idempotency-Key": "unfrozen-import"},
    ).json()["data"]["result"]
    response = client.post(
        "/api/v1/isic/training/preflight",
        json={"dataset_version_id": imported["version_id"], "params": {}},
        headers=admin_headers,
    )
    assert response.status_code == 200
    report = response.json()["data"]
    assert report["ok"] is False
    checks = {item["name"]: item["status"] for item in report["checks"]}
    assert checks["dataset_version"] == "fail"


def test_training_preflight_rejects_unknown_param(client, admin_headers, isic_version):
    response = client.post(
        "/api/v1/isic/training/preflight",
        json={"dataset_version_id": isic_version["version_id"], "params": {"mosaic": 0.5}},
        headers=admin_headers,
    )
    assert response.status_code == 200
    report = response.json()["data"]
    assert report["ok"] is False
    assert any(item["name"] == "params" and item["status"] == "fail" for item in report["checks"])


def test_training_preflight_passes_on_frozen_version(client, admin_headers, isic_version):
    response = client.post(
        "/api/v1/isic/training/preflight",
        json={"dataset_version_id": isic_version["version_id"], "params": {"epochs": 1, "batch": 4}},
        headers=admin_headers,
    )
    assert response.status_code == 200
    report = response.json()["data"]
    assert report["ok"] is True, report
    assert report["label_schema"] == "isic2018-task3-v1"
    assert report["class_order"] == list(ISIC_CLASS_KEYS)


def test_training_job_registers_isic_model(client, admin_headers, settings, isic_version):
    response = client.post(
        "/api/v1/isic/training/jobs",
        json={"dataset_version_id": isic_version["version_id"], "params": {"epochs": 1, "batch": 4}},
        headers={**admin_headers, "Idempotency-Key": "isic-train-1"},
    )
    assert response.status_code == 202, response.text
    job = response.json()["data"]
    assert job["type"] == "isic_train"
    assert job["status"] == "completed", job
    model_id = job["result"]["model_id"]

    models = client.get("/api/v1/isic/models", headers=admin_headers).json()["data"]
    entry = next(item for item in models["items"] if item["model_id"] == model_id)
    assert entry["model_task"] == "isic_multiclass"
    assert entry["label_schema"] == "isic2018-task3-v1"
    assert entry["class_order"] == list(ISIC_CLASS_KEYS)
    # dummy 后端不得被当成有效研究结果
    assert entry["research_valid"] is False


def test_isic_job_types_registered_on_trainer():
    from app.workers.dispatcher import HANDLERS
    from app.workers.trainer_worker import WORKER_TYPES

    assert HANDLERS["isic_train"].endswith("handle_isic_train")
    assert HANDLERS["isic_evaluate"].endswith("handle_isic_evaluate")
    assert "isic_train" in WORKER_TYPES and "isic_evaluate" in WORKER_TYPES
    # 复用同一 trainer 角色，不新建角色
    assert "cxr_train" in WORKER_TYPES


# ---------------------------------------------------------------- 评价策略


def _train_model(client, admin_headers, version_id: str, key: str) -> str:
    response = client.post(
        "/api/v1/isic/training/jobs",
        json={"dataset_version_id": version_id, "params": {"epochs": 1, "batch": 4}},
        headers={**admin_headers, "Idempotency-Key": key},
    )
    assert response.status_code == 202, response.text
    return response.json()["data"]["result"]["model_id"]


def test_evaluation_on_val_records_metrics(client, admin_headers, isic_version):
    model_id = _train_model(client, admin_headers, isic_version["version_id"], "eval-train-1")
    response = client.post(
        "/api/v1/isic/evaluations",
        json={"model_id": model_id, "dataset_version_id": isic_version["version_id"], "split": "val"},
        headers={**admin_headers, "Idempotency-Key": "eval-val-1"},
    )
    assert response.status_code == 202, response.text
    job = response.json()["data"]
    assert job["type"] == "isic_evaluate"
    assert job["status"] == "completed", job
    metrics = job["result"]["metrics"]
    assert len(metrics["confusion_matrix"]) == 7
    assert len(metrics["per_class"]) == 7
    assert metrics["research_valid"] is False

    listing = client.get("/api/v1/isic/evaluations", headers=admin_headers).json()["data"]
    row = next(item for item in listing["items"] if item["evaluation_id"] == job["result"]["evaluation_id"])
    assert row["status"] == "completed"
    assert row["split"] == "val"


def test_test_split_requires_admin_and_purpose(client, researcher_headers, admin_headers, isic_version):
    model_id = _train_model(client, admin_headers, isic_version["version_id"], "eval-train-2")

    forbidden = client.post(
        "/api/v1/isic/evaluations",
        json={"model_id": model_id, "dataset_version_id": isic_version["version_id"], "split": "test"},
        headers={**researcher_headers, "Idempotency-Key": "eval-test-researcher"},
    )
    assert forbidden.status_code == 403

    no_purpose = client.post(
        "/api/v1/isic/evaluations",
        json={"model_id": model_id, "dataset_version_id": isic_version["version_id"], "split": "test"},
        headers={**admin_headers, "Idempotency-Key": "eval-test-nopurpose"},
    )
    assert no_purpose.status_code == 422
    assert no_purpose.json()["code"] == "isic_test_purpose_required"


def test_cxr_model_cannot_be_used_for_isic_evaluation(client, admin_headers, settings, isic_version, db):
    from app.db.models import ModelRegistry
    import uuid

    cxr_model = ModelRegistry(
        id=uuid.uuid4().hex,
        name="cxr-dummy-for-isic-guard",
        path=str(Path(settings.model_root) / "guard.json"),
        sha256="0" * 64,
        size=1,
        source="training",
        status="active",
        format="json",
        created_by="admin",
        model_task="cxr_multilabel",
        available=True,
        metadata_json={"label_schema": "cxr-five-v1"},
    )
    db.add(cxr_model)
    db.commit()

    response = client.post(
        "/api/v1/isic/evaluations",
        json={"model_id": cxr_model.id, "dataset_version_id": isic_version["version_id"], "split": "val"},
        headers={**admin_headers, "Idempotency-Key": "eval-cxr-model"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "isic_evaluation_preflight_failed"


def test_isic_private_resources_require_owner(client, admin_headers, researcher_headers, researcher, isic_version, db):
    from app.db.models import DatasetResource, ModelRegistry, JobTask
    model_id = _train_model(client, admin_headers, isic_version["version_id"], "owner-model")
    request = {"dataset_version_id": isic_version["version_id"], "params": {"epochs": 1}}
    preflight = client.post('/api/v1/isic/training/preflight', headers=researcher_headers, json=request)
    assert preflight.json()['data']['ok'] is False
    assert client.post('/api/v1/isic/training/jobs', json=request,
                       headers={**researcher_headers, 'Idempotency-Key': 'other-data'}).status_code == 422
    # Owning the dataset alone must not grant use of another owner's trained model.
    resource = db.get(DatasetResource, isic_version['dataset_id'])
    resource.owner_user_id = researcher.id
    db.commit()
    evaluation = {'model_id': model_id, 'dataset_version_id': isic_version['version_id'], 'split': 'val'}
    assert client.post('/api/v1/isic/evaluations', json=evaluation,
                       headers={**researcher_headers, 'Idempotency-Key': 'other-model'}).status_code == 422
    model = db.get(ModelRegistry, model_id)
    model.model_task = 'isic_backbone'
    db.commit()
    request['base_model_id'] = model_id
    assert not client.post('/api/v1/isic/training/preflight', json=request,
                           headers=researcher_headers).json()['data']['ok']
    model.trusted_source = 'base'
    db.commit()
    assert client.post('/api/v1/isic/training/preflight', json=request,
                       headers=researcher_headers).json()['data']['ok']
    assert db.query(JobTask).filter(JobTask.owner_id == str(researcher.id)).count() == 0


def test_isic_evaluation_retry_keeps_one_run(client, admin_headers, isic_version, db):
    from app.db.models import EvaluationRun
    model_id = _train_model(client, admin_headers, isic_version['version_id'], 'retry-model')
    payload = {'model_id': model_id, 'dataset_version_id': isic_version['version_id'], 'split': 'val'}
    headers = {**admin_headers, 'Idempotency-Key': 'retry-evaluation'}
    first = client.post('/api/v1/isic/evaluations', headers=headers, json=payload)
    second = client.post('/api/v1/isic/evaluations', headers=headers, json=payload)
    assert first.status_code == second.status_code == 202, second.text
    assert first.json()['data']['job_id'] == second.json()['data']['job_id']
    assert db.query(EvaluationRun).count() == 1
    changed = client.post('/api/v1/isic/evaluations', headers=headers, json={**payload, 'split': 'calibration'})
    assert changed.status_code == 409
    assert db.query(EvaluationRun).count() == 1


def test_isic_running_cancel_updates_evaluation(client, admin_headers, isic_version, db, monkeypatch):
    from app.db.models import JobTask, EvaluationRun
    from app.db.session import get_session_factory
    from app.domains import job_service
    from app.domains.isic import evaluation_service
    model_id = _train_model(client, admin_headers, isic_version['version_id'], 'cancel-model')
    def cancel_run(settings, session, *, job_id, should_stop, **kwargs):
        with get_session_factory()() as other:
            job_service.cancel_job(other, other.get(JobTask, job_id))
        should_stop()
        raise AssertionError('cancel must stop execution')
    monkeypatch.setattr(evaluation_service, 'run', cancel_run)
    response = client.post('/api/v1/isic/evaluations', headers={**admin_headers, 'Idempotency-Key': 'cancel-eval'},
        json={'model_id': model_id, 'dataset_version_id': isic_version['version_id'], 'split': 'val'})
    assert response.status_code == 202, response.text
    assert response.json()['data']['status'] == 'cancelled'
    run = db.query(EvaluationRun).one()
    assert run.status == 'cancelled' and run.finished_at is not None


def test_stale_isic_evaluator_cannot_overwrite_new_attempt(client, admin_headers, isic_version, db, monkeypatch):
    from app.db.models import JobTask, EvaluationRun
    from app.db.session import get_session_factory
    from app.domains.isic import evaluation_service
    model_id = _train_model(client, admin_headers, isic_version['version_id'], 'stale-model')
    def replace_attempt(settings, session, *, job_id, should_stop, **kwargs):
        with get_session_factory()() as other:
            job = other.get(JobTask, job_id)
            job.attempt += 1
            job.lease_owner = 'new-worker'
            run = other.get(EvaluationRun, job.input_json['evaluation_id'])
            run.status = 'completed'
            run.metrics_json = {'new_attempt': True}
            other.commit()
        should_stop()
        raise AssertionError('stale attempt must stop execution')
    monkeypatch.setattr(evaluation_service, 'run', replace_attempt)
    response = client.post('/api/v1/isic/evaluations', headers={**admin_headers, 'Idempotency-Key': 'stale-eval'},
        json={'model_id': model_id, 'dataset_version_id': isic_version['version_id'], 'split': 'val'})
    assert response.status_code == 202, response.text
    run = db.query(EvaluationRun).one()
    assert run.status == 'completed' and run.metrics_json == {'new_attempt': True}
