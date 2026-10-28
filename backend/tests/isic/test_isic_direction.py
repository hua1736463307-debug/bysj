"""ISIC 方向：注册表、导入、冻结与任务隔离。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.directions import registry
from app.directions.isic import ISIC_DIRECTION_KEY, ISIC_LABEL_SCHEMA, ISIC_LABELS
from app.directions.cxr import CXR_DIRECTION_KEY
from app.schemas.cxr import LABEL_KEYS

CLASS_PLAN = {"MEL": 12, "NV": 20, "BCC": 10, "AKIEC": 8, "BKL": 6, "DF": 2, "VASC": 2}
COLUMNS = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")


def build_source(root: Path, *, split: str = "Validation", plan: dict | None = None) -> Path:
    plan = plan or CLASS_PLAN
    label_dir = root / f"ISIC2018_Task3_{split}_GroundTruth"
    image_dir = root / f"ISIC2018_Task3_{split}_Input"
    label_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    index = 0
    for column, count in plan.items():
        for _ in range(count):
            index += 1
            name = f"ISIC_{index:07d}"
            rng = np.random.default_rng(index)
            array = (rng.random((48, 48, 3)) * 255).astype("uint8")
            Image.fromarray(array, mode="RGB").save(image_dir / f"{name}.jpg", format="JPEG", quality=90)
            rows.append({"image": name, **{key: (1 if key == column else 0) for key in COLUMNS}})
    csv_path = label_dir / f"ISIC2018_Task3_{split}_GroundTruth.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", *COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return root


def configure_source(settings, root: Path, source_id: str = "isic-fixture") -> None:
    settings.dataset_sources = {**(settings.dataset_sources or {}), source_id: str(root)}


def import_split(client, admin_headers, settings, root, *, split: str = "val", key: str = "isic-import") -> dict:
    configure_source(settings, root)
    response = client.post(
        "/api/v1/isic/datasets/import",
        json={"source_id": "isic-fixture", "split": split, "name": "ISIC fixture", "license_note": "CC BY-NC 4.0"},
        headers={**admin_headers, "Idempotency-Key": key},
    )
    assert response.status_code == 202, response.text
    job = response.json()["data"]
    assert job["status"] == "completed", job
    return job["result"]


def test_registry_exposes_three_task_kinds():
    specs = {spec.key: spec for spec in registry.all_directions()}
    assert set(specs) == {"bccd_detect", CXR_DIRECTION_KEY, ISIC_DIRECTION_KEY}
    assert specs[CXR_DIRECTION_KEY].task_kind == "multilabel_classification"
    assert specs[ISIC_DIRECTION_KEY].task_kind == "single_label_classification"
    assert specs["bccd_detect"].task_kind == "object_detection"
    # 输出语义与任务语义必须匹配，不能混用
    assert specs[CXR_DIRECTION_KEY].analysis_output == "per_label_scores"
    assert specs[ISIC_DIRECTION_KEY].analysis_output == "single_label_probabilities"
    assert specs["bccd_detect"].analysis_output == "bounding_boxes"


def test_directions_do_not_share_label_lists():
    cxr = registry.require(CXR_DIRECTION_KEY)
    isic = registry.require(ISIC_DIRECTION_KEY)
    assert cxr.label_keys() == LABEL_KEYS
    assert isic.label_keys() == tuple(label.key for label in ISIC_LABELS)
    assert not set(cxr.label_keys()) & set(isic.label_keys())
    # 胸片四态不得出现在皮肤镜声明里
    assert "uncertain" not in json.dumps(isic.to_public())
    assert isic.label_schema == ISIC_LABEL_SCHEMA


def test_directions_endpoint_requires_auth(client):
    assert client.get("/api/v1/directions").status_code == 401


def test_directions_endpoint_lists_and_404s(client, admin_headers):
    response = client.get("/api/v1/directions", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total"] == 3
    assert {item["direction_key"] for item in data["items"]} == {"bccd_detect", "cxr", "isic2018"}

    detail = client.get(f"/api/v1/directions/{ISIC_DIRECTION_KEY}", headers=admin_headers)
    assert detail.status_code == 200
    payload = detail.json()["data"]
    assert payload["task_kind"] == "single_label_classification"
    assert len(payload["labels"]) == 7
    assert "path" not in json.dumps(payload).lower()

    assert client.get("/api/v1/directions/nope", headers=admin_headers).status_code == 404


def test_isic_schema_endpoint(client, admin_headers):
    response = client.get("/api/v1/isic/schema", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["task_kind"] == "single_label_classification"
    assert [item["key"] for item in data["labels"]] == [label.key for label in ISIC_LABELS]


def test_import_and_freeze_real_layout(client, admin_headers, settings, tmp_path):
    root = build_source(tmp_path / "isic")
    result = import_split(client, admin_headers, settings, root)
    assert result["asset_count"] == sum(CLASS_PLAN.values())
    assert result["report"]["class_counts"]["nv"] == 20
    assert result["report"]["lesion_groupings_available"] is False
    assert any("患者级隔离" in item or "lesion" in item for item in result["report"]["limitations"])

    freeze = client.post(
        f"/api/v1/isic/datasets/{result['dataset_id']}/versions",
        json={"split_seed": 42, "ratios": [0.7, 0.1, 0.1, 0.1]},
        headers={**admin_headers, "Idempotency-Key": "isic-freeze"},
    )
    assert freeze.status_code == 202, freeze.text
    job = freeze.json()["data"]
    assert job["status"] == "completed", job
    frozen = job["result"]
    assert frozen["state"] == "frozen"
    assert sum(frozen["split_counts"].values()) == sum(CLASS_PLAN.values())
    assert frozen["class_gate"] == [], frozen["class_gate"]
    assert frozen["rare_class_warnings"], "DF/VASC 各 2 例应被标为稀有类"

    summary = client.get(
        f"/api/v1/isic/datasets/{result['dataset_id']}/versions/{result['version_id']}/summary",
        headers=admin_headers,
    )
    assert summary.status_code == 200
    payload = summary.json()["data"]
    assert payload["group_level"] == "image"
    assert sum(payload["counts"].values()) == sum(CLASS_PLAN.values())


def test_import_rejects_missing_image(client, admin_headers, settings, tmp_path):
    root = build_source(tmp_path / "isic")
    victim = sorted((root / "ISIC2018_Task3_Validation_Input").glob("*.jpg"))[0]
    victim.unlink()
    configure_source(settings, root)
    response = client.post(
        "/api/v1/isic/datasets/import",
        json={"source_id": "isic-fixture", "split": "val"},
        headers={**admin_headers, "Idempotency-Key": "isic-missing"},
    )
    job = response.json()["data"]
    assert job["status"] == "failed"
    assert job["error"]["code"] == "isic_import_invalid"
    assert "缺失" in json.dumps(job["error"], ensure_ascii=False)


def test_import_rejects_multi_label_row(client, admin_headers, settings, tmp_path):
    root = build_source(tmp_path / "isic")
    csv_path = root / "ISIC2018_Task3_Validation_GroundTruth" / "ISIC2018_Task3_Validation_GroundTruth.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    mel, nv = header.index("MEL"), header.index("NV")
    first = lines[1].split(",")
    first[mel] = "1.0"
    first[nv] = "1.0"
    lines[1] = ",".join(first)
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    configure_source(settings, root)
    response = client.post(
        "/api/v1/isic/datasets/import",
        json={"source_id": "isic-fixture", "split": "val"},
        headers={**admin_headers, "Idempotency-Key": "isic-multihot"},
    )
    job = response.json()["data"]
    assert job["status"] == "failed"
    assert "恰好一个正类" in json.dumps(job["error"], ensure_ascii=False)


def test_import_rejects_unknown_column(client, admin_headers, settings, tmp_path):
    root = build_source(tmp_path / "isic")
    csv_path = root / "ISIC2018_Task3_Validation_GroundTruth" / "ISIC2018_Task3_Validation_GroundTruth.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    lines = [f"{line},EXTRA" for line in lines]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    configure_source(settings, root)
    response = client.post(
        "/api/v1/isic/datasets/import",
        json={"source_id": "isic-fixture", "split": "val"},
        headers={**admin_headers, "Idempotency-Key": "isic-extra-col"},
    )
    job = response.json()["data"]
    assert job["status"] == "failed"
    assert "不一致" in json.dumps(job["error"], ensure_ascii=False)


def test_unconfigured_source_is_refused(client, admin_headers, settings):
    settings.dataset_sources = {}
    settings.dataset_import_root = None
    response = client.post(
        "/api/v1/isic/datasets/import",
        json={"source_id": "unknown", "split": "val"},
        headers={**admin_headers, "Idempotency-Key": "isic-unknown"},
    )
    job = response.json()["data"]
    assert job["status"] == "failed"
    assert job["error"]["code"] in {"source_not_configured", "source_missing"}


def test_cross_direction_endpoints_reject(client, admin_headers, settings, tmp_path, cxr_dataset):
    root = build_source(tmp_path / "isic")
    imported = import_split(client, admin_headers, settings, root, key="isic-cross")
    # 用 CXR 接口读皮肤镜数据集
    response = client.get(f"/api/v1/cxr/datasets/{imported['dataset_id']}", headers=admin_headers)
    assert response.status_code == 409
    assert response.json()["code"] == "task_mismatch"
    # 用 ISIC 接口读胸片数据集
    response = client.get(f"/api/v1/isic/datasets/{cxr_dataset['dataset_id']}", headers=admin_headers)
    assert response.status_code == 409
    assert response.json()["code"] == "task_mismatch"
    # 用 ISIC 接口冻结胸片数据集
    response = client.post(
        f"/api/v1/isic/datasets/{cxr_dataset['dataset_id']}/versions",
        json={"split_seed": 42},
        headers={**admin_headers, "Idempotency-Key": "isic-freeze-cxr"},
    )
    assert response.status_code == 409
