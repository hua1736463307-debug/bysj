"""CXR 测试公共装置：合成 fixture、导入/冻结、dummy 模型登记。"""
from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path

from app.core.storage import ensure_dir, sha256_file
from app.db.models import ModelRegistry
from app.schemas.cxr import LABEL_DISPLAY, LABEL_KEYS

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_fixture_module():
    path = REPO_ROOT / "scripts" / "make_cxr_fixture.py"
    spec = importlib.util.spec_from_file_location("make_cxr_fixture", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_cxr_fixture"] = module
    spec.loader.exec_module(module)
    return module


def build_source(root: Path, *, patients: int = 40, duplicate_pairs: int = 2, size: int = 160) -> Path:
    module = _load_fixture_module()
    module.build(root, patients=patients, studies=1, size=size, seed=42, duplicate_pairs=duplicate_pairs)
    return root


def configure_source(settings, root: Path, source_id: str = "fixture") -> None:
    settings.cxr_sources = {**(settings.cxr_sources or {}), source_id: str(root)}
    settings.cxr_allow_synthetic = True


def import_fixture(client, admin_headers, settings, root: Path, *, source_id: str = "fixture",
                   max_patients: int = 10, idempotency_key: str | None = None) -> dict:
    configure_source(settings, root, source_id)
    response = client.post(
        "/api/v1/cxr/datasets/import",
        json={
            "source_id": source_id,
            "source_csv": "train.csv",
            "name": "合成胸片 fixture",
            "license_note": "synthetic engineering fixture",
            "max_patients": max_patients,
        },
        headers={**admin_headers, "Idempotency-Key": idempotency_key or f"import-{uuid.uuid4().hex[:8]}"},
    )
    assert response.status_code == 202, response.text
    job = response.json()["data"]
    assert job["status"] == "completed", job
    return job["result"]


def freeze(client, headers, dataset_id: str, *, seed: int = 42, key: str | None = None) -> dict:
    response = client.post(
        f"/api/v1/cxr/datasets/{dataset_id}/versions",
        json={"split_seed": seed, "ratios": [0.7, 0.1, 0.1, 0.1]},
        headers={**headers, "Idempotency-Key": key or f"freeze-{uuid.uuid4().hex[:8]}"},
    )
    assert response.status_code == 202, response.text
    return response.json()["data"]


def register_model(
    db,
    settings,
    *,
    owner_id: int,
    model_task: str = "cxr_multilabel",
    label_schema: str = "cxr-five-v1",
    backend: str = "dummy",
    name: str = "cxr-dummy",
    trusted_source: str = "platform",
) -> ModelRegistry:
    target = ensure_dir(Path(settings.model_root) / "cxr-dummy") / f"{uuid.uuid4().hex[:8]}.json"
    target.write_text(json.dumps({"backend": backend, "label_schema": label_schema}), encoding="utf-8")
    model = ModelRegistry(
        id=uuid.uuid4().hex,
        name=name,
        path=str(target),
        sha256=sha256_file(target),
        size=target.stat().st_size,
        source="training",
        status="active",
        format="json",
        created_by="admin",
        owner_id=owner_id,
        model_task=model_task,
        trusted_source=trusted_source,
        available=True,
        class_map_json={key: LABEL_DISPLAY[key] for key in LABEL_KEYS},
        metadata_json={
            "backend": backend,
            "label_schema": label_schema,
            "arch": "densenet121",
            "image_size": 224,
            "research_valid": False,
        },
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def dataset_id_of(client, headers) -> str:
    response = client.get("/api/v1/cxr/datasets", headers=headers)
    assert response.status_code == 200, response.text
    items = response.json()["data"]["items"]
    assert items, "没有 CXR 数据集"
    return items[0]["dataset_id"]


def version_id_of(client, headers, dataset_id: str) -> str:
    response = client.get(f"/api/v1/cxr/datasets/{dataset_id}", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()["data"]["latest_version"]["version_id"]


def asset_ids_of(client, headers, dataset_id: str, version_id: str, limit: int = 2) -> list[int]:
    response = client.get(
        f"/api/v1/datasets/{dataset_id}/assets",
        params={"version_id": version_id, "page_size": limit},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return [item["asset_id"] for item in response.json()["data"]["items"]][:limit]


def ready_dataset(client, admin_headers, settings, tmp_path, *, patients: int = 40) -> dict:
    """导入 + 冻结，返回 dataset_id/version_id/asset_ids。"""
    root = build_source(Path(tmp_path) / "src", patients=patients)
    imported = import_fixture(client, admin_headers, settings, root, max_patients=patients)
    dataset_id = imported["dataset_id"]
    job = freeze(client, admin_headers, dataset_id)
    assert job["status"] == "completed", job
    version_id = job["result"]["version_id"]
    return {
        "dataset_id": dataset_id,
        "version_id": version_id,
        "asset_ids": asset_ids_of(client, admin_headers, dataset_id, version_id, limit=3),
        "freeze": job["result"],
        "root": root,
    }
