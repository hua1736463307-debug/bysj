import hashlib
import json
import sys
import zipfile
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import run_grouped_revision as grouped


def make_package(root: Path, count: int = 4) -> Path:
    package = root / "package"
    package.mkdir()
    entries = {}
    labels = {}
    groups = {}
    with zipfile.ZipFile(package / "import.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for index in range(count):
            name = f"BloodImage_{index:05d}.jpg"
            body = ("image-%d" % index).encode()
            bundle.writestr(name, body)
            entries[name] = hashlib.sha256(body).hexdigest()
            label = "0 0.500000 0.500000 0.200000 0.200000\n"
            bundle.writestr(name[:-4] + ".txt", label)
            labels[name[:-4] + ".txt"] = hashlib.sha256(label.encode()).hexdigest()
            groups[name] = f"cluster-{index // 2}"
    (package / "source_groups.json").write_text(json.dumps(groups), encoding="utf-8")
    preparation = {
        "status": "prepared_not_imported",
        "images": count,
        "groups": len(set(groups.values())),
        "class_map": {"0": "RBC", "1": "WBC", "2": "Platelets"},
        "archive_sha256": grouped.sha256_file(package / "import.zip"),
        "group_map_sha256": grouped.sha256_file(package / "source_groups.json"),
        "group_level": "image_cluster",
        "fixed_split_seed": 42,
        "label_sha256": labels,
    }
    (package / "PREPARATION.json").write_text(json.dumps(preparation), encoding="utf-8")
    return package


def envelope(data):
    return {"code": "ok", "message": "ok", "data": data, "request_id": "test"}


class FakeServer:
    def __init__(self, count=4, fail_post=None):
        self.count = count
        self.fail_post = fail_post
        self.posts = {"import": 0, "freeze": 0}
        self.version_state = "draft"
        self.import_job = "import-job-1"
        self.freeze_job = "freeze-job-1"
        self.asset_mutator = None
        self.version_config = {"seed": 42, "group_level": "image_cluster"}

    def assets(self):
        items = []
        for index in range(self.count):
            name = f"BloodImage_{index:05d}.jpg"
            items.append({
                "asset_id": index + 1,
                "relative_path": name,
                "sha256": hashlib.sha256(("image-%d" % index).encode()).hexdigest(),
                "source_group": f"cluster-{index // 2}",
                "split": "train" if index < self.count - 2 else ("val" if index == self.count - 2 else "test"),
                "annotations": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}],
            })
        if self.asset_mutator:
            items = self.asset_mutator(items)
        return items

    def handler(self, request: httpx.Request):
        path = request.url.path
        if path.endswith("/datasets/import"):
            self.posts["import"] += 1
            if self.fail_post == "import":
                raise httpx.ConnectTimeout("simulated timeout", request=request)
            return httpx.Response(200, json=envelope({"job_id": self.import_job, "status": "pending"}))
        if path.endswith(f"/jobs/{self.import_job}"):
            return httpx.Response(200, json=envelope({
                "job_id": self.import_job, "status": "completed", "error": None,
                "result": {"dataset_id": "dataset-1", "version_id": "version-1"},
            }))
        if path.endswith(f"/jobs/{self.freeze_job}"):
            self.version_state = "frozen"
            return httpx.Response(200, json=envelope({
                "job_id": self.freeze_job, "status": "completed", "error": None,
                "result": {"version_id": "version-1", "state": "frozen"},
            }))
        if path.endswith("/versions") and request.method == "POST":
            self.posts["freeze"] += 1
            if self.fail_post == "freeze":
                raise httpx.ConnectTimeout("simulated timeout", request=request)
            return httpx.Response(200, json=envelope({"job_id": self.freeze_job, "status": "pending"}))
        if path.endswith("/dataset-versions/version-1"):
            return httpx.Response(200, json=envelope({
                "version_id": "version-1", "dataset_id": "dataset-1", "state": self.version_state,
                "split_config": {**self.version_config, "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}},
                "class_map": {"0": "RBC", "1": "WBC", "2": "Platelets"}, "split_manifest_hash": "h" * 64,
            }))
        if "/assets" in path:
            items = self.assets()
            page = int(request.url.params.get("page", "1"))
            size = int(request.url.params.get("page_size", "200"))
            chunk = items[(page - 1) * size: page * size]
            return httpx.Response(200, json=envelope({"items": chunk, "total": len(items),
                                                      "page": page, "page_size": size}))
        raise AssertionError(f"unexpected request {request.method} {path}")


def client_for(server: FakeServer) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(server.handler), timeout=5, trust_env=False)


def identity(package: Path, output: Path):
    return grouped.package_identity(package, "http://test", "run-1")


def test_import_then_rerun_does_not_post_twice(tmp_path):
    package = make_package(tmp_path)
    server = FakeServer(); client = client_for(server)
    state = tmp_path / "state.json"; ident = identity(package, tmp_path)

    job = grouped.import_dataset(client, "http://test/api/v1", "t", package, state, ident, poll_timeout=5)
    assert job["status"] == "completed" and server.posts["import"] == 1
    assert json.loads(state.read_text())["phase"] == "imported"

    grouped.import_dataset(client, "http://test/api/v1", "t", package, state, ident, poll_timeout=5)
    assert server.posts["import"] == 1, "rerun must only poll the existing job"


def test_uncertain_import_is_not_resent(tmp_path):
    package = make_package(tmp_path)
    server = FakeServer(fail_post="import"); client = client_for(server)
    state = tmp_path / "state.json"; ident = identity(package, tmp_path)

    with pytest.raises(grouped.ContractError):
        grouped.import_dataset(client, "http://test/api/v1", "t", package, state, ident, poll_timeout=5)
    saved = json.loads(state.read_text())
    assert saved["uncertain"] is True and "import_job_id" not in saved

    with pytest.raises(grouped.ContractError) as excinfo:
        grouped.import_dataset(client, "http://test/api/v1", "t", package, state, ident, poll_timeout=5)
    assert "uncertain" in str(excinfo.value).lower()
    assert server.posts["import"] == 1


def test_freeze_reuses_existing_frozen_version_and_rejects_conflict(tmp_path):
    package = make_package(tmp_path)
    server = FakeServer(); server.version_state = "frozen"; client = client_for(server)
    state = tmp_path / "state.json"; ident = identity(package, tmp_path)
    grouped.import_dataset(client, "http://test/api/v1", "t", package, state, ident, poll_timeout=5)
    saved = json.loads(state.read_text())

    result = grouped.freeze_version(client, "http://test/api/v1", "t", state, ident, poll_timeout=5)
    assert result["reused"] is True and server.posts["freeze"] == 0

    server.version_config = {"seed": 7, "group_level": "image_cluster"}
    with pytest.raises(grouped.ContractError):
        grouped.freeze_version(client, "http://test/api/v1", "t", state, ident, poll_timeout=5)
    assert server.posts["freeze"] == 0


def test_freeze_submits_once_and_records_job(tmp_path):
    package = make_package(tmp_path)
    server = FakeServer(); client = client_for(server)
    state = tmp_path / "state.json"; ident = identity(package, tmp_path)
    grouped.import_dataset(client, "http://test/api/v1", "t", package, state, ident, poll_timeout=5)

    result = grouped.freeze_version(client, "http://test/api/v1", "t", state, ident, poll_timeout=5)
    assert result["reused"] is False and server.posts["freeze"] == 1
    saved = json.loads(state.read_text())
    assert saved["phase"] == "frozen" and saved["freeze_job_id"] == "freeze-job-1"


def test_package_digest_mismatch_is_rejected(tmp_path):
    package = make_package(tmp_path)
    preparation = json.loads((package / "PREPARATION.json").read_text())
    preparation["archive_sha256"] = "0" * 64
    (package / "PREPARATION.json").write_text(json.dumps(preparation))
    with pytest.raises(grouped.ContractError):
        grouped.package_identity(package, "http://test", "run-1")


def test_state_identity_mismatch_is_rejected(tmp_path):
    package = make_package(tmp_path)
    server = FakeServer(); client = client_for(server)
    state = tmp_path / "state.json"; ident = identity(package, tmp_path)
    grouped.import_dataset(client, "http://test/api/v1", "t", package, state, ident, poll_timeout=5)

    other = dict(ident); other["archive_sha256"] = "b" * 64
    with pytest.raises(grouped.ContractError):
        grouped.import_dataset(client, "http://test/api/v1", "t", package, state, other, poll_timeout=5)


def test_pagination_incomplete_is_rejected(tmp_path):
    package = make_package(tmp_path)
    server = FakeServer(count=4); client = client_for(server)
    state = tmp_path / "state.json"; ident = identity(package, tmp_path)
    grouped.import_dataset(client, "http://test/api/v1", "t", package, state, ident, poll_timeout=5)

    original = server.handler

    def truncating(request):
        if "/assets" in request.url.path:
            return httpx.Response(200, json=envelope({"items": [], "total": 4, "page": 1, "page_size": 200}))
        return original(request)

    client = httpx.Client(transport=httpx.MockTransport(truncating), timeout=5)
    with pytest.raises(grouped.ContractError) as excinfo:
        grouped.verify_package(client, "http://test/api/v1", "t", package, state, ident, tmp_path)
    assert "pagination" in str(excinfo.value)


def test_verify_detects_missing_and_duplicate_assets(tmp_path):
    package = make_package(tmp_path)
    server = FakeServer(count=4); client = client_for(server)
    state = tmp_path / "state.json"; ident = identity(package, tmp_path)
    grouped.import_dataset(client, "http://test/api/v1", "t", package, state, ident, poll_timeout=5)

    server.asset_mutator = lambda items: items[:-1]
    with pytest.raises(grouped.ContractError):
        grouped.verify_package(client, "http://test/api/v1", "t", package, state, ident, tmp_path)

    server.asset_mutator = lambda items: items + [dict(items[0])]
    with pytest.raises(grouped.ContractError):
        grouped.verify_package(client, "http://test/api/v1", "t", package, state, ident, tmp_path)


def test_verify_detects_group_or_split_leak(tmp_path):
    package = make_package(tmp_path)
    server = FakeServer(count=4); client = client_for(server)
    state = tmp_path / "state.json"; ident = identity(package, tmp_path)
    grouped.import_dataset(client, "http://test/api/v1", "t", package, state, ident, poll_timeout=5)

    def leak(items):
        items[1]["source_group"] = "different-group"
        return items

    server.asset_mutator = leak
    with pytest.raises(grouped.ContractError):
        grouped.verify_package(client, "http://test/api/v1", "t", package, state, ident, tmp_path)
