import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("generate_lock", Path(__file__).parents[1] / "generate_lock.py")
lock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lock)


def report():
    return {"environment": {"sys_platform": "linux", "python_version": "3.12"}, "install": [
        {"metadata": {"name": "torch", "version": "2.6.0+cu124"}, "download_info": {
            "url": "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-linux_x86_64.whl",
            "archive_info": {"hashes": {"sha256": hashlib.sha256(b"wheel").hexdigest()}}}}]}


def render(value, **kwargs):
    return lock.render_lock(value, platform="linux", python_version="3.12", **kwargs)


def test_cuda_source_and_verified_wheel(tmp_path):
    (tmp_path / "torch-2.6.0+cu124-cp312-cp312-linux_x86_64.whl").write_bytes(b"wheel")
    result = render(report(), wheelhouse=tmp_path)
    assert "torch @ https://download.pytorch.org/whl/cu124/" in result
    assert "--hash=sha256:" in result


def test_offline_requires_verified_artifacts_and_preserves_provenance(tmp_path):
    with pytest.raises(ValueError, match="verified wheelhouse"):
        render(report(), offline=True)
    (tmp_path / "torch-2.6.0+cu124-cp312-cp312-linux_x86_64.whl").write_bytes(b"wheel")
    result = render(report(), offline=True, wheelhouse=tmp_path)
    assert "torch==2.6.0+cu124 --hash=sha256:" in result
    assert "# Verified source: https://download.pytorch.org/" in result
    assert "torch @" not in result


@pytest.mark.parametrize("fault", ["empty", "missing_hash", "source", "credentials", "target", "wheel_platform", "conflict"])
def test_reject_bad_reports(fault):
    value = report()
    item = value["install"][0]
    if fault == "empty": value["install"] = []
    elif fault == "missing_hash": item["download_info"]["archive_info"] = {}
    elif fault == "source": item["download_info"]["url"] = "https://untrusted.invalid/test.whl"
    elif fault == "credentials": item["download_info"]["url"] = "https://user:password@download.pytorch.org/test.whl"
    elif fault == "target": value["environment"]["python_version"] = "3.13"
    elif fault == "wheel_platform": item["download_info"]["url"] = item["download_info"]["url"].replace("linux_x86_64", "win_amd64")
    elif fault == "conflict":
        duplicate = copy.deepcopy(item)
        duplicate["metadata"]["version"] = "2.7.0"
        value["install"].append(duplicate)
    with pytest.raises(ValueError): render(value)


def test_failed_generation_preserves_last_good(tmp_path):
    target = tmp_path / "lock.txt"
    target.write_bytes(b"last-good\n")
    source = tmp_path / "report.json"
    source.write_text(json.dumps(report()), encoding="utf-8")
    with pytest.raises(ValueError, match="missing from wheelhouse"):
        lock.write_lock(source, target, platform="linux", python_version="3.12", wheelhouse=tmp_path)
    assert target.read_bytes() == b"last-good\n"


def test_corrupt_wheel_rejected(tmp_path):
    (tmp_path / "torch-2.6.0+cu124-cp312-cp312-linux_x86_64.whl").write_bytes(b"bad")
    with pytest.raises(ValueError, match="sha256 mismatch"): render(report(), wheelhouse=tmp_path)
