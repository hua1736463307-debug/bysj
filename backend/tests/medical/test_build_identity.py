import hashlib
import json

from app import version


def test_build_identity_uses_current_locks_and_runtime_sha(tmp_path, monkeypatch):
    (tmp_path / "backend").mkdir()
    core = tmp_path / "backend/requirements-core-linux-py312.lock.txt"
    core.write_bytes(b"current core")
    (tmp_path / "version.json").write_text(json.dumps({"code_sha": "stale", "requirements_lock_sha256": "stale"}))
    monkeypatch.setattr(version, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("BYSJ_CODE_SHA", "a"*40)
    version.build_info.cache_clear()
    try:
        info = version.build_info()
        assert info["code_sha"] == "a"*40
        assert info["requirements_lock_sha256"] == hashlib.sha256(core.read_bytes()).hexdigest()
        assert info["ml_lock_sha256"] is None
    finally:
        version.build_info.cache_clear()


def test_repository_root_is_parent_of_backend():
    assert (version.REPO_ROOT / "backend/app/version.py").resolve() == __import__('pathlib').Path(version.__file__).resolve()
