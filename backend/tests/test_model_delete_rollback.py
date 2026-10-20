import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import model_lifecycle
from database import JobEvent, JobTask, ModelRegistry
from dataset_deletion import restore_staged_path, stage_path_for_deletion
from model_lifecycle import (
    ModelDeletionJournal,
    delete_model_and_linked_training_task,
    delete_training_task_and_models,
    stage_registered_model_files,
)
from model_usage import payload_references_model


def _record(record_id, path, task_id=None):
    return ModelRegistry(
        id=record_id,
        name=Path(path).name,
        path=str(path),
        sha256=("a" * 64),
        size=1,
        source="trained",
        status="active",
        format="pt",
        task_id=task_id,
        created_by="test",
        metadata_json={},
    )


def _db():
    engine = create_engine("sqlite:///:memory:")
    JobTask.__table__.create(engine)
    JobEvent.__table__.create(engine)
    ModelRegistry.__table__.create(engine)
    return sessionmaker(bind=engine, future=True)()


def test_second_stage_failure_restores_first_file(tmp_path, monkeypatch):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    records = [
        SimpleNamespace(id="m1", name="first.pt", status="active", updated_at=datetime.now()),
        SimpleNamespace(id="m2", name="second.pt", status="active", updated_at=datetime.now()),
    ]
    paths = {"m1": first, "m2": second}
    real_stage = model_lifecycle.stage_path_for_deletion
    calls = {"count": 0}

    def flaky_stage(path, staged_path=None):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("injected second move failure")
        return real_stage(path, staged_path)

    monkeypatch.setenv("MODEL_DELETE_JOURNAL_DIR", str(tmp_path / "journals"))
    monkeypatch.setattr(model_lifecycle, "stage_path_for_deletion", flaky_stage)
    journal = ModelDeletionJournal("test", "stage-failure")

    with pytest.raises(OSError):
        stage_registered_model_files(
            records,
            lambda item: paths[item.id],
            lambda _record, _path: False,
            journal=journal,
        )

    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"
    assert records[0].status == "active"
    assert journal.data["state"] == "rolled_back"


def test_restore_conflict_keeps_both_copies(tmp_path):
    target = tmp_path / "model.pt"
    target.write_bytes(b"old")
    staged = stage_path_for_deletion(target)
    target.write_bytes(b"new")

    with pytest.raises(FileExistsError):
        restore_staged_path(staged, target)

    assert target.read_bytes() == b"new"
    assert staged.read_bytes() == b"old"


def test_database_commit_failure_restores_model_and_record(tmp_path, monkeypatch):
    db = _db()
    model_file = tmp_path / "trained.pt"
    model_file.write_bytes(b"weights")
    task = JobTask(
        id="task-1",
        type="training",
        status="completed",
        stage="completed",
        progress=100,
        message="done",
        owner_username="tester",
        input_json={},
        result_json={},
    )
    record = _record("model-1", model_file, "task-1")
    db.add_all([task, record])
    db.commit()

    monkeypatch.setenv("MODEL_DELETE_JOURNAL_DIR", str(tmp_path / "journals"))

    def fail_commit():
        raise RuntimeError("injected commit failure")

    monkeypatch.setattr(db, "commit", fail_commit)

    with pytest.raises(RuntimeError):
        delete_training_task_and_models(
            db,
            "task-1",
            resolve_path=lambda item: Path(item.path),
            is_protected=lambda _record, _path: False,
        )

    assert model_file.read_bytes() == b"weights"
    db.expire_all()
    assert db.query(JobTask).filter(JobTask.id == "task-1").first() is not None
    assert db.query(ModelRegistry).filter(ModelRegistry.id == "model-1").one().status == "active"


def test_cleanup_failure_is_reported_after_database_commit(tmp_path, monkeypatch):
    db = _db()
    model_file = tmp_path / "standalone.pt"
    model_file.write_bytes(b"weights")
    record = _record("model-2", model_file)
    db.add(record)
    db.commit()

    monkeypatch.setenv("MODEL_DELETE_JOURNAL_DIR", str(tmp_path / "journals"))

    def fail_purge(_staged):
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(model_lifecycle, "purge_staged_path", fail_purge)
    result = delete_model_and_linked_training_task(
        db,
        record,
        resolve_path=lambda item: Path(item.path),
        is_protected=lambda _record, _path: False,
    )

    assert result["cleanup_pending"] is True
    assert result["operation_id"]
    assert result["cleanup_failures"]
    assert not model_file.exists()
    assert db.query(ModelRegistry).filter(ModelRegistry.id == "model-2").one().status == "deleted"


def test_nested_runtime_payload_detects_model_reference():
    candidates = {"/app/data/models/custom.pt", "custom.pt", "model-3"}
    payload = {
        "shared_model": "/app/data/models/custom.pt",
        "shared_behavior_config": {"pose_model_path": "pose.pt"},
    }
    assert payload_references_model(payload, candidates)
    assert not payload_references_model({"shared_model": "other.pt"}, candidates)


def test_http_delete_does_not_directly_unlink_unregistered_models():
    source = (Path(__file__).resolve().parents[1] / "yolo_api.py").read_text(encoding="utf-8")
    start = source.index('async def delete_model(')
    end = source.index('@app.delete("/datasets/{dataset_id}")', start)
    route_source = source[start:end]
    assert "full_path.unlink()" not in route_source
    assert "模型尚未纳入模型登记" in route_source
