from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import json
import uuid

from app.api.deps import CurrentUser
from app.core.config import get_settings
from app.core.errors import ApiError
from app.db.models import DatasetResource, DatasetVersion, DatasetAsset
from app.domains.annotation.service import save_annotations
from app.domains.datasets.version_service import freeze_version

USER = CurrentUser(1, "researcher", "researcher", "researcher")
BOX = {"client_id": "new", "class_id": 0, "cx": .5, "cy": .5, "w": .4, "h": .4}


def seed(sessions, tmp_path):
    with sessions() as db:
        resource = DatasetResource(dataset_id=uuid.uuid4().hex, owner_user_id=1, owner_username=USER.username)
        db.add(resource)
        db.flush()
        version = DatasetVersion(dataset_id=resource.id, root_path=str(tmp_path), class_map_json={"0": "RBC"})
        db.add(version)
        db.flush()
        for i in range(9):
            db.add(DatasetAsset(dataset_version_id=version.id, relative_path=f"{i}.jpg", basename=f"{i}.jpg",
                content_hash=f"{i:064x}", width=100, height=100, source_group=f"image:{i}",
                annotations_json=[{**BOX, "client_id": "imported", "w": .2}]))
        db.commit()
        return version.id, db.query(DatasetAsset).order_by(DatasetAsset.id).first().id


def test_first_save_same_revision_only_one_winner(sessions, tmp_path):
    version_id, asset_id = seed(sessions, tmp_path)
    barrier = Barrier(2)
    def save(_):
        with sessions() as db:
            barrier.wait(10)
            try:
                return save_annotations(db, asset_id, USER, expected_revision=0, boxes=[BOX])["revision"]
            except ApiError as exc:
                return exc.code
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(save, range(2)))
    assert results.count(1) == 1
    assert results.count("revision_conflict") == 1


def test_owner_and_admin_saving_same_shared_revision_only_one_wins(sessions, tmp_path):
    _, asset_id = seed(sessions, tmp_path)
    barrier = Barrier(2)
    admin = CurrentUser(2, "admin", "admin", "admin")

    def save(user):
        with sessions() as db:
            barrier.wait(10)
            try:
                return save_annotations(db, asset_id, user, expected_revision=0, boxes=[BOX])["revision"]
            except ApiError as exc:
                return exc.code

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(save, (USER, admin)))
    assert results.count(1) == 1
    assert results.count("revision_conflict") == 1


def test_freeze_uses_saved_labels_and_rejects_late_save(sessions, tmp_path):
    version_id, asset_id = seed(sessions, tmp_path)
    with sessions() as db:
        save_annotations(db, asset_id, USER, expected_revision=0, boxes=[BOX])
        version = db.get(DatasetVersion, version_id)
        freeze_version(db, get_settings().model_copy(update={"data_root": tmp_path.parent}), version=version, seed=42, ratios=[.8, .1, .1])
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["assets"][0]["annotations"][0]["w"] == .4
        assert (tmp_path / "labels/0.txt").read_text().strip() == "0 0.500000 0.500000 0.400000 0.400000"
        try:
            save_annotations(db, asset_id, USER, expected_revision=1, boxes=[])
        except ApiError as exc:
            assert exc.code == "version_frozen"
        else:
            raise AssertionError("frozen annotation changed")


def test_simultaneous_freeze_only_one_winner(sessions, tmp_path):
    version_id, _ = seed(sessions, tmp_path)
    barrier = Barrier(2)
    def freeze(_):
        with sessions() as db:
            version = db.get(DatasetVersion, version_id)
            barrier.wait(10)
            try:
                return freeze_version(db, get_settings().model_copy(update={"data_root": tmp_path.parent}), version=version, seed=42, ratios=[.8,.1,.1])["state"]
            except ApiError as exc:
                return exc.code
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(freeze, range(2)))
    assert results.count("frozen") == 1
    assert any(code in {"version_frozen", "version_invalid_state"} for code in results)
