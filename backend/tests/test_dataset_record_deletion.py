import unittest
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import DatasetAsset, DatasetVersion, ModelRegistry
from dataset_versioning import delete_dataset_version_records


TEST_ENGINE = create_engine("sqlite+pysqlite:///:memory:", future=True)
TestSession = sessionmaker(bind=TEST_ENGINE, future=True)


class DatasetRecordDeletionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DatasetVersion.__table__.create(TEST_ENGINE, checkfirst=True)
        DatasetAsset.__table__.create(TEST_ENGINE, checkfirst=True)
        ModelRegistry.__table__.create(TEST_ENGINE, checkfirst=True)

    def setUp(self):
        self.db = TestSession()
        self.dataset_id = f"delete-check-{uuid.uuid4()}"
        self.original_version_id = str(uuid.uuid4())
        self.training_version_id = str(uuid.uuid4())

        self.db.add_all([
            self.make_version(self.original_version_id, "original"),
            self.make_version(self.training_version_id, "training"),
            self.make_asset(1, self.original_version_id),
            self.make_asset(2, self.training_version_id),
            self.make_model("original-model", self.original_version_id),
            self.make_model("training-model", self.training_version_id),
        ])
        self.db.commit()

    def tearDown(self):
        self.db.query(ModelRegistry).filter(ModelRegistry.name.in_(["original-model", "training-model"])).delete(synchronize_session=False)
        version_ids = [self.original_version_id, self.training_version_id]
        self.db.query(DatasetAsset).filter(DatasetAsset.dataset_version_id.in_(version_ids)).delete(synchronize_session=False)
        self.db.query(DatasetVersion).filter(DatasetVersion.id.in_(version_ids)).delete(synchronize_session=False)
        self.db.commit()
        self.db.close()

    def make_version(self, version_id, role):
        return DatasetVersion(
            id=version_id,
            dataset_id=self.dataset_id,
            dataset_role=role,
            status="ready",
            root_path=f"/tmp/{self.dataset_id}",
            manifest_path=f"/tmp/{version_id}.json",
            class_map_json={},
            split_config_json={},
            metadata_json={},
            content_hash=uuid.uuid4().hex,
            created_by="test",
        )

    @staticmethod
    def make_asset(asset_id, version_id):
        return DatasetAsset(
            id=asset_id,
            dataset_version_id=version_id,
            relative_path=f"images/{asset_id}.jpg",
            basename=f"{asset_id}.jpg",
            size=1,
            width=1,
            height=1,
            content_hash=uuid.uuid4().hex,
            annotations_json=[],
        )

    @staticmethod
    def make_model(name, version_id):
        return ModelRegistry(
            id=str(uuid.uuid4()),
            name=name,
            path=f"/tmp/{uuid.uuid4()}.pt",
            sha256=uuid.uuid4().hex,
            size=1,
            source="training",
            status="active",
            format="pt",
            dataset_version_id=version_id,
            created_by="test",
            metadata_json={},
        )

    def test_delete_removes_only_matching_role_and_clears_model_reference(self):
        deleted = delete_dataset_version_records(self.db, self.dataset_id, "original")
        self.db.commit()

        self.assertEqual(deleted, 1)
        self.assertIsNone(self.db.get(DatasetVersion, self.original_version_id))
        self.assertIsNotNone(self.db.get(DatasetVersion, self.training_version_id))
        self.assertEqual(
            self.db.query(DatasetAsset).filter(DatasetAsset.dataset_version_id == self.original_version_id).count(),
            0,
        )
        self.assertEqual(
            self.db.query(DatasetAsset).filter(DatasetAsset.dataset_version_id == self.training_version_id).count(),
            1,
        )
        self.assertIsNone(self.db.query(ModelRegistry).filter(ModelRegistry.name == "original-model").one().dataset_version_id)
        self.assertEqual(
            self.db.query(ModelRegistry).filter(ModelRegistry.name == "training-model").one().dataset_version_id,
            self.training_version_id,
        )


if __name__ == "__main__":
    unittest.main()
