import tempfile
import unittest
from pathlib import Path

from dataset_deletion import (
    purge_staged_directory,
    restore_staged_directory,
    stage_directory_for_deletion,
)


class DatasetDeleteQuarantineChecks(unittest.TestCase):
    def test_restore_returns_staged_directory_after_transaction_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "dataset"
            target.mkdir()
            (target / "asset.txt").write_text("asset", encoding="utf-8")

            staged = stage_directory_for_deletion(target)
            self.assertFalse(target.exists())
            self.assertTrue(staged.exists())

            restore_staged_directory(staged, target)
            self.assertTrue((target / "asset.txt").is_file())

    def test_purge_removes_only_staged_directory_after_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "dataset"
            target.mkdir()
            staged = stage_directory_for_deletion(target)

            purge_staged_directory(staged)
            self.assertFalse(staged.exists())
            self.assertTrue((target.parent / ".delete-quarantine").is_dir())

    def test_file_staging_supports_asset_rollback_and_purge(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "image.jpg"
            target.write_bytes(b"image")

            staged = stage_directory_for_deletion(target)
            self.assertFalse(target.exists())
            self.assertTrue(staged.is_file())

            restore_staged_directory(staged, target)
            self.assertEqual(target.read_bytes(), b"image")

            staged = stage_directory_for_deletion(target)
            purge_staged_directory(staged)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
