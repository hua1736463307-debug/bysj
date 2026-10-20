import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer
    from yolo_trainer import YOLOTrainer
    TRAINER_DEPENDENCIES = True
except ModuleNotFoundError:
    TRAINER_DEPENDENCIES = False


@unittest.skipUnless(TRAINER_DEPENDENCIES, "Ultralytics is unavailable")
class YOLOETrainerProfileTests(unittest.TestCase):
    def test_yoloe_load_uses_yoloe_wrapper(self):
        with patch("ultralytics.YOLOE") as yoloe:
            trainer = YOLOTrainer("yoloe-26m-seg.pt", training_profile="yoloe_pe_segment")
            trainer.load_model()
            yoloe.assert_called_once_with("yoloe-26m-seg.pt")

    def test_yoloe_train_passes_pe_seg_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = YOLOTrainer("yoloe-26m-seg.pt", training_profile="yoloe_pe_segment")
            trainer.model = MagicMock()
            trainer.model.train.return_value = MagicMock(save_dir=Path(directory) / "exp")
            trainer.data_config = str(Path(directory) / "dataset.yaml")
            with patch.object(trainer, "keep_only_best_pt"):
                trainer.train(save_dir=directory, name="exp")
            kwargs = trainer.model.train.call_args.kwargs
            self.assertIs(kwargs["trainer"], YOLOEPESegTrainer)
            self.assertEqual(kwargs["lr0"], 0.001)
            self.assertEqual(kwargs["epochs"], 150)

    def test_standard_train_does_not_override_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = YOLOTrainer("yolov8m-seg.pt", training_profile="standard_yolo")
            trainer.model = MagicMock()
            trainer.model.train.return_value = MagicMock(save_dir=Path(directory) / "exp")
            trainer.data_config = str(Path(directory) / "dataset.yaml")
            with patch.object(trainer, "keep_only_best_pt"):
                trainer.train(epochs=1, save_dir=directory, name="exp")
            self.assertNotIn("trainer", trainer.model.train.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
