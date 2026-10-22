import pytest
from pydantic import ValidationError
from app.schemas.training import TrainingParams, ultralytics_arguments


def test_explicit_zero_seed_and_augmentation_reach_actual_arguments():
    params = TrainingParams(seed=0, device="cuda:0", augmentation={"mosaic":0,"hsv_h":0,"close_mosaic":0},
                            early_stopping={"patience":0,"monitor":"val/mAP50-95"}).model_dump()
    args = ultralytics_arguments(params)
    assert args["seed"] == 0
    assert args["mosaic"] == args["hsv_h"] == args["close_mosaic"] == args["patience"] == 0
    assert args["device"] == "0"
    assert args["amp"] is False
    assert args["workers"] == 0
    assert "augmentation" not in args and "early_stopping" not in args


@pytest.mark.parametrize("payload", [
    {"unknown":1}, {"augmentation":{"typo":1}}, {"augmentation":{"mosaic":2}},
    {"early_stopping":{"monitor":"train/loss"}}, {"imgsz":65}, {"device":"0,1"},
])
def test_unsupported_or_invalid_experiment_parameter_rejected(payload):
    with pytest.raises(ValidationError):
        TrainingParams.model_validate(payload)
