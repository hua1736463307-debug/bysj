"""Helpers for resolving the dataset configuration used by a training job."""

from pathlib import Path
from functools import wraps
from threading import Lock


_TRAINING_SUBMISSION_LOCK = Lock()


def serialize_training_submission(function):
    """Keep submissions serialized while FastAPI runs them off the event loop."""
    @wraps(function)
    def submit(*args, **kwargs):
        with _TRAINING_SUBMISSION_LOCK:
            return function(*args, **kwargs)
    return submit


def can_reuse_prepared_segmentation_dataset(selected_version, source_version, profile: str) -> bool:
    """Reuse an explicitly registered segmentation set after annotation preflight.

    A newer annotation source still requires the usual workspace export.
    """
    metadata = selected_version.metadata_json or {}
    return (
        profile == "yoloe_pe_segment"
        and selected_version.dataset_role == "training"
        and selected_version.status == "ready"
        and bool(selected_version.id)
        and selected_version.id == source_version.id
        and metadata.get("required_geometry") == "polygon"
        and metadata.get("training_profile") == "yoloe_pe_segment"
        and metadata.get("training_task") == "segment"
    )


def resolve_training_dataset_config_path(dataset_path: str) -> Path:
    requested = Path(dataset_path)
    if requested.suffix.lower() in {".yaml", ".yml"}:
        return requested

    for filename in ("dataset.yaml", "data.yaml"):
        candidate = requested / filename
        if candidate.is_file():
            return candidate
    return requested / "dataset.yaml"
