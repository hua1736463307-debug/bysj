"""Task-scoped training workspace paths and guarded cleanup."""

import os
import shutil
from pathlib import Path
from typing import Optional


def get_training_workspace_root() -> Path:
    data_root = Path(os.getenv("DEEPVISION_DATA_DIR", "/app/data"))
    return Path(
        os.getenv(
            "DEEPVISION_TRAINING_WORKSPACES_DIR",
            str(data_root / "training_workspaces"),
        )
    ).resolve()


def cleanup_training_workspace(
    workspace_path: Optional[str],
    *,
    workspace_root: Optional[Path] = None,
) -> bool:
    if not workspace_path:
        return False

    root = Path(workspace_root or get_training_workspace_root()).resolve()
    target = Path(workspace_path).resolve()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"training workspace is outside the managed root: {target}") from exc

    if len(relative.parts) != 1:
        raise ValueError(f"training workspace must be a direct child of the managed root: {target}")
    if not target.exists():
        return False
    if not target.is_dir():
        raise ValueError(f"training workspace is not a directory: {target}")

    shutil.rmtree(target)
    return True
