"""Rollback-safe staging for managed dataset and model paths."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional


def _quarantine_root(target: Path) -> Path:
    return Path(target).parent / ".delete-quarantine"


def stage_path_for_deletion(target: Path, staged_path: Optional[Path] = None) -> Optional[Path]:
    """Atomically move a managed path into a sibling quarantine directory.

    A caller may provide a deterministic staged path when a persistent recovery
    journal needs to record the destination before the move occurs.
    """
    target = Path(target)
    if not target.exists():
        return None

    quarantine_root = _quarantine_root(target)
    quarantine_root.mkdir(parents=True, exist_ok=True)
    if staged_path is None:
        staged = quarantine_root / f"{target.name}-{uuid.uuid4().hex}"
    else:
        staged = Path(staged_path)
        if staged.parent.resolve() != quarantine_root.resolve():
            raise ValueError("staged deletion path must stay inside the sibling quarantine directory")
    if staged.exists():
        raise FileExistsError(f"deletion quarantine target already exists: {staged}")

    target.rename(staged)
    return staged


def restore_staged_path(staged: Optional[Path], target: Path) -> None:
    """Restore a staged path without ever overwriting a newly-created target."""
    if staged is None:
        return
    staged = Path(staged)
    target = Path(target)
    if not staged.exists():
        return
    if target.exists():
        raise FileExistsError(f"restore target already exists; staged copy kept at: {staged}")
    staged.rename(target)


def purge_staged_path(staged: Optional[Path]) -> None:
    """Permanently remove a staged path only after its database transaction commits."""
    if staged is None:
        return
    staged = Path(staged)
    if staged.is_dir():
        shutil.rmtree(staged)
    elif staged.exists():
        staged.unlink()


stage_directory_for_deletion = stage_path_for_deletion
restore_staged_directory = restore_staged_path
purge_staged_directory = purge_staged_path
