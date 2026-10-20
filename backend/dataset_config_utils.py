"""Dataset class metadata helpers without model/runtime dependencies."""

from pathlib import Path
from typing import Iterable, List

import yaml


def read_dataset_class_names(config_path: Path) -> List[str]:
    config_path = Path(config_path)
    if not config_path.is_file():
        return []
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    names = payload.get("names", [])
    if isinstance(names, list):
        return [str(value) for value in names]
    if isinstance(names, dict):
        try:
            keys = sorted(names, key=lambda value: int(value))
        except (TypeError, ValueError):
            keys = sorted(names, key=str)
        return [str(names[key]) for key in keys]
    return []


def requested_classes_match_dataset(config_path: Path, requested: Iterable[str]) -> bool:
    expected = read_dataset_class_names(config_path)
    selected = [str(value) for value in (requested or [])]
    return bool(expected) and selected == expected
