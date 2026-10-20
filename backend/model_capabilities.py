"""Model capabilities from explicit metadata and legacy weight basenames."""

import re
from typing import Any


def _weight_names(references):
    for reference in references:
        if isinstance(reference, dict):
            values = [reference.get(key) for key in (
                "base_model", "lineage_root_model", "parent_model", "path", "name",
            )]
        else:
            values = [reference]
        for value in values:
            yield str(value or "").replace("\\", "/").rsplit("/", 1)[-1].lower()


def model_supports_text_prompts(*references: Any) -> bool:
    """Registry declarations outrank filename hints; directory names are ignored."""
    metadata = [value for value in references if isinstance(value, dict)]
    for value in metadata:
        family = str(value.get("model_family") or "").strip().lower()
        if family in {"yolo", "yoloe"}:
            return family == "yoloe"
    for value in metadata:
        profile = str(value.get("training_profile") or "").strip().lower()
        if profile in {"standard_yolo", "yoloe_pe_segment"}:
            return profile == "yoloe_pe_segment"
    for value in metadata:
        capability = value.get("supports_text_prompts")
        if isinstance(capability, bool):
            return capability
    return any(re.match(r"^yoloe(?:[-_.]|$)", name) for name in _weight_names(references))


def model_annotation_geometry(*references: Any) -> str:
    """Do not infer masks from an experiment's descriptive name."""
    for value in references:
        if not isinstance(value, dict):
            continue
        geometry = str(value.get("geometry") or "").strip().lower()
        if geometry in {"rect", "bbox"}:
            return "rect"
        if geometry in {"polygon", "segmentation", "segment"}:
            return "polygon"
        task = str(value.get("task") or "").strip().lower()
        if task in {"detect", "segment"}:
            return "polygon" if task == "segment" else "rect"
        if isinstance(value.get("supports_segmentation"), bool):
            return "polygon" if value["supports_segmentation"] else "rect"
    return "polygon" if any(
        re.search(r"(?:^|[-_.])seg(?:ment)?(?:[-_.]|$)", name)
        for name in _weight_names(references)
    ) else "rect"
