import ast
import re
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


SOURCE_PATH = Path(__file__).resolve().parents[1] / "yolo_api.py"
FUNCTION_NAMES = {
    "normalize_annotation_prompt_classes",
    "_coerce_annotation_class_id",
    "normalize_annotation_class_bindings",
    "load_annotation_inference_model",
    "apply_annotation_prompt_classes",
    "enforce_annotation_prompt_results",
    "build_annotation_batch_request_signature",
    "annotation_batch_job_matches_signature",
}
MODULE_AST = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
SELECTED_FUNCTIONS = [
    node
    for node in MODULE_AST.body
    if isinstance(node, ast.FunctionDef) and node.name in FUNCTION_NAMES
]
NAMESPACE = {
    "Any": Any,
    "Dict": Dict,
    "HTTPException": HTTPException,
    "List": List,
    "Optional": Optional,
    "Path": Path,
    "re": re,
    "constants": type("Constants", (), {"DEFAULT_CONFIDENCE": 0.25}),
    "annotation_model_supports_text_prompts": lambda model_path, metadata=None: True,
    "get_model_class_names": lambda model: dict(getattr(model, "names", {}) or {}),
}
exec(compile(ast.Module(body=SELECTED_FUNCTIONS, type_ignores=[]), str(SOURCE_PATH), "exec"), NAMESPACE)

load_annotation_inference_model = NAMESPACE["load_annotation_inference_model"]
apply_annotation_prompt_classes = NAMESPACE["apply_annotation_prompt_classes"]
enforce_annotation_prompt_results = NAMESPACE["enforce_annotation_prompt_results"]
build_annotation_batch_request_signature = NAMESPACE["build_annotation_batch_request_signature"]
annotation_batch_job_matches_signature = NAMESPACE["annotation_batch_job_matches_signature"]


class EmbeddingPromptModel:
    def __init__(self):
        self.calls = []
        self.names = {0: "person", 1: "badge"}

    def get_text_pe(self, prompts):
        self.calls.append(("get_text_pe", prompts))
        return "embeddings"

    def set_classes(self, prompts, embeddings):
        self.calls.append(("set_classes", prompts, embeddings))
        self.names = dict(enumerate(prompts))


class IgnoredPromptModel:
    names = {0: "person", 1: "badge"}

    def set_classes(self, prompts):
        return None


class AnnotationPromptClassTests(unittest.TestCase):
    def test_prompted_models_use_explicit_yoloe_loader(self):
        ultralytics = types.ModuleType("ultralytics")
        ultralytics.YOLO = lambda path: ("YOLO", path)
        ultralytics.YOLOE = lambda path: ("YOLOE", path)

        with patch.dict(sys.modules, {"ultralytics": ultralytics}):
            self.assertEqual(
                load_annotation_inference_model("yoloe-26m-seg.pt", ["mobilephone"]),
                ("YOLOE", "yoloe-26m-seg.pt"),
            )
            self.assertEqual(
                load_annotation_inference_model("yolo11n.pt", []),
                ("YOLO", "yolo11n.pt"),
            )

    def test_prompt_vocabulary_is_applied_and_verified(self):
        model = EmbeddingPromptModel()

        applied = apply_annotation_prompt_classes(
            model,
            ["mobilephone"],
            supports_text_prompts=True,
        )

        self.assertEqual(applied, ["mobilephone"])
        self.assertEqual(model.names, {0: "mobilephone"})

    def test_non_prompt_model_and_ignored_set_classes_fail_closed(self):
        with self.assertRaises(HTTPException) as unsupported:
            apply_annotation_prompt_classes(
                EmbeddingPromptModel(),
                ["mobilephone"],
                supports_text_prompts=False,
            )
        self.assertEqual(unsupported.exception.status_code, 400)

        with self.assertRaises(HTTPException) as ignored:
            apply_annotation_prompt_classes(
                IgnoredPromptModel(),
                ["mobilephone"],
                supports_text_prompts=True,
            )
        self.assertEqual(ignored.exception.status_code, 500)

    def test_prompt_results_reject_person_and_badge(self):
        annotations = [
            {"model_class_name": "mobilephone", "class_name": "mobilephone"},
            {"model_class_name": "person", "class_name": "person"},
            {"model_class_name": "badge", "class_name": "badge"},
        ]

        with self.assertRaises(HTTPException) as raised:
            enforce_annotation_prompt_results(annotations, ["mobilephone"])

        self.assertIn("person", raised.exception.detail)
        self.assertIn("badge", raised.exception.detail)

    def test_batch_reuse_requires_complete_inference_signature(self):
        signature = build_annotation_batch_request_signature(
            "dataset-1",
            "",
            "/app/data/models/pretrained/yoloe-26m-seg.pt",
            0.25,
            ["mobilephone"],
            [0],
            None,
            ["mobilephone"],
            [],
            {"id": "model-1", "sha256": "abc", "supports_text_prompts": True},
        )
        matching_input = {
            "dataset_id": signature["dataset_id"],
            "search": signature["search"],
            "model_path": signature["model_path"],
            "confidence": signature["confidence"],
            "prompt_classes": signature["prompt_classes"],
            "prompt_page_indices": signature["prompt_page_indices"],
            "target_page_index": signature["target_page_index"],
            "annotation_classes": signature["annotation_classes"],
            "annotation_class_bindings": signature["annotation_class_bindings"],
            "model_metadata": {
                "id": "model-1",
                "sha256": "abc",
                "supports_text_prompts": True,
            },
        }

        self.assertTrue(annotation_batch_job_matches_signature(matching_input, signature))
        for field, value in (
            ("prompt_classes", ["person", "badge"]),
            ("confidence", 0.5),
            ("prompt_page_indices", [1]),
            ("target_page_index", 0),
        ):
            stale_input = {**matching_input, field: value}
            self.assertFalse(annotation_batch_job_matches_signature(stale_input, signature))


if __name__ == "__main__":
    unittest.main()
