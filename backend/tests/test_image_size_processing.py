from image_size_processing import MAX_TILE_CANDIDATES, MAX_TILES_PER_IMAGE, MOBILEPHONE_CONTEXT_POLICY_VERSION, SUPPORTED_IMAGE_SIZES, plan_annotation_tiles, process_image_for_training, render_annotation_tile, resolve_tiling_profile, select_adaptive_tile_size, select_image_size, summarize_image_size_results
from image_size_processing import resolve_tiling_profile
from image_size_processing import MOBILEPHONE_CONTEXT_POLICY_VERSION
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_size_processing import (
    SUPPORTED_IMAGE_SIZES,
    process_image_for_training,
    select_image_size,
    summarize_image_size_results,
)


class ImageSizeProcessingTests(unittest.TestCase):
    def _image(self, directory: str, name: str, size=(1600, 900)) -> Path:
        path = Path(directory) / name
        Image.new("RGB", size, (30, 60, 90)).save(path)
        return path

    def test_selects_largest_supported_size_without_unnecessary_upscale(self):
        self.assertEqual(select_image_size(1920, 1080), 640)
        self.assertEqual(select_image_size(1024, 800), 640)
        self.assertEqual(select_image_size(200, 100), 320)
        self.assertEqual(SUPPORTED_IMAGE_SIZES, (320, 416, 512, 640, 1280))

    def test_crop_keeps_bbox_and_transforms_to_square(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self._image(directory, "source.jpg")
            target = Path(directory) / "target.jpg"
            result = process_image_for_training(
                source,
                target,
                [{"yolo_line": "0 0.50 0.50 0.20 0.40"}],
            )
            self.assertEqual(result["operation"], "crop")
            self.assertEqual(result["target_size"], 640)
            self.assertTrue(result["all_annotations_preserved"])
            annotation = result["annotations"][0]
            self.assertAlmostEqual(annotation["x"], 0.5, places=5)
            self.assertAlmostEqual(annotation["y"], 0.5, places=5)
            self.assertAlmostEqual(annotation["width"], 0.355555, places=4)
            self.assertAlmostEqual(annotation["height"], 0.4, places=5)
            with Image.open(target) as output:
                self.assertEqual(output.size, (640, 640))

    def test_letterboxes_when_square_crop_would_cut_bbox(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self._image(directory, "source.jpg")
            target = Path(directory) / "target.jpg"
            result = process_image_for_training(
                source,
                target,
                [{"yolo_line": "0 0.50 0.50 1.00 0.80"}],
            )
            self.assertEqual(result["operation"], "letterbox")
            annotation = result["annotations"][0]
            self.assertAlmostEqual(annotation["x"], 0.5, places=5)
            self.assertAlmostEqual(annotation["width"], 1.0, places=5)
            self.assertGreater(annotation["height"], 0.0)
            with Image.open(target) as output:
                self.assertEqual(output.size, (640, 640))

    def test_polygon_is_preserved_without_partial_vertices(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self._image(directory, "source.png")
            target = Path(directory) / "target.png"
            result = process_image_for_training(
                source,
                target,
                [{"yolo_line": "2 0.40 0.30 0.60 0.30 0.60 0.70 0.40 0.70"}],
            )
            self.assertEqual(result["operation"], "crop")
            polygon = result["annotations"][0]
            self.assertEqual(polygon["type"], "polygon")
            self.assertEqual(len(polygon["points"]), 8)
            self.assertTrue(all(0 <= value <= 1 for value in polygon["points"]))

    def test_summary_records_100_percent_retention(self):
        summary = summarize_image_size_results(
            [
                {"target_size": 640, "operation": "crop", "all_annotations_preserved": True},
                {"target_size": 640, "operation": "letterbox", "all_annotations_preserved": True},
            ]
        )
        self.assertEqual(summary["annotation_retention"], "100_percent")
        self.assertTrue(summary["all_annotations_preserved"])
        self.assertEqual(summary["target_size_counts"], {"640": 2})

    def test_summary_records_annotation_retention(self):
        summary = summarize_image_size_results([
            {"target_size": 640, "operation": "crop", "all_annotations_preserved": True},
            {"target_size": 640, "operation": "letterbox", "all_annotations_preserved": True},
        ])
        self.assertEqual(summary["annotation_retention"], "100_percent")
        self.assertEqual(summary["target_size_counts"], {"640": 2})

    def test_adaptive_padding_uses_small_medium_and_large_thresholds(self):
        cases = (
            ("0 0.5 0.5 0.1 0.1", 150),
            ("0 0.5 0.5 0.2 0.2", 400),
            ("0 0.5 0.5 0.32 0.32", 832),
        )
        for yolo_line, expected_side in cases:
            with self.subTest(yolo_line=yolo_line):
                plans = plan_annotation_tiles(1000, 1000, [{"yolo_line": yolo_line}])
                self.assertEqual(len(plans), 1)
                self.assertFalse(plans[0]["fallback"])
                crop_box = plans[0]["crop_box"]
                self.assertEqual(crop_box[2] - crop_box[0], expected_side)

    def test_mobilephone_context_profile_keeps_a_larger_context_window(self):
        plans = plan_annotation_tiles(
            1000,
            1000,
            [{"yolo_line": "0 0.5 0.5 0.1 0.1"}],
            profile=MOBILEPHONE_CONTEXT_POLICY_VERSION,
            target_class_ids=[0],
        )
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]["policy_version"], MOBILEPHONE_CONTEXT_POLICY_VERSION)
        self.assertEqual(plans[0]["crop_box"][2] - plans[0]["crop_box"][0], 300)
        self.assertEqual(plans[0]["covered_annotation_indexes"], [0])

    def test_unknown_tiling_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知图像导出策略"):
            resolve_tiling_profile("mobilephone_context_v99")

    def test_mobilephone_profile_uses_640_and_1280_candidates(self):
        policy = resolve_tiling_profile(MOBILEPHONE_CONTEXT_POLICY_VERSION)
        plans = [{"annotation_short_side_ratios": [0.02]}]
        self.assertEqual(
            select_adaptive_tile_size(
                plans,
                candidates=policy["target_size_candidates"],
                minimum_annotation_pixels=policy["minimum_annotation_pixels"],
            ),
            1280,
        )

    def test_mobilephone_target_size_ignores_non_target_annotation_ratios(self):
        self.assertEqual(
            select_adaptive_tile_size(
                [{
                    "annotation_short_side_ratios": [0.001],
                    "target_annotation_short_side_ratios": [0.1],
                }],
                candidates=[640, 1280],
                minimum_annotation_pixels=32,
                ratio_field="target_annotation_short_side_ratios",
            ),
            640,
        )

    def test_mobilephone_context_keeps_edge_and_neighbor_annotations_complete(self):
        annotations = [
            {"yolo_line": "0 0.08 0.12 0.08 0.10"},
            {"yolo_line": "1 0.16 0.14 0.08 0.08"},
        ]
        plans = plan_annotation_tiles(
            1200,
            900,
            annotations,
            profile=MOBILEPHONE_CONTEXT_POLICY_VERSION,
            target_class_ids=[0],
        )
        covered = {index for plan in plans for index in plan["covered_annotation_indexes"]}
        self.assertEqual(covered, {0, 1})
        self.assertTrue(all(plan["crop_box"][0] >= 0 and plan["crop_box"][1] >= 0 for plan in plans if plan["crop_box"]))

    def test_mobilephone_context_falls_back_for_oversized_target(self):
        plans = plan_annotation_tiles(
            1000,
            1000,
            [{"yolo_line": "0 0.5 0.5 0.9 0.9"}],
            profile=MOBILEPHONE_CONTEXT_POLICY_VERSION,
            target_class_ids=[0],
        )
        self.assertEqual(len(plans), 1)
        self.assertTrue(plans[0]["fallback"])
        self.assertEqual(plans[0]["operation"], "letterbox")
        self.assertEqual(plans[0]["covered_annotation_indexes"], [0])

    def test_empty_annotations_are_rejected_by_tile_planner(self):
        with self.assertRaisesRegex(ValueError, "无标注图片"):
            plan_annotation_tiles(1000, 1000, [], profile=MOBILEPHONE_CONTEXT_POLICY_VERSION, target_class_ids=[0])

    def test_edge_target_is_shifted_without_being_clipped(self):
        plans = plan_annotation_tiles(1000, 800, [{"yolo_line": "0 0.05 0.05 0.1 0.1"}])
        self.assertEqual(plans[0]["crop_box"][:2], [0, 0])
        self.assertEqual(plans[0]["covered_annotation_indexes"], [0])

    def test_partial_neighbor_is_fully_included(self):
        annotations = [
            {"yolo_line": "0 0.40 0.50 0.10 0.10"},
            {"yolo_line": "0 0.46 0.50 0.10 0.10"},
        ]
        plans = plan_annotation_tiles(1000, 1000, annotations)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]["covered_annotation_indexes"], [0, 1])

    def test_distant_targets_are_all_covered_deterministically(self):
        annotations = [
            {"yolo_line": "0 0.15 0.15 0.05 0.05"},
            {"yolo_line": "1 0.85 0.85 0.05 0.05"},
        ]
        first = plan_annotation_tiles(1200, 800, annotations)
        second = plan_annotation_tiles(1200, 800, annotations)
        self.assertEqual(first, second)
        covered = {
            index
            for plan in first
            for index in plan["covered_annotation_indexes"]
        }
        self.assertEqual(covered, {0, 1})
        self.assertTrue(all(plan["covered_annotation_indexes"] for plan in first))

    def test_hard_limit_reserves_fallback_for_uncovered_targets(self):
        annotations = [
            {"yolo_line": f"0 {(index + 0.5) / 40:.6f} 0.5 0.005 0.05"}
            for index in range(40)
        ]
        plans = plan_annotation_tiles(10000, 1000, annotations)
        self.assertEqual(len(plans), MAX_TILES_PER_IMAGE)
        covered = {
            index
            for plan in plans
            for index in plan["covered_annotation_indexes"]
        }
        self.assertEqual(covered, set(range(len(annotations))))
        self.assertTrue(plans[-1]["fallback"])

    def test_soft_limit_merges_remaining_targets_when_possible(self):
        annotations = [
            {
                "yolo_line":
                    f"0 {(column + 0.5) / 7:.6f} {(row + 0.5) / 7:.6f} 0.01 0.01"
            }
            for row in range(7)
            for column in range(7)
        ]
        plans = plan_annotation_tiles(7000, 7000, annotations)
        self.assertLess(len(plans), MAX_TILES_PER_IMAGE)
        self.assertFalse(any(plan["fallback"] for plan in plans))
        self.assertGreater(max(len(plan["covered_annotation_indexes"]) for plan in plans), 1)

    def test_candidate_generation_is_bounded_and_keeps_full_coverage(self):
        annotations = [
            {"yolo_line": f"0 {(index + 0.5) / 300:.8f} 0.5 0.001 0.05"}
            for index in range(300)
        ]
        self.assertLess(MAX_TILE_CANDIDATES, len(annotations))
        plans = plan_annotation_tiles(30000, 1000, annotations)
        self.assertLessEqual(len(plans), MAX_TILES_PER_IMAGE)
        self.assertEqual(
            {
                index
                for plan in plans
                for index in plan["covered_annotation_indexes"]
            },
            set(range(len(annotations))),
        )
        self.assertTrue(plans[-1]["fallback"])

    def test_selects_one_dataset_size_for_annotation_pixel_coverage(self):
        plans = [{"annotation_short_side_ratios": [0.05] * 10}]
        self.assertEqual(select_adaptive_tile_size(plans), 512)

    def test_renders_bbox_and_polygon_crop_with_valid_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self._image(directory, "source.jpg", size=(1000, 1000))
            target = Path(directory) / "tile.jpg"
            annotations = [
                {"yolo_line": "0 0.30 0.30 0.10 0.10"},
                {"yolo_line": "1 0.25 0.25 0.35 0.25 0.35 0.35 0.25 0.35"},
            ]
            plan = plan_annotation_tiles(1000, 1000, annotations)[0]
            result = render_annotation_tile(source, target, annotations, plan, target_size=640)
            self.assertTrue(target.is_file())
            self.assertEqual(result["target_size"], 640)
            self.assertGreaterEqual(result["annotation_count"], 1)
            summary = summarize_image_size_results([result])
            self.assertEqual(summary["policy"], "dataset_adaptive_annotation_pixels")
            self.assertEqual(summary["selection"], "smallest_candidate_meeting_90_percent_at_24px")
            for annotation in result["annotations"]:
                if annotation["type"] == "polygon":
                    self.assertTrue(all(0 <= value <= 1 for value in annotation["points"]))
                else:
                    self.assertTrue(0 < annotation["width"] <= 1)
                    self.assertTrue(0 < annotation["height"] <= 1)


if __name__ == "__main__":
    unittest.main()
