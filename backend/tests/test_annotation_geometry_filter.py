import unittest

try:
    from dataset_versioning import filter_annotations_by_geometry
    DATASET_VERSIONING_AVAILABLE = True
except ModuleNotFoundError:
    DATASET_VERSIONING_AVAILABLE = False


@unittest.skipUnless(DATASET_VERSIONING_AVAILABLE, "dataset versioning dependencies are unavailable")
class AnnotationGeometryFilterTests(unittest.TestCase):
    def test_retains_only_requested_geometry(self):
        annotations = [
            {"type": "bbox", "yolo_line": "0 0.5 0.5 0.2 0.2"},
            {"type": "polygon", "yolo_line": "0 0.1 0.1 0.4 0.1 0.4 0.4"},
            {"yolo_line": "1 0.5 0.5 0.1 0.1"},
        ]

        self.assertEqual(
            filter_annotations_by_geometry(annotations, "bbox"),
            [annotations[0], annotations[2]],
        )
        self.assertEqual(
            filter_annotations_by_geometry(annotations, "polygon"),
            [annotations[1]],
        )

    def test_rejects_unknown_geometry(self):
        with self.assertRaisesRegex(ValueError, "bbox"):
            filter_annotations_by_geometry([], "mixed")


if __name__ == "__main__":
    unittest.main()
