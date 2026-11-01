import math
from pathlib import Path
import tempfile
import unittest

from ai.posture_detector import parse_detections, verify_weights


class PostureDetectorTests(unittest.TestCase):
    def test_parse_and_clip_model_boxes(self):
        result = parse_detections([
            [-10, -20, 500, 400, 0.8, 0],
            [20, 30, 40, 70, 0.6, 2],
        ], {0: "Fallen", 2: "Standing"}, (300, 400, 3))
        self.assertEqual(result[0].xyxy, (0, 0, 400, 300))
        self.assertEqual(result[0].label, "Fallen")
        self.assertEqual(result[1].confidence, 0.6)

    def test_prediction_suppresses_boxes_across_posture_classes(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from ai.posture_detector import PostureDetector

        detector = PostureDetector.__new__(PostureDetector)
        detector.model = Mock()
        detector.model.predict.return_value = [SimpleNamespace(boxes=None)]
        detector.image_size = 640
        detector.confidence = 0.4
        self.assertEqual(detector.predict(SimpleNamespace(shape=(480, 640, 3))), ())
        kwargs = detector.model.predict.call_args.kwargs
        self.assertTrue(kwargs["agnostic_nms"])
        self.assertEqual(kwargs["iou"], 0.5)
        self.assertEqual(kwargs["max_det"], 20)

    def test_invalid_boxes_do_not_escape(self):
        rows = [
            [0, 0, 10, 10, math.nan, 0], [0, 0, 10, 10, 0.8, 0.5],
            [10, 0, 0, 10, 0.8, 0], [0, 0, 10, 10, 1.5, 0],
            [0, 0, 10, 10, 0.8, 99], [0, 1],
        ]
        self.assertEqual(parse_detections(rows, {0: "Fallen"}, (300, 400, 3)), ())

    def test_reject_unverified_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            with self.assertRaises(ValueError):
                verify_weights(path)
            path.write_bytes(b"not a model")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_weights(path)


if __name__ == "__main__":
    unittest.main()
