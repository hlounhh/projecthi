import math
import unittest

from ai.pose_detector import parse_pose, _pose_getattr


class PoseParsingTests(unittest.TestCase):
    def test_preserves_coco17_with_confidence(self):
        points = [(10., 20., 0.9)] * 17
        result = parse_pose([[0, 0, 100, 200, 0.8, 0]], [points], (480, 640, 3))
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].keypoints), 17)
        self.assertEqual(result[0].keypoints[0], (10., 20., 0.9))

    def test_invalid_keypoint_is_marked_invisible(self):
        points = [(10., 20., 0.9)] * 16 + [(math.nan, 20., 1.)]
        result = parse_pose([[0, 0, 100, 200, 0.8, 0]], [points], (480, 640, 3))
        self.assertEqual(result[0].keypoints[-1], (0., 0., 0.))

    def test_invalid_geometry_and_shape_are_rejected(self):
        points = [(10., 20., 0.9)] * 17
        self.assertEqual(parse_pose([[100, 0, 0, 200, 0.8, 0]], [points], (480, 640, 3)), ())
        self.assertEqual(parse_pose([[0, 0, 100, 200, 0.8, 0]], [points[:10]], (480, 640, 3)), ())


if __name__ == "__main__":
    unittest.main()
