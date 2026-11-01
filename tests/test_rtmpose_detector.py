import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ai.pose_detector import PosePerson
from ai.rtmpose_detector import RTMPoseDetector, parse_rtmpose, verify_models
from tools.download_rtmpose import download


class RTMPoseTests(unittest.TestCase):
    def test_preserves_detector_confidence_and_coco17(self):
        people = parse_rtmpose([[1, 2, 90, 100, .73]], [0],
                               [[(10., 20.)] * 17], [[.9] * 17], (120, 160, 3))
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0].confidence, .73)
        self.assertEqual(people[0].keypoints, ((10., 20., .9),) * 17)

    def test_filters_nonhuman_low_confidence_and_invalid_geometry(self):
        for row, label in [([1, 2, 90, 100, .73], 1), ([1, 2, 90, 100, .1], 0),
                           ([90, 2, 1, 100, .73], 0), ([1, 2, 90, 100, math.nan], 0)]:
            self.assertEqual(parse_rtmpose([row], [label], [[(10, 20)] * 17],
                                           [[.9] * 17], (120, 160, 3)), ())

    def test_invalid_simcc_point_is_invisible(self):
        points = [(10, 20)] * 16 + [(math.nan, 20)]
        person = parse_rtmpose([[1, 2, 90, 100, .73]], [0], [points], [[.9] * 17],
                              (120, 160, 3))[0]
        self.assertEqual(person.keypoints[-1], (0., 0., 0.))

    def test_crop_translates_visible_joints_only(self):
        detector = RTMPoseDetector.__new__(RTMPoseDetector)
        person = PosePerson((0, 1, 20, 30), .8, ((2, 3, .9),) * 16 + ((0, 0, 0),))
        detector.predict = Mock(return_value=(person,))
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        result = detector.predict_crop(image, (10.2, 20.3, 60.1, 80.2))[0]
        self.assertEqual(result.xyxy, (10, 21, 30, 50))
        self.assertEqual(result.keypoints[0], (12, 23, .9))
        self.assertEqual(result.keypoints[-1], (0., 0., 0.))
        self.assertEqual(detector.predict.call_args.args[0].shape, (61, 51, 3))

    def test_empty_or_nonfinite_crop_does_not_run_network(self):
        detector = RTMPoseDetector.__new__(RTMPoseDetector)
        detector.predict = Mock()
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        for bounds in [(10, 20, 5, 40), (0, 0, math.nan, 50), (200, 0, 250, 30)]:
            self.assertEqual(detector.predict_crop(image, bounds), ())
        detector.predict.assert_not_called()

    def test_no_detections_does_not_hallucinate_full_image_pose(self):
        detector = RTMPoseDetector.__new__(RTMPoseDetector)
        detector.confidence = .4
        detector._detector = Mock(return_value=(np.empty((1, 0, 5)), np.empty((1, 0))))
        detector._pose = Mock()
        self.assertEqual(detector.predict(np.zeros((80, 120, 3), dtype=np.uint8)), ())
        detector._pose.assert_not_called()

    def test_predict_maps_detector_and_simcc_back_to_source_coordinates(self):
        detector = RTMPoseDetector.__new__(RTMPoseDetector)
        detector.confidence = .4
        detector.pose_input_size = (192, 256)
        detector._detector = Mock(return_value=(
            np.array([[[41.6, 20.8, 208, 187.2, .8], [0, 0, 416, 416, .99]]]),
            np.array([[0, 1]])))
        detector._bbox_xyxy2cs = Mock(return_value=(np.array([60., 50.]), np.array([80., 80.])))
        detector._affine = Mock(return_value=(np.zeros((256, 192, 3)), np.array([80., 80.])))
        detector._pose = Mock(return_value=("simcc_x", "simcc_y"))
        detector._simcc = Mock(return_value=(np.array([[[192., 256.]] * 17]), np.array([[.9] * 17])))
        people = detector.predict(np.zeros((100, 200, 3), dtype=np.uint8))
        self.assertEqual(len(people), 1)
        np.testing.assert_allclose(people[0].xyxy, (20, 10, 100, 90))
        np.testing.assert_allclose(people[0].keypoints, [(60, 50, .9)] * 17)
        self.assertEqual(people[0].confidence, .8)
        detector._pose.assert_called_once()
        np.testing.assert_allclose(detector._pose.call_args.args[0][0, 0],
                                   -np.array([123.675, 116.28, 103.53]) / (58.395, 57.12, 57.375))

    def test_models_are_verified_before_runtime_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("ai.rtmpose_detector._Runtime") as runtime:
                with self.assertRaisesRegex(ValueError, "assets missing"):
                    RTMPoseDetector(model_dir=directory)
                runtime.assert_not_called()
            (Path(directory) / "yolox-tiny.onnx").write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_models(directory)

    def test_valid_local_downloader_is_idempotent_without_network(self):
        data = b"verified-model"
        digest = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model.onnx").write_bytes(data)
            with patch("tools.download_rtmpose.ASSETS", (("model.zip", "model.onnx", "unused", digest),)):
                with patch("tools.download_rtmpose.urllib.request.urlopen") as network:
                    record = download(directory)
                    network.assert_not_called()
            self.assertEqual(record["models"][0]["sha256"], digest)

    def test_downloader_rejects_unpinned_archive_before_writing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            response = Mock()
            response.read.side_effect = [b"unexpected archive", b""]
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            with patch("tools.download_rtmpose.urllib.request.urlopen", return_value=response):
                with self.assertRaisesRegex(ValueError, "Archive SHA256 mismatch"):
                    download(directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()