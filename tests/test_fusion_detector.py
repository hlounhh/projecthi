from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ai.fusion_detector import FusionDetector, box_iou, fuse_people
from ai.pose_detector import PoseDetector, PosePerson


def pose(box):
    return SimpleNamespace(xyxy=box, confidence=0.9, keypoints=((10., 20., 0.9),) * 17)


def posture(box, label="fallen", confidence=0.8):
    return SimpleNamespace(xyxy=box, confidence=confidence, label=label)


class FusionTests(unittest.TestCase):
    def test_fallen_label_is_attached_to_matching_pose(self):
        box = (0, 0, 100, 200)
        people = fuse_people([pose(box)], [posture(box)])
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0].fallen_confidence, 0.8)
        self.assertEqual(len(people[0].keypoints), 17)

    def test_fallen_without_pose_remains_weak_observation(self):
        people = fuse_people([], [posture((0, 0, 100, 200))])
        self.assertEqual(people[0].keypoints, ())
        self.assertEqual(people[0].fallen_confidence, 0.8)

    def test_separate_people_do_not_share_fallen_signal(self):
        people = fuse_people([pose((0, 0, 100, 200)), pose((200, 0, 300, 200))],
                             [posture((0, 0, 100, 200))])
        self.assertEqual([p.fallen_confidence for p in people], [0.8, 0.])

    def test_auxiliary_cadence_uses_only_current_frame_evidence(self):
        detector = FusionDetector.__new__(FusionDetector)
        detector.posture_every, detector._sample = 3, 0
        box = (0, 0, 100, 200)
        detector.pose = Mock()
        detector.pose.predict.return_value = (pose(box),)
        detector.posture = Mock()
        detector.posture.predict.return_value = (posture(box),)
        outputs = [detector.predict(object())[0] for _ in range(4)]
        self.assertEqual(detector.posture.predict.call_count, 2)
        self.assertEqual([o.fallen_confidence for o in outputs], [0.8, 0, 0, 0.8])
        self.assertEqual([o.posture_observed for o in outputs], [True, False, False, True])
        detector.pose.predict.return_value = ()
        self.assertTrue(detector.predict(object())[0].posture_observed)
        self.assertEqual(detector.posture.predict.call_count, 3)

    def make_detector(self, **kwargs):
        with patch("ai.pose_detector.PoseDetector"), patch("ai.posture_detector.PostureDetector"):
            detector = FusionDetector(**kwargs)
        detector.pose.predict.return_value = ()
        detector.pose.predict_crop.return_value = ()
        detector.posture.predict.return_value = ()
        return detector

    def test_weak_pose_is_preserved_and_does_not_seed_recovery(self):
        detector = self.make_detector()
        weak = PosePerson((100, 100, 120, 150), 0.21, ((0., 0., 0.),) * 17)
        detector.pose.predict.return_value = (weak,)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            result = detector.predict(image)
            self.assertEqual(result[0].confidence, 0.21)
            self.assertEqual(result[0].xyxy, weak.xyxy)
        detector.pose.predict_crop.assert_not_called()
        self.assertEqual(detector._recent, [])

    def test_pose_threshold_is_lowered_but_posture_threshold_is_not(self):
        with patch("ai.pose_detector.PoseDetector") as pose_model, patch("ai.posture_detector.PostureDetector") as posture_model:
            FusionDetector(confidence=0.4)
            self.assertEqual(pose_model.call_args.kwargs["confidence"], 0.2)
            self.assertEqual(posture_model.call_args.kwargs["confidence"], 0.4)
            FusionDetector(confidence=0.1)
            self.assertEqual(pose_model.call_args.kwargs["confidence"], 0.1)

    def test_crop_mapping_clips_bounds_and_preserves_invisible_points(self):
        detector = PoseDetector.__new__(PoseDetector)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        local = PosePerson((1., 2., 15., 20.), 0.24,
                           ((3., 4., 0.8),) * 16 + ((0., 0., 0.),))
        detector.predict = Mock(return_value=(local,))
        result = detector.predict_crop(image, (19.8, 29.2, 250, 150))
        self.assertEqual(detector.predict.call_args.args[0].shape, (71, 181, 3))
        self.assertEqual(result[0].xyxy, (20., 31., 34., 49.))
        self.assertEqual(result[0].keypoints[0], (22., 33., 0.8))
        self.assertEqual(result[0].keypoints[-1], (0., 0., 0.))
        self.assertEqual(result[0].confidence, 0.24)
        self.assertEqual(detector.predict_crop(image, (210, 0, 220, 20)), ())
        self.assertEqual(detector.predict.call_count, 1)

    def test_missing_person_recovery_maps_actual_detection_not_anchor(self):
        detector = self.make_detector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.pose.predict.return_value = (pose((100, 100, 120, 150)),)
        detector.predict(image)
        detector.pose.predict.return_value = ()
        # Execute the real crop-coordinate mapping against a mocked model inference.
        crop_model = PoseDetector.__new__(PoseDetector)
        crop_model.predict = Mock(return_value=(PosePerson((24., 23., 42., 68.), 0.25,
                                                           ((28., 30., 0.8),) * 17),))
        detector.pose.predict_crop.side_effect = crop_model.predict_crop
        result = detector.predict(image)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].xyxy, (102., 103., 120., 148.))
        self.assertEqual(result[0].keypoints[0], (106., 110., 0.8))
        self.assertEqual(result[0].confidence, 0.25)
        self.assertEqual(result[0].fallen_confidence, 0.)
        detector.pose.predict_crop.assert_called_once()

    def test_crop_refresh_replaces_weak_torso_without_duplicate_or_stale_label(self):
        detector = self.make_detector(posture_every=5)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        box = (100, 100, 120, 150)
        detector.pose.predict.return_value = (pose(box),)
        detector.posture.predict.return_value = (posture(box),)
        self.assertEqual(detector.predict(image)[0].fallen_confidence, 0.8)
        detector.pose.predict.return_value = (PosePerson(box, 0.22, ((0., 0., 0.),) * 17),)
        fresh = PosePerson((101, 100, 121, 150), 0.25, ((110., 125., 0.8),) * 17)
        detector.pose.predict_crop.return_value = (fresh, fresh)
        result = detector.predict(image)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].xyxy, fresh.xyxy)
        self.assertEqual(result[0].confidence, 0.25)
        self.assertFalse(result[0].posture_observed)
        self.assertEqual(result[0].fallen_confidence, 0.)
        detector.predict(image)
        detector.predict(image)
        self.assertEqual(detector.pose.predict_crop.call_count, 1)
        detector.predict(image)
        self.assertEqual(detector.pose.predict_crop.call_count, 2)

    def test_weak_nearby_crop_requires_overlap_with_recent_person(self):
        detector = self.make_detector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        box = (100, 100, 120, 150)
        detector.pose.predict.return_value = (pose(box),)
        detector.predict(image)
        detector.pose.predict.return_value = ()
        detector.pose.predict_crop.return_value = (
            PosePerson((114, 100, 134, 150), .3, ((120., 120., .9),) * 17),)
        self.assertEqual(detector.predict(image), ())
        self.assertEqual(detector._recent[0].box, box)
        detector.pose.predict_crop.assert_called_once()

    def test_weak_full_frame_object_cannot_move_crop_anchor(self):
        detector = self.make_detector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        box = (100, 100, 120, 150)
        detector.pose.predict.return_value = (pose(box),)
        detector.predict(image)
        weak = PosePerson((114, 100, 134, 150), .3, ((120., 120., .9),) * 17)
        detector.pose.predict.return_value = (weak,)
        result = detector.predict(image)
        self.assertEqual(result[0].xyxy, weak.xyxy)
        self.assertEqual(detector._recent[0].box, box)
        detector.pose.predict_crop.assert_called_once()

    def test_weak_good_torso_retries_crop_without_extending_anchor_lifetime(self):
        detector = self.make_detector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        box = (100, 100, 120, 150)
        detector.pose.predict.return_value = (pose(box),)
        detector.predict(image)
        weak = PosePerson(box, .25, ((110., 125., .9),) * 17)
        detector.pose.predict.return_value = (weak,)
        detector.pose.predict_crop.return_value = (PosePerson(box, .22, weak.keypoints),)
        for sample in range(1, 11):
            output = detector.predict(image)
            self.assertEqual(output[0].confidence, .25)
            self.assertEqual(detector.pose.predict_crop.call_count, (sample + 2) // 3)
            self.assertEqual(detector._recent[0].confident_sample, 0)
        detector.predict(image)
        self.assertEqual(detector._recent, [])
        self.assertEqual(detector.pose.predict_crop.call_count, 4)

    def test_weak_overlapping_boxes_cannot_walk_anchor_toward_furniture(self):
        detector = self.make_detector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        box = (100, 100, 120, 150)
        detector.pose.predict.return_value = (pose(box),)
        detector.predict(image)
        for offset in (8, 16, 24):
            weak = PosePerson((100 + offset, 100, 120 + offset, 150), .3,
                              ((110. + offset, 125., .9),) * 17)
            detector.pose.predict.return_value = (weak,)
            detector.predict(image)
            self.assertEqual(detector._recent[0].box, box)
            self.assertEqual(detector._recent[0].confident_sample, 0)
        self.assertEqual(len(detector._recent), 1)

    def test_strong_crop_can_replace_weak_detection_with_confident_torso(self):
        detector = self.make_detector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        box = (100, 100, 120, 150)
        detector.pose.predict.return_value = (pose(box),)
        detector.predict(image)
        detector.pose.predict.return_value = (PosePerson(box, .25, pose(box).keypoints),)
        recovered = pose((101, 100, 121, 150))
        detector.pose.predict_crop.return_value = (recovered,)
        output = detector.predict(image)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].confidence, .9)
        self.assertEqual(detector._recent[0].box, recovered.xyxy)

    def test_good_full_frame_torso_needs_no_crop(self):
        detector = self.make_detector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.pose.predict.return_value = (pose((100, 100, 120, 150)),)
        for _ in range(12):
            self.assertEqual(len(detector.predict(image)), 1)
        detector.pose.predict_crop.assert_not_called()
        self.assertEqual(len(detector._recent), 1)
        self.assertEqual(detector._recent[0].confident_sample, 11)

    def test_one_crop_budget_and_no_cached_detections(self):
        detector = self.make_detector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.pose.predict.return_value = (pose((100, 100, 120, 150)), pose((300, 100, 320, 150)))
        detector.predict(image)
        detector.pose.predict.return_value = ()
        for _ in range(4):
            before = detector.pose.predict_crop.call_count
            self.assertEqual(detector.predict(image), ())
            self.assertLessEqual(detector.pose.predict_crop.call_count - before, 1)
        self.assertGreaterEqual(detector.pose.predict_crop.call_count, 2)
        bounds = [call.args[1] for call in detector.pose.predict_crop.call_args_list]
        self.assertNotEqual(bounds[0], bounds[1])

    def test_crop_rejects_unrelated_people_and_duplicates_of_other_person(self):
        detector = self.make_detector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        first, second = pose((100, 100, 120, 150)), pose((125, 100, 145, 150))
        detector.pose.predict.return_value = (first, second)
        detector.predict(image)
        detector.pose.predict.return_value = (second,)
        detector.pose.predict_crop.return_value = (pose((400, 300, 420, 350)),
                                                   pose((112, 100, 132, 150)))
        result = detector.predict(image)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].xyxy, second.xyxy)

    def test_refresh_cannot_append_a_second_detection_for_matched_anchor(self):
        detector = self.make_detector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.pose.predict.return_value = (pose((100, 100, 120, 150)),)
        detector.predict(image)
        weak = PosePerson((90, 100, 110, 150), 0.22, ((0., 0., 0.),) * 17)
        detector.pose.predict.return_value = (weak,)
        # Both observations are near the anchor, but not near each other.
        detector.pose.predict_crop.return_value = (pose((119, 100, 139, 150)),)
        result = detector.predict(image)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].xyxy, weak.xyxy)

    def test_expiry_is_sample_based_and_weak_matches_do_not_extend_it(self):
        detector = self.make_detector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        box = (100, 100, 120, 150)
        detector.pose.predict.return_value = (pose(box),)
        detector.predict(image)
        detector.pose.predict.return_value = (PosePerson(box, 0.2, ((0., 0., 0.),) * 17),)
        for _ in range(10):
            detector.predict(image)
        calls = detector.pose.predict_crop.call_count
        self.assertEqual(calls, 4)  # Samples 1, 4, 7, 10.
        for _ in range(3):
            detector.predict(image)
        self.assertEqual(detector.pose.predict_crop.call_count, calls)
        self.assertEqual(detector._recent, [])

    def test_reset_clears_history_and_cadence_without_reloading(self):
        detector = self.make_detector(posture_every=3)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.pose.predict.return_value = (pose((100, 100, 120, 150)),)
        detector.predict(image)
        model = detector.pose
        detector.reset()
        self.assertEqual(detector._sample, 0)
        self.assertEqual(detector._recent, [])
        self.assertIs(detector.pose, model)
        detector.pose.predict.return_value = ()
        self.assertEqual(detector.predict(image), ())
        detector.pose.predict_crop.assert_not_called()

    def test_recovery_can_be_disabled_and_never_zooms_whole_frame(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        for options, box in (({"recovery": False}, (100, 100, 120, 150)),
                             ({}, (0, 0, 640, 480))):
            detector = self.make_detector(**options)
            detector.pose.predict.return_value = (pose(box),)
            detector.predict(image)
            detector.pose.predict.return_value = ()
            self.assertEqual(detector.predict(image), ())
            detector.pose.predict_crop.assert_not_called()

    def test_recovery_sample_limits_are_explicit(self):
        for options in ({"recovery_max_samples": 11}, {"recovery_max_samples": 0},
                        {"recovery_every": 0}, {"recovery_max_samples": True}):
            with self.assertRaises(ValueError):
                self.make_detector(**options)

    def test_box_overlap_edge_cases(self):
        self.assertEqual(box_iou((0, 0, 100, 100), (100, 0, 200, 100)), 0)
        self.assertEqual(box_iou((0, 0, 100, 100), (0, 0, 100, 100)), 1)


if __name__ == "__main__":
    unittest.main()
