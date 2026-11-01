"""Opt-in real checkpoint smoke test; no network or accuracy claims."""

import os
import unittest


@unittest.skipUnless(os.getenv("RUN_MODEL_TESTS") == "1", "Set RUN_MODEL_TESTS=1 for local model smoke test")
class ModelIntegrationTests(unittest.TestCase):
    def test_verified_checkpoint_can_predict(self):
        import numpy as np
        from ai.posture_detector import PostureDetector

        detector = PostureDetector(image_size=320)
        self.assertEqual(set(detector.names.values()), {"fallen", "sitting", "standing"})
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        detections = detector.predict(image)
        self.assertIsInstance(detections, tuple)
        # Blank image detections are not ground-truth accuracy evidence.
        for detection in detections:
            self.assertIn(detection.label, detector.names.values())
            self.assertGreaterEqual(detection.confidence, 0)
            self.assertLessEqual(detection.confidence, 1)

    def test_official_pose_and_fusion_can_predict(self):
        import numpy as np
        from ai.fusion_detector import FusionDetector
        from ai.fall_engine import PersonObservation

        detector = FusionDetector(image_size=320)
        people = detector.predict(np.zeros((360, 640, 3), dtype=np.uint8))
        self.assertIsInstance(people, tuple)
        self.assertTrue(all(isinstance(person, PersonObservation) for person in people))
        self.assertGreater(detector.timings.pose_ms, 0)
        self.assertGreater(detector.timings.posture_ms, 0)


if __name__ == "__main__":
    unittest.main()
