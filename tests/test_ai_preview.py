from types import SimpleNamespace
import unittest

import numpy as np

from ai.fall_engine import PersonObservation
from utils.ai_preview import draw_detections


def observation(box, confidence):
    points = ((box[0] + 5, box[1] + 5, .9),) * 5 + (
        (box[0] + 5, box[1] + 10, .9),
        (box[0] + 20, box[1] + 10, .9)) + ((0, 0, 0),) * 10
    return PersonObservation(box, confidence, keypoints=points)


class SkeletonPreviewTests(unittest.TestCase):
    def test_unmatched_weak_skeleton_not_drawn_beside_real_person(self):
        person = observation((10, 10, 60, 80), .8)
        cushion = observation((110, 10, 180, 80), .3)
        result = SimpleNamespace(source_shape=(100, 200, 3), detections=(person, cushion))
        image = np.zeros(result.source_shape, dtype=np.uint8)
        tracks = (SimpleNamespace(xyxy=person.xyxy, visible=True),)
        draw_detections(image, result, boxes=False, tracks=tracks)
        self.assertTrue(np.any(image[:, :100]))
        self.assertFalse(np.any(image[:, 100:]))

    def test_accepted_weak_partial_person_keeps_skeleton(self):
        person = observation((10, 10, 60, 80), .25)
        result = SimpleNamespace(source_shape=(100, 200, 3), detections=(person,))
        image = np.zeros(result.source_shape, dtype=np.uint8)
        draw_detections(image, result, boxes=False,
                        tracks=(SimpleNamespace(xyxy=person.xyxy, visible=True),))
        self.assertTrue(np.any(image))

    def test_no_skeleton_for_missing_or_rejected_tracks(self):
        person = observation((10, 10, 60, 80), .8)
        result = SimpleNamespace(source_shape=(100, 200, 3), detections=(person,))
        for tracks in ((), (SimpleNamespace(xyxy=person.xyxy, visible=False),)):
            image = np.zeros(result.source_shape, dtype=np.uint8)
            draw_detections(image, result, boxes=False, tracks=tracks)
            self.assertFalse(np.any(image))


if __name__ == "__main__":
    unittest.main()
