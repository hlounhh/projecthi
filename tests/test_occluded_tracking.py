from dataclasses import replace
import unittest

from ai.fall_engine import FallConfig, FallEngine, CONFIRMED_FALL, UNOBSERVED
from tests.test_fall_engine import person, SHAPE


class OccludedTrackingTests(unittest.TestCase):
    def test_weak_observations_extend_existing_id_but_do_not_create_one(self):
        engine = FallEngine()
        weak = person(confidence=.25)
        self.assertEqual(engine.update((weak,), 0, SHAPE), ())
        self.assertEqual(engine.snapshot(0), ())
        engine.update((person(),), .1, SHAPE)
        identity = engine.snapshot(.1)[0].track_id
        for i in range(2, 31):
            engine.update((weak,), i / 10, SHAPE)
        self.assertEqual(engine.snapshot(3)[0].track_id, identity)
        self.assertTrue(engine.snapshot(3)[0].visible)

    def test_nearby_disjoint_weak_object_cannot_steal_person_id(self):
        engine = FallEngine()
        real = person()
        engine.update((real,), 0, SHAPE)
        cushion = replace(person(x=330, confidence=.3), xyxy=(300, 160, 420, 250))
        engine.update((cushion,), .1, SHAPE)
        view, = engine.snapshot(.1)
        self.assertEqual(view.track_id, 1)
        self.assertEqual(view.xyxy, real.xyxy)
        self.assertFalse(view.visible)
        engine.update((real, cushion), .2, SHAPE)
        view, = engine.snapshot(.2)
        self.assertEqual(view.track_id, 1)
        self.assertTrue(view.visible)

    def test_partial_upper_body_keeps_id_across_large_box_shrink(self):
        engine = FallEngine()
        full = person()
        engine.update((full,), 0, SHAPE)
        points = tuple(p if i < 11 else (0, 0, .1) for i, p in enumerate(full.keypoints))
        partial = replace(full, xyxy=(230, 120, 270, 170), keypoints=points)
        for i in range(1, 21):
            engine.update((partial,), i / 10, SHAPE)
        view, = engine.snapshot(2)
        self.assertEqual(view.track_id, 1)
        self.assertTrue(view.visible)
        self.assertFalse(view.observation_quality)
        self.assertEqual(view.state, UNOBSERVED)
        self.assertEqual(view.xyxy, partial.xyxy)

    def test_dormant_id_recovers_without_transmitting_motion(self):
        engine = FallEngine()
        for i in range(7):
            engine.update((person(),), i / 10, SHAPE)
        old = engine.snapshot(.6)[0].track_id
        engine.update((), 1.8, SHAPE)
        self.assertEqual(engine.snapshot(1.8), ())
        engine.update((person(),), 2, SHAPE)
        self.assertEqual(engine.snapshot(2)[0].track_id, old)
        self.assertIsNone(engine._tracks[old].baseline)
        events = []
        for i in range(21, 29):
            events.extend(engine.update((person(y=240, angle=80),), i / 10, SHAPE))
        self.assertNotIn(CONFIRMED_FALL, [e.state for e in events])

    def test_dormant_id_expires_and_clock_reset_clears_it(self):
        engine = FallEngine()
        engine.update((person(),), 0, SHAPE)
        engine.update((), 1.5, SHAPE)
        engine.update((person(),), 3.1, SHAPE)
        self.assertEqual(engine.snapshot(3.1)[0].track_id, 2)
        engine.update((), 4.5, SHAPE)
        engine.update((person(),), 0, SHAPE)
        self.assertEqual(engine.snapshot(0)[0].track_id, 3)

    def test_weak_pose_can_preserve_identity_but_never_confirm(self):
        engine = FallEngine()
        for i in range(7):
            engine.update((person(),), i / 10, SHAPE)
        events = []
        for i in range(7, 25):
            progress = min(1, (i - 6) / 8)
            weak = person(y=160 + 80 * progress, angle=80 * progress, confidence=.25)
            events.extend(engine.update((weak,), i / 10, SHAPE))
        self.assertEqual(engine.snapshot(2.4)[0].track_id, 1)
        self.assertTrue(engine.snapshot(2.4)[0].visible)
        self.assertFalse(engine.snapshot(2.4)[0].observation_quality)
        self.assertNotIn(CONFIRMED_FALL, [e.state for e in events])

    def test_missing_person_not_reported_visible_and_does_not_verify(self):
        engine = FallEngine()
        engine.update((person(),), 0, SHAPE)
        engine.update((), .1, SHAPE)
        view = engine.snapshot(.1)[0]
        self.assertFalse(view.visible)
        self.assertEqual(view.state, UNOBSERVED)

    def test_one_visible_torso_side_can_warn_but_cannot_confirm(self):
        engine = FallEngine()
        observation = person(y=240, angle=80)
        points = list(observation.keypoints)
        points[5] = points[11] = (0, 0, .1)
        observation = replace(observation, keypoints=tuple(points))
        events = []
        for i in range(25):
            events.extend(engine.update((observation,), i / 10, SHAPE))
        view = engine.snapshot(2.4)[0]
        self.assertTrue(view.visible)
        self.assertFalse(view.observation_quality)
        self.assertTrue(any(e.reason == "persistent_low_pose" for e in events))
        self.assertNotIn(CONFIRMED_FALL, [e.state for e in events])

    def test_two_dormant_people_are_not_merged_by_one_observation(self):
        engine = FallEngine()
        first = replace(person(), xyxy=(200, 100, 300, 300))
        second = replace(person(), xyxy=(230, 100, 330, 300))
        engine.update((first, second), 0, SHAPE)
        engine.update((), 1.2, SHAPE)
        shared = replace(person(), xyxy=(215, 100, 315, 300))
        engine.update((shared,), 1.3, SHAPE)
        self.assertEqual(engine.snapshot(1.3)[0].track_id, 3)
        self.assertEqual(len(engine._dormant), 2)

    def test_visible_legs_not_required_to_preserve_risk_box(self):
        engine = FallEngine()
        for i in range(7):
            engine.update((person(),), i / 10, SHAPE)
        engine.update((person(y=240, angle=80, fallen=.95),), .8, SHAPE)
        engine.update((person(y=240, angle=80, partial=True),), .9, SHAPE)
        view = engine.snapshot(.9)[0]
        self.assertTrue(view.visible)
        self.assertIsNotNone(view.event_id)
        self.assertFalse(view.observation_quality)

    def test_binary_decision_preserves_confirmation_through_occlusion_until_recovery(self):
        engine = FallEngine()
        for t, obs in ((0, person()), (.2, person()), (.4, person()), (.6, person()),
                       (.8, person(y=190, angle=35)), (1, person(y=240, angle=80, fallen=.95))):
            engine.update((obs,), t, SHAPE)
        self.assertFalse(engine.snapshot(1)[0].fall_detected)
        for t in (1.2, 1.4, 1.6, 1.8):
            engine.update((person(y=240, angle=80, fallen=.95),), t, SHAPE)
        self.assertTrue(engine.snapshot(1.8)[0].fall_detected)
        engine.update((), 1.9, SHAPE)
        self.assertTrue(engine.snapshot(1.9)[0].fall_detected)
        self.assertFalse(engine.snapshot(1.9)[0].visible)
        for i in range(20, 81):
            engine.update((person(),), i / 10, SHAPE)
        self.assertFalse(engine.snapshot(8)[0].fall_detected)

    def test_invalid_tracking_threshold_rejected(self):
        with self.assertRaises(ValueError):
            FallConfig(min_tracking_confidence=.5)


if __name__ == "__main__":
    unittest.main()
