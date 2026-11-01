from dataclasses import replace
import unittest

from ai.fall_engine import FallConfig, FallEngine, CONFIRMED_FALL
from tests.test_fall_engine import person, SHAPE, seated


class OcclusionEvidenceTests(unittest.TestCase):
    def seed(self):
        engine = FallEngine()
        for i in range(7):
            engine.update((person(),), i / 10, SHAPE)
        return engine

    def descent(self, engine):
        engine.update((person(y=200, angle=35),), .7, SHAPE)
        engine.update((person(y=240, angle=80),), .8, SHAPE)

    def test_no_observations_never_confirm(self):
        engine = self.seed()
        self.descent(engine)
        events = []
        for i in range(9, 40):
            events.extend(engine.update((), i / 10, SHAPE))
        self.assertFalse(any(e.state == CONFIRMED_FALL for e in events))

    def test_interrupted_down_poses_require_a_new_continuous_verification(self):
        engine = self.seed()
        self.descent(engine)
        events = []
        for i in range(9, 17):
            obs = (person(y=240, angle=80),) if i % 3 == 2 else ()
            events.extend(engine.update(obs, i / 10, SHAPE))
        self.assertFalse(any(e.state == CONFIRMED_FALL for e in events))
        for i in range(17, 30):
            events.extend(engine.update((person(y=240, angle=80),), i / 10, SHAPE))
        confirmed = [e for e in events if e.state == CONFIRMED_FALL]
        self.assertEqual(len(confirmed), 1)
        self.assertGreaterEqual(confirmed[0].timestamp, 2.4)
        self.assertAlmostEqual(dict(confirmed[0].features)["landing_observed_seconds"], .6)

    def test_static_lying_or_sitting_cannot_seed_occluded_fall(self):
        for observation in (person(y=240, angle=80), seated()):
            engine = FallEngine()
            events = []
            for i in range(40):
                events.extend(engine.update((observation,) if i % 3 == 0 else (), i / 10, SHAPE))
            self.assertFalse(any(e.state == CONFIRMED_FALL for e in events))

    def test_long_gap_weak_detection_stale_and_ambiguity_invalidate_bilateral_candidate(self):
        for kind in ("long_gap", "weak", "stale", "ambiguous"):
            engine = self.seed()
            self.descent(engine)
            if kind == "long_gap":
                engine.update((), 1.9, SHAPE)
                start = 20
            elif kind == "weak":
                engine.update((person(y=240, angle=80, confidence=.25),), .9, SHAPE)
                start = 10
            elif kind == "stale":
                engine.mark_unobserved(.9)
                start = 10
            else:
                engine._missing(engine._tracks[1], .9, "ambiguous_association", [])
                start = 10
            events = []
            for i in range(start, start + 20):
                events.extend(engine.update((person(y=240, angle=80),), i / 10, SHAPE))
            confirmed = [e for e in events if e.state == CONFIRMED_FALL]
            if kind == "weak":
                self.assertIsNone(engine._tracks[1].occlusion.candidate)
                self.assertEqual(len(confirmed), 1)
                self.assertEqual(confirmed[0].reason, "partial_landing_confirmed")
                self.assertAlmostEqual(dict(confirmed[0].features)["partial_landing_observed_seconds"], .6)
            else:
                self.assertFalse(confirmed, kind)

    def test_upright_recovery_invalidates_occlusion_candidate(self):
        engine = self.seed()
        self.descent(engine)
        engine.update((person(),), .9, SHAPE)
        self.assertIsNone(engine._tracks[1].occlusion.candidate)
        events = []
        for i in range(10, 29):
            events.extend(engine.update((), i / 10, SHAPE))
        self.assertFalse(any(e.reason == "occlusion_fall_confirmed" for e in events))

    def test_one_visible_shoulder_with_observed_motion_and_extended_legs_can_verify(self):
        engine = self.seed()
        self.descent(engine)
        observation = person(y=240, angle=80)
        points = list(observation.keypoints)
        points[5] = (*points[5][:2], .1)
        observation = replace(observation, keypoints=tuple(points))
        events = []
        for i in range(9, 30):
            events.extend(engine.update((observation,), i / 10, SHAPE))
        confirmed = [e for e in events if e.state == CONFIRMED_FALL]
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].reason, "partial_landing_confirmed")
        self.assertEqual(engine._tracks[1].occlusion.observed_down, 0.)

    def test_descent_time_does_not_extend_upright_baseline(self):
        engine = FallEngine()
        for timestamp in (0., .1, .2):
            engine.update((person(),), timestamp, SHAPE)
        engine.update((person(y=200, angle=35),), .3, SHAPE)
        engine.update((person(y=240, angle=80),), .4, SHAPE)
        self.assertIsNone(engine._tracks[1].occlusion.candidate)

    def test_dormant_identity_keeps_no_occlusion_evidence(self):
        for watchdog in (False, True):
            with self.subTest(watchdog=watchdog):
                engine = FallEngine(FallConfig(max_track_gap=.4))
                for i in range(7):
                    engine.update((person(),), i / 10, SHAPE)
                self.descent(engine)
                self.assertIsNotNone(engine._tracks[1].occlusion.candidate)
                if watchdog:
                    engine.mark_unobserved(1.3, reason="not_matched")
                else:
                    engine.update((), 1.3, SHAPE)
                self.assertIn(1, engine._dormant)
                self.assertIsNone(engine._dormant[1].occlusion.candidate)
                events = []
                for i in range(14, 31):
                    events.extend(engine.update((person(y=240, angle=80),), i / 10, SHAPE))
                self.assertIn(1, engine._tracks)
                self.assertFalse(any(e.reason == "occlusion_fall_confirmed" for e in events))

    def test_watchdog_always_invalidates_even_with_preserved_reason(self):
        for reason in ("not_matched", "pose_unreliable", "lower_body_unreliable", "observation_gap"):
            with self.subTest(reason=reason):
                engine = self.seed()
                self.descent(engine)
                engine.mark_unobserved(.9, reason=reason)
                self.assertIsNone(engine._tracks[1].occlusion.candidate)
                events = []
                for i in range(10, 30):
                    events.extend(engine.update((person(y=240, angle=80),), i / 10, SHAPE))
                self.assertFalse(any(e.state == CONFIRMED_FALL for e in events))

    def test_no_visible_legs_cannot_verify(self):
        engine = self.seed()
        self.descent(engine)
        events = []
        for i in range(9, 30):
            events.extend(engine.update((person(y=240, angle=80, partial=True),), i / 10, SHAPE))
        self.assertFalse(any(e.state == CONFIRMED_FALL for e in events))


if __name__ == "__main__":
    unittest.main()
