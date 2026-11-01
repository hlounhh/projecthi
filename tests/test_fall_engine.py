"""Deterministic heuristic tests, NOT a measurement of real-world fall accuracy.

Synthetic COCO17 poses exercise temporal safety gates without camera/model/NumPy.
"""

from dataclasses import FrozenInstanceError, fields, replace
import math
import unittest

from ai.fall_engine import (
    CONFIRMED_FALL, NORMAL, SUSPECTED_FALL, UNOBSERVED,
    FallConfig, FallEngine, FallEvent, PersonObservation,
)


SHAPE = (720, 1280, 3)


def person(x=250, y=160, angle=0, fallen=0, *, box=None, partial=False, confidence=0.95):
    """Rotate a 200px extended body around its shoulder/hip midpoint."""
    theta = math.radians(angle)
    dx, dy = math.sin(theta), math.cos(theta)
    # Shoulder->hip length is 65; shoulder->ankle length is 170.
    shoulder = (x - 32.5 * dx, y - 32.5 * dy)
    points = []
    for index in range(17):
        offset = -20 if index < 5 else 0 if index < 7 else 30 if index < 11 else 65 if index < 13 else 120 if index < 15 else 170
        side = -10 if index % 2 else 10
        points.append((shoulder[0] + offset * dx + side * dy,
                       shoulder[1] + offset * dy - side * dx,
                       0.1 if partial and index >= 13 else 0.95))
    if box is None:
        # Upright: width60 height200. Horizontal: width200 height60.
        w, h = 60 + 140 * dx, 60 + 140 * dy
        center = (shoulder[0] + 75 * dx, shoulder[1] + 75 * dy)
        box = (center[0] - w / 2, center[1] - h / 2,
               center[0] + w / 2, center[1] + h / 2)
    return PersonObservation(box, confidence, fallen, tuple(points))


def seated(x=250, y=260, fallen=0, crouched=False):
    """Upright torso, folded lower body (also usable for quick sitting)."""
    observation = person(x, y, fallen=fallen, box=(x - 55, y - 60, x + 65, y + 100))
    points = list(observation.keypoints)
    for index in (13, 14):
        points[index] = (x + 65, y + 40, 0.95)
    for index in (15, 16):
        points[index] = (x + 20, y + (35 if crouched else 85), 0.95)
    return replace(observation, keypoints=tuple(points))


def states(events):
    return [event.state for event in events]


class FallEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = FallEngine()

    def feed(self, timestamp, observation=None, engine=None):
        return (engine or self.engine).update(
            () if observation is None else (observation,), timestamp, SHAPE)

    def baseline(self, engine=None, x=250, start=0):
        events = []
        for offset in (0, 0.2, 0.4, 0.6):
            events.extend(self.feed(start + offset, person(x), engine))
        return events

    def fall(self, engine=None, x=250, start=0):
        engine = engine or self.engine
        events = self.baseline(engine, x, start)
        events.extend(self.feed(start + 0.8, person(x, 190, 35), engine))
        events.extend(self.feed(start + 1.0, person(x, 240, 80, 0.95), engine))
        for offset in (1.2, 1.4, 1.6, 1.8):
            events.extend(self.feed(start + offset, person(x, 240, 80, 0.95), engine))
        return events

    def assert_not_confirmed(self, events):
        self.assertNotIn(CONFIRMED_FALL, states(events))

    def test_sudden_side_fall_confirms_after_continuous_verification(self):
        events = self.fall()
        suspected = next(e for e in events if e.state == SUSPECTED_FALL)
        confirmed = [e for e in events if e.state == CONFIRMED_FALL]
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(suspected.event_id, confirmed[0].event_id)
        self.assertGreaterEqual(confirmed[0].timestamp - suspected.timestamp, 0.6)
        features = dict(confirmed[0].features)
        self.assertTrue(features["pose_reliable"])
        self.assertGreaterEqual(features["verified_seconds"], 0.6)
        self.assertEqual(confirmed[0].reason, "algorithmic_fall_confirmed")

    def test_forward_fall_can_use_strong_alternate_pose_geometry(self):
        self.baseline()
        # Oblique projected torso: less than the primary 45-degree turn, but a
        # compressed, wide silhouette AND extended, horizontally displaced limbs.
        initial = person(y=240, angle=40, fallen=0.9, box=(190, 210, 400, 325))
        points = list(initial.keypoints)
        points[15] = (430, 290, 0.95)
        points[16] = (435, 290, 0.95)
        fallen = replace(initial, keypoints=tuple(points))
        events = []
        for timestamp in (0.8, 1.0, 1.2, 1.4, 1.6):
            events.extend(self.feed(timestamp, fallen))
        self.assertIn(CONFIRMED_FALL, states(events))

    def test_geometry_does_not_require_fallen_classifier(self):
        self.baseline()
        events = []
        for timestamp in (0.8, 1.0, 1.2, 1.4, 1.6):
            events.extend(self.feed(timestamp, person(y=240, angle=80)))
        self.assertIn(SUSPECTED_FALL, states(events))
        self.assertIn(CONFIRMED_FALL, states(events))

    def curled_landing(self):
        observation = person(y=240, angle=80)
        points = list(observation.keypoints)
        # Fold the legs back toward the shoulders; torso motion remains reliable.
        for index in (13, 14, 15, 16):
            points[index] = (250, 265, 0.95)
        return replace(observation, xyxy=(200, 180, 390, 330), keypoints=tuple(points))

    def test_curled_rapid_landing_suspects_without_extended_geometry(self):
        self.baseline()
        events = self.feed(0.8, self.curled_landing())
        self.assertEqual(states(events), [SUSPECTED_FALL])
        features = dict(events[0].features)
        self.assertLess(features["body_extension"], self.engine.config.post_min_extension)
        self.assertFalse(features["post_fall_geometry"])
        self.assertGreaterEqual(features["drop"], self.engine.config.min_drop)
        for timestamp in (1.0, 1.2, 1.4, 1.6):
            events += self.feed(timestamp, self.curled_landing())
        self.assert_not_confirmed(events)

    def test_continuous_curled_then_extended_landing_can_verify_later(self):
        self.baseline()
        suspected = self.feed(0.8, self.curled_landing())[0]
        self.feed(1.0, self.curled_landing())
        events = []
        for timestamp in (1.2, 1.4, 1.6, 1.8, 2.0):
            events.extend(self.feed(timestamp, person(y=240, angle=80)))
        confirmed = next(e for e in events if e.state == CONFIRMED_FALL)
        self.assertEqual(confirmed.event_id, suspected.event_id)
        self.assertGreaterEqual(confirmed.timestamp - 1.2, self.engine.config.verify_seconds)

    def test_curled_landing_then_occlusion_cannot_borrow_motion_for_confirmation(self):
        self.baseline()
        suspected = self.feed(0.8, self.curled_landing())[0]
        gap = self.feed(1.0, person(y=240, angle=80, partial=True))
        self.assertEqual(states(gap), [UNOBSERVED])
        self.assertEqual(gap[0].event_id, suspected.event_id)
        events = []
        for timestamp in (1.2, 1.4, 1.6, 1.8, 2.0):
            events.extend(self.feed(timestamp, person(y=240, angle=80)))
        self.assert_not_confirmed(events)
        self.assertEqual(self.engine.snapshot(2.0)[0].event_id, suspected.event_id)

    def test_preparatory_bend_does_not_dilute_recent_descent(self):
        self.baseline()
        events = []
        for timestamp, y, angle in ((0.7, 163, 10), (0.8, 166, 35),
                                    (0.9, 180, 60), (1.0, 200, 80),
                                    (1.1, 225, 80)):
            events.extend(self.feed(timestamp, person(y=y, angle=angle)))
        suspected = next(e for e in events if e.state == SUSPECTED_FALL)
        self.assertGreaterEqual(dict(suspected.features)["descent_speed"], 0.65)
        self.assert_not_confirmed(events)

    def raised_leg_landing(self):
        observation = person(y=240, angle=20, box=(195, 180, 295, 290))
        points = list(observation.keypoints)
        for index in (13, 14, 15, 16):
            points[index] = (250, 210, 0.95)
        return replace(observation, keypoints=tuple(points))

    def test_long_upright_history_keeps_recent_motion_baseline(self):
        events = []
        for step in range(21):
            events.extend(self.feed(step * 0.2, person()))
        self.assertEqual(events, [])
        for timestamp in (4.2, 4.4, 4.6, 4.8, 5.0):
            events.extend(self.feed(timestamp, person(y=240, angle=80)))
        self.assertEqual(states(events).count(SUSPECTED_FALL), 1)
        self.assertEqual(states(events).count(CONFIRMED_FALL), 1)
        self.assertEqual(events[0].reason, "rapid_descent_and_geometry")
        self.assertEqual(events[0].timestamp, 4.2)

    def test_peak_descent_speed_survives_until_later_rotation(self):
        self.baseline()
        self.assertEqual(self.feed(0.8, person(y=185, angle=30)), ())
        self.assertEqual(self.feed(0.9, person(y=210, angle=30)), ())
        events = list(self.feed(1.0, person(y=218, angle=80)))
        self.assertEqual(states(events), [SUSPECTED_FALL])
        features = dict(events[0].features)
        self.assertAlmostEqual(features["descent_speed"], 1.25)
        self.assertGreaterEqual(features["drop"], self.engine.config.min_drop)
        for timestamp in (1.2, 1.4, 1.6, 1.8):
            events.extend(self.feed(timestamp, person(y=218, angle=80)))
        self.assertEqual(states(events).count(CONFIRMED_FALL), 1)

    def test_sequential_descent_then_delayed_rotation_confirms(self):
        self.baseline()
        for timestamp in (0.8, 1.0, 1.2, 1.4, 1.6):
            self.assertEqual(self.feed(timestamp, person(y=240, angle=30)), ())
        events = list(self.feed(1.8, person(y=240, angle=80)))
        self.assertEqual(states(events), [SUSPECTED_FALL])
        self.assertTrue(dict(events[0].features)["sequential_descent_landing"])
        for timestamp in (2.0, 2.2, 2.4, 2.6):
            events.extend(self.feed(timestamp, person(y=240, angle=80)))
        confirmed = next(e for e in events if e.state == CONFIRMED_FALL)
        self.assertEqual(confirmed.event_id, events[0].event_id)
        self.assertGreaterEqual(confirmed.timestamp - 1.8, self.engine.config.verify_seconds)

    def test_sequential_descent_then_compressed_raised_legs_confirms(self):
        self.baseline()
        for timestamp in (0.8, 1.0, 1.2, 1.4, 1.6):
            self.assertEqual(self.feed(timestamp, person(y=240, angle=30)), ())
        landing = self.raised_leg_landing()
        events = list(self.feed(1.8, landing))
        self.assertEqual(states(events), [SUSPECTED_FALL])
        features = dict(events[0].features)
        self.assertTrue(features["sequential_descent_landing"])
        self.assertLess(features["torso_angle"], self.engine.config.alternate_min_angle)
        self.assertLess(features["aspect"], self.engine.config.down_min_aspect)
        for timestamp in (2.0, 2.2, 2.4, 2.6):
            events.extend(self.feed(timestamp, landing))
        self.assertEqual(states(events).count(CONFIRMED_FALL), 1)

    def test_compressed_landing_requires_all_knees_and_ankles_raised(self):
        for index in (13, 14, 15, 16):
            for missing in (False, True):
                with self.subTest(index=index, missing=missing):
                    engine = FallEngine()
                    self.baseline(engine)
                    for timestamp in (0.8, 1.0, 1.2, 1.4, 1.6):
                        self.feed(timestamp, person(y=240, angle=30), engine)
                    landing = self.raised_leg_landing()
                    points = list(landing.keypoints)
                    points[index] = (250, 210, 0.1) if missing else (250, 280, 0.95)
                    landing = replace(landing, keypoints=tuple(points))
                    events = []
                    for timestamp in (1.8, 2.0, 2.2, 2.4, 2.6):
                        events.extend(self.feed(timestamp, landing, engine))
                    self.assertEqual(events, [])

    def test_sequential_descent_expires_before_late_landing(self):
        self.baseline()
        events = []
        for step in range(4, 14):
            events.extend(self.feed(step * 0.2, person(y=240, angle=30)))
        for timestamp in (2.8, 3.0, 3.2, 3.4, 3.6):
            events.extend(self.feed(timestamp, person(y=240, angle=80)))
        self.assertEqual(events, [])

    def test_sequential_descent_then_upright_recovery_cannot_qualify_lying(self):
        self.baseline()
        self.assertEqual(self.feed(0.8, person(y=240, angle=30)), ())
        events = []
        for timestamp in (1.0, 1.2, 1.4, 1.6, 1.8):
            events.extend(self.feed(timestamp, person(y=240)))
        for step in range(10, 26):
            events.extend(self.feed(step * 0.2, person(y=240, angle=80)))
        self.assertEqual(states(events), [SUSPECTED_FALL])
        self.assertEqual(events[0].reason, "persistent_low_pose")
        self.assert_not_confirmed(events)

    def test_missing_data_invalidates_untriggered_sequential_descent(self):
        for missing in ("unmatched", "frame_gap", "stale", "torso", "legs"):
            with self.subTest(missing=missing):
                engine = FallEngine()
                self.baseline(engine)
                track_id = engine.snapshot(0.6)[0].track_id
                self.assertEqual(self.feed(0.8, person(y=240, angle=30), engine), ())
                if missing == "unmatched":
                    events = list(self.feed(1.0, engine=engine))
                elif missing == "frame_gap":
                    events = list(self.feed(1.15, person(y=240, angle=30), engine))
                elif missing == "stale":
                    events = list(engine.mark_unobserved(1.0))
                else:
                    observation = person(y=240, angle=30, partial=missing == "legs")
                    if missing == "torso":
                        points = list(observation.keypoints)
                        points[5] = (0, 0, 0.1)
                        observation = replace(observation, keypoints=tuple(points))
                    events = list(self.feed(1.0, observation, engine))
                for timestamp in (1.2, 1.4, 1.6):
                    events.extend(self.feed(timestamp, person(y=240, angle=30), engine))
                for timestamp in (1.8, 2.0, 2.2, 2.4, 2.6):
                    events.extend(self.feed(timestamp, person(y=240, angle=80), engine))
                self.assertEqual(engine.snapshot(2.6)[0].track_id, track_id)
                self.assertNotIn(SUSPECTED_FALL, states(events))
                self.assert_not_confirmed(events)

    def test_slow_descent_then_raised_legs_cannot_borrow_motion(self):
        self.baseline()
        events = []
        for step in range(1, 21):
            events.extend(self.feed(0.6 + step * 0.2, person(y=160 + step * 4, angle=20)))
        for timestamp in (4.8, 5.0, 5.2, 5.4, 5.6, 5.8):
            events.extend(self.feed(timestamp, self.raised_leg_landing()))
        self.assertEqual(events, [])

    def test_fallen_label_on_seated_or_crouched_pose_does_not_suspect(self):
        for crouched in (False, True):
            with self.subTest(crouched=crouched):
                engine = FallEngine()
                events = self.baseline(engine)
                for step in range(4, 20):
                    events.extend(self.feed(step * 0.2, seated(fallen=0.95, crouched=crouched), engine))
                self.assertEqual(events, [])
                self.assertEqual(engine.snapshot(3.8)[0].state, NORMAL)

    def test_short_horizontal_pose_does_not_trigger_down_warning(self):
        events = []
        for timestamp in (0.0, 0.2, 0.4, 0.6):
            events.extend(self.feed(timestamp, person(y=240, angle=80)))
        events.extend(self.feed(0.8, person()))
        self.assertNotIn(SUSPECTED_FALL, states(events))

    def test_one_occluded_ankle_does_not_erase_observed_fall(self):
        self.baseline()
        observation = person(y=240, angle=80)
        points = list(observation.keypoints)
        points[16] = (0, 0, 0.1)
        observation = replace(observation, keypoints=tuple(points))
        events = []
        for timestamp in (0.8, 1.0, 1.2, 1.4, 1.6):
            events.extend(self.feed(timestamp, observation))
        self.assertIn(CONFIRMED_FALL, states(events))

    def test_observed_descent_and_visible_leg_can_confirm_with_one_shoulder_hidden(self):
        self.baseline()
        observation = person(y=240, angle=80, fallen=0.95)
        points = list(observation.keypoints)
        points[5] = (0, 0, 0.1)
        events = []
        for step in range(4, 15):
            events.extend(self.feed(step * 0.2, replace(observation, keypoints=tuple(points))))
        self.assertIn(UNOBSERVED, states(events))
        confirmed = [e for e in events if e.state == CONFIRMED_FALL]
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].reason, "partial_landing_confirmed")

    def test_initial_static_horizontal_stays_unresolved_not_confirmed(self):
        events = []
        for step in range(40):
            events.extend(self.feed(step * 0.2, person(y=240, angle=85, fallen=0.95)))
        self.assertEqual(states(events).count(SUSPECTED_FALL), 1)
        self.assert_not_confirmed(events)
        self.assertFalse(any(e.reason == "suspicion_timeout" for e in events))
        self.assertEqual(self.engine.snapshot(7.8)[0].state, SUSPECTED_FALL)

    def test_initial_horizontal_without_label_warns_without_claiming_fall(self):
        events = []
        for step in range(40):
            events.extend(self.feed(step * 0.2, person(y=240, angle=90)))
        self.assertEqual(states(events), [SUSPECTED_FALL])
        self.assertEqual(events[0].reason, "persistent_low_pose")
        self.assertGreaterEqual(events[0].timestamp, self.engine.config.down_seconds)
        self.assert_not_confirmed(events)

    def test_static_seated_does_not_fall(self):
        events = []
        for step in range(20):
            events.extend(self.feed(step * 0.2, seated()))
        self.assertEqual(events, [])

    def test_slow_lying_down_cannot_confirm(self):
        events = self.baseline()
        for step in range(1, 31):
            angle = min(85, step * 4)
            observation = person(y=160 + min(80, step * 4), angle=angle,
                                 fallen=0.95 if angle >= 65 else 0)
            events.extend(self.feed(0.6 + step * 0.2, observation))
        self.assert_not_confirmed(events)

    def test_quick_sit_and_crouch_do_not_fall(self):
        for crouched in (False, True):
            with self.subTest(crouched=crouched):
                engine = FallEngine()
                events = self.baseline(engine)
                for step in range(1, 15):
                    events.extend(self.feed(0.6 + step * 0.2, seated(crouched=crouched), engine))
                self.assert_not_confirmed(events)
                self.assertNotIn(SUSPECTED_FALL, states(events))

    def test_rapid_downward_motion_without_torso_rotation_is_not_a_fall(self):
        self.baseline()
        events = []
        for timestamp in (0.8, 1.0, 1.2, 1.4, 1.6):
            events.extend(self.feed(timestamp, person(y=250)))
        self.assertEqual(events, [])

    def test_rotation_without_descent_is_not_a_fall(self):
        self.baseline()
        events = []
        for timestamp in (0.8, 1.0, 1.2, 1.4, 1.6):
            events.extend(self.feed(timestamp, person(y=160, angle=85)))
        self.assert_not_confirmed(events)

    def test_bounding_boxes_without_pose_never_confirm_but_label_suspects_immediately(self):
        events = []
        for step in range(20):
            box = (200, 50, 260, 250) if step < 4 else (190, 250, 390, 310)
            observation = PersonObservation(box, 0.95, 0 if step < 4 else 0.95)
            emitted = self.feed(step * 0.2, observation)
            events.extend(emitted)
            if step == 4:
                self.assertIn(SUSPECTED_FALL, states(emitted))
                self.assertEqual(self.engine.snapshot(0.8)[0].state, UNOBSERVED)
        self.assert_not_confirmed(events)
        self.assertEqual(states(events).count(SUSPECTED_FALL), 1)

    def test_observed_descent_and_repeated_fallen_labels_can_verify_hidden_legs(self):
        self.baseline()
        events = []
        for step in range(1, 15):
            events.extend(self.feed(0.6 + step * 0.2,
                                    person(y=240, angle=80, fallen=0.95, partial=True)))
        self.assertIn(SUSPECTED_FALL, states(events))
        self.assertIn(UNOBSERVED, states(events))
        confirmed = [e for e in events if e.state == CONFIRMED_FALL]
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].reason, "partial_landing_confirmed")

    def test_partial_baseline_cannot_qualify_later_reliable_fall(self):
        for timestamp in (0, 0.2, 0.4, 0.6):
            self.feed(timestamp, person(partial=True))
        events = []
        for timestamp in (0.8, 1.0, 1.2, 1.4, 1.6):
            events.extend(self.feed(timestamp, person(y=240, angle=80, fallen=0.95)))
        self.assert_not_confirmed(events)

    def test_short_occlusion_requires_new_observed_down_evidence_to_confirm(self):
        self.baseline()
        suspected = self.feed(0.8, person(y=240, angle=80, fallen=0.95))[0]
        self.assertEqual(suspected.state, SUSPECTED_FALL)
        missing = self.feed(1.0)
        self.assertEqual(states(missing), [UNOBSERVED])
        self.assertEqual(missing[0].event_id, suspected.event_id)
        self.assertEqual(self.feed(1.2), ())
        self.assertEqual(self.engine.snapshot(1.2)[0].event_id, suspected.event_id)
        events = []
        for timestamp in (1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0):
            events.extend(self.feed(timestamp, person(y=240, angle=80, fallen=0.95)))
        confirmations = [e for e in events if e.state == CONFIRMED_FALL]
        self.assertEqual(len(confirmations), 1)
        self.assertGreaterEqual(confirmations[0].timestamp, 2.4)
        self.assertEqual(self.engine.snapshot(2.6)[0].event_id, suspected.event_id)

    def test_unreliable_pose_during_verification_clears_motion_evidence(self):
        self.baseline()
        self.feed(0.8, person(y=240, angle=80, fallen=0.95))
        self.feed(1.0, person(y=240, angle=80, fallen=0.95))
        gap = self.feed(1.2, person(y=240, angle=80, fallen=0.95, partial=True))
        self.assertEqual(states(gap), [UNOBSERVED])
        events = []
        for timestamp in (1.4, 1.6, 1.8, 2.0, 2.2):
            events.extend(self.feed(timestamp, person(y=240, angle=80, fallen=0.95)))
        self.assert_not_confirmed(events)

    def test_reacquisition_after_gap_cannot_use_pre_gap_baseline(self):
        self.baseline()
        self.feed(0.8)
        events = []
        for timestamp in (1.0, 1.2, 1.4, 1.6, 1.8):
            events.extend(self.feed(timestamp, person(y=240, angle=80, fallen=0.95)))
        self.assert_not_confirmed(events)

    def test_reacquisition_must_rebuild_full_baseline(self):
        self.baseline()
        self.feed(0.8)
        self.feed(1.0, person())
        events = []
        for timestamp in (1.2, 1.4, 1.6, 1.8, 2.0):
            events.extend(self.feed(timestamp, person(y=240, angle=80, fallen=0.95)))
        self.assert_not_confirmed(events)
        engine = FallEngine()
        self.baseline(engine)
        self.feed(0.8, engine=engine)
        self.baseline(engine, start=1.0)
        events = []
        for timestamp in (1.8, 2.0, 2.2, 2.4, 2.6):
            events.extend(self.feed(timestamp, person(y=240, angle=80, fallen=0.95), engine))
        self.assertIn(CONFIRMED_FALL, states(events))

    def test_large_time_gap_without_explicit_missing_frame_breaks_continuity(self):
        self.baseline()
        events = list(self.feed(1.1, person(y=240, angle=80, fallen=0.95)))
        for timestamp in (1.3, 1.5, 1.7, 1.9):
            events.extend(self.feed(timestamp, person(y=240, angle=80, fallen=0.95)))
        self.assertIn(UNOBSERVED, states(events))
        self.assert_not_confirmed(events)

    def test_multiple_tracks_keep_independent_motion_and_ids(self):
        events = []
        for step in range(10):
            timestamp = step * 0.2
            falling = person(250) if step < 4 else person(250, 240, 80, 0.95)
            static = person(800, 240, 80, 0.95)
            observations = (falling, static) if step % 2 else (static, falling)
            events.extend(self.engine.update(observations, timestamp, SHAPE))
        confirmed = [e for e in events if e.state == CONFIRMED_FALL]
        self.assertEqual(len(confirmed), 1)
        views = self.engine.snapshot(1.8)
        self.assertEqual(len(views), 2)
        self.assertEqual(confirmed[0].track_id, next(v.track_id for v in views if v.xyxy[0] < 500))
        self.assertEqual(len({v.track_id for v in views}), 2)

    def test_ambiguous_crossing_does_not_assert_identity_or_confirm(self):
        for timestamp in (0, 0.2, 0.4, 0.6):
            self.engine.update((person(250), person(330)), timestamp, SHAPE)
        old_ids = {view.track_id for view in self.engine.snapshot(0.6)}
        events = self.engine.update((person(290), person(290)), 0.8, SHAPE)
        self.assertEqual(states(events), [UNOBSERVED, UNOBSERVED])
        views = self.engine.snapshot(0.8)
        self.assertEqual({view.track_id for view in views}, old_ids)
        self.assertTrue(all(not v.observation_quality for v in views))
        for timestamp in (1.0, 1.2, 1.4, 1.6):
            events += self.engine.update((person(250, 240, 80, 0.95), person(330)), timestamp, SHAPE)
        self.assert_not_confirmed(events)

    def test_one_to_many_ambiguity_does_not_duplicate_track(self):
        self.baseline()
        events = self.engine.update((person(245), person(255)), 0.8, SHAPE)
        self.assertEqual(states(events), [UNOBSERVED])
        self.assertEqual(len(self.engine.snapshot(0.8)), 1)

    def test_nonmatch_and_offscreen_do_not_trigger_suspicion(self):
        self.baseline()
        events = self.engine.update((person(1000), person(-500, fallen=0.95)), 0.8, SHAPE)
        self.assertNotIn(SUSPECTED_FALL, states(events))
        self.assertIn(UNOBSERVED, states(events))
        self.assertEqual(len(self.engine.snapshot(0.8)), 2)

    def test_duplicate_timestamp_is_noop_even_with_different_shape_or_observations(self):
        self.baseline()
        before = self.engine.snapshot(0.6)
        self.assertEqual(self.engine.update((person(y=240, angle=80, fallen=1),),
                                            0.6, (360, 640)), ())
        self.assertEqual(self.engine.snapshot(0.6), before)
        self.assertEqual(self.engine.mark_unobserved(0.6), ())
        self.assertEqual(self.engine.snapshot(0.6), before)

    def test_duplicates_cannot_accrue_verification_samples_or_time(self):
        self.baseline()
        self.feed(0.8, person(y=240, angle=80, fallen=0.95))
        before = self.engine.snapshot(0.8)
        for _ in range(50):
            self.assertEqual(self.feed(0.8, person(y=240, angle=80, fallen=0.95)), ())
        self.assertEqual(self.engine.snapshot(0.8), before)
        self.assert_not_confirmed(self.feed(1.0, person(y=240, angle=80, fallen=0.95)))

    def test_skipped_posture_samples_do_not_rearm_fallen_label(self):
        events = []
        for step in range(40):
            observed = step % 3 == 0
            observation = replace(person(y=240, angle=80, fallen=0.95 if observed else 0),
                                  posture_observed=observed)
            events.extend(self.feed(step * 0.2, observation))
        self.assertEqual(states(events).count(SUSPECTED_FALL), 1)
        self.assert_not_confirmed(events)

    def test_shape_change_resets_tracks_and_temporal_history(self):
        self.baseline()
        old_id = self.engine.snapshot(0.6)[0].track_id
        events = list(self.engine.update((person(y=240, angle=80, fallen=0.95),),
                                         0.8, (800, 1280, 3)))
        self.assertNotEqual(self.engine.snapshot(0.8)[0].track_id, old_id)
        for timestamp in (1.0, 1.2, 1.4, 1.6):
            events.extend(self.engine.update((person(y=240, angle=80, fallen=0.95),),
                                             timestamp, (800, 1280, 3)))
        self.assert_not_confirmed(events)

    def test_backward_time_resets_without_reusing_ids(self):
        self.fall()
        old = self.engine.snapshot(1.8)[0]
        events = self.feed(0.0, person(y=240, angle=80, fallen=0.95))
        new = self.engine.snapshot(0.0)[0]
        self.assertNotEqual(old.track_id, new.track_id)
        self.assertNotEqual(old.event_id, new.event_id)
        self.assert_not_confirmed(events)
        self.engine.mark_unobserved(0.1)
        self.assertEqual(self.engine.mark_unobserved(0.0), ())
        self.assertEqual(self.engine.snapshot(0.0)[0].state, UNOBSERVED)

    def test_confirmed_fall_latches_and_deduplicates_while_person_stays_down(self):
        events = self.fall()
        for step in range(10, 80):
            events.extend(self.feed(step * 0.2, person(y=240, angle=80, fallen=0.95)))
        self.assertEqual(states(events).count(CONFIRMED_FALL), 1)
        self.assertEqual(states(events).count(SUSPECTED_FALL), 1)
        self.assertEqual(self.engine.snapshot(15.8)[0].state, CONFIRMED_FALL)

    def test_cooldown_requires_elapsed_time_and_observed_upright_recovery_to_rearm(self):
        events = self.fall()
        first_id = next(e.event_id for e in events if e.state == CONFIRMED_FALL)
        # Stay down past the cooldown deadline, then rebuild upright evidence.
        for step in range(10, 40):
            self.feed(step * 0.2, person(y=240, angle=80, fallen=0.95))
        for step in range(40, 49):
            events.extend(self.feed(step * 0.2, person()))
        for step in range(49, 55):
            events.extend(self.feed(step * 0.2, person(y=240, angle=80, fallen=0.95)))
        confirmations = [e for e in events if e.state == CONFIRMED_FALL]
        self.assertEqual(len(confirmations), 2)
        self.assertNotEqual(confirmations[-1].event_id, first_id)

    def test_stale_mark_only_transitions_once_preserves_id_and_blocks_confirmation(self):
        self.baseline()
        event = self.feed(0.8, person(y=240, angle=80, fallen=0.95))[0]
        stale = self.engine.mark_unobserved(1.0, "ai_stale")
        self.assertEqual(states(stale), [UNOBSERVED])
        self.assertEqual(stale[0].event_id, event.event_id)
        self.assertEqual(stale[0].reason, "ai_stale")
        self.assertEqual(self.engine.mark_unobserved(1.1, "stream_stale"), ())
        events = []
        for timestamp in (1.2, 1.4, 1.6, 1.8, 2.0):
            events.extend(self.feed(timestamp, person(y=240, angle=80, fallen=0.95)))
        self.assert_not_confirmed(events)

    def test_watchdog_time_does_not_reset_identity_for_inflight_source_result(self):
        self.baseline()
        suspected = self.feed(0.8, person(y=240, angle=80, fallen=0.95))[0]
        stale = self.engine.mark_unobserved(1.15, "ai_stale")
        self.assertEqual(states(stale), [UNOBSERVED])
        # The inference completed after watchdog time, but its SOURCE timestamp
        # is older. Neither reset the identity nor count this old work as recovery.
        self.assertEqual(self.feed(1.0, person(y=240, angle=80, fallen=0.95)), ())
        view = self.engine.snapshot(1.15)[0]
        self.assertEqual(view.track_id, suspected.track_id)
        self.assertEqual(view.event_id, suspected.event_id)
        self.assertEqual(view.state, UNOBSERVED)
        self.assertEqual(self.engine.mark_unobserved(1.15), ())
        self.assertEqual(self.engine.mark_unobserved(1.1), ())
        events = []
        for timestamp in (1.2, 1.4, 1.6, 1.8, 2.0):
            events.extend(self.feed(timestamp, person(y=240, angle=80, fallen=0.95)))
        self.assert_not_confirmed(events)
        self.assertEqual(self.engine.snapshot(2.0)[0].event_id, suspected.event_id)

    def test_fallen_label_during_long_upright_period_is_not_immediately_dismissed(self):
        for step in range(10):
            self.feed(step * 0.2, person())
        events = self.feed(2.0, person(fallen=0.95))
        self.assertEqual(states(events), [SUSPECTED_FALL])
        self.assertEqual(self.engine.snapshot(2.0)[0].state, SUSPECTED_FALL)

    def test_unobserved_suspicion_stays_unresolved_until_tracking_expires(self):
        engine = FallEngine(replace(FallConfig(), max_track_gap=5))
        self.baseline(engine)
        suspected = self.feed(0.8, person(y=240, angle=80, fallen=0.95), engine)[0]
        engine.mark_unobserved(1.0)
        self.assertEqual(engine.mark_unobserved(3.4), ())
        view = engine.snapshot(3.4)[0]
        self.assertEqual(view.state, UNOBSERVED)
        self.assertEqual(view.event_id, suspected.event_id)
        self.assertEqual(engine.mark_unobserved(3.6), ())
        self.assertEqual(engine.mark_unobserved(6.0), ())
        self.assertEqual(engine.snapshot(6.0), ())

    def test_unreliable_pose_cannot_dismiss_suspicion_on_timeout(self):
        self.baseline()
        suspected = self.feed(0.8, person(y=240, angle=80, fallen=0.95))[0]
        events = []
        for step in range(5, 25):
            events.extend(self.feed(step * 0.2, person(y=240, angle=80, fallen=0.95, partial=True)))
        self.assertEqual(states(events), [UNOBSERVED])
        self.assertEqual(self.engine.snapshot(4.8)[0].event_id, suspected.event_id)
        self.assertEqual(self.engine.snapshot(4.8)[0].state, UNOBSERVED)

    def test_expired_track_is_retired_and_cannot_transmit_baseline(self):
        self.baseline()
        old_id = self.engine.snapshot(0.6)[0].track_id
        events = self.feed(2.0, person(y=240, angle=80, fallen=0.95))
        self.assertEqual(states(events), [UNOBSERVED, SUSPECTED_FALL])
        self.assertNotEqual(self.engine.snapshot(2.0)[0].track_id, old_id)
        self.assert_not_confirmed(events)

    def test_suspicion_recovers_with_same_event_id(self):
        self.baseline()
        suspected = self.feed(0.8, person(y=240, angle=80, fallen=0.95))[0]
        events = []
        for timestamp in (1.0, 1.2, 1.4, 1.6, 1.8):
            events.extend(self.feed(timestamp, person()))
        recovered = next(e for e in events if e.reason == "upright_recovery")
        self.assertEqual(recovered.event_id, suspected.event_id)
        self.assertEqual(recovered.state, NORMAL)
        self.assert_not_confirmed(events)
        self.assertIsNone(self.engine.snapshot(1.8)[0].event_id)

    def test_snapshot_does_not_advance_time_or_verification(self):
        self.baseline()
        self.feed(0.8, person(y=240, angle=80, fallen=0.95))
        before = self.engine.snapshot(0.8)
        self.assertEqual(self.engine.snapshot(10)[0].state, UNOBSERVED)
        self.assertEqual(self.engine.snapshot(0.8), before)
        self.assert_not_confirmed(self.feed(1.0, person(y=240, angle=80, fallen=0.95)))

    def test_low_detection_confidence_is_missing_not_motion(self):
        self.baseline()
        events = self.feed(0.8, person(y=240, angle=80, fallen=0.95, confidence=0.1))
        self.assertEqual(states(events), [UNOBSERVED])

    def test_normalized_geometry_is_scale_invariant(self):
        observations = []
        for step in range(10):
            o = person() if step < 4 else person(y=240, angle=80, fallen=0.95)
            observations.append(replace(o, xyxy=tuple(v * 0.5 for v in o.xyxy),
                                        keypoints=tuple((x * 0.5, y * 0.5, c) for x, y, c in o.keypoints)))
        events = []
        for step, observation in enumerate(observations):
            events.extend(self.feed(step * 0.2, observation))
        self.assertIn(CONFIRMED_FALL, states(events))

    def test_configurable_verification_duration(self):
        engine = FallEngine(replace(FallConfig(), verify_seconds=1.2))
        events = self.fall(engine)
        self.assert_not_confirmed(events)
        for timestamp in (2.0, 2.2, 2.4):
            events.extend(self.feed(timestamp, person(y=240, angle=80, fallen=0.95), engine))
        self.assertIn(CONFIRMED_FALL, states(events))


class FallContractTests(unittest.TestCase):
    def test_frozen_contracts_are_deeply_immutable_and_accept_tuple_like_inputs(self):
        observation = PersonObservation([0, 0, 60, 200], 0.9, keypoints=[])
        self.assertIsInstance(observation.xyxy, tuple)
        self.assertEqual(observation.keypoints, ())
        with self.assertRaises(FrozenInstanceError):
            observation.confidence = 0.1
        event = FallEvent(1, "fall-1", SUSPECTED_FALL, 1.0, "fallen_label", (("drop", 0.5),))
        with self.assertRaises(FrozenInstanceError):
            event.state = CONFIRMED_FALL
        with self.assertRaises(TypeError):
            event.features[0] = ("drop", 1)
        engine = FallEngine()
        engine.update((person(),), 0, SHAPE)
        view = engine.snapshot(0)[0]
        with self.assertRaises(FrozenInstanceError):
            view.state = UNOBSERVED
        self.assertIsInstance(view.features, tuple)

    def test_all_config_fields_validate_nonfinite_values(self):
        config = FallConfig()
        for item in fields(config):
            invalid = (math.nan, math.inf, -math.inf, -1, 1, None) if item.name == "keypoint_smoothing" else (math.nan, math.inf, -math.inf, -1, True)
            for value in invalid:
                with self.subTest(field=item.name, value=value):
                    with self.assertRaises(ValueError):
                        replace(config, **{item.name: value})

    def test_config_ranges_and_relationships(self):
        for kwargs in (
            {"min_detection_confidence": 1.1}, {"min_visible_keypoints": 7},
            {"min_visible_keypoints": 18}, {"baseline_min_samples": 1},
            {"verify_min_samples": 1}, {"verify_seconds": 0},
            {"verify_seconds": 3}, {"max_track_gap": 0.1},
            {"motion_window": 0.1}, {"post_min_angle": 100},
            {"alternate_min_angle": 60}, {"association_max_area_ratio": 0.5},
            {"fallen_label_reset_threshold": 0.8},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    FallConfig(**kwargs)

    def test_observation_validation(self):
        for box in ((0, 0, 0, 1), (1, 0, 0, 1), (0, 0, math.nan, 1), (0, 0, 1)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                PersonObservation(box, 0.9)
        for score in (-1, math.nan, 1.1, True):
            with self.assertRaises(ValueError):
                PersonObservation((0, 0, 10, 20), score)
            with self.assertRaises(ValueError):
                PersonObservation((0, 0, 10, 20), 0.9, score)
        for points in (((1, 2, 0.9),), ((1, 2, 1.1),) * 17,
                       ((1, math.inf, 0.9),) * 17, ((1, 2),) * 17):
            with self.assertRaises(ValueError):
                PersonObservation((0, 0, 10, 20), 0.9, keypoints=points)

    def test_timestamp_shape_and_input_validation(self):
        engine = FallEngine()
        for timestamp in (-1, math.nan, math.inf, True):
            with self.assertRaises(ValueError):
                engine.update((), timestamp, SHAPE)
            with self.assertRaises(ValueError):
                engine.mark_unobserved(timestamp)
            with self.assertRaises(ValueError):
                engine.snapshot(timestamp)
        for shape in ((), (100,), (0, 10), (10, -1), (10.0, 20), (True, 20), (10, 20, 3, 4)):
            with self.assertRaises(ValueError):
                engine.update((), 0, shape)
        with self.assertRaises(TypeError):
            engine.update((object(),), 0, SHAPE)
        with self.assertRaises(TypeError):
            FallEngine(object())
        with self.assertRaises(ValueError):
            engine.mark_unobserved(0, "")
        self.assertEqual(engine.update((), 0, SHAPE), ())
        self.assertEqual(engine.snapshot(0), ())


if __name__ == "__main__":
    unittest.main()
