"""Filter behaviour, confidence gaps, and per-track temporal isolation."""

from dataclasses import replace
import math

import pytest

from ai.fall_engine import FallConfig, FallEngine
from ai.point_filter import KeypointSmoother
from tests.test_fall_engine import SHAPE, person


def points(x, confidence=0.95):
    return ((x, 100.0, confidence),) * 17


def test_stationary_jitter_is_reduced_but_fast_motion_remains_responsive():
    smoother = KeypointSmoother(scale=200)
    smoother.filter(points(100), 0)
    errors = []
    for index in range(1, 31):
        value = smoother.filter(points(100 + (-1) ** index * 2), index * 0.1)
        errors.append(abs(value[0][0] - 100))
    assert sum(errors) / len(errors) < 1.1
    moved = smoother.filter(points(200), 3.1)[0][0]
    assert moved > 180


def test_confidence_is_unchanged_and_reacquired_joint_starts_fresh():
    smoother = KeypointSmoother(scale=200)
    smoother.filter(points(100), 0)
    assert smoother.filter(points(900, 0.1), 0.1) == points(900, 0.1)
    assert smoother.filter(points(300), 0.2) == points(300)
    smoother.reset()
    assert smoother.filter(points(400), 0) == points(400)


def test_resolution_independent_and_irregular_sampling():
    first, second = KeypointSmoother(scale=200), KeypointSmoother(scale=800)
    for timestamp, x in ((0, 100), (0.05, 102), (0.19, 200), (0.3, 230)):
        a = first.filter(points(x), timestamp)[0][0]
        b = second.filter(tuple((px * 4, py * 4, c) for px, py, c in points(x)), timestamp)[0][0]
        assert b == pytest.approx(a * 4)
    for timestamp in (0.3, 0.2, math.nan, math.inf, True):
        with pytest.raises(ValueError):
            first.filter(points(100), timestamp)


def test_engine_keeps_raw_association_and_separates_filter_state():
    engine = FallEngine(FallConfig(keypoint_smoothing=True))
    a, b = person(x=250), person(x=700)
    engine.update((a, b), 0, SHAPE)
    engine.update((person(x=252), person(x=702)), 0.1, SHAPE)
    tracks = list(engine._tracks.values())
    assert tracks[0].keypoint_filter is not tracks[1].keypoint_filter
    assert tracks[0].keypoints == person(x=252).keypoints
    assert tracks[1].keypoints == person(x=702).keypoints
    engine.mark_unobserved(0.2)
    assert all(t.keypoint_filter is None for t in tracks)
    engine.update((person(x=300),), 0.3, SHAPE)
    active = next(t for t in engine._tracks.values() if t.visible)
    assert active.keypoint_filter._joints[5][0][0] * active.keypoint_filter.scale == pytest.approx(person(x=300).keypoints[5][0])


def test_weak_detection_and_missing_pose_clear_filter():
    engine = FallEngine(FallConfig(keypoint_smoothing=True))
    engine.update((person(),), 0, SHAPE)
    engine.update((replace(person(), confidence=0.25),), 0.1, SHAPE)
    assert engine._tracks[1].keypoint_filter is None
    engine.update((person(),), 0.2, SHAPE)
    engine.update((replace(person(), keypoints=()),), 0.3, SHAPE)
    assert engine._tracks[1].keypoint_filter is None