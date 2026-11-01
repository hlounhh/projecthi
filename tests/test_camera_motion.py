import numpy as np
import pytest

from ai.camera_motion import CameraMotionEstimator, MotionCompensatedFallEngine
from ai.fall_engine import CONFIRMED_FALL, SUSPECTED_FALL, FallEvent, PersonObservation


def _shifted(image, dx, dy):
    out = np.roll(image, dy, axis=0)
    return np.roll(out, dx, axis=1)


def test_estimator_recovers_shift():
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    estimator = CameraMotionEstimator()
    estimator.update(base)
    dx, dy, _, confidence = estimator.update(_shifted(base, 12, -8))
    assert (dx, dy) == pytest.approx((12.0, -8.0), abs=3.0)
    assert confidence >= 0.5


def test_estimator_confidence_low_on_noise():
    rng = np.random.default_rng(1)
    first = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    second = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    estimator = CameraMotionEstimator()
    estimator.update(first)
    _, _, _, confidence = estimator.update(second)
    assert confidence < 0.3


def _observation(x, confidence=0.9):
    return PersonObservation((x, 100.0, x + 60.0, 200.0), confidence)


class _FakeEngine:
    def __init__(self, events=()):
        self.events = tuple(events)
        self.calls = []

    def update(self, observations, timestamp, frame_shape):
        self.calls.append((observations, timestamp, frame_shape))
        return self.events


def test_untrusted_motion_downgrades_confirmed():
    event = FallEvent(1, "fall-0", CONFIRMED_FALL, 1.0, "reason", ())
    fake = _FakeEngine([event])
    engine = MotionCompensatedFallEngine(fake)
    rng = np.random.default_rng(3)
    engine.update((_observation(100.0),), 0.5, (240, 320),
                  frame=rng.integers(0, 255, (240, 320), dtype=np.uint8))
    result = engine.update((_observation(100.0),), 1.0, (240, 320),
                           frame=rng.integers(0, 255, (240, 320), dtype=np.uint8))
    assert engine.trusted is False
    assert result[0].state == SUSPECTED_FALL


def test_without_frame_passes_through():
    event = FallEvent(1, "fall-0", CONFIRMED_FALL, 1.0, "reason", ())
    fake = _FakeEngine([event])
    engine = MotionCompensatedFallEngine(fake)
    result = engine.update((_observation(100.0),), 1.0, (240, 320))
    assert result[0].state == CONFIRMED_FALL
    assert len(fake.calls) == 1


def test_trusted_motion_shifts_and_keeps_confirmed():
    rng = np.random.default_rng(5)
    base = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    event = FallEvent(1, "fall-0", CONFIRMED_FALL, 2.0, "reason", ())
    fake = _FakeEngine([event])
    engine = MotionCompensatedFallEngine(fake)
    engine.update((_observation(100.0),), 1.0, (240, 320), frame=base)
    assert engine.trusted is True
    result = engine.update((_observation(100.0),), 2.0, (240, 320), frame=_shifted(base, 10, 0))
    assert result[0].state == CONFIRMED_FALL
    observations, _, _ = fake.calls[-1]
    assert observations[0].xyxy[0] == pytest.approx(90.0, abs=4.0)
