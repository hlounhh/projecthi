"""Visible motion and fresh partial poses, including furniture-height landings."""

from dataclasses import replace

import pytest

from ai.fall_engine import CONFIRMED_FALL, FallConfig, FallEngine
from tests.test_fall_engine import SHAPE, person, seated


def hidden_upper(y=240):
    observation = person(y=y, angle=80)
    return replace(observation, keypoints=tuple(
        (x, y, confidence if i >= 11 else .1)
        for i, (x, y, confidence) in enumerate(observation.keypoints)))


def seed(y=160):
    engine = FallEngine()
    for i in range(7):
        engine.update((person(y=y),), i / 10, SHAPE)
    engine.update((person(y=y + 80, angle=30),), .7, SHAPE)
    assert engine._tracks[1].partial.candidate is not None
    return engine


def feed(engine, observation, start=8, end=30):
    events = []
    for i in range(start, end):
        events.extend(engine.update((observation,), i / 10, SHAPE))
    return [e for e in events if e.state == CONFIRMED_FALL]


@pytest.mark.parametrize('origin_y', [120, 300])
def test_hidden_shoulders_can_be_verified_from_extended_visible_legs_at_any_image_height(origin_y):
    engine = seed(origin_y)
    confirmed = feed(engine, hidden_upper(origin_y + 80))
    assert len(confirmed) == 1
    assert confirmed[0].reason == 'partial_landing_confirmed'
    assert dict(confirmed[0].features)['partial_landing_observed_seconds'] >= .6 - 1e-9


@pytest.mark.parametrize('observation', [hidden_upper(), person(y=240, angle=80, fallen=.95), seated()])
def test_static_low_postures_without_observed_descent_cannot_confirm(observation):
    assert not feed(FallEngine(), observation, start=0, end=50)


def test_body_disappearing_after_descent_does_not_verify_any_time():
    engine = seed()
    missing = replace(hidden_upper(), keypoints=tuple((x, y, .1) for x, y, _ in hidden_upper().keypoints))
    assert not feed(engine, missing)
    assert engine._tracks[1].partial.candidate is None


def test_low_seated_and_folded_leg_postures_do_not_count_as_horizontal_landing():
    engine = seed()
    crouch = seated(y=240, crouched=True)
    crouch = replace(crouch, xyxy=(200, 220, 330, 310))
    assert not feed(engine, crouch)


@pytest.mark.parametrize('failure', ['stale', 'ambiguous', 'long_gap', 'displaced_joints'])
def test_invalid_identity_cannot_reuse_partial_motion(failure):
    engine = seed()
    landing = hidden_upper()
    start = 9
    if failure == 'stale':
        engine.mark_unobserved(.8, reason='pose_unreliable')
    elif failure == 'ambiguous':
        engine.update((landing, landing), .8, SHAPE)
    elif failure == 'long_gap':
        engine.update((), 1.5, SHAPE)
        start = 16
    else:
        engine.update((), .8, SHAPE)
        landing = replace(landing, keypoints=tuple((x + 400, y, confidence)
                                                  for x, y, confidence in landing.keypoints))
    assert not feed(engine, landing, start=start, end=40)


def test_weak_predictions_without_confident_landmarks_do_not_extend_grace():
    engine = seed()
    weak = replace(hidden_upper(), confidence=.25)
    weak = replace(weak, keypoints=tuple((x, y, min(confidence, .55))
                                        for x, y, confidence in weak.keypoints))
    assert not feed(engine, weak)
    assert engine._tracks[1].partial.candidate is None


def test_existing_motion_can_be_verified_by_weak_detection_with_many_confident_landmarks():
    engine = seed()
    confirmed = feed(engine, replace(person(y=240, angle=80), confidence=.25))
    assert len(confirmed) == 1
    assert confirmed[0].reason == 'partial_landing_confirmed'
    assert dict(confirmed[0].features)['partial_weak_detection']


def test_repeated_brief_reacquisitions_cannot_extend_original_motion_deadline():
    engine = seed()
    started = engine._tracks[1].partial.candidate[0]
    for i in range(8, 45):
        observation = hidden_upper() if i % 5 == 0 else replace(hidden_upper(), confidence=.25)
        assert not any(e.state == CONFIRMED_FALL for e in engine.update((observation,), i / 10, SHAPE))
        candidate = engine._tracks[1].partial.candidate
        assert candidate is None or candidate[0] == started
    assert engine._tracks[1].partial.candidate is None
    assert not feed(engine, hidden_upper(), start=45, end=60)


def test_short_gap_requires_fresh_identity_and_restarts_observed_time():
    engine = seed()
    feed(engine, hidden_upper(), end=11)
    for t in (1.1, 1.2, 1.3, 1.4):
        engine.update((), t, SHAPE)
    for t in (1.5, 1.6):
        assert not any(e.state == CONFIRMED_FALL for e in engine.update((hidden_upper(),), t, SHAPE))
        assert engine._tracks[1].partial.observed_down == 0
    assert not feed(engine, hidden_upper(), start=17, end=23)
    assert len(feed(engine, hidden_upper(), start=23, end=28)) == 1


def test_brief_missing_frames_earn_no_landing_duration():
    engine = seed()
    feed(engine, hidden_upper(), end=10)
    evidence = engine._tracks[1].partial
    observed = evidence.observed_down
    for t in (1., 1.1):
        engine.update((), t, SHAPE)
        assert evidence.observed_down == observed
    engine.update((hidden_upper(),), 1.2, SHAPE)
    assert evidence.observed_down == pytest.approx(observed + .1)


def test_upright_recovery_cancels_motion_before_later_lying():
    engine = seed()
    assert not feed(engine, person(y=240), start=8, end=17)
    assert not feed(engine, hidden_upper(), start=17, end=40)


def test_slow_descent_cannot_seed_partial_confirmation():
    engine = FallEngine()
    for i in range(30):
        engine.update((person(y=160 + i * 2),), i / 10, SHAPE)
    assert not feed(engine, hidden_upper(), start=30, end=55)


def test_invalid_partial_drop_configuration_is_rejected():
    with pytest.raises(ValueError):
        FallConfig(partial_min_drop=-.1)