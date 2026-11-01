"""A measured fall can finish curled with hidden legs; missing time cannot verify it."""

from dataclasses import replace

import pytest

from ai.fall_engine import CONFIRMED_FALL, FallEngine
from tests.test_fall_engine import SHAPE, person, seated


def compact_landing(*, torso_y=315, one_side=False, confidence=0.95):
    observation = person(y=torso_y, box=(200, torso_y - 35, 350, torso_y + 65))
    points = list(observation.keypoints)
    for index in (5, 6):
        points[index] = (240 if index == 5 else 260, torso_y - 15, 0.95)
    for index in (11, 12):
        points[index] = (240 if index == 11 else 260, torso_y + 15, 0.95)
    for index in (13, 14, 15, 16):
        points[index] = (*points[index][:2], 0.1)
    if one_side:
        points[5] = (*points[5][:2], 0.1)
    return replace(observation, keypoints=tuple(points), confidence=confidence)


def seed_fall():
    engine = FallEngine()
    for index in range(7):
        engine.update((person(),), index / 10, SHAPE)
    events = engine.update((person(y=240, angle=80),), 0.7, SHAPE)
    assert engine._tracks[1].occlusion.motion_landing is not None
    assert not any(event.state == CONFIRMED_FALL for event in events)
    return engine


def feed_landing(engine, start=8, end=20, observation=None):
    events = []
    for index in range(start, end):
        events.extend(engine.update((observation or compact_landing(),), index / 10, SHAPE))
    return events


def confirmations(events):
    return [event for event in events if event.state == CONFIRMED_FALL]


def test_rapid_fall_then_foreshortened_curled_landing_confirms_once():
    engine = seed_fall()
    events = feed_landing(engine, end=35)
    confirmed = confirmations(events)
    assert len(confirmed) == 1
    assert confirmed[0].reason == 'curled_landing_confirmed'
    assert confirmed[0].timestamp >= 1.4
    assert dict(confirmed[0].features)['curled_landing_observed_seconds'] >= 0.6 - 1e-9
    assert not dict(confirmed[0].features)['lower_body_reliable']


def test_static_low_pose_and_sitting_cannot_supply_fall_motion():
    for observation in (compact_landing(), replace(compact_landing(), fallen_confidence=0.99),
                        seated(), person(y=240, angle=80)):
        engine = FallEngine()
        events = feed_landing(engine, start=0, end=50, observation=observation)
        assert not confirmations(events)


def test_fast_sitting_without_rapid_rotation_does_not_confirm():
    engine = FallEngine()
    for index in range(7):
        engine.update((person(),), index / 10, SHAPE)
    engine.update((person(y=240, angle=10),), 0.7, SHAPE)
    assert engine._tracks[1].occlusion.motion_landing is None
    assert not confirmations(feed_landing(engine))


@pytest.mark.parametrize('kind', ['weak', 'stale', 'ambiguous', 'long_gap', 'upright'])
def test_invalidated_motion_cannot_be_reused_for_later_low_pose(kind):
    engine = seed_fall()
    feed_landing(engine, end=10)
    start = 11
    if kind == 'weak':
        engine.update((compact_landing(confidence=0.25),), 1.0, SHAPE)
    elif kind == 'stale':
        engine.mark_unobserved(1.0)
    elif kind == 'ambiguous':
        engine._missing(engine._tracks[1], 1.0, 'ambiguous_association', [])
    elif kind == 'long_gap':
        engine.update((), 1.0, SHAPE)
        engine.update((), 1.1, SHAPE)
        engine.update((), 1.7, SHAPE)
        start = 18
    else:
        engine.update((person(),), 1.0, SHAPE)
    assert not confirmations(feed_landing(engine, start=start, end=35))


def test_one_visible_torso_side_never_supplies_verification():
    engine = seed_fall()
    assert not confirmations(feed_landing(engine, observation=compact_landing(one_side=True)))


def test_deep_bending_with_stationary_hips_is_not_a_landing():
    engine = seed_fall()
    observation = replace(compact_landing(), xyxy=(190, 205, 390, 335))
    points = list(observation.keypoints)
    # The shoulders dip deeply, but the pelvis stays near its original upright
    # level. A mean-torso drop alone would accept this compact bend.
    for index in (5, 6):
        points[index] = (points[index][0], 325, 0.95)
    for index in (11, 12):
        points[index] = (points[index][0], 225, 0.95)
    observation = replace(observation, keypoints=tuple(points))
    assert not confirmations(feed_landing(engine, observation=observation))


def test_missing_hip_resets_the_landing_verification_budget():
    engine = seed_fall()
    assert not confirmations(feed_landing(engine, end=13))
    observation = compact_landing()
    points = list(observation.keypoints)
    points[11] = (*points[11][:2], 0.1)
    engine.update((replace(observation, keypoints=tuple(points)),), 1.3, SHAPE)
    assert engine._tracks[1].occlusion.observed_curled == 0
    assert not confirmations(feed_landing(engine, start=14, end=22))
    assert len(confirmations(engine.update((compact_landing(),), 2.2, SHAPE))) == 1


def test_shallow_crouching_cannot_confirm_or_keep_motion_indefinitely():
    engine = seed_fall()
    assert not confirmations(feed_landing(engine, end=41, observation=compact_landing(torso_y=220)))
    assert engine._tracks[1].occlusion.motion_landing is None
    assert not confirmations(feed_landing(engine, start=41, end=65))


def test_sparse_samples_do_not_count_all_missing_time():
    engine = seed_fall()
    events = []
    for timestamp in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8):
        events.extend(engine.update((compact_landing(),), timestamp, SHAPE))
    assert not confirmations(events)
    events.extend(engine.update((compact_landing(),), 2.0, SHAPE))
    assert len(confirmations(events)) == 1


def test_expiring_older_occlusion_evidence_does_not_erase_newer_motion():
    engine = seed_fall()
    evidence = engine._tracks[1].occlusion
    evidence.candidate = (0.1, 200, 127.5)
    assert not confirmations(feed_landing(engine, end=30, observation=compact_landing(torso_y=220)))
    engine.update((compact_landing(torso_y=260),), 3.0, SHAPE)
    events = feed_landing(engine, start=31, end=38)
    assert len(confirmations(events)) == 1
    assert confirmations(events)[0].timestamp == pytest.approx(3.7)


@pytest.mark.parametrize('scale', [0.5, 2.0])
def test_same_fall_at_different_image_scales_has_same_confirmation_time(scale):
    engine = FallEngine()
    shape = (round(SHAPE[0] * scale), round(SHAPE[1] * scale), 3)
    events = []
    for index in range(20):
        observation = person() if index < 7 else person(y=240, angle=80) if index == 7 else compact_landing()
        observation = replace(observation,
                              xyxy=tuple(value * scale for value in observation.xyxy),
                              keypoints=tuple((x * scale, y * scale, c) for x, y, c in observation.keypoints))
        events.extend(engine.update((observation,), index / 10, shape))
    confirmed = confirmations(events)
    assert len(confirmed) == 1
    assert confirmed[0].timestamp == pytest.approx(1.4)