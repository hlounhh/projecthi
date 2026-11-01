from dataclasses import replace

import pytest

from ai.fall_engine import UNOBSERVED, FallConfig, FallEngine
from tests.test_curled_landing import compact_landing, confirmations, feed_landing, seed_fall
from tests.test_fall_engine import SHAPE, person, seated


def seated_baseline(engine=None):
    engine = engine or FallEngine()
    for index in range(7):
        engine.update((seated(y=160),), index / 10, SHAPE)
    return engine


@pytest.mark.parametrize('hidden_legs', [False, True])
def test_observed_fall_can_start_from_stable_seated_pose(hidden_legs):
    engine = seated_baseline()
    assert engine._tracks[1].baseline is not None
    events = list(engine.update((person(y=240, angle=80),), .7, SHAPE))
    landing = compact_landing() if hidden_legs else person(y=240, angle=80)
    events.extend(feed_landing(engine, observation=landing, end=30))
    assert len(confirmations(events)) == 1


def test_static_sitting_and_sit_to_stand_do_not_confirm():
    engine = seated_baseline()
    events = []
    for index in range(7, 40):
        events.extend(engine.update((seated(y=160),), index / 10, SHAPE))
    for index in range(40, 55):
        events.extend(engine.update((person(),), index / 10, SHAPE))
    assert not events
    assert engine._upright(engine._tracks[1].baseline)


def test_seated_reference_does_not_follow_gradual_drift():
    engine = seated_baseline()
    reference = engine._tracks[1].baseline
    for index in range(7, 15):
        engine.update((seated(y=160 + (index - 6)),), index / 10, SHAPE)
    assert engine._tracks[1].baseline == reference
    assert engine._tracks[1].baseline_seen_at == pytest.approx(1.4)


def test_reference_freshness_cannot_inflate_measured_descent_speed():
    engine = seated_baseline(FallEngine(FallConfig(motion_window=.4)))
    for index, y in ((7, 165.6), (8, 168.4), (9, 171.2), (10, 171.2)):
        engine.update((seated(y=y),), index / 10, SHAPE)
    events = []
    for index in range(11, 15):
        events.extend(engine.update((person(y=171.2 + (index - 10) * 8.4,
                                            angle=(index - 10) * 20),), index / 10, SHAPE))
    assert engine._tracks[1].occlusion.motion_landing is None
    assert not confirmations(events)


@pytest.mark.parametrize('kind', ['legs', 'torso', 'moving', 'too_short'])
def test_unreliable_or_unstable_seated_history_cannot_seed_motion(kind):
    engine = FallEngine()
    for index in range(7 if kind != 'too_short' else 3):
        observation = seated(y=160 + index * 10 if kind == 'moving' else 160)
        points = list(observation.keypoints)
        for joint in ((13, 14, 15, 16) if kind == 'legs' else (5,) if kind == 'torso' else ()):
            points[joint] = (*points[joint][:2], .1)
        engine.update((replace(observation, keypoints=tuple(points)),), index / 10, SHAPE)
    assert engine._tracks[1].baseline is None
    engine.update((person(y=240, angle=80),), .7 if kind != 'too_short' else .3, SHAPE)
    assert engine._tracks[1].occlusion.motion_landing is None


@pytest.mark.parametrize('origin', [person(), seated(y=160)])
def test_no_fall_motion_can_be_inferred_across_an_occlusion(origin):
    engine = FallEngine()
    for index in range(7):
        engine.update((origin,), index / 10, SHAPE)
    engine.update((), .7, SHAPE)
    events = feed_landing(engine, start=8, end=35)
    assert not confirmations(events)
    assert all(track.occlusion.motion_landing is None for track in engine._tracks.values())


@pytest.mark.parametrize('seated_origin', [False, True])
@pytest.mark.parametrize('hidden_legs', [False, True])
def test_short_occlusion_retains_observed_motion_but_restarts_verification(seated_origin, hidden_legs):
    engine = seated_baseline() if seated_origin else seed_fall()
    if seated_origin:
        engine.update((person(y=240, angle=80),), .7, SHAPE)
    landing = compact_landing() if hidden_legs else person(y=240, angle=80)
    feed_landing(engine, end=12, observation=landing)
    token = engine._tracks[1].occlusion.motion_landing
    event_id = engine.snapshot(1.1)[0].event_id
    engine.update((), 1.2, SHAPE)
    engine.update((), 1.3, SHAPE)
    assert engine.snapshot(1.3)[0].state == UNOBSERVED
    assert engine._tracks[1].occlusion.motion_landing == token
    assert engine._tracks[1].occlusion.observed_curled == 0
    for timestamp in (1.4, 1.5):
        assert not confirmations(engine.update((landing,), timestamp, SHAPE))
        assert engine._tracks[1].occlusion.observed_curled == 0
    assert not confirmations(feed_landing(engine, start=16, end=22, observation=landing))
    confirmed = confirmations(engine.update((landing,), 2.2, SHAPE))
    assert len(confirmed) == 1
    assert confirmed[0].event_id == event_id
    assert dict(confirmed[0].features)['landing_observed_seconds'] == pytest.approx(.6)


def test_repeated_occlusions_do_not_extend_the_original_motion_deadline():
    engine = seed_fall()
    original = engine._tracks[1].occlusion.motion_landing
    events = []
    for index in range(8, 45):
        observations = () if index % 5 == 0 else (compact_landing(),)
        events.extend(engine.update(observations, index / 10, SHAPE))
        token = engine._tracks[1].occlusion.motion_landing
        assert token is None or token == original
    assert engine._tracks[1].occlusion.motion_landing is None
    assert not confirmations(events)
    assert not confirmations(feed_landing(engine, start=45, end=60))


def test_suspicion_does_not_keep_grace_alive_while_torso_is_hidden():
    engine = seed_fall()
    feed_landing(engine, end=10)
    observation = compact_landing(one_side=True)
    for index in range(10, 19):
        engine.update((observation,), index / 10, SHAPE)
    assert engine._tracks[1].occlusion.motion_landing is None
    assert not confirmations(feed_landing(engine, start=19, end=40))


def test_box_overlap_alone_cannot_transfer_motion_to_displaced_torso():
    engine = seed_fall()
    feed_landing(engine, end=10)
    engine.update((), 1.0, SHAPE)
    observation = compact_landing()
    observation = replace(observation, keypoints=tuple((x + 400, y, confidence)
                                                      for x, y, confidence in observation.keypoints))
    engine.update((observation,), 1.1, SHAPE)
    assert engine._tracks[1].occlusion.motion_landing is None
    assert not confirmations(feed_landing(engine, start=12, end=35))


@pytest.mark.parametrize('kind', ['weak', 'stale', 'ambiguous', 'long_gap', 'upright', 'dormant'])
def test_invalid_identity_or_expired_observation_cannot_resume_confirmation(kind):
    engine = seed_fall()
    feed_landing(engine, end=10)
    start = 11
    if kind == 'weak':
        engine.update((compact_landing(confidence=.25),), 1.0, SHAPE)
    elif kind == 'stale':
        engine.mark_unobserved(1.0)
    elif kind == 'ambiguous':
        engine.update((compact_landing(), compact_landing()), 1.0, SHAPE)
    elif kind in ('long_gap', 'dormant'):
        engine.update((), 1.7 if kind == 'long_gap' else 2.0, SHAPE)
        start = 18 if kind == 'long_gap' else 21
    else:
        engine.update((person(),), 1.0, SHAPE)
    assert not confirmations(feed_landing(engine, start=start, end=42))


@pytest.mark.parametrize('kwargs', [
    {'occlusion_grace_seconds': .1}, {'baseline_max_position_change': 1},
    {'baseline_max_height_change': 1}, {'baseline_max_angle_change': 90},
    {'baseline_seated_max_aspect': .7},
])
def test_invalid_reference_and_grace_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FallConfig(**kwargs)