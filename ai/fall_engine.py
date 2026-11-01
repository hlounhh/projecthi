"""Conservative, geometric fall hypotheses, not medical diagnoses.

Call update once per inference result, with its source_received_at (not inference
completion time). Call mark_unobserved for a stale source/AI result. No clocks,
models, cameras, or third-party packages are used here. Geometry cannot establish
identity through crossings; ambiguous matches invalidate temporal evidence.
"""

from dataclasses import dataclass, field, fields, replace
import math
from typing import Iterable

from ai.occlusion_evidence import OcclusionEvidence
from ai.partial_landing import PartialLanding
from ai.point_filter import KeypointSmoother


NORMAL = "BINH THUONG"
SUSPECTED_FALL = "NGHI NGA"
CONFIRMED_FALL = "DA NGA"
UNOBSERVED = "MAT DAU"
Features = tuple[tuple[str, float | bool], ...]


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _probability(value):
    return _finite(value) and 0 <= value <= 1


@dataclass(frozen=True)
class PersonObservation:
    xyxy: tuple[float, float, float, float]
    confidence: float
    fallen_confidence: float = 0.0
    keypoints: tuple[tuple[float, float, float], ...] = ()
    posture_observed: bool = True

    def __post_init__(self):
        if type(self.posture_observed) is not bool:
            raise ValueError("posture_observed must be boolean.")
        box = tuple(self.xyxy)
        points = tuple(tuple(point) for point in self.keypoints)
        if (len(box) != 4 or not all(_finite(v) for v in box)
                or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("xyxy must be a finite, positive-area pixel box.")
        if not _probability(self.confidence) or not _probability(self.fallen_confidence):
            raise ValueError("Observation confidences must be in [0, 1].")
        if len(points) not in (0, 17) or any(
            len(p) != 3 or not all(_finite(v) for v in p) or not _probability(p[2])
            for p in points
        ):
            raise ValueError("keypoints must be empty or COCO17 (x, y, confidence).")
        object.__setattr__(self, "xyxy", box)
        object.__setattr__(self, "keypoints", points)


@dataclass(frozen=True)
class FallConfig:
    min_detection_confidence: float = 0.4
    min_tracking_confidence: float = 0.2
    min_keypoint_confidence: float = 0.5
    min_visible_keypoints: int = 8
    min_body_height: float = 32.0
    min_torso_fraction: float = 0.08
    fallen_label_threshold: float = 0.6
    fallen_label_reset_threshold: float = 0.3
    association_min_iou: float = 0.1
    association_max_distance: float = 0.9
    association_max_area_ratio: float = 4.0
    association_iou_weight: float = 0.65
    association_ambiguity_margin: float = 0.12
    max_track_gap: float = 1.0
    identity_retention_seconds: float = 3.0
    max_frame_gap: float = 0.3
    baseline_seconds: float = 0.4
    baseline_min_samples: int = 3
    baseline_max_angle: float = 25.0
    baseline_max_aspect: float = 0.8
    baseline_min_leg_extension: float = 0.35
    baseline_seated_max_aspect: float = 1.25
    baseline_max_position_change: float = 0.08
    baseline_max_angle_change: float = 10.0
    baseline_max_height_change: float = 0.15
    occlusion_grace_seconds: float = 0.7
    motion_window: float = 0.8
    min_drop: float = 0.28
    partial_min_drop: float = 0.18
    min_descent_speed: float = 0.65
    min_angle_change: float = 45.0
    min_angular_speed: float = 85.0
    post_min_angle: float = 55.0
    post_min_extension: float = 0.6
    alternate_min_angle: float = 35.0
    alternate_min_aspect: float = 1.25
    alternate_max_height_ratio: float = 0.65
    alternate_min_horizontal_extension: float = 0.6
    verify_seconds: float = 0.6
    verify_min_samples: int = 3
    suspicion_timeout: float = 2.5
    recovery_seconds: float = 0.6
    cooldown_seconds: float = 5.0
    down_seconds: float = 1.2
    down_min_aspect: float = 1.5
    keypoint_smoothing: bool = False
    smoothing_min_cutoff: float = 1.5
    smoothing_beta: float = 5.0
    smoothing_derivative_cutoff: float = 1.0

    def __post_init__(self):
        integers = {"min_visible_keypoints": (8, 17), "baseline_min_samples": (2, None),
                    "verify_min_samples": (2, None)}
        probabilities = {
            "min_detection_confidence", "min_tracking_confidence", "min_keypoint_confidence", "min_torso_fraction",
            "fallen_label_threshold", "fallen_label_reset_threshold", "association_min_iou",
            "association_iou_weight", "association_ambiguity_margin",
        }
        angles = {"baseline_max_angle", "baseline_max_angle_change", "min_angle_change", "post_min_angle", "alternate_min_angle"}
        for item in fields(self):
            name, value = item.name, getattr(self, item.name)
            if name == "keypoint_smoothing":
                if type(value) is not bool:
                    raise ValueError("keypoint_smoothing must be boolean.")
                continue
            if name in integers:
                low, high = integers[name]
                if type(value) is not int or value < low or (high is not None and value > high):
                    raise ValueError(f"Invalid {name}.")
            elif not _finite(value) or value <= 0:
                # A zero reset threshold / IoU gate / ambiguity margin is useful.
                if name not in probabilities or not _probability(value):
                    raise ValueError(f"{name} must be finite and positive.")
            if name in probabilities and not _probability(value):
                raise ValueError(f"{name} must be in [0, 1].")
            if name in angles and not 0 < value < 90:
                raise ValueError(f"{name} must be in (0, 90) degrees.")
        if not (self.min_tracking_confidence <= self.min_detection_confidence
                and self.fallen_label_reset_threshold < self.fallen_label_threshold
                and self.max_frame_gap <= self.max_track_gap
                and self.max_frame_gap <= self.motion_window
                and self.max_frame_gap <= self.occlusion_grace_seconds
                and self.baseline_max_aspect <= self.baseline_seated_max_aspect
                and self.baseline_max_position_change < 1
                and self.baseline_max_height_change < 1
                and self.partial_min_drop <= self.min_drop
                and self.verify_seconds < self.suspicion_timeout
                and self.baseline_max_angle < self.post_min_angle
                and self.alternate_min_angle <= self.post_min_angle
                and self.association_max_area_ratio >= 1):
            raise ValueError("Inconsistent fall thresholds or time budgets.")


@dataclass(frozen=True)
class FallEvent:
    track_id: int
    event_id: str | None
    state: str
    timestamp: float
    reason: str
    features: Features = ()


@dataclass(frozen=True)
class TrackView:
    track_id: int
    xyxy: tuple[float, float, float, float]
    state: str
    observation_quality: bool
    event_id: str | None = None
    features: Features = ()
    visible: bool = False
    observation_age: float = 0.0
    fall_detected: bool = False


@dataclass(frozen=True)
class _Sample:
    timestamp: float
    height: float
    torso_y: float
    angle: float
    aspect: float
    extension: float
    horizontal_extension: float
    leg_extension: float
    lower_body: bool
    raised_legs: bool
    hip_y: float
    torso_x: float


@dataclass
class _Track:
    track_id: int
    xyxy: tuple[float, float, float, float]
    last_seen: float
    state: str = NORMAL
    phase: str = NORMAL  # Preserved while the public state is UNOBSERVED.
    quality: bool = False
    visible: bool = True
    keypoints: tuple[tuple[float, float, float], ...] = ()
    keypoint_filter: KeypointSmoother | None = field(default=None, repr=False)
    event_id: str | None = None
    suspected_at: float | None = None
    label_latched: bool = False
    baseline: _Sample | None = None
    baseline_seen_at: float | None = None
    baseline_candidate: _Sample | None = None
    baseline_samples: int = 0
    motion_samples: list[_Sample] = field(default_factory=list)
    upright_since: float | None = None
    upright_samples: int = 0
    motion_height: float | None = None
    descent_base: _Sample | None = None
    down_since: float | None = None
    down_samples: int = 0
    down_latched: bool = False
    verify_since: float | None = None
    verify_samples: int = 0
    recovered_since: float | None = None
    cooldown_until: float | None = None
    features: dict[str, float | bool] = field(default_factory=dict)
    occlusion: OcclusionEvidence = field(default_factory=OcclusionEvidence)
    partial: PartialLanding = field(default_factory=PartialLanding)


def _iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection), max(area_a, area_b) / min(area_a, area_b)


class FallEngine:
    """Single-consumer tracker/FSM. All durations are seconds in input time.

    Events describe transitions only (no per-frame standing/sitting logs). A high
    fallen label emits SUSPECTED_FALL unless reliable folded-leg/vertical-torso
    geometry contradicts it. Without pose, UNOBSERVED also reports unavailable verification.
    Missing torso/ambiguous identity clears continuity. One visible leg can supply
    lower-body geometry. A separate curled-landing path permits hidden legs only
    after measured rapid fall motion, deep torso descent and sustained compact
    bilateral-torso observations. A persistent horizontal pose alone raises an
    inspection hypothesis without proving a fall occurred.
    A separate bounded occlusion path requires observed shoulder descent and repeated
    reliable down poses; missing time never contributes to its verification budget.
    Visibility is separate from verification quality. Weak detections cannot seed
    motion. An existing partial candidate can use them only with many confident
    landmarks and a continuous identity. Dormant IDs require unique strong box
    overlap to return; dormant identity recovery never restores temporal evidence.
    Stable standing or seated poses can supply a frozen pre-motion reference.
    Short occlusions retain only previously measured fall evidence; verification
    restarts after a conservative association and fresh torso observations.
    A partial-landing path can instead verify visible extended legs or repeated
    low poses corroborated by image labels after measured descent. It compares
    the same landmarks across frames and does not require floor contact.
    Missing data cannot dismiss an event. Seated-origin accuracy is not calibrated.

    CONFIRMED_FALL is an algorithmic hypothesis. It stays latched while the
    person remains down and cannot repeatedly confirm the same fall; it clears
    only after the cooldown has elapsed and a sustained upright pose is observed.
    """

    def __init__(self, config: FallConfig | None = None):
        self.config = config if config is not None else FallConfig()
        if not isinstance(self.config, FallConfig):
            raise TypeError("config must be FallConfig.")
        self._tracks: dict[int, _Track] = {}
        self._dormant: dict[int, _Track] = {}
        self._timestamp: float | None = None  # Last consumed source timestamp.
        self._unobserved_timestamp: float | None = None  # Separate watchdog watermark.
        self._shape: tuple[int, ...] | None = None
        # Do not reuse IDs after clock/shape resets.
        self._next_track = 1
        self._next_event = 1

    @staticmethod
    def _time(timestamp):
        if not _finite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and nonnegative.")

    def _advance(self, timestamp):
        self._time(timestamp)
        if self._timestamp is not None:
            if timestamp == self._timestamp:
                return False
            if timestamp < self._timestamp:
                self._tracks.clear()
                self._dormant.clear()
                self._shape = None
                self._unobserved_timestamp = None
        self._timestamp = timestamp
        return True

    def update(self, observations: Iterable[PersonObservation], timestamp: float,
               frame_shape: tuple[int, ...]) -> tuple[FallEvent, ...]:
        """Consume one processed frame (height, width[, channels]). Duplicates are no-ops."""
        shape = tuple(frame_shape)
        if len(shape) not in (2, 3) or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError("frame_shape must contain positive integer dimensions.")
        observations = tuple(observations)
        if any(not isinstance(o, PersonObservation) for o in observations):
            raise TypeError("observations must contain PersonObservation values.")
        if not self._advance(timestamp):
            return ()
        if self._shape is not None and self._shape != shape:
            self._tracks.clear()
            self._dormant.clear()
        self._shape = shape
        c = self.config
        height, width = shape[:2]
        observations = tuple(o for o in observations if o.confidence >= c.min_tracking_confidence
                             and o.xyxy[2] > 0 and o.xyxy[3] > 0
                             and o.xyxy[0] < width and o.xyxy[1] < height)
        events: list[FallEvent] = []
        if self._unobserved_timestamp is not None and timestamp <= self._unobserved_timestamp:
            # Consume delayed work once, but never use data captured before the
            # watchdog invalidation to rebuild baseline or verification.
            for track in self._tracks.values():
                self._missing(track, timestamp, "result_predates_stale_mark", events)
            return tuple(events)
        for track in tuple(self._tracks.values()):
            if timestamp - track.last_seen > c.max_frame_gap:
                self._missing(track, timestamp, "observation_gap", events)
            if timestamp - track.last_seen > c.max_track_gap:
                track.occlusion.clear()
                self._dormant[track.track_id] = self._tracks.pop(track.track_id)
        self._dormant = {i: t for i, t in self._dormant.items()
                         if timestamp - t.last_seen <= c.identity_retention_seconds}
        matches, eligible, ambiguous = self._associate(observations)
        matched_tracks = set(matches.values())
        for track in self._tracks.values():
            if track.track_id not in matched_tracks:
                reason = "ambiguous_association" if track.track_id in ambiguous else "not_matched"
                self._missing(track, timestamp, reason, events)
        unmatched = [i for i, o in enumerate(observations)
                     if i not in matches and i not in eligible
                     and o.confidence >= c.min_detection_confidence]
        recovered = self._recover_identities(observations, unmatched)
        for index, observation in enumerate(observations):
            track_id = matches.get(index)
            if track_id is None:
                if index in eligible or observation.confidence < c.min_detection_confidence:
                    continue
                track_id = recovered.get(index)
                if track_id is not None:
                    self._tracks[track_id] = self._dormant.pop(track_id)
                else:
                    track_id = self._next_track
                    self._next_track += 1
                    self._tracks[track_id] = _Track(track_id, observation.xyxy, timestamp)
            self._observe(self._tracks[track_id], observation, timestamp, events)
        return tuple(events)

    def mark_unobserved(self, timestamp: float, reason: str = "source_stale") -> tuple[FallEvent, ...]:
        """Invalidate continuity using watchdog time, separate from source time.

        A result already in flight may subsequently have an older source timestamp.
        It must not look like a source-clock rollback or inherit pre-stale motion.
        Only update advances the consumed source watermark. Duplicate/backward
        watchdog marks are no-ops; an actual backward update still resets tracks.
        """
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a nonempty string.")
        self._time(timestamp)
        if ((self._timestamp is not None and timestamp <= self._timestamp)
                or (self._unobserved_timestamp is not None
                    and timestamp <= self._unobserved_timestamp)):
            return ()
        self._unobserved_timestamp = timestamp
        events: list[FallEvent] = []
        for track in tuple(self._tracks.values()):
            track.occlusion.clear()
            track.partial.clear()
            self._missing(track, timestamp, reason, events)
            if timestamp - track.last_seen > self.config.max_track_gap:
                self._dormant[track.track_id] = self._tracks.pop(track.track_id)
        self._dormant = {i: t for i, t in self._dormant.items()
                         if timestamp - t.last_seen <= self.config.identity_retention_seconds}
        return tuple(events)

    def snapshot(self, timestamp: float) -> tuple[TrackView, ...]:
        """Read-only view; age can hide quality, but cannot advance the FSM or timers.

        Use mark_unobserved to commit staleness and emit its transition. A snapshot
        alone never manufactures verification time or resets input time.
        """
        self._time(timestamp)
        return tuple(TrackView(
            t.track_id, t.xyxy,
            t.state if timestamp - t.last_seen <= self.config.max_frame_gap else UNOBSERVED,
            t.quality and timestamp - t.last_seen <= self.config.max_frame_gap,
            t.event_id, tuple(sorted(t.features.items())),
            t.visible and timestamp - t.last_seen <= self.config.max_frame_gap,
            max(0.0, timestamp - t.last_seen),
            t.phase == CONFIRMED_FALL,
        ) for t in self._tracks.values())

    def _recover_identities(self, observations, indices):
        candidates = []
        for index in indices:
            box = observations[index].xyxy
            for track in self._dormant.values():
                overlap, ratio = _iou(track.xyxy, box)
                if overlap >= 0.5 and ratio <= 2:
                    candidates.append((index, track.track_id))
        return {i: t for i, t in candidates
                if sum(j == i for j, _ in candidates) == 1
                and sum(u == t for _, u in candidates) == 1}

    def _anchor_distance(self, track, observation):
        points = observation.keypoints
        if not points or not track.keypoints:
            return None
        pairs = [(a, b) for i, (a, b) in enumerate(zip(track.keypoints, points))
                 if i <= 12 and min(a[2], b[2]) >= self.config.min_keypoint_confidence]
        if len(pairs) < 3:
            return None
        distances = sorted(math.dist(a[:2], b[:2]) for a, b in pairs)
        scale = max(track.xyxy[3] - track.xyxy[1], observation.xyxy[3] - observation.xyxy[1])
        return distances[len(distances) // 2] / scale

    def _associate(self, observations):
        c = self.config
        rows: dict[int, list[tuple[float, int]]] = {}
        columns: dict[int, list[tuple[float, int]]] = {}
        for track in self._tracks.values():
            a = track.xyxy
            for index, observation in enumerate(observations):
                b = observation.xyxy
                overlap, area_ratio = _iou(a, b)
                distance = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2,
                                      (a[1] + a[3] - b[1] - b[3]) / 2) / max(a[3] - a[1], b[3] - b[1])
                anchor = self._anchor_distance(track, observation)
                partial_match = anchor is not None and anchor <= 0.25 and overlap >= c.association_min_iou
                if observation.confidence < c.min_detection_confidence and overlap < 0.3 and not partial_match:
                    continue
                if not partial_match and (area_ratio > c.association_max_area_ratio or (
                    overlap < c.association_min_iou and distance > c.association_max_distance
                )):
                    continue
                proximity = max(0.0, 1 - distance / c.association_max_distance)
                score = c.association_iou_weight * overlap + (1 - c.association_iou_weight) * proximity
                if partial_match:
                    score = max(score, 0.5 + 0.5 * (1 - anchor / 0.25))
                rows.setdefault(track.track_id, []).append((score, index))
                columns.setdefault(index, []).append((score, track.track_id))
        for values in (*rows.values(), *columns.values()):
            values.sort(reverse=True)
        ambiguous_tracks = {key for key, values in rows.items() if len(values) > 1
                            and values[0][0] - values[1][0] <= c.association_ambiguity_margin}
        ambiguous_observations = {key for key, values in columns.items() if len(values) > 1
                                  and values[0][0] - values[1][0] <= c.association_ambiguity_margin}
        # Quarantine every competitor of an ambiguous identity, not just its best pair.
        blocked_observations = ambiguous_observations | {
            index for track_id in ambiguous_tracks for _, index in rows[track_id]
        }
        ambiguous_tracks.update(track_id for index in ambiguous_observations
                                for _, track_id in columns[index])
        pairs = sorted(((score, track_id, index) for track_id, values in rows.items()
                        for score, index in values), reverse=True)
        matches, used = {}, set()
        for _, track_id, index in pairs:
            if (track_id in ambiguous_tracks or index in blocked_observations
                    or track_id in used or index in matches):
                continue
            matches[index] = track_id
            used.add(track_id)
        return matches, set(columns), ambiguous_tracks

    def _emit(self, track, timestamp, reason, events, state=None):
        events.append(FallEvent(track.track_id, track.event_id, state or track.state,
                                timestamp, reason, tuple(sorted(track.features.items()))))

    def _clear_continuity(self, track):
        track.keypoint_filter = None
        track.baseline = None
        track.baseline_seen_at = None
        track.baseline_candidate = None
        track.baseline_samples = 0
        track.motion_samples.clear()
        track.upright_since = None
        track.upright_samples = 0
        track.motion_height = None
        track.descent_base = None
        track.verify_since = None
        track.verify_samples = 0
        track.recovered_since = None
        track.down_since = None
        track.down_samples = 0

    def _dismiss(self, track, timestamp, reason, events):
        track.occlusion.clear()
        track.partial.clear()
        track.phase = NORMAL
        if track.state != UNOBSERVED:
            track.state = NORMAL
        self._emit(track, timestamp, reason, events, NORMAL)
        track.event_id = None
        track.suspected_at = None
        track.motion_height = None
        track.descent_base = None
        track.verify_since = None
        track.verify_samples = 0

    def _timeout(self, track, timestamp, events):
        if (track.phase == SUSPECTED_FALL and track.quality
                and track.suspected_at is not None
                and timestamp - track.suspected_at >= self.config.suspicion_timeout):
            self._dismiss(track, timestamp, "suspicion_timeout", events)

    def _missing(self, track, timestamp, reason, events):
        self._clear_continuity(track)
        if reason in ("not_matched", "observation_gap", "weak_detection"):
            track.partial.interrupt(timestamp, self.config)
        elif reason not in ("pose_unreliable", "lower_body_unreliable"):
            track.partial.clear()
        if reason not in ("not_matched", "pose_unreliable", "lower_body_unreliable", "observation_gap"):
            track.occlusion.clear()
        else:
            track.occlusion.interrupt(timestamp, self.config)
        track.quality = False
        track.visible = reason in ("pose_unreliable", "lower_body_unreliable", "weak_detection")
        track.features["pose_reliable"] = False
        if track.state != UNOBSERVED:
            track.state = UNOBSERVED
            self._emit(track, timestamp, reason, events)

    def _sample(self, observation, timestamp):
        c, points = self.config, observation.keypoints
        box = observation.xyxy
        height = box[3] - box[1]
        if height < c.min_body_height or not points:
            return None
        torso_sides = [(s, h) for s, h in ((5, 11), (6, 12))
                       if min(points[s][2], points[h][2]) >= c.min_keypoint_confidence]
        if not torso_sides:
            return None
        visible_legs = [ankle for knee, ankle in ((13, 15), (14, 16))
                        if points[knee][2] >= c.min_keypoint_confidence
                        and points[ankle][2] >= c.min_keypoint_confidence]
        lower_body = (len(torso_sides) == 2 and bool(visible_legs)
                      and sum(p[2] >= c.min_keypoint_confidence for p in points) >= c.min_visible_keypoints)
        shoulder = tuple(sum(points[s][i] for s, _ in torso_sides) / len(torso_sides) for i in (0, 1))
        hip = tuple(sum(points[h][i] for _, h in torso_sides) / len(torso_sides) for i in (0, 1))
        ankle = tuple(sum(points[j][i] for j in visible_legs) / len(visible_legs)
                      for i in (0, 1)) if visible_legs else hip
        torso = math.hypot(hip[0] - shoulder[0], hip[1] - shoulder[1])
        if torso < c.min_torso_fraction * height:
            return None
        angle = math.degrees(math.atan2(abs(hip[0] - shoulder[0]), abs(hip[1] - shoulder[1])))
        return _Sample(timestamp, height, (hip[1] + shoulder[1]) / 2, angle,
                       (box[2] - box[0]) / height,
                       math.dist(shoulder, ankle), abs(shoulder[0] - ankle[0]),
                       (ankle[1] - hip[1]) / height, lower_body,
                       lower_body and all(points[i][2] >= c.min_keypoint_confidence for i in (13, 14, 15, 16))
                       and max(points[i][1] for i in (13, 14, 15, 16)) < hip[1], hip[1],
                       (hip[0] + shoulder[0]) / 2)

    def _upright(self, sample):
        c = self.config
        return (sample.lower_body and sample.angle <= c.baseline_max_angle and sample.aspect <= c.baseline_max_aspect
                and sample.leg_extension >= c.baseline_min_leg_extension)

    def _reference_pose(self, sample):
        c = self.config
        return (sample.lower_body and sample.angle <= c.baseline_max_angle
                and sample.aspect <= c.baseline_seated_max_aspect and not sample.raised_legs)

    def _stable_reference(self, sample, reference):
        c = self.config
        return (self._reference_pose(sample)
                and self._upright(sample) == self._upright(reference)
                and math.hypot(sample.torso_x - reference.torso_x,
                               sample.torso_y - reference.torso_y) / reference.height <= c.baseline_max_position_change
                and abs(sample.hip_y - reference.hip_y) / reference.height <= c.baseline_max_position_change
                and abs(sample.angle - reference.angle) <= c.baseline_max_angle_change
                and abs(sample.height / reference.height - 1) <= c.baseline_max_height_change)

    def _update_baseline(self, track, sample):
        c = self.config
        if self._upright(sample):
            track.baseline_candidate = None
            track.baseline_samples = 0
            if track.upright_since is None:
                track.upright_since = sample.timestamp
            track.upright_samples += 1
            if ((track.baseline is None or not self._upright(track.baseline))
                    and sample.timestamp - track.upright_since + 1e-9 >= c.baseline_seconds
                    and track.upright_samples >= c.baseline_min_samples):
                track.baseline = sample
                track.baseline_seen_at = None
            return
        track.upright_since = None
        track.upright_samples = 0
        if track.baseline is not None:
            if self._stable_reference(sample, track.baseline):
                track.baseline_seen_at = sample.timestamp
                return
            seen_at = track.baseline_seen_at if track.baseline_seen_at is not None else track.baseline.timestamp
            if sample.timestamp - seen_at <= c.motion_window:
                track.baseline_candidate = None
                track.baseline_samples = 0
                return
            track.baseline = None
            track.baseline_seen_at = None
        if (track.phase != NORMAL or track.descent_base is not None
                or not self._reference_pose(sample)):
            track.baseline_candidate = None
            track.baseline_samples = 0
            return
        reference = track.baseline_candidate
        if reference is None or not self._stable_reference(sample, reference):
            track.baseline_candidate = sample
            track.baseline_samples = 1
            return
        track.baseline_samples += 1
        if (sample.timestamp - reference.timestamp + 1e-9 >= c.baseline_seconds
                and track.baseline_samples >= c.baseline_min_samples):
            track.baseline = reference
            track.baseline_seen_at = sample.timestamp
            track.baseline_candidate = None
            track.baseline_samples = 0

    def _post(self, sample, height):
        c = self.config
        extended = sample.extension / height >= c.post_min_extension
        tilted = sample.angle >= c.post_min_angle
        alternate = (sample.angle >= c.alternate_min_angle and sample.aspect >= c.alternate_min_aspect
                     and sample.height / height <= c.alternate_max_height_ratio
                     and sample.horizontal_extension / height >= c.alternate_min_horizontal_extension)
        compressed = (sample.aspect >= c.down_min_aspect
                      and sample.height / height <= c.alternate_max_height_ratio)
        inverted = sample.raised_legs and sample.height / height <= c.alternate_max_height_ratio
        return sample.lower_body and (inverted or (tilted and compressed) or extended and (tilted or alternate)), alternate

    def _motion(self, track, sample):
        c = self.config
        descent = track.descent_base
        if descent is not None:
            if sample.timestamp - descent.timestamp > c.suspicion_timeout:
                track.descent_base = None
            elif (sample.torso_y - descent.torso_y) / descent.height >= c.min_drop:
                if self._post(sample, descent.height)[0]:
                    track.features["sequential_descent_landing"] = True
                    track.motion_height = descent.height
                    return True
        track.motion_samples.append(sample)
        track.motion_samples[:] = [p for p in track.motion_samples
                                   if sample.timestamp - p.timestamp <= c.motion_window + 1e-9]
        base = track.baseline
        if base is None:
            return False
        standing = self._upright(base)
        if standing:
            candidates = [p for p in track.motion_samples if self._upright(p)]
        else:
            candidates = [p for p in track.motion_samples if self._stable_reference(p, base)]
        if candidates:
            if standing:
                base = track.baseline = candidates[0]
            else:
                # Rates use coordinates and time from the same actual pose.
                base = replace(candidates[0], height=base.height)
        elif not standing:
            return False
        dt = sample.timestamp - base.timestamp
        if dt <= 0 or dt > c.motion_window + 1e-9:
            return False
        drop = (sample.torso_y - base.torso_y) / base.height
        turn = sample.angle - base.angle
        post, alternate = self._post(sample, base.height)
        # Descent and rotation need not peak in the same frame.
        recent_speeds = [(b.torso_y - a.torso_y) / base.height / (b.timestamp - a.timestamp)
                         for a, b in zip(track.motion_samples, track.motion_samples[1:])
                         if a.timestamp >= base.timestamp
                         and 0.05 <= b.timestamp - a.timestamp <= c.max_frame_gap]
        speed = max([drop / dt, *recent_speeds])
        track.features.update(drop=drop, descent_speed=speed,
                              angle_change=turn, angular_speed=turn / dt,
                              body_extension=sample.extension / base.height,
                              post_fall_geometry=post)
        if (track.descent_base is None and sample.lower_body
                and drop >= c.min_drop and speed >= c.min_descent_speed):
            track.descent_base = base
        rapid_turn = turn >= c.min_angle_change and turn / dt >= c.min_angular_speed
        # A curled landing can precede the extended pose used for confirmation.
        # Strong descent + rotation is enough to ask for a human check, not confirm.
        return drop >= c.min_drop and speed >= c.min_descent_speed and (rapid_turn or (post and alternate))

    def _suspect(self, track, timestamp, reason, events):
        track.event_id = f"fall-{self._next_event}"
        self._next_event += 1
        track.suspected_at = timestamp
        # Recovery must be observed AFTER the hypothesis, not borrowed from an
        # upright period that preceded a contradictory classifier label.
        track.recovered_since = None
        track.state = track.phase = SUSPECTED_FALL
        self._emit(track, timestamp, reason, events)

    def _observe(self, track, observation, timestamp, events):
        c = self.config
        previous_state = track.state
        track.xyxy, track.last_seen = observation.xyxy, timestamp
        track.keypoints = observation.keypoints
        track.visible = True
        if c.keypoint_smoothing and observation.keypoints:
            if track.keypoint_filter is None:
                track.keypoint_filter = KeypointSmoother(
                    scale=max(observation.xyxy[2] - observation.xyxy[0],
                              observation.xyxy[3] - observation.xyxy[1]),
                    min_cutoff=c.smoothing_min_cutoff, beta=c.smoothing_beta,
                    derivative_cutoff=c.smoothing_derivative_cutoff)
            observation = replace(observation, keypoints=track.keypoint_filter.filter(
                observation.keypoints, timestamp, min_confidence=c.min_keypoint_confidence))
        else:
            track.keypoint_filter = None
        sample = self._sample(observation, timestamp)
        partial_fall = track.partial.observe(observation, timestamp, c)
        if track.phase == CONFIRMED_FALL:
            track.partial.clear()
        elif partial_fall and track.motion_height is None and track.occlusion.motion_landing is None:
            track.quality = sample is not None and sample.lower_body
            track.features = dict(track.partial.features)
            track.features.update(pose_reliable=track.quality,
                                  lower_body_reliable=track.quality,
                                  rapid_fall_motion_observed=True)
            if track.phase == NORMAL:
                self._suspect(track, timestamp, "visible_descent_before_partial_landing", events)
            track.state = track.phase = CONFIRMED_FALL
            track.cooldown_until = timestamp + c.cooldown_seconds
            self._emit(track, timestamp, "partial_landing_confirmed", events)
            track.partial.clear()
            track.occlusion.clear()
            self._clear_continuity(track)
            return
        if observation.confidence < c.min_detection_confidence:
            self._missing(track, timestamp, "weak_detection", events)
            return
        track.quality = sample is not None
        track.features = {"pose_reliable": track.quality,
                          "fallen_confidence": observation.fallen_confidence}
        label_rise = observation.fallen_confidence >= c.fallen_label_threshold and not track.label_latched
        if observation.posture_observed and observation.fallen_confidence <= c.fallen_label_reset_threshold:
            track.label_latched = False
        elif observation.fallen_confidence >= c.fallen_label_threshold:
            track.label_latched = True
        seated_geometry = (sample is not None and sample.lower_body
                           and sample.angle <= c.baseline_max_angle
                           and sample.aspect <= c.baseline_max_aspect
                           and not self._upright(sample))
        if track.phase == NORMAL and label_rise and not seated_geometry:
            self._suspect(track, timestamp, "fallen_label", events)
        landing = track.occlusion.motion_landing
        post_fall = sample is not None and landing is not None and self._post(sample, landing[1])[0]
        occluded_fall = track.occlusion.observe(
            observation, timestamp, sample, sample is not None and self._upright(sample), c,
            post_fall=post_fall)
        if occluded_fall and track.phase != CONFIRMED_FALL:
            if track.phase == NORMAL:
                self._suspect(track, timestamp, "observed_descent_before_occlusion", events)
            track.state = track.phase = CONFIRMED_FALL
            track.cooldown_until = timestamp + c.cooldown_seconds
            reason = track.occlusion.confirmation_reason or "occlusion_fall_confirmed"
            if reason in ("curled_landing_confirmed", "observed_landing_confirmed"):
                track.features["landing_observed_seconds"] = track.occlusion.observed_curled
                if reason == "curled_landing_confirmed":
                    track.features["curled_landing_observed_seconds"] = track.occlusion.observed_curled
                track.features["rapid_fall_motion_observed"] = True
                track.features["lower_body_reliable"] = sample.lower_body
                track.features["torso_angle"] = sample.angle
                track.features["aspect"] = sample.aspect
            else:
                track.features["occlusion_observed_seconds"] = track.occlusion.observed_down
            self._emit(track, timestamp, reason, events)
            track.occlusion.clear()
            self._clear_continuity(track)
            return
        if sample is None:
            self._missing(track, timestamp, "pose_unreliable", events)
            return
        # Resume the hypothesis, never its pre-gap baseline/verification timer.
        track.state = track.phase
        track.features.update(torso_angle=sample.angle, aspect=sample.aspect,
                              lower_body_reliable=sample.lower_body)
        if not sample.lower_body:
            track.quality = False
            track.features["pose_reliable"] = False
            track.baseline = None
            track.baseline_seen_at = None
            track.baseline_candidate = None
            track.baseline_samples = 0
            track.motion_samples.clear()
            track.motion_height = None
            track.descent_base = None
            track.upright_since = None
            track.upright_samples = 0
        upright = self._upright(sample)
        down = sample.angle >= c.post_min_angle and sample.aspect >= c.down_min_aspect
        if down:
            if track.down_since is None:
                track.down_since = timestamp
            track.down_samples += 1
            if (timestamp - track.down_since >= c.down_seconds
                    and track.down_samples >= c.verify_min_samples):
                if track.phase == NORMAL and not track.down_latched:
                    self._suspect(track, timestamp, "persistent_low_pose", events)
                track.down_latched = True
        else:
            track.down_since = None
            track.down_samples = 0
        if upright:
            track.down_latched = False
            if track.recovered_since is None:
                track.recovered_since = timestamp
            if timestamp - track.recovered_since >= c.recovery_seconds:
                track.descent_base = None
        else:
            track.recovered_since = None
        if track.phase == CONFIRMED_FALL:
            if (timestamp >= track.cooldown_until and track.recovered_since is not None
                    and timestamp - track.recovered_since >= c.recovery_seconds):
                self._dismiss(track, timestamp, "recovered_after_fall", events)
                self._clear_continuity(track)
            return
        motion = self._motion(track, sample)
        if motion:
            base = track.descent_base or track.baseline
            if base is not None:
                track.occlusion.arm_landing(base, timestamp)
            if track.phase == NORMAL:
                self._suspect(track, timestamp, "rapid_descent_and_geometry", events)
            if track.motion_height is None:
                track.motion_height = track.baseline.height
        if track.phase == SUSPECTED_FALL:
            if not sample.lower_body:
                track.baseline = None
                track.verify_since = None
                track.verify_samples = 0
                track.motion_height = None
                track.descent_base = None
                track.state = UNOBSERVED
                track.quality = False
                track.features["pose_reliable"] = False
                if previous_state != UNOBSERVED:
                    self._emit(track, timestamp, "lower_body_unreliable", events)
                track.features["verification_unavailable"] = True
                return
            post = track.motion_height is not None and self._post(sample, track.motion_height)[0]
            if post:
                if track.verify_since is None:
                    track.verify_since = timestamp
                track.verify_samples += 1
                if (timestamp - track.verify_since >= c.verify_seconds
                        and track.verify_samples >= c.verify_min_samples):
                    track.state = track.phase = CONFIRMED_FALL
                    track.cooldown_until = timestamp + c.cooldown_seconds
                    track.features["verified_seconds"] = timestamp - track.verify_since
                    self._emit(track, timestamp, "algorithmic_fall_confirmed", events)
                    self._clear_continuity(track)
                    return
            else:
                track.verify_since = None
                track.verify_samples = 0
                # Keep a reliably observed rapid-motion hypothesis through a
                # curled landing. Upright recovery or missing data invalidates it.
                if upright:
                    track.motion_height = None
            if (upright and track.recovered_since is not None
                    and timestamp - track.recovered_since >= c.recovery_seconds):
                self._dismiss(track, timestamp, "upright_recovery", events)
        if not down:
            self._timeout(track, timestamp, events)
        self._update_baseline(track, sample)
