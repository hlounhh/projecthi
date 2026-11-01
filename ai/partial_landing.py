"""Track visible anatomical landmarks through a descent and a hidden landing."""

from dataclasses import dataclass, field
import math
import statistics


@dataclass
class PartialLanding:
    history: list = field(default_factory=list)
    candidate: tuple | None = None
    last_seen: float | None = None
    last_down: float | None = None
    observed_down: float = 0.
    down_samples: int = 0
    label_times: list = field(default_factory=list)
    recovered_since: float | None = None
    identity_points: tuple = ()
    identity_box: tuple | None = None
    reacquire_samples: int = 0
    features: dict = field(default_factory=dict)

    def clear(self):
        self.history.clear()
        self.candidate = None
        self.last_seen = self.last_down = None
        self.observed_down = 0.
        self.down_samples = 0
        self.label_times.clear()
        self.recovered_since = None
        self.identity_points = ()
        self.identity_box = None
        self.reacquire_samples = 0
        self.features.clear()

    def interrupt(self, timestamp, config):
        self._expire(timestamp, config)
        self.history.clear()
        self.recovered_since = None
        if self.last_seen is not None and timestamp - self.last_seen > config.max_frame_gap + 1e-9:
            self.last_down = None
            self.observed_down = 0.
            self.down_samples = 0
            self.label_times.clear()
            if self.candidate is not None:
                self.reacquire_samples = 2
        if self.last_seen is None or timestamp - self.last_seen > config.occlusion_grace_seconds + 1e-9:
            self.clear()

    def _expire(self, timestamp, config):
        if (self.candidate is not None
                and timestamp - self.candidate[0] > config.suspicion_timeout + config.verify_seconds):
            self.candidate = None
            self.label_times.clear()
            self.reacquire_samples = 0
            self.last_down = None
            self.observed_down = 0.
            self.down_samples = 0

    @staticmethod
    def _upper_pairs(first, second, confidence):
        upper = [(first[i], second[i]) for i in (0, 5, 6)
                 if min(first[i][2], second[i][2]) >= confidence]
        if len(upper) >= 2:
            return upper
        torso = upper + [(first[i], second[i]) for i in (11, 12)
                         if min(first[i][2], second[i][2]) >= confidence]
        return torso if len(torso) >= 3 else []

    @staticmethod
    def _vertical(points, confidence, max_angle):
        sides = [(points[s], points[h]) for s, h in ((5, 11), (6, 12))
                 if min(points[s][2], points[h][2]) >= confidence]
        return bool(sides) and all(h[1] > s[1]
                                   and math.degrees(math.atan2(abs(h[0] - s[0]), h[1] - s[1])) <= max_angle
                                   for s, h in sides)

    def observe(self, observation, timestamp, config):
        self.features.clear()
        self._expire(timestamp, config)
        points = observation.keypoints
        body = (0, 5, 6, 11, 12, 13, 14, 15, 16)
        confidence = config.min_keypoint_confidence
        tracked_pose = (self.candidate is not None
                        and observation.confidence >= config.min_tracking_confidence
                        and sum(p[2] >= max(.75, confidence) for p in points) >= config.min_visible_keypoints)
        if ((observation.confidence < config.min_detection_confidence and not tracked_pose) or len(points) != 17
                or sum(points[i][2] >= confidence for i in body) < 3):
            self.interrupt(timestamp, config)
            return False
        if self.last_seen is not None and timestamp - self.last_seen > config.occlusion_grace_seconds + 1e-9:
            self.clear()
        if self.reacquire_samples or (self.candidate is not None and self.identity_points):
            common = [math.dist(self.identity_points[i][:2], points[i][:2]) for i in body
                      if min(self.identity_points[i][2], points[i][2]) >= confidence] if self.identity_points else []
            scale = max(self.identity_box[3] - self.identity_box[1], observation.xyxy[3] - observation.xyxy[1]) if self.identity_box else 1.
            if self.candidate is not None:
                scale = max(scale, self.candidate[1][1])
            if len(common) < 3 or max(common) / scale > config.association_max_distance:
                self.clear()
                return False
            elif self.reacquire_samples:
                self.identity_points, self.identity_box = points, observation.xyxy
                self.last_seen = timestamp
                if observation.posture_observed and observation.fallen_confidence >= config.fallen_label_threshold:
                    self.label_times.append((timestamp, observation.fallen_confidence))
                self.reacquire_samples -= 1
                return False
        gap = None if self.last_seen is None else timestamp - self.last_seen
        self.last_seen = timestamp
        self.identity_points, self.identity_box = points, observation.xyxy
        if gap is not None and gap > config.max_frame_gap + 1e-9:
            self.history.clear()
            self.last_down = None
            self.observed_down = 0.
            self.down_samples = 0
        height = observation.xyxy[3] - observation.xyxy[1]
        if height < config.min_body_height:
            self.interrupt(timestamp, config)
            return False
        self.history[:] = [p for p in self.history if timestamp - p[0] <= config.motion_window + 1e-9]
        vertical = self._vertical(points, confidence, config.baseline_max_angle)
        if self.candidate is not None:
            recovered = vertical and height / self.candidate[1][1] >= 1 - config.baseline_max_height_change
            if recovered:
                if self.recovered_since is None:
                    self.recovered_since = timestamp
                if timestamp - self.recovered_since >= config.recovery_seconds - 1e-9:
                    self.candidate = None
                    self.label_times.clear()
                    self.last_down = None
                    self.observed_down = 0.
                    self.down_samples = 0
            else:
                self.recovered_since = None
        upper = [points[i] for i in (0, 5, 6, 11, 12) if points[i][2] >= confidence]
        if self.candidate is None and len(upper) >= 2:
            bases = [p for p in self.history if p[3]]
            if len(bases) >= config.baseline_min_samples and bases[-1][0] - bases[0][0] >= config.baseline_seconds - 1e-9:
                base = bases[0]
                pairs = self._upper_pairs(base[2], points, confidence)
                dt = timestamp - base[0]
                if len(pairs) >= 2 and 0 < dt <= config.motion_window + 1e-9:
                    drop = statistics.median(b[1] - a[1] for a, b in pairs) / base[1]
                    recent = self.history + [(timestamp, height, points, vertical)]
                    speeds = []
                    for a, b in zip(recent, recent[1:]):
                        pair = self._upper_pairs(a[2], b[2], confidence)
                        if len(pair) >= 2 and .05 <= b[0] - a[0] <= config.max_frame_gap + 1e-9:
                            speeds.append(statistics.median(y[1] - x[1] for x, y in pair) / base[1] / (b[0] - a[0]))
                    speed = max([drop / dt, *speeds])
                    if drop >= config.partial_min_drop and speed >= config.min_descent_speed:
                        self.candidate = (timestamp, base)
        self.history.append((timestamp, height, points, vertical))
        if self.candidate is None:
            return False
        _, base = self.candidate
        scale, origin = base[1], base[2]
        pairs = self._upper_pairs(origin, points, confidence)
        drop = statistics.median(b[1] - a[1] for a, b in pairs) / scale if len(pairs) >= 2 else None
        horizontal = False
        for start, end in ((5, 15), (6, 16)):
            a, b = points[start], points[end]
            if min(a[2], b[2]) < confidence:
                continue
            dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
            horizontal |= (math.degrees(math.atan2(dx, dy)) >= config.post_min_angle
                           and dx / scale >= .35)
        for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
            a, b, end = points[hip], points[knee], points[ankle]
            if min(a[2], b[2], end[2]) < confidence:
                continue
            thigh = math.degrees(math.atan2(abs(b[0] - a[0]), abs(b[1] - a[1])))
            leg = math.degrees(math.atan2(abs(end[0] - a[0]), abs(end[1] - a[1])))
            horizontal |= (thigh >= config.post_min_angle and leg >= config.min_angle_change
                           and abs(end[0] - a[0]) / scale >= .25)
        compressed = height / scale <= config.alternate_max_height_ratio
        labelled = observation.posture_observed and observation.fallen_confidence >= config.fallen_label_threshold
        self.label_times[:] = [item for item in self.label_times
                               if timestamp - item[0] <= config.motion_window + 1e-9]
        if labelled:
            self.label_times.append((timestamp, observation.fallen_confidence))
        corroborated = (any(score >= .7 for _, score in self.label_times)
                        or (len(self.label_times) >= 2
                            and self.label_times[-1][0] - self.label_times[0][0] >= .2 - 1e-9))
        # A low box alone can be a seated person. Require an observed body
        # segment across the image, or a fresh model label AND visible descent.
        down = compressed and ((horizontal and (drop is None or drop >= config.partial_min_drop))
                               or (corroborated and drop is not None and drop >= config.min_drop))
        if not down:
            self.last_down = None
            self.observed_down = 0.
            self.down_samples = 0
            return False
        if self.last_down is not None:
            elapsed = timestamp - self.last_down
            if elapsed > config.max_frame_gap + 1e-9:
                self.observed_down = 0.
                self.down_samples = 0
            else:
                self.observed_down += min(elapsed, .1)
        self.last_down = timestamp
        self.down_samples += 1
        self.features.update(partial_landing_observed_seconds=self.observed_down,
                             partial_landing_height_ratio=height / scale,
                             partial_horizontal_segment=horizontal,
                             partial_weak_detection=observation.confidence < config.min_detection_confidence)
        return (self.observed_down + 1e-9 >= config.verify_seconds
                and self.down_samples >= config.verify_min_samples)