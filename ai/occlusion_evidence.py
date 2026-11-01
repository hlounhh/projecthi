"""Bounded evidence from actual poses; unobserved time never verifies a fall."""

from dataclasses import dataclass, field


@dataclass
class OcclusionEvidence:
    history: list = field(default_factory=list)
    candidate: tuple | None = None
    last_down: float | None = None
    observed_down: float = 0.
    down_samples: int = 0
    last_box: tuple | None = None
    last_seen: float | None = None
    motion_landing: tuple | None = None
    last_curled: float | None = None
    observed_curled: float = 0.
    curled_samples: int = 0
    confirmation_reason: str | None = None
    identity_points: tuple = ()
    identity_box: tuple | None = None
    identity_at: float | None = None
    reacquiring: bool = False
    reacquire_samples: int = 0
    reacquire_at: float | None = None

    def clear(self):
        self.history.clear()
        self.candidate = None
        self.last_down = None
        self.observed_down = 0.
        self.down_samples = 0
        self.last_box = None
        self.last_seen = None
        self._clear_landing()
        self.confirmation_reason = None
        self.identity_points = ()
        self.identity_box = None
        self.identity_at = None
        self.reacquiring = False
        self.reacquire_samples = 0
        self.reacquire_at = None

    def _expire(self, timestamp, config):
        lifetime = config.suspicion_timeout + config.verify_seconds
        if self.motion_landing is not None and timestamp - self.motion_landing[0] > lifetime:
            self._clear_landing()
        if self.candidate is not None and timestamp - self.candidate[0] > lifetime:
            self.candidate = None
            self.last_down = None
            self.observed_down = 0.
            self.down_samples = 0

    def interrupt(self, timestamp, config):
        self._expire(timestamp, config)
        if (self.identity_at is not None
                and timestamp - self.identity_at > min(config.occlusion_grace_seconds, config.max_track_gap) + 1e-9):
            self.clear()
            return
        self.history.clear()
        self.last_down = self.last_curled = None
        self.observed_down = self.observed_curled = 0.
        self.down_samples = self.curled_samples = 0
        self.reacquiring = self.motion_landing is not None or self.candidate is not None
        self.reacquire_samples = 0
        self.reacquire_at = None

    def _identity_continuous(self, observation, timestamp, config):
        if self.identity_at is None or self.identity_box is None:
            return False
        if timestamp - self.identity_at > min(config.occlusion_grace_seconds, config.max_track_gap) + 1e-9:
            return False
        a, b = self.identity_box, observation.xyxy
        intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in (a, b)]
        overlap = intersection / (sum(areas) - intersection)
        scale = max(a[3] - a[1], b[3] - b[1])
        displacement = sorted(((observation.keypoints[index][0] - point[0]) ** 2
                               + (observation.keypoints[index][1] - point[1]) ** 2) ** .5
                              for index, point in zip((5, 6, 11, 12), self.identity_points))
        return (overlap >= config.association_min_iou
                and max(areas) / min(areas) <= config.association_max_area_ratio
                and max(displacement) / scale <= config.association_max_distance)

    def _remember_identity(self, observation, timestamp):
        self.identity_points = tuple(observation.keypoints[index][:2] for index in (5, 6, 11, 12))
        self.identity_box = observation.xyxy
        self.identity_at = timestamp

    def _clear_landing(self):
        self.motion_landing = None
        self.last_curled = None
        self.observed_curled = 0.
        self.curled_samples = 0

    def arm_landing(self, base, timestamp):
        """Called only after the engine measures a reliable rapid fall motion."""
        if self.motion_landing is None:
            self.motion_landing = (timestamp, base.height, base.torso_y, base.hip_y)

    def _curled_landing(self, observation, timestamp, sample, config, post_fall):
        if self.motion_landing is None:
            return False
        started, height, torso_y, hip_y = self.motion_landing
        if timestamp - started > config.suspicion_timeout + config.verify_seconds:
            self._clear_landing()
            return False
        # A foreshortened, curled landing may hide both ankles and project a
        # vertical torso. Compensate for absent leg-extension evidence with
        # twice the usual descent, a compressed silhouette and both torso sides.
        points = observation.keypoints
        bilateral = (sample is not None and len(points) == 17
                     and all(points[i][2] >= config.min_keypoint_confidence for i in (5, 6, 11, 12)))
        compact = (bilateral and sample.height / height <= config.alternate_max_height_ratio
                   and sample.aspect >= config.alternate_min_aspect
                   and (sample.hip_y - hip_y) / height >= config.min_drop
                   and (sample.torso_y - torso_y) / height >= 2 * config.min_drop)
        extended = (bilateral and post_fall
                    and (sample.torso_y - torso_y) / height >= config.min_drop)
        if not (compact or extended):
            self.last_curled = None
            self.observed_curled = 0.
            self.curled_samples = 0
            return False
        if self.last_curled is not None:
            gap = timestamp - self.last_curled
            if gap > config.max_frame_gap + 1e-9:
                self.observed_curled = 0.
                self.curled_samples = 0
            else:
                # Sparse observations cannot borrow all the unobserved time.
                self.observed_curled += min(gap, .1)
        self.last_curled = timestamp
        self.curled_samples += 1
        if (self.observed_curled + 1e-9 >= config.verify_seconds
                and self.curled_samples >= config.verify_min_samples):
            self.confirmation_reason = "curled_landing_confirmed" if compact else "observed_landing_confirmed"
            return True
        return False

    def observe(self, observation, timestamp, sample, upright, config, *, post_fall=False):
        points, box = observation.keypoints, observation.xyxy
        self._expire(timestamp, config)
        if observation.confidence < config.min_detection_confidence:
            self.clear()
            return False
        previous_seen, previous_box = self.last_seen, self.last_box
        if previous_seen is not None:
            if timestamp - previous_seen > config.max_frame_gap + 1e-9:
                self.interrupt(timestamp, config)
            ax1, ay1, ax2, ay2 = previous_box
            bx1, by1, bx2, by2 = box
            overlap = max(0., min(ax2, bx2) - max(ax1, bx1)) * max(0., min(ay2, by2) - max(ay1, by1))
            if timestamp - previous_seen > config.occlusion_grace_seconds + 1e-9 or overlap <= 0:
                self.clear()
        self.last_seen, self.last_box = timestamp, box
        bilateral = (sample is not None and len(points) == 17
                     and all(points[index][2] >= config.min_keypoint_confidence for index in (5, 6, 11, 12)))
        if upright:
            self._clear_landing()
            self.candidate = None
            self.reacquiring = False
        if not bilateral:
            self.interrupt(timestamp, config)
            return False
        if self.reacquiring:
            if not self._identity_continuous(observation, timestamp, config):
                self.clear()
            else:
                if (self.reacquire_at is not None
                        and timestamp - self.reacquire_at > config.max_frame_gap + 1e-9):
                    self.reacquire_samples = 0
                self.reacquire_samples += 1
                self.reacquire_at = timestamp
                self._remember_identity(observation, timestamp)
                if self.reacquire_samples >= 2:
                    self.reacquiring = False
                return False
        self._remember_identity(observation, timestamp)
        if self._curled_landing(observation, timestamp, sample, config, post_fall):
            return True
        if sample is None:
            return False
        sides = [(s, h) for s, h in ((5, 11), (6, 12))
                 if min(points[s][2], points[h][2]) >= config.min_keypoint_confidence]
        shoulder_y = sum(points[s][1] for s, _ in sides) / len(sides)
        full = sample.lower_body and len(sides) == 2
        self.history[:] = [p for p in self.history if timestamp - p[0] <= config.motion_window + 1e-9]
        if upright:
            self.candidate = None
            self.last_down = None
            self.observed_down = 0.
            self.down_samples = 0
        if full:
            current = (timestamp, sample.height, shoulder_y, sample.angle, upright)
            bases = [p for p in self.history if p[4]]
            if self.candidate is None and len(bases) >= config.baseline_min_samples:
                base = max(bases, key=lambda p: p[1])
                recent = self.history + [current]
                speed = max(((b[2] - a[2]) / base[1] / (b[0] - a[0])
                             for a, b in zip(recent, recent[1:])
                             if 0.05 <= b[0] - a[0] <= config.max_frame_gap + 1e-9), default=0.)
                if (bases[-1][0] - bases[0][0] + 1e-9 >= config.baseline_seconds
                        and (shoulder_y - base[2]) / base[1] >= config.min_drop
                        and speed >= config.min_descent_speed
                        and sample.angle - base[3] >= config.alternate_min_angle):
                    self.candidate = (timestamp, base[1], base[2])
            self.history.append(current)
        if self.candidate is None:
            return False
        _, height, shoulder = self.candidate
        legs = all(points[i][2] >= config.min_keypoint_confidence for i in (11, 12, 13, 14, 15, 16))
        down = (full and legs and sample.angle >= config.post_min_angle
                and sample.aspect >= config.down_min_aspect
                and sample.height / height <= config.alternate_max_height_ratio
                and (shoulder_y - shoulder) / height >= config.min_drop)
        if not down:
            self.last_down = None
            self.observed_down = 0.
            self.down_samples = 0
            return False
        if self.last_down is not None:
            gap = timestamp - self.last_down
            if gap > .95:
                self.observed_down = 0.
                self.down_samples = 0
            else:
                self.observed_down += min(gap, .1)
        self.last_down = timestamp
        self.down_samples += 1
        return (self.observed_down + 1e-9 >= config.verify_seconds
                and self.down_samples >= config.verify_min_samples)
