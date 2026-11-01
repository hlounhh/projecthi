"""Per-person One Euro filtering; confidence and missing joints stay observed.

Coordinates use a fixed initial body span, so beta has the same meaning at
different image resolutions. Never share an instance between identities.
"""

import math


def _alpha(cutoff, dt):
    return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))


class KeypointSmoother:
    def __init__(self, *, scale, min_cutoff=1.5, beta=5.0, derivative_cutoff=1.0):
        for name, value in (("scale", scale), ("min_cutoff", min_cutoff),
                            ("beta", beta), ("derivative_cutoff", derivative_cutoff)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive.")
        self.scale = scale
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.reset()

    def reset(self):
        self._joints = [None] * 17
        self._timestamp = None

    def filter(self, points, timestamp, *, min_confidence=0.5):
        if len(points) != 17:
            raise ValueError("Expected COCO17 keypoints.")
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp < 0
                or self._timestamp is not None and timestamp <= self._timestamp):
            raise ValueError("Filter timestamps must be finite and strictly increasing.")
        self._timestamp = timestamp
        result = []
        for index, (x, y, confidence) in enumerate(points):
            if confidence < min_confidence:
                # Reacquisition starts fresh; never bridge an occluded joint.
                self._joints[index] = None
                result.append((x, y, confidence))
                continue
            raw = (x / self.scale, y / self.scale)
            previous = self._joints[index]
            if previous is None:
                filtered, derivative = raw, (0.0, 0.0)
            else:
                last_raw, last_filtered, last_derivative, last_time = previous
                dt = timestamp - last_time
                derivative_alpha = _alpha(self.derivative_cutoff, dt)
                derivative = tuple(derivative_alpha * ((v - old) / dt)
                                   + (1 - derivative_alpha) * old_derivative
                                   for v, old, old_derivative in zip(raw, last_raw, last_derivative))
                cutoff = self.min_cutoff + self.beta * math.hypot(*derivative)
                alpha = _alpha(cutoff, dt)
                filtered = tuple(alpha * v + (1 - alpha) * old
                                 for v, old in zip(raw, last_filtered))
            self._joints[index] = (raw, filtered, derivative, timestamp)
            result.append((filtered[0] * self.scale, filtered[1] * self.scale, confidence))
        return tuple(result)