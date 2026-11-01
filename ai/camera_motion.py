"""Camera motion compensation for person-facing cameras (PTZ tracking et al.)."""

from dataclasses import replace

import cv2
import numpy as np

from ai.fall_engine import CONFIRMED_FALL, SUSPECTED_FALL


class CameraMotionEstimator:
    """Phase correlation on a fixed-size grayscale pair; one global shift per frame.

    A fixed measurement size keeps the cost constant regardless of resolution.
    """

    WIDTH = 128

    def __init__(self):
        self.width = self.WIDTH
        self.height = None
        self._prev_gray = None

    def update(self, frame):
        if self.height is None:
            h, w = frame.shape[:2]
            self.height = max(32, round(self.width * h / w))
        source_shape = frame.shape[:2]
        # Shrink before grayscale: resizing color is cheaper than converting
        # full-resolution frames to gray.
        if frame.ndim == 3:
            frame = cv2.resize(frame, (self.width, self.height),
                               interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.resize(frame, (self.width, self.height),
                              interpolation=cv2.INTER_LINEAR)
        if self._prev_gray is None:
            small = gray.astype(np.float32) if gray.dtype == np.uint8 else gray
            self._prev_gray = small
            return 0.0, 0.0, 0.0, 1.0
        prev = self._prev_gray
        cur = gray.astype(np.float32) if gray.dtype == np.uint8 else gray
        h, w = prev.shape
        cross = np.fft.rfft2(prev) * np.conj(np.fft.rfft2(cur))
        corr = np.fft.irfft2(cross / np.maximum(np.abs(cross), 1e-9))
        peak_index = np.unravel_index(corr.argmax(), corr.shape)
        peak = float(corr[peak_index])
        # Z-score against the correlation floor: pure noise never yields a strong
        # outlier peak, a real match does. Map z in [4, 12] to confidence [0, 1].
        std = float(corr.std())
        confidence = max(0.0, min(1.0, (peak / std - 4.0) / 8.0)) if std > 1e-9 else 0.0
        dy = peak_index[0] if peak_index[0] <= h // 2 else peak_index[0] - h
        dx = peak_index[1] if peak_index[1] <= w // 2 else peak_index[1] - w
        # The returned shift is the observation offset between frames, not the
        # raw cross-correlation displacement.
        scale_y, scale_x = (s / m for s, m in zip(source_shape, (h, w)))
        dx, dy = -dx * scale_x, -dy * scale_y
        self._prev_gray = cur
        return float(dx), float(dy), 0.0, confidence

    def reset(self):
        self._prev_gray = None
        self.height = None


class MotionCompensatedFallEngine:
    """Stabilizes observations against global camera shift before FallEngine.update.

    An untrusted shift still passes the observations through, but CONFIRMED_FALL
    events are downgraded to SUSPECTED_FALL for later human confirmation.
    """

    def __init__(self, engine, *, estimator=None, min_confidence=0.35):
        self.engine = engine
        self.estimator = estimator or CameraMotionEstimator()
        self.min_confidence = float(min_confidence)
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.last_confidence = 1.0
        self.trusted = True

    def update(self, detections, timestamp, source_shape, *, frame=None):
        if frame is None:
            return self.engine.update(detections, timestamp, source_shape)
        dx, dy, _, confidence = self.estimator.update(frame)
        self.last_dx, self.last_dy, self.last_confidence = dx, dy, confidence
        self.trusted = confidence >= self.min_confidence
        if self.trusted and (dx or dy):
            detections = tuple(
                replace(obs,
                        xyxy=tuple(v - d for v, d in zip(obs.xyxy, (dx, dy, dx, dy))),
                        keypoints=tuple((p[0] - dx, p[1] - dy, p[2]) for p in obs.keypoints))
                for obs in detections)
        events = self.engine.update(detections, timestamp, source_shape)
        if not self.trusted:
            events = tuple(replace(e, state=SUSPECTED_FALL)
                           if e.state == CONFIRMED_FALL else e for e in events)
        return events

    def reset(self):
        self.last_dx, self.last_dy, self.last_confidence = 0.0, 0.0, 1.0
        self.trusted = True
        self.estimator.reset()
        if hasattr(self.engine, "reset"):
            self.engine.reset()

    def snapshot(self, *a, **kw):
        return self.engine.snapshot(*a, **kw)

    def mark_unobserved(self, *a, **kw):
        return self.engine.mark_unobserved(*a, **kw)

    def __getattr__(self, name):
        return getattr(self.engine, name)
