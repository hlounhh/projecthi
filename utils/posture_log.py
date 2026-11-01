"""Debounced scene posture summaries, not fall verification or person tracking."""

from collections import Counter
import math
import time

LABELS = {
    "standing": "Đang đứng",
    "sitting": "Đang ngồi",
    "fallen": "Có dấu hiệu ngã hoặc nằm thấp (chưa xác nhận ngã)",
}


class PostureReporter:
    def __init__(self, *, stable_seconds=0.5, missing_seconds=1.0, min_samples=3):
        if not all(math.isfinite(v) and v > 0 for v in (stable_seconds, missing_seconds)):
            raise ValueError("Posture time thresholds must be finite and positive.")
        if not isinstance(min_samples, int) or min_samples < 2:
            raise ValueError("At least two distinct inference samples are required.")
        self.stable_seconds = stable_seconds
        self.missing_seconds = missing_seconds
        self.min_samples = min_samples
        self._last_sequence = None
        self._last_counts = ()
        self._candidate = ()
        self._candidate_since = 0.0
        self._candidate_samples = 0
        self._last_seen = None

    def update(self, result, *, now=None) -> str | None:
        now = time.perf_counter() if now is None else now
        if self._last_seen is not None and now - self._last_seen >= self.missing_seconds:
            self._last_counts = ()
            self._candidate = ()
            self._candidate_samples = 0
        if result is None:
            # A brief missing/expired result must not reprint the same posture.
            self._candidate = ()
            self._candidate_samples = 0
            return None
        if result.source_sequence == self._last_sequence:
            return None
        self._last_sequence = result.source_sequence
        counts = Counter(d.label.lower() for d in result.detections if d.label.lower() in LABELS)
        signature = tuple(sorted(counts.items()))
        if not signature:
            self._candidate = ()
            self._candidate_samples = 0
            return None
        self._last_seen = now
        if signature != self._candidate:
            self._candidate = signature
            self._candidate_since = result.source_received_at
            self._candidate_samples = 1
            return None
        self._candidate_samples += 1
        if (self._candidate_samples < self.min_samples
                or result.source_received_at - self._candidate_since < self.stable_seconds
                or signature == self._last_counts):
            return None
        self._last_counts = signature
        # A box count is not a verified person count without tracking.
        return " | ".join(
            LABELS[label] + (f": {counts[label]} vùng nhận diện" if sum(counts.values()) > 1 else "")
            for label in LABELS if label in counts
        )
