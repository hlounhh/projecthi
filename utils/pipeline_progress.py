"""In-process usable-output deadline; cannot interrupt a blocked main thread."""

import math


class InferenceStalled(RuntimeError):
    pass


class InferenceProgress:
    def __init__(self, *, timeout_seconds=30.):
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("Inference progress timeout must be finite and positive.")
        self.timeout_seconds = timeout_seconds
        self._since = None
        self._sequence = -1
        self._last_time = None

    def stalled(self, snapshot, ai_status, *, now, stale_seconds, offline=False):
        if (isinstance(now, bool) or not isinstance(now, (int, float))
                or not math.isfinite(now) or now < 0
                or self._last_time is not None and now < self._last_time):
            raise ValueError("Progress time must be finite, nonnegative and monotonic.")
        self._last_time = now
        source_state = getattr(snapshot.state, "value", snapshot.state)
        worker_state = getattr(ai_status.state, "value", ai_status.state) if ai_status else None
        frame = snapshot.frame
        available = (source_state == "STREAMING" and frame is not None
                     and (offline or 0 <= now - frame.received_at <= stale_seconds)
                     and worker_state in ("RUNNING", "WAITING"))
        if not available:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        result = ai_status.result
        # Counts may increase for discarded stale predictions. Only published,
        # advancing source samples demonstrate usable inference progress.
        if result is not None and result.source_sequence > self._sequence:
            self._sequence = result.source_sequence
            self._since = now
        return now - self._since >= self.timeout_seconds
