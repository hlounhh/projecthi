"""Append-only local measurements; no invented camera/E2E latency estimates."""

from collections import deque
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
import threading
import uuid

from camera.rtsp_reader import StreamSnapshot


class SessionLogUnavailable(OSError):
    pass


class SessionFileHandler(logging.FileHandler):
    def __init__(self, *args, **kwargs):
        self.failed = threading.Event()
        super().__init__(*args, **kwargs)

    def handleError(self, record):
        # Called inside emit's exception handler, potentially on a camera thread.
        # Never format the failed record or print backend traceback/credentials.
        self.failed.set()


class SessionLogger(logging.Logger):
    def __init__(self, log_dir: Path):
        super().__init__("fall_camera", logging.INFO)
        self.propagate = False
        self._camera = logging.getLogger("camera")
        if any(getattr(handler, "_fall_session", False) for handler in self._camera.handlers):
            raise RuntimeError("Session logging is already active in this process.")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._handler = SessionFileHandler(log_dir / "camera.log", encoding="utf-8")
        self._failed = self._handler.failed
        self._handler._fall_session = True
        self._handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        self._camera_level = self._camera.level
        self.addHandler(self._handler)
        self._camera.addHandler(self._handler)
        self._camera.setLevel(logging.INFO)

    def check_health(self):
        if self._failed.is_set():
            raise SessionLogUnavailable("Session log persistence failed.")

    def close(self):
        handler = self._handler
        if handler is None:
            return
        self._handler = None
        self.removeHandler(handler)
        self._camera.removeHandler(handler)
        self._camera.setLevel(self._camera_level)
        try:
            handler.flush()
        finally:
            handler.close()
        self.check_health()


def setup_logging(log_dir: Path) -> SessionLogger:
    return SessionLogger(log_dir)


@contextmanager
def model_logs_to_file(session_logger):
    model_logger = logging.getLogger("ultralytics")

    def route(record):
        session_logger.handle(record)
        return record.levelno >= logging.WARNING

    model_logger.addFilter(route)
    try:
        yield
    finally:
        model_logger.removeFilter(route)


class FrameCounter:
    """Count distinct frames actually consumed, not GUI redraw iterations."""

    def __init__(self):
        self._times: deque[float] = deque(maxlen=600)
        self.last_sequence = 0
        self.skipped = 0
        self.consumed = 0

    def record(self, sequence: int, now: float) -> bool:
        if sequence <= self.last_sequence:
            return False
        self.skipped += max(0, sequence - self.last_sequence - 1)
        self.last_sequence = sequence
        self.consumed += 1
        self._times.append(now)
        return True

    def fps(self, now: float) -> float:
        while self._times and now - self._times[0] > 2:
            self._times.popleft()
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


class MetricsWriter:
    FIELDS = (
        "utc_time", "elapsed_s", "state", "frames_received", "frames_consumed",
        "frames_skipped_by_consumer", "connection_attempts", "read_fps",
        "consumer_fps", "last_read_ms", "post_read_age_ms", "source_media_ms",
        "frame_stale", "camera_to_screen_latency_ms", "ai_state", "ai_fps",
        "ai_processed", "ai_inference_ms", "ai_source_age_ms", "ai_boxes",
        "tracked_people", "unobserved_tracks", "suspected_tracks", "confirmed_tracks",
    )

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.path = log_dir / f"metrics_{stamp}_{uuid.uuid4().hex[:8]}.csv"
        self._file = self.path.open("x", newline="", encoding="utf-8")
        try:
            self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
            self._writer.writeheader()
            self._file.flush()
        except BaseException:
            self._file.close()
            raise
        self.started = time.perf_counter()

    def write(self, snapshot: StreamSnapshot, counter: FrameCounter, stale_seconds: float, ai_status=None,
              tracks=None):
        now = time.perf_counter()
        ai_result = ai_status.result if ai_status else None
        frame = snapshot.frame
        age = max(0, now - frame.received_at) if frame else None
        self._writer.writerow({
            "utc_time": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(now - self.started, 3),
            "state": snapshot.state.value,
            "frames_received": snapshot.frames_received,
            "frames_consumed": counter.consumed,
            "frames_skipped_by_consumer": counter.skipped,
            "connection_attempts": snapshot.connection_attempts,
            "read_fps": round(snapshot.read_fps, 2),
            "consumer_fps": round(counter.fps(now), 2),
            "last_read_ms": round(frame.read_ms, 3) if frame else "",
            "post_read_age_ms": round(age * 1000, 3) if age is not None else "",
            "source_media_ms": frame.media_time_ms if frame else "",
            "frame_stale": age is None or age > stale_seconds,
            # Cannot be inferred from read() arrival time. Requires external measurement.
            "camera_to_screen_latency_ms": "",
            "ai_state": ai_status.state.value if ai_status else "",
            "ai_fps": round(ai_status.inference_fps, 2) if ai_status else "",
            "ai_processed": ai_status.processed_count if ai_status else "",
            "ai_inference_ms": round(ai_result.inference_ms, 2) if ai_result else "",
            "ai_source_age_ms": round(max(0, now - ai_result.source_received_at) * 1000, 2) if ai_result else "",
            "ai_boxes": len(ai_result.detections) if ai_result else "",
            "tracked_people": len(tracks) if tracks is not None else "",
            "unobserved_tracks": sum(not t.observation_quality for t in tracks) if tracks is not None else "",
            "suspected_tracks": sum(t.state == "NGHI NGA" for t in tracks) if tracks is not None else "",
            "confirmed_tracks": sum(t.state == "DA NGA" for t in tracks)
            if tracks is not None else "",
        })
        self._file.flush()

    def close(self):
        self._file.close()
