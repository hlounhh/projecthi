"""Single-owner capture worker with a one-frame mailbox (no frame queue)."""

from collections import deque
from dataclasses import dataclass
from enum import Enum
import logging
import math
import threading
import time
from typing import Any, Callable

from config import Settings

logger = logging.getLogger(__name__)


class StreamState(str, Enum):
    CONNECTING = "CONNECTING"
    STREAMING = "STREAMING"
    RECONNECTING = "RECONNECTING"
    EOF = "EOF"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class FramePacket:
    # The published image must be treated as read-only; copy before drawing.
    image: Any
    sequence: int
    received_at: float  # perf_counter AFTER cap.read(), not camera exposure time
    read_ms: float
    media_time_ms: float | None


@dataclass(frozen=True)
class StreamSnapshot:
    state: StreamState
    frame: FramePacket | None
    frames_received: int
    connection_attempts: int
    read_fps: float
    error: str | None


def open_capture(settings: Settings):
    # Import only after config has set the FFmpeg/OpenCV environment variables.
    import cv2

    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    if settings.is_file:
        return cv2.VideoCapture(settings.source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(
        settings.source,
        cv2.CAP_FFMPEG,
        [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, settings.open_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, settings.read_timeout_ms,
        ],
    )


class LatestFrameReader:
    """Only the worker opens/reads/releases capture, avoiding release/read races.

    Consumers may skip frames. Sequence numbers expose these skips. Local files
    are paced at their nominal FPS, but arrival timestamps still use wall time.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        loop_video: bool = False,
        capture_factory: Callable = open_capture,
    ):
        self.settings = settings
        self.loop_video = loop_video
        self._factory = capture_factory
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = StreamState.STOPPED
        self._frame: FramePacket | None = None
        self._count = 0
        self._attempts = 0
        self._error: str | None = None
        self._arrivals: deque[float] = deque(maxlen=600)

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Reader chỉ được start một lần.")
        self._set_state(StreamState.CONNECTING)
        self._thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float | None = None) -> bool:
        self._stop.set()
        if self._thread is not None and self._thread.ident is not None:
            if timeout is None:
                timeout = (self.settings.open_timeout_ms + self.settings.read_timeout_ms) / 1000 + 2
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger.warning("Capture worker chưa dừng: backend có thể không tôn trọng timeout.")
                return False
        return True

    def snapshot(self) -> StreamSnapshot:
        now = time.perf_counter()
        with self._lock:
            self._prune(now)
            fps = 0.0
            if len(self._arrivals) >= 2:
                span = self._arrivals[-1] - self._arrivals[0]
                if span > 0:
                    fps = (len(self._arrivals) - 1) / span
            return StreamSnapshot(
                self._state, self._frame, self._count, self._attempts, fps, self._error
            )

    def _prune(self, now: float):
        while self._arrivals and now - self._arrivals[0] > 2.0:
            self._arrivals.popleft()

    def _set_state(self, state: StreamState, error: str | None = None):
        with self._lock:
            changed = (self._state, self._error) != (state, error)
            self._state = state
            self._error = error
            if state != StreamState.STREAMING:
                self._frame = None
                self._arrivals.clear()
        if changed:
            logger.info("Camera state=%s%s", state.value, f" reason={error}" if error else "")

    def _run(self):
        import cv2

        try:
            while not self._stop.is_set():
                capture = None
                terminal = False
                try:
                    with self._lock:
                        self._attempts += 1
                    self._set_state(StreamState.CONNECTING)
                    capture = self._factory(self.settings)
                    if not capture.isOpened():
                        raise RuntimeError("open failed")
                    file_fps = 0.0
                    if self.settings.is_file:
                        file_fps = float(capture.get(cv2.CAP_PROP_FPS))
                        if not math.isfinite(file_fps) or not 0 < file_fps <= 240:
                            file_fps = 25.0
                            logger.warning("Video thiếu FPS hợp lệ; dùng nhịp phát 25 FPS.")
                    deadline = time.perf_counter()
                    while not self._stop.is_set():
                        if self.settings.is_file and self._stop.wait(max(0, deadline - time.perf_counter())):
                            break
                        started = time.perf_counter()
                        ok, image = capture.read()
                        received = time.perf_counter()
                        if not ok or image is None or image.size == 0:
                            if self.settings.is_file:
                                if not self.loop_video:
                                    self._set_state(StreamState.EOF, "Hết video hoặc không đọc được khung hình.")
                                    terminal = True
                                else:
                                    self._set_state(StreamState.RECONNECTING, "Phát lại video.")
                            else:
                                self._set_state(StreamState.RECONNECTING, "Không nhận được khung hình; sẽ thử lại.")
                            break
                        media_ms = None
                        if self.settings.is_file:
                            value = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                            if math.isfinite(value) and value >= 0:
                                media_ms = value
                            # Never burst through old file frames to catch up after a delay.
                            deadline = max(deadline + 1 / file_fps, received)
                        with self._lock:
                            self._count += 1
                            self._frame = FramePacket(
                                image, self._count, received, (received - started) * 1000, media_ms
                            )
                            self._state = StreamState.STREAMING
                            self._error = None
                            self._arrivals.append(received)
                            self._prune(received)
                except Exception:
                    # Exception text from capture backends may contain the credential-bearing URL.
                    state = StreamState.ERROR if self.settings.is_file else StreamState.RECONNECTING
                    self._set_state(state, "Không mở/đọc được nguồn; kiểm tra mạng, cấu hình hoặc codec.")
                    terminal = self.settings.is_file
                finally:
                    if capture is not None:
                        try:
                            capture.release()
                        except Exception:
                            logger.warning("Không giải phóng được capture backend.")
                if terminal:
                    return
                if self._stop.wait(self.settings.reconnect_seconds):
                    break
        finally:
            if self._stop.is_set():
                self._set_state(StreamState.STOPPED)
