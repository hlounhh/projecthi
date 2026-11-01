"""Fixed media-time samples with bounded decode-ahead, never dropped."""

from collections import deque
import math
import threading
import time

from camera.rtsp_reader import FramePacket, StreamSnapshot, StreamState, open_capture


class VideoAnalysisReader:
    offline = True

    def __init__(self, settings, *, sample_fps=10, loop_video=False, capture_factory=open_capture):
        if not math.isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError("sample_fps must be finite and positive.")
        self.settings = settings
        self.sample_fps = sample_fps
        self.loop_video = loop_video
        self._factory = capture_factory
        self._condition = threading.Condition()
        self._stop = False
        self._thread = None
        self._state = StreamState.STOPPED
        self._frame = None
        self._count = 0
        self._attempts = 0
        self._error = None
        self._pending = deque()
        self._capacity = 4
        self._decoded = False

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Reader may only be started once.")
        self._state = StreamState.CONNECTING
        self._thread = threading.Thread(target=self._run, name="video-reader", daemon=True)
        self._thread.start()
        return self

    def snapshot(self):
        with self._condition:
            return StreamSnapshot(self._state, self._frame, self._count, self._attempts, 0.0, self._error)

    def acknowledge(self, sequence):
        with self._condition:
            if self._frame is None or sequence != self._frame.sequence:
                return
            self._pending.popleft()
            self._frame = self._pending[0] if self._pending else None
            if self._frame is None:
                self._state = StreamState.EOF if self._decoded else StreamState.CONNECTING
            self._condition.notify_all()

    def stop(self, timeout=5):
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True

    def _finish(self, state, error=None):
        with self._condition:
            self._state, self._error, self._frame = state, error, None

    def _run(self):
        import cv2

        try:
            while not self._stop:
                capture = None
                try:
                    with self._condition:
                        self._attempts += 1
                    capture = self._factory(self.settings)
                    if not capture.isOpened():
                        raise ValueError("Cannot open video.")
                    fps = float(capture.get(cv2.CAP_PROP_FPS))
                    if not math.isfinite(fps) or not 0 < fps <= 240:
                        raise ValueError("Video needs valid FPS for media-time analysis.")
                    index, next_sample = 0, 0.0
                    interval = max(1 / fps, 1 / self.sample_fps)
                    while not self._stop:
                        started = time.perf_counter()
                        if not capture.grab():
                            break
                        timestamp = index / fps
                        index += 1
                        if timestamp + 1e-9 < next_sample:
                            continue
                        ok, image = capture.retrieve()
                        received = time.perf_counter()
                        if not ok or image is None or image.size == 0:
                            raise ValueError("Cannot retrieve sampled frame.")
                        next_sample += interval
                        with self._condition:
                            self._count += 1
                            packet = FramePacket(image, self._count, received,
                                                 (received - started) * 1000, timestamp * 1000)
                            self._pending.append(packet)
                            self._frame = self._pending[0]
                            self._state = StreamState.STREAMING
                            self._condition.wait_for(lambda: self._stop or len(self._pending) < self._capacity)
                    with self._condition:
                        self._decoded = not self.loop_video
                        if not self._pending:
                            self._state = StreamState.EOF if self._decoded else StreamState.CONNECTING
                        self._condition.wait_for(lambda: self._stop or not self._pending)
                    if not self.loop_video:
                        return
                finally:
                    if capture is not None:
                        capture.release()
        except Exception:
            self._finish(StreamState.ERROR, "Không đọc được video hoặc FPS không hợp lệ.")
        finally:
            if self._stop:
                self._finish(StreamState.STOPPED)
