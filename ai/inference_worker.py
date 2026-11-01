"""Asynchronous inference over the camera's latest-frame mailbox, never a queue."""

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
import threading
import time
from typing import Any, Protocol

from camera.rtsp_reader import StreamSnapshot, StreamState


class Detector(Protocol):
    def predict(self, image: Any) -> tuple[Any, ...]:
        """Return immutable detections; the worker does not inspect their contents."""
        ...


class FrameReader(Protocol):
    def snapshot(self) -> StreamSnapshot:
        ...


class WorkerState(str, Enum):
    STOPPED = "STOPPED"
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    EOF = "EOF"
    ERROR = "ERROR"


@dataclass(frozen=True)
class InferenceResult:
    detections: tuple[Any, ...]
    source_sequence: int
    source_received_at: float
    completed_at: float
    inference_ms: float
    source_shape: tuple[int, ...]
    source_media_ms: float | None = None


@dataclass(frozen=True)
class WorkerSnapshot:
    state: WorkerState
    result: InferenceResult | None
    inference_fps: float
    error: str | None
    processed_count: int


class InferenceWorker:
    """Single-owner inference; live results expire by source arrival time.

    Offline readers retain each sample until the application acknowledges it.
    Stop is bounded but cannot interrupt an executing model call.
    """

    _POLL_SECONDS = 0.01
    _ERROR = "Inference worker failed; check the detector and camera source."
    _SOURCE_ERROR = "Camera source is unavailable."

    def __init__(
        self,
        reader: FrameReader,
        detector: Detector,
        *,
        max_fps: float = 10.0,
        stale_seconds: float = 1.0,
        max_result_age_seconds: float = 1.0,
    ):
        for name, value in (
            ("max_fps", max_fps),
            ("stale_seconds", stale_seconds),
            ("max_result_age_seconds", max_result_age_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        self._reader = reader
        self._offline = getattr(reader, "offline", False)
        self._detector = detector
        self._interval = 1.0 / max_fps
        self._stale_seconds = stale_seconds
        self._result_age = min(stale_seconds, max_result_age_seconds)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = WorkerState.STOPPED
        self._result: InferenceResult | None = None
        self._error: str | None = None
        self._count = 0
        self._completions: deque[float] = deque(maxlen=600)
        # Invalidate in-flight work if a consumer observes a disconnect/stall.
        self._source_epoch = 0

    def start(self):
        with self._lock:
            if self._thread is not None or self._stop.is_set():
                raise RuntimeError("Inference worker may only be started once.")
            self._state = WorkerState.WAITING
            self._thread = threading.Thread(
                target=self._run, name="inference-worker", daemon=True
            )
            self._thread.start()
        return self

    def stop(self, timeout: float = 1.0) -> bool:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative.")
        with self._lock:
            self._stop.set()
            self._result = None
            if self._state not in (WorkerState.ERROR, WorkerState.EOF):
                self._state = WorkerState.STOPPED
            thread = self._thread
        if thread is not None and thread.ident is not None:
            thread.join(timeout)
            return not thread.is_alive()
        return True

    def snapshot(self, source: StreamSnapshot | None = None) -> WorkerSnapshot:
        # Read the camera outside our lock; never hold a lock across prediction.
        with self._lock:
            active = not self._stop.is_set() and self._thread is not None
        if active:
            try:
                if source is None:
                    source = self._reader.snapshot()
                self._observe_source(source, time.perf_counter())
            except Exception:
                self._fail()
        now = time.perf_counter()
        with self._lock:
            if not self._offline and self._result is not None and now - self._result.source_received_at > self._result_age:
                self._result = None
            while self._completions and now - self._completions[0] > 2.0:
                self._completions.popleft()
            fps = 0.0
            if len(self._completions) >= 2:
                span = self._completions[-1] - self._completions[0]
                if span > 0:
                    fps = (len(self._completions) - 1) / span
            return WorkerSnapshot(self._state, self._result, fps, self._error, self._count)

    def _observe_source(self, source: StreamSnapshot, now: float) -> bool:
        with self._lock:
            if self._stop.is_set():
                return False
            if source.state in (StreamState.EOF, StreamState.ERROR):
                self._state = WorkerState.EOF if source.state == StreamState.EOF else WorkerState.ERROR
                self._error = None if source.state == StreamState.EOF else self._SOURCE_ERROR
                self._result = None
                self._source_epoch += 1
                self._stop.set()
                return False
            fresh = (
                source.state == StreamState.STREAMING
                and source.frame is not None
                and (self._offline or now - source.frame.received_at <= self._stale_seconds)
            )
            if not fresh:
                self._result = None
                self._state = WorkerState.WAITING
                self._source_epoch += 1
            return fresh

    def _fail(self):
        # Never expose backend exception messages/tracebacks (possibly credentials).
        with self._lock:
            self._state = WorkerState.ERROR
            self._error = self._ERROR
            self._result = None
            self._stop.set()

    def _run(self):
        last_sequence = -1
        next_start = 0.0
        try:
            while not self._stop.is_set():
                source = self._reader.snapshot()
                now = time.perf_counter()
                fresh = self._observe_source(source, now)
                if self._stop.is_set():
                    break
                packet = source.frame
                if not fresh or packet.sequence <= last_sequence:
                    self._stop.wait(self._POLL_SECONDS)
                    continue
                delay = next_start - now
                if delay > 0:
                    # Take another snapshot after waiting, not a queued packet.
                    self._stop.wait(min(delay, self._POLL_SECONDS))
                    continue
                with self._lock:
                    if self._stop.is_set():
                        break
                    epoch = self._source_epoch
                    self._state = WorkerState.RUNNING
                shape = tuple(packet.image.shape)
                last_sequence = packet.sequence
                started = time.perf_counter()
                next_start = started + self._interval
                detections = tuple(self._detector.predict(packet.image))
                completed = time.perf_counter()
                result = InferenceResult(
                    detections, packet.sequence, packet.received_at, completed,
                    (completed - started) * 1000, shape, packet.media_time_ms,
                )
                with self._lock:
                    self._count += 1
                    self._completions.append(completed)
                # A slow prediction must not resurrect disconnected/stale frames.
                live = self._reader.snapshot()
                checked_at = time.perf_counter()
                live_fresh = self._observe_source(live, checked_at)
                with self._lock:
                    if (
                        not self._stop.is_set()
                        and live_fresh
                        and epoch == self._source_epoch
                        and (self._offline or checked_at - packet.received_at <= self._result_age)
                    ):
                        self._result = result
        except Exception:
            self._fail()
