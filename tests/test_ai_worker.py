"""Worker tests use no camera, network, NumPy, OpenCV or model runtime."""

from dataclasses import dataclass, FrozenInstanceError
import sys
import threading
import time
import unittest
from unittest.mock import patch

from ai.inference_worker import InferenceWorker, WorkerState
from camera.rtsp_reader import FramePacket, StreamSnapshot, StreamState


@dataclass(frozen=True)
class FakeImage:
    sequence: int
    shape: tuple[int, ...] = (24, 32, 3)


@dataclass(frozen=True)
class FakeDetection:
    label: str = "person"


class FakeReader:
    def __init__(self):
        self._lock = threading.Lock()
        self.state = StreamState.CONNECTING
        self.frame = None
        self.error = None

    def publish(self, sequence, received_at=None):
        with self._lock:
            self.state = StreamState.STREAMING
            self.frame = FramePacket(
                FakeImage(sequence), sequence,
                time.perf_counter() if received_at is None else received_at,
                0.0, None,
            )

    def set_state(self, state, *, keep_frame=False, error=None):
        with self._lock:
            self.state = state
            self.error = error
            if not keep_frame:
                self.frame = None

    def snapshot(self):
        with self._lock:
            return StreamSnapshot(self.state, self.frame, 0, 0, 0.0, self.error)


class FakeDetector:
    def __init__(self, block=False):
        self.calls = []
        self.times = []
        self.threads = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = block
        self.detections = (FakeDetection(),)

    def predict(self, image):
        self.calls.append(image.sequence)
        self.times.append(time.perf_counter())
        self.threads.append(threading.current_thread())
        self.entered.set()
        if self.block and not self.release.wait(2):
            raise RuntimeError("Fake detector was not released")
        return self.detections


def wait_until(predicate, timeout=2):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("Condition not reached within timeout")


class InferenceWorkerTests(unittest.TestCase):
    def make_worker(self, reader=None, detector=None, **kwargs):
        reader = reader if reader is not None else FakeReader()
        detector = detector if detector is not None else FakeDetector()
        worker = InferenceWorker(reader, detector, **kwargs)
        self.addCleanup(self.cleanup_worker, worker, detector)
        return reader, detector, worker

    @staticmethod
    def cleanup_worker(worker, detector):
        if isinstance(detector, FakeDetector):
            detector.release.set()
        worker.stop(timeout=1)

    def test_construct_and_infer_without_cv2_or_torch(self):
        with patch.dict(sys.modules, {"cv2": None, "torch": None}):
            reader, detector, worker = self.make_worker()
            reader.publish(1)
            worker.start()
            wait_until(lambda: worker.snapshot().result is not None)
        snapshot = worker.snapshot()
        result = snapshot.result
        self.assertEqual(snapshot.state, WorkerState.RUNNING)
        self.assertIs(result.detections, detector.detections)
        self.assertEqual(result.source_sequence, 1)
        self.assertEqual(result.source_received_at, reader.snapshot().frame.received_at)
        self.assertEqual(result.source_shape, (24, 32, 3))
        self.assertGreaterEqual(result.completed_at, result.source_received_at)
        self.assertGreaterEqual(result.inference_ms, 0)
        self.assertEqual(snapshot.processed_count, 1)
        with self.assertRaises(FrozenInstanceError):
            result.source_sequence = 2
        with self.assertRaises(FrozenInstanceError):
            snapshot.state = WorkerState.ERROR
        self.assertTrue(detector.threads[0].daemon)
        self.assertNotEqual(detector.threads[0], threading.current_thread())

    def test_slow_inference_only_takes_latest_frame_without_blocking_consumer(self):
        reader, detector, worker = self.make_worker(detector=FakeDetector(block=True), max_fps=1000)
        reader.publish(1)
        worker.start()
        self.assertTrue(detector.entered.wait(1))
        started = time.perf_counter()
        for sequence in range(2, 101):
            reader.publish(sequence)
            self.assertIsNone(worker.snapshot().result)
        self.assertLess(time.perf_counter() - started, 0.5)
        self.assertEqual(detector.calls, [1])
        detector.release.set()
        wait_until(lambda: (result := worker.snapshot().result) is not None and result.source_sequence == 100)
        self.assertEqual(worker.snapshot().processed_count, 2)
        self.assertEqual(detector.calls, [1, 100])
        self.assertEqual(worker.snapshot().result.source_sequence, 100)
        self.assertEqual(len(set(detector.threads)), 1)

    def test_does_not_infer_the_same_sequence_twice(self):
        reader, detector, worker = self.make_worker(max_fps=1000)
        reader.publish(5)
        worker.start()
        wait_until(lambda: worker.snapshot().result is not None)
        # Even replacing the image/timestamp must not cause duplicate inference.
        reader.publish(5)
        time.sleep(0.04)
        self.assertEqual(detector.calls, [5])
        reader.publish(6)
        wait_until(lambda: worker.snapshot().processed_count == 2)
        self.assertEqual(detector.calls, [5, 6])

    def test_stale_or_missing_source_is_not_inferred(self):
        reader, detector, worker = self.make_worker(stale_seconds=0.1)
        reader.publish(1, received_at=time.perf_counter() - 1)
        worker.start()
        time.sleep(0.03)
        self.assertEqual(worker.snapshot().state, WorkerState.WAITING)
        self.assertEqual(detector.calls, [])
        reader.set_state(StreamState.STREAMING)
        time.sleep(0.02)
        self.assertIsNone(worker.snapshot().result)
        self.assertEqual(detector.calls, [])
        reader.publish(2)
        wait_until(lambda: worker.snapshot().result is not None)

    def test_snapshot_immediately_filters_disconnect_missing_and_stale_sources(self):
        for invalid in ("disconnect", "missing", "stale"):
            with self.subTest(invalid=invalid):
                reader, _, worker = self.make_worker(max_fps=0.1)
                reader.publish(1)
                worker.start()
                wait_until(lambda: worker.snapshot().result is not None)
                if invalid == "disconnect":
                    reader.set_state(StreamState.RECONNECTING, keep_frame=True)
                elif invalid == "missing":
                    reader.set_state(StreamState.STREAMING)
                else:
                    reader.publish(2, received_at=time.perf_counter() - 10)
                self.assertIsNone(worker.snapshot().result)
                self.assertEqual(worker.snapshot().state, WorkerState.WAITING)
                self.assertTrue(worker.stop())

    def test_result_age_uses_source_arrival_not_completion_or_latest_frame(self):
        reader, detector, worker = self.make_worker(
            max_fps=0.1, stale_seconds=5, max_result_age_seconds=0.06,
        )
        source_time = time.perf_counter()
        reader.publish(1, received_at=source_time)
        worker.start()
        wait_until(lambda: worker.snapshot().result is not None)
        while time.perf_counter() - source_time <= 0.08:
            reader.publish(2)
            time.sleep(0.002)
        self.assertIsNone(worker.snapshot().result)
        self.assertEqual(detector.calls, [1])

    def test_rechecks_source_after_inference_even_without_consumer_polling(self):
        reader, detector, worker = self.make_worker(detector=FakeDetector(block=True))
        reader.publish(1)
        worker.start()
        self.assertTrue(detector.entered.wait(1))
        reader.set_state(StreamState.RECONNECTING)
        detector.release.set()
        wait_until(lambda: worker.snapshot().processed_count == 1)
        self.assertIsNone(worker.snapshot().result)
        self.assertEqual(worker.snapshot().state, WorkerState.WAITING)

    def test_discards_result_which_expires_during_inference(self):
        reader, detector, worker = self.make_worker(
            detector=FakeDetector(block=True), stale_seconds=1, max_result_age_seconds=0.02,
        )
        reader.publish(1)
        worker.start()
        self.assertTrue(detector.entered.wait(1))
        time.sleep(0.04)
        detector.release.set()
        wait_until(lambda: worker.snapshot().processed_count == 1)
        self.assertIsNone(worker.snapshot().result)

    def test_observed_disconnect_invalidates_inflight_work_after_reconnect(self):
        reader, detector, worker = self.make_worker(detector=FakeDetector(block=True), max_fps=0.1)
        reader.publish(1)
        worker.start()
        self.assertTrue(detector.entered.wait(1))
        reader.set_state(StreamState.RECONNECTING)
        self.assertIsNone(worker.snapshot().result)
        reader.publish(2)
        detector.release.set()
        wait_until(lambda: worker.snapshot().processed_count == 1)
        self.assertIsNone(worker.snapshot().result)

    def test_exception_is_terminal_sanitized_and_clears_prior_result(self):
        class BrokenDetector(FakeDetector):
            def predict(self, image):
                if image.sequence == 2:
                    raise RuntimeError("rtsp://admin:TOP_SECRET@camera/live")
                return super().predict(image)

        reader, _, worker = self.make_worker(detector=BrokenDetector(), max_fps=1000)
        reader.publish(1)
        worker.start()
        wait_until(lambda: worker.snapshot().result is not None)
        reader.publish(2)
        wait_until(lambda: worker.snapshot().state == WorkerState.ERROR)
        self.assertIsNone(worker.snapshot().result)
        self.assertNotIn("TOP_SECRET", repr(worker.snapshot()))
        self.assertNotIn("rtsp", worker.snapshot().error)
        self.assertEqual(worker.snapshot().processed_count, 1)
        self.assertTrue(worker.stop())
        self.assertEqual(worker.snapshot().state, WorkerState.ERROR)

    def test_reader_exception_is_sanitized(self):
        class BrokenReader:
            def snapshot(self):
                raise RuntimeError("TOP_SECRET")

        _, detector, worker = self.make_worker(reader=BrokenReader())
        worker.start()
        wait_until(lambda: worker.snapshot().state == WorkerState.ERROR)
        self.assertNotIn("TOP_SECRET", repr(worker.snapshot()))
        self.assertEqual(detector.calls, [])
        self.assertTrue(worker.stop())

    def test_eof_is_normal_terminal_state_not_error(self):
        reader, detector, worker = self.make_worker()
        reader.set_state(StreamState.EOF, error="Hết video")
        worker.start()
        wait_until(lambda: worker.snapshot().state == WorkerState.EOF)
        self.assertIsNone(worker.snapshot().error)
        self.assertIsNone(worker.snapshot().result)
        self.assertEqual(detector.calls, [])
        self.assertTrue(worker.stop())
        self.assertEqual(worker.snapshot().state, WorkerState.EOF)

    def test_camera_error_is_terminal_and_sanitized(self):
        reader, _, worker = self.make_worker()
        reader.set_state(StreamState.ERROR, error="TOP_SECRET")
        worker.start()
        wait_until(lambda: worker.snapshot().state == WorkerState.ERROR)
        self.assertIsNotNone(worker.snapshot().error)
        self.assertNotIn("TOP_SECRET", repr(worker.snapshot()))
        self.assertTrue(worker.stop())

    def test_bounded_stop_during_blocked_inference_never_publishes_late_result(self):
        reader, detector, worker = self.make_worker(detector=FakeDetector(block=True))
        reader.publish(1)
        worker.start()
        self.assertTrue(detector.entered.wait(1))
        started = time.perf_counter()
        self.assertFalse(worker.stop(timeout=0.01))
        self.assertLess(time.perf_counter() - started, 0.5)
        self.assertEqual(worker.snapshot().state, WorkerState.STOPPED)
        self.assertIsNone(worker.snapshot().result)
        detector.release.set()
        self.assertTrue(worker.stop(timeout=1))
        self.assertIsNone(worker.snapshot().result)

    def test_rate_limits_starts_and_takes_latest_after_wait(self):
        reader, detector, worker = self.make_worker(max_fps=10, stale_seconds=2)
        reader.publish(1)
        worker.start()
        wait_until(lambda: worker.snapshot().processed_count == 1)
        reader.publish(2)
        time.sleep(0.02)
        self.assertEqual(detector.calls, [1])
        reader.publish(3)
        wait_until(lambda: worker.snapshot().processed_count == 2)
        reader.publish(4)
        wait_until(lambda: worker.snapshot().processed_count == 3)
        self.assertEqual(detector.calls, [1, 3, 4])
        for previous, following in zip(detector.times, detector.times[1:]):
            self.assertGreaterEqual(following - previous, 0.099)
        self.assertGreater(worker.snapshot().inference_fps, 0)
        self.assertLessEqual(worker.snapshot().inference_fps, 10.2)

    def test_stop_interrupts_long_rate_limit_wait(self):
        reader, _, worker = self.make_worker(max_fps=0.01)
        reader.publish(1)
        worker.start()
        wait_until(lambda: worker.snapshot().processed_count == 1)
        reader.publish(2)
        self.assertTrue(worker.stop(timeout=0.2))
        self.assertEqual(worker.snapshot().state, WorkerState.STOPPED)
        self.assertIsNone(worker.snapshot().result)

    def test_start_once_and_stop_before_start(self):
        _, _, worker = self.make_worker()
        worker.start()
        with self.assertRaises(RuntimeError):
            worker.start()
        self.assertTrue(worker.stop())
        with self.assertRaises(RuntimeError):
            worker.start()
        _, _, never_started = self.make_worker()
        self.assertEqual(never_started.snapshot().state, WorkerState.STOPPED)
        self.assertTrue(never_started.stop())
        with self.assertRaises(RuntimeError):
            never_started.start()

    def test_rejects_invalid_limits_and_unbounded_stop(self):
        for key in ("max_fps", "stale_seconds", "max_result_age_seconds"):
            for value in (0, -1, float("inf"), float("nan")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    InferenceWorker(FakeReader(), FakeDetector(), **{key: value})
        _, _, worker = self.make_worker()
        for value in (-1, float("inf"), float("nan")):
            with self.subTest(timeout=value), self.assertRaises(ValueError):
                worker.stop(value)


if __name__ == "__main__":
    unittest.main()
