"""Offline sampling must not depend on detector speed or wall-clock age."""

from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ai.inference_worker import InferenceWorker
from camera.rtsp_reader import StreamState
from camera.video_reader import VideoAnalysisReader
from config import Settings


@dataclass(frozen=True)
class Image:
    index: int
    size: int = 1
    shape: tuple = (100, 200, 3)


class Capture:
    def __init__(self):
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return 30.0

    def grab(self):
        if self.index == 30:
            return False
        self.index += 1
        return True

    def retrieve(self):
        return True, Image(self.index - 1)

    def release(self):
        self.released = True


class Detector:
    def __init__(self, delay=0):
        self.delay = delay
        self.calls = []

    def predict(self, image):
        self.calls.append(image.index)
        if self.delay:
            threading.Event().wait(self.delay)
        return ()


class VideoAnalysisTests(unittest.TestCase):
    def analyze(self, delay):
        capture = Capture()
        settings = Settings(str(Path("video.mp4")), True)
        reader = VideoAnalysisReader(settings, sample_fps=10, capture_factory=lambda _: capture)
        detector = Detector(delay)
        worker = InferenceWorker(reader, detector, max_fps=1000,
                                 stale_seconds=0.001, max_result_age_seconds=0.001)
        samples = []
        with patch.dict(sys.modules, {"cv2": SimpleNamespace(CAP_PROP_FPS=5)}):
            reader.start()
            worker.start()
            try:
                deadline = time.perf_counter() + 5
                last_sequence = 0
                while time.perf_counter() < deadline:
                    result = worker.snapshot().result
                    if result and result.source_sequence != last_sequence:
                        samples.append(result.source_media_ms)
                        last_sequence = result.source_sequence
                        reader.acknowledge(last_sequence)
                    if reader.snapshot().state == StreamState.EOF:
                        break
                    threading.Event().wait(0.002)
                else:
                    self.fail("Offline analysis stalled")
            finally:
                reader.stop()
                worker.stop()
        self.assertTrue(capture.released)
        return samples, detector.calls

    def test_sampling_and_last_result_survive_slow_inference(self):
        fast = self.analyze(0)
        slow = self.analyze(0.02)
        self.assertEqual(fast, slow)
        self.assertEqual(fast[1], list(range(0, 30, 3)))
        self.assertEqual(len(fast[0]), 10)
        self.assertAlmostEqual(fast[0][-1], 900)

    def test_decode_ahead_is_bounded_and_acknowledgement_is_exact(self):
        reader = VideoAnalysisReader(Settings("fake", True), capture_factory=lambda _: Capture())
        with patch.dict(sys.modules, {"cv2": SimpleNamespace(CAP_PROP_FPS=5)}):
            reader.start()
            try:
                deadline = time.perf_counter() + 1
                while reader.snapshot().frames_received < 4 and time.perf_counter() < deadline:
                    threading.Event().wait(0.002)
                snapshot = reader.snapshot()
                self.assertEqual(snapshot.frames_received, 4)
                self.assertEqual(snapshot.frame.sequence, 1)
                reader.acknowledge(99)
                self.assertEqual(reader.snapshot().frame.sequence, 1)
                reader.acknowledge(1)
                self.assertEqual(reader.snapshot().frame.sequence, 2)
            finally:
                reader.stop()

    def test_stop_unblocks_unacknowledged_frame(self):
        reader = VideoAnalysisReader(Settings("fake", True), capture_factory=lambda _: Capture())
        with patch.dict(sys.modules, {"cv2": SimpleNamespace(CAP_PROP_FPS=5)}):
            reader.start()
            self.assertTrue(reader.stop(timeout=1))
        self.assertEqual(reader.snapshot().state, StreamState.STOPPED)


if __name__ == "__main__":
    unittest.main()
