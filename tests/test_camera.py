"""Deterministic capture tests: no credentials, camera, network or OpenCV needed."""

from collections import deque
from contextlib import contextmanager
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from camera.rtsp_reader import LatestFrameReader, StreamState
from config import Settings


class FakeImage:
    size = 1


class FakeCapture:
    def __init__(self, frames=(), opened=True):
        self.frames = deque(frames)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if self.frames:
            return True, self.frames.popleft()
        return False, None

    def get(self, prop):
        return 100.0 if prop == 5 else 0.0

    def release(self):
        self.released = True


class BlockingCapture(FakeCapture):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.blocked = threading.Event()
        self.resume = threading.Event()

    def read(self):
        self.calls += 1
        if self.calls <= 3:
            return True, FakeImage()
        self.blocked.set()
        self.resume.wait(2)
        return False, None


@contextmanager
def fake_cv2():
    with patch.dict(sys.modules, {"cv2": SimpleNamespace(CAP_PROP_FPS=5, CAP_PROP_POS_MSEC=0)}):
        yield


def wait_until(predicate, timeout=2):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Condition not reached within timeout")


class CameraTests(unittest.TestCase):
    def settings(self, is_file=False):
        return Settings("fake-source", is_file, reconnect_seconds=0.1)

    def test_latest_mailbox_replaces_old_frames(self):
        capture = BlockingCapture()
        with fake_cv2():
            reader = LatestFrameReader(self.settings(), capture_factory=lambda _: capture).start()
            try:
                self.assertTrue(capture.blocked.wait(1))
                snapshot = reader.snapshot()
                self.assertEqual(snapshot.frames_received, 3)
                self.assertEqual(snapshot.frame.sequence, 3)
                self.assertEqual(snapshot.state, StreamState.STREAMING)
            finally:
                capture.resume.set()
                self.assertTrue(reader.stop())
        self.assertTrue(capture.released)
        self.assertIsNone(reader.snapshot().frame)

    def test_reconnect_after_failed_open(self):
        failed = FakeCapture(opened=False)
        working = BlockingCapture()
        captures = deque([failed, working])
        with fake_cv2():
            reader = LatestFrameReader(self.settings(), capture_factory=lambda _: captures.popleft()).start()
            try:
                self.assertTrue(working.blocked.wait(1))
                self.assertEqual(reader.snapshot().connection_attempts, 2)
                self.assertTrue(failed.released)
            finally:
                working.resume.set()
                self.assertTrue(reader.stop())

    def test_file_eof_does_not_reconnect(self):
        capture = FakeCapture([FakeImage(), FakeImage()])
        with fake_cv2():
            reader = LatestFrameReader(self.settings(True), capture_factory=lambda _: capture).start()
            try:
                wait_until(lambda: reader.snapshot().state == StreamState.EOF)
                snapshot = reader.snapshot()
                self.assertEqual(snapshot.frames_received, 2)
                self.assertEqual(snapshot.connection_attempts, 1)
                self.assertIsNone(snapshot.frame)
                self.assertTrue(capture.released)
            finally:
                reader.stop()

    def test_disconnect_clears_frame_and_releases_capture(self):
        capture = FakeCapture([FakeImage()])
        settings = Settings("fake-source", False, reconnect_seconds=30)
        with fake_cv2():
            reader = LatestFrameReader(settings, capture_factory=lambda _: capture).start()
            try:
                wait_until(lambda: reader.snapshot().state == StreamState.RECONNECTING)
                self.assertIsNone(reader.snapshot().frame)
                self.assertTrue(capture.released)
                started = time.perf_counter()
                self.assertTrue(reader.stop(timeout=1))
                self.assertLess(time.perf_counter() - started, 1)
            finally:
                reader.stop()

    def test_error_messages_do_not_include_credentials(self):
        def broken(_):
            raise RuntimeError("rtsp://admin:TOP_SECRET@camera/live")

        with fake_cv2():
            reader = LatestFrameReader(self.settings(True), capture_factory=broken).start()
            try:
                wait_until(lambda: reader.snapshot().state == StreamState.ERROR)
                self.assertNotIn("TOP_SECRET", reader.snapshot().error)
            finally:
                reader.stop()

    def test_start_twice_rejected(self):
        with fake_cv2():
            reader = LatestFrameReader(self.settings(True), capture_factory=lambda _: FakeCapture()).start()
            try:
                with self.assertRaises(RuntimeError):
                    reader.start()
            finally:
                reader.stop()


if __name__ == "__main__":
    unittest.main()
