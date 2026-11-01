"""Real codec integration test, skipped explicitly when OpenCV is unavailable."""

import importlib.util
from pathlib import Path
import tempfile
import time
import unittest

from camera.rtsp_reader import LatestFrameReader, StreamState
from config import Settings


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV chưa được cài")
class VideoIntegrationTests(unittest.TestCase):
    def test_generated_video_opens_and_reaches_eof(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (160, 120))
            self.assertTrue(writer.isOpened(), "MJPG encoder unavailable")
            try:
                for index in range(10):
                    writer.write(np.full((120, 160, 3), index * 20, dtype=np.uint8))
            finally:
                writer.release()
            reader = LatestFrameReader(Settings(str(path), True)).start()
            saw_frame = False
            try:
                deadline = time.perf_counter() + 5
                while time.perf_counter() < deadline:
                    snapshot = reader.snapshot()
                    if snapshot.frame:
                        saw_frame = True
                        self.assertEqual(snapshot.frame.image.shape, (120, 160, 3))
                    if snapshot.state in {StreamState.EOF, StreamState.ERROR}:
                        break
                    time.sleep(0.005)
                self.assertTrue(saw_frame)
                self.assertEqual(reader.snapshot().state, StreamState.EOF)
                self.assertEqual(reader.snapshot().frames_received, 10)
            finally:
                self.assertTrue(reader.stop())


if __name__ == "__main__":
    unittest.main()
