from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ai.inference_worker import InferenceWorker
from camera.rtsp_reader import LatestFrameReader
from camera.video_reader import VideoAnalysisReader
from config import Settings


class StartupCleanupTests(unittest.TestCase):
    def test_real_components_stop_safely_after_thread_start_fails(self):
        settings = Settings("unused-test-source", False, log_dir=Path("unused-test-logs"))
        components = (
            LatestFrameReader(settings),
            VideoAnalysisReader(settings),
            InferenceWorker(SimpleNamespace(offline=False), SimpleNamespace()),
        )
        for component in components:
            with self.subTest(component=type(component).__name__):
                with patch("threading.Thread.start", side_effect=RuntimeError("test-start-failure")):
                    with self.assertRaises(RuntimeError):
                        component.start()
                self.assertTrue(component.stop())
                self.assertTrue(component.stop())


if __name__ == "__main__":
    unittest.main()
