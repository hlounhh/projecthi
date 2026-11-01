from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from camera.rtsp_reader import StreamState
from config import Settings
import main


class MainCleanupSafetyTests(unittest.TestCase):
    def test_partial_pipeline_start_failure_keeps_resources_for_cleanup(self):
        for kind in ("live_reader", "video_reader", "worker"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                log_dir = Path(directory)
                offline = kind == "video_reader"
                settings = Settings("unused-test-source", offline, log_dir=log_dir)
                reader = Mock()
                reader.start.return_value = reader
                reader.stop.return_value = True
                reader.snapshot.return_value = SimpleNamespace(state=StreamState.CONNECTING, frame=None)
                worker = Mock()
                worker.stop.return_value = True
                target = worker if kind == "worker" else reader
                target.start.side_effect = RuntimeError("private-partial-start")
                argv = ["main.py", "--ai", "--headless"]
                if offline:
                    argv += ["--video", "unused-test-video"]
                stderr = io.StringIO()
                with patch("sys.argv", argv), \
                        patch("main.load_settings", return_value=settings), \
                        patch("camera.rtsp_reader.LatestFrameReader", return_value=reader), \
                        patch("camera.video_reader.VideoAnalysisReader", return_value=reader), \
                        patch("ai.fusion_detector.FusionDetector"), \
                        patch("ai.inference_worker.InferenceWorker", return_value=worker), \
                        patch("utils.metrics.setup_logging", return_value=Mock()) as logging, \
                        patch("utils.metrics.MetricsWriter") as metrics, \
                        redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                    self.assertEqual(main.run(), 1)
                reader.stop.assert_called_once()
                if kind == "worker":
                    worker.stop.assert_called_once_with(timeout=5)
                else:
                    worker.stop.assert_not_called()
                metrics.return_value.close.assert_called_once()
                logging.return_value.close.assert_called_once()
                self.assertNotIn("private-partial-start", stderr.getvalue())

    def test_metrics_startup_failure_closes_real_logger(self):
        from utils.metrics import setup_logging

        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            settings = Settings("unused-test-source", False, log_dir=log_dir)
            streams = []

            def start_logging(path):
                logger = setup_logging(path)
                streams.append(logger._handler.stream)
                return logger

            with patch("sys.argv", ["main.py", "--ai", "--headless"]), \
                    patch("main.load_settings", return_value=settings), \
                    patch("utils.metrics.setup_logging", side_effect=start_logging), \
                    patch("utils.metrics.MetricsWriter", side_effect=OSError("private-test-error")), \
                    patch("ai.fusion_detector.FusionDetector") as detector, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main.run(), 1)
                self.assertEqual(main.run(), 1)
            detector.assert_not_called()
            self.assertEqual(len(streams), 2)
            self.assertTrue(all(stream.closed for stream in streams))
            self.assertNotIn("private-test-error", (log_dir / "camera.log").read_text(encoding="utf-8"))

    def test_metrics_disk_failure_still_stops_workers_and_closes_journal(self):
        from utils.fall_log import FallEventWriter

        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            settings = Settings("unused-local-test-source", False, log_dir=log_dir)
            snapshot = SimpleNamespace(state=StreamState.STREAMING, frame=None, frames_received=1)
            reader = Mock()
            reader.start.return_value = reader
            reader.snapshot.return_value = snapshot
            reader.stop.return_value = True
            worker = Mock()
            worker.start.return_value = worker
            worker.snapshot.return_value = SimpleNamespace(state=None, result=None)
            worker.stop.return_value = True
            writers = []

            def writer(*args, **kwargs):
                instance = FallEventWriter(*args, **kwargs)
                writers.append(instance)
                return instance

            stderr = io.StringIO()
            with patch("sys.argv", ["main.py", "--ai", "--headless"]), \
                    patch("main.load_settings", return_value=settings), \
                    patch("camera.rtsp_reader.LatestFrameReader", return_value=reader), \
                    patch("ai.fusion_detector.FusionDetector"), \
                    patch("ai.inference_worker.InferenceWorker", return_value=worker), \
                    patch("utils.fall_log.FallEventWriter", side_effect=writer), \
                    patch("utils.metrics.setup_logging", return_value=Mock()), \
                    patch("utils.metrics.MetricsWriter") as metrics, \
                    redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                metrics.return_value.write.side_effect = OSError("secret backend message")
                self.assertEqual(main.run(), 1)
            reader.stop.assert_called_once()
            worker.stop.assert_called_once()
            metrics.return_value.close.assert_called_once()
            self.assertTrue(writers[0]._file.closed)
            self.assertIn("[metrics_persistence] (OSError)", stderr.getvalue())
            self.assertNotIn("secret backend message", stderr.getvalue())

    def test_final_log_write_failure_still_cleans_up(self):
        from utils.metrics import setup_logging

        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            settings = Settings("unused-test-source", False, log_dir=log_dir)
            reader = Mock()
            reader.start.return_value = reader
            reader.snapshot.return_value = SimpleNamespace(state=StreamState.EOF, frame=None,
                                                           frames_received=1, error=None)
            reader.stop.return_value = True
            worker = Mock()
            worker.start.return_value = worker
            worker.snapshot.return_value = SimpleNamespace(state=None, result=None)
            worker.stop.return_value = True
            streams = []

            def start_logging(path):
                logger = setup_logging(path)
                stream = logger._handler.stream
                streams.append(stream)
                original = logger.info

                def info(message, *args):
                    if message.startswith("Session cleanup"):
                        with patch.object(stream, "write", side_effect=OSError("secret-final-write")):
                            return original(message, *args)
                    return original(message, *args)

                logger.info = info
                return logger

            stderr = io.StringIO()
            with patch("sys.argv", ["main.py", "--ai", "--headless"]), \
                    patch("main.load_settings", return_value=settings), \
                    patch("camera.rtsp_reader.LatestFrameReader", return_value=reader), \
                    patch("ai.fusion_detector.FusionDetector"), \
                    patch("ai.inference_worker.InferenceWorker", return_value=worker), \
                    patch("utils.metrics.setup_logging", side_effect=start_logging), \
                    patch("utils.metrics.MetricsWriter"), \
                    redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                self.assertEqual(main.run(), 1)
            self.assertNotIn("secret-final-write", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertTrue(streams[0].closed)
            reader.stop.assert_called_once()
            worker.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
