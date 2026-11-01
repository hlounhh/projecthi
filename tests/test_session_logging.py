import logging
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.metrics import model_logs_to_file, setup_logging


class SessionLoggingTests(unittest.TestCase):
    def test_model_details_go_to_file_and_warnings_remain_visible(self):
        model_logger = logging.getLogger("ultralytics")
        stdout = io.StringIO()
        console = logging.StreamHandler(stdout)
        with tempfile.TemporaryDirectory() as directory:
            logger = setup_logging(Path(directory))
            try:
                with patch.object(model_logger, "handlers", [console]), \
                        patch.object(model_logger, "level", logging.INFO), \
                        patch.object(model_logger, "propagate", False), \
                        model_logs_to_file(logger):
                    model_logger.info("Loading model on CPU")
                    model_logger.warning("Model warning")
                logger.check_health()
            finally:
                logger.close()
            self.assertEqual(stdout.getvalue(), "Model warning\n")
            contents = (Path(directory) / "camera.log").read_text(encoding="utf-8")
            self.assertIn("Loading model on CPU", contents)
            self.assertIn("Model warning", contents)

    def test_model_log_filter_is_removed_after_loading_failure(self):
        model_logger = logging.getLogger("ultralytics")
        previous_filters = model_logger.filters[:]
        with self.assertRaises(RuntimeError):
            with model_logs_to_file(unittest.mock.Mock()):
                raise RuntimeError("test loading failure")
        self.assertEqual(model_logger.filters, previous_filters)

    def test_sequential_sessions_use_new_directory_and_detach_camera_handler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = logging.getLogger("camera")
            previous_handlers = camera.handlers[:]
            previous_level = camera.level
            first = setup_logging(root / "first")
            handler = first._handler
            stream = handler.stream
            try:
                first.info("first-session")
                logging.getLogger("camera.rtsp_reader").info("first-camera")
            finally:
                first.close()
            first.close()
            self.assertTrue(stream.closed)
            self.assertEqual(camera.handlers, previous_handlers)
            self.assertEqual(camera.level, previous_level)
            second = setup_logging(root / "second")
            try:
                second.info("second-session")
                logging.getLogger("camera.rtsp_reader").info("second-camera")
            finally:
                second.close()
            before = (root / "first" / "camera.log").read_text(encoding="utf-8")
            after = (root / "second" / "camera.log").read_text(encoding="utf-8")
            self.assertIn("first-camera", before)
            self.assertNotIn("second-session", before)
            self.assertIn("second-camera", after)
            self.assertNotIn("first-session", after)

    def test_unrelated_handlers_are_preserved_and_not_owned(self):
        camera = logging.getLogger("camera")
        unrelated = logging.NullHandler()
        camera.addHandler(unrelated)
        try:
            with tempfile.TemporaryDirectory() as directory:
                logger = setup_logging(Path(directory))
                logger.close()
            self.assertIn(unrelated, camera.handlers)
        finally:
            camera.removeHandler(unrelated)

    def test_overlapping_session_is_refused_without_creating_new_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = setup_logging(root / "first")
            try:
                with self.assertRaises(RuntimeError):
                    setup_logging(root / "second")
                self.assertFalse((root / "second").exists())
                first.info("still-owned")
            finally:
                first.close()

    def test_failed_file_open_does_not_attach_handler(self):
        camera = logging.getLogger("camera")
        before = camera.handlers[:]
        with tempfile.TemporaryDirectory() as directory, \
                patch("utils.metrics.SessionFileHandler", side_effect=OSError("test open failure")):
            with self.assertRaises(OSError):
                setup_logging(Path(directory))
        self.assertEqual(camera.handlers, before)

    def test_worker_emit_failure_latches_without_raw_traceback_or_thread_exception(self):
        from contextlib import redirect_stderr
        import io
        import threading
        from utils.metrics import SessionLogUnavailable

        with tempfile.TemporaryDirectory() as directory:
            logger = setup_logging(Path(directory))
            stream = logger._handler.stream
            stderr = io.StringIO()
            try:
                with patch.object(stream, "write", side_effect=OSError("rtsp://secret-backend")), \
                        patch("logging.raiseExceptions", True), redirect_stderr(stderr):
                    thread = threading.Thread(target=lambda: logging.getLogger("camera.rtsp_reader").warning("test-record"))
                    thread.start()
                    thread.join(timeout=2.)
                    self.assertFalse(thread.is_alive())
                    with self.assertRaises(SessionLogUnavailable):
                        logger.check_health()
                self.assertEqual(stderr.getvalue(), "")
                with self.assertRaises(SessionLogUnavailable):
                    logger.close()
                self.assertTrue(stream.closed)
                logger.close()
            finally:
                logger.close()

    def test_final_emit_failure_is_reported_by_close_after_detaching(self):
        from utils.metrics import SessionLogUnavailable

        with tempfile.TemporaryDirectory() as directory:
            logger = setup_logging(Path(directory))
            handler = logger._handler
            stream = handler.stream
            with patch.object(stream, "write", side_effect=OSError("secret")):
                logger.info("final-test-message")
            with self.assertRaises(SessionLogUnavailable):
                logger.close()
            self.assertTrue(stream.closed)
            self.assertNotIn(handler, logging.getLogger("camera").handlers)

    def test_flush_failure_still_closes_and_detaches(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = setup_logging(Path(directory))
            handler = logger._handler
            stream = handler.stream
            with patch.object(handler, "flush", side_effect=OSError("test flush failure")):
                with self.assertRaises(OSError):
                    logger.close()
            self.assertTrue(stream.closed)
            self.assertNotIn(handler, logging.getLogger("camera").handlers)
            logger.close()


if __name__ == "__main__":
    unittest.main()
