from contextlib import redirect_stderr
import io
import unittest

from utils.cleanup import cleanup_resources


class CleanupTests(unittest.TestCase):
    def test_failure_does_not_skip_remaining_resources_or_leak_exception_message(self):
        calls = []

        def broken():
            calls.append("broken")
            raise OSError("rtsp://private-secret")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            success = cleanup_resources((("file", broken), ("reader", lambda: False),
                                         ("last", lambda: calls.append("last"))))
        self.assertFalse(success)
        self.assertEqual(calls, ["broken", "last"])
        self.assertIn("OSError", stderr.getvalue())
        self.assertNotIn("private-secret", stderr.getvalue())

    def test_success(self):
        self.assertTrue(cleanup_resources((("file", lambda: None),)))


if __name__ == "__main__":
    unittest.main()
