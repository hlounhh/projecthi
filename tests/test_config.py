import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from config import ConfigError, load_settings


class ConfigTests(unittest.TestCase):
    def load(self, values, video=None):
        with patch.dict(os.environ, values, clear=True):
            return load_settings(video, env_file=Path("__missing_test_env__"))

    def test_requires_password_without_video(self):
        with self.assertRaises(ConfigError):
            self.load({})

    def test_password_is_encoded_and_hidden_in_repr(self):
        settings = self.load({"CAMERA_PASSWORD": "secret@#:/ word"})
        self.assertIn("secret%40%23%3A%2F%20word", settings.source)
        self.assertNotIn("secret", repr(settings))
        self.assertIn("subtype=1", settings.source)

    def test_custom_url_overrides_fields(self):
        url = "rtsp://user:private@localhost:8554/live"
        self.assertEqual(self.load({"RTSP_URL": url}).source, url)

    def test_explicit_stream_overrides_shell_and_custom_url(self):
        from urllib.parse import parse_qs, urlsplit

        with patch.dict(os.environ, {
            "CAMERA_SUBTYPE": "1",
            "RTSP_URL": "rtsp://user:private@localhost/live?channel=1&subtype=1&subtype=1",
        }, clear=True):
            settings = load_settings(env_file=Path("__missing_test_env__"), stream_subtype=0)
        query = parse_qs(urlsplit(settings.source).query)
        self.assertEqual(query["subtype"], ["0"])
        self.assertEqual(query["channel"], ["1"])
        self.assertNotIn("private", repr(settings))

    def test_invalid_values(self):
        for extra in (
            {"OPEN_TIMEOUT_MS": "nan"}, {"READ_TIMEOUT_MS": "1.5"},
            {"RTSP_TRANSPORT": "http"}, {"DISPLAY_WIDTH": "0"},
            {"CAMERA_HOST": "host:554"}, {"CAMERA_PORT": "70000"},
            {"RTSP_URL": "https://localhost/video"},
            {"RTSP_URL": "rtsp://localhost:invalid/video"},
        ):
            with self.subTest(extra=extra), self.assertRaises(ConfigError):
                self.load({"CAMERA_PASSWORD": "private", **extra})

    def test_file_does_not_require_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "sample.avi"
            video.touch()
            settings = self.load({}, str(video))
            self.assertTrue(settings.is_file)
            self.assertEqual(settings.source, str(video.resolve()))

    def test_missing_file(self):
        with self.assertRaises(ConfigError):
            self.load({}, "__missing_video__.avi")


if __name__ == "__main__":
    unittest.main()
