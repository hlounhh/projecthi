from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from ai.fall_engine import FallEvent
from utils.fall_log import FallEventWriter


class FallLogTests(unittest.TestCase):
    def test_event_journal_preserves_source_time_and_id(self):
        event = FallEvent(2, "fall-1", "NGHI NGA", 12.3, "fallen_label",
                          (("fallen_confidence", 0.7),))
        with tempfile.TemporaryDirectory() as directory:
            writer = FallEventWriter(Path(directory))
            stdout = io.StringIO()
            try:
                with redirect_stdout(stdout):
                    writer.write(event)
            finally:
                writer.close()
            row = json.loads(writer.path.read_text(encoding="utf-8"))
            self.assertEqual(row["event_id"], "fall-1")
            self.assertEqual(row["timestamp"], 12.3)
            self.assertEqual(stdout.getvalue(), "")
            self.assertNotIn("Đang đứng", stdout.getvalue())
            self.assertNotIn("Đang ngồi", stdout.getvalue())

    def test_missing_observation_is_quiet_but_journal_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = FallEventWriter(Path(directory))
            stdout = io.StringIO()
            try:
                with redirect_stdout(stdout):
                    writer.write(FallEvent(1, None, "MAT DAU", 0, "pose_unreliable"))
                    writer.write(FallEvent(1, "fall-1", "DA NGA", .5, "algorithmic_fall_confirmed"))
                    for timestamp in (1, 2, 5, 11):
                        writer.write(FallEvent(1, "fall-1", "MAT DAU", timestamp, "not_matched"))
                    writer.write(FallEvent(1, "fall-1", "BINH THUONG", 12, "recovered_after_fall"))
            finally:
                writer.close()
            rows = [json.loads(line) for line in writer.path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 7)
            self.assertEqual(sum(row["state"] == "MAT DAU" for row in rows), 5)
            self.assertEqual(len(stdout.getvalue().splitlines()), 2)
            self.assertIn("ĐÃ NGÃ", stdout.getvalue())
            self.assertIn("ĐÃ QUAN SÁT PHỤC HỒI SAU NGÃ", stdout.getvalue())

    def test_low_pose_warning_and_recovery_are_quiet(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = FallEventWriter(Path(directory))
            stdout = io.StringIO()
            try:
                with redirect_stdout(stdout):
                    writer.write(FallEvent(1, "fall-1", "NGHI NGA", 1.2, "persistent_low_pose"))
                    writer.write(FallEvent(1, "fall-1", "BINH THUONG", 3.0, "recovered_after_fall"))
            finally:
                writer.close()
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(len(writer.path.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
