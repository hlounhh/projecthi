import csv
from pathlib import Path
import tempfile
import time
import unittest

from camera.rtsp_reader import FramePacket, StreamSnapshot, StreamState
from utils.metrics import FrameCounter, MetricsWriter


class MetricsTests(unittest.TestCase):
    def test_counts_distinct_frames_and_skips(self):
        counter = FrameCounter()
        self.assertTrue(counter.record(1, 10))
        self.assertFalse(counter.record(1, 10.1))
        self.assertTrue(counter.record(4, 10.5))
        self.assertEqual(counter.skipped, 2)
        self.assertEqual(counter.consumed, 2)
        self.assertEqual(counter.fps(10.5), 2)
        self.assertEqual(counter.fps(14), 0)

    def test_csv_records_ai_measurements_separately(self):
        from ai.inference_worker import InferenceResult, WorkerSnapshot, WorkerState

        with tempfile.TemporaryDirectory() as directory:
            metrics = MetricsWriter(Path(directory))
            received = time.perf_counter() - 0.1
            packet = FramePacket(None, 1, received, 5.0, None)
            snapshot = StreamSnapshot(StreamState.STREAMING, packet, 1, 1, 25, None)
            result = InferenceResult((), 1, received, time.perf_counter(), 70.0, (480, 640, 3))
            ai_status = WorkerSnapshot(WorkerState.RUNNING, result, 9.5, None, 12)
            metrics.write(snapshot, FrameCounter(), 1, ai_status)
            metrics.close()
            with metrics.path.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["ai_processed"], "12")
            self.assertEqual(row["ai_inference_ms"], "70.0")
            self.assertEqual(row["ai_boxes"], "0")
            self.assertEqual(row["camera_to_screen_latency_ms"], "")

    def test_csv_does_not_invent_e2e_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = MetricsWriter(Path(directory))
            packet = FramePacket(None, 1, time.perf_counter() - 2, 5.0, None)
            snapshot = StreamSnapshot(StreamState.STREAMING, packet, 1, 1, 10, None)
            metrics.write(snapshot, FrameCounter(), stale_seconds=1)
            metrics.close()
            with metrics.path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["camera_to_screen_latency_ms"], "")
            self.assertEqual(rows[0]["frame_stale"], "True")
            self.assertGreaterEqual(float(rows[0]["post_read_age_ms"]), 2000)


if __name__ == "__main__":
    unittest.main()
