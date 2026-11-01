from types import SimpleNamespace
import unittest

from utils.pipeline_progress import InferenceProgress


def source(sequence=1, *, state="STREAMING", received_at=0.):
    return SimpleNamespace(state=state, frame=SimpleNamespace(sequence=sequence, received_at=received_at))


def worker(sequence=None, *, count=0, state="RUNNING"):
    result = None if sequence is None else SimpleNamespace(source_sequence=sequence)
    return SimpleNamespace(state=state, result=result, processed_count=count)


class InferenceProgressTests(unittest.TestCase):
    def check(self, progress, now, *, packet=None, status=None, offline=True):
        return progress.stalled(packet or source(), status or worker(), now=now,
                                stale_seconds=1., offline=offline)

    def test_no_first_result_reaches_deadline_even_with_new_camera_frames(self):
        progress = InferenceProgress(timeout_seconds=2.)
        self.assertFalse(self.check(progress, 0.))
        self.assertFalse(self.check(progress, 1., packet=source(10)))
        self.assertTrue(self.check(progress, 2., packet=source(20)))

    def test_repeated_cached_output_and_increasing_count_do_not_reset_deadline(self):
        progress = InferenceProgress(timeout_seconds=2.)
        self.assertFalse(self.check(progress, 0., status=worker(1)))
        self.assertFalse(self.check(progress, 1., status=worker(1, count=10)))
        self.assertTrue(self.check(progress, 2., status=worker(1, count=20)))

    def test_advancing_published_sequence_resets_deadline(self):
        progress = InferenceProgress(timeout_seconds=2.)
        self.assertFalse(self.check(progress, 0., status=worker(1)))
        self.assertFalse(self.check(progress, 1., status=worker(2)))
        self.assertFalse(self.check(progress, 2.9, status=worker(2)))
        self.assertTrue(self.check(progress, 3., status=worker(2)))

    def test_source_outage_resets_grace_but_old_result_cannot_count_again(self):
        progress = InferenceProgress(timeout_seconds=2.)
        self.check(progress, 0., status=worker(2))
        self.assertFalse(self.check(progress, 10., packet=source(state="RECONNECTING")))
        self.assertFalse(self.check(progress, 20., status=worker(2)))
        self.assertTrue(self.check(progress, 22., status=worker(2)))

    def test_live_stale_or_future_source_is_not_inference_hang(self):
        progress = InferenceProgress(timeout_seconds=2.)
        for now in (0., 10., 20.):
            self.assertFalse(self.check(progress, now, packet=source(received_at=now + 1), offline=False))
        self.assertFalse(self.check(progress, 30., packet=source(received_at=0.), offline=False))
        self.assertFalse(self.check(progress, 31., packet=source(received_at=31.), offline=False))
        self.assertTrue(self.check(progress, 33., packet=source(received_at=33.), offline=False))

    def test_terminal_workers_do_not_raise_inference_progress_alarm(self):
        progress = InferenceProgress(timeout_seconds=2.)
        self.check(progress, 0.)
        for index, state in enumerate(("ERROR", "EOF", "STOPPED"), 10):
            self.assertFalse(self.check(progress, float(index), status=worker(state=state)))

    def test_actual_blocked_synthetic_worker_is_detected_without_models(self):
        from ai.inference_worker import InferenceWorker
        from tests.test_ai_worker import FakeReader, FakeDetector

        reader = FakeReader()
        reader.offline = True
        reader.publish(1)
        detector = FakeDetector(block=True)
        actual_worker = InferenceWorker(reader, detector).start()
        try:
            self.assertTrue(detector.entered.wait(1))
            progress = InferenceProgress(timeout_seconds=2.)
            for now, expected in ((0., False), (2., True)):
                self.assertEqual(progress.stalled(reader.snapshot(), actual_worker.snapshot(), now=now,
                                                 stale_seconds=1., offline=True), expected)
            self.assertFalse(actual_worker.stop(timeout=0.))
        finally:
            detector.release.set()
            self.assertTrue(actual_worker.stop(timeout=1.))

    def test_invalid_timeout_and_clock_rejected(self):
        for value in (True, 0, -1, float("nan"), float("inf"), "2"):
            with self.assertRaises(ValueError):
                InferenceProgress(timeout_seconds=value)
        progress = InferenceProgress()
        self.check(progress, 2.)
        for value in (1., True, -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.check(progress, value)


if __name__ == "__main__":
    unittest.main()
