from types import SimpleNamespace
import unittest

from utils.posture_log import PostureReporter


def result(sequence, *labels, timestamp=None):
    return SimpleNamespace(source_sequence=sequence,
                           source_received_at=sequence * 0.1 if timestamp is None else timestamp,
                           detections=tuple(SimpleNamespace(label=label) for label in labels))


class PostureReporterTests(unittest.TestCase):
    def feed(self, reporter, sequence, *labels):
        return reporter.update(result(sequence, *labels), now=sequence * 0.1)

    def stabilize(self, reporter, start, *labels):
        outputs = [self.feed(reporter, sequence, *labels) for sequence in range(start, start + 8)]
        return [message for message in outputs if message]

    def test_requires_stable_distinct_results(self):
        reporter = PostureReporter()
        self.assertIsNone(self.feed(reporter, 1, "standing"))
        self.assertIsNone(self.feed(reporter, 1, "standing"))
        self.assertEqual(self.stabilize(reporter, 2, "standing"), ["Đang đứng"])
        self.assertEqual(self.stabilize(reporter, 10, "standing"), [])
        self.assertEqual(self.stabilize(reporter, 18, "sitting"), ["Đang ngồi"])

    def test_alternating_labels_are_not_reported(self):
        reporter = PostureReporter()
        for sequence in range(1, 40):
            label = "sitting" if sequence % 2 else "standing"
            self.assertIsNone(self.feed(reporter, sequence, label))

    def test_fallen_is_not_confirmed(self):
        messages = self.stabilize(PostureReporter(), 1, "fallen")
        self.assertEqual(len(messages), 1)
        self.assertIn("chưa xác nhận ngã", messages[0])
        self.assertNotIn("Đã ngã", messages[0])

    def test_brief_dropout_does_not_reprint(self):
        reporter = PostureReporter()
        self.stabilize(reporter, 1, "standing")
        self.assertIsNone(self.feed(reporter, 9))
        self.assertIsNone(reporter.update(None, now=0.95))
        self.assertEqual(self.stabilize(reporter, 10, "standing"), [])

    def test_long_missing_source_allows_reacquisition(self):
        reporter = PostureReporter()
        self.stabilize(reporter, 1, "standing")
        reporter.update(None, now=2)
        self.assertEqual(self.stabilize(reporter, 21, "standing"), ["Đang đứng"])

    def test_multiple_boxes_are_not_claimed_as_people(self):
        reporter = PostureReporter()
        message = self.stabilize(reporter, 1, "standing", "sitting", "standing")[0]
        self.assertIn("Đang đứng: 2 vùng nhận diện", message)
        self.assertIn("Đang ngồi: 1 vùng nhận diện", message)
        self.assertNotIn("người", message)

    def test_transient_duplicate_does_not_change_log(self):
        reporter = PostureReporter()
        self.stabilize(reporter, 1, "standing")
        self.assertIsNone(self.feed(reporter, 9, "standing", "standing"))
        self.assertEqual(self.stabilize(reporter, 10, "standing"), [])

    def test_repeated_sample_cannot_satisfy_minimum_samples(self):
        reporter = PostureReporter()
        self.feed(reporter, 1, "standing")
        self.assertIsNone(reporter.update(result(1, "standing"), now=0.8))
        self.assertIsNone(self.feed(reporter, 9, "standing"))
        self.assertEqual(self.feed(reporter, 10, "standing"), "Đang đứng")

    def test_unknown_label_is_not_injected_into_log(self):
        reporter = PostureReporter()
        self.assertEqual(self.stabilize(reporter, 1, "UNKNOWN\nsecret"), [])

    def test_invalid_thresholds(self):
        for options in ({"stable_seconds": 0}, {"missing_seconds": float("nan")}, {"min_samples": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                PostureReporter(**options)


if __name__ == "__main__":
    unittest.main()
