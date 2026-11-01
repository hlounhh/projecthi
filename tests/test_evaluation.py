import unittest

from utils.evaluation import score_events, summarize


def event(timestamp, state="DA NGA"):
    return {"timestamp": timestamp, "state": state}


class EvaluationTests(unittest.TestCase):
    def test_one_match_per_fall_duplicates_are_false_alarms(self):
        score = score_events([event(4), event(3), event(1), event(7)],
                             {"falls": [{"onset": 2, "end": 5}]}, 10)
        self.assertEqual((score["tp"], score["fn"], score["fp"]), (1, 0, 3))
        self.assertEqual(score["delays_media_s"], [1])

    def test_suspicion_is_not_confirmation_and_early_warning_is_separate(self):
        score = score_events([event(1, "NGHI NGA"), event(3, "NGHI NGA")],
                             {"falls": [{"onset": 2, "end": 5}]}, 10)
        self.assertEqual((score["tp"], score["fn"], score["fp"]), (0, 1, 0))
        self.assertEqual(score["false_warnings"], 1)

    def test_negative_exposure_and_false_alarm_rate(self):
        scores = [score_events([event(3)], {"falls": []}, 30),
                  score_events([], {"falls": []}, 30)]
        result = summarize(iter(scores))
        self.assertEqual(result["negative_clip_seconds"], 60)
        self.assertEqual(result["false_confirmations_per_negative_hour"], 60)
        self.assertEqual(result["clip_error_rate"], .5)
        self.assertEqual(result["tn_clips"], 1)
        self.assertIsNone(result["recall"])

    def test_multiple_falls_and_missing_event(self):
        score = score_events([event(3)], {"falls": [{"onset": 2, "end": 4},
                                                  {"onset": 6, "end": 8}]}, 10)
        self.assertEqual((score["tp"], score["fn"]), (1, 1))

    def test_empty_evaluation_has_no_invented_accuracy(self):
        result = summarize([])
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["clip_error_rate"])
        self.assertIsNone(result["clip_error_wilson95"])

    def test_invalid_annotations_rejected(self):
        for falls in ([{"onset": 5, "end": 2}], [{"onset": -1, "end": 4}],
                      [{"onset": 2, "end": 11}], [{"onset": float("nan"), "end": 4}],
                      [{"onset": 2, "end": 5}, {"onset": 4, "end": 7}]):
            with self.subTest(falls=falls), self.assertRaises(ValueError):
                score_events([], {"falls": falls}, 10)


if __name__ == "__main__":
    unittest.main()
