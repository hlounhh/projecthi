import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.compare_evaluations import compare_reports, render_markdown


def video(name="video.mp4", falls=None, confirmations=(), warnings=()):
    result = {
        "video": name, "video_sha256": "a" * 64, "duration": 10.0,
        "extraction_settings": {"image_size": 416, "sample_fps": 10},
        "fall_config": {"keypoint_smoothing": False},
        "events": [{"state": "DA NGA", "timestamp": t} for t in confirmations]
                  + [{"state": "NGHI NGA", "timestamp": t} for t in warnings],
    }
    if falls is not None:
        result["reference"] = {"falls": falls, "status": "provisional_visual"}
    return result


FALLS = [{"onset": 2, "end": 5}]


def report(*videos):
    return {"videos": list(videos), "summary": {"tp": 999}}


class ComparisonTests(unittest.TestCase):
    def test_recomputes_scores_and_accepts_config_changes(self):
        baseline = report(video(falls=FALLS, confirmations=(3,)))
        candidate = copy.deepcopy(baseline)
        candidate["videos"][0]["fall_config"]["keypoint_smoothing"] = True
        result = compare_reports(baseline, candidate)
        self.assertTrue(result["passed"])
        self.assertEqual(result["baseline_summary"]["tp"], 1)
        self.assertEqual(result["config_changes"]["video.mp4"]["keypoint_smoothing"],
                         {"baseline": False, "candidate": True})

    def test_per_clip_regression_cannot_be_hidden_by_an_improvement(self):
        before = report(video("a.mp4", FALLS, (3,)), video("b.mp4", FALLS))
        after = report(video("a.mp4", FALLS), video("b.mp4", FALLS, (3,)))
        result = compare_reports(before, after)
        self.assertFalse(result["passed"])
        self.assertEqual(result["regressions"]["lost_reference_falls"], 1)
        self.assertEqual(result["candidate_summary"]["tp"], 1)

    def test_replaced_matched_interval_is_a_regression(self):
        falls = FALLS + [{"onset": 6, "end": 9}]
        result = compare_reports(report(video(falls=falls, confirmations=(3,))),
                                 report(video(falls=falls, confirmations=(7,))))
        self.assertFalse(result["passed"])
        self.assertEqual(result["videos"][0]["lost_reference_falls"], [0])

    def test_false_confirmations_warnings_and_delay_gate_separately(self):
        before = report(video(falls=FALLS, confirmations=(3,)))
        after = report(video(falls=FALLS, confirmations=(3.4, 8), warnings=(1,)))
        result = compare_reports(before, after, max_delay_increase_s=.2)
        self.assertFalse(result["passed"])
        self.assertEqual(result["regressions"]["added_false_confirmations"], 1)
        self.assertEqual(result["regressions"]["added_false_warnings"], 1)
        allowed = compare_reports(before, after, max_added_fp=1, max_added_warnings=1,
                                  max_delay_increase_s=.5)
        self.assertTrue(allowed["passed"])

    def test_mismatched_inputs_rejected(self):
        before = report(video(falls=FALLS))
        for key, value in (("video_sha256", "b" * 64), ("duration", 11),
                           ("extraction_settings", {"sample_fps": 5}),
                           ("reference", {"falls": []}), ("video", "other.mp4")):
            after = copy.deepcopy(before)
            after["videos"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare_reports(before, after)

    def test_empty_duplicate_and_invalid_events_rejected(self):
        good = report(video(falls=[]))
        cases = [report(), report(video(), video()),
                 report(video(falls=[], confirmations=(float("nan"),))),
                 report(video(falls=[], confirmations=(11,)))]
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                compare_reports(bad, good)

    def test_unscored_counts_never_become_accuracy(self):
        result = compare_reports(report(video(confirmations=(3,))),
                                 report(video(confirmations=(3, 7))))
        self.assertIsNone(result["passed"])
        self.assertEqual(result["unscored_videos"], ["video.mp4"])
        self.assertEqual(result["candidate_summary"]["clips"], 0)
        self.assertIn("unscored", render_markdown(result).lower())

    def test_partial_labels_are_explicit(self):
        before = report(video(falls=[]), video("unlabeled.mp4"))
        result = compare_reports(before, before)
        self.assertTrue(result["passed"])
        self.assertEqual(result["scored_videos"], ["video.mp4"])
        self.assertEqual(result["unscored_videos"], ["unlabeled.mp4"])

    def test_negative_or_nonfinite_tolerances_rejected(self):
        same = report(video(falls=[]))
        for kwargs in ({"max_added_fp": -1}, {"max_added_fn": True},
                       {"max_delay_increase_s": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compare_reports(same, same, **kwargs)

    def test_explicit_model_comparison_still_requires_same_sampling(self):
        before = report(video(falls=[]))
        after = copy.deepcopy(before)
        after["videos"][0]["extraction_settings"]["posture_every"] = 2
        with self.assertRaises(ValueError):
            compare_reports(before, after)
        result = compare_reports(before, after, allow_extraction_change=True)
        self.assertTrue(result["passed"])
        self.assertTrue(result["warnings"])
        self.assertEqual(result["extraction_changes"]["video.mp4"]["posture_every"]["candidate"], 2)
        after["videos"][0]["extraction_settings"]["sample_fps"] = 5
        with self.assertRaises(ValueError):
            compare_reports(before, after, allow_extraction_change=True)

    def test_cli_writes_json_and_markdown_and_returns_regression_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, candidate = root / "before.json", root / "after.json"
            baseline.write_text(json.dumps(report(video(falls=FALLS, confirmations=(3,)))),
                                encoding="utf-8")
            candidate.write_text(json.dumps(report(video(falls=FALLS))), encoding="utf-8")
            command = [sys.executable, "-m", "tools.compare_evaluations", str(baseline),
                       str(candidate), "--output", str(root / "comparison.json"),
                       "--markdown", str(root / "comparison.md")]
            process = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(process.returncode, 1, process.stderr)
            self.assertFalse(json.loads((root / "comparison.json").read_text())["passed"])
            self.assertIn("video.mp4", (root / "comparison.md").read_text())
            again = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(again.returncode, 2)


if __name__ == "__main__":
    unittest.main()