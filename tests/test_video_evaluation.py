import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.evaluate_videos import extraction_provenance, replay


class CachedReplayTests(unittest.TestCase):
    def test_replay_uses_app_thresholds_and_event_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            data = {"video": "video.mp4", "duration": 2, "sha256": "test",
                    "settings": {"image_size": 416, "sample_fps": 10, "posture_every": 3},
                    "rows": [{"timestamp": 0, "shape": [720, 1280, 3],
                              "people": [{"xyxy": [200, 200, 400, 300], "confidence": .9,
                                          "fallen_confidence": .5}]}]}
            (cache / "video.json").write_text(json.dumps(data), encoding="utf-8")
            result = replay(cache, {"video.mp4": {"falls": []}})[0]
            self.assertEqual(len(result["suspected"]), 1)
            self.assertEqual(result["score"]["false_warnings"], 1)
            self.assertEqual(result["score"]["tp"], 0)
            self.assertEqual(result["score"]["fp"], 0)
            self.assertEqual(result["unreliable_sample_fraction"], 1)
            self.assertEqual(result["fall_config"]["fallen_label_threshold"], .4)
            self.assertEqual(result["fall_config"]["fallen_label_reset_threshold"], .2)
            overridden = replay(cache, config={"fallen_label_threshold": .6})[0]
            self.assertEqual(overridden["suspected"], [])


class ExtractionProvenanceTests(unittest.TestCase):
    def test_torch_ensemble_records_pinned_members_and_square_canvas(self):
        posture = extraction_provenance("yolo", "ensemble")["models"]["posture"]
        self.assertNotIn("source_sha256", posture)
        self.assertEqual(posture["preprocessing"], "static_square_letterbox_rect_false")
        self.assertEqual([member["name"] for member in posture["ensemble_members"]],
                         ["baseline", "fall-finetuned-v1", "fall-finetuned-v2"])

    def test_openvino_ensemble_records_parent_manifest_only(self):
        import ai.ensemble_openvino as ensemble_openvino

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ensemble_openvino.BUNDLE_FILE).write_text(json.dumps({
                "schema_version": 1, "task": "detect", "image_size": 416,
                "precision": "int8", "members": [], "preprocessing": "shared_backend"}))

            posture = extraction_provenance("yolo", "ensemble", posture_model_dir=root)["models"]["posture"]
            raw = (root / ensemble_openvino.BUNDLE_FILE).read_bytes()
            self.assertEqual(posture["manifest_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(posture["manifest"], "ensemble.json")
            self.assertEqual(posture["members"], [])
            self.assertEqual(posture["preprocessing"], "shared_backend")
            self.assertNotIn("ensemble_members", posture)


if __name__ == "__main__":
    unittest.main()
