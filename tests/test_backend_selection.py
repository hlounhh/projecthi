"""Backend selection must be explicit and must not fall back after failure."""

import hashlib
import json
from unittest.mock import patch

import pytest

from ai.fusion_detector import FusionDetector
from ai.openvino_detector import verify_bundle
from ai.pose_detector import POSE_SHA256


def test_native_defaults_do_not_initialize_optional_backends():
    with patch("ai.pose_detector.PoseDetector") as pose, \
            patch("ai.posture_detector.PostureDetector") as posture, \
            patch("ai.rtmpose_detector.RTMPoseDetector") as rtmpose, \
            patch("ai.openvino_detector.OpenVINOPoseDetector") as exported:
        detector = FusionDetector()
        assert detector.pose is pose.return_value
        assert detector.posture is posture.return_value
        rtmpose.assert_not_called()
        exported.assert_not_called()


def test_rtmpose_and_openvino_dispatch_preserve_thresholds_and_cadence():
    with patch("ai.rtmpose_detector.RTMPoseDetector") as pose, \
            patch("ai.openvino_detector.OpenVINOPostureDetector") as posture:
        detector = FusionDetector(confidence=0.4, image_size=416, threads=3,
                                  posture_every=2, pose_backend="rtmpose",
                                  posture_backend="openvino", posture_model_dir="posture")
        pose.assert_called_once_with(confidence=0.2, image_size=416, threads=3)
        posture.assert_called_once_with("posture", confidence=0.4, image_size=416, threads=3)
        assert detector.posture_every == 2
    with patch("ai.openvino_detector.OpenVINOPoseDetector", side_effect=ValueError("invalid bundle")), \
            patch("ai.pose_detector.PoseDetector") as fallback:
        with pytest.raises(ValueError, match="invalid bundle"):
            FusionDetector(pose_backend="openvino", pose_model_dir="pose")
        fallback.assert_not_called()


@pytest.mark.parametrize("kwargs", [
    {"pose_backend": "unknown"}, {"posture_backend": "unknown"},
    {"pose_model_dir": "unused"}, {"posture_model_dir": "unused"},
])
def test_invalid_selection_fails_before_loading_models(kwargs):
    with patch("ai.pose_detector.PoseDetector") as model:
        with pytest.raises(ValueError):
            FusionDetector(**kwargs)
        model.assert_not_called()


def make_bundle(tmp_path):
    directory = tmp_path / "fp32_openvino_model"
    directory.mkdir()
    files = {}
    for name in ("model.xml", "model.bin", "metadata.yaml"):
        content = name.encode()
        (directory / name).write_bytes(content)
        files[name] = hashlib.sha256(content).hexdigest()
    manifest = {"schema_version": 1, "task": "pose", "image_size": 416,
                "source_sha256": POSE_SHA256, "files": files}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory, manifest


def test_export_manifest_covers_bytes_task_source_size_and_containment(tmp_path):
    directory, manifest = make_bundle(tmp_path)
    assert verify_bundle(directory, "pose", 416) == directory.resolve()
    for task, size in (("detect", 416), ("pose", 640)):
        with pytest.raises(ValueError):
            verify_bundle(directory, task, size)
    (directory / "model.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(directory, "pose", 416)
    (directory / "model.bin").write_bytes(b"model.bin")
    manifest["files"]["../outside"] = hashlib.sha256(b"outside").hexdigest()
    (tmp_path / "outside").write_bytes(b"outside")
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="path"):
        verify_bundle(directory, "pose", 416)