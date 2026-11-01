"""Calibration guardrails do not need optional inference libraries."""

from types import SimpleNamespace

import pytest

from tools.export_openvino import calibration_files, export_bundle


def test_int8_refuses_missing_representative_data(tmp_path):
    with pytest.raises(ValueError, match="requires --calibration-images"):
        export_bundle("pose", tmp_path / "export", int8=True)
    with pytest.raises(ValueError, match="no images"):
        calibration_files(tmp_path, 10)
    assert not (tmp_path / "export").exists()


def test_calibration_subset_spans_input_collection(tmp_path):
    for index in range(10):
        (tmp_path / f"{index:02}.jpg").touch()
    (tmp_path / "ignore.txt").touch()
    assert [p.name for p in calibration_files(tmp_path, 3)] == ["00.jpg", "04.jpg", "09.jpg"]
    assert len(calibration_files(tmp_path, 1)) == 1
    with pytest.raises(ValueError, match="positive"):
        calibration_files(tmp_path, 0)


def test_export_refuses_existing_output_before_loading_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="already exists"):
        export_bundle("pose", tmp_path)


def test_thread_limited_compilation_replaces_delegated_backend():
    from ai.openvino_detector import _replace_compilation

    class Wrapper:
        def __init__(self):
            self.backend = SimpleNamespace(ov_compiled_model="original")

        def __getattr__(self, name):
            return getattr(self.backend, name)

    wrapper = Wrapper()
    yolo = SimpleNamespace(predictor=SimpleNamespace(model=wrapper))
    _replace_compilation(yolo, "thread-limited")
    assert wrapper.backend.ov_compiled_model == "thread-limited"
    assert "ov_compiled_model" not in wrapper.__dict__