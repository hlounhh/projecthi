"""Ensemble OpenVINO export contract: shared preprocessing and manifest pinning."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ai.ensemble_openvino import OpenVINOEnsemble, letterbox, verify_ensemble_bundle


MEMBERS = (("baseline", Path("baseline.pt"), "a" * 64),
           ("fall-finetuned-v1", Path("v1.pt"), "b" * 64),
           ("fall-finetuned-v2", Path("v2.pt"), "c" * 64))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_member(root, name, source_sha256, *, precision="openvino_int8", image_size=416,
                 names=None):
    folder = root / name / ("int8_openvino_model" if precision == "openvino_int8"
                            else f"{precision.split('_')[-1]}_openvino_model")
    folder.mkdir(parents=True)
    names = {"0": "fallen", "1": "sitting", "2": "standing"} if names is None else names
    (folder / "model.xml").write_text("<net/>", encoding="utf-8")
    (folder / "model.bin").write_bytes(b"weights")
    metadata = {"task": "detect", "names": names}
    (folder / "metadata.yaml").write_text(
        "task: detect\nnames:\n" + "".join(f"  {k}: {v}\n" for k, v in names.items()),
        encoding="utf-8")
    files = {p.name: digest(p) for p in folder.iterdir()}
    manifest = {"schema_version": 1, "task": "detect", "image_size": image_size,
                "source_sha256": source_sha256, "precision": precision, "files": files,
                "metadata": metadata}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def write_bundle(root, *, image_size=416, precision="openvino_int8", tamper=None):
    entries = []
    for name, _path, source_sha256 in MEMBERS:
        folder = write_member(root, name, source_sha256, precision=precision,
                             image_size=image_size)
        if tamper == name:
            (folder / "model.bin").write_bytes(b"changed")
        entries.append({"name": name, "source_sha256": source_sha256,
                        "precision": precision,
                        "directory": str(folder.relative_to(root)).replace("\\", "/"),
                        "manifest_sha256": digest(folder / "manifest.json")})
    (root / "ensemble.json").write_text(json.dumps({
        "schema_version": 1, "task": "detect", "image_size": image_size,
        "precision": "int8", "members": entries,
    }), encoding="utf-8")
    return root


def test_letterbox_matches_ultralytics_rectangle_geometry():
    from ultralytics.data.augment import LetterBox

    rng = np.random.default_rng(0)
    for height, width in ((720, 1280), (1080, 1920), (480, 640), (333, 519)):
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        expected = LetterBox(new_shape=(416, 416), auto=False)(image=image)
        tensor, left, top, ratio = letterbox(image, 416)
        actual = np.ascontiguousarray(tensor[0].transpose(1, 2, 0)[:, :, ::-1] * 255.)
        assert np.allclose(actual, expected, atol=1.0)
        new_w, new_h = round(width * ratio), round(height * ratio)
        assert (left, top) == ((416 - new_w) // 2, (416 - new_h) // 2)


def test_bundle_verification_pins_each_member(tmp_path):
    bundle = write_bundle(tmp_path)
    verified = verify_ensemble_bundle(bundle, image_size=416, members=MEMBERS)
    assert [name for name, _folder, _local, _precision in verified] == [name for name, _, _ in MEMBERS]


def test_bundle_verification_rejects_changed_member_file(tmp_path):
    bundle = write_bundle(tmp_path, tamper="fall-finetuned-v2")
    with pytest.raises(ValueError, match="Checksum"):
        verify_ensemble_bundle(bundle, image_size=416, members=MEMBERS)


def test_bundle_verification_rejects_unpinned_source(tmp_path):
    bundle = write_bundle(tmp_path)
    manifest = json.loads((bundle / "ensemble.json").read_text(encoding="utf-8"))
    manifest["members"][0]["source_sha256"] = "d" * 64
    (bundle / "ensemble.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="nguồn đã pin"):
        verify_ensemble_bundle(bundle, image_size=416, members=MEMBERS)


def test_bundle_verification_rejects_wrong_image_size(tmp_path):
    bundle = write_bundle(tmp_path, image_size=320)
    with pytest.raises(ValueError, match="kích thước"):
        verify_ensemble_bundle(bundle, image_size=416, members=MEMBERS)


def _runtime():
    """Decode helper without compiling graphs."""
    runtime = OpenVINOEnsemble.__new__(OpenVINOEnsemble)
    runtime.confidence = 0.4
    runtime.image_size = 416
    return runtime


def _output(entries, classes=3):
    columns = []
    for cx, cy, w, h, scores in entries:
        columns.append([cx, cy, w, h, *scores])
    return np.asarray(columns, dtype=np.float32).T[None]


def test_decode_maps_boxes_back_through_shared_letterbox():
    runtime = _runtime()
    member = SimpleNamespace(names={0: "fallen", 1: "sitting", 2: "standing"})
    # A 1280x720 frame letterboxed to 416: ratio 0.325, top pad 91.
    output = _output([(208., 208., 64., 32., (0.9, 0.05, 0.05)),
                      (50., 150., 10., 10., (0.1, 0.8, 0.1))])
    detections = runtime._decode(output, member, 0, 91, 0.325, (720, 1280, 3))
    assert len(detections) == 2
    fallen = detections[0]
    assert fallen.label == "fallen"
    assert fallen.class_id == 0
    assert fallen.confidence == pytest.approx(0.9, abs=1e-6)
    # Undo the shared padding and scale so the box lands back on the frame.
    x1, y1, x2, y2 = fallen.xyxy
    assert x1 == pytest.approx((176 - 0) / 0.325, abs=1e-3)
    assert y1 == pytest.approx((192 - 91) / 0.325, abs=1e-3)
    assert x2 == pytest.approx((240 - 0) / 0.325, abs=1e-3)
    assert y2 == pytest.approx((224 - 91) / 0.325, abs=1e-3)
    # argmax picks the highest class score, not necessarily index 0.
    assert detections[1].label == "sitting"
    assert detections[1].confidence == pytest.approx(0.8, abs=1e-6)


def test_decode_drops_low_confidence_and_clips_to_frame():
    runtime = _runtime()
    member = SimpleNamespace(names={0: "fallen", 1: "sitting", 2: "standing"})
    output = _output([(5., 5., 40., 40., (0.2, 0.1, 0.1))])
    assert runtime._decode(output, member, 0, 0, 1.0, (100, 100, 3)) == ()
    output = _output([(5., 5., 40., 40., (0.9, 0.1, 0.1))])
    (detection,) = runtime._decode(output, member, 0, 0, 1.0, (100, 100, 3))
    assert detection.xyxy == (0.0, 0.0, 25.0, 25.0)


def test_decode_accepts_transposed_output_layout():
    runtime = _runtime()
    member = SimpleNamespace(names={0: "fallen", 1: "sitting", 2: "standing"})
    output = _output([(100., 100., 40., 40., (0.9, 0.1, 0.1))])
    transposed = np.ascontiguousarray(output.transpose(0, 2, 1))
    assert runtime._decode(transposed, member, 0, 0, 1.0, (416, 416, 3))[0].label == "fallen"


def test_ensemble_rejects_unknown_backend_without_loading_models():
    from ai.posture_ensemble import EnsemblePostureDetector

    with pytest.raises(ValueError, match="torch hoặc openvino"):
        EnsemblePostureDetector(backend="onnx")
    with pytest.raises(ValueError, match="thư mục export"):
        EnsemblePostureDetector(backend="openvino")
    with pytest.raises(ValueError, match="model_dir"):
        EnsemblePostureDetector(backend="torch", model_dir="somewhere")


def test_torch_members_share_static_square_canvas():
    from ai.posture_ensemble import EnsemblePostureDetector

    calls = {}

    class FakeModel:
        def predict(self, **kwargs):
            calls.update(kwargs)
            return [SimpleNamespace(boxes=None)]

    detector = SimpleNamespace(image_size=416, confidence=0.4)
    member = SimpleNamespace(name="baseline", model=FakeModel(),
                             names={0: "fallen"}, confidence=0.4)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    assert EnsemblePostureDetector._run_member(detector, member, image) == ()
    assert calls["rect"] is False
    assert calls["imgsz"] == 416
    assert calls["agnostic_nms"] is True
