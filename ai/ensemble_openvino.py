"""OpenVINO ensemble runtime with one shared preprocessing pass per frame.

The stock Ultralytics path letterboxes the frame and decodes/NMSes inside every
member call. With three members that repeated Python overhead dominates a
416px INT8 graph on CPU. This module keeps the integrity guarantees of the
other export backends -- a parent manifest pins every member's source digest
and every member manifest covers its own files -- and replaces the repeated
pipeline with one letterbox plus a small vectorised decode per member.
"""

import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path

from ai.posture_detector import ROOT

SCHEMA_VERSION = 1
BUNDLE_FILE = "ensemble.json"
SUPPORTED_PRECISIONS = ("openvino_int8", "openvino_fp32", "openvino_fp16")
REQUIRED_MEMBER_FILES = {"model.xml", "model.bin", "metadata.yaml"}
PAD_VALUE = 114


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Không đọc được manifest ensemble ({Path(path).name}).") from exc
    if not isinstance(value, dict):
        raise ValueError("Manifest ensemble không phải một object JSON.")
    return value


def _verify_member_files(directory, *, image_size, precision):
    """Recheck one member's own manifest and every file it lists."""
    directory = Path(directory).resolve()
    local = _read_json(directory / "manifest.json")
    if (local.get("schema_version") != SCHEMA_VERSION or local.get("task") != "detect"
            or local.get("image_size") != image_size or local.get("precision") != precision):
        raise ValueError(f"Manifest thành viên ensemble không khớp ({directory.name}).")
    files = local.get("files")
    if not isinstance(files, dict) or not REQUIRED_MEMBER_FILES <= files.keys():
        raise ValueError("Manifest thành viên phải bao phủ graph, trọng số và metadata.")
    for name, expected in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("Đường dẫn manifest thành viên không hợp lệ.")
        if _digest(path) != expected:
            raise ValueError(f"Checksum tệp ensemble không khớp ({name}).")
    return local


def verify_ensemble_bundle(directory, *, image_size, members):
    """Validate the parent manifest against the pinned member sources."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("Thư mục export ensemble không tồn tại.")
    manifest = _read_json(directory / BUNDLE_FILE)
    if (manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("task") != "detect"
            or manifest.get("image_size") != image_size):
        raise ValueError("Manifest ensemble không khớp task hoặc kích thước ảnh.")
    expected = [(name, sha) for name, _path, sha in members]
    exported = manifest.get("members")
    if not isinstance(exported, list) or len(exported) != len(expected):
        raise ValueError("Manifest ensemble phải liệt kê đúng các thành viên đã pin.")
    verified = []
    for entry, (name, source_sha256) in zip(exported, expected):
        if not isinstance(entry, dict):
            raise ValueError("Mục thành viên ensemble không hợp lệ.")
        if entry.get("name") != name or entry.get("source_sha256") != source_sha256:
            raise ValueError(f"Thành viên ensemble không khớp nguồn đã pin ({name}).")
        precision = entry.get("precision")
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(f"Độ chính xác export không hỗ trợ ({name}).")
        folder = (directory / str(entry.get("directory", ""))).resolve()
        if not folder.is_relative_to(directory) or not folder.is_dir():
            raise ValueError(f"Thư mục thành viên ensemble không hợp lệ ({name}).")
        if not folder.name.endswith("_openvino_model"):
            raise ValueError(f"Thành viên ensemble phải là thư mục *_openvino_model ({name}).")
        local_manifest = folder / "manifest.json"
        if entry.get("manifest_sha256") != _digest(local_manifest):
            raise ValueError(f"Manifest thành viên ensemble đã thay đổi ({name}).")
        local = _verify_member_files(folder, image_size=image_size, precision=precision)
        verified.append((name, folder, local, precision))
    return verified


def letterbox(image, image_size):
    """Match YOLO's static, centred, non-stretched letterbox exactly."""
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Khung hình rỗng.")
    ratio = min(image_size / height, image_size / width)
    new_w, new_h = round(width * ratio), round(height * ratio)
    left = round((image_size - new_w) / 2 - 0.1)
    top = round((image_size - new_h) / 2 - 0.1)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((image_size, image_size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None],
                                  dtype=np.float32) / 255.0
    return tensor, left, top, ratio


class _Member:
    def __init__(self, name, compiled, names):
        self.name = name
        self.compiled = compiled
        self.names = names


class OpenVINOEnsemble:
    """Compiled per-member graphs sharing one letterbox per frame."""

    def __init__(self, directory, *, confidence=0.4, image_size=640, threads=2,
                 members=None):
        from ai.posture_ensemble import MEMBERS

        if (isinstance(confidence, bool) or not isinstance(confidence, Real)
                or not math.isfinite(confidence) or not 0 < confidence <= 1):
            raise ValueError("Confidence phải trong (0, 1].")
        if type(image_size) is not int or image_size not in (320, 416, 512, 640):
            raise ValueError("Kích thước AI phải là 320, 416, 512 hoặc 640.")
        if type(threads) is not int or not 1 <= threads <= 8:
            raise ValueError("Số CPU threads phải trong [1, 8].")
        pinned = tuple(MEMBERS if members is None else members)
        verified = verify_ensemble_bundle(directory, image_size=image_size, members=pinned)

        os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        import openvino as ov
        import yaml

        core = ov.Core()
        if os.name == "nt":
            core.set_property("CPU", {"INFERENCE_PRECISION_HINT": ov.Type.f32})
            os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX512_CORE_FP16")
        properties = {"INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY"}
        loaded = []
        for name, folder, _local_manifest, _precision in verified:
            metadata = yaml.safe_load((folder / "metadata.yaml").read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or metadata.get("task") != "detect":
                raise ValueError(f"Metadata thành viên ensemble không hợp lệ ({name}).")
            names = {int(key): str(value) for key, value in (metadata.get("names") or {}).items()}
            if not names or sorted(names) != list(range(len(names))):
                raise ValueError(f"Nhãn lớp ensemble không hợp lệ ({name}).")
            model = core.read_model(str(folder / "model.xml"))
            inputs = model.inputs
            if (len(inputs) != 1
                    or list(inputs[0].shape) != [1, 3, image_size, image_size]
                    or inputs[0].element_type != ov.Type.f32):
                raise ValueError(f"Input ensemble phải là float32 batch 1 vuông ({name}).")
            outputs = model.outputs
            if len(outputs) != 1:
                raise ValueError(f"Ensemble yêu cầu đúng một đầu ra ({name}).")
            if 4 + len(names) not in tuple(outputs[0].shape):
                raise ValueError(f"Đầu ra ensemble không khớp số lớp ({name}).")
            compiled = core.compile_model(model, "CPU", properties)
            loaded.append(_Member(name, compiled, names))
        if len({member.names.get(0, "").lower() for member in loaded}) != 1:
            raise ValueError("Các thành viên ensemble không cùng lớp chính (fallen index 0).")
        self.members = tuple(loaded)
        self.names = self.members[-1].names
        self.confidence = confidence
        self.image_size = image_size

    def _decode(self, output, member, left, top, ratio, shape):
        import numpy as np
        from ai.posture_detector import parse_detections

        array = np.asarray(output, dtype=np.float32)
        if array.ndim != 3 or array.shape[0] != 1:
            raise ValueError("Đầu ra ensemble phải có dạng [1, C, N].")
        channels = 4 + len(member.names)
        if array.shape[1] == channels:
            array = array[0]
        elif array.shape[2] == channels:
            array = array[0].T
        else:
            raise ValueError("Đầu ra ensemble không khớp số kênh.")
        scores = array[4:, :]
        class_ids = scores.argmax(axis=0)
        best = scores[class_ids, np.arange(scores.shape[1])]
        keep = np.flatnonzero(best >= self.confidence)
        if keep.size == 0:
            return ()
        centers, sides = array[:2, keep], array[2:4, keep]
        corners = np.concatenate((centers - sides / 2, centers + sides / 2), axis=0)
        corners -= np.array([[left], [top], [left], [top]], dtype=np.float32)
        corners /= ratio
        rows = np.concatenate((corners, best[keep][None],
                               class_ids[keep][None].astype(np.float32)), axis=0)
        return parse_detections(rows.T.tolist(), member.names, shape)

    def predict(self, image):
        from ai.posture_ensemble import merge_detections

        tensor, left, top, ratio = letterbox(image, self.image_size)
        detections = []
        for member in self.members:
            raw = member.compiled([tensor])
            output = next(iter(raw.values()))
            detections.extend(self._decode(output, member, left, top, ratio, image.shape))
        return merge_detections(detections)
