"""Optional CPU ONNX inference from local, verified project export bundles."""

import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path

from ai.pose_detector import POSE_SHA256, PoseDetector, parse_pose
from ai.posture_detector import EXPECTED_SHA256, ROOT, parse_detections


def verify_onnx_bundle(directory, task, image_size):
    if task not in ("pose", "detect"):
        raise ValueError("Unsupported exported model task.")
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected_source = POSE_SHA256 if task == "pose" else EXPECTED_SHA256
    if (manifest.get("schema_version") != 1 or manifest.get("task") != task
            or manifest.get("image_size") != image_size
            or manifest.get("source_sha256") != expected_source):
        raise ValueError("Export task, source checkpoint, or image size does not match.")
    if manifest.get("models", {}).get("onnx") != "model.onnx":
        raise ValueError("Expected the bundle's model.onnx graph.")
    files = manifest.get("files")
    if not isinstance(files, dict) or "model.onnx" not in files:
        raise ValueError("Export manifest must cover model.onnx.")
    for name, expected in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("Invalid export manifest path.")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("Export file checksum does not match.")
    return directory / "model.onnx"


def _reject_external_data(model):
    """Inspect tensors in graphs, attributes, sparse tensors and local functions."""
    from google.protobuf.message import Message
    from onnx import TensorProto

    def visit(message):
        if isinstance(message, TensorProto) and (
                message.data_location == TensorProto.EXTERNAL or message.external_data):
            raise ValueError("External ONNX weight files are not supported.")
        for field, value in message.ListFields():
            if field.message_type is None:
                continue
            if isinstance(value, Message):
                visit(value)
            else:
                for child in value:
                    visit(child)

    visit(model)


def _load(directory, task, confidence, image_size, threads):
    if (isinstance(confidence, bool) or not isinstance(confidence, Real)
            or not math.isfinite(confidence) or not 0 < confidence <= 1):
        raise ValueError("Confidence must be a finite number in (0, 1].")
    if type(image_size) is not int or image_size not in (320, 416, 512, 640):
        raise ValueError("Invalid inference image size.")
    if type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("Invalid inference thread count.")
    if directory is None:
        raise ValueError("An ONNX export bundle directory is required; run tools/export_openvino.py.")
    path = verify_onnx_bundle(directory, task, image_size)
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO, settings

    # Our small exports embed all weights. Refuse graphs that could load external
    # initializer files outside the manifest's verified graph.
    graph = onnx.load_model(str(path), load_external_data=False)
    _reject_external_data(graph)
    settings.update({"sync": False})
    torch.set_num_threads(threads)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 3, image_size, image_size] or inputs[0].type != "tensor(float)":
        raise ValueError("Expected static float32 batch-one square ONNX input.")
    return YOLO(str(path), task=task), session


def _replace_session(yolo, session):
    wrapper = yolo.predictor.model
    backend = getattr(wrapper, "backend", wrapper)
    if not hasattr(backend, "session"):
        raise RuntimeError("Installed Ultralytics ONNX backend is incompatible.")
    backend.session = session


class _ONNXDetector:
    def _initialize(self, directory, task, confidence, image_size, threads):
        self.model, self._session = _load(directory, task, confidence, image_size, threads)
        self.confidence, self.image_size = confidence, image_size
        import numpy as np
        blank = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        self.predict(blank)
        _replace_session(self.model, self._session)
        self.predict(blank)

    def _result(self, image):
        return self.model.predict(source=image, device="cpu", imgsz=self.image_size,
                                  conf=self.confidence, rect=False, verbose=False,
                                  save=False, show=False, agnostic_nms=True,
                                  iou=0.5, max_det=20)[0]


class ONNXPoseDetector(_ONNXDetector, PoseDetector):
    def __init__(self, directory, *, confidence=0.4, image_size=416, threads=2):
        self._initialize(directory, "pose", confidence, image_size, threads)

    def predict(self, image):
        result = self._result(image)
        if result.boxes is None or result.keypoints is None:
            return ()
        return parse_pose(result.boxes.data.cpu().tolist(), result.keypoints.data.cpu().tolist(), image.shape)


class ONNXPostureDetector(_ONNXDetector):
    def __init__(self, directory, *, confidence=0.4, image_size=416, threads=2):
        self._initialize(directory, "detect", confidence, image_size, threads)
        self.names = {int(key): str(value) for key, value in self.model.names.items()}

    def predict(self, image):
        result = self._result(image)
        if result.boxes is None:
            return ()
        names = {int(key): str(value) for key, value in self.model.names.items()}
        return parse_detections(result.boxes.data.cpu().tolist(), names, image.shape)