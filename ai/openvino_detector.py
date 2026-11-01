"""Local OpenVINO exports with explicit model metadata and integrity checks."""

import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path

from ai.pose_detector import POSE_SHA256, PoseDetector, parse_pose
from ai.posture_detector import EXPECTED_SHA256, ROOT, parse_detections


def verify_bundle(directory, task, image_size):
    if task not in ("pose", "detect"):
        raise ValueError("Unsupported exported model task.")
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    source = POSE_SHA256 if task == "pose" else EXPECTED_SHA256
    if (manifest.get("schema_version") != 1 or manifest.get("task") != task
            or manifest.get("image_size") != image_size
            or manifest.get("source_sha256") != source):
        raise ValueError("Export task, source checkpoint, or image size does not match.")
    files = manifest.get("files")
    if not isinstance(files, dict) or not {"model.xml", "model.bin", "metadata.yaml"} <= files.keys():
        raise ValueError("Export manifest must cover the graph, weights and metadata.")
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
    if not directory.name.endswith("_openvino_model"):
        raise ValueError("Expected an exported *_openvino_model directory.")
    return directory


def _load(directory, task, confidence, image_size, threads):
    if task not in ("pose", "detect"):
        raise ValueError("Unsupported exported model task.")
    if (isinstance(confidence, bool) or not isinstance(confidence, Real)
            or not math.isfinite(confidence) or not 0 < confidence <= 1):
        raise ValueError("Confidence must be a finite number in (0, 1].")
    if directory is None:
        raise ValueError("An OpenVINO model directory is required; run tools/export_openvino.py.")
    if image_size not in (320, 416, 512, 640) or type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("Invalid inference size or thread count.")
    directory = verify_bundle(directory, task, image_size)
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import openvino as ov
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    # Compile explicitly to keep the requested CPU thread budget. AutoBackend
    # would otherwise compile with its own default performance settings.
    core = ov.Core()
    if os.name == "nt":
        # Match Ultralytics' Windows CPU settings before our own compilation.
        core.set_property("CPU", {"INFERENCE_PRECISION_HINT": ov.Type.f32})
        os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX512_CORE_FP16")
    model = core.read_model(str(directory / "model.xml"))
    compiled = core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": threads,
                                                "PERFORMANCE_HINT": "LATENCY"})
    yolo = YOLO(str(directory), task=task)
    return yolo, compiled


def _replace_compilation(yolo, compiled):
    # Newer AutoBackend delegates reads to .backend; assigning directly to the
    # wrapper would silently shadow the attribute without changing inference.
    wrapper = yolo.predictor.model
    backend = getattr(wrapper, "backend", wrapper)
    if not hasattr(backend, "ov_compiled_model"):
        raise RuntimeError("Installed Ultralytics OpenVINO backend is incompatible.")
    backend.ov_compiled_model = compiled


class OpenVINOPoseDetector(PoseDetector):
    def __init__(self, directory, *, confidence=0.4, image_size=416, threads=2):
        self.model, self._compiled = _load(directory, "pose", confidence, image_size, threads)
        self.confidence, self.image_size = confidence, image_size
        self._warmup()

    def _warmup(self):
        import numpy as np
        # Let Ultralytics initialize preprocessing/NMS, then use our compilation
        # for all subsequent frames and warm it once before starting the camera.
        self.predict(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
        _replace_compilation(self.model, self._compiled)
        self.predict(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

    def predict(self, image):
        result = self.model.predict(source=image, device="cpu", imgsz=self.image_size,
                                    conf=self.confidence, rect=False, verbose=False,
                                    save=False, show=False, agnostic_nms=True,
                                    iou=0.5, max_det=20)[0]
        if result.boxes is None or result.keypoints is None:
            return ()
        return parse_pose(result.boxes.data.cpu().tolist(), result.keypoints.data.cpu().tolist(), image.shape)


class OpenVINOPostureDetector:
    def __init__(self, directory, *, confidence=0.4, image_size=416, threads=2):
        self.model, self._compiled = _load(directory, "detect", confidence, image_size, threads)
        self.confidence, self.image_size = confidence, image_size
        import numpy as np
        blank = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        self.predict(blank)
        _replace_compilation(self.model, self._compiled)
        self.predict(blank)
        self.names = {int(key): str(value) for key, value in self.model.names.items()}

    def predict(self, image):
        result = self.model.predict(source=image, device="cpu", imgsz=self.image_size,
                                    conf=self.confidence, rect=False, verbose=False,
                                    save=False, show=False, agnostic_nms=True,
                                    iou=0.5, max_det=20)[0]
        if result.boxes is None:
            return ()
        names = {int(key): str(value) for key, value in self.model.names.items()}
        return parse_detections(result.boxes.data.cpu().tolist(), names, image.shape)