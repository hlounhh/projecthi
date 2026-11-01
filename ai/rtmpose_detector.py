"""Optional, local-only YOLOX-tiny + RTMPose-s COCO17 inference.

The official exports have fixed inputs (416 square and 192x256). ``image_size``
is accepted for the shared detector interface; it does not resize the networks.
Detection confidence and SimCC keypoint scores need separate calibration from
YOLO pose. No accuracy improvement is implied without labelled video evaluation.
"""

import hashlib
import math
from pathlib import Path

from ai.pose_detector import PosePerson, parse_pose

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "rtmpose-s-body7-20230504"
MODEL_SHA256 = {
    "yolox-tiny.onnx": "ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250",
    "rtmpose-s.onnx": "9aeb635b83f86aea45cf45d85798f7eba1a162de8e0d721c44e54fe5eebaf47d",
}


def verify_models(model_dir):
    """Verify exact official exports before passing any data to native runtimes."""
    directory = Path(model_dir)
    for name, expected in MODEL_SHA256.items():
        path = directory / name
        if not path.is_file():
            raise ValueError("RTMPose assets missing; run python tools/download_rtmpose.py")
        if path.stat().st_size > 50_000_000:
            raise ValueError(f"RTMPose asset exceeds size limit: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"RTMPose SHA256 mismatch: {name}")


def parse_rtmpose(detections, labels, keypoints, scores, shape, *, confidence=0.4):
    """Convert matched end-to-end detector and SimCC outputs to PosePerson."""
    boxes, points = [], []
    for row, label, joints, values in zip(detections, labels, keypoints, scores):
        if len(row) != 5 or len(joints) != 17 or len(values) != 17:
            continue
        if float(label) != 0 or not math.isfinite(float(row[4])) or float(row[4]) < confidence:
            continue
        if any(len(point) != 2 for point in joints):
            continue
        boxes.append([*row, 0])
        points.append([(*point, score) for point, score in zip(joints, values)])
    return parse_pose(boxes, points, shape)


class _Runtime:
    def __init__(self, path, backend, threads):
        self.backend = backend
        if backend == "onnxruntime":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("RTMPose requires a working onnxruntime installation.") from exc
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(path), sess_options=options,
                                               providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        else:
            try:
                from openvino import Core
            except ImportError as exc:
                raise RuntimeError("RTMPose OpenVINO backend requires openvino.") from exc
            self.session = Core().compile_model(str(path), "CPU", {
                "PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": threads,
                "NUM_STREAMS": 1, "INFERENCE_PRECISION_HINT": "f32",
            })

    def __call__(self, image):
        import numpy as np
        tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)
        if self.backend == "onnxruntime":
            return self.session.run(None, {self.input_name: tensor})
        result = self.session([tensor])
        return [result[output] for output in self.session.outputs]


class RTMPoseDetector:
    """Two-stage pose estimator with the same predict/crop contract as YOLO."""

    def __init__(self, *, confidence=0.4, image_size=640, threads=2,
                 model_dir=None, backend="onnxruntime"):
        if not math.isfinite(confidence) or not 0 < confidence <= 1:
            raise ValueError("Confidence must be in (0, 1].")
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 8:
            raise ValueError("CPU threads must be an integer in [1, 8].")
        if backend not in ("onnxruntime", "openvino"):
            raise ValueError("RTMPose backend must be onnxruntime or openvino.")
        directory = Path(model_dir) if model_dir is not None else MODEL_DIR
        verify_models(directory)
        try:
            from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine
            from rtmlib.tools.pose_estimation.post_processings import get_simcc_maximum
        except ImportError as exc:
            raise RuntimeError("RTMPose requires rtmlib; install the optional AI dependencies.") from exc
        self._bbox_xyxy2cs = bbox_xyxy2cs
        self._affine = top_down_affine
        self._simcc = get_simcc_maximum
        self.confidence = confidence
        self.image_size = image_size
        self.detector_input_size = (416, 416)
        self.pose_input_size = (192, 256)
        self._detector = _Runtime(directory / "yolox-tiny.onnx", backend, threads)
        self._pose = _Runtime(directory / "rtmpose-s.onnx", backend, threads)

    def predict(self, image):
        import cv2
        import numpy as np
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
            raise ValueError("Expected a nonempty HxWx3 BGR image.")
        height, width = image.shape[:2]
        ratio = min(416 / height, 416 / width)
        resized_width = max(1, int(width * ratio))
        resized_height = max(1, int(height * ratio))
        padded = np.full((416, 416, 3), 114, dtype=np.uint8)
        padded[:resized_height, :resized_width] = cv2.resize(image, (resized_width, resized_height))
        # The pinned HumanArt YOLOX export already performs class-aware NMS.
        detections, labels = self._detector(padded)
        detections, labels = detections[0], labels[0]
        boxes = []
        for row, label in zip(detections, labels):
            if (len(row) != 5 or label != 0 or not np.isfinite(row).all()
                    or not self.confidence <= row[4] <= 1):
                continue
            x1, y1, x2, y2 = row[:4] / ratio
            x1, x2 = np.clip([x1, x2], 0, width)
            y1, y2 = np.clip([y1, y2], 0, height)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2, float(row[4])])
        # Bound per-frame top-down cost; no synthetic full-image pose on empty frames.
        boxes.sort(key=lambda row: row[4], reverse=True)
        people = []
        for row in boxes[:20]:
            center, scale = self._bbox_xyxy2cs(np.asarray(row[:4]), padding=1.25)
            crop, scale = self._affine(self.pose_input_size, scale, center, image)
            crop = (crop.astype(np.float32) - (123.675, 116.28, 103.53)) / (58.395, 57.12, 57.375)
            simcc_x, simcc_y = self._pose(crop)
            keypoints, scores = self._simcc(simcc_x, simcc_y)
            keypoints = keypoints / 2 / self.pose_input_size * scale + center - scale / 2
            people.extend(parse_rtmpose([row], [0], keypoints, scores, image.shape,
                                        confidence=self.confidence))
        return tuple(people)

    def predict_crop(self, image, bounds):
        """Return fresh crop detections in absolute coordinates, preserving invisible joints."""
        height, width = image.shape[:2]
        if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
            return ()
        x1, y1, x2, y2 = bounds
        x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
        x2, y2 = min(width, math.ceil(x2)), min(height, math.ceil(y2))
        if x2 <= x1 or y2 <= y1:
            return ()
        return tuple(PosePerson(
            (person.xyxy[0] + x1, person.xyxy[1] + y1,
             person.xyxy[2] + x1, person.xyxy[3] + y1), person.confidence,
            tuple((x + x1, y + y1, score) if score > 0 else (0., 0., 0.)
                  for x, y, score in person.keypoints))
            for person in self.predict(image[y1:y2, x1:x2]))