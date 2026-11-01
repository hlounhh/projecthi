"""Pinned Hugging Face image detector. A 'Fallen' box is NOT a fall event."""

from dataclasses import dataclass
import hashlib
import importlib
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "97261b03632a9715a27481b2cfdac6a78e26d2d9"
DEFAULT_WEIGHTS = ROOT / "models" / "human-fall-detection" / REVISION / "best.pt"
EXPECTED_SHA256 = "3f56ad30358d5c63bf8dbc0c1299cf68818c3d291dfb10c94107b94110aadd4c"

# Static pickle inspection found only these Torch/Ultralytics classes and helpers.
# Use the restricted unpickler, never torch.load(weights_only=False) on this file.
SAFE_CLASSES = {
    "torch.nn.modules.activation": ("SiLU",),
    "torch.nn.modules.batchnorm": ("BatchNorm2d",),
    "torch.nn.modules.container": ("ModuleList", "Sequential"),
    "torch.nn.modules.conv": ("Conv2d",),
    "torch.nn.modules.linear": ("Identity",),
    "torch.nn.modules.pooling": ("MaxPool2d",),
    "torch.nn.modules.upsampling": ("Upsample",),
    "ultralytics.nn.modules.block": (
        "Attention", "Bottleneck", "C2PSA", "C3k", "C3k2", "DFL", "PSABlock", "SPPF",
    ),
    "ultralytics.nn.modules.conv": ("Concat", "Conv", "DWConv"),
    "ultralytics.nn.modules.head": ("Detect",),
    "ultralytics.nn.tasks": ("DetectionModel",),
}


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    class_id: int
    label: str
    confidence: float


def verify_weights(path: Path, expected_sha256=EXPECTED_SHA256):
    if not path.is_file():
        raise ValueError("Chưa có model. Chạy: python -I tools/download_model.py")
    if path.stat().st_size > 20_000_000:
        raise ValueError("Checkpoint vượt kích thước cho phép.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("SHA256 model không khớp phiên bản đã kiểm tra; không nạp checkpoint.")


def parse_detections(boxes, names, shape) -> tuple[Detection, ...]:
    height, width = shape[:2]
    output = []
    for row in boxes:
        if len(row) != 6:
            continue
        x1, y1, x2, y2, score, class_value = (float(value) for value in row)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score, class_value)):
            continue
        if not class_value.is_integer() or not 0 <= score <= 1:
            continue
        class_id = int(class_value)
        if class_id not in names:
            continue
        x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
        y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        output.append(Detection((x1, y1, x2, y2), class_id, str(names[class_id]), score))
    return tuple(output)


class PostureDetector:
    def __init__(self, weights=DEFAULT_WEIGHTS, *, confidence=0.4, image_size=640, threads=2):
        if not math.isfinite(confidence) or not 0 < confidence <= 1:
            raise ValueError("Confidence phải trong (0, 1].")
        if image_size not in (320, 416, 512, 640):
            raise ValueError("Kích thước AI phải là 320, 416, 512 hoặc 640.")
        if not 1 <= threads <= 8:
            raise ValueError("Số CPU threads phải trong [1, 8].")
        path = Path(weights).resolve()
        verify_weights(path)
        # Local-only inference; no telemetry, hub sync, image saves or repo scripts.
        os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        import torch
        from ultralytics import YOLO, settings

        settings.update({"sync": False})
        torch.set_num_threads(threads)
        allowlist = [getattr(importlib.import_module(module), name)
                     for module, names in SAFE_CLASSES.items() for name in names]
        with torch.serialization.safe_globals(allowlist):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        network = checkpoint.get("ema")
        if network is None:
            network = checkpoint.get("model")
        if not isinstance(network, torch.nn.Module):
            raise ValueError("Checkpoint không có mạng nhận diện hợp lệ.")
        # YAML creates a local architecture only, without loading any pickle/download.
        self.model = YOLO("yolo11n.yaml", task="detect")
        self.model.model = network.float().eval()
        self.model.overrides = {"task": "detect", "model": str(path), "imgsz": image_size}
        self.names = {int(key): str(value) for key, value in self.model.names.items()}
        self.confidence = confidence
        self.image_size = image_size
        # Initialize/fuse/warm up before camera starts; exclude it from steady FPS.
        import numpy as np
        self.predict(np.zeros((image_size, image_size, 3), dtype=np.uint8))

    def predict(self, image) -> tuple[Detection, ...]:
        result = self.model.predict(
            source=image, device="cpu", imgsz=self.image_size, conf=self.confidence,
            # All three classes represent a person: suppress overlapping boxes
            # across labels, not just duplicates within the same label.
            agnostic_nms=True, iou=0.5,
            verbose=False, save=False, show=False, max_det=20,
        )[0]
        if result.boxes is None:
            return ()
        rows = result.boxes.data.cpu().tolist()
        return parse_detections(rows, self.names, image.shape)
