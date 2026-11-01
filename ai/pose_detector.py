"""Restricted loading of an official pretrained COCO17 pose model."""

from dataclasses import dataclass
import importlib
import math
import os

from ai.posture_detector import ROOT, SAFE_CLASSES, verify_weights

POSE_WEIGHTS = ROOT / "models" / "yolo11n-pose-v8.3.0" / "yolo11n-pose.pt"
POSE_SHA256 = "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"


def _pose_getattr(obj, name):
    # The pinned official pickle reconstructs Detect.forward through getattr.
    # Do not allow unrestricted dynamic attribute access in the unpickler.
    from ultralytics.nn.modules.head import Detect
    if obj is Detect and name == "forward":
        return Detect.forward
    raise ValueError("Unexpected checkpoint attribute reference.")


@dataclass(frozen=True)
class PosePerson:
    xyxy: tuple[float, float, float, float]
    confidence: float
    keypoints: tuple[tuple[float, float, float], ...]


def parse_pose(boxes, keypoints, shape):
    height, width = shape[:2]
    people = []
    for row, points in zip(boxes, keypoints):
        if len(row) != 6 or len(points) != 17:
            continue
        x1, y1, x2, y2, score, class_id = map(float, row)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score, class_id)):
            continue
        if class_id != 0 or not 0 <= score <= 1:
            continue
        x1, x2 = max(0., min(width, x1)), max(0., min(width, x2))
        y1, y2 = max(0., min(height, y1)), max(0., min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        parsed = []
        for point in points:
            if len(point) != 3:
                break
            x, y, conf = map(float, point)
            if (not all(math.isfinite(v) for v in (x, y, conf))
                    or not 0 <= x <= width or not 0 <= y <= height or not 0 <= conf <= 1):
                parsed.append((0., 0., 0.))
            else:
                parsed.append((x, y, conf))
        if len(parsed) == 17:
            people.append(PosePerson((x1, y1, x2, y2), score, tuple(parsed)))
    return tuple(people)


class PoseDetector:
    def __init__(self, *, confidence=0.4, image_size=640, threads=2):
        verify_weights(POSE_WEIGHTS, POSE_SHA256)
        os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        import torch
        import numpy as np
        from ultralytics import YOLO, settings
        from ultralytics.nn.modules.head import Pose
        from ultralytics.nn.tasks import PoseModel

        settings.update({"sync": False})
        torch.set_num_threads(threads)
        classes = [getattr(importlib.import_module(module), name)
                   for module, names in SAFE_CLASSES.items() for name in names]
        classes.extend([Pose, PoseModel, (_pose_getattr, "builtins.getattr")])
        with torch.serialization.safe_globals(classes):
            checkpoint = torch.load(POSE_WEIGHTS, map_location="cpu", weights_only=True)
        network = checkpoint.get("ema")
        if network is None:
            network = checkpoint.get("model")
        if not isinstance(network, PoseModel):
            raise ValueError("Checkpoint is not the expected pose architecture.")
        self.model = YOLO("yolo11n-pose.yaml", task="pose")
        self.model.model = network.float().eval()
        self.model.overrides = {"task": "pose", "model": str(POSE_WEIGHTS), "imgsz": image_size}
        self.confidence = confidence
        self.image_size = image_size
        self.predict(np.zeros((image_size, image_size, 3), dtype=np.uint8))

    def predict_crop(self, image, bounds):
        """Infer on one image slice; return only fresh detections in image coordinates."""
        height, width = image.shape[:2]
        x1, y1, x2, y2 = bounds
        x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
        x2, y2 = min(width, math.ceil(x2)), min(height, math.ceil(y2))
        if x2 <= x1 or y2 <= y1:
            return ()
        return tuple(PosePerson(
            (p.xyxy[0] + x1, p.xyxy[1] + y1, p.xyxy[2] + x1, p.xyxy[3] + y1),
            p.confidence,
            tuple((x + x1, y + y1, confidence) if confidence > 0 else (0., 0., 0.)
                  for x, y, confidence in p.keypoints))
            for p in self.predict(image[y1:y2, x1:x2]))

    def predict(self, image):
        result = self.model.predict(source=image, device="cpu", imgsz=self.image_size,
                                    conf=self.confidence, verbose=False, save=False, show=False,
                                    agnostic_nms=True, iou=0.5, max_det=20)[0]
        if result.boxes is None or result.keypoints is None:
            return ()
        return parse_pose(result.boxes.data.cpu().tolist(), result.keypoints.data.cpu().tolist(), image.shape)
