"""Three-member posture ensemble: pinned baseline + recall-tuned v1 + balanced v2.

Same security posture as ai/posture_detector.py: every member file is pinned by
size and SHA256 before load, and the restricted safe-globals unpickler is the
only loader. The output contract matches PostureDetector.predict exactly, so
ai/fusion_detector.py needs no change on how it consumes detections.

Two backends are available. "torch" (default) runs each member through the full
Ultralytics predict pipeline. "openvino" reuses one shared letterbox per frame
and a custom decode/agnostic-NMS pass per exported member, which removes most
of the repeated Python/Ultralytics overhead on CPU.
"""

from dataclasses import dataclass
import hashlib
import importlib
import math
import os
from pathlib import Path

from ai.posture_detector import Detection, SAFE_CLASSES, parse_detections

ROOT = Path(__file__).resolve().parents[1]

BASELINE_REVISION = "97261b03632a9715a27481b2cfdac6a78e26d2d9"
V1 = ROOT / "models" / "fall-finetuned-v1" / "best.pt"
V2 = ROOT / "models" / "fall-finetuned-v2" / "best.pt"
BASELINE = ROOT / "models" / "human-fall-detection" / BASELINE_REVISION / "best.pt"

# Pinned member checksums (20 MiB cap applies to every file, like the single model).
MEMBERS = (
    ("baseline", BASELINE, "3f56ad30358d5c63bf8dbc0c1299cf68818c3d291dfb10c94107b94110aadd4c"),
    ("fall-finetuned-v1", V1, "185e2e7fdcb0290b9ba248bdca13cae8d415820e19d8f2d2609cba270ca61684"),
    ("fall-finetuned-v2", V2, "f30c177513b856cd11296694746a19e126bb05436ebab3e52af7d13d77be0315"),
)

MAX_DETECTIONS = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned_checkpoint(path, expected_sha256, *, label, image_size):
    """Verify a member against its pinned digest, then load it with safe globals."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Thiếu model ensemble ({label}).")
    if path.stat().st_size > 20_000_000:
        raise ValueError(f"Checkpoint {label} vượt kích thước cho phép.")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"SHA256 model {label} không khớp; không nạp checkpoint.")
    import torch
    from ultralytics import YOLO

    allowlist = [getattr(importlib.import_module(module), name)
                 for module, names in SAFE_CLASSES.items() for name in names]
    with torch.serialization.safe_globals(allowlist):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    network = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(network, torch.nn.Module):
        raise ValueError(f"Checkpoint {label} không có mạng nhận diện hợp lệ.")
    # YAML creates a local architecture only, without loading any pickle/download.
    wrapper = YOLO("yolo11n.yaml", task="detect")
    wrapper.model = network.float().eval()
    wrapper.overrides = {"task": "detect", "model": str(path), "imgsz": image_size}
    names = {int(key): str(value) for key, value in wrapper.names.items()}
    return wrapper, names


@dataclass(frozen=True)
class _Member:
    name: str
    model: object
    names: dict
    confidence: float


def merge_detections(detections) -> tuple[Detection, ...]:
    """Class-agnostic NMS over merged member detections, best score first."""
    if not detections:
        return ()
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for candidate in ordered:
        if all(detection_overlap(candidate.xyxy, kept_box.xyxy) < 0.5 for kept_box in kept):
            kept.append(candidate)
        if len(kept) >= MAX_DETECTIONS:
            break
    return tuple(kept)


class EnsemblePostureDetector:
    """Runs each pinned posture model and merges detections with agnostic NMS.

    Merged boxes keep the highest member confidence for their location, so a
    fall confirmed by several members stays a single box with its best score;
    a fall only one member sees still survives. Same .predict() contract as
    PostureDetector (tuple[Detection, ...]).
    """

    def __init__(self, *, confidence=0.4, image_size=640, threads=2, backend="torch",
                 model_dir=None):
        if not math.isfinite(confidence) or not 0 < confidence <= 1:
            raise ValueError("Confidence phải trong (0, 1].")
        if image_size not in (320, 416, 512, 640):
            raise ValueError("Kích thước AI phải là 320, 416, 512 hoặc 640.")
        if not 1 <= threads <= 8:
            raise ValueError("Số CPU threads phải trong [1, 8].")
        if backend not in ("torch", "openvino"):
            raise ValueError("Ensemble backend phải là torch hoặc openvino.")
        if backend == "torch" and model_dir is not None:
            raise ValueError("model_dir chỉ dùng với ensemble backend openvino.")
        if backend == "openvino" and model_dir is None:
            raise ValueError("Ensemble OpenVINO cần thư mục export đã kiểm tra.")

        self.confidence = confidence
        self.image_size = image_size
        self.backend = backend
        self._runtime = None
        if backend == "openvino":
            from ai.ensemble_openvino import OpenVINOEnsemble
            self._runtime = OpenVINOEnsemble(model_dir, confidence=confidence,
                                             image_size=image_size, threads=threads)
            self.members = ()
            self.names = self._runtime.names
        else:
            self.members, self.names = self._load_members(image_size, threads, confidence)

        import numpy as np
        self.predict(np.zeros((image_size, image_size, 3), dtype=np.uint8))  # warm-up

    @staticmethod
    def _load_members(image_size, threads, confidence):
        os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        import torch
        from ultralytics import settings

        settings.update({"sync": False})
        torch.set_num_threads(threads)
        members = []
        for name, path, expected_sha256 in MEMBERS:
            wrapper, names = load_pinned_checkpoint(path, expected_sha256,
                                                    label=name, image_size=image_size)
            members.append(_Member(name, wrapper, names, confidence))
        if len({member.names.get(0, "").lower() for member in members}) != 1:
            raise ValueError("Các thành viên ensemble không cùng lớp chính (fallen index 0).")
        return tuple(members), members[-1].names  # baseline naming, for callers reading labels

    def _run_member(self, member: _Member, image):
        # rect=False keeps the static square letterbox used by every exported
        # backend, so Torch and OpenVINO members see the same canvas.
        result = member.model.predict(
            source=image, device="cpu", imgsz=self.image_size, conf=self.confidence,
            agnostic_nms=True, iou=0.5, rect=False,
            verbose=False, save=False, show=False, max_det=MAX_DETECTIONS,
        )[0]
        if result.boxes is None:
            return ()
        rows = result.boxes.data.cpu().tolist()
        return parse_detections(rows, member.names, image.shape)

    def predict(self, image) -> tuple[Detection, ...]:
        if self._runtime is not None:
            return self._runtime.predict(image)
        detections = []
        for member in self.members:
            detections.extend(self._run_member(member, image))
        return merge_detections(detections)


def detection_overlap(first, second) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0., x2 - x1) * max(0., y2 - y1)
    first_area = max(0., first[2] - first[0]) * max(0., first[3] - first[1])
    second_area = max(0., second[2] - second[0]) * max(0., second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.
