"""Compare warmed inference on identical local video frames, with output drift checks."""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.export_openvino import load_verified_model, sha256


def sample_frames(video, count):
    import cv2

    video = Path(video).resolve()
    if not video.is_file() or count < 1:
        raise ValueError("Provide a local video and a positive frame count.")
    capture = cv2.VideoCapture(str(video))
    frames = []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            raise ValueError("Video frame count is unavailable.")
        indices = sorted({round(i * (total - 1) / max(count - 1, 1)) for i in range(count)})
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Failed to decode benchmark frame {index}.")
            frames.append((index, image))
    finally:
        capture.release()
    return frames


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def output_drift(reference, candidate):
    """Class-aware greedy IoU pairing; a diagnostic, never ground-truth accuracy."""
    matched, reference_count, candidate_count = 0, 0, 0
    overlaps, scores, keypoints = [], [], []
    for expected, actual in zip(reference, candidate):
        a = expected.boxes.data.cpu().tolist() if expected.boxes is not None else []
        b = actual.boxes.data.cpu().tolist() if actual.boxes is not None else []
        reference_count += len(a)
        candidate_count += len(b)
        pairs = sorted(((box_iou(x, y), i, j) for i, x in enumerate(a)
                        for j, y in enumerate(b) if x[5] == y[5]), reverse=True)
        used_a, used_b = set(), set()
        for overlap, i, j in pairs:
            if overlap < 0.5 or i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            matched += 1
            overlaps.append(overlap)
            scores.append(abs(a[i][4] - b[j][4]))
            if expected.keypoints is not None and actual.keypoints is not None:
                for p, q in zip(expected.keypoints.data[i].cpu().tolist(), actual.keypoints.data[j].cpu().tolist()):
                    if p[2] >= 0.5 and q[2] >= 0.5:
                        keypoints.append(math.hypot(p[0] - q[0], p[1] - q[1]))
    mean = lambda values: round(statistics.mean(values), 6) if values else None
    return {"reference_detections": reference_count, "candidate_detections": candidate_count,
            "matched_iou_ge_0_5": matched, "unmatched_reference": reference_count - matched,
            "unmatched_candidate": candidate_count - matched, "matched_mean_iou": mean(overlaps),
            "matched_mean_score_delta": mean(scores), "confident_keypoint_mean_shift_px": mean(keypoints)}


def benchmark(bundle, video, *, frames=12, repeats=2, warmup=3, threads=2,
              confidence=0.4, include_onnx=False):
    if min(frames, repeats, warmup, threads) < 1 or threads > 8:
        raise ValueError("Frames/repeats/warmup must be positive; threads must be 1..8.")
    import cv2
    import torch
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or sha256(path) != expected:
            raise ValueError(f"Export file integrity failed: {name}")
    task, size = manifest["task"], manifest["image_size"]
    native, _, digest = load_verified_model(task, size, threads)
    if manifest["source_sha256"] != digest:
        raise ValueError("Export source digest does not match the verified native model.")
    cv2.setNumThreads(1)
    torch.set_num_threads(threads)
    sampled = sample_frames(video, frames)
    entries = [("pytorch", native), ("pytorch_rect", native)] + [(name, bundle / path) for name, path in manifest["models"].items()
                                       if include_onnx or name != "onnx"]
    rows, reference = [], None
    for name, model in entries:
        started = time.perf_counter()
        detector = None
        if not name.startswith("pytorch"):
            if name.startswith("openvino_"):
                from ai.openvino_detector import OpenVINOPoseDetector, OpenVINOPostureDetector
                detector_class = OpenVINOPoseDetector if task == "pose" else OpenVINOPostureDetector
                detector = detector_class(model, confidence=confidence, image_size=size, threads=threads)
                model = detector.model
            else:
                from ai.onnx_detector import ONNXPoseDetector, ONNXPostureDetector
                detector_class = ONNXPoseDetector if task == "pose" else ONNXPostureDetector
                detector = detector_class(bundle, confidence=confidence, image_size=size, threads=threads)
                model = detector.model
        def predict(image):
            return model.predict(image, device="cpu", imgsz=size, conf=confidence,
                                 rect=name == "pytorch_rect", verbose=False, save=False, show=False,
                                 agnostic_nms=True, iou=0.5, max_det=20)[0]
        for _ in range(warmup):
            predict(sampled[0][1])
        initialization_ms = (time.perf_counter() - started) * 1000
        latencies, outputs = [], []
        for repeat in range(repeats):
            for _, image in sampled:
                started = time.perf_counter()
                result = predict(image)
                latencies.append((time.perf_counter() - started) * 1000)
                if repeat == 0:
                    outputs.append(result)
        if reference is None:
            reference = outputs
        rows.append({"backend": name, "rect": name == "pytorch_rect",
                     "load_and_warmup_ms": round(initialization_ms, 2),
                     "mean_ms": round(statistics.mean(latencies), 2),
                     "median_ms": round(statistics.median(latencies), 2),
                     "p90_ms": round(sorted(latencies)[math.ceil(0.9 * len(latencies)) - 1], 2),
                     "throughput_fps": round(1000 / statistics.mean(latencies), 2),
                     "drift_vs_pytorch": output_drift(reference, outputs)})
        print(json.dumps(rows[-1]), flush=True)
    return {"task": task, "image_size": size, "threads": threads, "opencv_threads": 1,
            "confidence": confidence, "bundle": str(bundle),
            "video": str(Path(video).resolve()), "video_sha256": sha256(video),
            "frame_indices": [index for index, _ in sampled], "repeats": repeats, "warmup": warmup,
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "ultralytics", "numpy", "openvino")},
            "note": "Serial warmed preprocessing+inference+NMS; no decode/display/fall-engine time. "
                    "PyTorch initial safe loading is excluded from load_and_warmup_ms. "
                    "Output drift is not labeled detection or fall-event accuracy. "
                    "OpenVINO INFERENCE_NUM_THREADS and ONNX intra-op threads use the requested budget.",
            "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--include-onnx", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark(args.bundle, args.video, frames=args.frames, repeats=args.repeats,
                       warmup=args.warmup, threads=args.threads, confidence=args.confidence,
                       include_onnx=args.include_onnx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()