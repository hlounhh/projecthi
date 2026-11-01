"""Local CPU profiling on the same decoded frames, without preview or file pacing."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--size", type=int, default=416)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--posture-every", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--pose-backend", choices=("yolo", "rtmpose", "onnx", "openvino"), default="yolo")
    parser.add_argument("--pose-model-dir", type=Path)
    parser.add_argument("--posture-backend", choices=("torch", "onnx", "openvino"), default="torch")
    parser.add_argument("--posture-model-dir", type=Path)
    parser.add_argument("--cv-threads", type=int, choices=range(1, 9))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import cv2
    import torch
    from ai.fusion_detector import FusionDetector

    cap = cv2.VideoCapture(args.video)
    frames = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        indices = {round(t * fps) for t in (4, 5, 6, 6.4, 6.7, 7, 7.5, 8, 9, 10, 12, 14)}
        index = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if index in indices:
                frames.append(image)
            index += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError("No benchmark frames decoded.")
    if args.output and args.output.exists():
        raise ValueError("Choose a new benchmark output path.")
    detector = None
    def predict(image):
        if not args.worker:
            return detector.predict(image)
        error = []
        def run():
            try:
                detector.predict(image)
            except Exception as exc:
                error.append(exc)
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        if error:
            raise error[0]

    results = []
    cv_profiles = [args.cv_threads] if args.cv_threads else list(dict.fromkeys((cv2.getNumThreads(), 1)))
    for cv_threads in cv_profiles:
        cv2.setNumThreads(cv_threads)
        for threads in args.threads:
            # ORT/OpenVINO thread budgets are fixed at session compilation, so
            # changing Torch's global setting alone cannot benchmark them.
            detector = FusionDetector(image_size=args.size, threads=threads,
                                      posture_every=args.posture_every,
                                      pose_backend=args.pose_backend, pose_model_dir=args.pose_model_dir,
                                      posture_backend=args.posture_backend, posture_model_dir=args.posture_model_dir)
            torch.set_num_threads(threads)
            predict(frames[0])
            detector.reset()
            pose, posture, total = [], [], []
            for image in frames:
                started = time.perf_counter()
                predict(image)
                total.append((time.perf_counter() - started) * 1000)
                pose.append(detector.timings.pose_ms)
                posture.append(detector.timings.posture_ms)
            row = {"size": args.size, "inference_threads": threads,
                              "pose_backend": args.pose_backend, "posture_backend": args.posture_backend,
                              "posture_every": args.posture_every,
                              "pose_model_dir": str(args.pose_model_dir) if args.pose_model_dir else None,
                              "posture_model_dir": str(args.posture_model_dir) if args.posture_model_dir else None,
                              "opencv_threads": cv_threads, "frames": len(frames),
                              "pose_mean_ms": round(statistics.mean(pose), 2),
                              "posture_mean_ms": round(statistics.mean(posture), 2),
                              "total_mean_ms": round(statistics.mean(total), 2),
                              "total_p90_ms": round(sorted(total)[int(0.9 * (len(total) - 1))], 2),
                              "total_p95_ms": round(sorted(total)[math.ceil(0.95 * len(total)) - 1], 2),
                              "throughput_fps": round(1000 / statistics.mean(total), 2)}
            results.append(row)
            print(json.dumps(row), flush=True)
    if args.output:
        report = {"video": str(Path(args.video).resolve()),
                  "note": "Twelve decoded frames; pose+posture+recovery prediction, excluding decode, "
                          "fall logic, display, startup, pacing and notification. Not a live scheduling test.",
                  "results": results}
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
