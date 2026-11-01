"""Cache local observations once; replay temporal logic without rerunning models."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def extraction_provenance(pose_backend, posture_backend, pose_model_dir=None, posture_model_dir=None):
    """Record the verified model identity and preprocessing of a fresh extraction."""
    from ai.pose_detector import POSE_SHA256
    from ai.posture_detector import EXPECTED_SHA256

    models = {}
    from ai.posture_ensemble import MEMBERS
    for task, backend, directory in (("pose", pose_backend, pose_model_dir),
                                      ("posture", posture_backend, posture_model_dir)):
        if task == "posture" and backend == "ensemble" and directory:
            # Explicit OpenVINO ensemble export: record the parent and member manifests.
            from ai.ensemble_openvino import BUNDLE_FILE
            raw = (Path(directory) / BUNDLE_FILE).read_bytes()
            bundle = json.loads(raw)
            models[task] = {"manifest": BUNDLE_FILE,
                            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                            "precision": bundle.get("precision"),
                            "image_size": bundle.get("image_size"),
                            "members": bundle.get("members"),
                            "preprocessing": bundle.get("preprocessing")}
            continue
        if task == "posture" and backend == "ensemble":
            # Torch ensemble: members are pinned here and letterboxed to a static square.
            models[task] = {"ensemble_members": [
                {"name": name, "path": str(Path(path)), "sha256": sha}
                for name, path, sha in MEMBERS
            ], "preprocessing": "static_square_letterbox_rect_false"}
        elif backend in ("onnx", "openvino"):
            raw = (Path(directory) / "manifest.json").read_bytes()
            manifest = json.loads(raw)
            models[task] = {"manifest_sha256": hashlib.sha256(raw).hexdigest(),
                            "source_sha256": manifest["source_sha256"], "files": manifest["files"],
                            "preprocessing": "static_square_letterbox_rect_false"}
        elif backend == "rtmpose":
            from ai.rtmpose_detector import MODEL_SHA256
            models[task] = {"files": dict(MODEL_SHA256),
                            "runtime": "onnxruntime",
                            "preprocessing": "yolox_416_top_left_pad_rtmpose_192x256_affine"}
        else:
            models[task] = {"source_sha256": POSE_SHA256 if task == "pose" else EXPECTED_SHA256,
                            "preprocessing": "ultralytics_default_rectangular_letterbox"}
    versions = {}
    for package in ("torch", "ultralytics", "numpy", "opencv-python", "opencv-contrib-python",
                    "onnx", "onnxruntime", "openvino", "rtmlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    import cv2
    versions["cv2_runtime"] = cv2.__version__
    return {"models": models, "versions": versions}


def extract(videos, output, image_size=416, *, posture_every=3, pose_backend="yolo",
            pose_model_dir=None, posture_backend="torch", posture_model_dir=None,
            threads=2):
    import cv2
    from ai.fusion_detector import FusionDetector

    output.mkdir(parents=True, exist_ok=False)
    detector = FusionDetector(image_size=image_size, posture_every=posture_every,
                              threads=threads, pose_backend=pose_backend,
                              pose_model_dir=pose_model_dir, posture_backend=posture_backend,
                              posture_model_dir=posture_model_dir)
    provenance = extraction_provenance(pose_backend, posture_backend, pose_model_dir, posture_model_dir)
    for video in videos:
        detector.reset()
        cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not 0 < fps <= 240 or count <= 0:
            cap.release()
            raise ValueError(f"Invalid video metadata: {video.name}")
        duration = count / fps
        thumbnail_indices = {round(i * (count - 1) / 23) for i in range(24)}
        thumbnails, rows = [], []
        index, next_sample = 0, 0.
        started = time.perf_counter()
        try:
            while cap.grab():
                timestamp = index / fps
                sample = timestamp + 1e-9 >= next_sample
                thumbnail = index in thumbnail_indices
                if sample or thumbnail:
                    ok, image = cap.retrieve()
                    if not ok:
                        raise ValueError(f"Cannot decode {video.name} frame {index}")
                    if thumbnail:
                        small = cv2.resize(image, (320, 180))
                        cv2.rectangle(small, (0, 0), (150, 23), (0, 0, 0), -1)
                        cv2.putText(small, f"{video.name} {timestamp:.2f}s", (5, 17),
                                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                        thumbnails.append(small)
                    if sample:
                        next_sample += .1
                        people = detector.predict(image)
                        rows.append({"timestamp": timestamp, "shape": image.shape,
                                     "people": [asdict(p) for p in people],
                                     "timings": asdict(detector.timings)})
                index += 1
        finally:
            cap.release()
        import numpy as np
        canvas = np.zeros((6 * 180, 4 * 320, 3), dtype=np.uint8)
        for i, image in enumerate(thumbnails):
            y, x = (i // 4) * 180, (i % 4) * 320
            canvas[y:y + 180, x:x + 320] = image
        cv2.imwrite(str(output / f"{video.stem}_frames.jpg"), canvas)
        record = {"video": video.name, "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                  "fps": fps, "duration": duration, "decoded_frames": index,
                  "settings": {"image_size": image_size, "sample_fps": 10, "posture_every": posture_every,
                               "pose_backend": pose_backend, "posture_backend": posture_backend,
                               "threads": threads,
                               "provenance": provenance,
                               "pose_confidence": detector.pose.confidence,
                               "local_crop_recovery": detector.recovery},
                  "rows": rows}
        (output / f"{video.stem}.json").write_text(json.dumps(record), encoding="utf-8")
        print(f"{video.name}: {duration:.2f}s, {len(rows)} samples, {time.perf_counter()-started:.1f}s compute", flush=True)


def replay(cache, labels=None, config=None):
    from ai.fall_engine import CONFIRMED_FALL, SUSPECTED_FALL, FallConfig, FallEngine, PersonObservation

    results = []
    for path in sorted(cache.glob("video*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        values = {"fallen_label_threshold": 0.4, "fallen_label_reset_threshold": 0.2}
        values.update(config or {})
        engine = FallEngine(FallConfig(**values))
        events, unreliable, absent = [], 0, 0
        identities, visible_samples, retained_samples = set(), 0, 0
        for row in data["rows"]:
            people = tuple(PersonObservation(**p) for p in row["people"])
            events.extend(engine.update(people, row["timestamp"], tuple(row["shape"])))
            tracks = engine.snapshot(row["timestamp"])
            unreliable += any(not t.observation_quality for t in tracks)
            absent += not tracks
            identities.update(t.track_id for t in tracks)
            visible_samples += any(t.visible for t in tracks)
            retained_samples += bool(tracks) and not any(t.visible for t in tracks)
        confirmations = [asdict(e) for e in events if e.state == CONFIRMED_FALL]
        suspicions = [asdict(e) for e in events if e.state == SUSPECTED_FALL]
        result = {"video": data["video"], "duration": data["duration"],
                  "video_sha256": data["sha256"], "extraction_settings": data["settings"],
                  "fall_config": asdict(engine.config),
                  "confirmed": confirmations, "suspected": suspicions,
                  "events": [asdict(e) for e in events],
                  "distinct_track_ids": len(identities),
                  "visible_track_sample_fraction": visible_samples / max(1, len(data["rows"])),
                  "retained_box_sample_fraction": retained_samples / max(1, len(data["rows"])),
                  "no_track_sample_fraction": absent / max(1, len(data["rows"])),
                  "unreliable_sample_fraction": unreliable / max(1, len(data["rows"]))}
        if labels and data["video"] in labels:
            label = labels[data["video"]]
            result["reference"] = label
            from utils.evaluation import score_events
            result["score"] = score_events(result["events"], label, data["duration"])
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", nargs="+", type=Path)
    parser.add_argument("--size", type=int, choices=(320, 416, 512, 640), default=416)
    parser.add_argument("--posture-every", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--pose-backend", choices=("yolo", "rtmpose", "onnx", "openvino"), default="yolo")
    parser.add_argument("--pose-model-dir", type=Path)
    parser.add_argument("--posture-backend", choices=("torch", "onnx", "openvino", "ensemble"), default="torch")
    parser.add_argument("--posture-model-dir", type=Path)
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.extract:
        extract(args.extract, args.cache, args.size, posture_every=args.posture_every,
                pose_backend=args.pose_backend, pose_model_dir=args.pose_model_dir,
                posture_backend=args.posture_backend, posture_model_dir=args.posture_model_dir,
                threads=args.threads)
    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else None
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    results = replay(args.cache, labels, config)
    from utils.evaluation import summarize
    summary = summarize(r["score"] for r in results if "score" in r)
    report = {"evaluation_set": "development; provisional visual labels, not held-out",
              "config_overrides": config, "summary": summary, "videos": results}
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
    if labels:
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    for result in results:
        print(result["video"], "confirmed", [round(e["timestamp"], 2) for e in result["confirmed"]],
              "suspected", [(round(e["timestamp"], 2), e["reason"]) for e in result["suspected"]], flush=True)


if __name__ == "__main__":
    main()
