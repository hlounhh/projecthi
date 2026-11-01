"""Extract fresh local model observations around a selected video interval."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--recovery-every", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.start < args.end or args.recovery_every < 1:
        parser.error("Choose a new output directory, an ordered nonnegative interval, and a positive recovery cadence.")
    import cv2
    from ai.fusion_detector import FusionDetector
    from ai.fall_engine import FallConfig, FallEngine
    from ai.inference_worker import InferenceResult
    from utils.ai_preview import draw_detections
    from utils.fall_preview import draw_tracks

    args.output.mkdir(parents=True, exist_ok=False)
    detector = FusionDetector(image_size=416, posture_every=3,
                              recovery_every=args.recovery_every)
    engine = FallEngine(FallConfig(fallen_label_threshold=.4, fallen_label_reset_threshold=.2))
    cap = cv2.VideoCapture(str(args.video), cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        cap.release()
        raise ValueError("Cannot open local video.")
    start_frame = max(0, int(args.start * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rows, events = [], []
    index, next_sample, next_image = start_frame, args.start, args.start
    try:
        while index / fps <= args.end and cap.grab():
            timestamp = index / fps
            if timestamp + 1e-9 >= next_sample:
                next_sample += .1
                ok, image = cap.retrieve()
                if not ok:
                    raise ValueError("Cannot decode frame.")
                people = detector.predict(image)
                current = engine.update(people, timestamp, image.shape)
                events.extend(asdict(e) for e in current)
                rows.append({"timestamp": timestamp, "shape": image.shape,
                             "people": [asdict(p) for p in people],
                             "timings": asdict(detector.timings)})
                if timestamp >= next_image:
                    next_image += .5
                    tracks = engine.snapshot(timestamp)
                    result = InferenceResult(people, index, 0., 0., 0., image.shape)
                    draw_detections(image, result, boxes=False, tracks=tracks)
                    draw_tracks(image, tracks, image.shape)
                    cv2.imwrite(str(args.output / f"frame_{timestamp:.2f}.jpg"), image)
            index += 1
    finally:
        cap.release()
    record = {"video": args.video.name, "interval": [args.start, args.end],
              "cold_start": True, "settings": {"image_size": 416, "sample_fps": 10,
                                                   "posture_every": 3,
                                                   "recovery_every": args.recovery_every},
              "rows": rows, "events": events,
              "fall_config": asdict(engine.config)}
    (args.output / "observations.json").write_text(json.dumps(record), encoding="utf-8")
    print("Samples:", len(rows), "confirmations:",
          [round(e["timestamp"], 2) for e in events if e["state"] == "DA NGA"], flush=True)


if __name__ == "__main__":
    main()
