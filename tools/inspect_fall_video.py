"""Local per-frame diagnostic; samples in media time, not real-time benchmarking."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--size", type=int, default=416)
    args = parser.parse_args()
    import cv2
    from ai.fusion_detector import FusionDetector
    from ai.fall_engine import FallEngine

    detector = FusionDetector(image_size=args.size)
    engine = FallEngine()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    index = 0
    next_sample = 0.0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            timestamp = index / fps
            if timestamp + 1e-9 >= next_sample:
                next_sample += 0.1
                people = detector.predict(image)
                events = engine.update(people, timestamp, image.shape)
                tracks = engine.snapshot(timestamp)
                if 5 <= timestamp <= 9:
                    print(round(timestamp, 2), [
                        {"fallen": round(p.fallen_confidence, 2),
                         "legs": [round(p.keypoints[i][2], 2) for i in (5,6,11,12,13,14,15,16)]
                         if p.keypoints else [], "box": [round(v) for v in p.xyxy]}
                        for p in people], [(t.state, dict(t.features)) for t in tracks])
                for event in events:
                    print("EVENT", event)
            index += 1
    finally:
        cap.release()


if __name__ == "__main__":
    main()
