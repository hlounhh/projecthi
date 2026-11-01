"""Trace cached geometry without model inference."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.fall_engine import FallConfig, FallEngine, PersonObservation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--start", type=float, default=5)
    parser.add_argument("--end", type=float, default=10)
    args = parser.parse_args()
    data = json.loads(args.cache.read_text(encoding="utf-8"))
    engine = FallEngine(FallConfig(fallen_label_threshold=.4, fallen_label_reset_threshold=.2))
    for row in data["rows"]:
        people = tuple(PersonObservation(**p) for p in row["people"])
        events = engine.update(people, row["timestamp"], tuple(row["shape"]))
        if not args.start <= row["timestamp"] <= args.end:
            continue
        parts = []
        for track in engine._tracks.values():
            observation = next((p for p in people if p.xyxy == track.xyxy), None)
            sample = engine._sample(observation, row["timestamp"]) if observation else None
            base = track.baseline
            parts.append({"id": track.track_id, "state": track.state,
                          "angle": round(sample.angle, 1) if sample else None,
                          "height": round(sample.height) if sample else None,
                          "torso_y": round(sample.torso_y) if sample else None,
                          "aspect": round(sample.aspect, 2) if sample else None,
                          "legs": sample.lower_body if sample else None,
                          "base": round(base.timestamp, 2) if base else None,
                          "motion": track.motion_height,
                          "verify": track.verify_since,
                          "occlusion_candidate": track.occlusion.candidate,
                          "occlusion_budget": round(track.occlusion.observed_down, 2),
                          "occlusion_upright_span": round(
                              max((p[0] for p in track.occlusion.history if p[4]), default=0.)
                              - min((p[0] for p in track.occlusion.history if p[4]), default=0.), 2),
                          "features": {k: round(v, 2) if type(v) is float else v
                                       for k, v in track.features.items()},
                          "kp": [round(observation.keypoints[i][2], 2) for i in (5, 6, 11, 12, 13, 14, 15, 16)]
                                if observation and observation.keypoints else []})
        print(round(row["timestamp"], 2), json.dumps(parts), [(e.state, e.reason) for e in events])


if __name__ == "__main__":
    main()
