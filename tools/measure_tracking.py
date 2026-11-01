"""Describe cached tracking continuity; these counts are not identity accuracy."""

import argparse
import json
from pathlib import Path
import sys
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.fall_engine import FallConfig, FallEngine, PersonObservation


def measure(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    engine = FallEngine(FallConfig(fallen_label_threshold=.4, fallen_label_reset_threshold=.2))
    identities, events = set(), []
    counts = dict(samples=0, detected=0, visible_track=0, retained_box_only=0, reliable_pose=0)
    inference_ms = []
    for row in data["rows"]:
        people = tuple(PersonObservation(**p) for p in row["people"])
        events.extend(engine.update(people, row["timestamp"], tuple(row["shape"])))
        tracks = engine.snapshot(row["timestamp"])
        counts["samples"] += 1
        counts["detected"] += bool(people)
        counts["visible_track"] += any(t.visible for t in tracks)
        counts["retained_box_only"] += bool(tracks) and not any(t.visible for t in tracks)
        counts["reliable_pose"] += any(t.observation_quality for t in tracks)
        identities.update(t.track_id for t in tracks)
        inference_ms.append(sum(row["timings"].values()))
    times = sorted(inference_ms)
    return {"video": data["video"], "duration": data["duration"], "counts": counts,
            "distinct_track_ids": len(identities),
            "mean_model_ms": sum(times) / max(1, len(times)),
            "p95_model_ms": times[min(len(times) - 1, int(len(times) * .95))] if times else None,
            "warnings": [{"time": e.timestamp, "reason": e.reason} for e in events if e.state == "NGHI NGA"],
            "confirmed": [e.timestamp for e in events if e.state == "DA NGA"],
            "events": [asdict(e) for e in events]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("caches", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [measure(path) for path in args.caches]
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(records, stream, indent=2, ensure_ascii=False)
    for record in records:
        print(json.dumps({k: v for k, v in record.items() if k != "events"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
