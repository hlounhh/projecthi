"""Replay saved interval observations without loading models or decoding video."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.fall_engine import CONFIRMED_FALL, FallConfig, FallEngine, PersonObservation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("observations", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = []
    for path in args.observations:
        data = json.loads(path.read_text(encoding="utf-8"))
        engine = FallEngine(FallConfig(**data["fall_config"]))
        events = []
        for row in data["rows"]:
            people = tuple(PersonObservation(**p) for p in row["people"])
            events.extend(engine.update(people, row["timestamp"], tuple(row["shape"])))
        confirmations = [asdict(e) for e in events if e.state == CONFIRMED_FALL]
        reports.append({"cache": str(path), "video": data["video"],
                        "interval": data["interval"], "cold_start": data["cold_start"],
                        "samples": len(data["rows"]), "fall_config": asdict(engine.config),
                        "confirmed": confirmations, "events": [asdict(e) for e in events]})
        print(path.parent.name, "confirmed", [round(e["timestamp"], 2) for e in confirmations])
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"evaluation_type": "cached development intervals, not fresh model inference",
                   "intervals": reports}, stream, indent=2)


if __name__ == "__main__":
    main()
