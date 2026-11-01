"""Compare paired evaluate_videos reports, rescoring events against identical labels.

Exit codes: 0 = labeled regression gate passed, 1 = regression, 2 = invalid
inputs or no timestamp labels. Cached replay measures temporal logic only.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.evaluation import score_events, summarize


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _index(report):
    if not isinstance(report, dict) or not isinstance(report.get("videos"), list) or not report["videos"]:
        raise ValueError("Report must contain a nonempty videos list.")
    indexed = {}
    for video in report["videos"]:
        if not isinstance(video, dict):
            raise ValueError("Every video must be an object.")
        name = video.get("video")
        if not isinstance(name, str) or not name or name in indexed:
            raise ValueError("Video names must be nonempty and unique.")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(video.get("video_sha256", ""))):
            raise ValueError(f"{name}: missing or invalid video SHA256.")
        duration = video.get("duration")
        if not _number(duration) or duration <= 0:
            raise ValueError(f"{name}: invalid duration.")
        if not isinstance(video.get("extraction_settings"), dict) or not video["extraction_settings"]:
            raise ValueError(f"{name}: missing extraction settings.")
        if not isinstance(video.get("fall_config"), dict):
            raise ValueError(f"{name}: missing fall configuration.")
        if not isinstance(video.get("events"), list):
            raise ValueError(f"{name}: missing events list.")
        for event in video["events"]:
            if (not isinstance(event, dict) or not isinstance(event.get("state"), str)
                    or not _number(event.get("timestamp"))
                    or not 0 <= event["timestamp"] <= duration):
                raise ValueError(f"{name}: invalid event state or timestamp.")
        if "reference" in video:
            reference = video["reference"]
            if not isinstance(reference, dict) or not isinstance(reference.get("falls"), list):
                raise ValueError(f"{name}: reference must contain timestamped falls.")
            try:
                score_events(video["events"], reference, duration)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{name}: invalid timestamp annotation: {error}") from error
        indexed[name] = video
    return indexed


def _matches(video):
    """Map reference interval index to first matched confirmation delay."""
    matches = {}
    for event in sorted(video["events"], key=lambda item: item["timestamp"]):
        if event["state"] != "DA NGA":
            continue
        for index, fall in enumerate(video["reference"]["falls"]):
            if index not in matches and fall["onset"] <= event["timestamp"] <= fall["end"]:
                matches[index] = event["timestamp"] - fall["onset"]
                break
    return matches


def compare_reports(baseline, candidate, *, max_added_fn=0, max_added_fp=0,
                    max_added_warnings=0, max_delay_increase_s=0.0,
                    allow_extraction_change=False):
    """Return a JSON-safe comparison; improvements never cancel regressions elsewhere.

    max_added_fn counts lost previously matched reference falls, including a loss
    hidden by a newly matched fall in the same clip. Delays compare the same fall.
    Unlabeled videos remain descriptive, and cannot make a gate pass alone.
    """
    budgets = {"max_added_fn": max_added_fn, "max_added_fp": max_added_fp,
               "max_added_warnings": max_added_warnings,
               "max_delay_increase_s": max_delay_increase_s}
    for key, value in budgets.items():
        if not _number(value) or value < 0 or (key != "max_delay_increase_s" and type(value) is not int):
            raise ValueError(f"{key} must be a nonnegative {'number' if key == 'max_delay_increase_s' else 'integer'}.")
    before, after = _index(baseline), _index(candidate)
    if before.keys() != after.keys():
        raise ValueError("Reports must contain exactly the same video names.")
    if type(allow_extraction_change) is not bool:
        raise ValueError("allow_extraction_change must be boolean.")
    rows, scored, unscored, before_scores, after_scores = [], [], [], [], []
    extraction_changes = {}
    changes = {}
    lost_total = fp_total = warning_total = 0
    max_delay = 0.0
    for name in sorted(before):
        old, new = before[name], after[name]
        for key in ("video_sha256", "duration", "reference"):
            if old.get(key) != new.get(key):
                raise ValueError(f"{name}: {key} differs; comparison requires paired inputs.")
        if old["extraction_settings"] != new["extraction_settings"]:
            if not allow_extraction_change:
                raise ValueError(f"{name}: extraction_settings differs; use --allow-extraction-change for model/cadence comparisons.")
            old_rate = old["extraction_settings"].get("sample_fps")
            new_rate = new["extraction_settings"].get("sample_fps")
            if not _number(old_rate) or old_rate <= 0 or old_rate != new_rate:
                raise ValueError(f"{name}: model/cadence comparisons require identical sample_fps.")
            extraction_changes[name] = {
                key: {"baseline": old["extraction_settings"].get(key),
                      "candidate": new["extraction_settings"].get(key)}
                for key in sorted(old["extraction_settings"].keys() | new["extraction_settings"].keys())
                if old["extraction_settings"].get(key) != new["extraction_settings"].get(key)
            }
        changes[name] = {
            key: {"baseline": old["fall_config"].get(key), "candidate": new["fall_config"].get(key)}
            for key in sorted(old["fall_config"].keys() | new["fall_config"].keys())
            if old["fall_config"].get(key) != new["fall_config"].get(key)
        }
        old_counts = Counter(event["state"] for event in old["events"])
        new_counts = Counter(event["state"] for event in new["events"])
        row = {"video": name, "scored": "reference" in old,
               "baseline_confirmations": old_counts["DA NGA"],
               "candidate_confirmations": new_counts["DA NGA"],
               "baseline_warnings": old_counts["NGHI NGA"],
               "candidate_warnings": new_counts["NGHI NGA"],
               "baseline_warning_times_media_s": [e["timestamp"] for e in old["events"] if e["state"] == "NGHI NGA"],
               "candidate_warning_times_media_s": [e["timestamp"] for e in new["events"] if e["state"] == "NGHI NGA"]}
        if "reference" in old:
            scored.append(name)
            a = score_events(old["events"], old["reference"], old["duration"])
            b = score_events(new["events"], new["reference"], new["duration"])
            before_scores.append(a)
            after_scores.append(b)
            old_matches, new_matches = _matches(old), _matches(new)
            lost = sorted(old_matches.keys() - new_matches.keys())
            gained = sorted(new_matches.keys() - old_matches.keys())
            delays = [{"reference_index": index, "baseline_media_s": old_matches[index],
                       "candidate_media_s": new_matches[index],
                       "increase_media_s": new_matches[index] - old_matches[index]}
                      for index in sorted(old_matches.keys() & new_matches.keys())]
            added_fp = max(0, b["fp"] - a["fp"])
            added_warnings = max(0, b["false_warnings"] - a["false_warnings"])
            lost_total += len(lost)
            fp_total += added_fp
            warning_total += added_warnings
            max_delay = max(max_delay, *(item["increase_media_s"] for item in delays)) if delays else max_delay
            row.update(baseline_score=a, candidate_score=b, lost_reference_falls=lost,
                       gained_reference_falls=gained, added_false_confirmations=added_fp,
                       added_false_warnings=added_warnings, paired_delays=delays)
        else:
            unscored.append(name)
        rows.append(row)
    failures = []
    for observed, limit, description in (
        (lost_total, max_added_fn, "lost previously detected reference falls"),
        (fp_total, max_added_fp, "added false confirmations per clip"),
        (warning_total, max_added_warnings, "added false warnings per clip"),
        (max_delay, max_delay_increase_s + 1e-9, "maximum delay increase on the same reference fall"),
    ):
        if observed > limit:
            failures.append(f"{description}: {observed:g} exceeds {limit:g}")
    return {"schema_version": 1, "passed": not failures if scored else None,
            "scope": "Development replay; provisional labels are not an independent held-out evaluation. "
                     "Unscored counts are descriptive. Media-time delay is not end-to-end latency.",
            "scored_videos": scored, "unscored_videos": unscored,
            "baseline_summary": summarize(before_scores), "candidate_summary": summarize(after_scores),
            "thresholds": budgets,
            "regressions": {"lost_reference_falls": lost_total,
                            "added_false_confirmations": fp_total,
                            "added_false_warnings": warning_total,
                            "max_paired_delay_increase_media_s": max_delay},
            "failures": failures, "config_changes": changes,
            "extraction_changes": extraction_changes,
            "warnings": ["Extraction settings differ: this compares model/cadence pipelines, not just temporal logic."] if extraction_changes else [],
            "videos": rows}


def render_markdown(comparison):
    status = {True: "PASS", False: "REGRESSION", None: "UNSCORED"}[comparison["passed"]]
    lines = [f"Evaluation comparison: {status}", "", comparison["scope"], "",
             f"Scored clips: {len(comparison['scored_videos'])}; unscored clips: {len(comparison['unscored_videos'])}.", "",
             "| Video | Confirmations before / after | Warnings before / after | TP before / after | FN before / after | FP before / after |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in comparison["videos"]:
        counts = f"{row['baseline_confirmations']} / {row['candidate_confirmations']}"
        metrics = [f"{row['baseline_score'][key]} / {row['candidate_score'][key]}"
                   if row["scored"] else "unscored" for key in ("tp", "fn", "fp")]
        safe_name = row["video"].replace("|", "\\|").replace("\n", " ")
        warnings = f"{row['baseline_warnings']} / {row['candidate_warnings']}"
        lines.append(f"| {safe_name} | {counts} | {warnings} | {' | '.join(metrics)} |")
    lines += ["", "Regression totals: " + json.dumps(comparison["regressions"], ensure_ascii=False), ""]
    lines += [f"- {failure}" for failure in comparison["failures"]]
    lines += [f"- {warning}" for warning in comparison["warnings"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, help="Write JSON; existing files are refused.")
    parser.add_argument("--markdown", type=Path, help="Write Markdown; existing files are refused.")
    parser.add_argument("--max-added-fn", type=int, default=0)
    parser.add_argument("--max-added-fp", type=int, default=0)
    parser.add_argument("--max-added-warnings", type=int, default=0)
    parser.add_argument("--max-delay-increase-s", type=float, default=0)
    parser.add_argument("--allow-extraction-change", action="store_true",
                        help="Explicitly compare changed model/cadence settings; sampling rate must remain identical.")
    args = parser.parse_args(argv)
    try:
        destinations = [path for path in (args.output, args.markdown) if path is not None]
        if len({path.resolve() for path in destinations}) != len(destinations):
            raise ValueError("JSON and Markdown output paths must differ.")
        for destination in destinations:
            if destination.exists() or not destination.parent.is_dir():
                raise ValueError(f"Output must be new and its parent must exist: {destination}")
        result = compare_reports(
            json.loads(args.baseline.read_text(encoding="utf-8-sig")),
            json.loads(args.candidate.read_text(encoding="utf-8-sig")),
            max_added_fn=args.max_added_fn, max_added_fp=args.max_added_fp,
            max_added_warnings=args.max_added_warnings,
            max_delay_increase_s=args.max_delay_increase_s,
            allow_extraction_change=args.allow_extraction_change)
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        text = render_markdown(result)
        if args.markdown:
            with args.markdown.open("x", encoding="utf-8") as stream:
                stream.write(text)
        print(text)
        return 2 if result["passed"] is None else 0 if result["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Evaluation comparison failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())