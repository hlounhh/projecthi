"""Event-level scoring; warnings and confirmed falls are separate endpoints."""

import math


def score_events(events, reference, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite.")
    intervals = reference["falls"]
    previous_end = -1
    for fall in intervals:
        onset, end = fall["onset"], fall["end"]
        if not (math.isfinite(onset) and math.isfinite(end)
                and 0 <= onset < end <= duration and onset > previous_end):
            raise ValueError("Fall intervals must be ordered, disjoint and within the clip.")
        previous_end = end
    detections = sorted((e for e in events if e["state"] == "DA NGA"),
                        key=lambda e: e["timestamp"])
    warnings = [e for e in events if e["state"] == "NGHI NGA"]
    used, delays = set(), []
    false_alarms = 0
    for event in detections:
        match = next((i for i, fall in enumerate(intervals)
                      if i not in used and fall["onset"] <= event["timestamp"] <= fall["end"]), None)
        if match is None:
            false_alarms += 1
        else:
            used.add(match)
            delays.append(event["timestamp"] - intervals[match]["onset"])
    false_warnings = sum(not any(f["onset"] <= e["timestamp"] <= f["end"] for f in intervals)
                         for e in warnings)
    return {"tp": len(used), "fn": len(intervals) - len(used), "fp": false_alarms,
            "tn_clip": int(not intervals and not detections),
            "negative_clip_seconds": duration if not intervals else 0,
            "negative_clip_fp": false_alarms if not intervals else 0,
            "false_warnings": false_warnings, "delays_media_s": delays}


def summarize(scores):
    scores = list(scores)
    tp, fn, fp = (sum(s[k] for s in scores) for k in ("tp", "fn", "fp"))
    errors = sum(bool(s["fp"] or s["fn"]) for s in scores)
    negative_seconds = sum(s["negative_clip_seconds"] for s in scores)
    n = len(scores)
    # Descriptive only: clips from one room/person are not independent trials.
    z = 1.96
    if n:
        p = errors / n
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        interval = [max(0, center - half), min(1, center + half)]
    else:
        interval = None
    return {"clips": n, "tp": tp, "fn": fn, "fp": fp,
            "tn_clips": sum(s["tn_clip"] for s in scores),
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "clip_error_rate": errors / n if n else None,
            "clip_error_wilson95": interval,
            "false_warnings": sum(s["false_warnings"] for s in scores),
            "negative_clip_seconds": negative_seconds,
            "false_confirmations_per_negative_hour":
                sum(s["negative_clip_fp"] for s in scores) * 3600 / negative_seconds
                if negative_seconds else None}
