"""Local fall-event records; no external notification or posture chatter."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from ai.fall_engine import CONFIRMED_FALL, NORMAL


class FallEventWriter:
    def __init__(self, log_dir: Path):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.session_id = str(uuid.uuid4())
        self.path = log_dir / f"fall_events_{stamp}_{self.session_id[:8]}.jsonl"
        self._file = self.path.open("x", encoding="utf-8")
        self._confirmed_events = set()

    def write(self, event):
        record = asdict(event)
        record["session_id"] = self.session_id
        record["utc_recorded_at"] = datetime.now(timezone.utc).isoformat()
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        state = getattr(event.state, "value", event.state)
        if state == CONFIRMED_FALL:
            self._confirmed_events.add(event.event_id)
            message = "ĐÃ NGÃ — hệ thống phát hiện, kiểm tra ngay"
        elif state == NORMAL and event.event_id in self._confirmed_events:
            self._confirmed_events.discard(event.event_id)
            message = "ĐÃ QUAN SÁT PHỤC HỒI SAU NGÃ"
        else:
            return
        print(f"[#{event.track_id}] {message}")

    def close(self):
        self._file.close()
