"""Generate a local synthetic camera-pipeline fixture, NOT fall-test evidence."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    parser = argparse.ArgumentParser(description="Tạo video kiểm thử pipeline, không có dữ liệu người/ngã.")
    parser.add_argument("--output", default="data/test_videos/synthetic.avi")
    parser.add_argument("--seconds", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300:
        parser.error("--seconds phải trong [1, 300].")
    path = Path(args.output).resolve()
    if path.exists():
        parser.error("Tệp đã tồn tại; chọn tên khác để không ghi đè.")
    if path.suffix.lower() != ".avi":
        parser.error("Hãy dùng đuôi .avi (codec MJPG).")
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = 20
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 360))
    if not writer.isOpened():
        raise SystemExit("Không mở được MJPG VideoWriter.")
    try:
        for index in range(args.seconds * fps):
            image = np.full((360, 640, 3), 35, dtype=np.uint8)
            x = 30 + (index * 5) % 580
            cv2.circle(image, (x, 230), 20, (60, 200, 240), -1)
            cv2.putText(image, "SYNTHETIC PIPELINE TEST - NOT A FALL DATASET", (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
            cv2.putText(image, f"frame={index} media={index / fps:.3f}s", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 1)
            writer.write(image)
    finally:
        writer.release()
    print(f"Đã tạo {path}. Thời gian trong video là thời gian media, không phải đồng hồ camera.")


if __name__ == "__main__":
    main()
