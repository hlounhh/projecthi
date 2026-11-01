"""Inspect one RTSP stream without saving images or printing credentials."""

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ConfigError, load_settings
from camera.rtsp_reader import open_capture


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure RTSP read rate; NOT camera latency.")
    parser.add_argument("--subtype", type=int, choices=(0, 1), default=1)
    parser.add_argument("--transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--low-latency", action="store_true")
    parser.add_argument("--seconds", type=int, default=8)
    args = parser.parse_args()
    if not 3 <= args.seconds <= 120:
        parser.error("--seconds must be between 3 and 120")
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(exc)
        return 2
    parts = urlsplit(settings.source)
    query = dict(parse_qsl(parts.query))
    query["subtype"] = str(args.subtype)
    settings = replace(settings, source=urlunsplit(parts._replace(query=urlencode(query))))
    options = f"rtsp_transport;{args.transport}"
    if args.low_latency:
        # Experimental FFmpeg input options; success does not prove lower latency.
        options += "|fflags;nobuffer|flags;low_delay|analyzeduration;0|probesize;32768"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
    import cv2

    capture = None
    try:
        started = time.perf_counter()
        capture = open_capture(settings)
        if not capture.isOpened():
            print("Cannot open stream. No credentials printed.")
            return 1
        codec_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((codec_value >> (8 * i)) & 255) for i in range(4))
        print(f"subtype={args.subtype} transport={args.transport} low_latency={args.low_latency}")
        print(f"metadata: {capture.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
              f"{capture.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
              f"fps={capture.get(cv2.CAP_PROP_FPS):.2f} codec={codec!r}")
        first = None
        sample_start = None
        arrivals = []
        count = 0
        cpu_start = time.process_time()
        while True:
            ok, image = capture.read()
            now = time.perf_counter()
            if not ok or image is None:
                print("Read failed. No credentials printed.")
                return 1
            count += 1
            if first is None:
                first = now
                sample_start = now
                print(f"first_frame_s={first - started:.3f} decoded_shape={image.shape[1]}x{image.shape[0]}")
            # Ignore the initial burst; source metadata alone is not measured FPS.
            if now - first >= 2:
                arrivals.append(now)
            if now - first >= args.seconds:
                break
        span = arrivals[-1] - arrivals[0] if len(arrivals) >= 2 else 0
        fps = (len(arrivals) - 1) / span if span > 0 else 0
        print(f"frames_total={count} steady_read_fps={fps:.2f} "
              f"cpu_core_equivalents={(time.process_time() - cpu_start) / (now - sample_start):.2f}")
        print("Camera-to-screen latency: NOT MEASURED. Compare with an external clock.")
        return 0
    except Exception:
        print("Probe failed; raw backend error omitted to protect credentials.")
        return 1
    finally:
        if capture is not None:
            capture.release()


if __name__ == "__main__":
    raise SystemExit(main())
