"""Run the real preview and save its image region once a fall is confirmed."""

import argparse
import ctypes
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("screenshot")
    parser.add_argument("--at", type=float, help="Capture after this video timestamp instead of a confirmed fall.")
    parser.add_argument("--duration", type=float, default=100)
    args = parser.parse_args()
    target = Path(args.screenshot)
    if target.exists():
        parser.error("Screenshot already exists.")
    import cv2
    from PIL import ImageGrab
    import utils.fall_preview as preview
    import utils.preview as status_preview

    draw, show, wait, status_draw = preview.draw_tracks, cv2.imshow, cv2.waitKey, status_preview.draw_status
    confirmed, captured = False, False

    def draw_tracks(image, tracks, shape):
        nonlocal confirmed
        if args.at is None:
            confirmed |= any(t.state == "DA NGA" for t in tracks)
        draw(image, tracks, shape)

    def draw_status(image, labels):
        nonlocal confirmed
        if args.at is not None:
            fields = labels[0].split("|")
            if fields[0].strip() == "PHAN TICH VIDEO" and len(fields) > 1:
                confirmed = float(fields[1].strip().removesuffix("s")) >= args.at
        status_draw(image, labels)

    def imshow(window, image):
        nonlocal captured
        show(window, image)
        if confirmed and not captured:
            wait(1)
            ctypes.windll.user32.SetProcessDPIAware()
            x, y, width, height = cv2.getWindowImageRect(window)
            ImageGrab.grab(bbox=(x, y, x + width, y + height)).save(target)
            captured = True
            print(f"Preview screenshot: {target}")

    def wait_key(delay):
        key = wait(delay)
        return ord("q") if captured else key

    preview.draw_tracks, cv2.imshow, cv2.waitKey = draw_tracks, imshow, wait_key
    status_preview.draw_status = draw_status
    sys.argv = ["main.py", "--video", args.video, "--ai", "--duration", str(args.duration)]
    try:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"), run_name="__main__")
    finally:
        preview.draw_tracks, cv2.imshow, cv2.waitKey = draw, show, wait
        status_preview.draw_status = status_draw
    if not captured:
        raise RuntimeError("Requested preview was not captured.")


if __name__ == "__main__":
    main()
