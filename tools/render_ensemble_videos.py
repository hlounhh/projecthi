import os
import sys
from pathlib import Path
from types import SimpleNamespace
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.fusion_detector import FusionDetector
from ai.fall_engine import FallEngine
from utils.fall_preview import draw_tracks
from utils.ai_preview import draw_detections
from utils.preview import draw_status

STATE_MAP = {
    "BINH THUONG": "BINH THUONG",
    "NGHI NGA": "NGHI NGA",
    "DA NGA": "DA NGA",
    "MAT DAU": "MAT DAU",
}

def render_ensemble_video(video_in: str, video_out: str, max_fps=10):
    print(f"=== Dang xu ly: {video_in} -> {video_out} ===")
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        print(f"Loi: Khong mo duoc {video_in}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Resize để xuất web nhẹ, mượt mà (width=640)
    out_w = 640
    out_h = int(height * (out_w / width))
    if out_h % 2 != 0:
        out_h += 1

    detector = FusionDetector(
        confidence=0.4,
        image_size=416,
        threads=2,
        posture_every=3,
        pose_backend="yolo",
        posture_backend="ensemble"
    )
    engine = FallEngine()

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(video_out, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_out, fourcc, fps, (out_w, out_h))

    frame_idx = 0
    sample_interval = 1.0 / max_fps
    next_sample_time = 0.0
    last_people = ()
    last_tracks = ()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = frame_idx / fps

        if timestamp + 1e-6 >= next_sample_time:
            next_sample_time += sample_interval
            people = detector.predict(frame)
            engine.update(people, timestamp, frame.shape[:2])
            last_people = people
            last_tracks = engine.snapshot(timestamp)

        display_img = cv2.resize(frame, (out_w, out_h))

        # Vẽ skeleton/posture
        if last_people:
            mock_result = SimpleNamespace(
                detections=last_people,
                source_shape=frame.shape[:2]
            )
            draw_detections(display_img, mock_result, boxes=False, tracks=last_tracks)

        # Vẽ trạng thái track (DA NGA viền đỏ)
        if last_tracks:
            draw_tracks(display_img, last_tracks, frame.shape[:2])

        # Status header
        labels = [f"ENSEMBLE PYTORCH | {timestamp:.1f}s"]
        if not last_tracks:
            labels.append("KHONG CO NGUOI")
        elif any(t.state == "DA NGA" for t in last_tracks):
            labels.append("PHAT HIEN NGA!")
        elif any(t.state == "NGHI NGA" for t in last_tracks):
            labels.append("NGHI NGA - DANG THEO DOI")
        else:
            labels.append("THEO DOI BINH THUONG")

        draw_status(display_img, labels)
        writer.write(display_img)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"[{Path(video_in).name}] Frame {frame_idx}/{total_frames} ({timestamp:.1f}s)", flush=True)

    cap.release()
    writer.release()
    print(f"Xong {video_out} ({frame_idx}/{total_frames} frames)", flush=True)

if __name__ == "__main__":
    out_dir = Path("C:/Users/DELL/.zcode/v2/visualizations/6367141b4803484478728f02d31e99ae54f7f4c7f721142f791d1a9c980f456e")
    videos = ["video.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
    for v in videos:
        v_in = str(ROOT / v)
        v_out = str(out_dir / f"result_{v}")
        render_ensemble_video(v_in, v_out)
