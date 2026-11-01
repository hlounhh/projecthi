"""Local camera + pretrained pose tracking for experimental sudden-fall detection."""

import argparse
import math
from pathlib import Path
import sys
import time

from config import ConfigError, load_settings

# On-screen text must stay ASCII: OpenCV's Hershey fonts cannot render
# Vietnamese diacritics, so the preview uses unaccented Vietnamese.
STATE_PANEL_LABELS = {
    "CONNECTING": "DANG KET NOI",
    "STREAMING": "DANG PHAT HINH",
    "RECONNECTING": "THU KET NOI LAI",
    "EOF": "KET THUC NGUON",
    "ERROR": "LOI NGUON",
    "STOPPED": "DA DUNG",
}


def _ensure_utf8_output():
    # Windows console restore-point: Vietnamese UI strings must survive piping,
    # redirection and legacy cp1252 code pages without UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="IMOU camera — phát hiện nguy cơ ngã bằng --ai.")
    parser.add_argument("--ai", action="store_true", help="Bật pose + theo dõi chuyển động + bộ phát hiện ngã thử nghiệm.")
    parser.add_argument("--fall-config", help="Tệp JSON ghi đè ngưỡng FallConfig; chưa hiệu chỉnh trên dữ liệu thực.")
    parser.add_argument("--ai-fps", type=float, default=10, help="Giới hạn suy luận mỗi giây (mặc định 10).")
    parser.add_argument("--ai-progress-timeout", type=float, default=30.,
                        help="Dừng nếu nguồn sẵn sàng nhưng AI không xuất mẫu mới trong N giây (mặc định 30).")
    parser.add_argument("--ai-size", type=int, choices=(320, 416, 512, 640), default=416)
    parser.add_argument("--ai-conf", type=float, default=0.4)
    parser.add_argument("--ai-threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--pose-backend", choices=("yolo", "rtmpose", "onnx", "openvino"), default="openvino",
                        help="Pose backend (default: OpenVINO INT8).")
    parser.add_argument("--pose-model-dir", help="Export directory; defaults to the INT8 pose export for --ai-size.")
    parser.add_argument("--posture-backend", choices=("torch", "onnx", "openvino", "ensemble"), default="openvino",
                        help="Posture backend (default: OpenVINO INT8).")
    parser.add_argument("--posture-model-dir", help="Export directory; defaults to the INT8 posture export for --ai-size.")
    parser.add_argument("--smooth-keypoints", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable per-track One Euro filtering; overrides the JSON config.")
    parser.add_argument("--motion-compensate", action="store_true",
                        help="Compensate global camera shift (PTZ/tracking cameras); untrusted motion downgrades CONFIRMED to SUSPECTED.")
    parser.add_argument("--posture-every", type=int, choices=range(1, 11), default=3,
                        help="Chạy HF mỗi N mẫu khi pose thấy người; 1 = mọi mẫu.")
    parser.add_argument("--posture-stable", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--video", help="Dùng video cục bộ thay cho RTSP.")
    parser.add_argument("--stream", choices=("main", "sub"), help="Chọn luồng IMOU, ghi đè .env/biến môi trường.")
    parser.add_argument("--loop", action="store_true", help="Lặp video cục bộ.")
    parser.add_argument("--headless", action="store_true", help="Không mở cửa sổ; vẫn ghi CSV.")
    parser.add_argument("--duration", type=float, help="Dừng sau N giây, tính cả thời gian kết nối.")
    args = parser.parse_args()
    if not math.isfinite(args.ai_fps) or not 0 < args.ai_fps <= 60:
        parser.error("--ai-fps phải trong (0, 60].")
    if not math.isfinite(args.ai_progress_timeout) or args.ai_progress_timeout <= 0:
        parser.error("--ai-progress-timeout phải là số hữu hạn > 0.")
    if not math.isfinite(args.ai_conf) or not 0 < args.ai_conf <= 1:
        parser.error("--ai-conf phải trong (0, 1].")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration phải là số hữu hạn > 0.")
    if args.loop and not args.video:
        parser.error("--loop chỉ dùng cùng --video.")
    if args.stream and args.video:
        parser.error("--stream chỉ dùng với camera RTSP.")
    export_root = Path(__file__).resolve().parent / "models" / "accel"
    if args.pose_backend == "openvino" and args.pose_model_dir is None:
        args.pose_model_dir = str(export_root / f"pose-{args.ai_size}-int8" / "int8_openvino_model")
    if args.posture_backend == "openvino" and args.posture_model_dir is None:
        args.posture_model_dir = str(export_root / f"posture-{args.ai_size}-int8" / "int8_openvino_model")
    if args.posture_backend == "ensemble" and args.posture_model_dir is None:
        args.posture_model_dir = str(export_root / f"ensemble-{args.ai_size}-int8")
    return args


def run() -> int:
    _ensure_utf8_output()
    args = parse_args()
    try:
        subtype = {"main": 0, "sub": 1}.get(args.stream)
        settings = load_settings(args.video, stream_subtype=subtype)
    except ConfigError as exc:
        print(f"Lỗi cấu hình: {exc}", file=sys.stderr)
        return 2

    try:
        import cv2
        import numpy as np
        from camera.rtsp_reader import LatestFrameReader, StreamState
        from utils.metrics import FrameCounter, MetricsWriter, model_logs_to_file, setup_logging
        from utils.preview import draw_status, fit_preview
    except ImportError:
        print("Thiếu thư viện. Chạy: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2

    reader = None
    ai_worker = None
    fall_writer = None
    fall_engine = None
    last_ai_sequence = None
    track_source_shape = None
    metrics = None
    logger = None
    exit_code = 0
    stage = "logging_startup"
    try:
        from utils.pipeline_progress import InferenceProgress, InferenceStalled
        progress = InferenceProgress(timeout_seconds=args.ai_progress_timeout)
        logger = setup_logging(settings.log_dir)
        metrics = MetricsWriter(settings.log_dir)
        logger.info("Session started source_type=%s headless=%s", "file" if settings.is_file else "RTSP", args.headless)
        logger.check_health()
        print("Cảnh báo ngã tại máy | Q/Esc hoặc Ctrl+C để dừng."
              if args.ai else "Xem video | Q/Esc hoặc Ctrl+C để dừng.")
        if not settings.is_file:
            from urllib.parse import parse_qs, urlsplit
            selected = parse_qs(urlsplit(settings.source).query).get("subtype", ["unknown"])[0]
            # Never print the full source URL or arbitrary query values.
            selection = {"0": "MAIN (subtype=0)", "1": "SUB (subtype=1)"}.get(selected, "CUSTOM")
            logger.info("Camera stream: %s", selection)
        counter = FrameCounter()
        last_stream_state = None
        detector = None
        if args.ai:
            stage = "ai_configuration"
            import json
            from ai.fusion_detector import FusionDetector
            from ai.fall_engine import FallConfig, FallEngine
            from utils.fall_log import FallEventWriter
            config_values = {}
            if args.fall_config:
                config_values = json.loads(Path(args.fall_config).read_text(encoding="utf-8"))
                if not isinstance(config_values, dict):
                    raise ValueError("Fall config must be a JSON object.")
            config_values.setdefault("fallen_label_threshold", args.ai_conf)
            config_values.setdefault("fallen_label_reset_threshold", min(0.3, config_values["fallen_label_threshold"] / 2))
            if args.smooth_keypoints is not None:
                config_values["keypoint_smoothing"] = args.smooth_keypoints
            fall_engine = FallEngine(FallConfig(**config_values))
            fall_writer = FallEventWriter(settings.log_dir)
            motion_compensated = args.motion_compensate
            if args.motion_compensate:
                from ai.camera_motion import CameraMotionEstimator, MotionCompensatedFallEngine
                fall_engine = MotionCompensatedFallEngine(
                    fall_engine, estimator=CameraMotionEstimator())
            logger.info("Loading pose and posture models on CPU")
            stage = "model_loading"
            with model_logs_to_file(logger):
                detector = FusionDetector(confidence=args.ai_conf, image_size=args.ai_size,
                                          threads=args.ai_threads, posture_every=args.posture_every,
                                          pose_backend=args.pose_backend, pose_model_dir=args.pose_model_dir,
                                          posture_backend=args.posture_backend, posture_model_dir=args.posture_model_dir)
            logger.info("Inference pose_backend=%s posture_backend=%s posture_every=%s keypoint_smoothing=%s",
                        args.pose_backend, args.posture_backend, args.posture_every,
                        fall_engine.config.keypoint_smoothing)
            logger.info("Event log: %s", fall_writer.path)
        stage = "pipeline_startup"
        analysis_time = 0.0
        offline = settings.is_file and args.ai
        if offline:
            from camera.video_reader import VideoAnalysisReader
            reader = VideoAnalysisReader(settings, sample_fps=args.ai_fps, loop_video=args.loop)
            reader.start()
            logger.info("Video sample_fps=%g; playback follows inference speed", args.ai_fps)
        else:
            reader = LatestFrameReader(settings, loop_video=args.loop)
            reader.start()
        if detector is not None:
            from ai.inference_worker import InferenceWorker, WorkerState
            from utils.ai_preview import draw_detections
            from utils.fall_preview import draw_tracks
            ai_worker = InferenceWorker(reader, detector, max_fps=1000 if offline else args.ai_fps,
                                        stale_seconds=settings.stale_seconds)
            ai_worker.start()
        started = time.perf_counter()
        next_report = started
        window = "IMOU | Theo doi nga | Q / Esc de thoat" if args.ai else "IMOU | Giai doan 1 | Q / Esc de thoat"
        source_shape = None
        preview_base = None
        preview_sequence = None
        next_draw = 0.0
        preview_height = max(240, round(settings.display_width * 9 / 16))
        stage = "preview_startup"
        if not args.headless:
            # Handle aspect ratio ourselves: Win32 HighGUI can stretch WINDOW_NORMAL.
            cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
            cv2.resizeWindow(window, settings.display_width, round(settings.display_width * 9 / 16))
        while True:
            stage = "logging_persistence"
            logger.check_health()
            now = time.perf_counter()
            stage = "pipeline_snapshot"
            snapshot = reader.snapshot()
            ai_status = ai_worker.snapshot(snapshot) if ai_worker else None
            stalled = progress.stalled(snapshot, ai_status, now=now,
                                       stale_seconds=settings.stale_seconds, offline=offline) if ai_worker else False
            stage = "inference_progress"
            if stalled:
                raise InferenceStalled()
            if ai_status and ai_status.state == WorkerState.ERROR:
                print("Lỗi luồng suy luận [inference_worker]. GIÁM SÁT ĐÃ DỪNG.", file=sys.stderr)
                exit_code = 1
                break
            ai_result = ai_status.result if ai_status else None
            if snapshot.state != last_stream_state:
                last_stream_state = snapshot.state
                stream_message = {
                    StreamState.CONNECTING: "Đang kết nối camera/nguồn video...",
                    StreamState.RECONNECTING: "Mất nguồn hình, đang thử kết nối lại...",
                    StreamState.STREAMING: "Đã nhận hình.",
                }.get(snapshot.state)
                if stream_message:
                    logger.info("Source state: %s", snapshot.state.value)
                    if snapshot.state == StreamState.RECONNECTING:
                        print(stream_message)
            stage = "fall_analysis"
            packet = snapshot.frame
            if fall_engine is not None:
                events = ()
                if ai_result is not None and ai_result.source_sequence != last_ai_sequence:
                    last_ai_sequence = ai_result.source_sequence
                    analysis_time = ai_result.source_media_ms / 1000 if offline else ai_result.source_received_at
                    frame = packet.image if packet is not None and \
                        packet.image.shape[:2] == ai_result.source_shape else None
                    events = fall_engine.update(ai_result.detections, analysis_time,
                                                ai_result.source_shape,
                                                **({"frame": frame} if motion_compensated else {}))
                    track_source_shape = ai_result.source_shape
                    if offline:
                        reader.acknowledge(ai_result.source_sequence)
                elif ai_result is None and not offline:
                    events = fall_engine.mark_unobserved(now, reason="camera_or_inference_unavailable")
                stage = "event_log"
                for event in events:
                    fall_writer.write(event)
            tracks = fall_engine.snapshot(analysis_time if offline else now) if fall_engine else ()
            age = max(0, now - packet.received_at) if packet else None
            fresh = (
                packet is not None
                and (offline or age <= settings.stale_seconds)
                and snapshot.state == StreamState.STREAMING
            )
            is_new = fresh and packet.sequence > counter.last_sequence
            if packet is not None and source_shape != packet.image.shape[:2]:
                source_shape = packet.image.shape[:2]
                preview_height = max(1, round(source_shape[0] * settings.display_width / source_shape[1]))
                logger.info("Decoded frame: %sx%s", source_shape[1], source_shape[0])
                if not args.headless:
                    cv2.resizeWindow(window, settings.display_width, preview_height)
            stage = "preview"
            if not args.headless and now >= next_draw:
                next_draw = now + 1 / 30
                width = settings.display_width
                if fresh:
                    if preview_sequence != packet.sequence:
                        preview_base = cv2.resize(packet.image, (width, preview_height))
                        preview_sequence = packet.sequence
                    image = preview_base.copy()
                    # Old boxes are approximate overlays, never tracked positions.
                    if (ai_result and ai_result.source_shape == tuple(packet.image.shape)
                            and ai_result.source_sequence <= packet.sequence):
                        draw_detections(image, ai_result, boxes=False, tracks=tracks)
                else:
                    image = np.zeros((preview_height, width, 3), dtype=np.uint8)
                labels = [STATE_PANEL_LABELS.get(snapshot.state.value, snapshot.state.value)
                          if not offline else "PHAN TICH VIDEO"]
                if offline:
                    labels[0] += f" | {analysis_time:.1f}s"
                elif not fresh:
                    labels[0] += " | MAT HINH"
                if ai_status:
                    labels[0] += f" | AI {ai_status.inference_fps:.1f} FPS"
                    if not tracks:
                        labels.append("KHONG CO NGUOI")
                    elif any(not track.visible for track in tracks):
                        labels.append("BI CHE")
                    elif any(not track.observation_quality for track in tracks):
                        labels.append("BANG CHUNG YEU")
                draw_status(image, labels)
                # Risk borders must remain above the status panel.
                if (fresh and fall_engine is not None
                        and track_source_shape == tuple(packet.image.shape)):
                    draw_tracks(image, tracks, track_source_shape)
                _, _, viewport_width, viewport_height = cv2.getWindowImageRect(window)
                if viewport_width > 0 and viewport_height > 0:
                    image = fit_preview(image, viewport_width, viewport_height)
                cv2.imshow(window, image)
                if is_new:
                    counter.record(packet.sequence, time.perf_counter())
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            elif args.headless and is_new:
                counter.record(packet.sequence, time.perf_counter())

            if now >= next_report:
                stage = "metrics_persistence"
                metrics.write(snapshot, counter, settings.stale_seconds, ai_status,
                              tracks if fall_engine else None)
                next_report = now + settings.metrics_interval_seconds
            if snapshot.state in {StreamState.EOF, StreamState.ERROR}:
                logger.info("Source stopped: %s", snapshot.state.value)
                if snapshot.frames_received == 0 or snapshot.state == StreamState.ERROR:
                    print(f"Lỗi nguồn video: {snapshot.state.value}", file=sys.stderr)
                    exit_code = 1
                break
            if args.duration is not None and now - started >= args.duration:
                break
            time.sleep(0.005)
    except KeyboardInterrupt:
        if logger:
            logger.info("Session interrupted")
    except Exception as exc:
        # Exception messages may contain camera URLs; report the type only.
        error_type = type(exc).__name__
        if error_type == "InferenceStalled":
            print("AI không xuất mẫu mới trong thời hạn cấu hình; chưa có supervisor tự khởi động lại.", file=sys.stderr)
        print(f"Lỗi vận hành [{stage}] ({error_type}). GIÁM SÁT ĐÃ DỪNG. Kiểm tra thư viện, checkpoint model, quyền ghi log và codec. "
              "Thử bỏ --ai hoặc thêm --headless để tách lỗi AI/hiển thị.", file=sys.stderr)
        if logger:
            try:
                logger.error("Session failed stage=%s error_type=%s; raw exception omitted to protect credentials.",
                             stage, error_type)
            except Exception:
                pass
        exit_code = 1
    finally:
        from utils.cleanup import cleanup_resources
        actions = []
        if ai_worker:
            actions.append(("AI worker", lambda: ai_worker.stop(timeout=5)))
        if reader:
            actions.append(("video reader", reader.stop))
            if metrics:
                actions.append(("final metrics", lambda: metrics.write(
                    reader.snapshot(), counter, settings.stale_seconds,
                    ai_worker.snapshot() if ai_worker else None)))
        if metrics:
            actions.append(("metrics file", metrics.close))
        if fall_writer:
            actions.append(("event journal", fall_writer.close))
        if not args.headless:
            actions.append(("preview windows", cv2.destroyAllWindows))
        if not cleanup_resources(actions):
            exit_code = 1
        if logger:
            if not cleanup_resources((("session log", lambda: logger.info("Session cleanup exit_code=%d", exit_code)),
                                      ("session log file", logger.close))):
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
