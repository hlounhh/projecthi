"""Validated configuration. Camera credentials never belong in source control."""

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


class ConfigError(ValueError):
    pass


def _number(name: str, default: str, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, default))
    except ValueError:
        raise ConfigError(f"{name} phải là số.") from None
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ConfigError(f"{name} phải nằm trong [{minimum}, {maximum}].")
    return value


def _integer(name: str, default: str, minimum: int, maximum: int) -> int:
    value = _number(name, default, minimum, maximum)
    if not value.is_integer():
        raise ConfigError(f"{name} phải là số nguyên.")
    return int(value)


@dataclass(frozen=True)
class Settings:
    source: str = field(repr=False)
    is_file: bool
    open_timeout_ms: int = 3000
    read_timeout_ms: int = 2000
    reconnect_seconds: float = 2.0
    stale_seconds: float = 1.0
    display_width: int = 960
    metrics_interval_seconds: float = 1.0
    log_dir: Path = ROOT / "data" / "logs"


def load_settings(
    video: str | None = None,
    env_file: Path | None = None,
    *,
    stream_subtype: int | None = None,
) -> Settings:
    # Explicit shell variables take precedence over .env values.
    load_dotenv(env_file if env_file is not None else ROOT / ".env", override=False)
    is_file = video is not None
    if is_file:
        path = Path(video).expanduser().resolve()
        if not path.is_file():
            raise ConfigError("Không tìm thấy video. Hãy kiểm tra đường dẫn --video.")
        source = str(path)
    else:
        source = os.getenv("RTSP_URL", "").strip()
        if not source:
            host = os.getenv("CAMERA_HOST", "192.168.1.10").strip()
            user = os.getenv("CAMERA_USER", "admin")
            password = os.getenv("CAMERA_PASSWORD", "")
            if not password or password == "CHANGE_ME":
                raise ConfigError("Chưa cấu hình CAMERA_PASSWORD trong .env; hoặc dùng --video.")
            if not host or any(c in host for c in '/@?#: \\'):
                raise ConfigError("CAMERA_HOST phải là IPv4 hoặc hostname, không kèm cổng.")
            port = _integer("CAMERA_PORT", "554", 1, 65535)
            channel = _integer("CAMERA_CHANNEL", "1", 1, 64)
            subtype = _integer("CAMERA_SUBTYPE", "1", 0, 2)
            source = (
                f"rtsp://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}"
                f"/cam/realmonitor?channel={channel}&subtype={subtype}"
            )
        try:
            parsed = urlsplit(source)
            if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname:
                raise ValueError
            if parsed.port is not None and not 1 <= parsed.port <= 65535:
                raise ValueError
        except ValueError:
            raise ConfigError("RTSP_URL không hợp lệ; cần URL rtsp:// hoặc rtsps://.") from None
        if stream_subtype is not None:
            if stream_subtype not in (0, 1):
                raise ConfigError("Luồng chỉ chấp nhận 0 (chính) hoặc 1 (phụ).")
            # Explicit CLI selection overrides both shell variables and RTSP_URL.
            query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                     if key != "subtype"]
            query.append(("subtype", str(stream_subtype)))
            source = urlunsplit(parsed._replace(query=urlencode(query)))

    transport = os.getenv("RTSP_TRANSPORT", "tcp").strip().lower()
    if transport not in {"tcp", "udp"}:
        raise ConfigError("RTSP_TRANSPORT chỉ chấp nhận tcp hoặc udp.")
    # Set before opening any capture. Do not log the source, even on errors.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
    return Settings(
        source=source,
        is_file=is_file,
        open_timeout_ms=_integer("OPEN_TIMEOUT_MS", "3000", 100, 60000),
        read_timeout_ms=_integer("READ_TIMEOUT_MS", "2000", 100, 60000),
        reconnect_seconds=_number("RECONNECT_SECONDS", "2", 0.1, 300),
        stale_seconds=_number("STALE_SECONDS", "1", 0.1, 60),
        display_width=_integer("DISPLAY_WIDTH", "960", 320, 3840),
        metrics_interval_seconds=_number("METRICS_INTERVAL_SECONDS", "1", 0.1, 60),
        log_dir=Path(os.getenv("LOG_DIR", str(ROOT / "data" / "logs"))).expanduser().resolve(),
    )
