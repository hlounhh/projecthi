"""Track IDs and observation status, not medical or guaranteed safety labels."""

import cv2

from ai.fall_engine import CONFIRMED_FALL, UNOBSERVED


def draw_tracks(image, tracks, source_shape):
    height, width = source_shape[:2]
    sx, sy = image.shape[1] / width, image.shape[0] / height
    for track in tracks:
        x1, y1, x2, y2 = track.xyxy
        fall_detected = getattr(track, "fall_detected", track.state == CONFIRMED_FALL)
        visible = getattr(track, "visible", track.state != UNOBSERVED)
        color = (0, 0, 255) if fall_detected else (240, 240, 240) if visible else (0, 190, 255)
        start, end = (round(x1 * sx), round(y1 * sy)), (round(x2 * sx), round(y2 * sy))
        if visible or fall_detected:
            cv2.rectangle(image, start, end, color, 3 if fall_detected else 2)
        else:
            for a, b in ((start, (end[0], start[1])), ((end[0], start[1]), end),
                         (end, (start[0], end[1])), ((start[0], end[1]), start)):
                length = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
                for offset in range(0, length, 14):
                    p = tuple(round(a[i] + (b[i] - a[i]) * offset / max(1, length)) for i in (0, 1))
                    q = tuple(round(a[i] + (b[i] - a[i]) * min(offset + 7, length) / max(1, length)) for i in (0, 1))
                    cv2.line(image, p, q, color, 1)
        quality = getattr(track, "observation_quality", track.state != UNOBSERVED)
        status = "PHAT HIEN NGA" if fall_detected else "KHONG PHAT HIEN NGA" if visible and quality else "QUAN SAT KHONG RO"
        if not visible:
            status += " | HOP VI TRI CUOI"
        elif not quality:
            status += " | QUAN SAT HAN CHE"
        label = f"ID {track.track_id}: {status}"
        cv2.putText(image, label, (max(0, start[0]), min(image.shape[0] - 8, end[1] + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
