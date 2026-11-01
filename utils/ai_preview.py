"""Render posture boxes only, without promoting them to fall events."""

import cv2


def draw_detections(image, result, *, boxes=True, tracks=None):
    source_height, source_width = result.source_shape[:2]
    scale_x = image.shape[1] / source_width
    scale_y = image.shape[0] / source_height
    accepted_boxes = None if tracks is None else {t.xyxy for t in tracks if t.visible}
    for detection in result.detections:
        if accepted_boxes is not None and detection.xyxy not in accepted_boxes:
            continue
        x1, y1, x2, y2 = detection.xyxy
        start = (round(x1 * scale_x), round(y1 * scale_y))
        end = (round(x2 * scale_x), round(y2 * scale_y))
        # Same neutral box for every posture: not a confirmed emergency indicator.
        color = (220, 220, 220)
        if boxes:
            cv2.rectangle(image, start, end, color, 2)
        if hasattr(detection, "keypoints"):
            label = f"Nguoi {detection.confidence:.2f}"
            points = detection.keypoints
            for first, second in ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                                  (5, 11), (6, 12), (11, 12), (11, 13),
                                  (13, 15), (12, 14), (14, 16)):
                if len(points) == 17 and min(points[first][2], points[second][2]) >= 0.5:
                    a = (round(points[first][0] * scale_x), round(points[first][1] * scale_y))
                    b = (round(points[second][0] * scale_x), round(points[second][1] * scale_y))
                    cv2.line(image, a, b, color, 2)
        else:
            label = f"{detection.label} {detection.confidence:.2f} (tu the tu anh)"
        if boxes:
            cv2.putText(image, label, (start[0], max(16, start[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
