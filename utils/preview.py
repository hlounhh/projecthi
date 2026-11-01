"""Fit a preview into a resizable window without stretching or cropping."""

import cv2
import numpy as np


def draw_status(image, labels):
    scale = min(0.6, max(0.28, image.shape[1] / 1500))
    line_height = max(19, round(scale * 40))
    image[:min(image.shape[0], line_height * len(labels) + 12)] = (20, 20, 20)
    for index, label in enumerate(labels):
        cv2.putText(image, label, (8, 22 + index * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 1, cv2.LINE_AA)


def fit_preview(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("Preview dimensions must be positive.")
    source_height, source_width = image.shape[:2]
    if (source_width, source_height) == (width, height):
        return image
    ratio = min(width / source_width, height / source_height)
    fitted_width = min(width, max(1, round(source_width * ratio)))
    fitted_height = min(height, max(1, round(source_height * ratio)))
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    fitted = cv2.resize(image, (fitted_width, fitted_height), interpolation=interpolation)
    canvas = np.zeros((height, width, 3), dtype=image.dtype)
    left = (width - fitted_width) // 2
    top = (height - fitted_height) // 2
    canvas[top:top + fitted_height, left:left + fitted_width] = fitted
    return canvas
