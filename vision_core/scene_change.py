"""scene_change — gray absdiff, threshold, dilate, contours, change score."""

from __future__ import annotations

import cv2
import numpy as np

from vision_core.contracts import Frame, SceneChangeResult

_DEFAULT_THRESH = 16
_KERNEL = None


def _bgr(image: Frame | np.ndarray) -> np.ndarray:
    if isinstance(image, Frame):
        return image.image_bgr
    return image


def _kernel() -> np.ndarray:
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return _KERNEL


def scene_change(
    prev: Frame | np.ndarray,
    cur: Frame | np.ndarray,
    *,
    threshold: int = _DEFAULT_THRESH,
) -> SceneChangeResult:
    """Return a binary change map, contour boxes, and a [0, 1] change_score."""
    prev_bgr = _bgr(prev)
    cur_bgr = _bgr(cur)
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, cur_gray)
    _ok, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.dilate(binary, _kernel(), iterations=2)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
    change_score = float(np.count_nonzero(binary)) / float(binary.size)
    return SceneChangeResult(
        change_score=change_score,
        binary=binary,
        contours=list(contours),
        boxes=boxes,
    )
