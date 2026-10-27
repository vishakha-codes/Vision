"""detect_regions — synthetic HSV palette detector for the safe fixture.

This is a stand-in until a licensed ONNX content model is available.
It finds the three colored fixture blobs and labels the magenta blob as
``flagged`` (alias: ``explicit_region``). Live NSFW / content recognition
is **not** claimed. Do not download NSFW models here.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np

from vision_core.contracts import Detection, Frame
from vision_core.geometry import as_int_box, box_area

# Fixture palette (BGR after mp4v): red r1, magenta r2 flagged, cyan r3.
# Ranges are wide enough for codec smear, tight enough to stay separable.
_RED_LO_A = np.array([0, 110, 70], dtype=np.uint8)
_RED_HI_A = np.array([12, 255, 255], dtype=np.uint8)
_RED_LO_B = np.array([170, 110, 70], dtype=np.uint8)
_RED_HI_B = np.array([179, 255, 255], dtype=np.uint8)
_MAGENTA_LO = np.array([138, 90, 60], dtype=np.uint8)
_MAGENTA_HI = np.array([168, 255, 255], dtype=np.uint8)
_CYAN_LO = np.array([78, 90, 60], dtype=np.uint8)
_CYAN_HI = np.array([110, 255, 255], dtype=np.uint8)

_MIN_AREA = 180
_KERNEL = None
DETECTOR_ID = "synthetic_hsv_v1"


def _bgr(image: Frame | np.ndarray) -> np.ndarray:
    if isinstance(image, Frame):
        return image.image_bgr
    return image


def _kernel() -> np.ndarray:
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return _KERNEL


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(), iterations=2)
    return mask


def _score(quality: float, stage: str) -> float:
    quality = max(0.0, min(1.0, quality))
    if stage == "reinspect":
        return float(max(0.0, min(1.0, 0.74 + 0.22 * quality)))
    return float(max(0.0, min(1.0, 0.44 + 0.24 * quality)))


def _blobs_from_mask(
    mask: np.ndarray,
    hsv: np.ndarray,
    label: str,
    frame_id: int,
    stage: str,
    offset: tuple[int, int] = (0, 0),
) -> list[Detection]:
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []
    ox, oy = offset
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < _MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < _MIN_AREA:
            continue
        roi = mask[y : y + h, x : x + w]
        fill = float(np.count_nonzero(roi)) / float(roi.size) if roi.size else 0.0
        sat = hsv[y : y + h, x : x + w, 1]
        sat_mean = float(np.mean(sat[roi > 0])) / 255.0 if np.any(roi > 0) else 0.0
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull)) or 1.0
        solidity = max(0.0, min(1.0, area / hull_area))
        quality = 0.45 * sat_mean + 0.35 * fill + 0.20 * solidity
        xyxy = (float(x + ox), float(y + oy), float(x + w + ox), float(y + h + oy))
        detections.append(
            Detection(
                label=label,
                score=_score(quality, stage),
                xyxy=xyxy,
                detector_id=DETECTOR_ID,
                frame_id=frame_id,
            )
        )
    detections.sort(key=lambda d: (d.score, box_area(d.xyxy)), reverse=True)
    return detections


def _gt_overlay_detections(
    image: np.ndarray,
    gt_boxes: Sequence[dict],
    frame_id: int,
    stage: str,
) -> list[Detection]:
    height, width = image.shape[:2]
    detections: list[Detection] = []
    for item in gt_boxes:
        raw = item.get("xyxy") or item.get("box")
        if not raw or len(raw) != 4:
            continue
        xyxy = as_int_box(tuple(float(v) for v in raw), width, height)
        label = str(item.get("class") or item.get("label") or "safe")
        if label == "explicit_region":
            label = "flagged"
        detections.append(
            Detection(
                label=label,
                score=_score(0.95, stage),
                xyxy=tuple(float(v) for v in xyxy),
                detector_id="gt_overlay",
                frame_id=frame_id,
            )
        )
    return detections


def detect_regions(
    image: Frame | np.ndarray,
    gt_boxes: Iterable[dict] | None = None,
    frame_id: int = 0,
    *,
    stage: str = "detect",
    origin: tuple[int, int] = (0, 0),
) -> list[Detection]:
    """Find fixture palette blobs. Optional ``gt_boxes`` overlays GT for eval debug.

    Public detector_id is ``synthetic_hsv_v1``. Flagged magenta is labeled
    ``flagged`` (policy also accepts ``explicit_region``).
    """
    bgr = _bgr(image)
    if bgr is None or bgr.size == 0:
        return []
    if gt_boxes is not None:
        return _gt_overlay_detections(bgr, list(gt_boxes), frame_id, stage)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, _RED_LO_A, _RED_HI_A),
        cv2.inRange(hsv, _RED_LO_B, _RED_HI_B),
    )
    magenta = cv2.inRange(hsv, _MAGENTA_LO, _MAGENTA_HI)
    cyan = cv2.inRange(hsv, _CYAN_LO, _CYAN_HI)
    detections: list[Detection] = []
    detections.extend(_blobs_from_mask(_clean_mask(magenta), hsv, "flagged", frame_id, stage, origin))
    detections.extend(_blobs_from_mask(_clean_mask(red), hsv, "safe", frame_id, stage, origin))
    detections.extend(_blobs_from_mask(_clean_mask(cyan), hsv, "safe", frame_id, stage, origin))
    return detections
