"""reinspect_crop — higher-res second look at a box + margin.

This is the agent tool hook: an uncertain first-pass score should call
this before committing to a mask. For the synthetic detector we re-run
HSV detection on an upscaled crop.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision_core.contracts import Detection, Frame
from vision_core.detect_regions import detect_regions
from vision_core.geometry import as_int_box, box_iou, expand_box


@dataclass
class ReinspectResult:
    label: str
    score: float
    xyxy: tuple[float, float, float, float]
    detections: list[Detection]
    crop_xyxy: tuple[int, int, int, int]


def _bgr(image: Frame | np.ndarray) -> np.ndarray:
    if isinstance(image, Frame):
        return image.image_bgr
    return image


def reinspect_crop(
    image: Frame | np.ndarray,
    box: tuple[float, float, float, float],
    margin: int = 20,
    frame_id: int = 0,
    *,
    scale: float = 2.0,
) -> ReinspectResult:
    """Crop ``box+margin``, optionally upscale, re-run ``detect_regions``."""
    bgr = _bgr(image)
    height, width = bgr.shape[:2]
    crop_xyxy = as_int_box(expand_box(box, margin, width, height), width, height)
    x1, y1, x2, y2 = crop_xyxy
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return ReinspectResult(label="unknown", score=0.0, xyxy=box, detections=[], crop_xyxy=crop_xyxy)

    used = crop
    scale_used = 1.0
    if scale > 1.0 and min(crop.shape[0], crop.shape[1]) < 160:
        used = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scale_used = scale

    raw = detect_regions(used, frame_id=frame_id, stage="reinspect")
    remapped: list[Detection] = []
    for det in raw:
        dx1, dy1, dx2, dy2 = det.xyxy
        remapped.append(
            Detection(
                label=det.label,
                score=det.score,
                xyxy=(
                    x1 + dx1 / scale_used,
                    y1 + dy1 / scale_used,
                    x1 + dx2 / scale_used,
                    y1 + dy2 / scale_used,
                ),
                detector_id=f"{det.detector_id}:reinspect",
                frame_id=frame_id,
            )
        )

    if not remapped:
        return ReinspectResult(label="unknown", score=0.0, xyxy=box, detections=[], crop_xyxy=crop_xyxy)

    remapped.sort(key=lambda d: (box_iou(d.xyxy, box), d.score), reverse=True)
    best = remapped[0]
    return ReinspectResult(
        label=best.label,
        score=best.score,
        xyxy=best.xyxy,
        detections=remapped,
        crop_xyxy=crop_xyxy,
    )
