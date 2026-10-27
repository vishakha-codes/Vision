"""extract_tiles — ROI crops + boxes from a change map / connected components."""

from __future__ import annotations

import cv2
import numpy as np

from vision_core.contracts import Frame, SceneChangeResult, Tile
from vision_core.geometry import as_int_box, expand_box


def _bgr(image: Frame | np.ndarray) -> np.ndarray:
    if isinstance(image, Frame):
        return image.image_bgr
    return image


def _binary(change: SceneChangeResult | np.ndarray) -> np.ndarray:
    if isinstance(change, SceneChangeResult):
        return change.binary
    return change


def extract_tiles(
    frame: Frame | np.ndarray,
    change: SceneChangeResult | np.ndarray,
    margin: int = 12,
    *,
    min_area: int = 80,
) -> list[Tile]:
    """Crop changed components. Tiny compression / sensor noise is skipped."""
    image = _bgr(frame)
    height, width = image.shape[:2]
    binary = _binary(change)
    if binary is None or binary.size == 0:
        return []
    if binary.shape[:2] != (height, width):
        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)

    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    tiles: list[Tile] = []
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        xyxy = expand_box((x, y, x + w, y + h), margin, width, height)
        x1, y1, x2, y2 = as_int_box(xyxy, width, height)
        crop = image[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        component = labels[y1:y2, x1:x2] == label
        tile_score = float(np.count_nonzero(component)) / float(component.size) if component.size else 0.0
        tiles.append(
            Tile(
                image_bgr=crop,
                xyxy=(x1, y1, x2, y2),
                area=area,
                change_score=tile_score,
            )
        )
    tiles.sort(key=lambda t: t.area, reverse=True)
    return tiles
