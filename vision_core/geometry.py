"""Box geometry used by tracking, policy, and eval."""

from __future__ import annotations


Box = tuple[float, float, float, float]


def box_area(xyxy: Box) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def clip_box(xyxy: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = xyxy
    x1 = min(max(0.0, x1), float(max(0, width - 1)))
    y1 = min(max(0.0, y1), float(max(0, height - 1)))
    x2 = min(max(x1 + 1.0, x2), float(width))
    y2 = min(max(y1 + 1.0, y2), float(height))
    return (x1, y1, x2, y2)


def expand_box(xyxy: Box, margin: float, width: int, height: int) -> Box:
    x1, y1, x2, y2 = xyxy
    return clip_box((x1 - margin, y1 - margin, x2 + margin, y2 + margin), width, height)


def as_int_box(xyxy: Box, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = clip_box(xyxy, width, height)
    ix1 = int(round(x1))
    iy1 = int(round(y1))
    ix2 = int(round(x2))
    iy2 = int(round(y2))
    ix1 = min(max(0, ix1), max(0, width - 1))
    iy1 = min(max(0, iy1), max(0, height - 1))
    ix2 = min(max(ix1 + 1, ix2), width)
    iy2 = min(max(iy1 + 1, iy2), height)
    return ix1, iy1, ix2, iy2


def shift_box(xyxy: Box, dx: float, dy: float, width: int, height: int) -> Box:
    x1, y1, x2, y2 = xyxy
    return clip_box((x1 + dx, y1 + dy, x2 + dx, y2 + dy), width, height)


def box_center(xyxy: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def blend_boxes(pred: Box, meas: Box, alpha: float) -> Box:
    """EMA toward the measurement: alpha=1 keeps only meas."""
    a = max(0.0, min(1.0, alpha))
    return tuple(a * m + (1.0 - a) * p for p, m in zip(pred, meas))  # type: ignore[return-value]


def hysteresis_mask_on(
    currently_on: bool,
    score: float,
    on_threshold: float,
    off_threshold: float,
) -> bool:
    """Mask turns on at a higher score than it turns off."""
    if off_threshold > on_threshold:
        off_threshold = on_threshold
    if currently_on:
        return score >= off_threshold
    return score >= on_threshold
