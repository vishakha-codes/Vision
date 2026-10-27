"""Capture-pixel ↔ Cocoa-point mapping for the live overlay.

macOS Retina: mss / Quartz usually grab *backing pixels* (2×), while
NSWindow frames are in *logical points*. If this scale is wrong, mask
windows miss the content. Y also flips: capture is top-left, Cocoa is
bottom-left.

Window-server bounds are global top-left points (``points_xyxy_to_capture``)
and ``NSScreen.visibleFrame`` is a bottom-left Cocoa rect
(``cocoa_rect_to_capture``); both map to capture pixels of the grab.

This module is pure math so tests do not need a display.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DisplayGeom:
    """Primary-display geometry used to place overlay windows.

    ``*_pt`` values are logical points. ``*_px`` are backing pixels from
    the last grab. ``scale_*`` is px/pt (typically 2.0 on Retina).
    """

    left_pt: float
    top_pt: float
    width_pt: float
    height_pt: float
    width_px: int
    height_px: int
    scale_x: float
    scale_y: float
    cocoa_origin_x: float
    cocoa_origin_y: float
    cocoa_height_pt: float

    @property
    def scale(self) -> float:
        return 0.5 * (self.scale_x + self.scale_y)


def scale_from_sizes(px_w: int, px_h: int, pt_w: float, pt_h: float) -> tuple[float, float]:
    sx = float(px_w) / float(pt_w) if pt_w else 1.0
    sy = float(px_h) / float(pt_h) if pt_h else 1.0
    return sx, sy


def capture_xyxy_to_cocoa(
    xyxy: tuple[float, float, float, float],
    geom: DisplayGeom,
) -> tuple[float, float, float, float]:
    """Map a capture-pixel box to a Cocoa frame ``(x, y, w, h)`` in points.

    ``xyxy`` is top-left origin in the grabbed image. The returned ``y`` is
    the bottom of the window in the Cocoa global coordinate system.
    """
    x1, y1, x2, y2 = xyxy
    scale_x = geom.scale_x if geom.scale_x else 1.0
    scale_y = geom.scale_y if geom.scale_y else 1.0
    left_pt = geom.left_pt + (x1 / scale_x)
    top_pt = geom.top_pt + (y1 / scale_y)
    width_pt = max(1.0, (x2 - x1) / scale_x)
    height_pt = max(1.0, (y2 - y1) / scale_y)
    cocoa_x = geom.cocoa_origin_x + (left_pt - geom.left_pt)
    cocoa_y = (
        geom.cocoa_origin_y
        + geom.cocoa_height_pt
        - (top_pt - geom.top_pt)
        - height_pt
    )
    return float(cocoa_x), float(cocoa_y), float(width_pt), float(height_pt)


def points_xyxy_to_capture(
    bounds: tuple[float, float, float, float],
    geom: DisplayGeom,
) -> tuple[float, float, float, float]:
    """Global top-left **points** (``CGWindowListCopyWindowInfo`` bounds,
    given as x1, y1, x2, y2) → capture pixels of the grabbed display
    (Retina: × ``scale``). May fall outside the frame (other displays)."""
    x1, y1, x2, y2 = bounds
    sx = geom.scale_x if geom.scale_x else 1.0
    sy = geom.scale_y if geom.scale_y else 1.0
    return (
        float((x1 - geom.left_pt) * sx),
        float((y1 - geom.top_pt) * sy),
        float((x2 - geom.left_pt) * sx),
        float((y2 - geom.top_pt) * sy),
    )


def cocoa_rect_to_capture(
    rect: tuple[float, float, float, float],
    geom: DisplayGeom,
) -> tuple[float, float, float, float]:
    """A Cocoa ``(x, y, w, h)`` rect in points (bottom-left origin, e.g.
    ``NSScreen.visibleFrame``) → capture-pixel ``xyxy`` (top-left origin).
    Inverse of :func:`capture_xyxy_to_cocoa`."""
    x, y, w, h = rect
    sx = geom.scale_x if geom.scale_x else 1.0
    sy = geom.scale_y if geom.scale_y else 1.0
    top_pt = geom.cocoa_origin_y + geom.cocoa_height_pt - (y + h)  # from the display's top, in points
    left_pt = x - geom.cocoa_origin_x
    return (float(left_pt * sx), float(top_pt * sy), float((left_pt + w) * sx), float((top_pt + h) * sy))
