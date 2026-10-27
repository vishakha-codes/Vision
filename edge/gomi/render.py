"""Paint a morphed clay figure. OpenCV for the HUD; AppKit for the island."""

from __future__ import annotations

import cv2
import numpy as np

from edge.gomi.catalog import DEFAULT_SKIN, Figure, Shape, load_catalog
from edge.gomi.morph import Morph
from edge.gomi.states import GomiState

_CATALOG = None
_HUD = None


def catalog():
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = load_catalog()
    return _CATALOG


def hud_morph() -> Morph:
    global _HUD
    if _HUD is None:
        _HUD = Morph(catalog(), DEFAULT_SKIN, "idle")
    return _HUD


def _scale(v: float, size: float, origin: float) -> float:
    return origin + v / 160.0 * size


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def _blend(canvas: np.ndarray, color: tuple[int, int, int], mask: np.ndarray, alpha: float) -> None:
    if alpha <= 0.01 or mask is None:
        return
    a = float(max(0.0, min(1.0, alpha)))
    region = mask.astype(bool)
    if not region.any():
        return
    src = canvas[region].astype(np.float32)
    paint = np.array(_bgr(color), dtype=np.float32)
    canvas[region] = np.clip(src * (1.0 - a) + paint * a, 0, 255).astype(np.uint8)


def _ellipse_mask(shape_hw: tuple[int, int], cx: int, cy: int, rx: int, ry: int) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if rx < 1 or ry < 1:
        return mask
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1, cv2.LINE_AA)
    return mask


def paint_cv(canvas: np.ndarray, fig: Figure, origin: tuple[float, float], size: float) -> None:
    if canvas.size == 0 or size < 8:
        return
    h, w = canvas.shape[:2]
    ox, oy = origin

    def pt(x: float, y: float) -> tuple[int, int]:
        return int(round(_scale(x, size, ox))), int(round(_scale(y, size, oy)))

    order = (
        "shadow",
        "puff-l",
        "puff-r",
        "ear-l",
        "ear-r",
        "ear-l-inner",
        "ear-r-inner",
        "body",
        "highlight",
        "blush",
        "left-eye",
        "right-eye",
        "left-pupil",
        "right-pupil",
        "mouth",
    )
    for key in order:
        shape = fig.shapes.get(key)
        if shape is None or shape.opacity <= 0.02:
            continue
        if shape.kind == "ellipse":
            cx, cy = pt(shape.cx, shape.cy)
            rx = max(1, int(round(shape.rx / 160.0 * size)))
            ry = max(1, int(round(shape.ry / 160.0 * size)))
            if key == "body":
                mask = _ellipse_mask((h, w), cx, cy, rx, ry)
                _blend(canvas, fig.bottom, mask, shape.opacity)
                hx, hy = pt(shape.cx - shape.rx * 0.18, shape.cy - shape.ry * 0.22)
                hmask = _ellipse_mask((h, w), hx, hy, max(1, int(rx * 0.72)), max(1, int(ry * 0.62)))
                _blend(canvas, fig.top, hmask, 0.92 * shape.opacity)
            else:
                color = fig.top if shape.clay else shape.fill
                if key == "highlight":
                    color = (255, 255, 255)
                mask = _ellipse_mask((h, w), cx, cy, rx, ry)
                _blend(canvas, color, mask, shape.opacity)
        elif shape.kind == "path" and len(shape.numbers) >= 6:
            pts = []
            for i in range(0, len(shape.numbers) - 1, 2):
                pts.append(pt(shape.numbers[i], shape.numbers[i + 1]))
            arr = np.array(pts, dtype=np.int32)
            if key == "mouth" and shape.stroke is not None:
                if len(pts) >= 3:
                    cv2.line(canvas, pts[0], pts[1], _bgr(shape.stroke), max(1, int(round(shape.stroke_w / 160.0 * size))), cv2.LINE_AA)
                    cv2.line(canvas, pts[1], pts[2], _bgr(shape.stroke), max(1, int(round(shape.stroke_w / 160.0 * size))), cv2.LINE_AA)
            elif shape.opacity > 0.02:
                color = fig.bottom if shape.clay else shape.fill
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [arr], _bgr(color), cv2.LINE_AA)
                cv2.addWeighted(overlay, shape.opacity, canvas, 1.0 - shape.opacity, 0, canvas)


def paint_face(
    canvas: np.ndarray,
    *,
    origin: tuple[int, int],
    size: int,
    state: GomiState | str,
    t: float,
    skin: str = DEFAULT_SKIN,
) -> None:
    morph = hud_morph()
    name = state.value if isinstance(state, GomiState) else str(state)
    if name not in {"idle", "alert", "working", "clear", "sleep"}:
        from edge.gomi.character import state_from_mood

        name = state_from_mood(name).value
    morph.retarget(name, t, skin=skin)
    paint_cv(canvas, morph.pose_at(t), origin, float(size))


def paint_ns(fig: Figure, origin_x: float, origin_y: float, size: float) -> None:
    """AppKit draw. Caller is inside drawRect_ (flipped view, y down)."""
    from AppKit import NSBezierPath, NSColor, NSGradient, NSMakePoint, NSMakeRect

    def px(x: float, y: float) -> tuple[float, float]:
        return _scale(x, size, origin_x), _scale(y, size, origin_y)

    def color(rgb: tuple[int, int, int], alpha: float):
        return NSColor.colorWithCalibratedRed_green_blue_alpha_(
            rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, max(0.0, min(1.0, alpha))
        )

    order = (
        "shadow",
        "ear-l",
        "ear-r",
        "puff-l",
        "puff-r",
        "body",
        "ear-l-inner",
        "ear-r-inner",
        "highlight",
        "blush",
        "left-eye",
        "right-eye",
        "left-pupil",
        "right-pupil",
        "mouth",
    )
    for key in order:
        shape: Shape | None = fig.shapes.get(key)
        if shape is None or shape.opacity <= 0.02:
            continue
        if shape.kind == "ellipse":
            cx, cy = px(shape.cx, shape.cy)
            rx = max(0.6, shape.rx / 160.0 * size)
            ry = max(0.6, shape.ry / 160.0 * size)
            rect = NSMakeRect(cx - rx, cy - ry, rx * 2.0, ry * 2.0)
            path = NSBezierPath.bezierPathWithOvalInRect_(rect)
            if key == "body":
                grad = NSGradient.alloc().initWithStartingColor_endingColor_(
                    color(fig.top, shape.opacity),
                    color(fig.bottom, shape.opacity),
                )
                grad.drawInBezierPath_relativeCenterPosition_(path, NSMakePoint(-0.15, -0.28))
            else:
                rgb = (255, 255, 255) if key == "highlight" else (fig.top if shape.clay else shape.fill)
                color(rgb, shape.opacity).setFill()
                path.fill()
        elif shape.kind == "path" and len(shape.numbers) >= 6:
            path = NSBezierPath.bezierPath()
            pts = [px(shape.numbers[i], shape.numbers[i + 1]) for i in range(0, len(shape.numbers) - 1, 2)]
            if key == "mouth" and shape.stroke is not None and len(pts) >= 3:
                path.moveToPoint_(pts[0])
                path.curveToPoint_controlPoint1_controlPoint2_(pts[2], pts[1], pts[1])
                path.setLineWidth_(max(1.1, shape.stroke_w / 160.0 * size))
                path.setLineCapStyle_(1)
                color(shape.stroke, shape.opacity).setStroke()
                path.stroke()
            elif len(pts) >= 3:
                path.moveToPoint_(pts[0])
                for p in pts[1:]:
                    path.lineToPoint_(p)
                path.closePath()
                rgb = fig.top if shape.clay else shape.fill
                color(rgb, shape.opacity).setFill()
                path.fill()
