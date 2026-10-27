"""Gomi pose timing, plus the HUD hook for the clay SVG skins.

The product face is the SVG catalog (pebble / fox / ghost / bear / cloud),
not the old lantern-seed doodle. Not a Coucou / Mochi sprite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from edge.gomi.states import GomiState

# RGB 0-255. Teal pebble, apricot belly, lantern-seed — Gomi's look.
_PALETTE = {
    GomiState.IDLE: {
        "body": (47, 155, 134),
        "belly": (240, 178, 122),
        "outline": (18, 48, 44),
        "eye": (24, 32, 36),
        "white": (246, 244, 236),
        "seed": (255, 214, 90),
        "seed_glow": (255, 168, 60),
        "nose": (176, 98, 64),
        "mouth": (56, 40, 36),
        "rim": (64, 196, 168),
    },
    GomiState.ALERT: {
        "body": (217, 119, 58),
        "belly": (245, 208, 138),
        "outline": (74, 32, 16),
        "eye": (28, 24, 22),
        "white": (255, 246, 232),
        "seed": (255, 92, 64),
        "seed_glow": (255, 140, 80),
        "nose": (168, 72, 40),
        "mouth": (64, 32, 24),
        "rim": (255, 140, 72),
    },
    GomiState.WORKING: {
        "body": (196, 160, 53),
        "belly": (243, 226, 160),
        "outline": (64, 48, 12),
        "eye": (32, 28, 18),
        "white": (255, 250, 232),
        "seed": (255, 196, 64),
        "seed_glow": (232, 176, 48),
        "nose": (160, 96, 40),
        "mouth": (56, 40, 20),
        "rim": (232, 196, 72),
    },
    GomiState.CLEAR: {
        "body": (62, 201, 160),
        "belly": (255, 228, 181),
        "outline": (16, 64, 52),
        "eye": (22, 36, 32),
        "white": (255, 252, 240),
        "seed": (180, 255, 140),
        "seed_glow": (120, 230, 150),
        "nose": (188, 110, 70),
        "mouth": (48, 56, 40),
        "rim": (96, 230, 180),
    },
    GomiState.SLEEP: {
        "body": (58, 85, 80),
        "belly": (138, 122, 104),
        "outline": (24, 32, 32),
        "eye": (28, 32, 32),
        "white": (188, 192, 184),
        "seed": (74, 106, 98),
        "seed_glow": (52, 76, 72),
        "nose": (96, 80, 68),
        "mouth": (40, 44, 44),
        "rim": (88, 108, 104),
    },
}

_MOOD_TO_STATE = {
    "watching": GomiState.IDLE,
    "idle": GomiState.IDLE,
    "unsure": GomiState.WORKING,
    "working": GomiState.WORKING,
    "covering": GomiState.ALERT,
    "alert": GomiState.ALERT,
    "lost": GomiState.SLEEP,
    "sleep": GomiState.SLEEP,
    "clear": GomiState.CLEAR,
}


@dataclass(frozen=True)
class RGBA:
    r: int
    g: int
    b: int
    a: int = 255

    def bgr(self) -> tuple[int, int, int]:
        return (self.b, self.g, self.r)

    def ns(self) -> tuple[float, float, float, float]:
        return (self.r / 255.0, self.g / 255.0, self.b / 255.0, self.a / 255.0)


@dataclass(frozen=True)
class EllipseOp:
    cx: float
    cy: float
    rx: float
    ry: float
    fill: RGBA
    angle: float = 0.0


@dataclass(frozen=True)
class PolyOp:
    points: tuple[tuple[float, float], ...]
    fill: RGBA
    stroke: RGBA | None = None
    stroke_w: float = 1.6


@dataclass(frozen=True)
class LineOp:
    x1: float
    y1: float
    x2: float
    y2: float
    color: RGBA
    w: float = 2.0


DrawOp = EllipseOp | PolyOp | LineOp


@dataclass(frozen=True)
class GomiPose:
    bob: float
    blink: float
    sway_deg: float
    pulse: float
    look_x: float
    look_y: float
    smile: float
    squash: float
    sx: float = 1.0
    sy: float = 1.0


def _ease_in_out(u: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, u)))


def _ease_out_cubic(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return 1.0 - (1.0 - u) ** 3


def blink_envelope(t: float) -> float:
    """Lid close ~70ms, open ~130ms, then rest. Occasional second blink."""
    period = 3.35
    phase = t % period
    close_s, open_s, gap = 0.07, 0.13, 0.22
    if phase < close_s:
        return _ease_in_out(phase / close_s)
    if phase < close_s + open_s:
        return 1.0 - _ease_out_cubic((phase - close_s) / open_s)
    # Every other cycle: a second quick blink, Coucou-style 22% feel without RNG.
    if int(t // period) % 4 == 1:
        start = close_s + open_s + gap
        if start <= phase < start + close_s:
            return _ease_in_out((phase - start) / close_s)
        if start + close_s <= phase < start + close_s + open_s:
            return 1.0 - _ease_out_cubic((phase - start - close_s) / open_s)
    return 0.0


def breathe_scale(t: float, amp: float = 0.04) -> tuple[float, float]:
    wave = math.sin(t * 1.8)
    return (1.0 - wave * amp * 0.57, 1.0 + wave * amp)


def state_from_mood(mood: str) -> GomiState:
    key = (mood or "watching").strip().lower()
    return _MOOD_TO_STATE.get(key, GomiState.IDLE)


def palette_for(state: GomiState) -> dict[str, tuple[int, int, int]]:
    return _PALETTE.get(state, _PALETTE[GomiState.IDLE])


def pose_at(state: GomiState, t: float) -> GomiPose:
    """Blink, breathe, antenna sway, seed pulse — driven only by time + state."""
    blink = blink_envelope(t)
    if state is GomiState.SLEEP:
        sx, sy = breathe_scale(t, 0.025)
        return GomiPose(
            bob=math.sin(t * 1.05) * 1.1,
            blink=1.0,
            sway_deg=math.sin(t * 0.55) * 5.0,
            pulse=0.16 + 0.08 * math.sin(t * 1.15),
            look_x=0.0,
            look_y=0.12,
            smile=-0.45,
            squash=0.07,
            sx=sx,
            sy=sy,
        )
    if state is GomiState.ALERT:
        sx, sy = breathe_scale(t * 2.4, 0.055)
        return GomiPose(
            bob=math.sin(t * 6.8) * 1.7,
            blink=0.0,
            sway_deg=math.sin(t * 8.4) * 16.0,
            pulse=0.55 + 0.45 * (0.5 + 0.5 * math.sin(t * 13.0)),
            look_x=0.12 * math.sin(t * 9.0),
            look_y=-0.08,
            smile=0.05,
            squash=-0.07,
            sx=sx,
            sy=sy,
        )
    if state is GomiState.WORKING:
        sx, sy = breathe_scale(t * 1.4, 0.03)
        return GomiPose(
            bob=math.sin(t * 3.4) * 1.3,
            blink=blink,
            sway_deg=t * 70.0,
            pulse=0.45 + 0.25 * math.sin(t * 5.0),
            look_x=0.18 * math.sin(t * 2.2),
            look_y=-0.04,
            smile=0.15,
            squash=0.0,
            sx=sx,
            sy=sy,
        )
    if state is GomiState.CLEAR:
        sx, sy = breathe_scale(t, 0.05)
        return GomiPose(
            bob=abs(math.sin(t * 3.0)) * 2.2,
            blink=blink * 0.35,
            sway_deg=math.sin(t * 2.0) * 10.0,
            pulse=0.7 + 0.25 * math.sin(t * 3.4),
            look_x=0.0,
            look_y=-0.02,
            smile=0.85,
            squash=-0.03,
            sx=sx,
            sy=sy,
        )
    sx, sy = breathe_scale(t, 0.04)
    return GomiPose(
        bob=math.sin(t * 2.15) * 2.2,
        blink=blink,
        sway_deg=math.sin(t * 1.35) * 8.0,
        pulse=0.45 + 0.35 * math.sin(t * 2.8),
        look_x=0.05 * math.sin(t * 0.7),
        look_y=0.03 * math.sin(t * 0.45),
        smile=0.35,
        squash=0.0,
        sx=sx,
        sy=sy,
    )


def _rgb(pal: dict[str, tuple[int, int, int]], key: str, alpha: int = 255) -> RGBA:
    r, g, b = pal[key]
    return RGBA(r, g, b, alpha)


def gomi_ops(
    *,
    x: float,
    y: float,
    size: float,
    state: GomiState,
    pose: GomiPose,
) -> list[DrawOp]:
    """Layout Gomi inside a square. y grows downward (OpenCV / flipped NSView)."""
    pal = palette_for(state)
    body = _rgb(pal, "body")
    belly = _rgb(pal, "belly")
    outline = _rgb(pal, "outline")
    white = _rgb(pal, "white")
    eye = _rgb(pal, "eye")
    seed = _rgb(pal, "seed")
    glow = _rgb(pal, "seed_glow", 140)
    nose = _rgb(pal, "nose")
    mouth = _rgb(pal, "mouth")

    squash = (1.0 + pose.squash) * pose.sx
    stretch = (1.0 - pose.squash * 0.6) * pose.sy
    s = max(12.0, size)
    cx = x + s * 0.50
    cy = y + s * 0.52 + pose.bob

    def pt(nx: float, ny: float) -> tuple[float, float]:
        return (cx + (nx - 0.50) * s * squash, cy + (ny - 0.50) * s * stretch)

    ops: list[DrawOp] = []

    # Soft ground shadow.
    ops.append(
        EllipseOp(
            cx,
            y + s * 0.90 + pose.bob * 0.15,
            s * 0.28,
            s * 0.07,
            RGBA(8, 10, 12, 90),
        )
    )

    # Side pebble nubs — Gomi is not a perfect circle.
    ops.append(EllipseOp(*pt(0.16, 0.50), s * 0.10, s * 0.12, body))
    ops.append(EllipseOp(*pt(0.86, 0.52), s * 0.09, s * 0.11, body))

    # Feet stubs.
    ops.append(EllipseOp(*pt(0.34, 0.88), s * 0.10, s * 0.055, body))
    ops.append(EllipseOp(*pt(0.66, 0.88), s * 0.10, s * 0.055, body))

    body_pts = tuple(
        pt(nx, ny)
        for nx, ny in (
            (0.46, 0.10),
            (0.72, 0.20),
            (0.86, 0.40),
            (0.84, 0.66),
            (0.70, 0.84),
            (0.50, 0.90),
            (0.30, 0.84),
            (0.16, 0.64),
            (0.16, 0.38),
            (0.28, 0.20),
        )
    )
    ops.append(PolyOp(body_pts, body, outline, max(1.2, s * 0.03)))
    ops.append(EllipseOp(*pt(0.50, 0.62), s * 0.20, s * 0.18, belly))

    # Lantern-seed antenna (signature). Working = spinning; others sway.
    ang = math.radians(pose.sway_deg)
    root = pt(0.40, 0.16)
    tip = (
        root[0] + math.sin(ang) * s * 0.22,
        root[1] - math.cos(ang) * s * 0.26,
    )
    ops.append(LineOp(root[0], root[1], tip[0], tip[1], outline, max(1.4, s * 0.035)))
    glow_r = s * (0.10 + 0.05 * pose.pulse)
    ops.append(EllipseOp(tip[0], tip[1], glow_r * 1.55, glow_r * 1.55, glow))
    for i in range(5):
        petal = math.radians(pose.sway_deg + i * 72.0)
        px = tip[0] + math.cos(petal) * s * 0.055
        py = tip[1] + math.sin(petal) * s * 0.055
        ops.append(EllipseOp(px, py, s * 0.034, s * 0.022, seed, pose.sway_deg + i * 72.0))
    ops.append(EllipseOp(tip[0], tip[1], s * 0.032, s * 0.032, seed))

    # Face.
    eye_h = max(0.12, 1.0 - pose.blink)
    left = pt(0.38, 0.42)
    right = pt(0.64, 0.41)
    if state is GomiState.SLEEP:
        ops.append(
            LineOp(
                left[0] - s * 0.07,
                left[1],
                left[0] + s * 0.07,
                left[1] + s * 0.02,
                eye,
                max(1.6, s * 0.045),
            )
        )
        ops.append(
            LineOp(
                right[0] - s * 0.06,
                right[1] + s * 0.02,
                right[0] + s * 0.07,
                right[1],
                eye,
                max(1.6, s * 0.045),
            )
        )
    else:
        rx_l, ry_l = s * 0.085, s * 0.10 * eye_h
        rx_r, ry_r = s * 0.075, s * 0.09 * eye_h
        if ry_l > 0.6:
            ops.append(EllipseOp(left[0], left[1], rx_l, ry_l, white))
            ops.append(
                EllipseOp(
                    left[0] + pose.look_x * s * 0.04,
                    left[1] + pose.look_y * s * 0.04,
                    s * 0.038,
                    s * 0.046 * eye_h,
                    eye,
                )
            )
            ops.append(
                EllipseOp(
                    left[0] + s * 0.018,
                    left[1] - s * 0.022 * eye_h,
                    s * 0.012,
                    s * 0.012,
                    white,
                )
            )
        if ry_r > 0.6:
            ops.append(EllipseOp(right[0], right[1], rx_r, ry_r, white))
            ops.append(
                EllipseOp(
                    right[0] + pose.look_x * s * 0.04,
                    right[1] + pose.look_y * s * 0.04,
                    s * 0.034,
                    s * 0.042 * eye_h,
                    eye,
                )
            )
            ops.append(
                EllipseOp(
                    right[0] + s * 0.016,
                    right[1] - s * 0.02 * eye_h,
                    s * 0.011,
                    s * 0.011,
                    white,
                )
            )

    if s >= 28:
        nose_pts = (pt(0.50, 0.50), pt(0.46, 0.56), pt(0.54, 0.56))
        ops.append(PolyOp(nose_pts, nose, None, 0.0))

    smile = pose.smile
    mx, my = pt(0.50, 0.60)
    if smile < 0:
        ops.append(
            LineOp(mx - s * 0.07, my + s * 0.01, mx + s * 0.07, my + s * 0.01, mouth, max(1.2, s * 0.03))
        )
    else:
        ops.append(LineOp(mx - s * 0.08, my, mx - s * 0.02, my + s * (0.02 + 0.03 * smile), mouth, max(1.3, s * 0.03)))
        ops.append(LineOp(mx - s * 0.02, my + s * (0.02 + 0.03 * smile), mx + s * 0.08, my, mouth, max(1.3, s * 0.03)))

    return ops


def paint_cv(canvas: np.ndarray, ops: list[DrawOp]) -> None:
    if canvas.size == 0:
        return
    h, w = canvas.shape[:2]
    for op in ops:
        if isinstance(op, EllipseOp):
            rx, ry = max(0.6, op.rx), max(0.6, op.ry)
            cx, cy = int(round(op.cx)), int(round(op.cy))
            if cx < -20 or cy < -20 or cx > w + 20 or cy > h + 20:
                continue
            cv2.ellipse(
                canvas,
                (cx, cy),
                (int(round(rx)), int(round(ry))),
                op.angle,
                0,
                360,
                op.fill.bgr(),
                -1,
                cv2.LINE_AA,
            )
        elif isinstance(op, PolyOp):
            pts = np.array(op.points, dtype=np.int32)
            cv2.fillPoly(canvas, [pts], op.fill.bgr(), cv2.LINE_AA)
            if op.stroke is not None and op.stroke_w > 0:
                cv2.polylines(canvas, [pts], True, op.stroke.bgr(), max(1, int(round(op.stroke_w))), cv2.LINE_AA)
        elif isinstance(op, LineOp):
            cv2.line(
                canvas,
                (int(round(op.x1)), int(round(op.y1))),
                (int(round(op.x2)), int(round(op.y2))),
                op.color.bgr(),
                max(1, int(round(op.w))),
                cv2.LINE_AA,
            )


def draw_gomi(
    canvas: np.ndarray,
    *,
    origin: tuple[int, int],
    size: int,
    state: GomiState | str,
    t: float = 0.0,
    skin: str = "pebble",
) -> None:
    """Clay SVG skin (default pebble). The lantern-seed drawing is not the product face."""
    from edge.gomi.render import paint_face

    paint_face(canvas, origin=origin, size=size, state=state, t=t, skin=skin)


def draw_companion_hud(
    canvas: np.ndarray,
    mood: str,
    line: str,
    *,
    origin: tuple[int, int] | None = None,
    t: float = 0.0,
) -> None:
    """Small Gomi card in the OpenCV HUD. Signature matches the old island slot."""
    if canvas.size == 0:
        return
    height, width = canvas.shape[:2]
    island_w, island_h = 124, 132
    x0, y0 = origin if origin is not None else (8, max(8, height - island_h - 8))
    x1, y1 = min(width, x0 + island_w), min(height, y0 + island_h)
    if x1 <= x0 or y1 <= y0:
        return
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (18, 20, 24), thickness=-1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (86, 96, 92), thickness=1)
    state = state_from_mood(mood)
    face = min(island_w - 16, island_h - 36)
    draw_gomi(canvas, origin=(x0 + 8, y0 + 4), size=face, state=state, t=t, skin="pebble")
    label = state.value
    cv2.putText(
        canvas,
        f"Gomi · {label}",
        (x0 + 8, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (220, 226, 222),
        1,
        cv2.LINE_AA,
    )
    if line:
        cv2.putText(
            canvas,
            line[:64],
            (x1 + 8, min(height - 10, y1 - 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
