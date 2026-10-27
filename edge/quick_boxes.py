"""Fast-path picture boxes for pre-frost (capture thread, a few ms).

The one quick finder (``quick_picture_boxes``). Recall-first: a wrong
pre-frost costs a few frames of over-mask until the analyzer clears it; a
miss is an exposure. Two cheap passes over the rect, unioned:

1. **Colour pass** (~1/4 scale BGR): exactly-flat page / UI colours are the
   background (``regions.estimate_background``, remembered across calls so a
   small strip still knows the page colour); everything else → open (text
   strokes vanish) → light close → connected components → keep solid,
   tonally rich blobs (``regions.picture_features`` veto for text on a panel).
2. **Pale pass** (the 1/4-res gray frame the scroll follower already has):
   page-coloured but *textured* areas (low-contrast photos, products on
   white): background must be page-coloured **and** flat; a wide open drops
   glyph edges.

Both passes obey the same caps: nothing on label 0 (wallpaper, menu bar,
Dock, system UI) when a window label map is given, components are cut at
window borders (no box spans two windows), and no box is larger than
``MASK_MAX_FRAC`` (40 %) of the frame.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import cv2
import numpy as np

from vision_core.live.regions import (
    _to_bgr,
    background_mask,
    downscale_int,
    estimate_background,
    picture_features,
)
from vision_core.live.windows import MASK_MAX_FRAC, label_borders

Box = tuple[float, float, float, float]
# (label map, (capture px per label px x, y)); 0 = system UI / wallpaper.
Labels = tuple[np.ndarray, tuple[float, float]]
QuickFn = Callable[..., list[Box]]

# Pale pass (1/4-res gray)
MIN_SIDE_SMALL = 5  # small px (≈ 20 capture px at 1/4)
MIN_FILL = 0.9  # pictures are solid; text blobs keep inter-glyph gaps
MIN_FILL_TALL = 0.75
TALL_SMALL = 24
MIN_TEXTURE = 1.0  # mean |Laplacian| on the small gray (flat = button / fill)
TEXT_MAX_H = 14  # small px
TEXT_MIN_TEXTURE = 40.0
BG_SHARE = 0.12
BG_TOL = 5
FLAT_LAP = 2  # |Laplacian| on the 1/4 gray at or below this = flat fill
_OPEN = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_OPEN_WIDE = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

_PAGE_BG: list[tuple[int, int, int]] = []  # page colours seen on recent full-ish calls


# --------------------------------------------------------------------------- shared helpers


def _labels_for(labels: Labels | None, origin: tuple[float, float], f: tuple[float, float], shape: tuple[int, int]) -> np.ndarray | None:
    """Label raster aligned to a working image whose pixel (u, v) is the
    capture block starting at ``origin + (u·fx, v·fy)``."""
    if labels is None:
        return None
    lab, (sx, sy) = labels
    sh, sw = shape[:2]
    lh, lw = lab.shape[:2]
    ys = np.clip(((origin[1] + (np.arange(sh) + 0.5) * f[1]) / sy).astype(np.int64), 0, lh - 1)
    xs = np.clip(((origin[0] + (np.arange(sw) + 0.5) * f[0]) / sx).astype(np.int64), 0, lw - 1)
    return lab[np.ix_(ys, xs)]


def _apply_labels(fg: np.ndarray, lab: np.ndarray | None) -> np.ndarray:
    """No foreground on system UI, and a 1-px cut at every window border."""
    if lab is None:
        return fg
    fg = fg.copy()
    fg[(lab == 0) | label_borders(lab)] = 0
    return fg


def _contains(a, b, slack: float = 2.0) -> bool:
    return a[0] - slack <= b[0] and a[1] - slack <= b[1] and a[2] + slack >= b[2] and a[3] + slack >= b[3]


# --------------------------------------------------------------------------- colour pass


def _colour_boxes(frame: np.ndarray, rect: tuple[int, int, int, int], full_call: bool, labels: Labels | None,
                  t0: float, max_ms: float) -> list[Box]:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = rect
    scale = min(1.0, W / 3456.0)
    min_px = max(8.0, 24.0 * scale)
    f = max(1 if W < 1000 else 2, int(round(W / 864.0)))  # ~864 px wide analysis
    small = downscale_int(frame[y1:y2, x1:x2], f)
    if small.shape[0] < 2 or small.shape[1] < 2:
        return []
    colors = estimate_background(small, min_frac=0.02, max_colors=6)
    for c in _PAGE_BG:
        if c not in colors:
            colors.append(c)
    if full_call:
        _PAGE_BG[:] = colors[:8]
    fg = cv2.bitwise_not(background_mask(small, colors, 1))
    lab = _labels_for(labels, (x1, y1), (f, f), small.shape)
    fg = _apply_labels(fg, lab)
    # Open first (text strokes vanish), then a light close fills speckle
    # holes; closing first would weld text lines to pictures.
    k = max(2, int(round(10.0 * scale / f)) + 1)
    solid = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    if f >= 4:
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        if lab is not None:
            solid[lab == 0] = 0  # the close must not bridge a window border
            solid[label_borders(lab)] = 0
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    sh, sw = solid.shape[:2]
    order = sorted(range(1, n), key=lambda i: -int(stats[i][4]))
    out: list[Box] = []
    min_w = max(2, int(math.ceil(min_px / f)))
    for i in order:
        if (time.perf_counter() - t0) * 1000.0 > max_ms:
            break
        x, y, w, h, area = (int(v) for v in stats[i])
        at_edge = y == 0 or x == 0 or y + h >= sh or x + w >= sw
        # Away from the rect edge a picture is at least ~min size; at the edge
        # (content scrolling in) a thin strip counts.
        if max(w, h) < 2 * min_w or (min(w, h) < 2 * min_w and not (at_edge and min(w, h) >= 2)):
            continue
        if area < 0.5 * w * h:
            continue  # ragged: text / icons, not a rectangle of picture
        step = max(1, int(math.ceil(max(w, h) / 64.0)))  # features on ≤ 64 px: enough for a veto
        feats = picture_features(small[y : y + h : step, x : x + w : step], fg[y : y + h : step, x : x + w : step])
        if feats["modes2"] > 0.8 and feats["flat"] > 0.6:
            continue  # bimodal and flat: text on a panel / UI paint
        if feats["std"] < 1.0:
            continue
        out.append((float(x1 + x * f), float(y1 + y * f), float(min(x2, x1 + (x + w) * f)), float(min(y2, y1 + (y + h) * f))))
    return out


# --------------------------------------------------------------------------- pale pass


def _bg_levels(crop: np.ndarray, flat: np.ndarray | None = None) -> list[int]:
    hist = cv2.calcHist([crop], [0], flat, [64], [0, 256]).ravel()
    total = float(np.count_nonzero(flat)) if flat is not None else float(crop.size)
    total = total or 1.0
    levels = []
    for b in np.argsort(hist)[::-1][:3]:
        if hist[b] / total < BG_SHARE:
            break
        lo, hi = int(b) * 4, int(b) * 4 + 4
        sel = crop[(crop >= lo) & (crop < hi)]
        levels.append(int(np.median(sel)) if sel.size else lo + 2)
    return levels


def _blobs(fg: np.ndarray, lap: np.ndarray, min_side: int, *, min_fill: float | None = None) -> list[tuple[int, int, int, int]]:
    solid = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _OPEN)
    n, _labels, stats, _c = cv2.connectedComponentsWithStats(solid, connectivity=8)
    shape = fg.shape
    out: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        bx, by = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        # A blob cut by the rect edge may be the visible sliver of a picture
        # scrolling in: allow it to be thin along that axis.
        cut_y = by == 0 or by + bh >= shape[0]
        cut_x = bx == 0 or bx + bw >= shape[1]
        if bw < (2 if cut_x else min_side) or bh < (2 if cut_y else min_side):
            continue
        if bw < min_side and bh < min_side:
            continue
        fill = float(np.count_nonzero(fg[by : by + bh, bx : bx + bw])) / (bw * bh)
        need = min_fill if min_fill is not None else (MIN_FILL_TALL if bh >= TALL_SMALL else MIN_FILL)
        if fill < need:
            continue
        texture = float(lap[by : by + bh, bx : bx + bw].mean())
        if texture < MIN_TEXTURE:
            continue
        if bh < TEXT_MAX_H and texture > TEXT_MIN_TEXTURE:
            continue  # a line of text (or its sliver at the edge): glyph edges everywhere
        out.append((bx, by, bx + bw, by + bh))
    return out


def _pale_boxes(small: np.ndarray, rect: Box, scale: tuple[float, float], labels: Labels | None) -> list[Box]:
    """Picture-like boxes (capture px) on the 1/4 gray, both background tests."""
    h, w = small.shape[:2]
    x1 = int(max(0, np.floor(rect[0] / scale[0])))
    y1 = int(max(0, np.floor(rect[1] / scale[1])))
    x2 = int(min(w, np.ceil(rect[2] / scale[0])))
    y2 = int(min(h, np.ceil(rect[3] / scale[1])))
    if x2 - x1 < MIN_SIDE_SMALL or y2 - y1 < MIN_SIDE_SMALL:
        return []
    crop = small[y1:y2, x1:x2]
    lab = _labels_for(labels, (x1 * scale[0], y1 * scale[1]), scale, crop.shape)
    lap = np.abs(cv2.Laplacian(crop, cv2.CV_16S, ksize=3))
    flat = (lap <= FLAT_LAP).astype(np.uint8) * 255
    near = np.zeros(crop.shape, np.uint8)
    for level in _bg_levels(crop, flat):
        near |= cv2.inRange(crop, max(0, level - BG_TOL), min(255, level + BG_TOL))
    # Anything not page-coloured (keeps grid gutters as separators) ...
    out = _blobs(_apply_labels(cv2.bitwise_not(near), lab), lap, MIN_SIDE_SMALL)
    # ... and page-coloured but textured (background must also be flat).
    pale = cv2.bitwise_and(cv2.bitwise_not(cv2.bitwise_and(near, flat)), near)
    pale = _apply_labels(cv2.morphologyEx(pale, cv2.MORPH_OPEN, _OPEN_WIDE), lab)
    for b in _blobs(pale, lap, MIN_SIDE_SMALL, min_fill=MIN_FILL_TALL):
        if not any(_contains(o, b) for o in out):
            out.append(b)
    return [
        ((x1 + bx1) * scale[0], (y1 + by1) * scale[1], (x1 + bx2) * scale[0], (y1 + by2) * scale[1])
        for bx1, by1, bx2, by2 in out
    ]


def _texty(frame: np.ndarray, b: Box) -> bool:
    """The colour pass's veto on a pale-pass box: bimodal and flat = a text
    block / UI paint, not a (low-contrast) picture."""
    x1, y1, x2, y2 = (int(round(v)) for v in b)
    crop = frame[max(0, y1) : y2, max(0, x1) : x2]
    if crop.size == 0:
        return True
    step = max(1, int(math.ceil(max(crop.shape[:2]) / 48.0)))
    feats = picture_features(crop[::step, ::step])
    return feats["modes2"] > 0.8 and feats["flat"] > 0.6


# --------------------------------------------------------------------------- the finder


def quick_picture_boxes(
    frame_bgr: np.ndarray,
    rect: Box | None = None,
    *,
    small: np.ndarray | None = None,
    scale: tuple[float, float] | None = None,
    labels: Labels | None = None,
    max_ms: float = 4.0,
) -> list[Box]:
    """Recall-first picture-like boxes inside ``rect`` (capture px).

    ``small`` / ``scale``: the 1/4-res gray frame and its capture-px-per-px
    scale (computed here when missing). ``labels``: window label map (0 =
    system UI). Boxes may be loose and may merge adjacent thumbnails; text
    blocks, system UI and anything over 40 % of the frame are never returned.
    The colour pass stops early at ``max_ms`` and returns what it has.
    """
    t0 = time.perf_counter()
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    frame = _to_bgr(frame_bgr)
    H, W = frame.shape[:2]
    r = (0, 0, W, H) if rect is None else (
        max(0, int(math.floor(rect[0]))),
        max(0, int(math.floor(rect[1]))),
        min(W, int(math.ceil(rect[2]))),
        min(H, int(math.ceil(rect[3]))),
    )
    min_px = max(8.0, 24.0 * min(1.0, W / 3456.0))
    if r[2] - r[0] < min_px or r[3] - r[1] < min_px:
        return []
    full_call = rect is None or (r[2] - r[0]) * (r[3] - r[1]) >= 0.5 * W * H
    boxes = _colour_boxes(frame, r, full_call, labels, t0, max_ms)
    # The pale pass always runs (≤ ~1 ms on the 1/4 gray): it is what finds
    # low-contrast pictures the colour pass calls background.
    if small is None:
        from edge.scroll_follow import small_scale, to_small_gray

        small = to_small_gray(frame)
        scale = small_scale(frame.shape, small.shape)
    elif scale is None:
        scale = (W / float(small.shape[1]), H / float(small.shape[0]))
    for b in _pale_boxes(small, (float(r[0]), float(r[1]), float(r[2]), float(r[3])), scale, labels):
        if any(_contains(o, b, slack=4.0) for o in boxes) or _texty(frame, b):
            continue
        boxes.append(b)
    cap = MASK_MAX_FRAC * float(W * H)
    return [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) <= cap]


class QuickBoxes:
    """Callable wrapper with a per-frame time budget over several rects.
    ``fn`` injects another finder (tests); the default is
    :func:`quick_picture_boxes`."""

    def __init__(self, fn: QuickFn | None = None) -> None:
        self.fn = fn
        self.kind = "injected" if fn is not None else "quick_picture_boxes"
        self.last_ms = 0.0

    def __call__(
        self,
        frame_bgr: np.ndarray,
        rects: list[Box],
        small: np.ndarray,
        scale: tuple[float, float],
        *,
        max_ms: float = 4.0,
        labels: Labels | None = None,
    ) -> list[Box]:
        t0 = time.perf_counter()
        out: list[Box] = []
        for rect in rects:
            left = max_ms - (time.perf_counter() - t0) * 1000.0
            if left <= 0.2:
                break
            if self.fn is not None:
                got = list(self.fn(frame_bgr, rect, max_ms=left))
            else:
                got = quick_picture_boxes(frame_bgr, rect, small=small, scale=scale, labels=labels, max_ms=left)
            out.extend(tuple(float(v) for v in b) for b in got)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return out
