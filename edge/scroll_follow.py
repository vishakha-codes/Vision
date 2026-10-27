"""Scroll follow: move masks with the content under them, every frame.

Pure OpenCV / NumPy, no threads, no AppKit. Runs on a 1/4-resolution gray
copy of each captured frame so it stays a few ms even on a Retina grab.

1. ``PhaseShift``: ``cv2.phaseCorrelate`` on one window's low-res crop (the
   engine runs it per app window; the whole frame when no window list is
   known). A scroll is a pure translation, so the peak gives (dx, dy)
   directly. Pixels the window does not own (a window in front, the menu
   bar) are masked out with ``valid`` so a dragged front window cannot pass
   its motion to the window behind it. A low response (navigation, video,
   nothing in common) means "no shift".
2. ``relocalize``: ``cv2.matchTemplate`` of the patch last seen under one
   mask, searched around both "did not move" and "moved with the page".
   This catches sub-pane scrolling (a sidebar that stays put while the main
   column scrolls, or an inner scroller that moves alone).

Capture excludes the overlay window, so the content under a mask is visible
in the grab and can be used as the template.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

Box = tuple[float, float, float, float]

# 1/4 resolution: two pyrDowns (Gaussian, so text does not alias).
SMALL_LEVELS = 2
PHASE_MIN_RESPONSE = 0.08  # below this the frames do not share a translation
PHASE_MIN_SHIFT = 0.35  # small px; below this treat as "did not move"
MATCH_MIN_SCORE = 0.72  # TM_CCOEFF_NORMED needed to trust a local match
TEMPLATE_MIN_STD = 4.0  # flatter patches carry no position information
TEMPLATE_MAX = 96  # small px per side (= 384 capture px on a 1/4 grab)
TEMPLATE_MIN = 6  # small px per side
TEMPLATE_INNER = 0.7  # use the inner 70 % of the box
WIDE_MIN_SCORE = 0.9
EDGE_MARGIN = 2.0  # small px kept clear of the frame border
SEARCH_MARGIN = 10  # small px around each candidate position
SEARCH_MARGIN_WIDE_Y = 110  # second try: fast / jumpy vertical scrolls (≈ 440 px)


def fill_invalid(img: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    """Float copy of ``img`` with pixels outside ``valid`` set to the mean of
    the valid ones (a flat fill carries no translation signal)."""
    out = img.astype(np.float32)
    if valid is None:
        return out
    v = valid > 0
    if v.all():
        return out
    mean = float(out[v].mean()) if v.any() else 0.0
    out[~v] = mean
    return out


def to_small_gray(frame_bgr: np.ndarray, levels: int = SMALL_LEVELS) -> np.ndarray:
    """BGR (or gray) capture → low-res gray uint8."""
    if frame_bgr.ndim == 3:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_bgr
    for _ in range(levels):
        gray = cv2.pyrDown(gray)
    return gray


def small_scale(frame_shape: tuple[int, ...], small_shape: tuple[int, ...]) -> tuple[float, float]:
    """Capture px per small px on each axis."""
    return (
        float(frame_shape[1]) / float(max(1, small_shape[1])),
        float(frame_shape[0]) / float(max(1, small_shape[0])),
    )


@dataclass
class Shift:
    dx: float = 0.0  # capture px, + = content moved right
    dy: float = 0.0  # capture px, + = content moved down
    response: float = 0.0
    ok: bool = False  # True when the frames share a confident translation

    @property
    def moved(self) -> bool:
        return self.ok and (self.dx != 0.0 or self.dy != 0.0)


class PhaseShift:
    """``cv2.phaseCorrelate`` with a cached Hanning window, cropped to DFT
    friendly sizes (558 rows → 540 is ~2× faster than the odd size)."""

    def __init__(self, min_response: float = PHASE_MIN_RESPONSE) -> None:
        self.min_response = float(min_response)
        self._win: np.ndarray | None = None
        self._win_shape: tuple[int, int] | None = None

    @staticmethod
    def _crop_shape(h: int, w: int) -> tuple[int, int]:
        def best(n: int) -> int:
            # largest DFT-friendly size ≤ n (getOptimalDFTSize rounds up)
            for m in range(n, max(8, n // 2), -1):
                if cv2.getOptimalDFTSize(m) == m:
                    return m
            return n

        return best(h), best(w)

    def __call__(
        self,
        prev_small: np.ndarray,
        cur_small: np.ndarray,
        scale: tuple[float, float],
        valid: np.ndarray | None = None,
    ) -> Shift:
        """``valid`` (uint8/bool, same shape): pixels this window owns in both
        frames; the rest are flattened before correlating."""
        if prev_small is None or cur_small is None or prev_small.shape != cur_small.shape:
            return Shift()
        h, w = cur_small.shape[:2]
        if h < 8 or w < 8:
            return Shift()
        ch, cw = self._crop_shape(h, w)
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        vv = None if valid is None else valid[y0 : y0 + ch, x0 : x0 + cw]
        if vv is not None and np.count_nonzero(vv) < 0.2 * vv.size:
            return Shift()  # the window is mostly hidden: no reliable translation
        a = fill_invalid(prev_small[y0 : y0 + ch, x0 : x0 + cw], vv)
        b = fill_invalid(cur_small[y0 : y0 + ch, x0 : x0 + cw], vv)
        if self._win is None or self._win_shape != (ch, cw):
            self._win = cv2.createHanningWindow((cw, ch), cv2.CV_32F)
            self._win_shape = (ch, cw)
        if float(a.std()) < 1.0 or float(b.std()) < 1.0:
            return Shift()
        (sx, sy), response = cv2.phaseCorrelate(a, b, self._win)
        response = float(response)
        if not np.isfinite(response) or response < self.min_response:
            return Shift(response=max(0.0, response) if np.isfinite(response) else 0.0)
        if abs(sx) > cw / 3 or abs(sy) > ch / 3:
            return Shift(response=response)
        if abs(sx) < PHASE_MIN_SHIFT:
            sx = 0.0
        if abs(sy) < PHASE_MIN_SHIFT:
            sy = 0.0
        return Shift(dx=float(sx) * scale[0], dy=float(sy) * scale[1], response=response, ok=True)


@dataclass
class Relocalized:
    dx: float  # capture px
    dy: float
    score: float
    textured: bool
    ok: bool  # confident match (or flat patch, where position is moot)


def _subpixel(res: np.ndarray, x: int, y: int) -> tuple[float, float]:
    def fit(m1: float, c: float, p1: float) -> float:
        den = m1 - 2.0 * c + p1
        if abs(den) < 1e-9:
            return 0.0
        return float(np.clip(0.5 * (m1 - p1) / den, -0.5, 0.5))

    h, w = res.shape[:2]
    ox = fit(res[y, x - 1], res[y, x], res[y, x + 1]) if 0 < x < w - 1 else 0.0
    oy = fit(res[y - 1, x], res[y, x], res[y + 1, x]) if 0 < y < h - 1 else 0.0
    return ox, oy


def _clip_int(box: Box, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        int(max(0, min(w, np.floor(x1)))),
        int(max(0, min(h, np.floor(y1)))),
        int(max(0, min(w, np.ceil(x2)))),
        int(max(0, min(h, np.ceil(y2)))),
    )


def _match(
    tmpl: np.ndarray,
    cur_small: np.ndarray,
    tx: int,
    ty: int,
    candidates: list[tuple[float, float]],
    margin_x: int,
    margin_y: int,
) -> tuple[float, float, float, bool] | None:
    """Best TM_CCOEFF_NORMED match in a window spanning every candidate.
    The last field is True when the peak sits on a window side that was
    *not* clipped by the frame: the real peak may lie outside, so the
    caller must not trust it (this is what lost sub-pane scrolls)."""
    th, tw = tmpl.shape[:2]
    h, w = cur_small.shape[:2]
    xs = [tx + c[0] for c in candidates]
    ys = [ty + c[1] for c in candidates]
    sx1 = int(max(0, np.floor(min(xs)) - margin_x))
    sy1 = int(max(0, np.floor(min(ys)) - margin_y))
    sx2 = int(min(w, np.ceil(max(xs)) + tw + margin_x))
    sy2 = int(min(h, np.ceil(max(ys)) + th + margin_y))
    if sx2 - sx1 < tw or sy2 - sy1 < th:
        return None
    window = cur_small[sy1:sy2, sx1:sx2]
    res = cv2.matchTemplate(window, tmpl, cv2.TM_CCOEFF_NORMED)
    _min_v, max_v, _min_l, (bx, by) = cv2.minMaxLoc(res)
    if not np.isfinite(max_v):
        return None
    rh, rw = res.shape[:2]
    at_edge = (
        (by == 0 and sy1 > 0)
        or (by == rh - 1 and sy2 < h)
        or (bx == 0 and sx1 > 0)
        or (bx == rw - 1 and sx2 < w)
    )
    ox, oy = _subpixel(res, bx, by)
    return (sx1 + bx + ox - tx, sy1 + by + oy - ty, float(max_v), bool(at_edge))


def relocalize(
    prev_small: np.ndarray,
    cur_small: np.ndarray,
    box: Box,
    scale: tuple[float, float],
    pred: tuple[float, float] = (0.0, 0.0),
    *,
    extra: tuple[tuple[float, float], ...] | list[tuple[float, float]] = (),
    clip: Box | None = None,
    min_score: float = MATCH_MIN_SCORE,
) -> Relocalized:
    """Where did the content under ``box`` (capture px, in ``prev``) go?

    ``pred`` is the predicted motion in capture px (the local pane shift,
    else the global shift); ``extra`` are more candidates (e.g. the mask's
    last velocity). "Stayed", ``pred`` and ``extra`` are all searched; the
    best normalized correlation wins. Returns motion in capture px.
    """
    sx, sy = scale
    h, w = prev_small.shape[:2]
    pdx, pdy = pred[0] / sx, pred[1] / sy
    bx1, by1, bx2, by2 = box[0] / sx, box[1] / sy, box[2] / sx, box[3] / sy
    if clip is not None:
        # Template from inside the moving pane only (a mask that slid under
        # static chrome must not drag chrome pixels into its template).
        cx1, cy1, cx2, cy2 = clip[0] / sx, clip[1] / sy, clip[2] / sx, clip[3] / sy
        if min(bx2, cx2) - max(bx1, cx1) >= TEMPLATE_MIN and min(by2, cy2) - max(by1, cy1) >= TEMPLATE_MIN:
            bx1, by1, bx2, by2 = max(bx1, cx1), max(by1, cy1), min(bx2, cx2), min(by2, cy2)
    # The template must be on screen in prev, and land on screen at the
    # predicted spot, otherwise the window cannot contain the true match.
    # Stay EDGE_MARGIN px inside the frame: pyrDown's border rows differ
    # between frames and pull the match by a pixel.
    m = EDGE_MARGIN
    rx1 = max(bx1, m, m - pdx)
    ry1 = max(by1, m, m - pdy)
    rx2 = min(bx2, w - m, w - m - pdx)
    ry2 = min(by2, h - m, h - m - pdy)
    if rx2 - rx1 < TEMPLATE_MIN or ry2 - ry1 < TEMPLATE_MIN:
        rx1, ry1 = max(bx1, m), max(by1, m)
        rx2, ry2 = min(bx2, w - m), min(by2, h - m)
    if rx2 - rx1 < TEMPLATE_MIN or ry2 - ry1 < TEMPLATE_MIN:
        return Relocalized(pred[0], pred[1], 0.0, False, False)
    # Central crop: bounded cost for big pictures, and the box border (which
    # may already overlap static chrome or a neighbour) stays out of it.
    cx, cy = 0.5 * (rx1 + rx2), 0.5 * (ry1 + ry2)
    inner_w = max(TEMPLATE_MIN, (rx2 - rx1) * TEMPLATE_INNER)
    inner_h = max(TEMPLATE_MIN, (ry2 - ry1) * TEMPLATE_INNER)
    half_w = min(TEMPLATE_MAX, rx2 - rx1, inner_w) / 2.0
    half_h = min(TEMPLATE_MAX, ry2 - ry1, inner_h) / 2.0
    tx1, ty1, tx2, ty2 = _clip_int((cx - half_w, cy - half_h, cx + half_w, cy + half_h), w, h)
    tmpl = prev_small[ty1:ty2, tx1:tx2]
    if tmpl.shape[0] < TEMPLATE_MIN or tmpl.shape[1] < TEMPLATE_MIN:
        return Relocalized(pred[0], pred[1], 0.0, False, False)
    if float(tmpl.std()) < TEMPLATE_MIN_STD:
        # Flat patch (blank card, solid colour): follow the prediction.
        return Relocalized(pred[0], pred[1], 0.0, False, True)
    candidates = [(0.0, 0.0)]
    for cdx, cdy in [(pdx, pdy)] + [(e[0] / sx, e[1] / sy) for e in extra]:
        if any(abs(cdx - c[0]) <= 1.0 and abs(cdy - c[1]) <= 1.0 for c in candidates):
            continue
        candidates.append((cdx, cdy))
    got = _match(tmpl, cur_small, tx1, ty1, candidates, SEARCH_MARGIN, SEARCH_MARGIN)
    if got is None or got[2] < min_score or got[3]:
        # Wide vertical search (fast / jumpy scroll). A grid of similar
        # thumbnails can match a neighbour, so demand a clearly good peak.
        wide = _match(tmpl, cur_small, tx1, ty1, candidates, SEARCH_MARGIN, SEARCH_MARGIN_WIDE_Y)
        if wide is not None and wide[2] >= WIDE_MIN_SCORE and (got is None or got[3] or wide[2] > got[2]):
            got = wide
    if got is None:
        return Relocalized(pred[0], pred[1], 0.0, True, False)
    mdx, mdy, score, at_edge = got
    if score < min_score or at_edge:
        return Relocalized(pred[0], pred[1], score, True, False)
    for cdx, cdy in candidates:
        if abs(mdx - cdx) <= 1.0 and abs(mdy - cdy) <= 1.0:
            # Agrees with a phase-correlation estimate: that is the more
            # precise sub-pixel value (no per-mask drift).
            return Relocalized(cdx * sx, cdy * sy, score, True, True)
    return Relocalized(mdx * sx, mdy * sy, score, True, True)


@dataclass
class ChangeInfo:
    fraction: float = 0.0  # changed share of the frame after scroll compensation
    box: Box | None = None  # capture px bbox of changed pixels
    large: bool = False  # navigation / app switch: mark masks stale
    boxes: list[Box] = field(default_factory=list)  # capture px, per changed blob
    revealed: Box | None = None  # capture px bbox of content scrolled in at the frame edge


CHANGE_PIXEL_DELTA = 18
CHANGE_MIN_FRACTION = 0.0005
LARGE_CHANGE_FRACTION = 0.30
COMPONENT_MIN_AREA = 12  # small px²
COMPONENT_MAX = 12
_OPEN = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
_MERGE = np.ones((5, 5), np.uint8)


def _warp(img: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    ref = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpAffine(
        np.full_like(img, 255), m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return ref, valid


def changed_mask(
    prev_small: np.ndarray, cur_small: np.ndarray, dx_small: float = 0.0, dy_small: float = 0.0
) -> tuple[np.ndarray, np.ndarray | None]:
    """Binary mask of pixels that differ after undoing a shift (small px),
    and the mask of pixels revealed by that shift (None if no shift).
    A 3×3 open drops 1-px residue that sub-pixel warps leave on text."""
    if dx_small or dy_small:
        ref, valid = _warp(prev_small, dx_small, dy_small)
    else:
        ref, valid = prev_small, None
    diff = cv2.absdiff(ref, cur_small)
    _t, mask = cv2.threshold(diff, CHANGE_PIXEL_DELTA, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN)
    revealed = None
    if valid is not None:
        revealed = cv2.bitwise_not(valid)
        mask[revealed > 0] = 255
    return mask, revealed


def mask_boxes(
    mask: np.ndarray,
    scale: tuple[float, float],
    offset: tuple[int, int] = (0, 0),
    *,
    min_area: int = COMPONENT_MIN_AREA,
    max_n: int = COMPONENT_MAX,
) -> list[Box]:
    """Connected blobs of a binary mask (merged by a 5×5 dilation) → capture px."""
    if mask.size == 0 or not cv2.countNonZero(mask):
        return []
    merged = cv2.dilate(mask, _MERGE)
    n, _labels, stats, _c = cv2.connectedComponentsWithStats(merged, connectivity=8)
    order = sorted(range(1, n), key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
    out: list[Box] = []
    ox, oy = offset
    h, w = mask.shape[:2]
    for i in order[:max_n]:
        if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        # undo the dilation's 2 px grow, but never past the mask edge
        x1, y1 = min(x + 2, x + bw - 1), min(y + 2, y + bh - 1)
        x2, y2 = max(x + bw - 2, x1 + 1), max(y + bh - 2, y1 + 1)
        if x == 0:
            x1 = 0
        if y == 0:
            y1 = 0
        if x + bw >= w:
            x2 = w
        if y + bh >= h:
            y2 = h
        out.append(((ox + x1) * scale[0], (oy + y1) * scale[1], (ox + x2) * scale[0], (oy + y2) * scale[1]))
    return out


def detect_change(
    prev_small: np.ndarray,
    cur_small: np.ndarray,
    shift: Shift,
    scale: tuple[float, float],
    *,
    large_fraction: float = LARGE_CHANGE_FRACTION,
    valid: np.ndarray | None = None,
) -> ChangeInfo:
    """``absdiff`` gate on the low-res frames, after undoing a page scroll.
    Content newly revealed by a scroll counts as changed (it needs a look).
    ``valid``: only these pixels count (a window's own pixels in both frames);
    the changed fraction is a share of them."""
    if prev_small is None or prev_small.shape != cur_small.shape:
        h, w = cur_small.shape[:2]
        full = (0.0, 0.0, w * scale[0], h * scale[1])
        return ChangeInfo(1.0, full, True, [full])
    dx = shift.dx / scale[0] if shift.moved else 0.0
    dy = shift.dy / scale[1] if shift.moved else 0.0
    mask, revealed = changed_mask(prev_small, cur_small, dx, dy)
    if dx or dy:
        # Keep only pixels that differ under BOTH hypotheses (moved with the
        # page, stayed put): static chrome next to a scrolling page drops out.
        raw, _r = changed_mask(prev_small, cur_small)
        mask = cv2.bitwise_and(mask, raw)
        if revealed is not None:
            # The scrolled-in strip is thin (a few px at 1/4): no open here.
            _t, raw_thin = cv2.threshold(cv2.absdiff(prev_small, cur_small), CHANGE_PIXEL_DELTA // 2, 255, cv2.THRESH_BINARY)
            strip = cv2.bitwise_and(revealed, cv2.dilate(raw_thin, _MERGE))
            mask = cv2.bitwise_or(mask, strip)
    denom = float(mask.size)
    if valid is not None:
        v = (valid > 0).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, v)
        denom = float(max(1, cv2.countNonZero(v)))
    n = int(cv2.countNonZero(mask))
    frac = n / denom
    if frac < CHANGE_MIN_FRACTION:
        return ChangeInfo(frac, None, False)
    x, y, bw, bh = cv2.boundingRect(mask)
    box = (x * scale[0], y * scale[1], (x + bw) * scale[0], (y + bh) * scale[1])
    rev_box = None
    if revealed is not None and cv2.countNonZero(revealed):
        rx, ry, rw, rh = cv2.boundingRect(revealed)
        rev_box = (rx * scale[0], ry * scale[1], (rx + rw) * scale[0], (ry + rh) * scale[1])
    # After undoing a page scroll only the revealed strip remains, so a
    # big remainder means the content itself was replaced (navigation).
    large = frac >= large_fraction
    boxes = [box] if large else mask_boxes(mask, scale)
    return ChangeInfo(frac, box, large, boxes, rev_box)


LOCAL_MIN_SIDE = 40  # small px: smaller blobs are new content, not a pane


def local_shift(
    prev_small: np.ndarray,
    cur_small: np.ndarray,
    box: Box,
    scale: tuple[float, float],
    *,
    min_response: float = 0.2,
) -> Shift:
    """Phase correlation inside one changed blob (an inner scroller / pane
    that moved on its own while the rest of the screen stayed put)."""
    h, w = cur_small.shape[:2]
    x1, y1, x2, y2 = _clip_int((box[0] / scale[0], box[1] / scale[1], box[2] / scale[0], box[3] / scale[1]), w, h)
    if x2 - x1 < LOCAL_MIN_SIDE or y2 - y1 < LOCAL_MIN_SIDE:
        return Shift()
    ch, cw = PhaseShift._crop_shape(y2 - y1, x2 - x1)
    a = prev_small[y1 : y1 + ch, x1 : x1 + cw].astype(np.float32)
    b = cur_small[y1 : y1 + ch, x1 : x1 + cw].astype(np.float32)
    if float(a.std()) < 2.0 or float(b.std()) < 2.0:
        return Shift()
    win = cv2.createHanningWindow((cw, ch), cv2.CV_32F)
    (sx, sy), response = cv2.phaseCorrelate(a, b, win)
    response = float(response)
    if not np.isfinite(response) or response < min_response:
        return Shift(response=max(0.0, response) if np.isfinite(response) else 0.0)
    if abs(sx) > cw / 2.5 or abs(sy) > ch / 2.5:
        return Shift(response=response)
    if abs(sx) < PHASE_MIN_SHIFT:
        sx = 0.0
    if abs(sy) < PHASE_MIN_SHIFT:
        sy = 0.0
    return Shift(dx=float(sx) * scale[0], dy=float(sy) * scale[1], response=response, ok=True)


def residual_boxes(
    prev_small: np.ndarray,
    cur_small: np.ndarray,
    box: Box,
    shift: Shift,
    scale: tuple[float, float],
) -> list[Box]:
    """Inside ``box``, what is still different after undoing ``shift``: the
    strip the pane scrolled in, plus anything that really changed."""
    h, w = cur_small.shape[:2]
    x1, y1, x2, y2 = _clip_int((box[0] / scale[0], box[1] / scale[1], box[2] / scale[0], box[3] / scale[1]), w, h)
    if x2 <= x1 or y2 <= y1:
        return []
    dx, dy = shift.dx / scale[0], shift.dy / scale[1]
    pad = int(np.ceil(max(abs(dx), abs(dy)))) + 2
    px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
    px2, py2 = min(w, x2 + pad), min(h, y2 + pad)
    mask, _rev = changed_mask(prev_small[py1:py2, px1:px2], cur_small[py1:py2, px1:px2], dx, dy)
    # Pixels that came from outside the blob in prev are new to the blob.
    inner = mask[y1 - py1 : y2 - py1, x1 - px1 : x2 - px1].copy()
    m = np.zeros_like(inner)
    sx1 = int(np.clip(np.floor(dx), 0, x2 - x1))
    sx2 = int(np.clip(np.ceil(x2 - x1 + dx), 0, x2 - x1))
    sy1 = int(np.clip(np.floor(dy), 0, y2 - y1))
    sy2 = int(np.clip(np.ceil(y2 - y1 + dy), 0, y2 - y1))
    m[sy1:sy2, sx1:sx2] = 255
    inner[m == 0] = 255
    return mask_boxes(inner, scale, (x1, y1), min_area=4)
