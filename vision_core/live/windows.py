"""Window-aware geometry for the live path (pure NumPy / OpenCV, no AppKit).

The window server tells us, for every frame, which app window owns which
pixels (``ScreenLayout``: bounds, layer, front-to-back order) and where the
menu bar and Dock are (``visible_xyxy`` = ``NSScreen.visibleFrame``). This
module turns that into:

- ``WindowGeometry``: each app window's **visible region** (its bounds ∩ the
  visible frame, minus every window in front of it) as a few disjoint rects;
- ``label_map``: a raster where 0 = wallpaper / desktop / system UI (menu
  bar, Dock, Notification Centre, popovers, our own windows) and 1..N = app
  windows, so the proposer and the quick finder only look inside windows and
  never across a window border;
- ``clip_specs``: every mask cut to its own window's *current* visible
  region (≤ a few rects), moved with its window if the window was dragged
  since the analysed frame, and never over the menu bar or the Dock;
- ``cap_coverage``: when masks would cover most of a window, drop the
  speculative ones (pre-frost, unsure) and raise a "busy" warning instead
  of blurring the whole screen;
- ``HeavyTracker``: which windows have had several UNSAFE masks recently
  (adaptive pre-frost).

Boxes are ``(x1, y1, x2, y2)`` in capture pixels.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable, Sequence

import numpy as np

from vision_core.live.contracts import Box, MaskSpec, ScreenLayout, VerdictState, WindowInfo

SYSTEM_LABEL = 0
MIN_WINDOW_PX = 8.0  # smaller windows (tracking areas, 1-px helpers) are ignored
MIN_ALPHA = 0.05  # fully transparent windows neither own nor hide pixels
OCCLUDER_MIN_ALPHA = 0.3  # a system-UI window this see-through does not hide what is behind it
OCCLUDER_MAX_FRAC = 0.6  # a system-UI window covering most of the screen is a transparent overlay
MAX_PIECES = 8  # rects per clipped mask (the rest are slivers)
MASK_MAX_FRAC = 0.40  # no single proposal / quick box larger than this share of the frame
BUSY_COVER_FRAC = 0.60  # masks above this share of a window (or the screen) → "busy" warning
BUSY_SHED_FRAC = 0.85  # ...and above this, speculative masks are dropped (no whole-window blur)
BUSY_WARNING = {"code": "busy", "text": "Lots to cover here"}


# --------------------------------------------------------------------------- rect algebra


def inter(a: Box, b: Box) -> Box | None:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def area(b: Box | None) -> float:
    if b is None:
        return 0.0
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_minus(a: Box, b: Box, min_px: float = 0.0) -> list[Box]:
    """``a`` minus ``b`` as up to four disjoint slabs (top, bottom, left, right).
    Slabs thinner than ``min_px`` are dropped (0 = keep every positive slab)."""
    if inter(a, b) is None:
        return [a]
    out: list[Box] = []
    if b[1] - a[1] > min_px:
        out.append((a[0], a[1], a[2], b[1]))
    if a[3] - b[3] > min_px:
        out.append((a[0], b[3], a[2], a[3]))
    y1, y2 = max(a[1], b[1]), min(a[3], b[3])
    if b[0] - a[0] > min_px:
        out.append((a[0], y1, b[0], y2))
    if a[2] - b[2] > min_px:
        out.append((b[2], y1, a[2], y2))
    return out


def rects_minus(rects: Iterable[Box], b: Box) -> list[Box]:
    out: list[Box] = []
    for r in rects:
        out.extend(box_minus(r, b))
    return out


def rects_inter(rects: Iterable[Box], b: Box) -> list[Box]:
    return [c for c in (inter(r, b) for r in rects) if c is not None]


def rects_area(rects: Iterable[Box]) -> float:
    """Area of disjoint rects."""
    return float(sum(area(r) for r in rects))


def bbox(rects: Sequence[Box]) -> Box | None:
    if not rects:
        return None
    return (min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects))


def edges_touched(box: Box, W: float, H: float, tol: float) -> int:
    return int(box[0] <= tol) + int(box[1] <= tol) + int(box[2] >= W - tol) + int(box[3] >= H - tol)


# --------------------------------------------------------------------------- layout geometry


class WindowGeometry:
    """Visible region of every app window for one ``ScreenLayout``."""

    def __init__(self, layout: ScreenLayout) -> None:
        W, H = layout.frame_wh
        self.layout = layout
        self.frame: Box = (0.0, 0.0, float(W), float(H))
        vis = inter(tuple(float(v) for v in layout.visible_xyxy), self.frame) if layout.visible_xyxy else None
        self.content: Box = vis or self.frame  # the screen minus the menu bar and the Dock
        self.system_rects: list[Box] = box_minus(self.frame, self.content)
        self.apps: list[WindowInfo] = []  # front to back
        self.visible: dict[int, list[Box]] = {}
        self.bounds: dict[int, Box] = {}
        fronts: list[Box] = []
        frame_area = float(W * H) or 1.0
        for w in layout.windows:
            if w.alpha < MIN_ALPHA:
                continue
            b = inter(tuple(float(v) for v in w.xyxy), self.frame)
            if b is None or b[2] - b[0] < MIN_WINDOW_PX or b[3] - b[1] < MIN_WINDOW_PX:
                continue
            if w.is_app:
                if w.window_id in self.visible:
                    continue
                region = rects_inter([b], self.content)
                for f in fronts:
                    region = rects_minus(region, f)
                self.apps.append(w)
                self.visible[w.window_id] = region
                self.bounds[w.window_id] = b
                fronts.append(b)
            elif w.alpha >= OCCLUDER_MIN_ALPHA and area(b) < OCCLUDER_MAX_FRAC * frame_area:
                fronts.append(b)  # menu, popover, Dock, notification: hides what is behind it
        self.labels: dict[int, int] = {w.window_id: i + 1 for i, w in enumerate(self.apps)}
        self.ids: list[int | None] = [None] + [w.window_id for w in self.apps]

    # -- queries -------------------------------------------------------------
    def window_id_of(self, label: int) -> int | None:
        return self.ids[label] if 0 < label < len(self.ids) else None

    def visible_area(self, window_id: int | None) -> float:
        if window_id is None:
            return area(self.content)
        return rects_area(self.visible.get(window_id, ()))

    def visible_bbox(self, window_id: int) -> Box | None:
        return bbox(self.visible.get(window_id, []))

    def owner_of(self, box: Box) -> int | None:
        """App window holding most of ``box`` (None = mostly system UI / wallpaper)."""
        best, best_a = None, 0.0
        for wid, rects in self.visible.items():
            a = rects_area(rects_inter(rects, box))
            if a > best_a:
                best, best_a = wid, a
        if best is None or best_a < 0.5 * area(box):
            return None
        return best

    def inside_window(self, box: Box, window_id: int | None, tol: float = 2.0) -> bool:
        """``box`` lies within the window's bounds (it does not cross a window border)."""
        if window_id is None:
            return False
        b = self.bounds.get(window_id)
        if b is None:
            return False
        return box[0] >= b[0] - tol and box[1] >= b[1] - tol and box[2] <= b[2] + tol and box[3] <= b[3] + tol

    def in_system_ui(self, box: Box) -> float:
        """Share of ``box`` on the menu bar / Dock strips."""
        a = area(box)
        if a <= 0:
            return 0.0
        return rects_area(rects_inter(self.system_rects, box)) / a


_GEOM_CACHE: dict[int, WindowGeometry] = {}


def geometry(layout: ScreenLayout) -> WindowGeometry:
    """``WindowGeometry`` for a layout (cached on the layout object's id)."""
    key = id(layout)
    g = _GEOM_CACHE.get(key)
    if g is None or g.layout is not layout:
        if len(_GEOM_CACHE) > 16:
            _GEOM_CACHE.clear()
        g = WindowGeometry(layout)
        _GEOM_CACHE[key] = g
    return g


def label_map(layout: ScreenLayout, shape: tuple[int, int], scale: tuple[float, float]) -> np.ndarray:
    """int32 raster of ``shape`` (rows, cols); map px (u, v) ↔ capture px
    (u·sx, v·sy). 0 = system UI / wallpaper, k = ``geometry(layout).apps[k-1]``.
    Window edges are rounded *inward*, so a map pixel is labelled only when
    it lies wholly inside that window's visible region."""
    g = geometry(layout)
    h, w = shape[:2]
    sx, sy = float(scale[0]) or 1.0, float(scale[1]) or 1.0
    lab = np.zeros((h, w), np.int32)
    for win in g.apps:
        k = g.labels[win.window_id]
        for r in g.visible[win.window_id]:
            x1, y1 = int(math.ceil(r[0] / sx - 1e-6)), int(math.ceil(r[1] / sy - 1e-6))
            x2, y2 = int(math.floor(r[2] / sx + 1e-6)), int(math.floor(r[3] / sy + 1e-6))
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                lab[y1:y2, x1:x2] = k
    return lab


def label_borders(lab: np.ndarray) -> np.ndarray:
    """True where a pixel's 4-neighbour has another label (cut components there)."""
    b = np.zeros(lab.shape, bool)
    b[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    b[:, :-1] |= lab[:, 1:] != lab[:, :-1]
    b[1:, :] |= lab[1:, :] != lab[:-1, :]
    b[:-1, :] |= lab[1:, :] != lab[:-1, :]
    return b


# --------------------------------------------------------------------------- mask clipping


def window_motion(then: ScreenLayout | None, now: ScreenLayout | None, window_id: int | None) -> tuple[float, float]:
    """How far a window's top-left moved between two layouts (0, 0 if unknown)."""
    if then is None or now is None or window_id is None or then is now:
        return 0.0, 0.0
    a, b = then.window(window_id), now.window(window_id)
    if a is None or b is None:
        return 0.0, 0.0
    return float(b.xyxy[0] - a.xyxy[0]), float(b.xyxy[1] - a.xyxy[1])


def clip_box(box: Box, layout: ScreenLayout | None, window_id: int | None, frame_wh: tuple[int, int] | None = None) -> list[Box]:
    """One mask box → the rects it may cover: its window's visible region
    (or, without a known window, the screen minus the menu bar and Dock)."""
    if layout is None:
        if frame_wh is None:
            return [box]
        c = inter(box, (0.0, 0.0, float(frame_wh[0]), float(frame_wh[1])))
        return [c] if c is not None else []
    g = geometry(layout)
    if window_id is not None and window_id in g.visible:
        return rects_inter(g.visible[window_id], box)
    if window_id is not None and not layout.stale and layout.window(window_id) is None:
        return []  # its window closed (a late result must not paint the window behind)
    c = inter(box, g.content)
    return [c] if c is not None else []


def clip_specs(
    specs: Sequence[MaskSpec],
    layout: ScreenLayout | None,
    *,
    frame_layout: ScreenLayout | None = None,
    pad_px: float = 0.0,
    frame_wh: tuple[int, int] | None = None,
    min_px: float = 1.0,
) -> list[MaskSpec]:
    """Masks as they may be drawn *now*: padded, moved with their window if
    it was dragged since ``frame_layout``, then cut to the window's current
    visible region (front windows, the menu bar and the Dock are never
    covered on behalf of a window behind them). A mask cut into pieces gets
    ``id~k`` ids; a single piece keeps its id."""
    out: list[MaskSpec] = []
    for m in specs:
        x1, y1, x2, y2 = (float(v) for v in m.xyxy)
        if pad_px:
            x1, y1, x2, y2 = x1 - pad_px, y1 - pad_px, x2 + pad_px, y2 + pad_px
        dx, dy = window_motion(frame_layout, layout, m.window_id)
        box = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        pieces = [p for p in clip_box(box, layout, m.window_id, frame_wh) if p[2] - p[0] >= min_px and p[3] - p[1] >= min_px]
        if len(pieces) > MAX_PIECES:
            pieces = sorted(pieces, key=area, reverse=True)[:MAX_PIECES]
        if len(pieces) == 1 and pieces[0] == (x1, y1, x2, y2):
            out.append(m if not pad_px else replace(m, xyxy=pieces[0]))
            continue
        for k, p in enumerate(pieces):
            rid = m.region_id if len(pieces) == 1 else f"{m.region_id}~{k}"
            out.append(replace(m, region_id=rid, xyxy=p))
    return out


# --------------------------------------------------------------------------- busy cap

_SPECULATIVE_ORDER = {VerdictState.PENDING: 0, VerdictState.UNSURE: 1}


def _union_px(boxes: Sequence[Box], clip: Box, grid: float) -> float:
    if not boxes:
        return 0.0
    w = max(1, int(math.ceil((clip[2] - clip[0]) / grid)))
    h = max(1, int(math.ceil((clip[3] - clip[1]) / grid)))
    m = np.zeros((h, w), bool)
    for b in boxes:
        c = inter(b, clip)
        if c is None:
            continue
        a = int((c[0] - clip[0]) // grid)
        bb = int((c[1] - clip[1]) // grid)
        cc = int(math.ceil((c[2] - clip[0]) / grid))
        d = int(math.ceil((c[3] - clip[1]) / grid))
        m[bb:d, a:cc] = True
    return float(np.count_nonzero(m)) * grid * grid


def cap_coverage(
    specs: Sequence[MaskSpec],
    layout: ScreenLayout | None,
    frame_wh: tuple[int, int],
    *,
    cap_frac: float = BUSY_COVER_FRAC,
    shed_frac: float = BUSY_SHED_FRAC,
    grid: float = 16.0,
) -> tuple[list[MaskSpec], bool]:
    """Busy screens: masks over ``cap_frac`` of a window's visible area (of
    the screen for masks without a window) raise the "busy" warning; over
    ``shed_frac`` speculative masks (pre-frost first, then unsure, largest
    first) are dropped until back at ``cap_frac`` — a warning instead of a
    whole-window blur. Confirmed UNSAFE masks are always kept.
    Returns (kept, busy).

    Measured on the eval_live web pages: shedding already at 60 % doubled
    exposure with a late analyzer (image grids are legitimately > 60 %
    pictures), so shedding waits for ~whole-window cover."""
    if not specs or cap_frac <= 0:
        return list(specs), False
    W, H = frame_wh
    groups: dict[int | None, list[MaskSpec]] = {}
    for m in specs:
        groups.setdefault(m.window_id, []).append(m)
    g = geometry(layout) if layout is not None else None
    keep: list[MaskSpec] = []
    busy = False
    for wid, ms in groups.items():
        if g is not None and wid is not None and wid in g.visible:
            clip = g.visible_bbox(wid) or g.content
            limit = cap_frac * max(1.0, g.visible_area(wid))
        else:
            clip = g.content if g is not None else (0.0, 0.0, float(W), float(H))
            limit = cap_frac * max(1.0, area(clip))
        cur = list(ms)
        covered = _union_px([m.xyxy for m in cur], clip, grid)
        if covered <= limit:
            keep.extend(cur)
            continue
        busy = True
        if covered <= limit * max(1.0, shed_frac / cap_frac):
            keep.extend(cur)
            continue
        spec = sorted(
            (m for m in cur if m.verdict in _SPECULATIVE_ORDER),
            key=lambda m: (_SPECULATIVE_ORDER[m.verdict], -area(m.xyxy)),
        )
        for m in spec:
            cur.remove(m)
            covered = _union_px([c.xyxy for c in cur], clip, grid)
            if covered <= limit:
                break
        keep.extend(cur)
    return keep, busy


# --------------------------------------------------------------------------- heavy windows


class HeavyTracker:
    """Windows that recently had several UNSAFE masks ("heavy" pages).

    ``observe`` every active UNSAFE mask (by a stable key) on each frame; a
    window is heavy while at least ``n`` distinct keys were seen within the
    last ``window_s`` seconds, so it decays ``window_s`` after the hits stop.
    """

    def __init__(self) -> None:
        self._seen: dict[int | None, dict[str, float]] = {}

    def clear(self) -> None:
        self._seen.clear()

    def observe(self, window_id: int | None, key: str, t_ms: float) -> None:
        self._seen.setdefault(window_id, {})[str(key)] = float(t_ms)

    def count(self, window_id: int | None, t_ms: float, window_s: float) -> int:
        seen = self._seen.get(window_id)
        if not seen:
            return 0
        horizon = float(t_ms) - 1000.0 * float(window_s)
        for k in [k for k, t in seen.items() if t < horizon]:
            del seen[k]
        return len(seen)

    def heavy(self, window_id: int | None, t_ms: float, n: int, window_s: float) -> bool:
        return self.count(window_id, t_ms, window_s) >= max(1, int(n))

    def heavy_windows(self, t_ms: float, n: int, window_s: float) -> set[int | None]:
        return {w for w in list(self._seen) if self.heavy(w, t_ms, n, window_s)}


def prefrost_settings(policy: object, mode: object = None) -> tuple[str, int, float]:
    """(mode, heavy_masks, heavy_window_s) from a loaded mode policy, with the
    planned defaults when the policy predates the ``prefrost`` block:
    aggressive "always", medium "adaptive", off "never"."""
    default = "never"
    name = str(getattr(mode, "value", mode) or getattr(getattr(policy, "mode", None), "value", "") or "")
    if name == "aggressive":
        default = "always"
    elif name == "medium":
        default = "adaptive"
    pf = str(getattr(policy, "prefrost_mode", default) or default).strip().lower()
    if pf not in ("always", "adaptive", "never"):
        pf = default
    n = int(getattr(policy, "prefrost_heavy_masks", 3) or 3)
    s = float(getattr(policy, "prefrost_heavy_window_s", 10.0) or 10.0)
    return pf, n, s
