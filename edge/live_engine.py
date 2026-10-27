"""Live runtime: capture → scroll follow → analyzer → masks → overlay.

Threads::

    CaptureWorker   grab + window list (rate-limited, monotonic t) →
                    EngineCore.on_frame (fast path: per-window
                    phaseCorrelate + matchTemplate move every mask on
                    *every* frame; dragged windows move their own masks)
                    → latest-frame-wins slot
    VisionWorker    owns the LiveAnalyzer; analyzes the newest frame when
                    the gate reports change, a re-look, or a ~500 ms keyframe
                    → EngineCore.apply_result (re-based to the current frame)
    main (AppKit)   NSApplication.run + NSTimer @ 60 Hz: pull the mask set,
                    re-clip it to the *current* window list, move overlay
                    subviews, publish island state
    island bridge   Unix-socket JSON lines (``edge.island_bridge``)

Window awareness (``ScreenLayout`` from ``edge.capture``): every mask
belongs to one app window and is only ever drawn inside that window's
visible region (never over the menu bar, the Dock or a window in front).
Without a window list (mss, eval web pages, tests) the whole frame is one
region, as before.

``EngineCore`` holds all the logic and is used unchanged by the
deterministic, single-threaded ``run_headless`` for the eval harness.

Quit: q / Esc in the terminal, Ctrl-C, or the island's guardian ``quit``.
This is not zero-exposure: capture sees pixels after they are on screen.
"""

from __future__ import annotations

import inspect
import os
import select
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Iterable

import cv2
import numpy as np

from edge.mask_manager import MaskManager, shift_box
from edge.quick_boxes import QuickBoxes
from edge.scroll_follow import (
    PhaseShift,
    Shift,
    detect_change,
    local_shift,
    relocalize,
    residual_boxes,
    small_scale,
    to_small_gray,
)
from vision_core.live.contracts import (
    AnalyzeResult,
    Box,
    MaskSpec,
    ProtectionMode,
    ScreenLayout,
    VerdictState,
)
from vision_core.live.windows import (
    BUSY_WARNING,
    HeavyTracker,
    box_minus,
    cap_coverage,
    clip_specs,
    geometry,
    inter,
    label_map,
    prefrost_settings,
    rects_inter,
    rects_minus,
)

try:  # the medium agent's rate-limited JSON-lines log (no pixels, no titles)
    from edge.engine_log import EngineLog
except Exception:  # noqa: BLE001 - optional
    EngineLog = None  # type: ignore[assignment]

KEYFRAME_MS = 500.0
# Pass the changed-area hint to the analyzer. Off: the analyzer's `changed`
# path loses clipped pictures (eval_live engine[auto] latency 0: 26% exposed
# with the hint vs 3.4% without). Full-frame looks are cache-cheap.
ANALYZE_CHANGED_HINT = os.environ.get("SG_ANALYZE_HINT") == "1"
TARGET_FPS = 12.0
OVERLAY_HZ = 60.0
HEADLESS_FRAME_MS = 1000.0 / TARGET_FPS
SMALL_HISTORY = 48  # low-res frames kept to re-base late analyzer results
ALERT_HOLD_S = 1.5
WARNING_HOLD_MS = 2000.0  # keep a "busy" warning up this long after the last trigger
WINDOW_GONE_MAX_MS = 1500.0  # a closed window's masks never outlive a stale window list longer than this
PANE_MIN_RESPONSE = 0.3
FOLLOW_PAD_SMALL = 1.5  # small px of pad on masks that moved since confirmed
PREFROST_EXPAND = 12  # small px of context around a revealed strip
PREFROST_MAX_MS = 4.0  # fast-path budget for picture boxes per frame
PREFROST_MIN_PX = 24.0  # capture px; thinner boxes only at a revealed edge
PREFROST_TOUCH = 1.0  # small px: "touches" tolerance for continuation
PREFROST_COVERED = 0.9  # share of a candidate already under masks → skip (less = gaps)
FOLLOW_MAX_WINDOWS = 8  # per-window scroll follow for at most this many (largest) windows
FOLLOW_MIN_SMALL = 16  # small px: smaller windows are not scroll-followed


def _union(a: Box | None, b: Box | None) -> Box | None:
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _inter(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _covered_share(a: Box, boxes: list[Box], grid: float = 8.0) -> float:
    """Share of ``a`` under the union of ``boxes`` (rasterized, coarse)."""
    if not boxes:
        return 0.0
    w = max(1, int(np.ceil((a[2] - a[0]) / grid)))
    h = max(1, int(np.ceil((a[3] - a[1]) / grid)))
    m = np.zeros((h, w), bool)
    for b in boxes:
        x1 = int(np.floor(max(0.0, b[0] - a[0]) / grid))
        y1 = int(np.floor(max(0.0, b[1] - a[1]) / grid))
        x2 = int(np.ceil(min(a[2] - a[0], b[2] - a[0]) / grid))
        y2 = int(np.ceil(min(a[3] - a[1], b[3] - a[1]) / grid))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
    return float(m.mean())


def _box_minus(a: Box, b: Box, min_px: float = 4.0) -> list[Box]:
    """``a`` minus ``b`` as up to four slabs (thinner than ``min_px`` dropped)."""
    return box_minus(a, b, min_px)


def _pane_at(
    box: Box, panes: list[tuple[Box, float, float]], loose: bool = False
) -> tuple[Box, float, float] | None:
    """The moving pane that holds this box's centre, if any."""
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    for pane in panes:
        p = pane[0]
        if p[0] <= cx <= p[2] and p[1] <= cy <= p[3]:
            return pane
    if loose:
        # A vertical scroller spans its column: same x-range is enough.
        for pane in panes:
            p = pane[0]
            if p[0] <= cx <= p[2] and pane[2] and not pane[1]:
                return pane
    return None


def _inside(a: Box, b: Box) -> bool:
    return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]


def _group_panes(panes: list[tuple[Box, float, float]], tol: float) -> list[tuple[Box, float, float]]:
    out: list[tuple[Box, float, float]] = []
    for box, dx, dy in sorted(panes, key=lambda p: -(p[0][2] - p[0][0]) * (p[0][3] - p[0][1])):
        for i, (ob, odx, ody) in enumerate(out):
            if abs(odx - dx) <= tol and abs(ody - dy) <= tol:
                out[i] = (_union(ob, box), odx, ody)
                break
        else:
            out.append((box, dx, dy))
    return out


def _union_ok(track: Box, cand: Box, max_widen: float = 1.3) -> bool:
    """Grow along the scroll (column) only: a candidate may lengthen a mask
    but not widen it much (merged boxes, a neighbour)."""
    u = _union(track, cand)
    return (u[2] - u[0]) <= max_widen * max(1.0, track[2] - track[0])


def _same_column(a: Box, b: Box) -> bool:
    """Same picture column: a vertical scroll continues a picture downward
    or upward, never sideways into its neighbour."""
    ox = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    return ox / max(1.0, a[2] - a[0]) >= 0.8 and ox / max(1.0, b[2] - b[0]) >= 0.6


def _continues(a: Box, b: Box, tol: float) -> bool:
    """``a`` touches ``b`` along an edge (within ``tol``)."""
    near_y = abs(a[1] - b[3]) <= tol or abs(a[3] - b[1]) <= tol
    near_x = abs(a[0] - b[2]) <= tol or abs(a[2] - b[0]) <= tol
    ovx = min(a[2], b[2]) - max(a[0], b[0]) > 0
    ovy = min(a[3], b[3]) - max(a[1], b[1]) > 0
    return (near_y and ovx) or (near_x and ovy)


def _touches_any(a: Box, rects: list[Box], tol: float) -> bool:
    return any(_overlaps(a, r) or _continues(a, r, tol) for r in rects)


def _merge_rects(rects: list[tuple[Box, Box, int | None]]) -> list[tuple[Box, list[Box], int | None]]:
    """Union overlapping search rects (of one window) so a picture is looked at once."""
    out: list[tuple[Box, list[Box], int | None]] = []
    for grown, core, wid in sorted(rects, key=lambda r: (r[0][1], r[0][0])):
        for i, (g, cores, w) in enumerate(out):
            if w == wid and _overlaps(g, grown):
                out[i] = (_union(g, grown), cores + [core], w)
                break
        else:
            out.append((grown, [core], wid))
    return out


@dataclass
class AnalyzeRequest:
    seq: int
    t_ms: float
    changed: Box | None  # None = whole frame
    full: bool
    reason: str


@dataclass
class _Region:
    """One scroll-follow region: an app window (or the whole frame)."""

    wid: int | None
    rect: Box  # capture px, now (clipped to the frame)
    prev_rect: Box | None  # the same window-local area in the previous frame
    moved: tuple[float, float] = (0.0, 0.0)  # window drag since the previous frame
    new: bool = False  # appeared, resized, or no previous frame: all of it is fresh


def _window_regions(
    layout: ScreenLayout | None, prev: ScreenLayout | None, W: int, H: int
) -> list[_Region]:
    frame = (0.0, 0.0, float(W), float(H))
    if layout is None:
        return [_Region(None, frame, frame)]
    g = geometry(layout)
    gp = geometry(prev) if prev is not None else None
    out: list[_Region] = []
    for win in g.apps[: FOLLOW_MAX_WINDOWS * 4]:
        wid = win.window_id
        now = tuple(float(v) for v in win.xyxy)
        rect = g.bounds[wid]
        old = prev.window(wid) if prev is not None else None
        if old is None or gp is None or wid not in gp.bounds:
            out.append(_Region(wid, rect, None, new=True))
            continue
        then = tuple(float(v) for v in old.xyxy)
        if abs((now[2] - now[0]) - (then[2] - then[0])) > 0.5 or abs((now[3] - now[1]) - (then[3] - then[1])) > 0.5:
            out.append(_Region(wid, rect, None, new=True))  # resized / zoom animation
            continue
        dx, dy = now[0] - then[0], now[1] - then[1]
        # Window-local area visible in both frames.
        loc_now = (rect[0] - now[0], rect[1] - now[1], rect[2] - now[0], rect[3] - now[1])
        pb = gp.bounds[wid]
        loc_then = (pb[0] - then[0], pb[1] - then[1], pb[2] - then[0], pb[3] - then[1])
        loc = inter(loc_now, loc_then)
        if loc is None:
            out.append(_Region(wid, rect, None, new=True))
            continue
        cur = (now[0] + loc[0], now[1] + loc[1], now[0] + loc[2], now[1] + loc[3])
        prv = (then[0] + loc[0], then[1] + loc[1], then[0] + loc[2], then[1] + loc[3])
        out.append(_Region(wid, cur, prv, moved=(dx, dy)))
    return out


class EngineCore:
    """Scroll follow + mask persistence + analysis gate. Not thread-safe:
    the threaded engine wraps every call in one lock.

    ``prefrost``: True = pre-frost new pictures everywhere ("always"),
    False = never; ``prefrost_mode`` ("always" | "adaptive" | "never")
    overrides it. "adaptive" pre-frosts only in *heavy* windows: windows
    with ≥ ``heavy_masks`` UNSAFE masks active within ``heavy_window_s``."""

    def __init__(
        self,
        *,
        keyframe_ms: float = KEYFRAME_MS,
        prefrost: bool = False,
        quick: QuickBoxes | None = None,
        prefrost_mode: str | None = None,
        heavy_masks: int = 3,
        heavy_window_s: float = 10.0,
    ) -> None:
        self.keyframe_ms = float(keyframe_ms)
        self.prefrost_mode = prefrost_mode or ("always" if prefrost else "never")
        self.heavy_masks = int(heavy_masks)
        self.heavy_window_s = float(heavy_window_s)
        self.heavy = HeavyTracker()
        self.quick = quick if quick is not None else QuickBoxes()
        self.prefrost_ms = 0.0
        self.debug_log: list | None = None  # (seq, box, decision) when set
        self.known: list[tuple[Box, int | None]] = []  # analyzed, not masked (SAFE) pictures, followed
        self._last_frame: np.ndarray | None = None
        self._motion: deque[tuple[int, dict, list]] = deque(maxlen=SMALL_HISTORY * 4)
        self._follow_hist: deque[float] = deque(maxlen=512)
        self.masks = MaskManager()
        self._phase: dict[int | None, PhaseShift] = {}
        self.seq = -1
        self.t_ms = 0.0
        self.frame_shape: tuple[int, int] | None = None
        self.scale = (4.0, 4.0)
        self.prev_small: np.ndarray | None = None
        self._smalls: deque[tuple[int, np.ndarray]] = deque(maxlen=SMALL_HISTORY)
        self.layout: ScreenLayout | None = None
        self.prev_layout: ScreenLayout | None = None
        self._lab: np.ndarray | None = None  # label map at small scale (current frame)
        self._lab_prev: np.ndarray | None = None
        self._deferred: list[tuple[Box, int | None]] = []  # fresh rects seen while the window list was stale
        self.pending_change: Box | None = None
        self.urgent = True  # first frame: analyze the whole thing now
        self.last_analyze_t: float | None = None
        self.last_result_seq = -1
        self.change_seq = -1  # last frame where the fast path saw something the analyzer must look at
        self.results = 0
        self.last_counts: dict[str, int] = {}
        self.last_shift = Shift()
        self.last_change = 0.0
        self.follow_ms = 0.0  # last fast-path cost
        self.follow_ms_max = 0.0
        self.weak_matches = 0
        self.version = 0
        self.warning: dict | None = None
        self._warning_t: float | None = None
        self._specs: list[MaskSpec] = []
        self._drawn: list[MaskSpec] = []

    @property
    def prefrost(self) -> bool:
        return self.prefrost_mode != "never"

    @prefrost.setter
    def prefrost(self, on: bool) -> None:
        self.prefrost_mode = "always" if on else "never"

    def set_prefrost_policy(self, mode: str, heavy_masks: int = 3, heavy_window_s: float = 10.0) -> None:
        self.prefrost_mode = mode if mode in ("always", "adaptive", "never") else "never"
        self.heavy_masks = int(heavy_masks)
        self.heavy_window_s = float(heavy_window_s)

    def heavy_set(self) -> set:
        """Heavy windows now (only meaningful for an adaptive policy)."""
        if self.prefrost_mode != "adaptive":
            return set()
        return self.heavy.heavy_windows(self.t_ms, self.heavy_masks, self.heavy_window_s)

    def heavy_window(self, window_id: int | None) -> bool:
        return self.heavy.heavy(window_id, self.t_ms, self.heavy_masks, self.heavy_window_s)

    def _prefrost_here(self, window_id: int | None) -> bool:
        if self.prefrost_mode == "always":
            return True
        if self.prefrost_mode == "adaptive":
            return self.heavy_window(window_id)
        return False

    # ---- fast path -------------------------------------------------------

    def reset(self) -> None:
        """Forget motion state (pause / resume, mode change to OFF)."""
        self.masks.clear()
        self.prev_small = None
        self._smalls.clear()
        self._motion.clear()
        self.known = []
        self._last_frame = None
        self.pending_change = None
        self._deferred = []
        self.heavy.clear()
        self.urgent = True
        self._publish()

    def _labels(self, layout: ScreenLayout | None, shape: tuple[int, int]) -> np.ndarray | None:
        if layout is None:
            return None
        return label_map(layout, shape, self.scale)

    def _small_rect(self, rect: Box, shape: tuple[int, int]) -> tuple[int, int, int, int]:
        sx, sy = self.scale
        h, w = shape[:2]
        x1 = int(max(0, min(w, np.ceil(rect[0] / sx - 1e-6))))
        y1 = int(max(0, min(h, np.ceil(rect[1] / sy - 1e-6))))
        x2 = int(max(0, min(w, np.floor(rect[2] / sx + 1e-6))))
        y2 = int(max(0, min(h, np.floor(rect[3] / sy + 1e-6))))
        return x1, y1, x2, y2

    def on_frame(
        self,
        frame_bgr: np.ndarray,
        t_ms: float,
        small: np.ndarray | None = None,
        windows: ScreenLayout | None = None,
    ) -> int:
        """Move every mask with its content (and pre-frost new pictures where
        the policy wants it). ``windows``: the window list captured with this
        frame. Returns this frame's seq."""
        t0 = time.perf_counter()
        if small is None:
            small = to_small_gray(frame_bgr)
        self.seq += 1
        seq = self.seq
        self.t_ms = float(t_ms)
        h, w = frame_bgr.shape[:2]
        if self.frame_shape != (h, w):
            self.frame_shape = (h, w)
            self.masks.frame_size = (float(w), float(h))
            self.prev_small = None
            self.layout = None
        if windows is not None and tuple(windows.frame_wh) != (w, h):
            windows = None
        self.scale = small_scale(frame_bgr.shape, small.shape)
        self.prev_layout, self.layout = self.layout, windows
        self._lab_prev, self._lab = self._lab, self._labels(windows, small.shape)
        prev = self.prev_small
        if prev is not None and prev.shape != small.shape:
            prev = None
        sx, sy = self.scale
        g = geometry(windows) if windows is not None else None
        gp = geometry(self.prev_layout) if self.prev_layout is not None else None

        self._windows_gone(windows)
        motions: dict[int | None, tuple[float, float]] = {}
        panes_all: list[tuple[Box, float, float, int | None]] = []
        fresh: list[tuple[Box, int | None]] = []
        best_shift, best_area, max_frac = Shift(), -1.0, 0.0
        regions = _window_regions(windows, self.prev_layout, w, h)
        if g is not None:
            # Largest windows first; small ones are not scroll-followed.
            regions.sort(key=lambda r: -(r.rect[2] - r.rect[0]) * (r.rect[3] - r.rect[1]))
        for k, reg in enumerate(regions):
            wid = reg.wid
            vis_now = g.visible.get(wid, []) if g is not None else [reg.rect]
            if prev is None or reg.new:
                motions[wid] = (0.0, 0.0)
                if prev is not None and wid is not None:
                    # A window appeared or was resized: all of it needs a look.
                    self.masks.mark_stale(seq, wid)
                    self.known = [kb for kb in self.known if kb[1] != wid]
                    self.urgent = True
                fresh.extend((r, wid) for r in vis_now)
                if prev is None:
                    max_frac = 1.0
                continue
            cx1, cy1, cx2, cy2 = self._small_rect(reg.rect, small.shape)
            px1, py1, px2, py2 = self._small_rect(reg.prev_rect, small.shape)
            cw, ch = min(cx2 - cx1, px2 - px1), min(cy2 - cy1, py2 - py1)
            if cw < 4 or ch < 4 or (g is not None and k >= FOLLOW_MAX_WINDOWS) or (g is not None and min(cw, ch) < FOLLOW_MIN_SMALL):
                motions[wid] = reg.moved
                if reg.moved != (0.0, 0.0):
                    self.masks.move_window(wid, *reg.moved)
                continue
            cur_c = small[cy1 : cy1 + ch, cx1 : cx1 + cw]
            prev_c = prev[py1 : py1 + ch, px1 : px1 + cw]
            valid = None
            if self._lab is not None and self._lab_prev is not None and g is not None and gp is not None:
                kn, kp = g.labels.get(wid, -1), gp.labels.get(wid, -1)
                valid = ((self._lab[cy1 : cy1 + ch, cx1 : cx1 + cw] == kn) & (self._lab_prev[py1 : py1 + ch, px1 : px1 + cw] == kp)).astype(np.uint8)
                if valid.all():
                    valid = None
            ox, oy = cx1 * sx, cy1 * sy  # crop origin, capture px (current frame)
            dragged = reg.moved != (0.0, 0.0)
            if dragged:
                shift = Shift(0.0, 0.0, 1.0, True)  # content moved with the window, exactly
            else:
                phase = self._phase.setdefault(wid, PhaseShift())
                shift = phase(prev_c, cur_c, self.scale, valid)
            change = detect_change(prev_c, cur_c, shift, self.scale, valid=valid)
            gdx, gdy = (shift.dx, shift.dy) if shift.ok else (0.0, 0.0)
            area = float(cw * ch)
            if area > best_area:
                best_shift, best_area = shift, area
            max_frac = max(max_frac, change.fraction)
            motions[wid] = (reg.moved[0] + gdx, reg.moved[1] + gdy)
            cbox = None if change.box is None else shift_box(change.box, ox, oy)
            cboxes = [shift_box(b, ox, oy) for b in change.boxes]

            # Panes that moved on their own (inner scrollers) inside this
            # window: a local phase correlation per changed blob. What is
            # left after undoing that motion is content the pane scrolled in.
            panes: list[tuple[Box, float, float]] = []
            region_fresh: list[Box] = []
            if not change.large and not dragged:
                still: list[Box] = []
                for blob in cboxes:
                    lblob = shift_box(blob, -ox, -oy)
                    ls = local_shift(prev_c, cur_c, lblob, self.scale)
                    # A pane is a region that moved, and not with the page.
                    if (
                        ls.moved
                        and ls.response >= PANE_MIN_RESPONSE
                        and (abs(ls.dx - gdx) > sx or abs(ls.dy - gdy) > sy)
                        and max(abs(ls.dx) / sx, abs(ls.dy) / sy) >= 1.0
                    ):
                        panes.append((blob, ls.dx, ls.dy))
                    else:
                        still.append(blob)
                # One inner scroller shows up as many blobs (gutters do not
                # change); blobs that moved alike are one pane.
                panes = _group_panes(panes, tol=1.5 * max(self.scale))
                for pane_box, pdx, pdy in panes:
                    lp = shift_box(pane_box, -ox, -oy)
                    region_fresh.extend(
                        shift_box(b, ox, oy) for b in residual_boxes(prev_c, cur_c, lp, Shift(pdx, pdy, 1.0, True), self.scale)
                    )
                region_fresh.extend(b for b in still if _pane_at(b, panes) is None or not _inside(b, _pane_at(b, panes)[0]))
            elif dragged and not change.large:
                region_fresh.extend(cboxes)  # content that changed while the window moved
            panes_all.extend((p[0], p[1], p[2], wid) for p in panes)

            # Move this window's masks.
            if dragged:
                self.masks.move_window(wid, *reg.moved)
            elif cbox is not None or shift.moved:
                prev_full_small = self.prev_small
                for track in list(self.masks.tracks.values()):
                    if g is not None and track.window_id != wid:
                        continue
                    pane = _pane_at(track.box, panes, loose=True)
                    if not shift.moved and pane is None and not _overlaps(track.box, cbox):
                        continue  # nothing moved and nothing changed under it
                    pred = (pane[1], pane[2]) if pane is not None else (gdx, gdy)
                    extra = [(gdx, gdy), track.vel]
                    if pane is None:
                        extra.extend((p[1], p[2]) for p in panes[:3])
                    clip = pane[0] if pane is not None else (reg.rect if g is not None else None)
                    rel = relocalize(prev_full_small, small, track.box, self.scale, pred, extra=extra, clip=clip)
                    if rel.ok:
                        self.masks.move(track.key, rel.dx, rel.dy)
                        track.vel = (rel.dx, rel.dy)
                        track.weak_frames = 0
                    else:
                        # Fail closed: keep covering where the content most
                        # likely went, and ask the analyzer to look there.
                        self.masks.move(track.key, pred[0], pred[1])
                        track.vel = pred
                        track.weak_frames += 1
                        self.weak_matches += 1
                        self.pending_change = _union(self.pending_change, track.box)
            if change.large:
                self.masks.mark_stale(seq, wid)
                self.urgent = True
                self.known = [kb for kb in self.known if kb[1] != wid]
                region_fresh = list(vis_now)
                if wid is None:
                    self.pending_change = None
            elif cbox is not None:
                if self.pending_change is not None and shift.moved and g is None:
                    self.pending_change = shift_box(self.pending_change, shift.dx, shift.dy)
                self.pending_change = _union(self.pending_change, cbox)
            # Only what this window shows now is fresh for it.
            if g is not None:
                clipped: list[Box] = []
                for b in region_fresh:
                    clipped.extend(rects_inter(vis_now, b))
                region_fresh = clipped
                # Revealed by a window in front moving away / closing.
                if gp is not None and wid in gp.visible:
                    then = [shift_box(r, *reg.moved) for r in gp.visible[wid]]
                    revealed = list(vis_now)
                    for r in then:
                        revealed = rects_minus(revealed, r)
                    for r in revealed:
                        if (r[2] - r[0]) >= sx and (r[3] - r[1]) >= sy:
                            region_fresh.append(r)
                            self.pending_change = _union(self.pending_change, r)
            fresh.extend((b, wid) for b in region_fresh)
        self._motion.append((seq, motions, panes_all))
        self.last_shift = best_shift
        self.last_change = max_frac
        if prev is None:
            self.masks.mark_stale(seq)
            self.urgent = True
            self.pending_change = None
            self.known = []
        self._move_known(motions, panes_all)
        self.masks.drop_offscreen(w, h)
        self._observe_heavy()
        self.prefrost_ms = 0.0
        if windows is not None and windows.stale:
            # The window list may not match the pixels (an open / close
            # animation): add no new cover now, look again next frame.
            self._deferred.extend(fresh)
        else:
            fresh = self._deferred + fresh if self._deferred else fresh
            self._deferred = []
            if self.prefrost and fresh:
                self._prefrost(frame_bgr, small, fresh, seq)
        if self.urgent or self.pending_change is not None or fresh or self._deferred:
            self.change_seq = seq
        self.masks.record_all(seq)
        self.prev_small = small
        self._last_frame = frame_bgr
        self._smalls.append((seq, small))
        self.follow_ms = (time.perf_counter() - t0) * 1000.0
        self.follow_ms_max = max(self.follow_ms_max, self.follow_ms)
        self._follow_hist.append(self.follow_ms)
        self._publish()
        return seq

    def _windows_gone(self, layout: ScreenLayout | None) -> None:
        """Masks of a window that left the window list go, once the list is
        steady; while it is stale (the close / minimise animation, which
        ``edge.capture.WindowLister`` flags for ~300 ms after any change)
        they stay. Tracks without a window get one."""
        if layout is None:
            return
        g = geometry(layout)
        gone: list[str] = []
        for key, track in self.masks.tracks.items():
            if track.window_id is None:
                track.window_id = g.owner_of(track.box)
                continue
            if layout.window(track.window_id) is not None:
                track.gone_since_ms = None
                continue
            if track.gone_since_ms is None:
                track.gone_since_ms = self.t_ms
            if layout.stale and self.t_ms - track.gone_since_ms < WINDOW_GONE_MAX_MS:
                continue
            gone.append(key)
        for key in gone:
            del self.masks.tracks[key]

    def _observe_heavy(self) -> None:
        for track in self.masks.tracks.values():
            if track.verdict is VerdictState.UNSAFE and not track.prefrost:
                self.heavy.observe(track.window_id, track.region_id.split("#")[0], self.t_ms)

    # ---- fast-path pre-frost --------------------------------------------

    def _move_known(self, motions: dict, panes: list[tuple[Box, float, float, int | None]]) -> None:
        if not self.known or self.frame_shape is None:
            return
        h, w = self.frame_shape
        kept = []
        for box, wid in self.known:
            pane = _pane_at(box, [p for p in panes if p[3] == wid], loose=True)
            dx, dy = (pane[1], pane[2]) if pane is not None else motions.get(wid, (0.0, 0.0))
            nb = shift_box(box, dx, dy)
            if nb[2] > 0 and nb[3] > 0 and nb[0] < w and nb[1] < h:
                kept.append((nb, wid))
        self.known = kept

    def _prefrost(self, frame_bgr: np.ndarray, small: np.ndarray, fresh: list[tuple[Box, int | None]], seq: int) -> None:
        """Cover picture-like boxes in newly revealed / changed areas NOW, as
        PENDING frost, so a late analyzer cannot leave them exposed. Only
        inside app windows (never on system UI), each box capped at its
        window, and in "adaptive" mode only in heavy windows."""
        h, w = self.frame_shape
        sx, sy = self.scale
        ex, ey = PREFROST_EXPAND * sx, PREFROST_EXPAND * sy
        g = geometry(self.layout) if self.layout is not None else None
        rects = []
        for r, wid in fresh:
            if (r[2] - r[0]) * (r[3] - r[1]) <= 0:
                continue
            if g is not None and wid is None:
                continue
            if not self._prefrost_here(wid):
                continue
            bound = g.bounds.get(wid) if g is not None else None
            grown = (max(0.0, r[0] - ex), max(0.0, r[1] - ey), min(float(w), r[2] + ex), min(float(h), r[3] + ey))
            if bound is not None:
                grown = inter(grown, bound) or grown
            rects.append((grown, r, wid))
        if not rects:
            return
        # Biggest fresh areas first (a newly shown page or window), so a
        # string of small strips cannot eat the whole per-frame budget.
        rects = sorted(_merge_rects(rects), key=lambda r: -(r[0][2] - r[0][0]) * (r[0][3] - r[0][1]))
        t0 = time.perf_counter()
        tol = PREFROST_TOUCH * max(sx, sy)
        labels = (self._lab, self.scale) if self._lab is not None else None
        for grown, cores, wid in rects:
            left = PREFROST_MAX_MS - (time.perf_counter() - t0) * 1000.0
            if left <= 0.2:
                break
            for cand in self.quick(frame_bgr, [grown], small, self.scale, max_ms=left, labels=labels):
                if not any(_overlaps(cand, c) for c in cores):
                    continue  # old content next to the revealed area
                if (cand[2] - cand[0]) < PREFROST_MIN_PX or (cand[3] - cand[1]) < PREFROST_MIN_PX:
                    if not _touches_any(cand, cores, tol):
                        continue
                cwid = wid
                if g is not None:
                    cwid = g.owner_of(cand)
                    if cwid is None:
                        continue  # system UI / wallpaper: never pre-frosted
                    cand = inter(cand, g.bounds[cwid]) or cand
                self._place_prefrost(cand, seq, tol, cwid)
        self.prefrost_ms = (time.perf_counter() - t0) * 1000.0

    def _place_prefrost(self, cand: Box, seq: int, tol: float, wid: int | None = None) -> None:
        area = max(1.0, (cand[2] - cand[0]) * (cand[3] - cand[1]))
        grow = None
        hits = []
        best = 0.0
        for track in self.masks.tracks.values():
            if track.window_id != wid:
                continue
            inter_a = _inter(cand, track.box)
            if inter_a > 0:
                hits.append(track.box)
            if (inter_a > 0 or _continues(cand, track.box, tol)) and _same_column(cand, track.box):
                score = inter_a + 1.0
                if score > best:
                    grow, best = track, score
        covered = _covered_share(cand, hits) * area
        log = self.debug_log
        if grow is not None and 0 < covered / area < 0.98 and _union_ok(grow.box, cand):
            grow.box = _union(grow.box, cand)  # partly covered: extend the cover
            if log is not None:
                log.append((seq, cand, f"extend {grow.key}"))
            return
        if covered / area >= PREFROST_COVERED:
            if log is not None:
                log.append((seq, cand, "covered"))
            return
        known = [b for b, kw in self.known if kw == wid]
        if _covered_share(cand, [b for b in known if _overlaps(cand, b)]) >= 0.5:
            if log is not None:
                log.append((seq, cand, "known"))
            return  # judged SAFE already
        for box in known:
            if _continues(cand, box, tol) and _same_column(cand, box):
                if log is not None:
                    log.append((seq, cand, "known-continues"))
                return  # the rest of a SAFE picture (no gutter in between)
        if grow is not None and _union_ok(grow.box, cand):
            grow.box = _union(grow.box, cand)  # fail closed: extend the cover
            if log is not None:
                log.append((seq, cand, f"grow {grow.key}"))
            return
        if log is not None:
            log.append((seq, cand, "added"))
        self.masks.add_prefrost(cand, seq, window_id=wid)

    # ---- analysis gate ---------------------------------------------------

    def take_request(
        self, seq: int, t_ms: float, *, force: bool = False, windows: ScreenLayout | None = None
    ) -> AnalyzeRequest | None:
        """Should frame ``seq`` be analyzed? Clears the pending gate state.
        Never on a frame whose window list is stale (an open / close
        animation): its result would attribute pixels to the wrong window,
        so the existing masks stay and the next steady frame is analyzed."""
        if windows is not None and windows.stale:
            self.urgent = True
            return None
        keyframe = self.last_analyze_t is None or (t_ms - self.last_analyze_t) >= self.keyframe_ms
        full = self.urgent or keyframe
        if not (full or self.pending_change is not None or force):
            return None
        if self.urgent:
            reason = "change"
        elif keyframe:
            reason = "keyframe"
        elif self.pending_change is not None:
            reason = "gate"
        else:
            reason = "scheduled"
        changed = None if full else self.pending_change
        if changed is not None and self.frame_shape is not None:
            h, w = self.frame_shape
            changed = (
                max(0.0, changed[0]),
                max(0.0, changed[1]),
                min(float(w), changed[2]),
                min(float(h), changed[3]),
            )
        self.pending_change = None
        self.urgent = False
        self.last_analyze_t = float(t_ms)
        return AnalyzeRequest(seq=seq, t_ms=float(t_ms), changed=changed, full=full, reason=reason)

    def _small_at(self, seq: int) -> np.ndarray | None:
        for s, small in reversed(self._smalls):
            if s == seq:
                return small
        return None

    def _advance(self, box: Box, seq: int, window_id: int | None = None) -> tuple[float, float]:
        """Predicted motion of ``box`` from frame ``seq`` to now: per frame,
        the shift of the moving pane under it, else its window's motion
        (drag + scroll), else the page shift."""
        dx = dy = 0.0
        cur = box
        for s, motions, panes in self._motion:
            if s <= seq:
                continue
            pane = _pane_at(cur, [p for p in panes if p[3] == window_id], loose=True)
            if pane is not None:
                mdx, mdy = pane[1], pane[2]
            elif window_id in motions:
                mdx, mdy = motions[window_id]
            elif None in motions:
                mdx, mdy = motions[None]
            else:
                mdx, mdy = 0.0, 0.0
            dx += mdx
            dy += mdy
            cur = shift_box(cur, mdx, mdy)
        return dx, dy

    def apply_result(self, result: AnalyzeResult, seq: int, frame_bgr: np.ndarray | None = None) -> None:
        """Reconcile an analyzer result for frame ``seq`` into the current
        frame (``self.seq``), even if it arrives many frames late. PENDING
        masks in the result are the analyzer's (it applies the same
        pre-frost policy, ``modes.masks_for(..., heavy=)``)."""
        cur = self.prev_small
        analyzed = self._small_at(seq)
        if analyzed is None and frame_bgr is not None:
            analyzed = to_small_gray(frame_bgr)
        same_frame = seq == self.seq

        def place(mask: MaskSpec, _seq: int) -> tuple[Box, bool]:
            box = tuple(float(v) for v in mask.xyxy)
            if same_frame:
                return box, True
            motion = self._advance(box, seq, getattr(mask, "window_id", None))
            if analyzed is None or cur is None or analyzed.shape != cur.shape:
                return shift_box(box, *motion), False
            rel = relocalize(analyzed, cur, box, self.scale, motion)
            if rel.ok:
                return shift_box(box, rel.dx, rel.dy), True
            return shift_box(box, *motion), False

        before = {}
        for key, track in self.masks.tracks.items():
            then = track.box_at(seq)
            seen = None if then is None else shift_box(then, *self._advance(then, seq, track.window_id))
            before[key] = (track.box, seen, track.window_id)
        uncertain = self.masks.reconcile(result, seq, self.seq, place)
        for key in uncertain:
            track = self.masks.tracks.get(key)
            if track is not None:
                self.pending_change = _union(self.pending_change, track.box)
                self.change_seq = self.seq
        # Remember analyzed pictures that got no mask (SAFE): the fast path
        # must not pre-frost them again as they scroll.
        masked = {m.region_id for m in result.masks}
        known = []
        for region in result.regions:
            if region.region_id in masked:
                continue
            verdict = result.verdicts.get(region.region_id)
            if verdict is not None and verdict.state is not VerdictState.SAFE:
                continue
            box = tuple(float(v) for v in region.xyxy)
            wid = getattr(region, "window_id", None)
            known.append((box if same_frame else shift_box(box, *self._advance(box, seq, wid)), wid))
        if self.masks.last_large_change is None or seq >= self.masks.last_large_change:
            self.known = known
        if self.prefrost and self._last_frame is not None and self.prev_small is not None:
            # Never uncover without a look: where a mask shrank or went away,
            # anything revealed after the analyzed frame is still unjudged.
            freed: list[tuple[Box, int | None]] = []
            for key, (box, seen, wid) in before.items():
                if key in self.masks.tracks and self.masks.tracks[key].box == box:
                    continue
                # Only what the analyzed frame could not have seen.
                freed.extend((b, wid) for b in ([box] if seen is None else _box_minus(box, seen)))
            if freed:
                self._prefrost(self._last_frame, self.prev_small, freed, self.seq)
        if self.frame_shape is not None:
            h, w = self.frame_shape
            self.masks.drop_offscreen(w, h)
        self.last_result_seq = max(self.last_result_seq, seq)
        self.results += 1
        self.last_counts = dict(getattr(result, "counts", {}) or {})
        self._observe_heavy()
        self._publish()

    # ---- read side -------------------------------------------------------

    def _publish(self) -> None:
        if self.frame_shape is None:
            self._specs = []
            self._drawn = []
        else:
            h, w = self.frame_shape
            self._specs = self.masks.specs(w, h, FOLLOW_PAD_SMALL * max(self.scale))
            self._drawn = self.draw(self._specs, self.layout)
        self.version += 1

    def draw(
        self,
        specs: list[MaskSpec],
        layout: ScreenLayout | None,
        *,
        frame_layout: ScreenLayout | None = None,
        pad_px: float = 0.0,
        now_ms: float | None = None,
    ) -> list[MaskSpec]:
        """Masks as they may be drawn under ``layout``: cut to each mask's
        window's visible region (moved with the window since
        ``frame_layout``), never on the menu bar / Dock, and capped at the
        busy-screen limit (speculative masks dropped, warning raised)."""
        if self.frame_shape is None:
            return []
        h, w = self.frame_shape
        drawn = clip_specs(specs, layout, frame_layout=frame_layout if frame_layout is not None else layout,
                           pad_px=pad_px, frame_wh=(w, h))
        drawn, busy = cap_coverage(drawn, layout, (w, h))
        t = self.t_ms if now_ms is None else now_ms
        if busy:
            self.warning, self._warning_t = dict(BUSY_WARNING), t
        elif self._warning_t is not None and t - self._warning_t >= WARNING_HOLD_MS:
            self.warning, self._warning_t = None, None
        return drawn

    def visible(self) -> list[MaskSpec]:
        """Masks to draw on the latest frame (capture px, window-clipped)."""
        return list(self._drawn)

    def specs(self) -> list[MaskSpec]:
        """Masks before window clipping (capture px), for a re-clip against
        a newer window list (the 60 Hz overlay tick)."""
        return list(self._specs)


def _prefrost_policy(mode, analyzer=None) -> tuple[str, int, float]:
    """(prefrost mode, heavy masks, heavy window s) for a protection mode, from
    the analyzer's loaded policy when it is for that mode, else the policy
    file; planned defaults when the policy has no ``prefrost`` block."""
    try:
        pm = ProtectionMode(getattr(mode, "value", mode))
    except ValueError:
        return "never", 3, 10.0
    policy = getattr(analyzer, "policy", None)
    if policy is None or getattr(policy, "mode", None) != pm:
        try:
            from vision_core.live.modes import load_mode_policy

            policy = load_mode_policy(pm)
        except Exception:  # noqa: BLE001 - defaults below
            policy = None
    return prefrost_settings(policy, pm)


def _copy_spec(m: MaskSpec) -> MaskSpec:
    return MaskSpec(region_id=m.region_id, xyxy=tuple(m.xyxy), style=m.style, reason=m.reason, verdict=m.verdict,
                    window_id=getattr(m, "window_id", None))


def _takes(analyzer, name: str) -> bool:
    fn = getattr(analyzer, "analyze", None)
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _takes_windows(analyzer) -> bool:
    return _takes(analyzer, "windows")


def analyze_frame(
    analyzer,
    frame: np.ndarray,
    *,
    changed: Box | None,
    t_ms: float,
    windows: ScreenLayout | None,
    heavy_windows: set | None = None,
):
    """``analyzer.analyze`` with the window list (and the engine's heavy-window
    signal for adaptive pre-frost) when it takes them."""
    kw: dict = {}
    if windows is not None and _takes_windows(analyzer):
        kw["windows"] = windows
    if heavy_windows is not None and _takes(analyzer, "heavy_windows"):
        kw["heavy_windows"] = heavy_windows
    return analyzer.analyze(frame, changed=changed, t_ms=t_ms, **kw)


def run_headless(
    frames: Iterable[np.ndarray],
    analyzer,
    *,
    t_ms: Iterable[float] | None = None,
    analyze_every: int = 1,
    analyzer_latency_frames: int = 0,
    windows: Iterable[ScreenLayout | None] | None = None,
) -> list[list[MaskSpec]]:
    """Deterministic, single-threaded engine run (no AppKit, no threads).

    For frame ``i``: the fast path moves masks; a result launched
    ``analyzer_latency_frames`` frames ago is applied; then, if no analysis
    is in flight and ``i % analyze_every == 0`` (or a large change asked for
    an immediate look), ``analyzer.analyze`` runs on frame ``i`` and its
    result is held back for ``analyzer_latency_frames`` frames (0 = applied
    to frame ``i`` itself). ``windows``: per-frame window lists (as the live
    capture sends them). Returns the masks drawn on each frame (window-
    clipped, busy-capped).
    """
    every = max(1, int(analyze_every))
    latency = max(0, int(analyzer_latency_frames))
    pf_mode, pf_n, pf_s = _prefrost_policy(getattr(analyzer, "mode", None), analyzer)
    core = EngineCore(prefrost_mode=pf_mode, heavy_masks=pf_n, heavy_window_s=pf_s)
    times = None if t_ms is None else iter(t_ms)
    wins = None if windows is None else iter(windows)
    out: list[list[MaskSpec]] = []
    inflight: tuple[int, int, AnalyzeResult, np.ndarray] | None = None
    for i, frame in enumerate(frames):
        t = float(next(times)) if times is not None else i * HEADLESS_FRAME_MS
        layout = next(wins, None) if wins is not None else None
        seq = core.on_frame(frame, t, windows=layout)
        if inflight is not None and inflight[0] <= i:
            core.apply_result(inflight[2], inflight[1], inflight[3])
            inflight = None
        if inflight is None and (i % every == 0 or core.urgent):
            req = core.take_request(seq, t, force=True, windows=layout)
            if req is not None:
                result = analyze_frame(analyzer, frame, changed=req.changed if ANALYZE_CHANGED_HINT else None, t_ms=t,
                                       windows=layout, heavy_windows=core.heavy_set())
                if latency == 0:
                    core.apply_result(result, seq, frame)
                else:
                    inflight = (i + latency, seq, result, frame)
        out.append([_copy_spec(m) for m in core.visible()])
    return out


# ---------------------------------------------------------------------------
# Threaded runtime
# ---------------------------------------------------------------------------


class LatestSlot:
    """Latest-frame-wins handoff from capture to vision."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: tuple[int, float, np.ndarray, ScreenLayout | None] | None = None

    def put(self, seq: int, t_ms: float, frame: np.ndarray, windows: ScreenLayout | None = None) -> None:
        with self._cond:
            self._item = (seq, t_ms, frame, windows)
            self._cond.notify_all()

    def wait_newer(self, seq: int, timeout: float) -> tuple[int, float, np.ndarray, ScreenLayout | None] | None:
        with self._cond:
            if self._item is None or self._item[0] <= seq:
                self._cond.wait(timeout)
            item = self._item
        if item is None or item[0] <= seq:
            return None
        return item

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()


class AnalyzerUnavailable(RuntimeError):
    """The real analyzer (NudeNet via cv2.dnn / onnxruntime) could not load."""


def load_analyzer(mode: ProtectionMode):
    """The real analyzer, or :class:`AnalyzerUnavailable`. There is no
    stand-in: an engine without a working detector must say it is not
    protecting, never draw "no masks" as if the screen were clean."""
    try:
        from vision_core.live.analyzer import make_analyzer

        try:
            params = inspect.signature(make_analyzer).parameters
        except (TypeError, ValueError):
            params = {}
        analyzer = make_analyzer(mode=mode) if "mode" in params else make_analyzer()
        if getattr(analyzer, "mode", None) != mode:
            analyzer.set_mode(mode)
    except Exception as exc:  # noqa: BLE001 - any failure means: not protecting
        raise AnalyzerUnavailable(f"{type(exc).__name__}: {exc}"[:300]) from exc
    return analyzer, "live"


class CaptureWorker(threading.Thread):
    def __init__(self, engine: LiveEngine) -> None:
        super().__init__(name="sg-capture", daemon=True)
        self.engine = engine

    def run(self) -> None:
        eng = self.engine
        cap = eng.capture
        while not eng.stop_event.is_set():
            try:
                if eng.idle:
                    eng.stop_event.wait(0.2)
                    continue
                frame = cap.read()
                if frame is None:
                    eng.capture_lost(getattr(cap, "reason", "") or "capture unavailable")
                    if eng.stop_event.wait(1.0):
                        break
                    if hasattr(cap, "available") and getattr(cap, "_grabber", None) is not None:
                        cap.available = True  # retry the same grabber
                    continue
                eng.on_frame(frame.image_bgr, frame.t_ms, getattr(cap, "geom", None), getattr(cap, "windows", None))
                eng.capture_error = None
                ev, eng._mode_event = getattr(eng, "_mode_event", None), None
                eng.log_tick(ev)
            except Exception as exc:  # noqa: BLE001 - never let the capture thread die silently
                eng.report_capture_error(exc)
                eng.log_tick()
                if eng.stop_event.wait(0.5):
                    break


class VisionWorker(threading.Thread):
    def __init__(self, engine: LiveEngine) -> None:
        super().__init__(name="sg-vision", daemon=True)
        self.engine = engine

    def run(self) -> None:
        eng = self.engine
        seen = -1
        while not eng.stop_event.is_set():
            eng.apply_pending_mode()
            item = eng.slot.wait_newer(seen, timeout=0.1)
            if item is None or eng.idle:
                continue
            seq, t_ms, frame, windows = item
            with eng.lock:
                req = eng.core.take_request(seq, t_ms, windows=windows)
                epoch = eng.mode_epoch
                heavy = eng.core.heavy_set()
            seen = seq
            if req is None:
                continue
            t0 = time.perf_counter()
            try:
                result = analyze_frame(eng.analyzer, frame, changed=req.changed if ANALYZE_CHANGED_HINT else None,
                                       t_ms=t_ms, windows=windows, heavy_windows=heavy)
            except Exception as exc:  # keep masks; report, retry next frame
                eng.analyzer_error = f"{type(exc).__name__}: {exc}"
                with eng.lock:
                    eng.core.urgent = True
                eng.stop_event.wait(0.2)
                continue
            dt = (time.perf_counter() - t0) * 1000.0
            with eng.lock:
                eng.analyze_ms = dt
                eng.analyzer_error = None
                if eng.idle or eng.mode_epoch != epoch:
                    continue  # mode changed mid-analysis: result is for the old mode
                eng.core.apply_result(result, seq, frame)
            if eng.cloud is not None:
                try:
                    eng.cloud.on_result(result)
                except Exception:  # noqa: BLE001 - the cloud side never breaks the vision loop
                    pass


class _TerminalKeys(threading.Thread):
    """q / Esc in the launching terminal quits (cbreak, restored on exit)."""

    def __init__(self, on_quit) -> None:
        super().__init__(name="sg-keys", daemon=True)
        self.on_quit = on_quit
        self._stop = threading.Event()
        self._fd: int | None = None
        self._old = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            return
        try:
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            self._fd = None
            return
        super().start()

    def run(self) -> None:
        while not self._stop.is_set() and self._fd is not None:
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.1)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            try:
                ch = os.read(self._fd, 1)
            except OSError:
                return
            if ch in (b"q", b"Q", b"\x1b"):
                self.on_quit("terminal")
                return

    def close(self) -> None:
        self._stop.set()
        if self._fd is not None and self._old is not None:
            try:
                import termios

                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except Exception:
                pass
        self._fd = None


class LiveEngine:
    """Owns the threads, overlay, and island bridge for ``--source live``."""

    def __init__(
        self,
        *,
        capture,
        analyzer,
        mode: ProtectionMode,
        overlay=None,
        bridge=None,
        analyzer_kind: str = "live",
        keyframe_ms: float = KEYFRAME_MS,
    ) -> None:
        self.capture = capture
        self.analyzer = analyzer
        self.analyzer_kind = analyzer_kind
        self.overlay = overlay
        self.bridge = bridge
        pf_mode, pf_n, pf_s = _prefrost_policy(mode, analyzer)
        self.core = EngineCore(keyframe_ms=keyframe_ms, prefrost_mode=pf_mode, heavy_masks=pf_n, heavy_window_s=pf_s)
        self.lock = threading.Lock()
        self.slot = LatestSlot()
        self.stop_event = threading.Event()
        self.stop_reason = ""
        self.mode = mode  # effective mode (the agent may raise it for an app)
        self.pin_mode = mode  # the parent's mode (PIN); protection never goes below it
        self.cloud = None  # edge.cloud_glue.CloudGlue when the Mac is set up for the cloud
        self.last_frame_mono: float | None = None
        self.mode_epoch = 0
        self._pending_mode: ProtectionMode | None = None
        self.paused = False
        self.geom = None
        self.capture_ok = True
        self.capture_reason = ""
        self.analyze_ms = 0.0
        self.analyzer_error: str | None = None
        self.capture_error: str | None = None
        self._capture_error_sent: str | None = None
        self.window_watch = None  # edge.capture.WindowWatcher: fresh window list for the 60 Hz clip
        self.log = None
        if EngineLog is not None:
            try:
                self.log = EngineLog()
            except Exception:  # noqa: BLE001 - logging is optional
                self.log = None
        self._log_t = 0.0
        self._fps_t: deque[float] = deque(maxlen=24)
        self._drawn_version = -1
        self._last_counts = (0, 0)
        self._alert_until = 0.0
        self._threads: list[threading.Thread] = []
        self._keys: _TerminalKeys | None = None
        self._timer = None
        self._app = None
        self.overlay_updates = 0

    # ---- shared state ----------------------------------------------------

    @property
    def idle(self) -> bool:
        return self.paused or self.mode is ProtectionMode.OFF

    def on_frame(self, frame_bgr: np.ndarray, t_ms: float, geom, windows: ScreenLayout | None = None) -> None:
        small = to_small_gray(frame_bgr)  # outside the lock
        with self.lock:
            if self.idle:
                return
            seq = self.core.on_frame(frame_bgr, t_ms, small, windows)
            self.geom = geom
            self.last_frame_mono = time.monotonic()  # edge.redacted_capture: snapshot freshness
            self.capture_ok = True
            self.capture_reason = ""
            self._fps_t.append(time.perf_counter())
        self.slot.put(seq, t_ms, frame_bgr, windows)

    def report_capture_error(self, exc: BaseException) -> None:
        """A capture-loop failure: shown in ``status()`` / on the island, and
        the loop keeps going."""
        msg = f"capture loop: {type(exc).__name__}: {exc}"[:200]
        self.capture_error = msg
        if self.bridge is not None and msg != self._capture_error_sent:
            self._capture_error_sent = msg
            try:
                self.bridge.send_error(msg)
            except Exception:  # noqa: BLE001
                pass

    def log_tick(self, event: str | None = None) -> None:
        """One engine-log snapshot per capture tick (``EngineLog`` keeps ~1 Hz;
        snapshots with ``error`` / ``event`` bypass that). Masks, counts,
        warning, errors; no pixels, no titles. No-op without ``edge.engine_log``."""
        if self.log is None:
            return
        with self.lock:
            masks = self.core.visible()
            shape = self.core.frame_shape
            counts = dict(self.core.last_counts)
            seq = self.core.seq
            warning = self.core.warning
        area = float(shape[0] * shape[1]) if shape else 1.0
        snap = {
            "mode": self.mode.value,
            "frame": seq,
            "regions": int(counts.get("n_regions", 0)),
            "classified": int(counts.get("n_classified", 0)),
            "second_look": int(counts.get("n_second_look", 0)),
            "masks": [
                {
                    "xyxy": [round(float(v), 1) for v in m.xyxy],
                    "reason": m.reason,
                    "verdict": m.verdict.value,
                    "window_id": m.window_id,
                    "screen_frac": round((m.xyxy[2] - m.xyxy[0]) * (m.xyxy[3] - m.xyxy[1]) / area, 4),
                }
                for m in masks
            ],
            "warning": warning,
        }
        errors = {k: v for k, v in (("analyzer", self.analyzer_error), ("capture", self.capture_error)) if v}
        err_key = tuple(sorted(errors.items()))
        if errors and err_key != getattr(self, "_logged_err", None):
            snap["error"] = errors  # new failure: bypasses the log's rate limit
        self._logged_err = err_key if errors else None
        if event:
            snap["event"] = event
        try:
            self.log.record(snap)
        except Exception:  # noqa: BLE001 - never let logging break capture
            pass

    def capture_lost(self, reason: str) -> None:
        with self.lock:
            first = self.capture_ok
            self.capture_ok = False
            self.capture_reason = reason
        if first and self.bridge is not None:
            self.bridge.send_error(reason.splitlines()[0][:200])

    def fps(self) -> float:
        ts = list(self._fps_t)
        if len(ts) < 2 or ts[-1] - ts[0] <= 0:
            return 0.0
        if time.perf_counter() - ts[-1] > 2.0:
            return 0.0
        return (len(ts) - 1) / (ts[-1] - ts[0])

    # ---- island / guardian controls (any thread) -------------------------

    def set_mode(self, mode: ProtectionMode) -> None:
        """The parent's mode (island, PIN-checked). The effective mode starts
        there; the cloud agent may raise it for an app, never lower it."""
        self.pin_mode = mode
        self._switch_mode(mode)
        if self.cloud is not None:
            try:
                app = self.cloud.front_app()
                if app:
                    self.cloud.refresh_mode(app)
            except Exception:  # noqa: BLE001
                pass

    def apply_effective_mode(self, mode: ProtectionMode) -> None:
        """Agent policy for the frontmost app (``AgentClient.effective_mode``,
        which only ever raises protection above ``pin_mode``)."""
        from agent.runtime.steward import MODE_RANK

        if MODE_RANK.get(mode.value, 0) < MODE_RANK.get(self.pin_mode.value, 0):
            mode = self.pin_mode
        if mode is not self.mode:
            self._switch_mode(mode)

    def _switch_mode(self, mode: ProtectionMode) -> None:
        with self.lock:
            self.mode = mode
            self.mode_epoch += 1
            self._pending_mode = mode
            self.core.set_prefrost_policy(*_prefrost_policy(mode))
            self._mode_event = f"mode:{mode.value}"
            if mode is ProtectionMode.OFF:
                self.core.reset()
            else:
                # Keep covering until the new mode's first result reconciles.
                self.core.masks.mark_stale(self.core.seq + 1)
                self.core.urgent = True
        self.slot.wake()

    def apply_pending_mode(self) -> None:
        """Vision thread only: the analyzer is thread-confined."""
        with self.lock:
            mode, self._pending_mode = self._pending_mode, None
        if mode is not None:
            try:
                self.analyzer.set_mode(mode)
            except Exception as exc:
                self.analyzer_error = f"set_mode failed: {exc}"

    def pause(self) -> None:
        with self.lock:
            self.paused = True
            self.core.reset()

    def resume(self) -> None:
        with self.lock:
            self.paused = False
            self.core.reset()

    def request_stop(self, reason: str = "") -> None:
        if not self.stop_event.is_set():
            self.stop_reason = reason
        self.stop_event.set()
        self.slot.wake()  # the 60 Hz timer notices and stops NSApp

    # ---- status ----------------------------------------------------------

    def status(self) -> dict:
        from edge.island_bridge import mood_for

        with self.lock:
            masks = self.core.visible()
            stale = self.core.masks.stale_count()
            started = self.core.results > 0
            capture_ok = self.capture_ok
            warning = self.core.warning
        # A mask cut into pieces by a window in front is still one mask.
        base: dict[str, MaskSpec] = {}
        for m in masks:
            base.setdefault(str(m.region_id).split("~")[0], m)
        masks = list(base.values())
        unsafe = sum(1 for m in masks if m.verdict is VerdictState.UNSAFE)
        unsure = sum(1 for m in masks if m.verdict is VerdictState.UNSURE)
        pending = sum(1 for m in masks if m.verdict is VerdictState.PENDING)
        now = time.monotonic()
        total = len(masks)
        if self.bridge is not None:
            prev_total, prev_unsafe = self._last_counts
            if total > prev_total:
                self.bridge.send_event("covered", count=total)
            elif total == 0 and prev_total > 0:
                self.bridge.send_event("cleared")
        if unsafe > self._last_counts[1]:
            self._alert_until = now + ALERT_HOLD_S
        self._last_counts = (total, unsafe)
        status, mood = mood_for(
            capture_ok=capture_ok,
            mode=self.mode,
            paused=self.paused,
            started=started,
            stale=stale,
            unsafe=unsafe,
            unsure=unsure,
            pending=pending,
            alert=now < self._alert_until,
        )
        reason = ""
        for want in (VerdictState.UNSAFE, VerdictState.UNSURE, VerdictState.PENDING):
            hit = next((m for m in masks if m.verdict is want and m.reason), None)
            if hit is not None:
                reason = hit.reason
                break
        if not capture_ok:
            reason = "CAPTURE_UNAVAILABLE"
        if self.analyzer is None and self.mode is not ProtectionMode.OFF:
            status, mood, reason = "NOT_PROTECTING", "lost", "ANALYZER_UNAVAILABLE"
        error = self.capture_error or self.analyzer_error or getattr(self.analyzer, "detector_error", None)
        return {
            "status": status,
            "mood": mood,
            "mode": self.mode.value,
            "masks": total,
            "pending": pending,
            "backend": self.backend() if self.analyzer is not None else "none",
            "fps": round(self.fps(), 1),
            "reason": reason,
            "warning": dict(warning) if warning and not self.idle else None,
            "error": error[:200] if error else None,
            "analyzer_error": self.analyzer_error,
        }

    def backend(self) -> str:
        """What judges pictures: the local detector backend, "+COOL" when the
        cloud lane is up, "COOL offline" when its breaker is open."""
        local = "local"
        if self.cloud is not None:
            return self.cloud.backend(local)
        return local

    # ---- main-thread tick -----------------------------------------------

    def tick(self) -> None:
        """Overlay + island refresh. Called at ~60 Hz on the main thread.

        Every tick re-clips the masks to the **current** window list (a
        window dragged on top cuts a mask within one tick; a dragged window
        carries its own masks), pads them for follow error first so the pad
        never crosses into a front window, the menu bar or the Dock."""
        if self.overlay is not None and getattr(self.overlay, "available", False):
            with self.lock:
                specs = [] if self.idle else self.core.specs()
                frame_layout = self.core.layout
                geom = self.geom
            layout_now = frame_layout
            if specs and self.window_watch is not None and geom is not None and frame_layout is not None:
                try:
                    fresh = self.window_watch.poll(geom)
                except Exception:  # noqa: BLE001 - fall back to the frame's list
                    fresh = None
                if fresh is not None and tuple(fresh.frame_wh) == tuple(frame_layout.frame_wh):
                    layout_now = fresh
            pad_px = float(getattr(self.overlay, "pad", 0.0)) * (geom.scale if geom is not None else 1.0)
            with self.lock:
                drawn = self.core.draw(specs, layout_now, frame_layout=frame_layout, pad_px=pad_px) if specs else []
            if self.overlay.update(drawn, geom, pad_pt=0.0):
                self.overlay_updates += 1
            ids = self.overlay.window_ids()
            if ids and hasattr(self.capture, "exclude_window_ids"):
                self.capture.exclude_window_ids = ids
            if ids and self.window_watch is not None:
                self.window_watch.exclude_window_ids = ids
        if self.cloud is not None:
            try:
                self.cloud.tick()
            except Exception:  # noqa: BLE001
                pass
        if self.bridge is not None:
            self.bridge.publish_state(self.status())

    # ---- lifecycle -------------------------------------------------------

    def start_workers(self) -> None:
        if self.analyzer is None:
            self._threads = []  # nothing to protect with: no capture, no masks; status says so
            return
        self._threads = [CaptureWorker(self), VisionWorker(self)]
        for t in self._threads:
            t.start()

    def join_workers(self, timeout: float = 3.0) -> None:
        self.request_stop(self.stop_reason or "shutdown")
        deadline = time.monotonic() + timeout
        for t in self._threads:
            t.join(max(0.05, deadline - time.monotonic()))

    def _stop_app(self) -> None:
        app = self._app
        if app is None:
            return
        try:
            from AppKit import NSEvent, NSEventTypeApplicationDefined

            app.stop_(None)
            event = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                NSEventTypeApplicationDefined, (0.0, 0.0), 0, 0.0, 0, None, 0, 0, 0
            )
            app.postEvent_atStart_(event, True)
        except Exception:
            pass

    def _run_appkit(self) -> bool:
        try:
            from AppKit import NSApplication
            from Foundation import NSRunLoop, NSRunLoopCommonModes, NSTimer
        except Exception:
            return False
        app = NSApplication.sharedApplication()
        self._app = app

        def _on_timer(_timer) -> None:
            try:
                if self.stop_event.is_set():
                    self._stop_app()
                    return
                self.tick()
            except Exception as exc:  # never let a tick kill the run loop
                print(f"overlay tick failed: {exc}", file=sys.stderr)

        timer = NSTimer.timerWithTimeInterval_repeats_block_(1.0 / OVERLAY_HZ, True, _on_timer)
        NSRunLoop.currentRunLoop().addTimer_forMode_(timer, NSRunLoopCommonModes)
        self._timer = timer
        try:
            app.finishLaunching()
        except Exception:
            pass
        while not self.stop_event.is_set():
            app.run()  # returns after _stop_app
        try:
            timer.invalidate()
        except Exception:
            pass
        self._app = None
        return True

    def run(self) -> str:
        """Block until quit. Returns the stop reason."""
        old_int = None
        if threading.current_thread() is threading.main_thread():
            old_int = signal.signal(signal.SIGINT, lambda *_a: self.request_stop("ctrl-c"))
        self._keys = _TerminalKeys(self.request_stop)
        self._keys.start()
        if self.bridge is not None:
            self.bridge.on_quit = lambda: self.request_stop("island")
        self.start_workers()
        try:
            ran = False
            if self.overlay is not None and getattr(self.overlay, "available", False):
                ran = self._run_appkit()
            if not ran:
                while not self.stop_event.wait(1.0 / 30.0):
                    self.tick()
        finally:
            self.join_workers()
            if self._keys is not None:
                self._keys.close()
            if old_int is not None:
                signal.signal(signal.SIGINT, old_int)
            if self.overlay is not None:
                try:
                    self.overlay.close()
                except Exception:
                    pass
        return self.stop_reason or "stopped"


def _start_island(sock_path) -> Any:
    try:
        from edge.island_launcher import start_island
    except Exception:
        return None
    try:
        params = inspect.signature(start_island).parameters
        kwargs = {}
        for name in ("sock_path", "socket_path"):
            if name in params:
                kwargs[name] = str(sock_path)
        return start_island(**kwargs) or True
    except Exception as exc:
        print(f"island launch failed: {exc}; running without the island", file=sys.stderr)
        return None


def _stop_island(handle: Any) -> None:
    if handle is None:
        return
    try:
        from edge.island_launcher import stop_island

        params = inspect.signature(stop_island).parameters
        if params and handle is not True:
            stop_island(handle)
        else:
            stop_island()
    except Exception:
        pass


def run_live(
    *,
    mode: str | None = None,
    fps: float = TARGET_FPS,
    overlay: bool = True,
    island: bool = True,
    capture=None,
    analyzer=None,
    max_seconds: float | None = None,
    cloud=None,
    chat: bool = True,
) -> dict:
    """``python -m edge.app --source live`` entry point. ``cloud``: inject a
    ``CloudGlue`` (tests); else built from the cloud config, if any.
    ``chat``: start the kids chat service (``SG_CHAT_SERVICE=0`` also disables)."""
    from edge.capture import PERMISSION_HELP, LiveCapture
    from edge.island_bridge import GuardianStore, IslandBridge
    from edge.screen_overlay import MaskOverlay

    store = GuardianStore()
    try:
        resolved = store.resolve_mode(mode)
    except OSError as exc:
        print(f"guardian.json not writable ({exc}); mode not persisted", file=sys.stderr)
        resolved = ProtectionMode(mode) if mode else ProtectionMode.AGGRESSIVE
    cap = capture if capture is not None else LiveCapture(target_fps=fps)
    if not getattr(cap, "available", False):
        print(getattr(cap, "reason", None) or PERMISSION_HELP, file=sys.stderr)
        try:
            cap.release()
        except Exception:
            pass
        return {"frames": 0, "status": "NOT_PROTECTING", "mood": "lost", "error": "capture unavailable", "mode": resolved.value}
    kind = "injected"
    load_error = ""
    if analyzer is None:
        try:
            analyzer, kind = load_analyzer(resolved)
        except AnalyzerUnavailable as exc:
            load_error = f"analyzer unavailable: {exc}"
            analyzer, kind = None, "none"
            print(f"ERROR: NOT PROTECTING. {load_error}", file=sys.stderr, flush=True)
    mask_overlay = None
    if overlay:
        mask_overlay = MaskOverlay()
        if mask_overlay.error:
            print(mask_overlay.error, file=sys.stderr)
    watch = None
    if mask_overlay is not None and mask_overlay.available:
        try:
            from edge.capture import WindowWatcher

            watch = WindowWatcher()
        except Exception:  # noqa: BLE001 - masks are then clipped to the frame's window list
            watch = None
    bridge = IslandBridge(guardian=store, hello={"mode": resolved.value, "opencv": cv2.__version__})
    engine = LiveEngine(capture=cap, analyzer=analyzer, mode=resolved, overlay=mask_overlay, bridge=bridge, analyzer_kind=kind)
    engine.window_watch = watch
    if analyzer is None:
        engine.analyzer_error = load_error
        if engine.log is not None:
            try:
                engine.log.record({"mode": resolved.value, "error": {"analyzer": load_error}, "event": "not_protecting"})
            except Exception:  # noqa: BLE001
                pass
    engine.cloud = cloud
    if cloud is None:
        try:
            from edge.cloud_glue import CloudGlue

            engine.cloud = CloudGlue.from_config(engine)  # None without cloud config: all local, as before
        except Exception as exc:  # noqa: BLE001
            print(f"cloud lane off: {exc}", file=sys.stderr)
            engine.cloud = None
    if engine.cloud is not None:
        engine.cloud.engine = engine
        engine.cloud.attach(analyzer)
    bridge.on_set_mode = engine.set_mode
    bridge.on_pause = engine.pause
    bridge.on_resume = engine.resume
    if not bridge.start():
        print(f"{bridge.error}; running without the island", file=sys.stderr)
        engine.bridge = None
    island_handle = _start_island(bridge.sock_path) if (island and engine.bridge is not None) else None
    if analyzer is None and engine.bridge is not None:
        engine.bridge.send_error(f"Not protecting: {load_error}"[:200])
    redact = None
    try:
        from edge.redacted_capture import RedactedCaptureServer

        redact = RedactedCaptureServer(engine)
        redact.start()
    except Exception as exc:  # noqa: BLE001 - explain-my-screen stays unavailable (fail closed)
        print(f"redacted snapshot socket off: {exc}", file=sys.stderr)
        redact = None
    chat_proc = None
    if chat:
        try:
            from kids_chat.launcher import start_chat_service

            chat_proc = start_chat_service()  # None if SG_CHAT_SERVICE=0 or already running
        except Exception as exc:  # noqa: BLE001
            print(f"kids chat service not started: {exc}", file=sys.stderr)
    if max_seconds is not None:
        threading.Timer(max_seconds, lambda: engine.request_stop("max-seconds")).start()
    print(
        f"Screen Guardian live engine: mode={resolved.value} analyzer={kind} "
        f"overlay={'on' if mask_overlay is not None and mask_overlay.available else 'off'} "
        f"island={'socket ' + str(bridge.sock_path) if engine.bridge is not None else 'off'}. "
        "q / Esc / Ctrl-C quits.",
        flush=True,
    )
    try:
        reason = engine.run()
    finally:
        _stop_island(island_handle)
        if redact is not None:
            try:
                redact.stop()
            except Exception:  # noqa: BLE001
                pass
        if chat_proc is not None:
            try:
                from kids_chat.launcher import stop_chat_service

                stop_chat_service(chat_proc)
            except Exception:  # noqa: BLE001
                pass
        if engine.cloud is not None:
            try:
                engine.cloud.close()  # session summary on quit
            except Exception:  # noqa: BLE001
                pass
        if engine.bridge is not None:
            engine.bridge.stop()
        try:
            cap.release()
        except Exception:
            pass
    return {
        "frames": engine.core.seq + 1,
        "results": engine.core.results,
        "status": engine.status()["status"],
        "mode": engine.mode.value,
        "stopped": reason,
        "follow_ms_max": round(engine.core.follow_ms_max, 2),
        "analyze_ms": round(engine.analyze_ms, 2),
        "opencv": cv2.__version__,
    }
