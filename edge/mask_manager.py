"""Mask persistence for the live engine.

Rules (fail closed):

- Masks are never wiped on a scene change. A large change marks them
  **stale**. A stale mask stays on screen until an analyzer result for a
  frame at or after that change arrives; only then is it reconciled.
- An analyzer result covers the whole frame it analyzed. Masks it returns
  are kept (moved to where that content is *now*); masks it does not return
  are dropped (region gone, or SAFE), unless they are stale and the result
  predates the change.
- Identity is stable by ``region_id`` so the overlay moves a view instead
  of recreating it. A result mask with a new id that lands on an existing
  mask (IoU ≥ 0.5 where that mask was at the analyzed frame) inherits the
  existing display key.
- Masks that scroll fully off screen are dropped.
- Every mask belongs to one app window (``window_id``; None without a
  window list). A large change in one window stales only that window's
  masks; a dragged window moves only its own masks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from vision_core.live.contracts import (
    AnalyzeResult,
    Box,
    MaskSpec,
    MaskStyle,
    VerdictState,
)

HISTORY_FRAMES = 96  # per-mask positions kept so late results can be re-based
ADOPT_IOU = 0.5
OFFSCREEN_MIN_PX = 6.0
OFFSCREEN_MIN_SHARE = 0.02


def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def shift_box(box: Box, dx: float, dy: float) -> Box:
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


EDGE_TOL = 8.0  # capture px (2 px at 1/4)


def stick_to_edges(before: Box, after: Box, width: float | None, height: float | None) -> Box:
    """A mask clipped by a screen edge keeps touching that edge while the
    content moves inward, so a picture scrolling *into* view stays covered
    (fail closed) until the analyzer sees its full extent."""
    if width is None or height is None:
        return after
    x1, y1, x2, y2 = after
    if before[1] <= EDGE_TOL and y1 > before[1]:
        y1 = min(y1, 0.0) if before[1] <= 0 else before[1]
    if before[3] >= height - EDGE_TOL and y2 < before[3]:
        y2 = max(y2, before[3])
    if before[0] <= EDGE_TOL and x1 > before[0]:
        x1 = min(x1, 0.0) if before[0] <= 0 else before[0]
    if before[2] >= width - EDGE_TOL and x2 < before[2]:
        x2 = max(x2, before[2])
    return (x1, y1, x2, y2)


def clip_box(box: Box, width: float, height: float) -> Box:
    return (
        max(0.0, min(width, box[0])),
        max(0.0, min(height, box[1])),
        max(0.0, min(width, box[2])),
        max(0.0, min(height, box[3])),
    )


@dataclass
class Track:
    key: str  # display identity, stable for the life of the mask
    region_id: str  # analyzer id (may be re-assigned on adoption)
    box: Box  # capture px of the *current* frame, unclipped
    style: MaskStyle = MaskStyle.FROST
    reason: str = ""
    verdict: VerdictState = VerdictState.UNSAFE
    born_seq: int = 0
    confirmed_seq: int = 0  # frame seq of the last analyzer result that kept it
    stale_since: int | None = None  # seq of the large change that staled it
    weak_frames: int = 0  # consecutive frames without a confident local match
    vel: tuple[float, float] = (0.0, 0.0)  # last frame's motion (capture px)
    travel: tuple[float, float] = (0.0, 0.0)  # summed motion since birth
    prefrost: bool = False  # added by the fast path, not (yet) by the analyzer
    window_id: int | None = None  # owning app window (None = no window list)
    gone_since_ms: float | None = None  # its window left the window list at this time
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_FRAMES))

    def record(self, seq: int) -> None:
        entry = (seq, self.box, self.travel)
        if self.history and self.history[-1][0] == seq:
            self.history[-1] = entry
        else:
            self.history.append(entry)

    def box_at(self, seq: int) -> Box | None:
        """Where this mask was on frame ``seq`` (None if unknown)."""
        for s, b, _t in reversed(self.history):
            if s <= seq:
                return b
        return None

    def travel_since(self, seq: int) -> tuple[float, float] | None:
        """Content motion since frame ``seq`` (growth / clipping excluded)."""
        for s, _b, t in reversed(self.history):
            if s <= seq:
                return (self.travel[0] - t[0], self.travel[1] - t[1])
        return None

    def spec(self, width: float, height: float, pad: float = 0.0) -> MaskSpec:
        box = self.box
        if pad > 0 and self.history:
            # Moved since the analyzer last placed it: sub-pixel follow error
            # accumulates, so cover a little more (fail closed).
            last = self.travel_since(self.confirmed_seq) if self.confirmed_seq >= 0 else None
            if last is None:
                last = self.travel
            if abs(last[0]) + abs(last[1]) >= pad:
                box = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
        return MaskSpec(
            region_id=self.key,
            xyxy=clip_box(box, width, height),
            style=self.style,
            reason=self.reason,
            verdict=self.verdict,
            window_id=self.window_id,
        )


# place(mask, analyzed_seq) → (box in current frame, confident)
PlaceFn = Callable[[MaskSpec, int], tuple[Box, bool]]


class MaskManager:
    def __init__(self) -> None:
        self.tracks: dict[str, Track] = {}
        self.frame_size: tuple[float, float] | None = None  # (w, h) of the current frame
        # Grow edge-clipped masks as content scrolls in. Off when the fast
        # path pre-frosts (it extends masks only where picture pixels appear).
        self.stick_edges = True
        self.last_large_change: int | None = None
        self.last_large_by_window: dict[int, int] = {}  # per-window large changes (window list known)
        self.dropped_offscreen = 0
        self.dropped_by_result = 0
        self.prefrost_added = 0
        self._prefrost_n = 0

    def __len__(self) -> int:
        return len(self.tracks)

    def clear(self) -> None:
        self.tracks.clear()

    def _by_region(self) -> dict[str, Track]:
        return {t.region_id: t for t in self.tracks.values()}

    def mark_stale(self, seq: int, window_id: int | None = None) -> None:
        """Large change on frame ``seq`` (in one window when ``window_id`` is
        given, else the whole screen). Keep masks, but stop trusting them.
        A mask that is already stale keeps its original change seq so a
        stream of changes (video) cannot hold it forever."""
        if window_id is None:
            self.last_large_change = seq
        else:
            self.last_large_by_window[window_id] = seq
        for track in self.tracks.values():
            if window_id is not None and track.window_id != window_id:
                continue
            if track.stale_since is None:
                track.stale_since = seq

    def large_change_for(self, window_id: int | None) -> int | None:
        """Latest large change that concerns a mask of this window."""
        a = self.last_large_change
        b = self.last_large_by_window.get(window_id) if window_id is not None else None
        if a is None:
            return b
        return a if b is None else max(a, b)

    def move_window(self, window_id: int, dx: float, dy: float) -> int:
        """A window was dragged: its masks move with it, exactly."""
        n = 0
        for track in self.tracks.values():
            if track.window_id == window_id and (dx or dy):
                track.box = shift_box(track.box, dx, dy)
                track.travel = (track.travel[0] + dx, track.travel[1] + dy)
                track.vel = (dx, dy)
                n += 1
        return n

    def stale_count(self) -> int:
        return sum(1 for t in self.tracks.values() if t.stale_since is not None)

    def add_prefrost(self, box: Box, seq: int, window_id: int | None = None) -> Track:
        """Fast-path PENDING frost on a picture-like box nobody covers yet.
        It lives until an analyzer result for a frame ≥ ``seq`` drops it."""
        self._prefrost_n += 1
        key = f"pf{self._prefrost_n}"
        track = Track(
            key=key,
            region_id=key,
            box=box,
            style=MaskStyle.FROST,
            reason="PENDING_PREFROST",
            verdict=VerdictState.PENDING,
            born_seq=seq,
            confirmed_seq=-1,
            prefrost=True,
            window_id=window_id,
        )
        track.record(seq)
        self.tracks[key] = track
        self.prefrost_added += 1
        return track

    def _stick(self, before: Box, after: Box) -> Box:
        if self.frame_size is None or not self.stick_edges:
            return after
        return stick_to_edges(before, after, *self.frame_size)

    def move(self, key: str, dx: float, dy: float) -> None:
        track = self.tracks.get(key)
        if track is not None and (dx or dy):
            track.box = self._stick(track.box, shift_box(track.box, dx, dy))
            track.travel = (track.travel[0] + dx, track.travel[1] + dy)

    def record_all(self, seq: int) -> None:
        for track in self.tracks.values():
            track.record(seq)

    def drop_offscreen(self, width: float, height: float) -> list[str]:
        gone: list[str] = []
        for key, track in self.tracks.items():
            x1, y1, x2, y2 = clip_box(track.box, width, height)
            vis_w, vis_h = x2 - x1, y2 - y1
            area = max(1.0, (track.box[2] - track.box[0]) * (track.box[3] - track.box[1]))
            if (
                vis_w < OFFSCREEN_MIN_PX
                or vis_h < OFFSCREEN_MIN_PX
                or (vis_w * vis_h) / area < OFFSCREEN_MIN_SHARE
            ):
                gone.append(key)
        for key in gone:
            del self.tracks[key]
        self.dropped_offscreen += len(gone)
        return gone

    def reconcile(
        self,
        result: AnalyzeResult,
        seq: int,
        cur_seq: int,
        place: PlaceFn,
    ) -> set[str]:
        """Apply an analyzer result for frame ``seq`` while the engine is at
        ``cur_seq``. Returns keys whose position is uncertain (re-analyze)."""
        uncertain: set[str] = set()
        by_region = self._by_region()
        claimed: set[str] = set()
        for mask in result.masks:
            wid = getattr(mask, "window_id", None)
            large = self.large_change_for(wid)
            pre_change = large is not None and seq < large
            track = by_region.get(mask.region_id)
            if track is not None and track.key in claimed:
                track = None
            if track is None:
                track = self._adopt(mask.xyxy, seq, claimed, wid)
            moved = None if track is None else track.travel_since(seq)
            if track is not None and moved is not None:
                # Same content: re-base the analyzer box by how far this
                # mask's content has moved since the analyzed frame.
                xyxy = tuple(float(v) for v in mask.xyxy)
                box = self._stick(xyxy, shift_box(xyxy, moved[0], moved[1]))
                confident = True
            else:
                xyxy = tuple(float(v) for v in mask.xyxy)
                box, confident = place(mask, seq)
                box = self._stick(xyxy, box)
                if not confident and pre_change:
                    # Content analyzed before a navigation and not found now.
                    continue
            if track is None:
                key = mask.region_id
                n = 1
                while key in self.tracks:
                    n += 1
                    key = f"{mask.region_id}#{n}"
                track = Track(key=key, region_id=mask.region_id, box=box, born_seq=cur_seq)
                self.tracks[key] = track
            track.region_id = mask.region_id
            track.box = box
            track.style = mask.style
            track.reason = mask.reason
            track.verdict = mask.verdict
            track.confirmed_seq = seq
            track.weak_frames = 0
            track.prefrost = False
            track.window_id = wid
            track.gone_since_ms = None
            track.stale_since = large if pre_change else None
            track.record(cur_seq)
            claimed.add(track.key)
            if not confident:
                uncertain.add(track.key)
        drop: list[str] = []
        for key, track in self.tracks.items():
            if key in claimed:
                continue
            if track.born_seq > seq:
                continue  # newer than this result
            if track.stale_since is not None and seq < track.stale_since:
                continue  # result predates the change; keep covering
            drop.append(key)
        for key in drop:
            del self.tracks[key]
        self.dropped_by_result += len(drop)
        return uncertain

    def _adopt(self, xyxy: Box, seq: int, claimed: set[str], window_id: int | None = None) -> Track | None:
        best: Track | None = None
        best_iou = ADOPT_IOU
        for track in self.tracks.values():
            if track.key in claimed:
                continue
            if window_id is not None and track.window_id is not None and track.window_id != window_id:
                continue
            then = track.box_at(seq)
            if then is None:
                continue
            iou = box_iou(then, xyxy)
            if iou >= best_iou:
                best, best_iou = track, iou
        return best

    def specs(self, width: float, height: float, pad: float = 0.0) -> list[MaskSpec]:
        return [t.spec(width, height, pad) for t in sorted(self.tracks.values(), key=lambda t: t.key)]
