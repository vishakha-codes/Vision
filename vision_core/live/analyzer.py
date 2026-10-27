"""``LiveAnalyzer`` implementation: regions → cache → cv2.dnn → mode → masks.

One call of :meth:`Analyzer.analyze`:

1. **Propose** picture regions (only inside ``changed`` when given; regions
   remembered from earlier calls cover the rest of the frame).
2. **Identity**: a proposal keeps the id of a remembered region with the same
   perceptual hash (scroll / revisit) or, failing that, the same rect (IoU ≥
   0.9, content changed in place → after two changes it is a ``video``).
3. **Cache**: evidence keyed by dHash + aHash + size bucket. A hit costs a
   hash compare; the verdict is re-derived for the current mode and rect.
4. **Classify** cache misses in batches of letterboxed crops (whole picture,
   plus 2× tiles when the picture is large) until ``budget_ms`` is spent;
   the rest stay PENDING (AGGRESSIVE pre-frosts them) for the next call.
5. **Second look** for UNSURE pictures: margin crop + 2× tiles, rate-limited
   by a token bucket (~4/s) instead of a lifetime budget.
6. **Masks** for every region on the frame, per mode, each capped at its
   window and tagged with its ``window_id``.

**Windows** (``analyze(..., windows=ScreenLayout)``): proposals only look
inside app windows (``vision_core.live.windows.label_map``), static toolbar
bands are learned per window (and only from frames where that window's
content scrolled), and large regions get window-scoped, short-lived cache
entries, so wallpaper and layout can never bring back an old verdict.

**Cloud lane** (``cloud_lane=``, optional, ``vision_core.live.cloud_lane``):
answers that came back since the last call are merged into the cache first;
pictures the local model could not settle are submitted after the verdict
step (UNSURE after the second look or with an empty second-look bucket;
large SAFE pictures with a weak score; every job when the local detector
fails). ``submit`` never blocks. Without a lane nothing changes.

Thread-confined: one vision worker owns an instance; no locking.
"""

from __future__ import annotations

import inspect
import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from vision_core.live.budget import TokenBucket
from vision_core.live.cache import VerdictCache, hamming, size_bucket
from vision_core.live.contracts import (
    AnalyzeResult,
    Box,
    MaskSpec,
    PictureRegion,
    ProtectionMode,
    ScreenLayout,
    Verdict,
    VerdictState,
)
from vision_core.live.detector import NudeNetDetector, PartDetector, RawDet
from vision_core.live.modes import (
    REASON_CLIPPED,
    Evidence,
    ModePolicy,
    NormDet,
    decide,
    load_mode_policy,
    masks_for,
    needs_second_look,
)
from vision_core.live.regions import (
    KIND_TILE,
    KIND_VIDEO,
    RegionOptions,
    RegionProposer,
    WindowView,
    _iou,
    downscale_int,
    tile_grid,
    work_factor,
)
from vision_core.live.windows import HeavyTracker, geometry, inter, label_map, prefrost_settings, window_motion

_ID_HAMMING = 6
_SAME_RECT_IOU = 0.9
CLOUD_LARGE_PX = 900.0  # long side of a SAFE picture worth a cloud look when its score is weak
CLOUD_LARGE_MIN_SCORE = 0.08
_VIDEO_CHANGES = 2
_ROI_MARGIN_PX = 24.0
BIG_REGION_FRAC = 0.20  # regions over this share of the screen: window-scoped, short-lived cache entries
BIG_REGION_TTL_MS = 3000.0
BAND_HISTORY = 5  # recent static-band estimates kept per window (debugging)
BAND_SCROLL_ROWS = 0.30  # a scroll changes at least this share of a window's rows
SHADOW_SIDE, SHADOW_TOP, SHADOW_BOTTOM = 0.015, 0.008, 0.035  # drop-shadow reach (× frame width)


@dataclass
class _Known:
    region: PictureRegion
    first_seen_ms: float
    changes: int = 0
    carry: Evidence | None = None  # last evidence before an in-place content change
    # Scroll tracking: the unclipped rect (current coords) and the evidence
    # from a view where the whole picture was visible.
    full_xyxy: Box | None = None
    full_ev: Evidence | None = None
    static: int = 0  # consecutive calls with the same rect and content


@dataclass
class _Band:
    """Static toolbar / status bands of one window (or the whole frame)."""

    bounds: tuple[float, float, float, float]
    top: float | None = None
    bottom: float | None = None
    tops: deque = field(default_factory=lambda: deque(maxlen=BAND_HISTORY))
    bottoms: deque = field(default_factory=lambda: deque(maxlen=BAND_HISTORY))
    line: float | None = None  # toolbar border found at creation


def incomplete(reg: PictureRegion) -> bool:
    """Cut by the frame / a pane edge along the scroll axis."""
    return "t" in reg.clipped or "b" in reg.clipped


def _toolbar_line(frame: np.ndarray, max_frac: float = 0.25) -> float | None:
    """y just below a thin full-width line in the top quarter (a toolbar's
    bottom border) that differs from the rows on both sides; else None."""
    H = frame.shape[0]
    g = cv2.cvtColor(np.ascontiguousarray(frame[: int(H * max_frac), ::4]), cv2.COLOR_BGR2GRAY).astype(np.int16)
    if g.shape[0] < 5:
        return None
    mean = g.mean(axis=1)
    std = g.std(axis=1)
    for y in range(g.shape[0] - 3, 1, -1):
        if std[y] > 3.0:
            continue
        above = y - 1
        below = y + 1
        while below < g.shape[0] - 1 and std[below] <= 3.0 and abs(mean[below] - mean[y]) < 2.0:
            below += 1  # lines may be 2–3 px thick
        if below - y > 4:
            continue
        if std[above] <= 3.0 and abs(mean[above] - mean[y]) >= 8.0 and abs(mean[below] - mean[y]) >= 8.0:
            return float(below)
    return None


def _shift(b: Box, dx: float, dy: float) -> Box:
    return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


@dataclass
class _Job:
    region: PictureRegion
    crops: list[Box] = field(default_factory=list)


def _first_run(rows: np.ndarray, min_len: int = 3) -> tuple[int, int] | None:
    """(first, last) busy row of the span that starts and ends with a run of
    ``min_len`` consecutive busy rows (isolated rows — a clock tick, a
    spinner — are ignored)."""
    if rows.size < min_len:
        return None
    d = np.diff(rows)
    first = last = None
    for i in range(rows.size - min_len + 1):
        if np.all(d[i : i + min_len - 1] == 1):
            first = int(rows[i])
            break
    for i in range(rows.size - 1, min_len - 2, -1):
        if np.all(d[i - min_len + 1 : i] == 1):
            last = int(rows[i])
            break
    if first is None or last is None or last < first:
        return None
    return first, last


def _evidence_unsafe(value: object) -> bool:
    """Cache predicate: reuse at any size when any box is fairly strong."""
    if not isinstance(value, Evidence):
        return False
    pool = list(value.dets) + list(value.dets2 or ())
    return any(d.score >= 0.45 for d in pool)


def _intersects(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class Analyzer:
    """See module docstring. Build with :func:`make_analyzer`."""

    def __init__(
        self,
        mode: ProtectionMode = ProtectionMode.AGGRESSIVE,
        *,
        detector: PartDetector | None = None,
        model_path: str | Path | None = None,
        backend: str = "auto",
        policy_dir: str | Path | None = None,
        budget_ms: float | None = None,
        second_look_rate_per_s: float | None = None,
        second_look_burst: float | None = None,
        cache: VerdictCache | None = None,
        region_opts: RegionOptions | None = None,
        tile_long_px: float = 800.0,
        max_batch: int = 16,
        warmup: bool = True,
        cloud_lane=None,
    ) -> None:
        self._policy_dir = policy_dir
        self._budget_override = budget_ms
        self._sl_rate = second_look_rate_per_s
        self._sl_burst = second_look_burst
        self._model_path = model_path
        self._backend = backend
        self._warmup = warmup
        self._detector = detector
        self.cloud_lane = cloud_lane  # vision_core.live.cloud_lane.CloudLane (or a test double)
        # (dhash, ahash) → what was escalated (region, window, local score, labels, reason):
        # read by the engine's cloud callback to build agent events. No pixels.
        self.escalations: OrderedDict[tuple[int, int], dict] = OrderedDict()
        self._scoped: OrderedDict[tuple[int, int], tuple] = OrderedDict()  # cloud keys of window-scoped entries
        self.detector_error: str | None = None
        self.proposer = RegionProposer(region_opts)
        self.cache = cache or VerdictCache(reuse_any_size=_evidence_unsafe)
        self.tile_long_px = float(tile_long_px)
        self.max_batch = max(1, int(max_batch))
        self._known: dict[str, _Known] = {}
        self._frame_wh: tuple[int, int] | None = None
        self._crop_ms = 5.0  # EMA of detector cost per crop
        self._prev_small: np.ndarray | None = None
        self._bands: dict[int | None, _Band] = {}
        self._layout: ScreenLayout | None = None
        self._prev_layout: ScreenLayout | None = None
        self._small: np.ndarray | None = None
        self._f = 1
        self.heavy = HeavyTracker()  # windows with several recent UNSAFE masks (adaptive pre-frost)
        self._masks_heavy_kw: bool | None = None
        self.mode = ProtectionMode(mode)
        self.policy: ModePolicy = load_mode_policy(self.mode, policy_dir)
        self.bucket = self._make_bucket()
        if self.mode is not ProtectionMode.OFF:
            self.detector  # load + warm up now, not on the first frame

    # ------------------------------------------------------------ config

    def _make_bucket(self) -> TokenBucket:
        rate = self._sl_rate if self._sl_rate is not None else self.policy.second_look_rate_per_s
        burst = self._sl_burst if self._sl_burst is not None else self.policy.second_look_burst
        return TokenBucket(rate_per_s=rate, burst=burst)

    @property
    def budget_ms(self) -> float:
        if self._budget_override is not None:
            return float(self._budget_override)
        return float(self.policy.budget_ms)

    @property
    def backend(self) -> str:
        """Detector backend actually in use (``cv2`` / ``ort``), else the one
        requested; ``LiveEngine.status`` shows it on the island."""
        det = self._detector
        got = getattr(det, "backend", "") if det is not None else ""
        return str(got or self._backend or "local")

    # Static bands of the whole frame (no window info), kept for callers / tests.
    @property
    def _clip_top(self) -> float | None:
        band = self._bands.get(None)
        return band.top if band is not None else None

    @property
    def _clip_bottom(self) -> float | None:
        band = self._bands.get(None)
        return band.bottom if band is not None else None

    @property
    def detector(self) -> PartDetector:
        if self._detector is None:
            self._detector = NudeNetDetector(self._model_path, backend=self._backend, warmup=self._warmup)
        return self._detector

    def set_mode(self, mode: ProtectionMode) -> None:
        """Switch mode. Cached evidence is mode-independent, so it is kept."""
        self.mode = ProtectionMode(mode)
        self.policy = load_mode_policy(self.mode, self._policy_dir)
        self.bucket = self._make_bucket()
        if self.mode is not ProtectionMode.OFF:
            self.detector

    def reset(self) -> None:
        """Forget remembered regions (the cache stays)."""
        self._known.clear()
        self._bands.clear()
        self.heavy.clear()

    # ------------------------------------------------------------ main

    def analyze(
        self,
        frame_bgr: np.ndarray,
        *,
        changed: Box | None = None,
        t_ms: float | None = None,
        windows: ScreenLayout | None = None,
        heavy_windows: set | None = None,
    ) -> AnalyzeResult:
        """See ``LiveAnalyzer.analyze``. ``heavy_windows`` (optional): the
        engine's heavy-window signal (windows with several recent UNSAFE
        masks); an adaptive pre-frost policy pre-frosts PENDING pictures only
        there. Without it the analyzer uses its own tracker of its masks."""
        t0 = time.perf_counter()
        # Frame 0 at t = 0.0 is a real time; only a missing t uses our clock.
        now_ms = float(t_ms) if t_ms is not None else time.monotonic() * 1000.0
        timings: dict[str, float] = {}
        counts: dict[str, int] = {}
        if self.mode is ProtectionMode.OFF or frame_bgr is None or frame_bgr.size == 0:
            timings["total"] = (time.perf_counter() - t0) * 1000.0
            return AnalyzeResult(timings_ms=timings, counts=counts)
        frame = frame_bgr if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3 else cv2.cvtColor(
            frame_bgr, cv2.COLOR_BGRA2BGR if frame_bgr.ndim == 3 else cv2.COLOR_GRAY2BGR
        )
        H, W = frame.shape[:2]
        if self._frame_wh != (W, H):
            self._known.clear()
            self._frame_wh = (W, H)
            self._prev_small = None
            self._bands.clear()
            self._prev_layout = None
        if windows is not None and tuple(windows.frame_wh) != (W, H):
            windows = None  # captured for another frame size: unusable
        self._prev_layout, self._layout = self._layout, windows
        # Work-scale copy for scroll-shift verification of clipped pictures.
        self._f = work_factor(H, W, self.proposer.opts.work_long)
        self._small = cv2.cvtColor(downscale_int(frame, self._f), cv2.COLOR_BGR2GRAY)
        self._update_static_bands(frame)
        view = self._window_view(W, H)

        # 1. proposals (inside the changed rect when we already know the rest)
        regions, kept = self._propose(frame, changed, view)
        timings["propose"] = (time.perf_counter() - t0) * 1000.0
        for key, val in self.proposer.last.timings_ms.items():
            if key != "total":
                timings[f"propose.{key}"] = val

        # 2. identity against remembered regions
        t1 = time.perf_counter()
        current = self._assign_ids(regions, kept, now_ms)
        # 3. cloud answers into the cache (vision thread), then the cache
        if self.cloud_lane is not None:
            try:
                counts["n_cloud_applied"] = len(self.cloud_lane.apply(_ScopedCacheView(self.cache, self._scoped), now_ms))
            except Exception:  # noqa: BLE001 - the cloud lane must never break a frame
                counts["n_cloud_applied"] = 0
        evidence: dict[str, Evidence] = {}
        sources: dict[str, str] = {}
        misses: list[PictureRegion] = []
        decide_rect: dict[str, Box] = {}
        min_classify = self.proposer.min_side(W)
        n_tracked = 0
        for reg in current:
            k = self._known.get(reg.region_id)
            w, h = reg.xyxy[2] - reg.xyxy[0], reg.xyxy[3] - reg.xyxy[1]
            seen_bigger = (
                k is not None
                and k.full_xyxy is not None
                and (k.full_xyxy[1] < reg.xyxy[1] - self._f or k.full_xyxy[3] > reg.xyxy[3] + self._f)
            )
            occl = k is not None and k.full_xyxy is not None and self._occluded_part(reg, k.full_xyxy)
            known_ev = None if k is None else (k.full_ev if k.full_ev is not None else (k.carry if occl else None))
            if (incomplete(reg) and seen_bigger or occl) and known_ev is not None:
                # Partly scrolled out, or partly hidden by a window in front:
                # reuse what the whole picture showed.
                evidence[reg.region_id] = known_ev
                sources[reg.region_id] = "track"
                decide_rect[reg.region_id] = k.full_xyxy
                n_tracked += 1
                continue
            if incomplete(reg) and min(w, h) < min_classify:
                continue  # a sliver: too little to classify; fail-closed per mode
            ev = self._cache_get(reg, now_ms)
            if isinstance(ev, Evidence):
                evidence[reg.region_id] = ev
                sources[reg.region_id] = "cache"
            else:
                misses.append(reg)
        counts["n_cache_hits"] = sum(1 for v in sources.values() if v == "cache")
        counts["n_tracked"] = n_tracked
        timings["cache"] = (time.perf_counter() - t1) * 1000.0

        # 4. classify misses within the time budget
        t2 = time.perf_counter()
        misses.sort(key=lambda r: self._priority(r))
        n_crops = 0
        n_classified = 0
        jobs = [self._first_pass_job(r) for r in misses]
        i = 0
        while i < len(jobs):
            batch: list[_Job] = []
            n = 0
            while i < len(jobs) and (not batch or n + len(jobs[i].crops) <= self.max_batch):
                batch.append(jobs[i])
                n += len(jobs[i].crops)
                i += 1
            elapsed = (time.perf_counter() - t0) * 1000.0
            # budget ≤ 0: never classify (pure pre-frost). Otherwise the first
            # batch always runs so a slow proposal step cannot starve progress.
            if self.budget_ms <= 0.0 or (n_classified and elapsed + n * self._crop_ms > self.budget_ms):
                break
            for job, ev in zip(batch, self._run_jobs(frame, batch, now_ms)):
                if ev is None:
                    continue  # local detector failed: left PENDING, escalated to the cloud
                evidence[job.region.region_id] = ev
                sources[job.region.region_id] = "local"
                self._cache_put(job.region, ev, now_ms)
                n_classified += 1
            n_crops += n
        timings["classify"] = (time.perf_counter() - t2) * 1000.0

        # 5. second look for UNSURE (token bucket)
        t3 = time.perf_counter()
        n_second = 0
        now_s = now_ms / 1000.0
        for reg in current:
            ev = evidence.get(reg.region_id)
            if ev is None or sources.get(reg.region_id) == "track":
                continue
            verdict = decide(self.policy, reg, ev)
            if not needs_second_look(self.policy, verdict, ev):
                continue
            job = self._second_look_job(reg, W, H)
            elapsed = (time.perf_counter() - t0) * 1000.0
            if elapsed + len(job.crops) * self._crop_ms > self.budget_ms * 1.25:
                break
            if not self.bucket.take(now_s):
                break
            second = self._run_jobs(frame, [job], now_ms)[0]
            if second is None:
                continue
            ev2 = Evidence(ev.dets, second.dets, ev.latency_ms + second.latency_ms, ev.n_crops + second.n_crops)
            evidence[reg.region_id] = ev2
            sources[reg.region_id] = "local"
            self._cache_put(reg, ev2, now_ms)
            n_second += 1
            n_crops += len(job.crops)
        timings["second_look"] = (time.perf_counter() - t3) * 1000.0

        # 6. verdicts + masks for the whole frame
        t4 = time.perf_counter()
        verdicts: dict[str, Verdict] = {}
        masks: list[MaskSpec] = []
        n_pending = 0
        n_clip_frost = 0
        pf_mode, pf_n, pf_s = prefrost_settings(self.policy, self.mode)
        heavy_set: set = set()
        if pf_mode == "adaptive":
            heavy_set = set(heavy_windows) if heavy_windows is not None else set()
            heavy_set |= self.heavy.heavy_windows(now_ms, pf_n, pf_s)
        mask_start: dict[str, int] = {}
        for reg in current:
            mask_start[reg.region_id] = len(masks)
            self._heavy_now = reg.window_id in heavy_set
            ev = evidence.get(reg.region_id)
            src = sources.get(reg.region_id, "local")
            k = self._known.get(reg.region_id)
            if ev is None:
                carry = k.carry if k is not None else None
                if carry is not None:
                    carried = decide(self.policy, reg, carry, source="carry")
                    if carried.state in (VerdictState.UNSAFE, VerdictState.UNSURE):
                        # Video / in-place change: keep the cover until re-checked.
                        verdicts[reg.region_id] = carried
                        masks.extend(self._masks_for(reg, carried, (W, H), evidence=carry))
                        n_pending += 1
                        continue
                n_pending += 1
            elif k is not None and src != "track":
                k.carry = ev
                if not incomplete(reg):
                    k.full_ev = ev
                    k.full_xyxy = reg.xyxy
            full = decide_rect.get(reg.region_id)
            basis = replace(reg, xyxy=full) if full is not None else reg
            verdict = decide(self.policy, basis, ev, source=src)
            if (
                self.policy.clipped_fail_closed
                and incomplete(reg)
                and src != "track"
                and reg.kind != KIND_TILE
                and verdict.state in (VerdictState.SAFE, VerdictState.PENDING)
                and not self._clip_release(reg, verdict, k, H)
            ):
                # Scrolling in, never seen whole: what is still off-screen is
                # unknown, so cover it until the whole picture has been checked.
                verdict = Verdict(VerdictState.UNSURE, score=verdict.score, source="clipped")
                masks.extend(self._masks_for(reg, verdict, (W, H), reason=REASON_CLIPPED))
                verdicts[reg.region_id] = verdict
                n_clip_frost += 1
                continue
            verdicts[reg.region_id] = verdict
            masks.extend(self._masks_for(reg, verdict, (W, H), evidence=ev))
            band = self.policy.clipped_edge_band
            if band > 0.0 and incomplete(reg) and src != "track" and reg.kind != KIND_TILE:
                # Parts mode: a part cut by the edge is too partial to detect.
                # Cover unclassifiable slivers, and a band along the cut edge of
                # pictures that already showed a hit.
                x1, y1, x2, y2 = reg.xyxy
                w, h = x2 - x1, y2 - y1
                sliver = min(w, h) < min_classify
                rects: list[Box] = []
                if sliver and verdict.state in (VerdictState.PENDING, VerdictState.SAFE):
                    rects.append(reg.xyxy)
                elif verdict.state in (VerdictState.UNSAFE, VerdictState.UNSURE):
                    bh = min(h, band * w)
                    if "t" in reg.clipped:
                        rects.append((x1, y1, x2, y1 + bh))
                    if "b" in reg.clipped:
                        rects.append((x1, y2 - bh, x2, y2))
                for j, rect in enumerate(rects):
                    masks.append(MaskSpec(f"{reg.region_id}#e{j}", rect, self.policy.style, REASON_CLIPPED, verdict.state))
                    n_clip_frost += 1
        self._prev_small = self._small
        if self.cloud_lane is not None:
            counts["n_cloud_submitted"] = self._escalate(frame, current, evidence, sources, verdicts, now_ms)
        masks = self._cap_to_windows(masks, current, mask_start)
        for m in masks:
            if m.verdict is VerdictState.UNSAFE:
                self.heavy.observe(m.window_id, m.region_id.split("#")[0], now_ms)
        counts["n_clipped_frost"] = n_clip_frost
        timings["decide"] = (time.perf_counter() - t4) * 1000.0
        timings["total"] = (time.perf_counter() - t0) * 1000.0
        counts.update(
            n_regions=len(current),
            n_pictures=sum(1 for r in current if r.kind != KIND_TILE),
            n_tiles=sum(1 for r in current if r.kind == KIND_TILE),
            n_classified=n_classified,
            n_pending=n_pending,
            n_second_look=n_second,
            n_crops=n_crops,
            n_masks=len(masks),
        )
        return AnalyzeResult(regions=current, verdicts=verdicts, masks=masks, timings_ms=timings, counts=counts)

    # ------------------------------------------------------------ steps

    def _propose(
        self, frame: np.ndarray, changed: Box | None, view: WindowView | None = None
    ) -> tuple[list[PictureRegion], list[PictureRegion]]:
        H, W = frame.shape[:2]
        if view is not None:
            clip_ys: list[float] = []
            content_y: tuple[float | None, float | None] = (None, None)  # per window, in the labels
        else:
            clip_ys = [y for y in (self._clip_top, self._clip_bottom) if y is not None]
            content_y = (self._clip_top, self._clip_bottom)
        if changed is None or not self._known:
            return self.proposer.propose(frame, clip_ys=clip_ys, content_y=content_y, windows=view), []
        roi = (
            max(0.0, changed[0] - _ROI_MARGIN_PX),
            max(0.0, changed[1] - _ROI_MARGIN_PX),
            min(float(W), changed[2] + _ROI_MARGIN_PX),
            min(float(H), changed[3] + _ROI_MARGIN_PX),
        )
        # Grow the roi over every remembered picture it touches (until stable),
        # so a picture is never half re-proposed.
        grown = True
        while grown:
            grown = False
            for k in self._known.values():
                b = k.region.xyxy
                if _intersects(b, roi) and not (
                    b[0] >= roi[0] and b[1] >= roi[1] and b[2] <= roi[2] and b[3] <= roi[3]
                ):
                    roi = (min(roi[0], b[0]), min(roi[1], b[1]), max(roi[2], b[2]), max(roi[3], b[3]))
                    grown = True
        kept = [k.region for k in self._known.values() if not _intersects(k.region.xyxy, roi)]
        return self.proposer.propose(frame, roi, clip_ys=clip_ys, content_y=content_y, windows=view), kept

    # ------------------------------------------------------------ windows

    def _window_rects(self, W: int, H: int) -> dict[int | None, tuple[float, float, float, float]]:
        """Rects whose static bands are learned: each app window's bounds
        (clipped to the frame), or the whole frame without window info."""
        if self._layout is None:
            return {None: (0.0, 0.0, float(W), float(H))}
        g = geometry(self._layout)
        return {w.window_id: g.bounds[w.window_id] for w in g.apps}

    def _update_static_bands(self, frame: np.ndarray) -> None:
        """Static toolbar / status bands, per window: content scrolls under
        them, so their inner edge clips pictures. Learned from where the
        frame-to-frame change starts (top) and stops (bottom) **inside that
        window**, only on frames where most of its rows changed and a
        translation explains it (a scroll; a ticking clock, a spinner or a
        dragged window's shadow is not), and forgotten when the window
        resizes (moved with it when dragged)."""
        H, W = frame.shape[:2]
        rects = self._window_rects(W, H)
        for key in [k for k in self._bands if k not in rects]:
            del self._bands[key]
        prev, cur = self._prev_small, self._small
        f = float(self._f)
        for key, b in rects.items():
            band = self._bands.get(key)
            if band is None or band.bounds != b:
                if band is not None and key is not None:
                    dx, dy = window_motion(self._prev_layout, self._layout, key)
                    if (dx or dy) and (band.bounds[2] - band.bounds[0], band.bounds[3] - band.bounds[1]) == (b[2] - b[0], b[3] - b[1]):
                        # Dragged, same size: the bands move with the window.
                        band = _Band(b, None if band.top is None else band.top + dy, None if band.bottom is None else band.bottom + dy,
                                     deque((t + dy for t in band.tops), maxlen=BAND_HISTORY),
                                     deque((t + dy for t in band.bottoms), maxlen=BAND_HISTORY),
                                     None if band.line is None else band.line + dy)
                        self._bands[key] = band
                        continue
                x1, y1, x2, y2 = (int(round(v)) for v in b)
                line = _toolbar_line(frame[y1:y2, x1:x2]) if y2 - y1 >= 20 and x2 - x1 >= 20 else None
                band = _Band(b, None if line is None else y1 + line, None, line=None if line is None else y1 + line)
                self._bands[key] = band
                continue
            if band.line is None and band.top is None:
                x1, y1, x2, y2 = (int(round(v)) for v in b)
                line = _toolbar_line(frame[y1:y2, x1:x2]) if y2 - y1 >= 20 and x2 - x1 >= 20 else None
                if line is not None:
                    band.line = band.top = y1 + line
            if prev is None or cur is None or prev.shape != cur.shape:
                continue
            if key is not None and not self._layout_steady():
                continue  # windows moved / opened / closed: their pixels changed, not this page
            sx1, sy1 = int(math.ceil(b[0] / f)), int(math.ceil(b[1] / f))
            sx2, sy2 = int(math.floor(b[2] / f)), int(math.floor(b[3] / f))
            if sx2 - sx1 < 8 or sy2 - sy1 < 8:
                continue
            pc, cc = prev[sy1:sy2, sx1:sx2], cur[sy1:sy2, sx1:sx2]
            moved = cv2.absdiff(pc, cc) > 2
            own = None
            if key is not None:
                g = geometry(self._layout)
                own = label_map(self._layout, cur.shape, (f, f))[sy1:sy2, sx1:sx2] == g.labels.get(key, -1)
                moved &= own
            rows = np.flatnonzero(np.count_nonzero(moved, axis=1) >= max(4, moved.shape[1] // 50))
            if key is None:
                # No window list: the whole frame is one page (the old rule,
                # measured on the eval web pages; the desktop case has windows).
                if rows.size < 4:
                    continue
                run = (int(rows[0]), int(rows[-1]))
            else:
                if rows.size < max(4, BAND_SCROLL_ROWS * moved.shape[0]):
                    continue  # not a scroll of this window
                if not self._scrolled(pc, cc, own):
                    continue  # content changed in place (a video, a shadow): not a scroll
                run = _first_run(rows)
                if run is None:
                    continue
            top, bot = b[1] + run[0] * f, b[1] + (run[1] + 1) * f
            h = b[3] - b[1]
            # Running min / max per window (a scroll frame where the top rows
            # happen not to change says nothing about the toolbar); reset when
            # the window moves / resizes / the analyzer resets.
            if 0 < top - b[1] <= 0.25 * h:
                band.tops.append(top)
                band.top = top if band.top is None else min(band.top, top)
            if b[1] + 0.85 * h <= bot < b[3] - f:
                band.bottoms.append(bot)
                band.bottom = bot if band.bottom is None else max(band.bottom, bot)

    def _layout_steady(self) -> bool:
        """Same app windows at the same bounds as the previous call."""
        a, b = self._prev_layout, self._layout
        if a is None or b is None:
            return False
        sa = {w.window_id: w.xyxy for w in a.windows if w.is_app}
        sb = {w.window_id: w.xyxy for w in b.windows if w.is_app}
        return sa == sb

    @staticmethod
    def _scrolled(prev: np.ndarray, cur: np.ndarray, own: np.ndarray | None) -> bool:
        """A vertical translation explains the change (phase correlation)."""
        h, w = cur.shape[:2]
        if h < 16 or w < 16:
            return False
        a, b = prev.astype(np.float32), cur.astype(np.float32)
        if own is not None and not own.all():
            if own.any():
                a[~own] = float(a[own].mean())
                b[~own] = float(b[own].mean())
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
        return bool(np.isfinite(resp) and resp >= 0.08 and abs(dy) >= 0.5 and abs(dx) <= 1.0)

    def _pane(self, window_id: int | None) -> tuple[float, float]:
        """(top, bottom) of the scrolling content of a window (or the frame)."""
        band = self._bands.get(window_id)
        W, H = self._frame_wh or (0, 0)
        if band is None:
            band = self._bands.get(None)
            if band is None:
                return 0.0, float(H)
        top = band.top if band.top is not None else band.bounds[1]
        bottom = band.bottom if band.bottom is not None else band.bounds[3]
        return float(top), float(bottom)

    def _window_view(self, W: int, H: int) -> WindowView | None:
        """Work-scale labels for the proposer: app windows only, minus each
        window's static toolbar bands."""
        if self._layout is None:
            return None
        g = geometry(self._layout)
        f = self._f
        lab = label_map(self._layout, (H // f, W // f), (float(f), float(f)))
        bounds: dict[int, Box] = {}
        panes: dict[int, Box] = {}
        rows = (np.arange(lab.shape[0]) * f).astype(np.float64)
        for win in g.apps:
            k = g.labels[win.window_id]
            b = g.bounds[win.window_id]
            bounds[k] = b
            top, bottom = self._pane(win.window_id)
            top, bottom = max(top, b[1]), min(bottom, b[3])
            vis = g.visible_bbox(win.window_id)
            if vis is None:
                continue
            pane = inter((b[0], top, b[2], bottom), g.content)
            if pane is not None:
                panes[k] = pane
            outside = (rows < top - 0.5) | (rows + f > bottom + 0.5)
            if np.any(outside):
                x1, x2 = max(0, int(b[0] // f)), min(lab.shape[1], int(math.ceil(b[2] / f)))
                sub = lab[outside, x1:x2]
                sub[sub == k] = 0
                lab[outside, x1:x2] = sub
        # Drop shadows of front windows on the windows behind them.
        shadow = np.zeros(lab.shape, bool)
        mx, mt, mb = SHADOW_SIDE * W, SHADOW_TOP * W, SHADOW_BOTTOM * W
        for win in g.apps:
            k = g.labels[win.window_id]
            b = g.bounds[win.window_id]
            x1, y1 = max(0, int((b[0] - mx) // f)), max(0, int((b[1] - mt) // f))
            x2, y2 = min(lab.shape[1], int(math.ceil((b[2] + mx) / f))), min(lab.shape[0], int(math.ceil((b[3] + mb) / f)))
            if x2 > x1 and y2 > y1:
                sub = lab[y1:y2, x1:x2]
                shadow[y1:y2, x1:x2] |= (sub > 0) & (sub > k)  # windows behind (higher label = further back)
        return WindowView(labels=lab, factor=f, window_ids=list(g.ids), bounds=bounds, panes=panes,
                          shadow=shadow if shadow.any() else None)

    def _cap_to_windows(self, masks: list[MaskSpec], current: list[PictureRegion], start: dict[str, int]) -> list[MaskSpec]:
        """Tag every mask with its region's window and cap it at that
        window's bounds (a mask is never larger than its window)."""
        if not masks:
            return masks
        order = sorted(((i, rid) for rid, i in start.items()), key=lambda t: t[0])
        by_id = {r.region_id: r for r in current}
        g = geometry(self._layout) if self._layout is not None else None
        out: list[MaskSpec] = []
        j = 0
        owner: PictureRegion | None = None
        for idx, m in enumerate(masks):
            while j < len(order) and order[j][0] <= idx:
                owner = by_id.get(order[j][1])
                j += 1
            wid = owner.window_id if owner is not None else None
            box = tuple(float(v) for v in m.xyxy)
            if g is not None and wid is not None and wid in g.bounds:
                c = inter(box, g.bounds[wid])
                if c is None:
                    continue
                box = c
            out.append(replace(m, xyxy=box, window_id=wid))
        return out

    def _masks_for(self, reg: PictureRegion, verdict: Verdict, wh: tuple[int, int], **kw) -> list[MaskSpec]:
        """``modes.masks_for`` plus ``heavy=`` for an adaptive pre-frost
        policy (passed only when that version of ``masks_for`` takes it)."""
        if self._masks_heavy_kw is None:
            try:
                self._masks_heavy_kw = "heavy" in inspect.signature(masks_for).parameters
            except (TypeError, ValueError):
                self._masks_heavy_kw = False
        if self._masks_heavy_kw:
            kw["heavy"] = bool(getattr(self, "_heavy_now", False))
        return masks_for(self.policy, reg, verdict, wh, **kw)

    def _assign_ids(self, regions: list[PictureRegion], kept: list[PictureRegion], now_ms: float) -> list[PictureRegion]:
        """Give each proposal a stable id: same content (hash), same picture
        shifted by a scroll (clipped pictures change content), or same rect
        (content changed in place: video)."""
        old = dict(self._known)
        new_known: dict[str, _Known] = {}
        for reg in kept:
            if reg.region_id in old:
                new_known[reg.region_id] = old.pop(reg.region_id)
        taken = set(new_known)
        shifts: list[float] = []
        shift_pairs: list[tuple[Box, float, int | None]] = []
        pending: list[PictureRegion] = []
        # Pass 1: content hash (exact picture, possibly moved).
        for reg in regions:
            wb = size_bucket(reg.xyxy[2] - reg.xyxy[0], reg.xyxy[3] - reg.xyxy[1])
            match = None
            if incomplete(reg):
                pending.append(reg)  # its rect edge is not its picture edge: match by content shift
                continue
            for rid, k in old.items():
                if rid in taken or (k.region.kind == KIND_TILE) != (reg.kind == KIND_TILE) or incomplete(k.region):
                    continue
                kb = size_bucket(k.region.xyxy[2] - k.region.xyxy[0], k.region.xyxy[3] - k.region.xyxy[1])
                if kb == wb and hamming(k.region.content_hash, reg.content_hash) <= _ID_HAMMING:
                    match = rid
                    break
            if match is None:
                pending.append(reg)
                continue
            k = old[match]
            dx, dy = reg.xyxy[0] - k.region.xyxy[0], reg.xyxy[1] - k.region.xyxy[1]
            wdx, wdy = window_motion(self._prev_layout, self._layout, reg.window_id)
            if not (wdx or wdy):  # moved with a dragged window: not a scroll
                shifts.append(dy)
                shift_pairs.append((reg.xyxy, dy, reg.window_id))
            self._adopt(k, reg, match, dx, dy)
            new_known[match] = k
            taken.add(match)
        # Pass 2: scroll shift (clipped or growing pictures), then same rect.
        for reg in pending:
            match, dy = self._shift_match(reg, old, taken, shifts)
            if match is not None:
                k = old[match]
                self._adopt(k, reg, match, 0.0, dy)
            else:
                match = next(
                    (rid for rid, k in old.items() if rid not in taken and _iou(k.region.xyxy, reg.xyxy) >= _SAME_RECT_IOU),
                    None,
                )
                if match is not None:
                    k = old[match]
                    k.changes += 1
                    if k.changes >= _VIDEO_CHANGES and reg.kind != KIND_TILE:
                        reg.kind = KIND_VIDEO
                    k.full_ev = None  # new content: the old whole-picture view no longer applies
                    self._adopt(k, reg, match, 0.0, 0.0)
                else:
                    match = self._occlusion_match(reg, old, taken)
                    if match is not None:
                        # Partly hidden by a window in front: same picture, keep
                        # its whole-picture rect and evidence.
                        k = old[match]
                        reg.region_id = match
                        if k.region.kind == KIND_VIDEO and reg.kind != KIND_TILE:
                            reg.kind = KIND_VIDEO
                        k.static = 0
                        k.region = reg
            if match is None:
                rid = reg.region_id
                n = 1
                while rid in new_known or rid in taken or rid in old:
                    rid = f"{reg.region_id}~{n}"
                    n += 1
                reg.region_id = rid
                k = _Known(reg, now_ms, full_xyxy=reg.xyxy)
                match = rid
            else:
                k = old[match]
            reg.full_xyxy = k.full_xyxy
            new_known[match] = k
            taken.add(match)
        # Pass 3: pictures seen whole that are now mostly scrolled out (thin
        # strips the proposer cannot see) stay tracked while visible.
        if shifts:
            for rid, k in old.items():
                if rid in taken or k.full_ev is None or k.full_xyxy is None or k.region.kind == KIND_TILE:
                    continue
                if self._layout is not None and k.region.window_id is not None and self._layout.window(k.region.window_id) is None:
                    continue  # its window is gone
                dy = self._pane_shift(k.region.xyxy, k.region.window_id, shift_pairs)
                if dy is None or abs(dy) < 1.0:
                    continue
                full = _shift(k.full_xyxy, 0.0, dy)
                top, bottom = self._pane(k.region.window_id)
                vis = (max(0.0, full[0]), max(top, full[1]), min(float(self._frame_wh[0]), full[2]), min(bottom, full[3]))
                if vis[3] - vis[1] < 1.0 or vis[2] - vis[0] < 1.0:
                    continue
                if any(_iou(vis, r.xyxy) > 0.3 for r in regions):
                    continue
                diff = self._content_diff(k.region.xyxy, vis, dy) if vis[3] - vis[1] >= 4 * self._f else 0.0
                if diff is not None and diff > 6.0:
                    continue
                reg = replace(k.region, xyxy=vis, full_xyxy=full)
                reg.clipped = ("t" if vis[1] > full[1] + 0.5 else "") + ("b" if vis[3] < full[3] - 0.5 else "")
                if not reg.clipped:
                    continue
                k.region = reg
                k.full_xyxy = full
                new_known[rid] = k
                taken.add(rid)
        self._known = new_known
        return [k.region for k in new_known.values()]

    def _occluded_part(self, reg: PictureRegion, full: Box) -> bool:
        """``reg`` is the visible part of picture ``full``, the rest hidden by
        a window in front (or the menu bar / Dock)."""
        if self._layout is None or reg.window_id is None or reg.kind == KIND_TILE:
            return False
        a_full = (full[2] - full[0]) * (full[3] - full[1])
        a_reg = (reg.xyxy[2] - reg.xyxy[0]) * (reg.xyxy[3] - reg.xyxy[1])
        if a_full <= 0 or a_reg >= 0.95 * a_full:
            return False
        ix = max(0.0, min(full[2], reg.xyxy[2]) - max(full[0], reg.xyxy[0]))
        iy = max(0.0, min(full[3], reg.xyxy[3]) - max(full[1], reg.xyxy[1]))
        if ix * iy < 0.9 * a_reg:
            return False
        g = geometry(self._layout)
        vis = sum((r[2] - r[0]) * (r[3] - r[1]) for r in (inter(v, full) for v in g.visible.get(reg.window_id, [])) if r)
        return vis <= a_reg * 1.15  # what is not this region is hidden, not missing

    def _occlusion_match(self, reg: PictureRegion, old: dict[str, _Known], taken: set[str]) -> str | None:
        for rid, k in old.items():
            if rid in taken or k.region.kind == KIND_TILE or k.region.window_id != reg.window_id:
                continue
            if k.full_xyxy is not None and (k.full_ev is not None or k.carry is not None) and self._occluded_part(reg, k.full_xyxy):
                return rid
        return None

    def _clip_release(self, reg: PictureRegion, verdict: Verdict, k: _Known | None, H: int) -> bool:
        """A clipped picture that has stopped moving, shows a large part of
        itself and was checked SAFE is released (bounds the fail-closed cover
        to pictures that are scrolling in or barely visible)."""
        pol = self.policy
        if verdict.state is not VerdictState.SAFE or k is None or k.static < pol.clipped_release_static:
            return False
        w, h = reg.xyxy[2] - reg.xyxy[0], reg.xyxy[3] - reg.xyxy[1]
        top, bottom = self._pane(reg.window_id)
        # A static view this tall is no sliver either (a wide hero cut by the
        # toolbar): what is hidden is not on screen, what shows was checked.
        tall = h >= float(getattr(pol, "clipped_release_min_frac", 0.15)) * float(H)
        return tall or h >= pol.clipped_release_aspect * w or h >= pol.clipped_release_view * (bottom - top)

    @staticmethod
    def _pane_shift(box: Box, window_id: int | None, shifts_by: list[tuple[Box, float, int | None]]) -> float | None:
        """Scroll shift of the pane holding ``box``: the median dy of
        hash-matched pictures in the same window whose columns overlap it."""
        same = [(b, dy) for b, dy, w in shifts_by if w == window_id]
        local = [dy for b, dy in same if b[0] < box[2] and box[0] < b[2]]
        pool = local or [dy for _, dy in same]
        if not pool:
            return None
        return float(np.median(pool))

    @staticmethod
    def _adopt(k: _Known, reg: PictureRegion, rid: str, dx: float, dy: float) -> None:
        same = (
            k.region.content_hash == reg.content_hash
            and max(abs(a - b) for a, b in zip(k.region.xyxy, reg.xyxy)) <= 1.0
        )
        k.static = k.static + 1 if same else 0
        reg.region_id = rid
        if k.region.kind == KIND_VIDEO and reg.kind != KIND_TILE:
            reg.kind = KIND_VIDEO
        full = _shift(k.full_xyxy, dx, dy) if k.full_xyxy is not None else reg.xyxy
        if not incomplete(reg):
            full = reg.xyxy
        elif k.full_ev is None:
            # Still growing in: the full rect is at least what we have seen.
            full = (min(full[0], reg.xyxy[0]), min(full[1], reg.xyxy[1]), max(full[2], reg.xyxy[2]), max(full[3], reg.xyxy[3]))
        k.full_xyxy = full
        reg.full_xyxy = full
        k.region = reg

    def _shift_match(
        self, reg: PictureRegion, old: dict[str, _Known], taken: set[str], shifts: list[float]
    ) -> tuple[str | None, float]:
        """Old picture that, moved vertically, shows the same pixels as ``reg``."""
        if self._prev_small is None or self._small is None:
            return None, 0.0
        f = float(self._f)
        tol = 2.0 * f + 2.0
        best: tuple[float, str | None, float] = (1e9, None, 0.0)
        for rid, k in old.items():
            if rid in taken or k.region.kind == KIND_TILE or reg.kind == KIND_TILE:
                continue
            kv = k.region.xyxy
            if abs(kv[0] - reg.xyxy[0]) > tol or abs(kv[2] - reg.xyxy[2]) > tol:
                continue
            if not (incomplete(reg) or incomplete(k.region)):
                continue
            kf = k.full_xyxy or kv
            cands = {reg.xyxy[1] - kv[1], reg.xyxy[3] - kv[3], reg.xyxy[1] - kf[1], reg.xyxy[3] - kf[3]}
            cands |= {round(s) for s in shifts}
            for dy in cands:
                diff = self._content_diff(kv, reg.xyxy, dy)
                if diff is not None and diff < best[0]:
                    best = (diff, rid, float(dy))
        if best[1] is None or best[0] > 6.0:
            return None, 0.0
        return best[1], best[2]

    def _content_diff(self, old_box: Box, new_box: Box, dy: float) -> float | None:
        """Mean abs gray difference over the overlap of ``old_box + dy`` and ``new_box``."""
        f = self._f
        a = _shift(old_box, 0.0, dy)
        y1, y2 = max(a[1], new_box[1]), min(a[3], new_box[3])
        x1, x2 = max(a[0], new_box[0]), min(a[2], new_box[2])
        if (y2 - y1) < 4 * f or (x2 - x1) < 4 * f:
            return None
        cy1, cy2 = int(math.ceil(y1 / f)) + 1, int(math.floor(y2 / f)) - 1
        cx1, cx2 = int(math.ceil(x1 / f)) + 1, int(math.floor(x2 / f)) - 1
        oy = int(round(dy / f))
        if abs(dy / f - oy) > 0.34 or cy2 - cy1 < 2 or cx2 - cx1 < 2:
            # Shift not on the work grid: compare at an aligned offset anyway
            # (area-averaged pixels differ a little; the threshold allows it).
            pass
        cur = self._small[cy1:cy2, cx1:cx2]
        prev = self._prev_small[cy1 - oy : cy2 - oy, cx1:cx2]
        if cur.size == 0 or prev.shape != cur.shape:
            return None
        return float(cv2.absdiff(cur, prev).mean())

    def _priority(self, reg: PictureRegion) -> tuple:
        k = self._known.get(reg.region_id)
        first = k.first_seen_ms if k else 0.0
        area = (reg.xyxy[2] - reg.xyxy[0]) * (reg.xyxy[3] - reg.xyxy[1])
        return (reg.kind == KIND_TILE, first, -area)

    def _first_pass_job(self, reg: PictureRegion) -> _Job:
        x1, y1, x2, y2 = reg.xyxy
        crops: list[Box] = [reg.xyxy]
        long_side = max(x2 - x1, y2 - y1)
        # Big tiles too (a hero over the region cap is one tile): 2× tiles see the detail.
        if long_side >= self.tile_long_px and (reg.kind != KIND_TILE or reg.group_xyxy == reg.xyxy):
            tile = max(320.0, 0.6 * long_side)
            crops.extend(tile_grid(reg.xyxy, tile, 0.25))
        return _Job(reg, crops)

    def _second_look_job(self, reg: PictureRegion, W: int, H: int) -> _Job:
        x1, y1, x2, y2 = reg.xyxy
        m = self.policy.second_look_margin_frac
        mx, my = (x2 - x1) * m, (y2 - y1) * m
        box = (max(0.0, x1 - mx), max(0.0, y1 - my), min(float(W), x2 + mx), min(float(H), y2 + my))
        long_side = max(box[2] - box[0], box[3] - box[1])
        tile = max(96.0, 0.55 * long_side)
        return _Job(reg, [box] + tile_grid(box, tile, 0.25))

    # ------------------------------------------------------------ cloud lane

    def _submit(self, reg: PictureRegion, frame: np.ndarray, reason: str, *, score: float = 0.0,
                labels: tuple[str, ...] = ()) -> bool:
        x1, y1, x2, y2 = (int(round(v)) for v in reg.xyxy)
        crop = frame[max(0, y1) : y2, max(0, x1) : x2]
        w, h = float(x2 - x1), float(y2 - y1)
        key = (int(reg.content_hash), int(reg.ahash))
        big, scope = self._cache_scope(reg)
        if big:
            self._scoped[key] = (scope, (w, h))
            while len(self._scoped) > 256:
                self._scoped.popitem(last=False)
        try:
            ok = bool(self.cloud_lane.submit(region_id=reg.region_id, dhash=key[0], ahash=key[1], wh=(w, h),
                                             crop_bgr=crop, mode=self.mode.value, reason=reason))
        except Exception:  # noqa: BLE001 - never blocks, never breaks the frame
            return False
        if ok:
            self.escalations[key] = {"region_id": reg.region_id, "window_id": reg.window_id, "wh": (w, h),
                                     "l1_score": float(score), "labels": tuple(labels), "reason": reason}
            while len(self.escalations) > 256:
                self.escalations.popitem(last=False)
        return ok

    def _escalate(self, frame, current, evidence, sources, verdicts, now_ms: float) -> int:
        """Pictures the local model could not settle → the cloud lane."""
        from vision_core.live.cloud_lane import REASON_LARGE_WEAK, REASON_UNSURE

        n = 0
        for reg in current:
            if reg.kind == KIND_TILE or incomplete(reg) or sources.get(reg.region_id) in (None, "track"):
                continue
            ev = evidence.get(reg.region_id)
            v = verdicts.get(reg.region_id)
            if ev is None or v is None:
                continue
            w, h = reg.xyxy[2] - reg.xyxy[0], reg.xyxy[3] - reg.xyxy[1]
            reason = None
            if v.state is VerdictState.UNSURE and v.source != "clipped":
                # After the local second look, or when its bucket had no token.
                if ev.second_look or needs_second_look(self.policy, v, ev):
                    reason = REASON_UNSURE
            elif v.state is VerdictState.SAFE and max(w, h) >= CLOUD_LARGE_PX:
                if CLOUD_LARGE_MIN_SCORE <= v.score < self.policy.score_unsure:
                    reason = REASON_LARGE_WEAK
            if reason is not None and self._submit(reg, frame, reason, score=v.score, labels=v.labels):
                n += 1
        return n

    def _run_jobs(self, frame: np.ndarray, jobs: Sequence[_Job], now_ms: float = 0.0) -> list[Evidence | None]:
        """Detector on every crop of these jobs. With a cloud lane, a failing
        local detector leaves the jobs PENDING (None) and sends each picture
        to the cloud (``local_unavailable``); without one it raises, as before."""
        try:
            return self._run_jobs_local(frame, jobs)
        except Exception as exc:
            if self.cloud_lane is None:
                raise
            from vision_core.live.cloud_lane import REASON_LOCAL_UNAVAILABLE

            self.detector_error = f"{type(exc).__name__}: {exc}"[:200]
            for job in jobs:
                if job.region.kind != KIND_TILE:
                    self._submit(job.region, frame, REASON_LOCAL_UNAVAILABLE)
            return [None for _ in jobs]

    def _run_jobs_local(self, frame: np.ndarray, jobs: Sequence[_Job]) -> list[Evidence]:
        images: list[np.ndarray] = []
        owners: list[tuple[int, Box]] = []
        for j, job in enumerate(jobs):
            for c in job.crops:
                ix1, iy1 = int(math.floor(c[0])), int(math.floor(c[1]))
                ix2, iy2 = int(math.ceil(c[2])), int(math.ceil(c[3]))
                crop = frame[iy1:iy2, ix1:ix2]
                if crop.size == 0:
                    continue
                images.append(crop)
                owners.append((j, (float(ix1), float(iy1), float(ix2), float(iy2))))
        t = time.perf_counter()
        raw = self.detector.detect_batch(images) if images else []
        self.detector_error = None
        dt = (time.perf_counter() - t) * 1000.0
        if images:
            self._crop_ms = 0.7 * self._crop_ms + 0.3 * (dt / len(images))
        per_job: list[list[tuple[str, float, Box]]] = [[] for _ in jobs]
        for (j, (ox, oy, _, _)), dets in zip(owners, raw):
            for d in dets:
                b = d.xyxy
                per_job[j].append((d.label, d.score, (b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy)))
        out: list[Evidence] = []
        share = dt / max(1, len(jobs))
        for job, dets in zip(jobs, per_job):
            out.append(Evidence(self._normalise(job.region, _nms(dets)), None, share, len(job.crops)))
        return out

    @staticmethod
    def _normalise(reg: PictureRegion, dets: list[tuple[str, float, Box]]) -> tuple[NormDet, ...]:
        x1, y1, x2, y2 = reg.xyxy
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        out = []
        for label, score, b in dets:
            nb = (
                min(1.0, max(0.0, (b[0] - x1) / w)),
                min(1.0, max(0.0, (b[1] - y1) / h)),
                min(1.0, max(0.0, (b[2] - x1) / w)),
                min(1.0, max(0.0, (b[3] - y1) / h)),
            )
            if nb[2] - nb[0] > 0 and nb[3] - nb[1] > 0:
                out.append(NormDet(label, float(score), nb))
        return tuple(out)

    def _cache_scope(self, reg: PictureRegion) -> tuple[bool, object]:
        """(big, scope): a region over ~20 % of the screen is mostly layout,
        so its entry is scoped to its window, short-lived and never reused
        at another size."""
        w, h = reg.xyxy[2] - reg.xyxy[0], reg.xyxy[3] - reg.xyxy[1]
        W, H = self._frame_wh or (1, 1)
        big = w * h > BIG_REGION_FRAC * float(W * H)
        return big, (("win", reg.window_id) if big else None)

    def _cache_get(self, reg: PictureRegion, now_ms: float) -> Evidence | None:
        w, h = reg.xyxy[2] - reg.xyxy[0], reg.xyxy[3] - reg.xyxy[1]
        big, scope = self._cache_scope(reg)
        if not big:
            return self.cache.get(reg.content_hash, reg.ahash, (w, h), now_ms)
        return self.cache.get(reg.content_hash, reg.ahash, (w, h), now_ms, scope=scope)

    def _cache_put(self, reg: PictureRegion, ev: Evidence, now_ms: float) -> None:
        w, h = reg.xyxy[2] - reg.xyxy[0], reg.xyxy[3] - reg.xyxy[1]
        big, scope = self._cache_scope(reg)
        if not big:
            self.cache.put(reg.content_hash, reg.ahash, (w, h), ev, now_ms)
            return
        self.cache.put(reg.content_hash, reg.ahash, (w, h), ev, now_ms, scope=scope, ttl_ms=BIG_REGION_TTL_MS, any_size=False, refresh=False)


class _ScopedCacheView:
    """What ``CloudLane.apply`` sees of the verdict cache.

    Cloud answers are keyed by (dHash, aHash, size). Regions over ~20 % of
    the screen live in **window-scoped**, short-lived entries (so wallpaper
    and layout cannot bring an old verdict back). For keys the analyzer
    escalated from such a region, the answer is merged into that same scoped
    entry (same window scope, same short TTL, no any-size reuse); every
    other answer goes to the ordinary global entry. A cloud verdict can thus
    never leak from one window to another through a layout-dominated hash.
    """

    def __init__(self, cache: VerdictCache, scoped: dict) -> None:
        self.cache = cache
        self.scoped = scoped

    def get(self, dhash: int, ahash: int, wh, now_ms: float):
        hit = self.scoped.get((int(dhash), int(ahash)))
        if hit is not None:
            return self.cache.get(dhash, ahash, wh, now_ms, scope=hit[0])
        return self.cache.get(dhash, ahash, wh, now_ms)

    def put(self, dhash: int, ahash: int, wh, value, now_ms: float):
        hit = self.scoped.get((int(dhash), int(ahash)))
        if hit is not None:
            return self.cache.put(dhash, ahash, wh, value, now_ms, scope=hit[0], ttl_ms=BIG_REGION_TTL_MS,
                                  any_size=False, refresh=False)
        return self.cache.put(dhash, ahash, wh, value, now_ms)


def _nms(dets: list[tuple[str, float, Box]], iou: float = 0.45) -> list[tuple[str, float, Box]]:
    """Per-class NMS across the crops of one picture (whole view + tiles)."""
    if len(dets) <= 1:
        return list(dets)
    labels = sorted({d[0] for d in dets})
    ids = [labels.index(d[0]) for d in dets]
    xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for _, _, b in dets]
    keep = cv2.dnn.NMSBoxesBatched(xywh, [d[1] for d in dets], ids, 0.0, iou)
    return [dets[int(i)] for i in np.asarray(keep).reshape(-1)]


def make_analyzer(mode: ProtectionMode = ProtectionMode.AGGRESSIVE, **opts) -> Analyzer:
    """Factory used by the live engine and the eval harness.

    Options (all optional): ``detector`` (inject a ``PartDetector``),
    ``model_path`` (e.g. a future ``640m.onnx``), ``backend`` (``auto`` /
    ``cv2`` / ``ort``), ``budget_ms``, ``second_look_rate_per_s``,
    ``second_look_burst``, ``policy_dir``, ``cache``, ``region_opts``,
    ``tile_long_px``, ``max_batch``, ``warmup``, ``cloud_lane``.
    """
    return Analyzer(ProtectionMode(mode), **opts)
