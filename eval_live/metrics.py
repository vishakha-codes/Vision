"""Live-path metrics: proposals, coverage, exposure, over-mask, scroll-follow lag.

All geometry is in screen (capture) pixels. Pixel areas are counted on a
boolean raster with cell size ``grid`` (1 = exact; the runner uses 2 at
Retina so a frame costs ~1 ms). Masks and targets are rasterised with the
same rounding, so a mask that equals its target covers it exactly.

Targets depend on the protection semantics:

- **AGGRESSIVE** target = the whole visible flagged picture. Covering any
  other pixel is over-mask.
- **MEDIUM** target = the flagged picture's part boxes (stand-ins for what a
  body-part detector returns), padded by ``part_pad``. Covering the rest of
  a flagged picture, other pictures, text or UI is over-mask.

Exposure (per frame): some flagged target has visible, uncovered pixels
greater than ``EXPOSURE_FRAC`` (5%) of its full area. On desktop frames a
picture's pixels under a window in front (or the menu bar / Dock) are not
visible, so they neither count as exposed nor as target.

**Desktop frames** (``FrameGT.desktop``) also get window metrics, from the
drawn (true) window layout and each mask's ``window_id``:

- ``sys_px``: mask pixels on the menu bar, the Dock or the strip outside
  ``visibleFrame`` (must be 0);
- ``outside_own_px``: mask pixels outside the visible region of the mask's
  own window (no mask is larger than its window's visible content);
  ``front_leak_px``: the part of that on windows in front of it;
- ``max_mask_frac``: largest single mask / screen;
- ``mask_px`` vs ``flagged_px``: total mask area for one flagged picture.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Iterable, Sequence as Seq

import numpy as np

from eval_live.sequences import FrameGT, GTPicture
from vision_core.live.windows import WindowGeometry

Box = tuple[float, float, float, float]

EXPOSURE_FRAC = 0.05
FOLLOW_IOU = 0.8
IOU_THRESHOLDS = (0.5, 0.7)
MIN_EVENT_SIDE = 24  # px; slivers at the viewport edge make IoU meaningless


# --------------------------------------------------------------------------- geometry


def _g(v: float, grid: int, up: bool) -> int:
    return int(math.ceil(v / grid)) if up else int(math.floor(v / grid))


def _gbox(b: Box, grid: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x1 = max(0, min(w, _g(b[0], grid, False)))
    y1 = max(0, min(h, _g(b[1], grid, False)))
    x2 = max(0, min(w, _g(b[2], grid, True)))
    y2 = max(0, min(h, _g(b[3], grid, True)))
    return x1, y1, x2, y2


def rasterize(boxes: Iterable[Box], width: int, height: int, grid: int = 1) -> np.ndarray:
    shape = (_g(height, grid, True), _g(width, grid, True))
    out = np.zeros(shape, bool)
    for b in boxes:
        x1, y1, x2, y2 = _gbox(b, grid, shape)
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = True
    return out


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _clip(b: Box, c: Box) -> Box | None:
    x1, y1, x2, y2 = max(b[0], c[0]), max(b[1], c[1]), min(b[2], c[2]), min(b[3], c[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _expand(b: Box, pad: float) -> Box:
    return (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)


def follow_iou(target: Box, masks: Seq[Box], clip: Box, grid: int = 1) -> float:
    """IoU of ``target`` with the union of masks that belong to it.

    A mask belongs to the target when it mostly lies on it (≥ 50% of the
    mask) or overlaps it with IoU ≥ 0.3, so a neighbour's mask touching the
    edge does not count, and a target split into two masks still can.
    Masks are clipped to the target's pane first.
    """
    cands: list[Box] = []
    for m in masks:
        mc = _clip(m, clip)
        if mc is None:
            continue
        inter = _clip(mc, target)
        if inter is None:
            continue
        if _area(inter) >= 0.5 * _area(mc) or iou(mc, target) >= 0.3:
            cands.append(mc)
    if not cands:
        return 0.0
    if len(cands) == 1:
        return iou(cands[0], target)
    xs = [target[0]] + [c[0] for c in cands]
    ys = [target[1]] + [c[1] for c in cands]
    ox, oy = min(xs), min(ys)
    w = max([target[2]] + [c[2] for c in cands]) - ox
    h = max([target[3]] + [c[3] for c in cands]) - oy
    shift = lambda b: (b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy)  # noqa: E731
    u = rasterize([shift(c) for c in cands], int(math.ceil(w)), int(math.ceil(h)), grid)
    t = rasterize([shift(target)], int(math.ceil(w)), int(math.ceil(h)), grid)
    union = np.count_nonzero(u | t)
    return float(np.count_nonzero(u & t)) / union if union else 0.0


# --------------------------------------------------------------------------- per frame


def _targets(pic: GTPicture, semantics: str) -> tuple[list[Box], list[Box]]:
    """(visible target boxes, full target boxes) for one flagged picture."""
    if semantics == "aggressive":
        return [pic.xyxy], [pic.full_xyxy]
    return list(pic.parts), list(pic.full_parts)


def _union_area(boxes: Seq[Box], grid: int) -> float:
    if not boxes:
        return 0.0
    if len(boxes) == 1:
        return _area(boxes[0])
    ox = min(b[0] for b in boxes)
    oy = min(b[1] for b in boxes)
    w = max(b[2] for b in boxes) - ox
    h = max(b[3] for b in boxes) - oy
    r = rasterize([(b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy) for b in boxes], int(math.ceil(w)), int(math.ceil(h)), grid)
    return float(np.count_nonzero(r)) * grid * grid


def _hide(R: np.ndarray, rects: Iterable[Box], grid: int) -> np.ndarray:
    """``R`` with ``rects`` cleared (hidden pixels are not visible targets)."""
    for b in rects:
        x1, y1, x2, y2 = _gbox(b, grid, R.shape)
        if x2 > x1 and y2 > y1:
            R[y1:y2, x1:x2] = False
    return R


def _desktop_record(gt: FrameGT, masks: Seq[Box], owners: Seq[int | None] | None, M: np.ndarray,
                    width: int, height: int, grid: int) -> dict:
    g2 = grid * grid
    d = gt.desktop or {}
    truth = d.get("truth")
    vis = d.get("visible") or (0, 0, width, height)
    system = rasterize([(0, 0, width, height)], width, height, grid)
    _hide(system, [vis], grid)
    system |= rasterize([d["dock"]], width, height, grid) if d.get("dock") else False
    out: dict = {
        "sys_px": int(np.count_nonzero(M & system)) * g2,
        "max_mask_frac": max((_area(m) / float(width * height) for m in masks), default=0.0),
        "mask_px": int(np.count_nonzero(M)) * g2,
        "outside_own_px": 0,
        "front_leak_px": 0,
        "since_event": gt.frames_since_event,
    }
    if truth is None:
        return out
    g = WindowGeometry(truth)
    order = [w.window_id for w in g.apps]  # front to back
    own_cache: dict[int | None, np.ndarray] = {}

    def region(wid: int | None) -> np.ndarray:
        if wid not in own_cache:
            own_cache[wid] = rasterize(g.visible.get(wid, []), width, height, grid)
        return own_cache[wid]

    for k, m in enumerate(masks):
        wid = owners[k] if owners is not None and k < len(owners) else None
        mm = rasterize([m], width, height, grid)
        if wid is None:
            out["outside_own_px"] += int(np.count_nonzero(mm & system)) * g2
            continue
        out["outside_own_px"] += int(np.count_nonzero(mm & ~region(wid))) * g2
        if wid in order:
            for front in order[: order.index(wid)]:
                out["front_leak_px"] += int(np.count_nonzero(mm & region(front))) * g2
    return out


def frame_record(
    gt: FrameGT,
    masks: Seq[Box],
    width: int,
    height: int,
    *,
    grid: int = 1,
    part_pad: float = 12.0,
    owners: Seq[int | None] | None = None,
) -> dict:
    """Coverage / exposure / over-mask numbers for one frame. ``owners``:
    each mask's ``window_id`` (desktop window metrics)."""
    g2 = grid * grid
    M = rasterize(masks, width, height, grid)
    flagged = gt.flagged()
    F = _hide(rasterize([p.xyxy for p in flagged], width, height, grid), [h for p in flagged for h in p.hidden], grid)
    P = rasterize([p.xyxy for p in gt.pictures], width, height, grid)
    Tm = rasterize([_expand(b, part_pad) for p in flagged for b in p.parts], width, height, grid)
    Tp = rasterize([b for p in flagged for b in p.parts], width, height, grid)
    shape = M.shape
    exposed = {"aggressive": [], "medium": []}
    vis_px = cov_px = 0
    part_vis = part_cov = 0
    for p in flagged:
        x1, y1, x2, y2 = _gbox(p.xyxy, grid, shape)
        sl = M[y1:y2, x1:x2]
        seen = None
        if p.hidden:
            seen = F[y1:y2, x1:x2] if len(flagged) == 1 else _hide(rasterize([p.xyxy], width, height, grid), p.hidden, grid)[y1:y2, x1:x2]
            v = int(np.count_nonzero(seen))
            c = int(np.count_nonzero(sl & seen))
        else:
            v = sl.size
            c = int(np.count_nonzero(sl))
        vis_px += v
        cov_px += c
        if (v - c) * g2 > EXPOSURE_FRAC * p.full_area:
            exposed["aggressive"].append(p.gid)
        if p.parts:
            T = Tp[y1:y2, x1:x2]
            if p.hidden:
                T = T & seen  # parts under a window in front are not visible
            tv = int(np.count_nonzero(T))
            tc = int(np.count_nonzero(T & sl))
            part_vis += tv
            part_cov += tc
            full_parts = _union_area(p.full_parts, grid)
            if full_parts and (tv - tc) * g2 > EXPOSURE_FRAC * full_parts:
                exposed["medium"].append(p.gid)
    screen_px = M.size
    over_agg = int(np.count_nonzero(M & ~F))
    over_med = int(np.count_nonzero(M & ~Tm))
    desk = _desktop_record(gt, masks, owners, M, width, height, grid) if gt.desktop is not None else None
    return {
        "desktop": desk,
        "index": gt.index,
        "layout": gt.layout,
        "moving": gt.moving,
        "after_nav": gt.after_nav,
        "nav": gt.nav,
        "n_masks": len(masks),
        "n_flagged": len(flagged),
        "flagged_px": vis_px * g2,
        "covered_px": cov_px * g2,
        "coverage": (cov_px / vis_px) if vis_px else None,
        "parts_coverage": (part_cov / part_vis) if part_vis else None,
        "exposed": {k: sorted(v) for k, v in exposed.items()},
        "over": {
            "aggressive": {
                "px": over_agg * g2,
                "frac": over_agg / max(1, screen_px - int(np.count_nonzero(F))),
                "on_pictures_px": int(np.count_nonzero(M & P & ~F)) * g2,
                "on_text_ui_px": int(np.count_nonzero(M & ~P)) * g2,
            },
            "medium": {
                "px": over_med * g2,
                "frac": over_med / max(1, screen_px - int(np.count_nonzero(Tm))),
                "on_flagged_non_part_px": int(np.count_nonzero(M & F & ~Tm)) * g2,
                "on_pictures_px": int(np.count_nonzero(M & P & ~F)) * g2,
                "on_text_ui_px": int(np.count_nonzero(M & ~P)) * g2,
            },
        },
    }


# --------------------------------------------------------------------------- proposals


def match_proposals(gt: FrameGT, proposals: Seq[Box], thr: float) -> dict:
    """Greedy one-to-one IoU matching. Eligible GT = ≥ 50% visible and ≥ 24 px a side."""
    gts = gt.pictures
    elig = [
        (p.visible_area >= 0.5 * p.full_area)
        and (p.xyxy[2] - p.xyxy[0] >= MIN_EVENT_SIDE)
        and (p.xyxy[3] - p.xyxy[1] >= MIN_EVENT_SIDE)
        for p in gts
    ]
    pairs = sorted(
        ((iou(tuple(map(float, pr)), tuple(map(float, g.xyxy))), i, j) for i, pr in enumerate(proposals) for j, g in enumerate(gts)),
        reverse=True,
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
    n_elig = sum(elig)
    tp_g = sum(1 for j in used_g if elig[j])
    tp_flag = sum(1 for j in used_g if elig[j] and gts[j].flagged)
    n_flag = sum(1 for j, e in enumerate(elig) if e and gts[j].flagged)
    return {
        "n_gt": n_elig,
        "n_gt_flagged": n_flag,
        "n_prop": len(proposals),
        "tp_gt": tp_g,
        "tp_gt_flagged": tp_flag,
        "tp_prop": len(used_p),
    }


# --------------------------------------------------------------------------- events


def _target_keys(gt: FrameGT, semantics: str) -> dict[str, tuple[Box, Box, Box]]:
    out: dict[str, tuple[Box, Box, Box]] = {}
    for p in gt.flagged():
        vis, full = _targets(p, semantics)
        for k, (v, f) in enumerate(zip(vis, full)):
            key = p.gid if semantics == "aggressive" else f"{p.gid}#{k}"
            out[key] = (tuple(map(float, v)), tuple(map(float, f)), tuple(map(float, p.clip)))
    return out


def _big_enough(vis: Box, full: Box) -> bool:
    return (
        vis[2] - vis[0] >= MIN_EVENT_SIDE
        and vis[3] - vis[1] >= MIN_EVENT_SIDE
        and _area(vis) >= 0.2 * _area(full)
    )


def lag_events(gts: Seq[FrameGT], masks: Seq[Seq[Box]], semantics: str, *, grid: int = 1) -> list[dict]:
    """Scroll-follow lag: frames until mask IoU ≥ 0.8 with the moved target.

    An event starts on frame t when a target that was followed on t-1
    (IoU ≥ 0.8) moves. ``lag`` is the smallest k ≥ 0 with IoU ≥ 0.8 on
    t+k. If the sequence ends first the event is ``unresolved``; if the
    target leaves the screen first it is dropped.
    """
    keys = [_target_keys(g, semantics) for g in gts]
    ious: list[dict[str, float]] = []
    for kd, ms in zip(keys, masks):
        ious.append({k: follow_iou(v, ms, c, grid) for k, (v, _f, c) in kd.items() if _big_enough(v, _f)})
    events: list[dict] = []
    for t in range(1, len(gts)):
        for key, (vis, full, _c) in keys[t].items():
            prev = keys[t - 1].get(key)
            if prev is None or prev[1] == full:
                continue
            if ious[t - 1].get(key, 0.0) < FOLLOW_IOU or key not in ious[t]:
                continue
            lag = None
            for k in range(t, len(gts)):
                if key not in ious[k]:
                    lag = "left"
                    break
                if ious[k][key] >= FOLLOW_IOU:
                    lag = k - t
                    break
            if lag == "left":
                continue
            events.append({
                "frame": t,
                "key": key,
                "layout": gts[t].layout,
                "dy": gts[t].scroll_dy,
                "lag": lag,  # None = unresolved at sequence end
                "iou_at_move": round(ious[t][key], 4),
            })
    return events


def cover_events(gts: Seq[FrameGT], records: Seq[dict], semantics: str) -> list[dict]:
    """Time-to-cover: frames from a flagged target becoming exposable to covered.

    Starts when a flagged picture first has > 5% of its target visible (it
    scrolled in, the page changed, or a video turned flagged); ``frames`` is
    how many frames it stayed exposed (0 = covered on arrival).
    """
    def exposable(p: GTPicture) -> bool:
        vis, full = _targets(p, semantics)
        if semantics == "aggressive":
            return p.visible_area > EXPOSURE_FRAC * p.full_area
        tot = sum(_area(f) for f in full)
        return tot > 0 and sum(_area(v) for v in vis) > EXPOSURE_FRAC * tot

    out: list[dict] = []
    prev: set[str] = set()
    for t, (g, rec) in enumerate(zip(gts, records)):
        now = {p.gid for p in g.flagged() if exposable(p)}
        for gid in sorted(now - prev):
            frames = None
            for k in range(t, len(gts)):
                still = {p.gid for p in gts[k].flagged() if exposable(p)}
                if gid not in still:
                    frames = k - t  # left before being covered; exposed the whole time
                    break
                if gid not in records[k]["exposed"][semantics]:
                    frames = k - t
                    break
            out.append({"frame": t, "gid": gid, "layout": g.layout, "after_nav": g.after_nav, "frames": frames})
        prev = now
    return out


# --------------------------------------------------------------------------- sequence + summary


@dataclass
class SequenceEval:
    name: str
    width: int
    height: int
    records: list[dict] = field(default_factory=list)
    lags: dict[str, list[dict]] = field(default_factory=dict)
    covers: dict[str, list[dict]] = field(default_factory=dict)
    proposals: list[dict] = field(default_factory=list)  # per frame, per thr (analyzer level)


def evaluate_sequence(
    name: str,
    gts: Seq[FrameGT],
    masks: Seq[Seq[Box]],
    width: int,
    height: int,
    *,
    proposals: Seq[Seq[Box]] | None = None,
    grid: int = 1,
    part_pad: float | None = None,
    owners: Seq[Seq[int | None]] | None = None,
) -> SequenceEval:
    if len(masks) != len(gts):
        raise ValueError(f"{name}: {len(masks)} mask lists for {len(gts)} frames")
    masks = [[tuple(map(float, m)) for m in ms] for ms in masks]
    pad = part_pad if part_pad is not None else 12.0 * width / 1728.0
    ev = SequenceEval(name, width, height)
    ev.records = [
        frame_record(g, ms, width, height, grid=grid, part_pad=pad, owners=None if owners is None else owners[i])
        for i, (g, ms) in enumerate(zip(gts, masks))
    ]
    for sem in ("aggressive", "medium"):
        ev.lags[sem] = lag_events(gts, masks, sem, grid=grid)
        ev.covers[sem] = cover_events(gts, ev.records, sem)
    if proposals is not None:
        for g, props in zip(gts, proposals):
            row = {"layout": g.layout}
            for thr in IOU_THRESHOLDS:
                row[str(thr)] = match_proposals(g, props, thr)
            ev.proposals.append(row)
    return ev


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    vals = sorted(vals)
    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 3),
        "p50": round(vals[len(vals) // 2], 3),
        "p95": round(vals[min(len(vals) - 1, int(math.ceil(0.95 * len(vals))) - 1)], 3),
        "max": round(vals[-1], 3),
    }


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def desktop_summary(recs: Seq[dict], settle: int = 3) -> dict | None:
    """Window metrics over desktop frames. ``settle``: frames after a scene
    event (and after the start) before over-mask / area numbers count, i.e.
    once the analyzer's results have landed."""
    rows = [r for r in recs if r.get("desktop") is not None]
    if not rows:
        return None

    def settled(r: dict) -> bool:
        se = r["desktop"]["since_event"]
        return r["index"] >= settle and (se is None or se >= settle)

    one = [r for r in rows if r["n_flagged"] == 1 and settled(r) and not r["moving"] and r["flagged_px"] > 0]
    ratios = [r["desktop"]["mask_px"] / float(r["flagged_px"]) for r in one]
    clean = [r for r in rows if r["n_flagged"] == 0 and settled(r)]
    clean_over = [r["over"]["aggressive"]["frac"] for r in clean]
    d = [r["desktop"] for r in rows]
    return {
        "frames": len(rows),
        "sys_cover_px_max": max(x["sys_px"] for x in d),
        "sys_cover_frames": sum(1 for x in d if x["sys_px"] > 0),
        "max_mask_frac": round(max(x["max_mask_frac"] for x in d), 4),
        "outside_own_px_max": max(x["outside_own_px"] for x in d),
        "outside_own_frames": sum(1 for x in d if x["outside_own_px"] > 0),
        "front_leak_px_max": max(x["front_leak_px"] for x in d),
        "front_leak_frames": sum(1 for x in d if x["front_leak_px"] > 0),
        "one_flag_frames": len(one),
        "one_flag_area_ratio_max": round(max(ratios), 3) if ratios else None,
        "one_flag_area_ratio_mean": round(statistics.fmean(ratios), 3) if ratios else None,
        "clean_frames": len(clean),
        "clean_over_mask_max": round(max(clean_over), 4) if clean_over else None,
        "clean_over_mask_mean": round(statistics.fmean(clean_over), 4) if clean_over else None,
    }


def summarize(evals: Seq[SequenceEval], semantics: str, layout: str | None = None, *, settle: int = 3) -> dict:
    """Aggregate over sequences (optionally only frames/events of one layout).

    ``semantics`` picks the headline target ("aggressive" | "medium"); both
    over-mask variants are always reported. Desktop frames add a
    ``desktop`` block (see :func:`desktop_summary`).
    """
    keep = (lambda d: d["layout"] == layout) if layout else (lambda d: True)
    recs = [r for e in evals for r in e.records if keep(r)]
    with_flag = [r for r in recs if r["n_flagged"]]
    exp = [r for r in recs if r["exposed"][semantics]]
    scroll = [r for r in recs if r["moving"]]
    nav = [r for r in recs if r["after_nav"]]
    static = [r for r in recs if not r["moving"] and not r["after_nav"]]
    exp_of = lambda rs: sum(1 for r in rs if r["exposed"][semantics])  # noqa: E731
    covs = [r["coverage"] for r in with_flag if r["coverage"] is not None]
    pcovs = [r["parts_coverage"] for r in with_flag if r["parts_coverage"] is not None]
    lags = [ev for e in evals for ev in e.lags[semantics] if keep(ev)]
    resolved = [ev["lag"] for ev in lags if ev["lag"] is not None]
    covers = [ev for e in evals for ev in e.covers[semantics] if keep(ev)]
    ttc = [ev["frames"] for ev in covers if ev["frames"] is not None]
    out: dict = {
        "frames": len(recs),
        "frames_with_flagged": len(with_flag),
        "exposure": {
            "semantics": semantics,
            "frames": len(exp),
            "rate": _rate(len(exp), len(with_flag)),
            "during_scroll": {"frames": exp_of(scroll), "of": len(scroll), "rate": _rate(exp_of(scroll), len(scroll))},
            "after_navigation": {"frames": exp_of(nav), "of": len(nav), "rate": _rate(exp_of(nav), len(nav))},
            "static": {"frames": exp_of(static), "of": len(static), "rate": _rate(exp_of(static), len(static))},
            "picture_target_frames": sum(1 for r in recs if r["exposed"]["aggressive"]),
            "part_target_frames": sum(1 for r in recs if r["exposed"]["medium"]),
        },
        "coverage": {
            "picture_mean": round(statistics.fmean(covs), 4) if covs else None,
            "picture_min": round(min(covs), 4) if covs else None,
            "parts_mean": round(statistics.fmean(pcovs), 4) if pcovs else None,
        },
        "over_mask": {},
        "scroll_follow_lag_frames": {
            "semantics": semantics,
            "events": len(lags),
            "unresolved": len(lags) - len(resolved),
            **{k: v for k, v in _stats([float(x) for x in resolved]).items() if k != "n"},
            "lag0_rate": _rate(sum(1 for x in resolved if x == 0), len(lags)),
        },
        "time_to_cover_frames": {
            "appearances": len(covers),
            "never_covered": len(covers) - len(ttc),
            "covered_on_arrival_rate": _rate(sum(1 for x in ttc if x == 0), len(covers)),
            **{k: v for k, v in _stats([float(x) for x in ttc]).items() if k != "n"},
        },
    }
    for sem in ("aggressive", "medium"):
        o = [r["over"][sem] for r in recs]
        block = {
            "frac_mean": round(statistics.fmean([x["frac"] for x in o]), 5) if o else None,
            "px_mean": round(statistics.fmean([x["px"] for x in o]), 1) if o else None,
            "frames_any": sum(1 for x in o if x["px"] > 0),
        }
        for key in ("on_pictures_px", "on_text_ui_px", "on_flagged_non_part_px"):
            if o and key in o[0]:
                block[key + "_mean"] = round(statistics.fmean([x[key] for x in o]), 1)
        out["over_mask"][sem] = block
    desk = desktop_summary(recs, settle)
    if desk is not None:
        out["desktop"] = desk
    props = [p for e in evals for p in e.proposals if keep(p)]
    if props:
        pm: dict = {}
        for thr in IOU_THRESHOLDS:
            rows = [p[str(thr)] for p in props]
            s = {k: sum(r[k] for r in rows) for k in rows[0]}
            pm[f"iou{thr}"] = {
                "recall": _rate(s["tp_gt"], s["n_gt"]),
                "recall_flagged": _rate(s["tp_gt_flagged"], s["n_gt_flagged"]),
                "precision": _rate(s["tp_prop"], s["n_prop"]),
                "n_gt": s["n_gt"],
                "n_prop": s["n_prop"],
            }
        out["proposals"] = pm
    return out


def layouts_of(evals: Seq[SequenceEval]) -> list[str]:
    return sorted({r["layout"] for e in evals for r in e.records})
