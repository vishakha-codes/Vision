"""Picture-region proposals on a screenshot (photos, thumbnails, hero images,
video frames, pictures clipped at the frame edge).

Screens are mostly flat page / UI colours with text drawn on them; pictures
are the places where that stops. So the pipeline, at ~1/3 scale on Retina:

1. **Background colours.** Exactly-flat pixels (no neighbour differs by more
   than 1 level) are histogrammed in 5-bit colour bins; bins holding ≥ 1 % of
   the frame are page / UI backgrounds. ``fg`` = every pixel not within
   ``bg_tol`` of one of them (``cv2.inRange``), so a flat sky inside a photo
   stays foreground while the page around it does not.
2. **Solid mask.** ``close 3×3`` fills speckle holes, ``open k×k`` removes
   strokes thinner than ~15 capture px (text, rules, borders).
3. **Connected components** of the solid mask big enough to hold a picture.
4. **Recursive XY-cut** of the raw ``fg`` inside each component: full rows /
   columns of background are gutters, so a thumbnail grid splits into its
   cells and text lines fall below the minimum size and vanish.
5. **Picture-vs-text/UI score** per leaf: gray-level entropy, share of
   mid-tones vs the two dominant tones (text is bimodal ink / paper),
   colourfulness (Hasler–Süsstrunk), share of gradient energy that is
   axis-aligned (UI and glyphs are H/V; photos are not), and fill.
6. Edges **refined at full resolution** (±1 work px), NMS / containment
   merge (a container holding ≥ 2 accepted pictures is dropped, the
   pictures are kept), fragment merge.
7. **Tile fallback:** solid, textured cells not covered by an accepted region
   (collages, a picture mixed with UI) get overlapping tiles (≥ 25 %
   overlap) so there are no coverage gaps.

**Windows** (``WindowView``, built by the analyzer from the window list):
only pixels inside app windows are foreground (wallpaper, menu bar, Dock and
window shadows are label 0), components are cut at window borders, and every
region carries its ``window_id``. **Safety caps**, whatever the score: no
region over ``max_region_frac`` (40 %) of the frame, none touching 3 or more
frame edges, none crossing its window's border.

Region ids are content-derived (dHash + size bucket) so the same picture keeps
its id after a scroll; duplicates on one frame get a ``~n`` suffix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from vision_core.live.cache import hash_crop, size_bucket
from vision_core.live.contracts import Box, PictureRegion

KIND_PICTURE = "picture"
KIND_VIDEO = "video"
KIND_TILE = "tile"  # tile-fallback crop over a picture-like area (not a picture rect)


@dataclass
class RegionOptions:
    work_long: int = 1152  # long edge of the analysis image (3456 → factor 3)
    min_side_px: int = 96  # capture px, short side, at Retina width (3456)
    ref_width: int = 3456  # min_side / solid_px scale with frame width (1× screen → 48 px)
    min_side_floor: int = 40
    sliver_px: int = 16  # short side of a picture cut by the frame / a pane edge (Retina px)
    sliver_floor: int = 6
    bg_tol: int = 1  # L∞ distance to a background colour
    bg_min_frac: float = 0.01  # share of the frame for a flat colour to be "background"
    max_bg_colors: int = 8
    solid_px: int = 15  # capture px; thinner strokes are removed from the solid mask
    gap_frac: float = 0.01  # a gutter row / column may hold this share of fg
    accept_score: float = 0.45
    max_region_frac: float = 0.40  # safety cap: no region larger than this share of the frame
    max_frame_edges: int = 3  # safety cap: no region touching this many frame edges
    max_group_frac: float = 0.60  # a tile group (what a tile hit covers) may be larger, never 3+ edges
    merge_iou: float = 0.5
    contain_frac: float = 0.8
    fallback_tiles: bool = True
    tile_px: int = 640  # fallback tile size (capture px), clamped to the area
    tile_overlap: float = 0.25
    max_tiles: int = 24
    max_regions: int = 96


@dataclass
class Leaf:
    """A candidate before acceptance (kept for debugging / tile fallback)."""

    work_xyxy: tuple[int, int, int, int]
    score: float
    feats: dict[str, float] = field(default_factory=dict)
    accepted: bool = False


@dataclass
class WindowView:
    """Window labels for one frame at the proposer's work scale.

    ``labels``: int32 raster of the **whole frame** at ``factor`` (capture px
    per work px); 0 = system UI / wallpaper / static toolbar band, k ≥ 1 =
    an app window. ``window_ids[k]`` is its id, ``bounds[k]`` its rect and
    ``panes[k]`` its scrolling content rect (capture px): pictures touching
    a pane edge are clipped there."""

    labels: np.ndarray
    factor: int
    window_ids: list[int | None]
    bounds: dict[int, Box] = field(default_factory=dict)
    panes: dict[int, Box] = field(default_factory=dict)
    # Work px of a back window under a front window's drop shadow (darkened
    # page colour there is still background, not a picture).
    shadow: np.ndarray | None = None

    def crop(self, rx1: int, ry1: int, shape: tuple[int, int], arr: np.ndarray | None = None) -> np.ndarray:
        f = self.factor
        src = self.labels if arr is None else arr
        sh, sw = shape[:2]
        y0, x0 = ry1 // f, rx1 // f
        out = np.zeros((sh, sw), src.dtype)
        part = src[y0 : y0 + sh, x0 : x0 + sw]
        out[: part.shape[0], : part.shape[1]] = part
        return out


def shadow_background(small: np.ndarray, shadow: np.ndarray, colors: list[tuple[int, int, int]]) -> np.ndarray:
    """In shadow pixels, a page colour darkened by a uniform factor (0.35–1)
    is background too. Returns a 0/255 mask of such pixels."""
    out = np.zeros(small.shape[:2], np.uint8)
    if not colors or not np.any(shadow):
        return out
    px = small[shadow].astype(np.float32)
    hit = np.zeros(px.shape[0], bool)
    for c in colors:
        cc = np.asarray(c, np.float32) + 1.0
        alpha = (px.sum(axis=1) + 3.0) / float(cc.sum())
        pred = alpha[:, None] * cc - 1.0
        hit |= (np.abs(px - pred).max(axis=1) <= 3.0 + 0.02 * float(cc.max())) & (alpha >= 0.35) & (alpha <= 1.02)
    out[shadow] = np.where(hit, 255, 0).astype(np.uint8)
    return out


def _borders(lab: np.ndarray) -> np.ndarray:
    b = np.zeros(lab.shape, bool)
    d = lab[:, 1:] != lab[:, :-1]
    b[:, 1:] |= d
    b[:, :-1] |= d
    d = lab[1:, :] != lab[:-1, :]
    b[1:, :] |= d
    b[:-1, :] |= d
    return b


def _mode_label(lab: np.ndarray, mask: np.ndarray | None = None) -> int:
    vals = lab[mask] if mask is not None and mask.any() else lab.reshape(-1)
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0
    return int(np.bincount(vals).argmax())


@dataclass
class ProposalDebug:
    factor: int = 1
    bg_colors: list[tuple[int, int, int]] = field(default_factory=list)
    leaves: list[Leaf] = field(default_factory=list)
    fg: np.ndarray | None = None
    solid: np.ndarray | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- utils


def work_factor(height: int, width: int, work_long: int, max_factor: int = 3) -> int:
    """Integer downscale; capped so an 8 px gutter keeps ≥ 2 clean work px."""
    f = int(round(max(height, width) / float(max(64, work_long))))
    return max(1, min(int(max_factor), f))


def downscale_int(bgr: np.ndarray, f: int) -> np.ndarray:
    """Integer-factor area downscale (fast path; trims the < f px remainder)."""
    if f <= 1:
        return bgr
    h, w = bgr.shape[:2]
    return cv2.resize(bgr[: (h // f) * f, : (w // f) * f], (w // f, h // f), interpolation=cv2.INTER_AREA)


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def estimate_background(
    small: np.ndarray, *, min_frac: float = 0.01, max_colors: int = 8
) -> list[tuple[int, int, int]]:
    """Dominant exactly-flat colours (page, cards, toolbars)."""
    sub = np.ascontiguousarray(small[::3, ::3])
    h, w = sub.shape[:2]
    dx = cv2.absdiff(sub[:, 1:], sub[:, :-1]).reshape(h, w - 1, 3).max(axis=2)
    dy = cv2.absdiff(sub[1:, :], sub[:-1, :]).reshape(h - 1, w, 3).max(axis=2)
    flat = np.ones((h, w), dtype=bool)
    flat[:, :-1] &= dx <= 1
    flat[:-1, :] &= dy <= 1
    px = sub[flat]
    if px.size == 0:
        return []
    # 15-bit bins find the candidates fast; exact colours inside the top bins
    # keep 250-grey page and 255-white cards apart (same 5-bit bin).
    q = px >> 3
    code = (q[:, 0].astype(np.int32) << 10) | (q[:, 1].astype(np.int32) << 5) | q[:, 2].astype(np.int32)
    counts = np.bincount(code, minlength=1 << 15)
    need = max(16, int(min_frac * sub.shape[0] * sub.shape[1]))
    found: list[tuple[int, tuple[int, int, int]]] = []
    for c in np.argsort(counts)[::-1][: max_colors * 2]:
        if counts[c] < need:
            break
        sel = px[code == c].astype(np.int32) & 7
        low = np.bincount((sel[:, 0] << 6) | (sel[:, 1] << 3) | sel[:, 2], minlength=512)
        base = ((int(c) >> 10) << 3, ((int(c) >> 5) & 31) << 3, (int(c) & 31) << 3)
        for v in np.flatnonzero(low >= need).tolist():
            found.append((int(low[v]), (base[0] | (v >> 6), base[1] | ((v >> 3) & 7), base[2] | (v & 7))))
    found.sort(key=lambda t: -t[0])
    return [c for _, c in found[:max_colors]]


def background_mask(img: np.ndarray, colors: list[tuple[int, int, int]], tol: int) -> np.ndarray:
    """255 where a pixel is within ``tol`` of any background colour."""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for c in colors:
        lo = np.array([max(0, v - tol) for v in c], dtype=np.uint8)
        hi = np.array([min(255, v + tol) for v in c], dtype=np.uint8)
        mask |= cv2.inRange(img, lo, hi)
    return mask


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` runs of True."""
    if flags.size == 0:
        return []
    d = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def xy_cut(
    mask: np.ndarray, min_w: int, min_h: int, *, gap_frac: float = 0.01, depth: int = 0
) -> list[tuple[int, int, int, int]]:
    """Recursive XY-cut of a binary mask. Returns tight leaf boxes (x1, y1, x2, y2).

    Segments smaller than ``min_w × min_h`` are dropped early: they cannot
    contain a picture, which is also what removes text lines.
    """
    if mask.size == 0:
        return []
    rows = np.count_nonzero(mask, axis=1)
    ys = np.flatnonzero(rows)
    if ys.size == 0:
        return []
    cols = np.count_nonzero(mask, axis=0)
    xs = np.flatnonzero(cols)
    y1, y2 = int(ys[0]), int(ys[-1]) + 1
    x1, x2 = int(xs[0]), int(xs[-1]) + 1
    if (x2 - x1) < min_w or (y2 - y1) < min_h:
        return []
    if depth > 24:
        return [(x1, y1, x2, y2)]
    rows = rows[y1:y2]
    cols = cols[x1:x2]
    out: list[tuple[int, int, int, int]] = []
    for axis, prof, span in ((0, rows, x2 - x1), (1, cols, y2 - y1)):
        busy = prof > max(0, int(gap_frac * span))
        segs = _runs(busy)
        if len(segs) <= 1:
            continue
        for a, b in segs:
            if axis == 0:
                sub = mask[y1 + a : y1 + b, x1:x2]
                for sx1, sy1, sx2, sy2 in xy_cut(sub, min_w, min_h, gap_frac=gap_frac, depth=depth + 1):
                    out.append((x1 + sx1, y1 + a + sy1, x1 + sx2, y1 + a + sy2))
            else:
                sub = mask[y1:y2, x1 + a : x1 + b]
                for sx1, sy1, sx2, sy2 in xy_cut(sub, min_w, min_h, gap_frac=gap_frac, depth=depth + 1):
                    out.append((x1 + a + sx1, y1 + sy1, x1 + a + sx2, y1 + sy2))
        return out
    return [(x1, y1, x2, y2)]


# ----------------------------------------------------------------------- scoring


def picture_features(crop_bgr: np.ndarray, fg: np.ndarray | None = None) -> dict[str, float]:
    """Cheap picture-vs-text/UI features on a (small-scale) crop."""
    crop = _to_bgr(crop_bgr)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    n = float(gray.size) or 1.0
    # Tone features on a 1–99 % contrast stretch: a pale / low-contrast photo
    # spreads out, while text stays bimodal (ink → 0, paper → 255).
    h256 = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    cdf = np.cumsum(h256) / n
    lo = int(np.searchsorted(cdf, 0.01))
    hi = int(np.searchsorted(cdf, 0.99))
    if hi - lo >= 4:
        lut = np.clip((np.arange(256) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        stretched = cv2.LUT(gray, lut)
    else:
        stretched = gray
    hist = cv2.calcHist([stretched], [0], None, [64], [0, 256]).ravel() / n
    nz = hist[hist > 0]
    entropy = float(-(nz * np.log2(nz)).sum())  # bits, 0..6
    # Bimodality: mass in the two dominant tones (± one bin). Text ≈ 1.
    top = np.argsort(hist)[::-1]
    taken = np.zeros(64, dtype=bool)
    for idx in top[:2]:
        taken[max(0, idx - 1) : idx + 2] = True
    modes2 = float(hist[taken].sum())
    b, g, r = (crop[..., i].astype(np.float32) for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    colour_sigma = float(math.sqrt(rg.var() + yb.var()))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(gx) + np.abs(gy)
    energy = float(mag.sum()) + 1e-6
    # Axis-aligned: one component dominates the other by 4× (≈ ±14°).
    ax = (np.abs(gx) > 4.0 * np.abs(gy)) | (np.abs(gy) > 4.0 * np.abs(gx))
    hv_frac = float(mag[ax].sum()) / energy
    edge_density = float(np.count_nonzero(mag > 40.0)) / n
    fill = float(np.count_nonzero(fg)) / float(fg.size) if fg is not None and fg.size else 1.0
    # Exactly-flat share: UI paint and text paper are flat, photos carry noise.
    c16 = crop.astype(np.int16)
    flat_x = np.abs(c16[:, 1:] - c16[:, :-1]).max(axis=2) <= 1 if crop.shape[1] > 1 else np.ones(1, bool)
    flat_y = np.abs(c16[1:, :] - c16[:-1, :]).max(axis=2) <= 1 if crop.shape[0] > 1 else np.ones(1, bool)
    flat = 0.5 * float(flat_x.mean()) + 0.5 * float(flat_y.mean())
    return {
        "entropy": entropy,
        "modes2": modes2,
        "colour": colour_sigma,
        "hv": hv_frac,
        "edges": edge_density,
        "fill": fill,
        "flat": flat,
        "std": float(gray.std()),
    }


def picture_score(feats: dict[str, float]) -> float:
    """0..1 picture-likeness from :func:`picture_features`."""

    def ramp(v: float, lo: float, hi: float) -> float:
        return float(min(1.0, max(0.0, (v - lo) / (hi - lo))))

    s_tone = ramp(feats["entropy"], 2.0, 4.0)
    s_mid = 1.0 - ramp(feats["modes2"], 0.45, 0.85)
    s_col = ramp(feats["colour"], 4.0, 20.0)
    s_orient = 1.0 - ramp(feats["hv"], 0.45, 0.85)
    s_fill = ramp(feats["fill"], 0.5, 0.9)
    s_std = ramp(feats["std"], 0.8, 3.0)
    s_tex = 1.0 - ramp(feats.get("flat", 0.0), 0.3, 0.8)
    score = 0.25 * s_tone + 0.20 * s_mid + 0.10 * s_col + 0.15 * s_orient + 0.10 * s_fill + 0.20 * s_tex
    score *= s_std
    # Smooth picture (pale product shot, soft-focus photo, gradient art): a
    # solid non-background rectangle with spread-out tones. Text / UI panels
    # fail on bimodality.
    if feats["fill"] >= 0.6 and feats["modes2"] <= 0.5 and feats["entropy"] >= 2.4 and feats["std"] >= 1.2:
        score = max(score, 0.5)
    return float(score)


# ------------------------------------------------------------------ geometry


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _inside_frac(a: Box, b: Box) -> float:
    """Share of ``a`` inside ``b``."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


def tile_grid(box: Box, tile: float, overlap: float = 0.25) -> list[Box]:
    """Tiles of ``tile`` px covering ``box`` with at least ``overlap`` overlap."""
    x1, y1, x2, y2 = box
    out: list[Box] = []

    def starts(a: float, b: float) -> list[float]:
        span = b - a
        if span <= tile:
            return [a]
        step = tile * (1.0 - overlap)
        n = int(math.ceil((span - tile) / step)) + 1
        real = (span - tile) / (n - 1)
        return [a + i * real for i in range(n)]

    t = float(tile)
    for ty in starts(y1, y2):
        for tx in starts(x1, x2):
            out.append((tx, ty, min(x2, tx + t), min(y2, ty + t)))
    return out


def refine_edges(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    colors: list[tuple[int, int, int]],
    tol: int,
    reach: int,
) -> tuple[int, int, int, int]:
    """Snap each side to the full-res bg→picture transition within ±``reach``."""
    if not colors:
        return box
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    iy1, iy2 = min(y2 - 1, y1 + reach), max(y1 + 1, y2 - reach)
    ix1, ix2 = min(x2 - 1, x1 + reach), max(x1 + 1, x2 - reach)

    def col_frac(xa: int, xb: int) -> np.ndarray:
        strip = frame[iy1:iy2, xa:xb]
        return 1.0 - background_mask(strip, colors, tol).mean(axis=0) / 255.0

    def row_frac(ya: int, yb: int) -> np.ndarray:
        strip = frame[ya:yb, ix1:ix2]
        return 1.0 - background_mask(strip, colors, tol).mean(axis=1) / 255.0

    a, b = max(0, x1 - reach), min(W, x1 + reach + 1)
    if b > a and iy2 > iy1:
        idx = np.flatnonzero(col_frac(a, b) >= 0.5)
        if idx.size:
            x1 = a + int(idx[0])
    a, b = max(0, x2 - reach - 1), min(W, x2 + reach)
    if b > a and iy2 > iy1:
        idx = np.flatnonzero(col_frac(a, b) >= 0.5)
        if idx.size:
            x2 = a + int(idx[-1]) + 1
    a, b = max(0, y1 - reach), min(H, y1 + reach + 1)
    if b > a and ix2 > ix1:
        idx = np.flatnonzero(row_frac(a, b) >= 0.5)
        if idx.size:
            y1 = a + int(idx[0])
    a, b = max(0, y2 - reach - 1), min(H, y2 + reach)
    if b > a and ix2 > ix1:
        idx = np.flatnonzero(row_frac(a, b) >= 0.5)
        if idx.size:
            y2 = a + int(idx[-1]) + 1
    if x2 - x1 < 2 or y2 - y1 < 2:
        return box
    return x1, y1, x2, y2


def region_key(dhash: int, w: float, h: float) -> str:
    bw, bh = size_bucket(w, h)
    return f"p{dhash:016x}-{bw}x{bh}"


def make_region(
    frame: np.ndarray, xyxy: Box, *, kind: str = KIND_PICTURE, score: float = 0.0
) -> PictureRegion:
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    crop = frame[y1:y2, x1:x2]
    dh, ah = hash_crop(crop)
    rid = region_key(dh, x2 - x1, y2 - y1)
    if kind == KIND_TILE:
        rid = f"t{x1}_{y1}_{x2 - x1}x{y2 - y1}"
    return PictureRegion(
        region_id=rid,
        xyxy=(float(x1), float(y1), float(x2), float(y2)),
        kind=kind,
        picture_score=float(score),
        content_hash=dh,
        ahash=ah,
    )


def dedupe_ids(regions: list[PictureRegion]) -> list[PictureRegion]:
    """Same picture twice on one frame → ``id~1``, ``id~2`` (top-left first)."""
    seen: dict[str, int] = {}
    for reg in sorted(regions, key=lambda r: (r.xyxy[1], r.xyxy[0])):
        n = seen.get(reg.region_id, 0)
        seen[reg.region_id] = n + 1
        if n:
            reg.region_id = f"{reg.region_id}~{n}"
    return regions


def clip_line_rows(small: np.ndarray, fg: np.ndarray, *, min_frac: float = 0.85, max_lines: int = 8) -> list[int]:
    """Rows (work px) that are long uniform lines: toolbar / pane borders that
    content scrolls under. Frame edges are handled separately."""
    h, w = fg.shape[:2]
    busy = np.count_nonzero(fg, axis=1) >= int(min_frac * w)
    rows = np.flatnonzero(busy)
    if rows.size == 0:
        return []
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    out: list[int] = []
    for r in rows.tolist():
        runs = _runs(fg[r] > 0)
        if not runs or max(b - a for a, b in runs) < min_frac * w:
            continue  # card borders etc. have gaps; a toolbar / pane line does not
        vals = gray[r][fg[r] > 0]
        if vals.size and float(vals.std()) <= 4.0:
            out.append(int(r))
            if len(out) >= max_lines:
                break
    return out


def _clip_sides(cap: Box, pane: Box, lines_cap: list[float], tol: float) -> str:
    """Sides of ``cap`` cut by its pane (the frame, or its window's scrolling
    content rect) or by a toolbar line."""
    x1, y1, x2, y2 = cap
    px1, py1, px2, py2 = pane
    sides = ""
    if y1 <= py1 + tol or any(abs(y1 - ly) <= tol for ly in lines_cap):
        sides += "t"
    if y2 >= py2 - tol or any(abs(y2 - ly) <= tol for ly in lines_cap):
        sides += "b"
    if x1 <= px1 + tol:
        sides += "l"
    if x2 >= px2 - tol:
        sides += "r"
    return sides


# ------------------------------------------------------------------ proposer


class RegionProposer:
    """Stateless per call except for the remembered page background colours
    (used when proposing inside a small ``roi``)."""

    def __init__(self, opts: RegionOptions | None = None) -> None:
        self.opts = opts or RegionOptions()
        self.bg_colors: list[tuple[int, int, int]] = []
        self.last = ProposalDebug()

    def sliver_side(self, frame_width: int) -> float:
        o = self.opts
        return max(float(o.sliver_floor), o.sliver_px * min(1.0, frame_width / float(o.ref_width)))

    def min_side(self, frame_width: int) -> float:
        o = self.opts
        return max(float(o.min_side_floor), o.min_side_px * min(1.0, frame_width / float(o.ref_width)))

    def propose(
        self,
        frame_bgr: np.ndarray,
        roi: Box | None = None,
        *,
        debug: bool = False,
        clip_ys: Sequence[float] = (),
        content_y: tuple[float | None, float | None] = (None, None),
        windows: WindowView | None = None,
    ) -> list[PictureRegion]:
        """``clip_ys``: extra horizontal clip lines in capture px (y where
        scrolled content starts / ends, e.g. under a static toolbar).
        ``content_y``: (top, bottom) of the scrolling content; static bands
        outside it (toolbars) are never proposed. ``windows``: window labels
        (see :class:`WindowView`); then only app-window pixels are looked at
        and ``content_y`` is per window (encoded in the labels / panes)."""
        import time

        o = self.opts
        t0 = time.perf_counter()
        frame = _to_bgr(frame_bgr)
        H, W = frame.shape[:2]
        rx1, ry1, rx2, ry2 = 0, 0, W, H
        if roi is not None:
            rx1 = max(0, int(math.floor(roi[0])))
            ry1 = max(0, int(math.floor(roi[1])))
            rx2 = min(W, int(math.ceil(roi[2])))
            ry2 = min(H, int(math.ceil(roi[3])))
        if rx2 - rx1 < self.min_side(W) or ry2 - ry1 < self.min_side(W):
            self.last = ProposalDebug()
            return []
        full = (rx1, ry1, rx2, ry2) == (0, 0, W, H)
        f = work_factor(H, W, o.work_long)
        if windows is not None and windows.factor != f:
            windows = None  # built for another scale: ignore rather than misalign
        min_side = self.min_side(W)
        view = frame[ry1:ry2, rx1:rx2]
        small = downscale_int(view, f)
        t_scale = time.perf_counter()
        lab = windows.crop(rx1, ry1, small.shape) if windows is not None else None
        colors = estimate_background(small if lab is None else _masked_for_bg(small, lab), min_frac=o.bg_min_frac, max_colors=o.max_bg_colors)
        if full:
            self.bg_colors = list(colors)
        else:
            for c in self.bg_colors:
                if c not in colors and len(colors) < o.max_bg_colors + 4:
                    colors.append(c)
        bg = background_mask(small, colors, o.bg_tol)
        if windows is not None and windows.shadow is not None:
            bg |= shadow_background(small, windows.crop(rx1, ry1, small.shape, windows.shadow), colors)
        fg = cv2.bitwise_not(bg)
        if lab is not None:
            fg[lab == 0] = 0  # wallpaper, menu bar, Dock, shadows, toolbar bands
        else:
            ctop, cbot = content_y
            if ctop is not None:
                fg[: max(0, int(math.ceil((ctop - ry1) / f)))] = 0
            if cbot is not None:
                fg[max(0, int(math.floor((cbot - ry1) / f))) :] = 0
        k = max(3, int(round(o.solid_px / f)))
        solid = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        if lab is not None:
            solid[(lab == 0) | _borders(lab)] = 0  # components never cross a window border
        t_masks = time.perf_counter()

        min_w = max(4, int(math.floor(min_side / f)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
        leaves: list[Leaf] = []
        leaf_lab: list[int] = []
        kernel3 = np.ones((3, 3), np.uint8)
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if w < min_w or h < min_w:
                continue
            comp = (labels[y : y + h, x : x + w] == i).astype(np.uint8)
            wl = _mode_label(lab[y : y + h, x : x + w], comp > 0) if lab is not None else 0
            comp = cv2.dilate(comp, kernel3)
            if lab is not None:
                comp[lab[y : y + h, x : x + w] != wl] = 0
            prof = cv2.bitwise_and(fg[y : y + h, x : x + w], fg[y : y + h, x : x + w], mask=comp)
            for lx1, ly1, lx2, ly2 in xy_cut(prof > 0, min_w, min_w, gap_frac=o.gap_frac):
                gx1, gy1, gx2, gy2 = x + lx1, y + ly1, x + lx2, y + ly2
                feats = picture_features(small[gy1:gy2, gx1:gx2], fg[gy1:gy2, gx1:gx2])
                leaves.append(Leaf((gx1, gy1, gx2, gy2), picture_score(feats), feats))
                leaf_lab.append(wl)
        t_leaves = time.perf_counter()

        boxes: list[tuple[Box, float, int]] = []
        big: list[tuple[Box, int]] = []  # picture-like but over the region cap
        for leaf, wl in zip(leaves, leaf_lab):
            if leaf.score < o.accept_score:
                continue
            lx1, ly1, lx2, ly2 = leaf.work_xyxy
            cap = (rx1 + lx1 * f, ry1 + ly1 * f, rx1 + lx2 * f, ry1 + ly2 * f)
            cap = (min(cap[0], W - 1), min(cap[1], H - 1), min(cap[2], W), min(cap[3], H))
            # Work px at the roi / frame border are complete if the leaf touches it.
            if lx2 * f >= (rx2 - rx1) - f:
                cap = (cap[0], cap[1], rx2, cap[3])
            if ly2 * f >= (ry2 - ry1) - f:
                cap = (cap[0], cap[1], cap[2], ry2)
            cap = refine_edges(frame, cap, colors, o.bg_tol, reach=f + 1)
            cap = self._in_window(cap, wl, windows)
            if cap is None:
                continue
            w, h = cap[2] - cap[0], cap[3] - cap[1]
            if min(w, h) < min_side:
                continue
            if not self._passes_caps(cap, W, H, f):
                if self._passes_caps(cap, W, H, f, max_frac=o.max_group_frac):
                    # Too big to be one picture region (the cap), small enough to
                    # be a hero image: looked at as one tile whose hit covers it.
                    big.append((tuple(float(v) for v in cap), wl))
                continue
            leaf.accepted = True
            boxes.append((tuple(float(v) for v in cap), leaf.score, wl))
        # Clip lines (frame edges + long uniform rows such as a toolbar border).
        line_rows = clip_line_rows(small, fg)
        for cy in clip_ys:
            r = int(round((float(cy) - ry1) / f)) - 1
            if 0 <= r < small.shape[0] - 1:
                line_rows = [q for q in line_rows if abs(q - r) > 2] + [r]
        lines_cap = [float(ry1 + (r + 1) * f) for r in line_rows] + [float(ry1 + r * f) for r in line_rows]
        for b, sc, wl in self._edge_slivers(frame, small, fg, f, (rx1, ry1), line_rows, colors, min_side, boxes, lab, windows):
            boxes.append((b, sc, wl))
        boxes = self._merge(boxes)
        boxes = boxes[: o.max_regions]
        regions = []
        frame_pane = (0.0, 0.0, float(W), float(H))
        for b, sc, wl in boxes:
            reg = make_region(frame, b, score=sc)
            pane = frame_pane
            if windows is not None and wl > 0:
                reg.window_id = windows.window_ids[wl] if wl < len(windows.window_ids) else None
                pane = windows.panes.get(wl) or windows.bounds.get(wl) or frame_pane
            reg.clipped = _clip_sides(reg.xyxy, pane, lines_cap, tol=float(2 * f + 1))
            regions.append(reg)
        t_regions = time.perf_counter()

        for b, wl in big:
            reg = make_region(frame, b, kind=KIND_TILE, score=0.0)
            reg.group_xyxy = b
            if windows is not None and 0 < wl < len(windows.window_ids):
                reg.window_id = windows.window_ids[wl]
            regions.append(reg)
        if o.fallback_tiles:
            regions.extend(
                self._fallback_tiles(frame, small, solid, f, (rx1, ry1), [b for b, _, _ in boxes] + [b for b, _ in big],
                                     min_side, lab, windows)
            )
        dedupe_ids(regions)
        t_end = time.perf_counter()
        self.last = ProposalDebug(
            factor=f,
            bg_colors=colors,
            leaves=leaves,
            fg=fg if debug else None,
            solid=solid if debug else None,
            timings_ms={
                "scale": (t_scale - t0) * 1e3,
                "masks": (t_masks - t_scale) * 1e3,
                "leaves": (t_leaves - t_masks) * 1e3,
                "regions": (t_regions - t_leaves) * 1e3,
                "tiles": (t_end - t_regions) * 1e3,
                "total": (t_end - t0) * 1e3,
            },
        )
        return regions

    # ------------------------------------------------------------ caps

    @staticmethod
    def _in_window(cap: Box, wl: int, windows: WindowView | None) -> Box | None:
        """Clip a box to its window's bounds (a region never crosses a window
        border); None when it has no window (system UI)."""
        if windows is None:
            return cap
        b = windows.bounds.get(wl)
        if wl <= 0 or b is None:
            return None
        x1, y1 = max(cap[0], b[0]), max(cap[1], b[1])
        x2, y2 = min(cap[2], b[2]), min(cap[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _passes_caps(self, cap: Box, W: int, H: int, f: int, max_frac: float | None = None) -> bool:
        """Safety caps, whatever the score: not most of the frame, not
        touching 3+ frame edges (the whole desktop, a wallpaper)."""
        o = self.opts
        w, h = cap[2] - cap[0], cap[3] - cap[1]
        if w * h > (o.max_region_frac if max_frac is None else max_frac) * float(W * H):
            return False
        tol = float(f + 1)
        edges = int(cap[0] <= tol) + int(cap[1] <= tol) + int(cap[2] >= W - tol) + int(cap[3] >= H - tol)
        return edges < o.max_frame_edges

    def _edge_slivers(
        self,
        frame: np.ndarray,
        small: np.ndarray,
        fg: np.ndarray,
        f: int,
        origin: tuple[int, int],
        line_rows: list[int],
        colors: list[tuple[int, int, int]],
        min_side: float,
        accepted: list[tuple[Box, float, int]],
        lab: np.ndarray | None = None,
        windows: WindowView | None = None,
    ) -> list[tuple[Box, float, int]]:
        """Pictures scrolled mostly out of view: thin strips hugging the frame's
        top / bottom edge, a toolbar line or a window's content edge, too thin
        for the solid mask."""
        o = self.opts
        H, W = frame.shape[:2]
        sh, sw = fg.shape[:2]
        ox, oy = origin
        min_long = max(4, int(math.floor(min_side / f)))
        s0 = max(2, int(math.ceil(self.sliver_side(W) / f)))
        band = max(s0 + 1, int(math.ceil(min_side / f)) + 2)
        starts: list[tuple[int, int, int, int]] = []  # (first row, direction, col lo, col hi)
        if oy == 0:
            starts.append((0, 1, 0, sw))
        if oy + sh * f >= H - f:
            starts.append((sh - 1, -1, 0, sw))
        rows_sorted = sorted(set(line_rows))
        groups: list[list[int]] = []
        for r in rows_sorted:
            if groups and r - groups[-1][-1] <= 1:
                groups[-1].append(r)
            else:
                groups.append([r])
        for g in groups:  # a thick line is one line
            starts.append((g[-1] + 1, 1, 0, sw))
            starts.append((g[0] - 1, -1, 0, sw))
        if windows is not None:
            for wl, pane in windows.panes.items():
                c_lo = max(0, int(math.ceil((pane[0] - ox) / f)))
                c_hi = min(sw, int(math.floor((pane[2] - ox) / f)))
                if c_hi - c_lo < min_long:
                    continue
                starts.append((int(math.ceil((pane[1] - oy) / f)), 1, c_lo, c_hi))
                starts.append((int(math.floor((pane[3] - oy) / f)) - 1, -1, c_lo, c_hi))
        out: list[tuple[Box, float, int]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for r0, d, c_lo, c_hi in starts:
            if not (0 <= r0 < sh) or (r0, d, c_lo, c_hi) in seen:
                continue
            seen.add((r0, d, c_lo, c_hi))
            rows = list(range(r0, r0 + d * band, d))
            rows = [r for r in rows if 0 <= r < sh]
            if len(rows) < s0:
                continue
            lo, hi = min(rows), max(rows) + 1
            strip = fg[lo:hi, c_lo:c_hi] > 0
            head = strip[: s0] if d > 0 else strip[-s0:]
            busy = head.mean(axis=0) >= 0.8
            for a, b in _runs(busy):
                if b - a < min_long:
                    continue
                cols = strip[:, a:b].mean(axis=1) >= 0.8
                order = cols if d > 0 else cols[::-1]
                depth = int(np.argmin(order)) if not order.all() else len(order)
                if depth < s0 or depth >= band - 1:
                    continue  # too thin, or a full-size picture (main path)
                a, b = a + c_lo, b + c_lo
                y_a = lo if d > 0 else hi - depth
                y_b = lo + depth if d > 0 else hi
                wl = _mode_label(lab[y_a:y_b, a:b]) if lab is not None else 0
                if lab is not None and wl == 0:
                    continue
                feats = picture_features(small[y_a:y_b, a:b], fg[y_a:y_b, a:b])
                score = picture_score(feats)
                if score < o.accept_score:
                    continue
                cap = (float(ox + a * f), float(oy + y_a * f), float(min(W, ox + b * f)), float(min(H, oy + y_b * f)))
                if d < 0 and y_b >= sh:
                    cap = (cap[0], cap[1], cap[2], float(H if oy + sh * f >= H - f else cap[3]))
                if any(_inside_frac(cap, acc) >= 0.5 for acc, _, _ in accepted + out):
                    continue
                cap = refine_edges(frame, tuple(int(v) for v in cap), colors, o.bg_tol, reach=f + 1)
                cap = self._in_window(tuple(float(v) for v in cap), wl, windows)
                if cap is None or not self._passes_caps(cap, W, H, f):
                    continue
                out.append((tuple(float(v) for v in cap), score, wl))
        return out

    def _merge(self, boxes: list[tuple[Box, float, int]]) -> list[tuple[Box, float, int]]:
        """Union near-duplicates, drop boxes mostly inside a bigger one. A box
        holding two or more accepted pictures is a container (a card, a
        collage frame, the desktop): keep the pictures, drop the container.
        Boxes in different windows never merge."""
        o = self.opts

        def area(b: Box) -> float:
            return (b[2] - b[0]) * (b[3] - b[1])

        items = sorted(boxes, key=lambda bs: -area(bs[0]))
        containers = set()
        for i, (box, _s, wl) in enumerate(items):
            kids = [
                j
                for j in range(i + 1, len(items))
                if items[j][2] == wl and _inside_frac(items[j][0], box) >= o.contain_frac and area(items[j][0]) < 0.8 * area(box)
            ]
            # Kids of one container may overlap each other (duplicates): count distinct ones.
            distinct: list[Box] = []
            for j in kids:
                if not any(_iou(items[j][0], d) >= o.merge_iou for d in distinct):
                    distinct.append(items[j][0])
            if len(distinct) >= 2:
                containers.add(i)
        items = [it for i, it in enumerate(items) if i not in containers]
        changed = True
        while changed:
            changed = False
            out: list[tuple[Box, float, int]] = []
            for box, score, wl in items:
                merged = False
                for j, (kb, ks, kl) in enumerate(out):
                    if kl != wl:
                        continue
                    if _iou(box, kb) >= o.merge_iou:
                        out[j] = (
                            (min(box[0], kb[0]), min(box[1], kb[1]), max(box[2], kb[2]), max(box[3], kb[3])),
                            max(score, ks),
                            kl,
                        )
                        merged = changed = True
                        break
                    if _inside_frac(box, kb) >= o.contain_frac:
                        out[j] = (kb, max(score, ks), kl)
                        merged = True
                        break
                if not merged:
                    out.append((box, score, wl))
            items = out
        return items

    def _fallback_tiles(
        self,
        frame: np.ndarray,
        small: np.ndarray,
        solid: np.ndarray,
        f: int,
        origin: tuple[int, int],
        accepted: list[Box],
        min_side: float,
        lab: np.ndarray | None = None,
        windows: WindowView | None = None,
    ) -> list[PictureRegion]:
        """Overlapping tiles over solid, textured cells no region covers."""
        o = self.opts
        cell = max(8, int(round(min_side / f)))
        sh, sw = solid.shape[:2]
        gh, gw = sh // cell, sw // cell
        if gh < 1 or gw < 1:
            return []
        trim = solid[: gh * cell, : gw * cell]
        sfrac = cv2.resize(trim, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        gray = cv2.cvtColor(small[: gh * cell, : gw * cell], cv2.COLOR_BGR2GRAY).astype(np.float32)
        m1 = cv2.resize(gray, (gw, gh), interpolation=cv2.INTER_AREA)
        m2 = cv2.resize(gray * gray, (gw, gh), interpolation=cv2.INTER_AREA)
        std = np.sqrt(np.maximum(0.0, m2 - m1 * m1))
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        tex = cv2.resize(lap, (gw, gh), interpolation=cv2.INTER_AREA)
        cand = (sfrac >= 0.9) & ((std >= 4.0) | (tex >= 2.0))
        if not np.any(cand):
            return []
        ox, oy = origin
        covered = np.zeros_like(cand)
        for b in accepted:
            cx1 = int((b[0] - ox) / f / cell)
            cy1 = int((b[1] - oy) / f / cell)
            cx2 = int(math.ceil((b[2] - ox) / f / cell))
            cy2 = int(math.ceil((b[3] - oy) / f / cell))
            covered[max(0, cy1) : max(0, cy2), max(0, cx1) : max(0, cx2)] = True
        cand &= ~covered
        H, W = frame.shape[:2]
        # Cell labels: a group never spans two windows.
        if lab is not None:
            cell_lab = lab[cell // 2 : gh * cell : cell, cell // 2 : gw * cell : cell][:gh, :gw]
            cand &= cell_lab > 0
            groups = [(int(k), cand & (cell_lab == k)) for k in np.unique(cell_lab[cand])]
        else:
            groups = [(0, cand)]
        tiles: list[PictureRegion] = []
        for wl, cmask in groups:
            n, _labels, stats, _ = cv2.connectedComponentsWithStats(cmask.astype(np.uint8), connectivity=8)
            for i in range(1, n):
                x, y, w, h, area = (int(v) for v in stats[i])
                if area < 2:
                    continue
                # Grow by one cell (partial border cells fail the 90 % test), then
                # shrink to the solid pixels inside, so tiles reach the picture edge.
                gx1, gy1 = max(0, (x - 1) * cell), max(0, (y - 1) * cell)
                gx2, gy2 = min(sw, (x + w + 1) * cell), min(sh, (y + h + 1) * cell)
                sub = solid[gy1:gy2, gx1:gx2] > 0
                if lab is not None:
                    sub &= lab[gy1:gy2, gx1:gx2] == wl
                ys, xs = np.nonzero(sub)
                if xs.size == 0:
                    continue
                box = (
                    float(ox + (gx1 + int(xs.min())) * f),
                    float(oy + (gy1 + int(ys.min())) * f),
                    float(min(W, ox + (gx1 + int(xs.max()) + 1) * f)),
                    float(min(H, oy + (gy1 + int(ys.max()) + 1) * f)),
                )
                box = self._in_window(box, wl, windows) if windows is not None else box
                if box is None:
                    continue
                # A tile hit covers its group (AGGRESSIVE), e.g. a full-bleed hero
                # too big to be one proposal. A group touching 3+ frame edges or
                # over ``max_group_frac`` is no picture (the desktop, a
                # wallpaper): then a hit covers just the tile.
                group = box if self._passes_caps(box, W, H, f, max_frac=o.max_group_frac) else None
                short = min(box[2] - box[0], box[3] - box[1])
                size = float(min(o.tile_px, max(min_side * 2, short)))
                for t in tile_grid(box, size, o.tile_overlap):
                    reg = make_region(frame, t, kind=KIND_TILE, score=0.0)
                    reg.group_xyxy = group
                    if windows is not None and 0 < wl < len(windows.window_ids):
                        reg.window_id = windows.window_ids[wl]
                    tiles.append(reg)
                    if len(tiles) >= o.max_tiles:
                        return tiles
        return tiles


def _masked_for_bg(small: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Background colours from app-window pixels only (label 0 → black, which
    is flat and would otherwise become a page colour; it is not used)."""
    if not np.any(lab > 0):
        return small
    ys, xs = np.nonzero(lab > 0)
    y1, y2, x1, x2 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    return np.ascontiguousarray(small[y1:y2, x1:x2])


def propose_regions(
    frame_bgr: np.ndarray, roi: Box | None = None, opts: RegionOptions | None = None
) -> list[PictureRegion]:
    """One-shot helper around :class:`RegionProposer`."""
    return RegionProposer(opts).propose(frame_bgr, roi)
