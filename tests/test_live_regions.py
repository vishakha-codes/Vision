"""Picture-region proposals on synthetic screenshots (cv2 drawing only)."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import pytest

from vision_core.live.cache import hamming, hash_crop
from vision_core.live.regions import (
    KIND_PICTURE,
    KIND_TILE,
    RegionOptions,
    RegionProposer,
    _iou,
    estimate_background,
    picture_features,
    picture_score,
    tile_grid,
    xy_cut,
)
from vision_core.live.synth import (
    SynthPage,
    draw_paragraph,
    make_page,
    make_ui_page,
    photo_texture,
    scroll_view,
)


@lru_cache(maxsize=None)
def _page(seed: int, dark: bool = False, gutter: int | None = None) -> SynthPage:
    return make_page(seed=seed, dark=dark, gutter=gutter)


@lru_cache(maxsize=None)
def _ui(seed: int, dark: bool = False) -> SynthPage:
    return make_ui_page(seed=seed, dark=dark)


def _pictures(regs):
    return [r for r in regs if r.kind != KIND_TILE]


def _recall(regs, gts, thr=0.7) -> float:
    pics = _pictures(regs)
    hit = sum(1 for g in gts if max([_iou(r.xyxy, g) for r in pics] or [0.0]) >= thr)
    return hit / max(1, len(gts))


def _overlap_frac(a, b) -> float:
    """Share of ``a`` covered by ``b``."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def test_recall_on_desktop_pages_with_varied_gutters() -> None:
    proposer = RegionProposer()
    hits = total = 0
    for seed, (dark, gutter) in enumerate([(False, 8), (True, 12), (False, 16), (True, 24), (False, 10), (False, 20)]):
        page = _page(seed, dark, gutter)
        assert len(page.pictures) >= 20
        regs = proposer.propose(page.image)
        pics = _pictures(regs)
        for g in page.pictures:
            total += 1
            hits += max([_iou(r.xyxy, g) for r in pics] or [0.0]) >= 0.7
    assert hits / total >= 0.95, f"recall {hits}/{total}"


def test_text_blocks_and_ui_are_not_proposed() -> None:
    proposer = RegionProposer()
    for page in [_page(0), _page(1, True, 12), _ui(0), _ui(1, True), _ui(2)]:
        regs = _pictures(proposer.propose(page.image))
        for reg in regs:
            # A proposal is never mostly text / UI.
            on_text = max((_overlap_frac(reg.xyxy, t) for t in page.texts), default=0.0)
            on_pic = max((_overlap_frac(reg.xyxy, g) for g in page.pictures), default=0.0)
            assert on_text < 0.5 or on_pic > 0.5, (reg.xyxy, on_text)
        for t in page.texts:
            assert all(_iou(r.xyxy, t) < 0.3 for r in regs), t


def test_ui_page_positives_found_and_whole_screen_never_proposed() -> None:
    proposer = RegionProposer()
    page = _ui(0)
    regs = proposer.propose(page.image)
    H, W = page.image.shape[:2]
    assert all((r.xyxy[2] - r.xyxy[0]) * (r.xyxy[3] - r.xyxy[1]) < 0.5 * W * H for r in regs)
    assert _recall(regs, page.pictures) >= 0.75  # flat cartoon may fragment


def test_text_only_page_proposes_nothing() -> None:
    rng = np.random.default_rng(3)
    img = np.full((2234, 3456, 3), 250, np.uint8)
    for col in range(3):
        draw_paragraph(img, (80 + col * 1130, 120, 80 + col * 1130 + 1000, 2150), rng)
    regs = RegionProposer().propose(img)
    assert _pictures(regs) == []


def test_fullscreen_picture_is_never_one_region_but_tiles_cover_it() -> None:
    """Safety cap: no region over 40 % of the frame / touching 3+ edges,
    whatever its score (that was the whole-desktop blur). A full-screen
    picture is still looked at, through fallback tiles whose hit covers only
    the tile (the group would be most of the frame)."""
    rng = np.random.default_rng(5)
    img = photo_texture(1117, 1728, rng, palette="vivid")
    regs = RegionProposer().propose(img)
    assert _pictures(regs) == []
    tiles = [r for r in regs if r.kind == KIND_TILE]
    assert tiles and all(r.group_xyxy is None for r in tiles)
    canvas = np.zeros((1117, 1728), bool)
    for t in tiles:
        x1, y1, x2, y2 = (int(v) for v in t.xyxy)
        canvas[y1:y2, x1:x2] = True
    assert canvas.mean() >= 0.9


@pytest.mark.parametrize("gutter", [8, 12, 24])
def test_thumbnail_grid_splits_along_gutters(gutter: int) -> None:
    rng = np.random.default_rng(gutter)
    img = np.full((1400, 2400, 3), 255, np.uint8)
    gts = []
    tw, th = 260, 200
    for r in range(4):
        for c in range(6):
            x, y = 100 + c * (tw + gutter), 100 + r * (th + gutter)
            img[y : y + th, x : x + tw] = photo_texture(th, tw, rng)
            gts.append((x, y, x + tw, y + th))
    regs = RegionProposer().propose(img)
    assert _recall(regs, gts, 0.8) >= 0.95
    assert len(_pictures(regs)) == len(gts)


def test_pictures_clipped_at_frame_edges() -> None:
    rng = np.random.default_rng(9)
    img = np.full((1117, 1728, 3), 250, np.uint8)
    gts = [(1500, 300, 1728, 700), (300, 900, 800, 1117), (0, 0, 400, 260)]
    for x1, y1, x2, y2 in gts:
        img[y1:y2, x1:x2] = photo_texture(y2 - y1, x2 - x1, rng)
    regs = RegionProposer().propose(img)
    assert _recall(regs, gts, 0.85) == 1.0


def test_low_contrast_and_text_overlay_pictures() -> None:
    rng = np.random.default_rng(11)
    img = np.full((1117, 1728, 3), 250, np.uint8)
    gts = [(100, 100, 500, 400), (600, 100, 1000, 400), (1100, 100, 1600, 500)]
    img[100:400, 100:500] = photo_texture(300, 400, rng, palette="light")
    img[100:400, 600:1000] = photo_texture(300, 400, rng, palette="gray", text_overlay=True)
    img[100:500, 1100:1600] = photo_texture(400, 500, rng, palette="dark", text_overlay=True)
    draw_paragraph(img, (100, 600, 1600, 1000), rng)
    regs = RegionProposer().propose(img)
    assert _recall(regs, gts) == 1.0


def test_region_ids_and_hashes_stable_across_scroll() -> None:
    page = make_page(3456, 2434, seed=4)
    a = RegionProposer().propose(scroll_view(page.image, 0, 2234))
    b = RegionProposer().propose(scroll_view(page.image, 40, 2234))
    ids_a = {r.region_id: r for r in _pictures(a) if r.xyxy[1] > 60 and r.xyxy[3] < 2234 - 60}
    ids_b = {r.region_id: r for r in _pictures(b)}
    same = [rid for rid in ids_a if rid in ids_b]
    assert len(same) >= 0.8 * len(ids_a)
    for rid in same:
        assert ids_b[rid].xyxy[1] == pytest.approx(ids_a[rid].xyxy[1] - 40, abs=1.0)


def test_roi_proposal_only_inside_roi() -> None:
    page = _page(2)
    roi = (1000.0, 150.0, 2500.0, 1400.0)
    regs = RegionProposer().propose(page.image, roi)
    assert regs
    for r in regs:
        assert r.xyxy[0] >= roi[0] - 1 and r.xyxy[2] <= roi[2] + 1
        assert r.xyxy[1] >= roi[1] - 1 and r.xyxy[3] <= roi[3] + 1
    inside = [g for g in page.pictures if g[0] >= roi[0] and g[2] <= roi[2] and g[1] >= roi[1] and g[3] <= roi[3]]
    assert _recall(regs, inside) >= 0.9


def test_tile_fallback_covers_collage_with_overlap() -> None:
    # Abutting pictures (no gutters) mixed with a UI strip the scorer rejects:
    # whatever is not accepted as a picture is still covered by tiles.
    rng = np.random.default_rng(2)
    img = np.full((1400, 2400, 3), 250, np.uint8)
    img[200:1200, 300:2100] = photo_texture(1000, 1800, rng)
    opts = RegionOptions(accept_score=1.1)  # force every leaf to be rejected
    regs = RegionProposer(opts).propose(img)
    tiles = [r for r in regs if r.kind == KIND_TILE]
    assert tiles and not _pictures(regs)
    cover = np.zeros(img.shape[:2], bool)
    for t in tiles:
        x1, y1, x2, y2 = (int(v) for v in t.xyxy)
        cover[y1:y2, x1:x2] = True
    assert cover[200:1200, 300:2100].mean() >= 0.97


def test_tile_grid_overlap_at_least_25pct() -> None:
    tiles = tile_grid((0, 0, 2000, 900), 640, 0.25)
    xs = sorted({t[0] for t in tiles})
    for a, b in zip(xs, xs[1:]):
        assert (a + 640 - b) / 640 >= 0.25 - 1e-6
    assert max(t[2] for t in tiles) == 2000 and max(t[3] for t in tiles) == 900


def test_xy_cut_splits_gutters_and_drops_small() -> None:
    m = np.zeros((100, 200), bool)
    m[10:60, 10:90] = True
    m[10:60, 92:190] = True  # 2 px gutter
    m[80:84, 10:190] = True  # a "text line": too short
    leaves = xy_cut(m, 30, 30)
    assert sorted(leaves) == [(10, 10, 90, 60), (92, 10, 190, 60)]


def test_background_estimate_separates_white_and_grey() -> None:
    img = np.full((300, 300, 3), 250, np.uint8)
    img[50:250, 50:250] = 255
    colors = estimate_background(img)
    assert (250, 250, 250) in colors and (255, 255, 255) in colors


def test_picture_score_separates_photo_from_text_panel() -> None:
    rng = np.random.default_rng(1)
    photo = photo_texture(120, 160, rng)
    panel = np.full((120, 160, 3), (90, 40, 160), np.uint8)
    cv2.putText(panel, "Sale", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    assert picture_score(picture_features(photo)) >= 0.6
    assert picture_score(picture_features(panel)) < 0.45


def test_dhash_survives_one_pixel_jitter_but_not_new_content() -> None:
    rng = np.random.default_rng(0)
    img = photo_texture(400, 500, rng)
    other = photo_texture(400, 500, np.random.default_rng(1))
    d0, a0 = hash_crop(img[2:398, 2:498])
    d1, a1 = hash_crop(img[3:399, 1:497])
    d2, _ = hash_crop(other[2:398, 2:498])
    assert hamming(d0, d1) <= 6 and hamming(a0, a1) <= 8
    assert hamming(d0, d2) > 12


def test_proposals_fast_on_retina_frame() -> None:
    import time

    proposer = RegionProposer()
    img = _page(1, True, 12).image
    proposer.propose(img)
    t0 = time.perf_counter()
    proposer.propose(img)
    assert (time.perf_counter() - t0) * 1000.0 < 120.0  # ~20 ms on M4 Pro; loose for CI
    assert all(r.kind in (KIND_PICTURE, KIND_TILE) for r in proposer.propose(img))


def test_recall_at_1x_resolution() -> None:
    proposer = RegionProposer()
    hits = total = 0
    for seed in range(4):
        page = make_page(1728, 1117, seed=seed, dark=seed % 2 == 1)
        pics = _pictures(proposer.propose(page.image))
        for g in page.pictures:
            total += 1
            hits += max([_iou(r.xyxy, g) for r in pics] or [0.0]) >= 0.7
    assert hits / total >= 0.95
