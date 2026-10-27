"""eval_live corpus + sequences: determinism and ground-truth sanity (small frames)."""

from __future__ import annotations

import numpy as np
import pytest

from eval_live.corpus import LAYOUTS, make_page
from eval_live.sequences import make_sequence, scroll_profile

W = 864  # scale 0.5 keeps these tests fast
SIZE = (864, 559)


def _overlap(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


@pytest.mark.parametrize("layout", LAYOUTS)
def test_page_is_deterministic(layout: str) -> None:
    a = make_page(layout, 3, width_px=W)
    b = make_page(layout, 3, width_px=W)
    assert np.array_equal(a.image, b.image)
    assert a.gt() == b.gt()
    c = make_page(layout, 4, width_px=W)
    assert not np.array_equal(a.image, c.image) or a.gt() != c.gt()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_page_gt_sanity(layout: str) -> None:
    page = make_page(layout, 0, width_px=W)
    H = page.height
    pics = page.pictures
    assert len(pics) >= 2
    assert page.height > SIZE[1] * 1.5  # taller than the screen, so it can scroll
    flags = [p.flagged for p in pics]
    assert any(flags) and not all(flags)
    for p in pics:
        x1, y1, x2, y2 = p.xyxy
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
        if p.flagged:
            assert p.parts
            for q in p.parts:
                assert x1 <= q[0] < q[2] <= x2 and y1 <= q[1] < q[3] <= y2
        else:
            assert p.parts == ()
    for i, a in enumerate(pics):
        for b in pics[i + 1 :]:
            assert not _overlap(a.xyxy, b.xyxy), (a.pic_id, b.pic_id)


def test_corpus_has_hard_cases() -> None:
    tags = {t for layout in LAYOUTS for p in make_page(layout, 0, width_px=W).pictures for t in p.tags}
    assert {"low_contrast", "text_overlay", "edge_touch"} <= tags


def test_scroll_profiles_respect_speed_bands() -> None:
    rng = np.random.default_rng(0)
    steps = np.diff(scroll_profile("smooth", 60, rng, 10**6))
    assert steps.min() >= 0 and steps.max() <= 40 and (steps[steps > 0] >= 10).all()
    jumps = np.abs(np.diff(scroll_profile("jumpy", 60, rng, 10**6)))
    assert all(j == 0 or 80 <= j <= 200 for j in jumps) and (jumps > 0).any()
    mom = np.abs(np.diff(scroll_profile("momentum", 60, rng, 10**6)))
    first = int(np.argmax(mom > 0))
    assert mom[first] >= 120 and mom[first + 1] < mom[first]  # flick, then decay


@pytest.mark.parametrize("name", ["search_jumpy", "nav_social_to_shop", "subpane_grid", "video_play"])
def test_sequence_gt_in_screen_coords(name: str) -> None:
    seq = make_sequence(name, size=SIZE, n=12)
    assert len(seq.gt) == 12
    frames = list(seq.frames())
    assert all(f.shape == (SIZE[1], SIZE[0], 3) for f in frames)
    assert np.array_equal(frames[5], seq.frame(5))  # re-render is identical
    for g in seq.gt:
        for p in g.pictures:
            x1, y1, x2, y2 = p.xyxy
            assert 0 <= x1 < x2 <= SIZE[0] and 0 <= y1 < y2 <= SIZE[1]
            f = p.full_xyxy
            assert f[0] <= x1 and f[1] <= y1 and x2 <= f[2] and y2 <= f[3]
    if name == "nav_social_to_shop":
        navs = [g.index for g in seq.gt if g.nav]
        assert navs == [6]
        assert seq.gt[6].frames_since_nav == 0 and seq.gt[7].after_nav
        before = {p.gid.split("/")[0] for p in seq.gt[5].pictures}
        after = {p.gid.split("/")[0] for p in seq.gt[6].pictures}
        assert before.isdisjoint(after)
    if name == "subpane_grid":
        left = [{p.gid: p.xyxy for p in g.pictures if p.gid.startswith("news")} for g in seq.gt]
        assert all(d == left[0] for d in left)  # static panel never moves
        assert any(g.moving for g in seq.gt)
    if name == "video_play":
        vid = [[p for p in g.pictures if p.kind == "video"][0] for g in seq.gt]
        assert [v.flagged for v in vid] == [i >= seq.video_flag_from for i in range(12)]
        assert not np.array_equal(frames[1], frames[2])  # content changes without scrolling
        assert seq.gt[2].scroll_dy == 0
