"""quick_picture_boxes (edge.quick_boxes, the one quick finder): recall-first
pre-frost boxes for revealed strips."""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from edge.quick_boxes import quick_picture_boxes
from vision_core.live.regions import _iou
from vision_core.live.synth import draw_paragraph, make_page, make_ui_page, photo_texture


@lru_cache(maxsize=None)
def _page(seed: int):
    return make_page(seed=seed, dark=seed % 2 == 1)


def _inside(g, b, frac: float = 0.9) -> bool:
    return (min(g[2], b[2]) - max(g[0], b[0])) >= frac * (g[2] - g[0]) and (
        min(g[3], b[3]) - max(g[1], b[1])
    ) >= frac * (g[3] - g[1])


def test_full_frame_recall_and_no_text() -> None:
    for seed in (1, 2):
        page = _page(seed)
        boxes = quick_picture_boxes(page.image, max_ms=50.0)
        hit = sum(1 for g in page.pictures if any(_inside(g, b) for b in boxes))
        assert hit / len(page.pictures) >= 0.9
        assert all(_iou(b, t) < 0.3 for b in boxes for t in page.texts)
    ui = make_ui_page(seed=0)
    boxes = quick_picture_boxes(ui.image, max_ms=50.0)
    assert all(_iou(b, t) < 0.3 for b in boxes for t in ui.texts)


def test_revealed_strip_boxes_stay_inside_rect_and_find_slivers() -> None:
    rng = np.random.default_rng(0)
    img = np.full((2234, 3456, 3), 250, np.uint8)
    draw_paragraph(img, (100, 1900, 1500, 2230), rng)
    img[2150:2234, 1700:2300] = photo_texture(84, 600, rng)  # scrolling in at the bottom
    img[2200:2234, 2400:3000] = photo_texture(34, 600, rng)  # a thin sliver
    rect = (0.0, 2034.0, 3456.0, 2234.0)
    boxes = quick_picture_boxes(img, rect)
    assert all(b[1] >= rect[1] and b[3] <= rect[3] for b in boxes)
    assert any(_inside((1700, 2150, 2300, 2234), b) for b in boxes)
    assert any(_inside((2400, 2200, 3000, 2234), b) for b in boxes)
    assert all(b[2] <= 1600 or b[0] >= 1600 for b in boxes)  # the text column is not returned


def test_never_returns_whole_page_or_text_only_page() -> None:
    rng = np.random.default_rng(1)
    img = np.full((2234, 3456, 3), 250, np.uint8)
    for col in range(3):
        draw_paragraph(img, (80 + col * 1130, 120, 80 + col * 1130 + 1000, 2150), rng)
    assert quick_picture_boxes(img, max_ms=50.0) == []
    full = photo_texture(1117, 1728, rng)
    assert quick_picture_boxes(full, max_ms=50.0) == []  # the whole screen is never a box


def test_timing_strip_and_full_frame() -> None:
    page = _page(3)
    quick_picture_boxes(page.image, (0, 2034, 3456, 2234))
    strip, full = [], []
    for _ in range(5):
        t0 = time.perf_counter()
        quick_picture_boxes(page.image, (0, 2034, 3456, 2234))
        strip.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        quick_picture_boxes(page.image, None, max_ms=1e9)
        full.append((time.perf_counter() - t0) * 1000.0)
    # ~1 ms / ~7 ms on M4 Pro; loose bounds for CI noise.
    assert min(strip) <= 4.0 * 2
    assert min(full) <= 10.0 * 2


def test_max_ms_bounds_the_call() -> None:
    page = _page(1)
    t0 = time.perf_counter()
    quick_picture_boxes(page.image, None, max_ms=0.0)
    assert (time.perf_counter() - t0) * 1000.0 < 20.0
