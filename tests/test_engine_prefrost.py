"""Fast-path pre-frost (AGGRESSIVE): pictures that scroll in, appear after a
navigation, or enter a sub-pane are covered before a late analyzer answers.
Synthetic cv2-drawn pages only."""

from __future__ import annotations

import numpy as np

from edge.live_engine import HEADLESS_FRAME_MS, EngineCore, run_headless
from edge.quick_boxes import QuickBoxes, quick_picture_boxes
from edge.scroll_follow import small_scale, to_small_gray
from engine_synth import KnownRectsAnalyzer, clip, make_page
from vision_core.live.contracts import ProtectionMode, VerdictState

W, H = 1280, 800
PICS = [(100, 120, 500, 440), (700, 980, 1100, 1300), (200, 1500, 640, 1860)]


def _coverage(box, masks) -> float:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    if x2 <= x1 or y2 <= y1:
        return 1.0
    canvas = np.zeros((H, W), np.uint8)
    for m in masks:
        a, b, c, d = (int(round(v)) for v in m.xyxy)
        canvas[max(0, b) : max(0, d), max(0, a) : max(0, c)] = 1
    return float(canvas[y1:y2, x1:x2].mean())


def _scroll(n: int, step: int = 40):
    page = make_page(3200, W, PICS, seed=7)
    frames = [np.ascontiguousarray(page[i * step : i * step + H]) for i in range(n)]

    def truth(i: int):
        o = i * step
        return [(f"r{k}", (x1, y1 - o, x2, y2 - o)) for k, (x1, y1, x2, y2) in enumerate(PICS)]

    return frames, truth


def _exposed_frames(out, truth, *, skip_first: int = 1) -> list[tuple[int, str, float]]:
    bad = []
    for i, masks in enumerate(out):
        if i < skip_first:
            continue
        for rid, box in truth(i):
            vis = clip(box, W, H)
            area = (box[2] - box[0]) * (box[3] - box[1])
            if vis[2] - vis[0] <= 0 or vis[3] - vis[1] <= 0:
                continue
            vis_area = (vis[2] - vis[0]) * (vis[3] - vis[1])
            uncovered = (1.0 - _coverage(vis, masks)) * vis_area
            if uncovered > 0.05 * area:
                bad.append((i, rid, uncovered / area))
    return bad


def test_scrolling_in_picture_is_prefrosted_before_late_result() -> None:
    frames, truth = _scroll(30)
    late = dict(analyze_every=1, analyzer_latency_frames=5)
    aggr = run_headless(frames, KnownRectsAnalyzer(truth, frame_ms=HEADLESS_FRAME_MS), **late)
    med = run_headless(
        frames, KnownRectsAnalyzer(truth, frame_ms=HEADLESS_FRAME_MS, mode=ProtectionMode.MEDIUM), **late
    )
    bad_aggr = _exposed_frames(aggr, truth)
    bad_med = _exposed_frames(med, truth)
    assert len(bad_med) > 5  # without pre-frost a 5-frame-late analyzer leaks
    assert len(bad_aggr) <= 2, bad_aggr
    reasons = {m.reason for masks in aggr for m in masks}
    assert "PENDING_PREFROST" in reasons
    assert all(m.reason != "PENDING_PREFROST" for masks in med for m in masks)


def test_prefrost_of_safe_picture_clears_after_result() -> None:
    """Pictures the analyzer calls SAFE lose their pending frost once a
    result for a frame at or after the pre-frost arrives."""
    page = make_page(H, W, [(300, 200, 800, 600)], seed=3)
    frames = [np.full_like(page, 244)] + [page.copy() for _ in range(9)]

    def none(_i):
        return []

    def safe(i):
        return [("s0", (300, 200, 800, 600))] if i >= 1 else []

    analyzer = KnownRectsAnalyzer(none, frame_ms=HEADLESS_FRAME_MS, safe=safe)
    out = run_headless(frames, analyzer, analyze_every=1, analyzer_latency_frames=3)
    # The picture appears on frame 1 (navigation-sized change): pre-frosted now.
    assert any(m.verdict is VerdictState.PENDING and m.reason == "PENDING_PREFROST" for m in out[1])
    assert _coverage((300, 200, 800, 600), out[1]) > 0.95
    # Frame 0's look lands on 3; the next look (frame 3, after the change)
    # lands on 6 and says SAFE: the pending frost goes away then, not before.
    assert _coverage((300, 200, 800, 600), out[5]) > 0.95
    assert out[6] == [] and out[-1] == []


def test_navigation_prefrosts_new_page_immediately() -> None:
    a = make_page(H, W, [(100, 100, 500, 420)], seed=5)
    b = make_page(H, W, [(600, 300, 1100, 700)], seed=9, bg=230)
    frames = [a.copy() for _ in range(5)] + [b.copy() for _ in range(6)]

    def truth(i):
        return [("a0", (100, 100, 500, 420))] if i < 5 else [("b0", (600, 300, 1100, 700))]

    out = run_headless(
        frames, KnownRectsAnalyzer(truth, frame_ms=HEADLESS_FRAME_MS), analyze_every=1, analyzer_latency_frames=4
    )
    assert _coverage((600, 300, 1100, 700), out[5]) > 0.95  # same frame as the switch
    assert _coverage((100, 100, 500, 420), out[5]) > 0.95  # old cover kept (stale)


def test_subpane_entering_picture_is_prefrosted() -> None:
    base = make_page(H, W, [(40, 80, 300, 330)], seed=11)
    inner = make_page(2400, 360, [(30, 620, 330, 900)], seed=12, bg=225)
    frames = []
    for i in range(10):
        f = base.copy()
        f[200:700, 800:1160] = inner[i * 30 : i * 30 + 500]
        frames.append(f)

    def truth(i):
        return [("side", (40, 80, 300, 330)), ("pane", (830, 200 + 620 - i * 30, 1130, 200 + 900 - i * 30))]

    out = run_headless(
        frames, KnownRectsAnalyzer(truth, frame_ms=HEADLESS_FRAME_MS), analyze_every=1, analyzer_latency_frames=4
    )
    for i in range(4, 10):
        x1, y1, x2, y2 = truth(i)[1][1]
        vis = (x1, max(200, y1), x2, min(700, y2))
        if vis[3] - vis[1] < 12:
            continue
        assert _coverage(vis, out[i]) > 0.9, (i, vis, [m.xyxy for m in out[i]])


def test_quick_boxes_finds_pictures_not_text() -> None:
    page = make_page(H, W, [(100, 120, 500, 440), (700, 500, 1100, 760)], seed=4)
    small = to_small_gray(page)
    scale = small_scale(page.shape, small.shape)
    boxes = quick_picture_boxes(page, (0, 0, W, H), small=small, scale=scale, max_ms=50.0)
    canvas = np.zeros((H, W), bool)
    for b in boxes:
        x1, y1, x2, y2 = (int(v) for v in b)
        canvas[y1:y2, x1:x2] = True
    pics = np.zeros((H, W), bool)
    pics[120:440, 100:500] = True
    pics[500:760, 700:1100] = True
    assert canvas[pics].mean() > 0.95  # recall
    assert canvas[~pics].mean() < 0.12  # mostly not text lines (recall-first)


def test_quick_boxes_uses_injected_fn_and_budget() -> None:
    calls = []

    def fake(frame, rect, max_ms=4.0):
        calls.append((rect, max_ms))
        return [(10, 10, 50, 50)]

    q = QuickBoxes(fake)
    frame = np.zeros((100, 100, 3), np.uint8)
    out = q(frame, [(0, 0, 100, 100)], to_small_gray(frame), (4.0, 4.0), max_ms=3.0)
    assert out == [(10.0, 10.0, 50.0, 50.0)] and q.kind == "injected"
    assert calls and calls[0][1] <= 3.0


def test_fast_path_budget_on_retina_scroll() -> None:
    page = make_page(2234 + 1600, 3456, [(200 + 700 * (k % 4), 200 + 600 * (k // 4), 700 + 700 * (k % 4), 600 + 600 * (k // 4)) for k in range(16)], seed=2)
    core = EngineCore(prefrost=True)
    for i in range(8):
        core.on_frame(np.ascontiguousarray(page[i * 80 : i * 80 + 2234]), i * 83.3)
    assert core.prefrost_ms < 8.0
    assert len(core.masks) >= 4  # the pictures on screen got pending frost
