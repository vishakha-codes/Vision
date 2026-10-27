"""Live engine: scroll follow + mask persistence via the headless API.

Synthetic, cv2-drawn pages only (textured abstract pictures + text)."""

from __future__ import annotations

import numpy as np

from edge.live_engine import HEADLESS_FRAME_MS, EngineCore, run_headless
from edge.scroll_follow import PhaseShift, relocalize, small_scale, to_small_gray
from engine_synth import KnownRectsAnalyzer, NullAnalyzer, clip, iou, make_page
from vision_core.live.contracts import AnalyzeResult, MaskSpec, ProtectionMode, VerdictState

W, H = 1280, 800
MEDIUM = ProtectionMode.MEDIUM  # pure follow / persistence: no fast-path pre-frost
RECTS = [(100, 120, 500, 440), (700, 300, 1100, 650), (200, 1500, 640, 1900), (760, 2100, 1180, 2400)]
STEP = 40


def _scroll_frames(n: int, step: int = STEP, rects=RECTS, seed: int = 3):
    page = make_page(3600, W, rects, seed=seed)
    offs = [i * step for i in range(n)]
    frames = [np.ascontiguousarray(page[o : o + H]) for o in offs]

    def truth(i: int):
        o = offs[min(i, n - 1)]
        return [(f"r{k}", (x1, y1 - o, x2, y2 - o)) for k, (x1, y1, x2, y2) in enumerate(rects)]

    return frames, truth


def _coverage(box, masks) -> float:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    if x2 <= x1 or y2 <= y1:
        return 1.0
    canvas = np.zeros((H, W), np.uint8)
    for m in masks:
        a, b, c, d = (int(round(v)) for v in m.xyxy)
        canvas[max(0, b) : max(0, d), max(0, a) : max(0, c)] = 1
    return float(canvas[y1:y2, x1:x2].mean())


def test_masks_follow_40px_scroll_within_one_frame() -> None:
    """Analyzer looks once (frame 0, result 2 frames late). Every later frame
    the masks must sit on the pictures (IoU ≥ 0.8) from the fast path alone."""
    frames, truth = _scroll_frames(14)
    analyzer = KnownRectsAnalyzer(truth, frame_ms=HEADLESS_FRAME_MS, mode=MEDIUM)
    out = run_headless(frames, analyzer, analyze_every=10_000, analyzer_latency_frames=2)
    assert len(analyzer.calls) == 1
    assert out[0] == [] and out[1] == []  # result not back yet
    checked = 0
    for i in range(2, len(frames)):
        masks = out[i]
        for rid, box in truth(i)[:2]:
            vis = clip(box, W, H)
            if vis[3] - vis[1] < 40:
                continue
            best = max((iou(vis, m.xyxy) for m in masks), default=0.0)
            assert best >= 0.8, (i, rid, best, vis, [m.xyxy for m in masks])
            checked += 1
    assert checked >= 15


def test_scroll_with_periodic_late_analyzer_never_reexposes() -> None:
    """Once a picture is covered it stays ≥ 95 % covered while scrolling,
    including pictures that scrolled in from the bottom edge."""
    frames, truth = _scroll_frames(60)
    analyzer = KnownRectsAnalyzer(truth, frame_ms=HEADLESS_FRAME_MS)
    out = run_headless(frames, analyzer, analyze_every=5, analyzer_latency_frames=3)
    covered: set[str] = set()
    for i, masks in enumerate(out):
        for rid, box in truth(i):
            vis = clip(box, W, H)
            if vis[2] - vis[0] < 8 or vis[3] - vis[1] < 8:
                continue
            cov = _coverage(vis, masks)
            if rid in covered:
                assert cov >= 0.95, (i, rid, cov)
            elif cov >= 0.95:
                covered.add(rid)
    assert covered == {"r0", "r1", "r2", "r3"}
    # Masks keep their identity while they move (overlay moves views): the
    # key covering r0 on frame 3 (a pre-frost adopted by the analyzer) stays.
    def key_on(i: int) -> str:
        box = clip(truth(i)[0][1], W, H)
        return max(out[i], key=lambda m: iou(box, m.xyxy)).region_id

    assert len({key_on(i) for i in range(3, 10)}) == 1


def test_masks_leaving_screen_are_dropped() -> None:
    frames, truth = _scroll_frames(24)
    analyzer = KnownRectsAnalyzer(truth, frame_ms=HEADLESS_FRAME_MS, mode=MEDIUM)
    out = run_headless(frames, analyzer, analyze_every=10_000, analyzer_latency_frames=0)
    assert {m.region_id for m in out[0]} == {"r0", "r1"}
    # r0 bottom (440) is off the top after 11 frames of 40 px; r1 after 17
    assert "r0" not in {m.region_id for m in out[12]}
    assert out[20] == []


def _nav_frames():
    rects_a = [(100, 100, 500, 420), (700, 380, 1100, 700)]
    rects_b = [(300, 200, 800, 600)]
    page_a = make_page(H, W, rects_a, seed=5)
    page_b = make_page(H, W, rects_b, seed=9, bg=230)
    frames = [page_a.copy() for _ in range(10)] + [page_b.copy() for _ in range(10)]

    def truth(i: int):
        if i < 10:
            return [(f"a{k}", r) for k, r in enumerate(rects_a)]
        return [("b0", rects_b[0])]

    return frames, truth, rects_a, rects_b


def test_navigation_keeps_masks_until_next_result() -> None:
    frames, truth, rects_a, rects_b = _nav_frames()
    analyzer = KnownRectsAnalyzer(truth, frame_ms=HEADLESS_FRAME_MS, mode=MEDIUM)
    out = run_headless(frames, analyzer, analyze_every=4, analyzer_latency_frames=3)
    # Frame 10 navigates while frame 8's look is still in flight (lands on
    # 11, predates the change, so it must not drop anything). The urgent
    # post-change look follows; until it lands the old masks stay up.
    post = [i for i, (idx, _c) in enumerate(analyzer.calls) if idx >= 10]
    assert post, analyzer.calls
    first_post_frame = analyzer.calls[post[0]][0]
    applied_at = first_post_frame + 3
    for i in range(10, applied_at):
        ids = {m.region_id for m in out[i]}
        assert {"a0", "a1"} <= ids, (i, ids)
    final = {m.region_id for m in out[applied_at]}
    assert final == {"b0"}, (applied_at, final)
    assert iou(out[applied_at][0].xyxy, rects_b[0]) > 0.95
    # The navigation itself asked for a full-frame look (changed=None).
    assert analyzer.calls[post[0]][1] is None


def test_navigation_then_safe_clears_and_pending_prefrost_renders() -> None:
    frames, _truth, rects_a, _rects_b = _nav_frames()

    class Scripted:
        mode = None

        def set_mode(self, mode):
            self.mode = mode

        def analyze(self, frame, *, changed=None, t_ms=0.0):
            idx = int(round(t_ms / HEADLESS_FRAME_MS))
            if idx < 10:
                return AnalyzeResult(
                    masks=[
                        MaskSpec("a0", rects_a[0], reason="PENDING_PREFROST", verdict=VerdictState.PENDING),
                    ]
                )
            return AnalyzeResult()  # new page: nothing to cover

    out = run_headless(frames, Scripted(), analyze_every=1, analyzer_latency_frames=2)
    assert out[2] and out[2][0].verdict is VerdictState.PENDING
    assert out[2][0].reason == "PENDING_PREFROST"
    assert out[11] and out[11][0].region_id == "a0"  # stale, still covering
    assert out[-1] == []


def test_stub_analyzer_runs_clean() -> None:
    frames, _truth = _scroll_frames(6)
    out = run_headless(frames, NullAnalyzer())
    assert out == [[] for _ in frames]


def test_custom_monotonic_t_ms_drives_keyframes() -> None:
    frames, truth = _scroll_frames(8)
    analyzer = KnownRectsAnalyzer(truth, frame_ms=1.0, mode=MEDIUM)
    times = [float(i) for i in range(8)]  # index == t_ms for this analyzer
    out = run_headless(frames, analyzer, t_ms=times, analyze_every=3)
    assert [c[0] for c in analyzer.calls] == [0, 3, 6]
    assert len(out) == 8


def _subpane_frames(n: int, step: int = 30):
    """Fixed sidebar (with a picture) + a small inner scroller that moves
    alone; most of the screen is static so the global shift is ~0."""
    rng_page = make_page(H, W, [(40, 80, 300, 330)], seed=11)
    inner = make_page(2400, 360, [(30, 140, 330, 420)], seed=12, bg=225)
    frames = []
    for i in range(n):
        f = rng_page.copy()
        f[200:700, 800:1160] = inner[i * step : i * step + 500]
        frames.append(f)

    def truth(i: int):
        return [("side", (40, 80, 300, 330)), ("pane", (830, 200 + 140 - i * step, 1130, 200 + 420 - i * step))]

    return frames, truth


def test_subpane_scroll_follows_with_match_template() -> None:
    frames, truth = _subpane_frames(8)
    analyzer = KnownRectsAnalyzer(truth, frame_ms=HEADLESS_FRAME_MS, mode=MEDIUM)
    out = run_headless(frames, analyzer, analyze_every=10_000, analyzer_latency_frames=1)
    for i in range(1, 8):
        by_id = {m.region_id: m for m in out[i]}
        assert iou(by_id["side"].xyxy, truth(i)[0][1]) > 0.95, i
        # The pane mask is clipped by nothing (pane box is inside the frame);
        # compare to the truth box, the inner scroller crops at y=200.
        want = clip(truth(i)[1][1], W, H)
        want = (want[0], max(200.0, want[1]), want[2], want[3])
        got = by_id["pane"].xyxy
        got = (got[0], max(200.0, got[1]), got[2], got[3])
        assert iou(got, want) >= 0.8, (i, got, want)


def test_relocalize_direct_subpane_and_phase_static() -> None:
    frames, truth = _subpane_frames(2)
    a, b = to_small_gray(frames[0]), to_small_gray(frames[1])
    scale = small_scale(frames[0].shape, a.shape)
    shift = PhaseShift()(a, b, scale)
    assert abs(shift.dy) < 4.0  # static screen dominates
    rel = relocalize(a, b, truth(0)[1][1], scale, (shift.dx, shift.dy))
    assert rel.ok and abs(rel.dy + 30.0) <= 2.0 and abs(rel.dx) <= 2.0
    still = relocalize(a, b, truth(0)[0][1], scale, (0.0, -30.0))
    assert still.ok and abs(still.dy) <= 1.0


def test_low_confidence_keeps_mask_and_requests_reanalysis() -> None:
    """Content under a mask is replaced by noise (video-like): the mask
    stays put (fail closed) and the area is queued for the analyzer."""
    page = make_page(H, W, [(300, 200, 700, 520)], seed=21)
    core = EngineCore()
    core.on_frame(page, 0.0)
    core.apply_result(AnalyzeResult(masks=[MaskSpec("v", (300, 200, 700, 520))]), 0)
    req = core.take_request(0, 0.0)
    assert req is not None  # first frame: urgent full look
    nxt = page.copy()
    rng = np.random.default_rng(1)
    nxt[200:520, 300:700] = rng.integers(0, 255, (320, 400, 3), dtype=np.uint8)
    core.on_frame(nxt, 50.0)
    masks = core.visible()
    assert len(masks) == 1 and iou(masks[0].xyxy, (300, 200, 700, 520)) > 0.98
    assert core.pending_change is not None
    req = core.take_request(1, 60.0)
    assert req is not None and req.changed is not None and req.reason == "gate"
    cx1, cy1, cx2, cy2 = req.changed
    assert cx1 <= 300 and cy1 <= 200 and cx2 >= 700 and cy2 >= 520
