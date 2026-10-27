"""Window-aware segmentation, clipping and safety caps (synthetic desktops,
cv2-drawn only). This is the "aggressive mode blurs the whole screen" bug:
these tests must keep it from coming back."""

from __future__ import annotations

import os
import time
from functools import lru_cache

import numpy as np
import pytest

from edge.capture import WindowLister, window_layout_from_info
from edge.coords import DisplayGeom, capture_xyxy_to_cocoa, cocoa_rect_to_capture, points_xyxy_to_capture
from edge.live_engine import EngineCore, run_headless
from edge.quick_boxes import quick_picture_boxes
from edge.scroll_follow import small_scale, to_small_gray
from eval_live.metrics import evaluate_sequence, summarize
from eval_live.oracle import OracleTruth, make_live_analyzer
from eval_live.sequences import make_sequence
from vision_core.live.analyzer import make_analyzer
from vision_core.live.cache import VerdictCache
from vision_core.live.contracts import MaskSpec, ProtectionMode, ScreenLayout, VerdictState, WindowInfo
from vision_core.live.windows import (
    HeavyTracker,
    cap_coverage,
    clip_specs,
    geometry,
    inter,
    label_map,
)


class NoHit:
    def detect_batch(self, images):
        return [[] for _ in images]


@lru_cache(maxsize=None)
def _desk(name: str, res: str = "retina", n: int = 12):
    return make_sequence(name, res=res, n=n)


def _windows_of(layout: ScreenLayout) -> dict[int, tuple]:
    return {w.window_id: w.xyxy for w in layout.windows if w.layer == 0}


def _system(layout: ScreenLayout, box) -> float:
    return geometry(layout).in_system_ui(box)


def _inside_one_window(layout: ScreenLayout, box, tol: float = 2.0) -> bool:
    for b in _windows_of(layout).values():
        if box[0] >= b[0] - tol and box[1] >= b[1] - tol and box[2] <= b[2] + tol and box[3] <= b[3] + tol:
            return True
    return False


# --------------------------------------------------------------------------- proposer / quick finder


@pytest.mark.parametrize("name", ["desk_static_photo", "desk_static_gradient", "desk_static_solid", "desk_cmdtab"])
def test_proposer_on_desktop_stays_inside_windows(name: str) -> None:
    seq = _desk(name)
    i = len(seq) - 1  # cmdtab: three overlapping windows
    frame, layout = seq.frame(i), seq.gt[i].windows
    W, H = seq.width, seq.height
    an = make_analyzer(ProtectionMode.AGGRESSIVE, detector=NoHit(), warmup=False)
    res = an.analyze(frame, t_ms=0.0, windows=layout)
    assert res.regions
    for r in res.regions:
        a = (r.xyxy[2] - r.xyxy[0]) * (r.xyxy[3] - r.xyxy[1])
        if r.kind != "tile":
            assert a <= 0.40 * W * H, r.xyxy
        assert _system(layout, r.xyxy) == 0.0, r.xyxy  # never on the menu bar / Dock
        assert r.window_id in _windows_of(layout), r.xyxy
        assert _inside_one_window(layout, r.xyxy), r.xyxy  # never spans two windows
        if r.group_xyxy is not None:
            assert _inside_one_window(layout, r.group_xyxy)


def test_proposer_without_window_list_never_returns_the_desktop() -> None:
    """Fallback (no window info): the safety caps alone still reject the
    whole-desktop region the bug produced."""
    seq = _desk("desk_static_photo")
    frame = seq.frame(0)
    W, H = seq.width, seq.height
    an = make_analyzer(ProtectionMode.AGGRESSIVE, detector=NoHit(), warmup=False)
    for r in an.analyze(frame, t_ms=0.0).regions:
        if r.kind == "tile":
            continue
        assert (r.xyxy[2] - r.xyxy[0]) * (r.xyxy[3] - r.xyxy[1]) <= 0.40 * W * H
        edges = (r.xyxy[0] <= 4) + (r.xyxy[1] <= 4) + (r.xyxy[2] >= W - 4) + (r.xyxy[3] >= H - 4)
        assert edges < 3, r.xyxy


@pytest.mark.parametrize("name", ["desk_static_photo", "desk_cmdtab", "desk_maximised"])
def test_quick_finder_on_desktop(name: str) -> None:
    seq = _desk(name)
    i = len(seq) - 1
    frame, layout = seq.frame(i), seq.gt[i].windows
    W, H = seq.width, seq.height
    small = to_small_gray(frame)
    scale = small_scale(frame.shape, small.shape)
    lab = label_map(layout, small.shape, scale)
    boxes = quick_picture_boxes(frame, None, small=small, scale=scale, labels=(lab, scale), max_ms=1e9)
    assert boxes
    for b in boxes:
        assert (b[2] - b[0]) * (b[3] - b[1]) <= 0.40 * W * H
        assert _system(layout, b) == 0.0, b
        assert _inside_one_window(layout, b, tol=8.0), b


def test_quick_finder_without_labels_obeys_area_cap() -> None:
    seq = _desk("desk_static_photo")
    frame = seq.frame(0)
    W, H = seq.width, seq.height
    for b in quick_picture_boxes(frame, None, max_ms=1e9):
        assert (b[2] - b[0]) * (b[3] - b[1]) <= 0.40 * W * H


# --------------------------------------------------------------------------- clipping


def _layout(*wins, W=1000, H=800, visible=(0, 30, 1000, 740), stale=False) -> ScreenLayout:
    return ScreenLayout((W, H), tuple(wins), visible, stale=stale)


def _cover(specs, box) -> float:
    canvas = np.zeros((800, 1000), bool)
    for m in specs:
        a, b, c, d = (int(round(v)) for v in m.xyxy)
        canvas[max(0, b) : max(0, d), max(0, a) : max(0, c)] = True
    x1, y1, x2, y2 = (int(v) for v in box)
    return float(canvas[y1:y2, x1:x2].mean())


def test_clip_cuts_back_window_mask_at_front_window_and_system_ui() -> None:
    front = WindowInfo(2, (500, 100, 900, 500))
    back = WindowInfo(1, (50, 0, 950, 790))
    lay = _layout(front, back)
    mask = MaskSpec("m", (300, 10, 700, 760), window_id=1)
    out = clip_specs([mask], lay)
    assert len(out) > 1 and all(m.window_id == 1 for m in out)
    assert _cover(out, (500, 100, 700, 500)) == 0.0  # the front window's pixels
    assert _cover(out, (300, 0, 700, 30)) == 0.0  # menu bar
    assert _cover(out, (300, 740, 700, 760)) == 0.0  # Dock strip
    assert _cover(out, (300, 30, 500, 740)) == 1.0  # the rest of the picture stays covered
    assert _cover(out, (500, 500, 700, 740)) == 1.0


def test_clip_pad_never_crosses_into_front_window() -> None:
    front = WindowInfo(2, (400, 100, 800, 500))
    back = WindowInfo(1, (0, 30, 1000, 740))
    out = clip_specs([MaskSpec("m", (100, 200, 398, 400), window_id=1)], _layout(front, back), pad_px=8.0)
    assert _cover(out, (400, 100, 800, 500)) == 0.0
    assert _cover(out, (92, 192, 398, 408)) == 1.0


def test_dragged_window_carries_its_masks_within_one_tick() -> None:
    then = _layout(WindowInfo(2, (100, 100, 400, 400)), WindowInfo(1, (0, 30, 1000, 740)))
    now = _layout(WindowInfo(2, (300, 150, 600, 450)), WindowInfo(1, (0, 30, 1000, 740)))
    out = clip_specs([MaskSpec("a", (150, 150, 250, 250), window_id=2)], now, frame_layout=then)
    assert len(out) == 1 and out[0].xyxy == (350.0, 200.0, 450.0, 300.0)
    # A back-window mask under the window that was dragged on top is cut at once.
    back = clip_specs([MaskSpec("b", (320, 160, 500, 300), window_id=1)], now, frame_layout=then)
    assert _cover(back, (320, 160, 500, 300)) == 0.0


def test_mask_without_window_is_kept_off_system_ui() -> None:
    lay = _layout(WindowInfo(1, (0, 30, 1000, 740)))
    out = clip_specs([MaskSpec("m", (0, 0, 1000, 800), window_id=None)], lay)
    assert len(out) == 1 and out[0].xyxy == (0.0, 30.0, 1000.0, 740.0)


def test_system_ui_windows_occlude_but_huge_transparent_ones_do_not() -> None:
    popover = WindowInfo(3, (200, 200, 300, 300), layer=101)
    overlay = WindowInfo(4, (0, 0, 1000, 800), layer=1500, alpha=1.0)  # e.g. a full-screen tool overlay
    lay = _layout(popover, overlay, WindowInfo(1, (0, 30, 1000, 740)))
    out = clip_specs([MaskSpec("m", (100, 100, 400, 400), window_id=1)], lay)
    assert _cover(out, (200, 200, 300, 300)) == 0.0
    assert _cover(out, (100, 100, 200, 400)) == 1.0


def test_busy_cap_warns_and_sheds_speculative_masks_only() -> None:
    lay = _layout(WindowInfo(1, (0, 30, 1000, 740)))
    unsafe = MaskSpec("u", (0, 30, 500, 400), verdict=VerdictState.UNSAFE, window_id=1)
    pend = [MaskSpec(f"p{i}", (500, 30 + 100 * i, 1000, 130 + 100 * i), verdict=VerdictState.PENDING,
                     reason="PENDING_PREFROST", window_id=1) for i in range(7)]
    pend.append(MaskSpec("p7", (0, 400, 500, 740), verdict=VerdictState.PENDING, reason="PENDING_PREFROST", window_id=1))
    kept, busy = cap_coverage([unsafe] + pend, lay, (1000, 800))
    assert busy
    assert unsafe in kept
    assert len(kept) < len(pend) + 1
    some, busy2 = cap_coverage([unsafe], lay, (1000, 800))
    assert some == [unsafe] and not busy2


def test_heavy_tracker_decays() -> None:
    h = HeavyTracker()
    for k in range(3):
        h.observe(7, f"m{k}", 1000.0)
    assert h.heavy(7, 2000.0, 3, 10.0)
    assert not h.heavy(8, 2000.0, 3, 10.0)
    assert not h.heavy(7, 12_000.0, 3, 10.0)


# --------------------------------------------------------------------------- capture → layout


def _geom() -> DisplayGeom:
    return DisplayGeom(0, 0, 1728, 1117, 3456, 2234, 2.0, 2.0, 0, 0, 1117)


def test_window_list_to_layout_is_retina_correct_and_drops_own_windows() -> None:
    info = [
        {"kCGWindowNumber": 1, "kCGWindowLayer": 1000, "kCGWindowOwnerPID": os.getpid(), "kCGWindowOwnerName": "Python",
         "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 1728, "Height": 1117}, "kCGWindowAlpha": 1.0},
        {"kCGWindowNumber": 2, "kCGWindowLayer": 27, "kCGWindowOwnerPID": 99, "kCGWindowOwnerName": "GuardianIsland",
         "kCGWindowBounds": {"X": 700, "Y": 0, "Width": 300, "Height": 60}, "kCGWindowAlpha": 1.0},
        {"kCGWindowNumber": 3, "kCGWindowLayer": 0, "kCGWindowOwnerPID": 5, "kCGWindowOwnerName": "Safari",
         "kCGWindowName": "private title", "kCGWindowBounds": {"X": 80, "Y": 60, "Width": 1220, "Height": 940}, "kCGWindowAlpha": 1.0},
        {"kCGWindowNumber": 4, "kCGWindowLayer": 0, "kCGWindowOwnerPID": 6, "kCGWindowOwnerName": "Notes",
         "kCGWindowBounds": {"X": 2000, "Y": 0, "Width": 800, "Height": 600}, "kCGWindowAlpha": 1.0},  # other display
    ]
    lay = window_layout_from_info(info, _geom(), exclude_pids={os.getpid()}, visible_cocoa=(0, 63, 1728, 1016))
    ids = [w.window_id for w in lay.windows]
    assert ids == [2, 3]
    island, safari = lay.windows
    assert island.layer >= 1 and not island.is_app  # the island hides masks, owns no pictures
    assert safari.xyxy == (160.0, 120.0, 2600.0, 2000.0)
    assert "private title" not in repr(lay)  # titles are compared, never kept
    assert lay.visible_xyxy == (0.0, 76.0, 3456.0, 2108.0)  # menu bar 38 pt, Dock 63 pt


def test_coords_round_trip() -> None:
    g = _geom()
    box = (100.0, 200.0, 900.0, 1000.0)
    x, y, w, h = capture_xyxy_to_cocoa(box, g)
    assert cocoa_rect_to_capture((x, y, w, h), g) == pytest.approx(box)
    assert points_xyxy_to_capture((10, 20, 30, 40), g) == (20.0, 40.0, 60.0, 80.0)


def test_window_lister_marks_animation_frames_stale() -> None:
    lister = WindowLister()
    base = [{"kCGWindowNumber": 3, "kCGWindowLayer": 0, "kCGWindowOwnerPID": 5, "kCGWindowOwnerName": "Safari",
             "kCGWindowBounds": {"X": 80, "Y": 60, "Width": 1220, "Height": 940}, "kCGWindowAlpha": 1.0}]
    assert not lister.build(base, _geom(), now=0.0).stale
    opened = base + [{"kCGWindowNumber": 4, "kCGWindowLayer": 0, "kCGWindowOwnerPID": 6, "kCGWindowOwnerName": "Notes",
                      "kCGWindowBounds": {"X": 300, "Y": 300, "Width": 400, "Height": 300}, "kCGWindowAlpha": 1.0}]
    assert lister.build(opened, _geom(), now=1.0).stale  # just appeared
    assert not lister.build(opened, _geom(), now=1.5).stale
    fading = [dict(opened[1], kCGWindowAlpha=0.4)] + base
    assert lister.build(fading, _geom(), now=2.0).stale


def test_always_present_dock_overlay_is_not_mission_control() -> None:
    """Recent macOS keeps a full-screen Dock window at the Dock level. It
    used to mark every frame stale, so the live engine analyzed nothing."""
    lister = WindowLister()
    safari = {"kCGWindowNumber": 3, "kCGWindowLayer": 0, "kCGWindowOwnerPID": 5, "kCGWindowOwnerName": "Safari",
              "kCGWindowBounds": {"X": 80, "Y": 60, "Width": 1220, "Height": 940}, "kCGWindowAlpha": 1.0}
    dock = {"kCGWindowNumber": 50, "kCGWindowLayer": 20, "kCGWindowOwnerPID": 2, "kCGWindowOwnerName": "Dock",
            "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 1728, "Height": 1117}, "kCGWindowAlpha": 1.0}
    assert not lister.build([dock, safari], _geom(), now=0.0).stale
    assert not lister.build([dock, safari], _geom(), now=5.0).stale
    mission = dict(dock, kCGWindowNumber=51, kCGWindowLayer=18)  # a new big Dock overlay appears
    assert lister.build([mission, dock, safari], _geom(), now=6.0).stale
    assert lister.build([mission, dock, safari], _geom(), now=10.0).stale
    assert not lister.build([dock, safari], _geom(), now=11.0).stale  # Mission Control closed
    assert not window_layout_from_info([dock, safari], _geom(), steady_dock_ids={50}).stale
    assert window_layout_from_info([dock, safari], _geom()).stale  # stateless call keeps the old rule


# --------------------------------------------------------------------------- cache / analyzer small fixes


def test_big_region_cache_entries_are_scoped_and_short_lived() -> None:
    c = VerdictCache(reuse_any_size=lambda v: True)
    c.put(1, 2, (2000, 1500), "UNSAFE", 0.0, scope=("win", 7), ttl_ms=3000.0, any_size=False, refresh=False)
    assert c.get(1, 2, (2000, 1500), 100.0) is None  # unscoped lookups never see it
    assert c.get(1, 2, (2000, 1500), 100.0, scope=("win", 8)) is None  # another window
    assert c.get(1, 2, (2000, 1500), 1000.0, scope=("win", 7)) == "UNSAFE"
    assert c.get(1, 2, (200, 150), 1000.0, scope=("win", 7)) is None  # no any-size reuse
    assert c.get(1, 2, (2000, 1500), 3500.0, scope=("win", 7)) is None  # hits did not extend it


def test_analyzer_backend_property_and_t_ms_zero_is_a_real_time() -> None:
    an = make_analyzer(ProtectionMode.AGGRESSIVE, detector=NoHit(), warmup=False)
    assert isinstance(an.backend, str) and an.backend
    frame = np.full((400, 600, 3), 240, np.uint8)
    an.analyze(frame, t_ms=0.0)
    # t = 0.0 must not switch to the monotonic clock (bucket + cache use it).
    assert an.bucket is not None
    an.analyze(frame, t_ms=None)


# --------------------------------------------------------------------------- engine + desktop eval (hard asserts)


def _engine(name: str, mode=ProtectionMode.AGGRESSIVE, latency: int = 0, res: str = "half", n: int = 12):
    seq = make_sequence(name, res=res, n=n)
    truth = OracleTruth(seq.gt, seq.width, seq.height)
    an, _label, _note = make_live_analyzer(mode, truth, "auto")
    out = run_headless(truth.iter_frames(seq.frames()), an, t_ms=[g.t_ms for g in seq.gt],
                       analyzer_latency_frames=latency, windows=[g.windows for g in seq.gt])
    ev = evaluate_sequence(seq.name, seq.gt, [[m.xyxy for m in f] for f in out], seq.width, seq.height, grid=2,
                           owners=[[m.window_id for m in f] for f in out])
    return seq, out, summarize([ev], "aggressive" if mode is ProtectionMode.AGGRESSIVE else "medium", settle=latency + 3)


DESK_ALL = ["desk_static_photo", "desk_static_gradient", "desk_static_solid", "desk_clean", "desk_drag",
            "desk_cmdtab", "desk_scroll_clock", "desk_nav_clean", "desk_open_close", "desk_maximised"]


@pytest.mark.parametrize("name", DESK_ALL)
def test_desktop_hard_asserts(name: str) -> None:
    seq, out, s = _engine(name)
    d = s["desktop"]
    assert d["sys_cover_px_max"] == 0  # menu bar + Dock never covered
    assert d["outside_own_px_max"] == 0  # no mask larger than its window's visible content
    assert d["front_leak_px_max"] == 0  # front window never covered for a window behind
    assert d["max_mask_frac"] < 0.40


@pytest.mark.parametrize("name", ["desk_static_photo", "desk_static_gradient", "desk_static_solid"])
def test_one_flagged_picture_costs_about_its_area(name: str) -> None:
    _seq, _out, s = _engine(name)
    d = s["desktop"]
    assert d["one_flag_frames"] > 0
    assert d["one_flag_area_ratio_max"] <= 1.2  # was 1.00 of the screen for 0.08 of it
    assert s["exposure"]["frames"] == 0


def test_clean_desktop_over_mask_after_results_land() -> None:
    _seq, _out, s = _engine("desk_clean")
    assert s["desktop"]["clean_over_mask_max"] <= 0.02


def test_flagged_then_clean_page_same_layout_has_no_masks() -> None:
    seq, out, s = _engine("desk_nav_clean", n=16)
    switch = 8
    assert any(m.verdict is VerdictState.UNSAFE for m in out[switch - 1])
    for i in range(switch + 3, len(out)):
        assert not any(m.verdict is VerdictState.UNSAFE for m in out[i]), (i, [m.xyxy for m in out[i]])
    assert s["desktop"]["clean_over_mask_max"] <= 0.02


def test_drag_window_across_flagged_picture() -> None:
    """Notes is dragged over the flagged picture and back: the mask stays on
    the picture's visible part and never follows the dragged window."""
    seq, out, s = _engine("desk_drag", n=16)
    assert s["exposure"]["frames"] == 0
    assert s["desktop"]["front_leak_px_max"] == 0
    flagged = seq.gt[0].flagged()[0].xyxy
    for i, ms in enumerate(out):
        notes = next(w for w in seq.gt[i].desktop["truth"].windows if w.window_id == 102).xyxy
        for m in ms:
            if m.verdict is VerdictState.UNSAFE:
                assert inter(m.xyxy, notes) is None, (i, m.xyxy)
                assert inter(m.xyxy, flagged) is not None


@pytest.mark.parametrize("name", ["desk_cmdtab", "desk_scroll_clock", "desk_open_close"])
def test_desktop_sequences_cover_the_flagged_picture(name: str) -> None:
    _seq, _out, s = _engine(name, n=16)
    assert s["exposure"]["rate"] is None or s["exposure"]["rate"] <= 0.2


def test_clock_tick_after_scroll_does_not_remerge() -> None:
    seq, out, _s = _engine("desk_scroll_clock", n=18)
    W, H = seq.width, seq.height
    for ms in out[12:]:
        assert all((m.xyxy[2] - m.xyxy[0]) * (m.xyxy[3] - m.xyxy[1]) < 0.4 * W * H for m in ms)


def test_medium_adaptive_prefrost_only_on_heavy_windows() -> None:
    core = EngineCore(prefrost_mode="adaptive", heavy_masks=2, heavy_window_s=10.0)
    assert core.prefrost and not core.heavy_window(5)
    core.t_ms = 100.0
    core.heavy.observe(5, "a", 100.0)
    core.heavy.observe(5, "b", 100.0)
    assert core.heavy_window(5) and core.heavy_set() == {5}
    core.t_ms = 20_000.0
    assert not core.heavy_window(5)


def test_capture_worker_survives_errors_and_reports_them() -> None:
    from edge.live_engine import CaptureWorker, LiveEngine
    from engine_synth import NullAnalyzer

    class Boom:
        available = True
        reason = ""
        geom = None
        n = 0

        def read(self):
            self.n += 1
            raise RuntimeError("grab exploded")

    cap = Boom()
    eng = LiveEngine(capture=cap, analyzer=NullAnalyzer(ProtectionMode.AGGRESSIVE), mode=ProtectionMode.AGGRESSIVE)
    w = CaptureWorker(eng)
    w.start()
    time.sleep(0.8)
    assert w.is_alive() and cap.n >= 2
    st = eng.status()
    assert "grab exploded" in (st["error"] or "")
    assert "warning" in st and "analyzer_error" in st
    eng.request_stop("test")
    w.join(2.0)
