"""Overlay controller (memory backend) and the threaded engine, no display."""

from __future__ import annotations

import time

import numpy as np
import pytest

from edge.coords import DisplayGeom
from edge.live_engine import LiveEngine, load_analyzer
from edge.screen_overlay import MaskOverlay, MemoryMaskBackend, overlay_level
from engine_synth import iou, make_page
from vision_core.contracts import Frame
from vision_core.live.contracts import AnalyzeResult, MaskSpec, MaskStyle, ProtectionMode, VerdictState


def _retina() -> DisplayGeom:
    return DisplayGeom(
        left_pt=0.0, top_pt=0.0, width_pt=1440.0, height_pt=900.0,
        width_px=2880, height_px=1800, scale_x=2.0, scale_y=2.0,
        cocoa_origin_x=0.0, cocoa_origin_y=0.0, cocoa_height_pt=900.0,
    )


def test_overlay_maps_retina_px_to_points_and_moves_views() -> None:
    backend = MemoryMaskBackend()
    overlay = MaskOverlay(backend=backend, pad_pt=4.0)
    geom = _retina()
    m = MaskSpec("r1", (200.0, 100.0, 600.0, 400.0), style=MaskStyle.FROST)
    assert overlay.update([m], geom) is True
    (rect, style), = backend.items.values()
    x, y, w, h = rect
    # 400×300 px at (200,100) on a 2× display = 200×150 pt, y flipped, padded
    assert (x, w, h) == pytest.approx((96.0, 208.0, 158.0))
    assert y == pytest.approx(900.0 - 50.0 - 150.0 - 4.0)
    assert style == "frost"
    assert backend.created == 1

    # Same masks → no backend call at all.
    assert overlay.update([m], geom) is False
    # Scroll: same id, new place → a move, not a new view.
    for dy in range(1, 6):
        overlay.update([MaskSpec("r1", (200.0, 100.0 - 40 * dy, 600.0, 400.0 - 40 * dy))], geom)
    assert backend.created == 1 and backend.moved == 5
    # Style change recreates; removal removes.
    overlay.update([MaskSpec("r1", (0, 0, 10, 10), style=MaskStyle.OPAQUE)], geom)
    assert backend.created == 2 and backend.items["r1"][1] == "opaque"
    overlay.update([], geom)
    assert backend.removed == 1 and overlay.mask_count() == 0
    overlay.close()
    assert backend.closed


def test_overlay_level_is_above_menus() -> None:
    assert overlay_level() > 101


class _FakeCapture:
    """Scrolls a synthetic page 40 px per read; monotonic t_ms."""

    def __init__(self, page: np.ndarray, height: int, fps: float = 60.0) -> None:
        self.page = page
        self.h = height
        self.i = 0
        self.available = True
        self.reason = ""
        self.geom = DisplayGeom(0, 0, page.shape[1], height, page.shape[1], height, 1, 1, 0, 0, height)
        self.exclude_window_ids: list[int] = []
        self.offsets: dict[int, int] = {}
        self._t0 = time.perf_counter()
        self._dt = 1.0 / fps

    def read(self):
        time.sleep(self._dt)
        off = min(40 * self.i, self.page.shape[0] - self.h)
        img = np.ascontiguousarray(self.page[off : off + self.h])
        self.offsets[id(img)] = off
        self.i += 1
        return Frame(self.i, (time.perf_counter() - self._t0) * 1000.0, img, img.shape[1], img.shape[0])

    def release(self) -> None:
        self.available = False


class _SlowKnownAnalyzer:
    def __init__(self, cap: _FakeCapture, rect, delay_s: float) -> None:
        self.cap = cap
        self.rect = rect
        self.delay = delay_s
        self.mode = ProtectionMode.AGGRESSIVE
        self.modes: list[ProtectionMode] = []
        self.calls = 0

    def set_mode(self, mode):
        self.mode = mode
        self.modes.append(mode)

    def analyze(self, frame, *, changed=None, t_ms=0.0):
        self.calls += 1
        off = self.cap.offsets[id(frame)]
        time.sleep(self.delay)
        x1, y1, x2, y2 = self.rect
        box = (x1, max(0, y1 - off), x2, min(frame.shape[0], y2 - off))
        if box[3] - box[1] < 8:
            return AnalyzeResult()
        return AnalyzeResult(masks=[MaskSpec("p", box, verdict=VerdictState.UNSAFE, reason="UNSAFE_PICTURE")])


def test_threaded_engine_follows_scroll_with_slow_analyzer() -> None:
    rect = (200, 150, 600, 450)  # fully on screen at the first look
    page = make_page(2400, 960, [rect], seed=4)
    cap = _FakeCapture(page, 600)
    analyzer = _SlowKnownAnalyzer(cap, rect, delay_s=0.12)  # ~7 frames late
    backend = MemoryMaskBackend()
    engine = LiveEngine(capture=cap, analyzer=analyzer, mode=ProtectionMode.AGGRESSIVE,
                        overlay=MaskOverlay(backend=backend))
    engine.start_workers()
    try:
        worst = 1.0
        checked = 0
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and cap.i < 14:
            engine.tick()
            with engine.lock:
                masks = engine.core.visible()
                seq = engine.core.seq
                results = engine.core.results
            if results and masks and cap.i < 14:
                # Ground truth for the frame the core is on (≥ the shown one).
                off = min(40 * seq, page.shape[0] - 600)
                want = (rect[0], max(0, rect[1] - off), rect[2], min(600, rect[3] - off))
                if want[3] - want[1] > 60:
                    worst = min(worst, iou(masks[0].xyxy, want))
                    checked += 1
            time.sleep(0.005)
    finally:
        engine.join_workers()
    assert analyzer.calls >= 2
    assert checked > 0 and worst >= 0.8, (checked, worst)
    assert backend.created == 1 and backend.moved >= 3  # one view, moved
    status = engine.status()
    assert status["mode"] == "aggressive" and status["status"] == "PROTECTING"


def test_engine_mode_off_and_pause_clear_masks() -> None:
    rect = (100, 50, 400, 300)
    page = make_page(1600, 640, [rect], seed=6)
    cap = _FakeCapture(page, 400)
    analyzer = _SlowKnownAnalyzer(cap, rect, delay_s=0.0)
    engine = LiveEngine(capture=cap, analyzer=analyzer, mode=ProtectionMode.AGGRESSIVE)
    engine.start_workers()
    try:
        assert _until(lambda: engine.core.visible())
        engine.set_mode(ProtectionMode.OFF)
        assert engine.core.visible() == []
        assert engine.status()["status"] == "OFF" and engine.status()["mood"] == "sleep"
        assert _until(lambda: ProtectionMode.OFF in analyzer.modes)  # applied on the vision thread
        engine.set_mode(ProtectionMode.MEDIUM)
        engine.pause()
        assert engine.core.visible() == [] and engine.status()["status"] == "PAUSED"
        engine.resume()
        assert engine.status()["status"] == "PROTECTING"
    finally:
        engine.join_workers()
    assert not any(t.is_alive() for t in engine._threads)


def _until(pred, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_load_analyzer_is_real_or_raises() -> None:
    analyzer, kind = load_analyzer(ProtectionMode.MEDIUM)
    assert kind == "live"
    assert analyzer.mode is ProtectionMode.MEDIUM
    assert type(analyzer).__name__ == "Analyzer"


def test_missing_analyzer_is_loudly_not_protecting(monkeypatch) -> None:
    import vision_core.live.analyzer as va

    from edge.live_engine import AnalyzerUnavailable, LiveEngine

    def boom(*a, **k):
        raise RuntimeError("model file missing")

    monkeypatch.setattr(va, "make_analyzer", boom)
    try:
        load_analyzer(ProtectionMode.AGGRESSIVE)
    except AnalyzerUnavailable as exc:
        assert "model file missing" in str(exc)
    else:
        raise AssertionError("must raise, never fall back to a no-mask stand-in")
    cap = type("C", (), {"available": True, "reason": "", "geom": None})()
    eng = LiveEngine(capture=cap, analyzer=None, mode=ProtectionMode.AGGRESSIVE)
    eng.analyzer_error = "analyzer unavailable: model file missing"
    st = eng.status()
    assert st["status"] == "NOT_PROTECTING" and st["reason"] == "ANALYZER_UNAVAILABLE"
    assert "model file missing" in st["error"] and st["backend"] == "none"
    eng.start_workers()
    assert eng._threads == []
