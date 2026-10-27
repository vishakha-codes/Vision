"""edge.redacted_capture against a fake engine, plus the real EngineCore (window-clipped masks)."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")

from edge import redacted_capture as rc  # noqa: E402
from kids_chat.snapshot import SnapshotRefused, verify  # noqa: E402


class FakeMasks:
    def __init__(self, stale=0):
        self.stale = stale

    def stale_count(self):
        return self.stale


def fake_engine(**over):
    frame = np.zeros((900, 1440, 3), np.uint8)
    frame[:, :] = (240, 240, 240)
    frame[100:400, 1000:1300] = np.random.default_rng(1).integers(0, 255, (300, 300, 3), dtype=np.uint8)  # "unsafe" picture
    mask = SimpleNamespace(xyxy=(1000.0, 100.0, 1300.0, 400.0), verdict=SimpleNamespace(name="UNSAFE"))
    core = SimpleNamespace(results=5, _last_frame=frame, masks=FakeMasks(), _deferred=[], pending_change=None, urgent=False,
                           last_result_seq=10, change_seq=10, seq=12, visible=lambda: [mask])
    geom = SimpleNamespace(left_pt=0.0, top_pt=0.0, width_pt=1440.0, height_pt=900.0)
    e = SimpleNamespace(lock=threading.Lock(), core=core, idle=False, capture_ok=True, geom=geom,
                        last_frame_mono=time.monotonic())
    for k, v in over.items():
        setattr(core if hasattr(core, k) else e, k, v)
    return e


def cap(engine, **kw):
    kw.setdefault("recheck", lambda bgr: False)
    kw.setdefault("opted_in", lambda: True)
    return rc.RedactedCapture(engine, **kw)


def test_snapshot_paints_masks_and_passes_client_verify():
    r = cap(fake_engine()).snapshot()
    assert r["ok"] is True and r["w"] == 1280 and r["h"] == 800 and r["all_checked"] is True
    snap = verify(r)  # the chat service's own check: masks must be solid
    assert snap.display == {"x": 0.0, "y": 0.0, "w": 1440.0, "h": 900.0}
    assert len(snap.masks_px) == 1


def test_engine_bug_that_skips_painting_is_caught_by_client(monkeypatch):
    monkeypatch.setattr(rc, "paint", lambda frame, boxes, pad=6: (frame.copy(), [list(b) for b in boxes]))
    r = cap(fake_engine()).snapshot()
    assert r["ok"] is True
    with pytest.raises(SnapshotRefused) as e:
        verify(r)
    assert e.value.reason == "mask_not_painted"


def test_capture_ahead_of_the_analyzer_on_a_still_screen_is_allowed():
    """Live, capture runs a frame or two ahead of the analyzer (seq 12 vs 10).
    With no change since the analyzed frame the snapshot must go through."""
    r = cap(fake_engine(last_result_seq=10, change_seq=8, seq=12)).snapshot()
    assert r["ok"] is True


@pytest.mark.parametrize("over,reason", [
    ({"idle": True}, "not_protecting"),
    ({"capture_ok": False}, "not_protecting"),
    ({"results": 0}, "no_frame"),
    ({"last_frame_mono": time.monotonic() - 5}, "stale_frame"),
    ({"masks": FakeMasks(stale=1)}, "unchecked_regions"),
    ({"last_result_seq": 9}, "unchecked_regions"),  # a change at frame 10 is not judged yet
    ({"urgent": True}, "unchecked_regions"),
])
def test_refusals(over, reason, monkeypatch):
    monkeypatch.setattr(rc, "CATCH_UP_S", 0.1)
    e = fake_engine(**over)
    if "last_frame_mono" in over:
        e.last_frame_mono = over["last_frame_mono"]
    r = cap(e).snapshot()
    assert r == {"type": "redacted_snapshot", "ok": False, "reason": reason}


def test_refuses_without_parent_opt_in_or_detector():
    assert cap(fake_engine(), opted_in=lambda: False).snapshot()["reason"] == "disabled"
    assert cap(fake_engine(), recheck=None).snapshot()["reason"] == "no_recheck"
    assert cap(fake_engine(), recheck=lambda b: True).snapshot()["reason"] == "nsfw_recheck"


def test_missing_freshness_hook_fails_closed():
    e = fake_engine()
    del e.last_frame_mono
    assert cap(e).snapshot()["reason"] == "stale_frame"


def test_three_frames():
    r = cap(fake_engine()).snapshot(frames=3, interval_ms=200)
    assert r["ok"] and len(r["frames"]) == 3


# --------------------------------------------------------------------------- the real engine


def _real_engine():
    from edge.live_engine import LiveEngine
    from vision_core.live.contracts import AnalyzeResult, MaskSpec, ProtectionMode, ScreenLayout, VerdictState, WindowInfo
    from engine_synth import NullAnalyzer

    cap = SimpleNamespace(available=True, reason="", geom=None)
    eng = LiveEngine(capture=cap, analyzer=NullAnalyzer(ProtectionMode.AGGRESSIVE), mode=ProtectionMode.AGGRESSIVE)
    frame = np.full((800, 1200, 3), 240, np.uint8)
    frame[100:400, 100:500] = np.random.default_rng(2).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    # A back window with the picture; a front window covering its right part.
    layout = ScreenLayout((1200, 800), (WindowInfo(2, (350.0, 50.0, 900.0, 500.0)), WindowInfo(1, (0.0, 40.0, 1200.0, 760.0))),
                          (0.0, 40.0, 1200.0, 760.0))
    geom = SimpleNamespace(left_pt=0.0, top_pt=0.0, width_pt=1200.0, height_pt=800.0, scale=1.0)
    eng.on_frame(frame, 0.0, geom, layout)
    eng.core.take_request(0, 0.0, windows=layout)
    eng.core.apply_result(AnalyzeResult(masks=[MaskSpec("p", (100.0, 100.0, 500.0, 400.0), window_id=1)]), 0, frame)
    return eng, frame, (MaskSpec, VerdictState)


def test_real_engine_paints_window_clipped_masks_in_capture_px():
    eng, frame, _ = _real_engine()
    r = cap(eng).snapshot()
    assert r["ok"] is True, r
    # The mask was cut at the front window (x = 350): painted boxes stay left of it (+ pad).
    assert r["masks"] and all(m[2] <= (350 + rc.PAD_PX) * r["w"] / 1200.0 + 1 for m in r["masks"])
    vis = eng.core.visible()
    assert vis and all(m.xyxy[2] <= 350.0 for m in vis)


def test_real_engine_refuses_while_anything_is_pending_or_busy(monkeypatch):
    monkeypatch.setattr(rc, "CATCH_UP_S", 0.1)
    eng, frame, (MaskSpec, VerdictState) = _real_engine()
    eng.core.masks.add_prefrost((600.0, 550.0, 800.0, 700.0), eng.core.seq, window_id=1)
    eng.core._publish()
    assert cap(eng).snapshot()["reason"] == "unchecked_regions"
    eng2, _f, _ = _real_engine()
    eng2.core.warning = {"code": "busy", "text": "Lots to cover here"}
    assert cap(eng2).snapshot()["reason"] == "unchecked_regions"
