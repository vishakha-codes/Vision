"""Cloud lane (L2) + agent (L3) wiring in the analyzer and the engine, with a
fake transport / fake agent. Without cloud config everything stays local."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from edge.cloud_glue import CloudGlue, cloud_wanted
from edge.live_engine import LiveEngine
from vision_core.live.analyzer import make_analyzer
from vision_core.live.cloud_lane import (
    AGENT_CLEAR,
    AGENT_CONFIRM,
    REASON_LARGE_WEAK,
    REASON_LOCAL_UNAVAILABLE,
    REASON_UNSURE,
    CloudLane,
    CloudLaneConfig,
)
from vision_core.live.contracts import ProtectionMode, ScreenLayout, VerdictState, WindowInfo
from vision_core.live.detector import RawDet
from engine_synth import NullAnalyzer
from vision_core.live.synth import photo_texture

LABEL = "FEMALE_BREAST_EXPOSED"


def _frame(seed: int = 0, size=(400, 300), frame_wh=(1200, 800)):
    img = np.full((frame_wh[1], frame_wh[0], 3), 245, np.uint8)
    w, h = size
    img[150 : 150 + h, 100 : 100 + w] = photo_texture(h, w, np.random.default_rng(seed))
    return img


class Weak:
    """Every crop: one weak hit (UNSURE in aggressive: 0.15 ≤ s < 0.25)."""

    def __init__(self, score: float = 0.2) -> None:
        self.score = score
        self.calls = 0

    def detect_batch(self, images):
        self.calls += 1
        return [[RawDet(LABEL, self.score, (0.0, 0.0, im.shape[1] * 0.5, im.shape[0] * 0.5))] for im in images]


class Broken:
    def detect_batch(self, images):
        raise RuntimeError("onnx session lost")


class FakeLane:
    """Records submits; ``answers`` are CloudResult-likes handed out by apply."""

    def __init__(self) -> None:
        self.submits: list[dict] = []
        self.state = "closed"
        self.counts = {"submitted": 0}
        self.injected: list[dict] = []
        self.budget = None
        self.closed = False

    def submit(self, **kw):
        self.submits.append(kw)
        self.counts["submitted"] += 1
        return True

    def apply(self, cache, now_ms):
        return []

    def inject_verdict(self, **kw):
        self.injected.append(kw)

    def set_budget(self, rate, burst=None):
        self.budget = rate

    def close(self):
        self.closed = True


def _lane(transport):
    cfg = CloudLaneConfig(url="https://l2.test:8443", token="t" * 16)
    return CloudLane(cfg, transport=transport)


def _wait(cond, timeout=3.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


# --------------------------------------------------------------------------- analyzer


def test_no_lane_is_exactly_local() -> None:
    an = make_analyzer(ProtectionMode.AGGRESSIVE, detector=Weak(), warmup=False)
    r = an.analyze(_frame(), t_ms=0.0)
    assert "n_cloud_applied" not in r.counts and "n_cloud_submitted" not in r.counts
    an2 = make_analyzer(ProtectionMode.AGGRESSIVE, detector=Broken(), warmup=False)
    with pytest.raises(RuntimeError):
        an2.analyze(_frame(), t_ms=0.0)  # as before: the vision worker reports it


def test_unsure_after_second_look_goes_to_the_cloud_and_a_clear_lands() -> None:
    seen = []

    def transport(body, timeout):
        seen.append(body)
        return {"verdict": {"state": "SAFE", "score": 0.0, "labels": []}, "dets": [], "source": "cloud",
                "timings_ms": {"total": 12.0}, "n_views": 3}

    lane = _lane(transport)
    an = make_analyzer(ProtectionMode.AGGRESSIVE, detector=Weak(), warmup=False, cloud_lane=lane)
    frame = _frame()
    r1 = an.analyze(frame, t_ms=0.0)
    pics = [r for r in r1.regions if r.kind == "picture"]
    assert pics and r1.verdicts[pics[0].region_id].state is VerdictState.UNSURE
    assert r1.counts["n_cloud_submitted"] >= 1
    assert _wait(lambda: lane.counts["ok"] >= 1)
    assert seen[0]["reason"] == REASON_UNSURE and "jpeg_b64" in seen[0] and "title" not in seen[0]
    r2 = an.analyze(frame, t_ms=100.0)
    assert r2.counts["n_cloud_applied"] >= 1
    assert r2.verdicts[pics[0].region_id].state is VerdictState.SAFE  # the cloud's clean look cleared it
    lane.close()


def test_agent_confirm_makes_it_unsafe() -> None:
    lane = _lane(lambda body, t: {"verdict": {"state": "UNSURE", "score": 0.2, "labels": [LABEL]}, "dets": []})
    an = make_analyzer(ProtectionMode.AGGRESSIVE, detector=Weak(), warmup=False, cloud_lane=lane)
    frame = _frame(1)
    r1 = an.analyze(frame, t_ms=0.0)
    rid = next(r.region_id for r in r1.regions if r.kind == "picture")
    key = next(iter(an.escalations))
    meta = an.escalations[key]
    assert _wait(lambda: lane.counts["ok"] >= 1)  # the L2 answer (UNSURE) first
    an.analyze(frame, t_ms=50.0)
    lane.inject_verdict(dhash=key[0], ahash=key[1], wh=meta["wh"], verdict=AGENT_CONFIRM, labels=(LABEL,))
    r2 = an.analyze(frame, t_ms=100.0)
    assert r2.verdicts[rid].state is VerdictState.UNSAFE
    lane.inject_verdict(dhash=key[0], ahash=key[1], wh=meta["wh"], verdict=AGENT_CLEAR)
    lane.close()


def test_large_weak_safe_picture_is_escalated() -> None:
    lane = FakeLane()
    an = make_analyzer(ProtectionMode.MEDIUM, detector=Weak(0.1), warmup=False, cloud_lane=lane)
    an.analyze(_frame(2, size=(950, 500), frame_wh=(2400, 1600)), t_ms=0.0)
    assert any(s["reason"] == REASON_LARGE_WEAK for s in lane.submits)
    assert all(s["crop_bgr"].shape[1] >= 900 for s in lane.submits if s["reason"] == REASON_LARGE_WEAK)


def test_local_detector_failure_leaves_pending_and_escalates() -> None:
    lane = FakeLane()
    an = make_analyzer(ProtectionMode.AGGRESSIVE, detector=Broken(), warmup=False, cloud_lane=lane)
    r = an.analyze(_frame(3), t_ms=0.0)
    pics = [x for x in r.regions if x.kind == "picture"]
    assert pics and all(r.verdicts[p.region_id].state is VerdictState.PENDING for p in pics)
    assert lane.submits and all(s["reason"] == REASON_LOCAL_UNAVAILABLE for s in lane.submits)
    assert an.detector_error and "onnx" in an.detector_error
    assert any(m.reason == "PENDING_PREFROST" for m in r.masks)  # aggressive keeps covering


def test_cloud_answer_for_a_big_region_lands_in_its_window_scoped_entry() -> None:
    """Regions over 20 % of the screen are cached per window; a cloud answer
    for one goes into that window's entry, never a global one."""
    lane = _lane(lambda body, t: {"verdict": {"state": "SAFE", "score": 0.0, "labels": []}, "dets": []})
    an = make_analyzer(ProtectionMode.AGGRESSIVE, detector=Weak(), warmup=False, cloud_lane=lane)
    frame = _frame(4, size=(700, 450))  # 0.33 of the 1200×800 frame → a big (window-scoped) region
    layout = ScreenLayout((1200, 800), (WindowInfo(7, (0.0, 0.0, 1200.0, 800.0)),), (0.0, 0.0, 1200.0, 800.0))
    an.analyze(frame, t_ms=0.0, windows=layout)
    assert an._scoped, "the 33 % picture is a window-scoped region"
    assert _wait(lambda: lane.counts["ok"] >= 1)
    an.analyze(frame, t_ms=100.0, windows=layout)
    key = next(iter(an._scoped))
    scope, wh = an._scoped[key]
    assert an.cache.get(key[0], key[1], wh, 200.0) is None  # nothing global
    assert an.cache.get(key[0], key[1], wh, 200.0, scope=scope) is not None
    lane.close()


# --------------------------------------------------------------------------- engine glue


class FakeAgent:
    def __init__(self, raise_to: str | None = None) -> None:
        self.events: list[tuple[str, dict]] = []
        self.raise_to = raise_to
        self.flushed = 0

    def stubborn_unsure(self, **kw):
        self.events.append(("stubborn_unsure", kw))
        return True

    def heavy_page(self, **kw):
        self.events.append(("heavy_page", kw))
        return True

    def session_summary(self, **kw):
        self.events.append(("session_summary", kw))
        return True

    def flush_pending(self):
        self.flushed += 1
        return 0

    def effective_mode(self, pin_mode, app):
        if self.raise_to and app == "com.example.browser":
            return self.raise_to
        return pin_mode


def _engine(mode=ProtectionMode.MEDIUM):
    cap = SimpleNamespace(available=True, reason="", geom=None)
    eng = LiveEngine(capture=cap, analyzer=NullAnalyzer(mode), mode=mode)
    layout = ScreenLayout((1200, 800), (WindowInfo(5, (0.0, 0.0, 800.0, 600.0), owner_pid=0, owner="Example Browser"),),
                          (0.0, 0.0, 1200.0, 800.0))
    eng.core.layout = layout
    return eng


def test_cloud_off_without_config_and_under_pytest() -> None:
    assert not cloud_wanted()
    eng = _engine()
    assert CloudGlue.from_config(eng) is None
    st = eng.status()
    assert st["backend"] == "local" and st["warning"] is None


def test_glue_backend_strings() -> None:
    eng = _engine()
    lane = FakeLane()
    g = CloudGlue(eng, lane=lane)
    eng.cloud = g
    assert eng.status()["backend"] == "local+COOL"
    lane.state = "open"
    assert eng.status()["backend"] == "COOL offline"


def test_unsure_cloud_answer_becomes_a_stubborn_unsure_event_without_titles() -> None:
    eng = _engine()
    lane, agent = FakeLane(), FakeAgent()
    g = CloudGlue(eng, lane=lane, agent=agent)
    eng.analyzer.escalations = {(11, 22): {"window_id": 5, "l1_score": 0.2, "labels": (LABEL,), "wh": (400, 300)}}
    res = SimpleNamespace(dhash=11, ahash=22, wh=(400.0, 300.0), region_id="r1", state="UNSURE", score=0.22,
                          labels=(LABEL,), source="cloud", reason=REASON_UNSURE, rtt_ms=40.0, late=False)
    g.on_cloud(res)
    kind, kw = agent.events[-1]
    assert kind == "stubborn_unsure"
    assert kw["app"] == "example-browser"  # process name (no bundle for pid 0), never a title
    assert kw["mode_floor"] == "medium" and kw["l1_score"] == 0.2 and kw["l2_state"] == "UNSURE"
    assert kw["context"]["dhash"] == 11
    # A SAFE answer does not bother the agent.
    g.on_cloud(SimpleNamespace(**{**res.__dict__, "state": "SAFE"}))
    assert len(agent.events) == 1


def test_agent_confirm_clear_policy_and_mode_raise() -> None:
    eng = _engine(ProtectionMode.MEDIUM)
    lane, agent = FakeLane(), FakeAgent(raise_to="aggressive")
    g = CloudGlue(eng, lane=lane, agent=agent)
    eng.cloud = g
    g._front_app = "com.example.browser"
    ctx = {"dhash": 11, "ahash": 22, "wh": (400, 300), "labels": (LABEL,), "app": "com.example.browser"}
    g.on_agent({"type": "stubborn_unsure", "verdict": "CONFIRM", "policy": {"escalation_rate_per_s": 1.5}}, ctx)
    assert lane.injected[-1]["verdict"] == "CONFIRM" and lane.budget == 1.5
    assert eng.mode is ProtectionMode.AGGRESSIVE and eng.pin_mode is ProtectionMode.MEDIUM  # raised, PIN kept
    g.on_agent({"type": "stubborn_unsure", "verdict": "CLEAR"}, ctx)
    assert lane.injected[-1]["verdict"] == "CLEAR"
    agent.raise_to = None
    g.refresh_mode("com.example.browser")  # policy gone: back to the PIN mode
    assert eng.mode is ProtectionMode.MEDIUM
    # Nothing can take protection below the PIN mode.
    eng.apply_effective_mode(ProtectionMode.OFF)
    assert eng.mode is ProtectionMode.MEDIUM


def test_heavy_page_and_session_summary_cadence() -> None:
    now = [0.0]
    eng = _engine()
    agent = FakeAgent()
    g = CloudGlue(eng, lane=None, agent=agent, clock=lambda: now[0])
    reg = SimpleNamespace(region_id="pA", kind="picture", window_id=5)
    from vision_core.live.contracts import MaskSpec, Verdict

    result = SimpleNamespace(regions=[reg], verdicts={"pA": Verdict(VerdictState.UNSAFE)},
                             masks=[MaskSpec("pA", (0, 0, 10, 10), reason="UNSAFE_PARTS")], counts={})
    g.on_result(result)
    now[0] = 61.0
    g.tick()
    heavy = [kw for k, kw in agent.events if k == "heavy_page"]
    assert heavy and heavy[0]["app"] == "example-browser" and heavy[0]["pictures_per_min"] > 0
    now[0] = 61.0 + 30 * 60
    g.tick()
    summ = [kw for k, kw in agent.events if k == "session_summary"]
    assert summ and summ[0]["counts"].get("UNSAFE_PARTS") == 1
    g.close()  # on quit: one more summary
    assert sum(1 for k, _ in agent.events if k == "session_summary") == 2


def test_fake_lane_round_trip_through_the_engine_loop() -> None:
    """A fake L2 that answers CLEAR for the first picture: the verdict lands
    on the next analyzer call (the run used for the integration report)."""
    lane = _lane(lambda body, t: {"verdict": {"state": "SAFE", "score": 0.0, "labels": []}, "dets": [],
                                  "source": "cloud"})
    an = make_analyzer(ProtectionMode.AGGRESSIVE, detector=Weak(), warmup=False, cloud_lane=lane)
    from edge.live_engine import run_headless

    frames = [_frame(5) for _ in range(6)]
    out = run_headless(frames, an, t_ms=[i * 300.0 for i in range(6)])
    unsure_first = any(m.verdict is VerdictState.UNSURE for m in out[0])
    assert unsure_first
    assert _wait(lambda: lane.counts["applied"] >= 1 or lane.counts["ok"] >= 1)
    out2 = run_headless([_frame(5)] * 3, an, t_ms=[2000.0, 2300.0, 2600.0])
    assert not any(m.verdict in (VerdictState.UNSURE, VerdictState.UNSAFE) for m in out2[-1])
    lane.close()


def test_connect_timeouts_say_cool_unreachable() -> None:
    import httpx

    def transport(body, timeout):
        raise httpx.ConnectTimeout("timed out")

    cfg = CloudLaneConfig(url="https://l2.test:8443", token="t" * 16, breaker_failures=2, breaker_open_s=30.0)
    lane = CloudLane(cfg, transport=transport)
    for i in range(3):
        crop = np.random.default_rng(i).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        lane.submit(region_id=f"r{i}", dhash=i + 1, ahash=i + 7, wh=(64, 64), crop_bgr=crop, mode="aggressive")
        _wait(lambda: lane.counts["timeouts"] + lane.counts["failed"] >= i + 1 or lane.state == "open", 1.0)
    assert lane.state == "open"
    assert lane.problem == "COOL unreachable (IP not allowed?)"
    eng = _engine()
    g = CloudGlue(eng, lane=lane)
    eng.cloud = g
    g.tick()
    assert eng.status()["backend"] == "COOL unreachable (IP not allowed?)"
    assert any(e["event"] == "cloud_unreachable" for e in g.events)
    lane.close()
