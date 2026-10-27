"""Phase 4 laptop reflex: capture, flow-hold, policy toggle, privacy."""

from __future__ import annotations

import json
from dataclasses import asdict

import cv2
import numpy as np
import pytest

from edge.capture import LiveCapture, ReplayCapture, SAFE_FIXTURE, open_source
from edge.companion import draw_companion, mood_from, say
from edge.overlay import render_overlay, window_title
from edge.http_brain import PerceiveError, parse_perceive_response
from edge.reflex import (
    STATUS_CLOUD_LOOK,
    STATUS_LOCAL_HOLD,
    STATUS_NOT_PROTECTING,
    Reflex,
)
from vision_core.contracts import (
    PROTECTIVE_ACTIONS,
    ActionType,
    Frame,
    ProtectionAction,
    TrackedRegion,
    TrackMode,
    make_trace,
    new_trace_id,
    trace_to_dict,
)


def _frame(bgr: np.ndarray, frame_id: int = 0) -> Frame:
    h, w = bgr.shape[:2]
    return Frame(frame_id=frame_id, t_ms=frame_id * 33.3, image_bgr=bgr, width=w, height=h)


def _magenta(w: int = 240, h: int = 180, cx: int = 120, cy: int = 90, radius: int = 36) -> np.ndarray:
    image = np.full((h, w, 3), 28, dtype=np.uint8)
    cv2.circle(image, (cx, cy), radius, (200, 40, 200), thickness=-1)
    return image


def _primary(actions) -> ActionType | None:
    protective = [a for a in actions if a.action in PROTECTIVE_ACTIONS]
    if protective:
        return protective[0].action
    if actions:
        return actions[0].action
    return None


def test_replay_reads_safe_fixture() -> None:
    with ReplayCapture(loop=False) as cap:
        assert cap.available
        assert cap.path == SAFE_FIXTURE.resolve()
        frame = cap.read()
        assert frame is not None
        assert frame.image_bgr.ndim == 3
        assert frame.width == frame.image_bgr.shape[1]


def test_replay_refuses_path_outside_safe(tmp_path) -> None:
    outsider = tmp_path / "other.mp4"
    outsider.write_bytes(b"nope")
    with pytest.raises(ValueError, match="fixtures/safe"):
        ReplayCapture(outsider)


def test_live_capture_unavailable_without_grabber() -> None:
    live = LiveCapture()
    assert live.available is False
    assert live.read() is None
    with pytest.raises(RuntimeError, match="capture unavailable"):
        live.require()
    opened = open_source("live")
    assert opened.read() is None


def test_none_frame_is_not_protecting() -> None:
    reflex = Reflex("balanced")
    state = reflex.step(None)
    assert state.status == STATUS_NOT_PROTECTING
    assert state.mood == "lost"
    assert state.protecting is False
    assert "NOT_PROTECTING" in window_title(state)
    canvas = render_overlay(state)
    assert canvas.shape[0] >= 16


def test_pause_is_not_protecting() -> None:
    reflex = Reflex("strict", brain_every=1)
    frame = _frame(_magenta())
    live = reflex.step(frame)
    assert live.status != STATUS_NOT_PROTECTING
    paused = reflex.step(frame, paused=True)
    assert paused.status == STATUS_NOT_PROTECTING
    assert paused.mood == "lost"
    assert paused.paused is True


def test_same_clip_policies_change_action() -> None:
    image = _magenta()
    seen: dict[str, ActionType | None] = {}
    for name in ("strict", "balanced", "relaxed"):
        reflex = Reflex(name, brain_every=1)
        state = None
        for i in range(4):
            state = reflex.step(_frame(image, i))
        assert state is not None
        seen[name] = _primary(state.actions)
    assert seen["strict"] == ActionType.OPAQUE_MASK
    assert seen["balanced"] == ActionType.BLUR
    assert seen["relaxed"] in (ActionType.NO_ACTION, ActionType.REVIEW, None)
    assert seen["strict"] != seen["relaxed"]
    assert seen["balanced"] != seen["relaxed"]


def test_policy_toggle_on_same_reflex() -> None:
    reflex = Reflex("strict", brain_every=1)
    image = _magenta()
    first = reflex.step(_frame(image, 0))
    assert _primary(first.actions) == ActionType.OPAQUE_MASK
    reflex.set_policy("relaxed")
    second = reflex.step(_frame(image, 1))
    assert second.policy_name == "relaxed"
    assert _primary(second.actions) in (ActionType.NO_ACTION, ActionType.REVIEW, None)
    assert not any(a.action in (ActionType.OPAQUE_MASK, ActionType.BLUR) for a in second.actions)


def test_flow_hold_between_brain_ticks() -> None:
    reflex = Reflex(
        "strict",
        brain_every=99,
        brain_every_ms=10_000.0,
        brain_budget_ms=10_000.0,
        change_gate=9.0,
    )
    a = reflex.step(_frame(_magenta(cx=80, cy=80), 0))
    assert a.status == STATUS_CLOUD_LOOK
    assert a.protecting
    box_a = next(x.region.xyxy for x in a.actions if x.region is not None)
    b = reflex.step(_frame(_magenta(cx=150, cy=100), 1))
    assert b.status == STATUS_LOCAL_HOLD
    assert b.held is True
    box_b = next(x.region.xyxy for x in b.actions if x.region is not None)
    assert box_a != box_b


def test_slow_brain_holds_last_mask() -> None:
    reflex = Reflex(
        "strict",
        brain_every=99,
        brain_every_ms=10_000.0,
        brain_budget_ms=0.0,
        change_gate=9.0,
    )
    first = reflex.step(_frame(_magenta(), 0))
    assert first.status == STATUS_CLOUD_LOOK
    second = reflex.step(_frame(_magenta(), 1))
    assert second.status == STATUS_LOCAL_HOLD
    assert second.held is True
    assert second.protecting is True


def test_flow_hold_does_not_drop_lock_on_moving_blob() -> None:
    """Change-gate skips detect; DIS hold must keep the mask on the blob."""
    reflex = Reflex(
        "strict",
        brain_every=99,
        brain_every_ms=10_000.0,
        brain_budget_ms=10_000.0,
        change_gate=9.0,
    )
    first = reflex.step(_frame(_magenta(cx=70, cy=80), 0))
    assert first.protecting
    last = first
    for i in range(1, 20):
        last = reflex.step(_frame(_magenta(cx=70 + i * 4, cy=80), i))
        assert last.status == STATUS_LOCAL_HOLD
        assert last.protecting
        assert last.tracks, "flow-hold must not expire tracks as misses"
    box = next(x.region.xyxy for x in last.actions if x.region is not None)
    # Blob moved ~76 px; a frozen frame-0 mask would miss it entirely.
    assert box[0] > 90.0


def test_change_gate_still_keyframes() -> None:
    reflex = Reflex(
        "strict",
        brain_every=4,
        brain_every_ms=10_000.0,
        brain_budget_ms=10_000.0,
        change_gate=9.0,
    )
    looks = []
    for i in range(9):
        state = reflex.step(_frame(_magenta(cx=80 + i, cy=90), i))
        looks.append(state.status)
    assert looks[0] == STATUS_CLOUD_LOOK
    assert looks[1] == STATUS_LOCAL_HOLD
    assert looks[4] == STATUS_CLOUD_LOOK
    assert looks.count(STATUS_CLOUD_LOOK) >= 2


def test_overlay_strict_is_opaque_relaxed_is_not() -> None:
    image = _magenta(w=640, h=360, cx=420, cy=110, radius=40)
    strict = Reflex("strict", brain_every=1).step(_frame(image, 0))
    relaxed = Reflex("relaxed", brain_every=1).step(_frame(image, 0))
    s_canvas = render_overlay(strict)
    r_canvas = render_overlay(relaxed)
    region = next(a.region for a in strict.actions if a.region is not None)
    x1, y1, x2, y2 = (int(v) for v in region.xyxy)
    s_roi = s_canvas[y1:y2, x1:x2]
    r_roi = r_canvas[y1:y2, x1:x2]
    assert float(s_roi.mean()) < 40.0
    assert float(r_roi.mean()) > 40.0
    assert "strict" in window_title(strict)
    assert "relaxed" in window_title(relaxed)


def test_companion_moods_and_say() -> None:
    assert mood_from(status=STATUS_NOT_PROTECTING) == "lost"
    covering = ProtectionAction(
        ActionType.OPAQUE_MASK,
        TrackedRegion(1, (0, 0, 8, 8), "flagged", 0.9, 0.1, 0.0, TrackMode.TRACKED),
        "FLAGGED_STRICT_BLOCK",
        "strict-v2",
        new_trace_id(),
    )
    assert mood_from(status=STATUS_CLOUD_LOOK, actions=[covering]) == "covering"
    review = ProtectionAction(
        ActionType.REVIEW,
        None,
        "UNCERTAIN_REINSPECT",
        "relaxed-v2",
        new_trace_id(),
    )
    assert mood_from(status=STATUS_CLOUD_LOOK, actions=[review], band="mid") == "unsure"
    assert mood_from(status=STATUS_CLOUD_LOOK, actions=[], band="low") == "watching"
    assert "Covering" in say("FLAGGED_STRICT_BLOCK")
    canvas = np.full((200, 320, 3), 16, dtype=np.uint8)
    draw_companion(canvas, "watching", say("NO_ACTION"))
    assert canvas.mean() > 16


def test_traces_never_include_window_title_or_pixels() -> None:
    reflex = Reflex("balanced", brain_every=1)
    pixels = _magenta()
    state = reflex.step(_frame(pixels, 0))
    title = window_title(state)
    assert title
    for trace in state.traces:
        assert trace.image_persisted is False
        payload = trace_to_dict(trace)
        dumped = json.dumps(payload)
        assert "image_bgr" not in dumped
        assert "Screen Guardian |" not in dumped
        assert title not in dumped
        assert payload["image_persisted"] is False
        raw = json.dumps(asdict(trace), default=str)
        assert "image_persisted" in raw
        for value in payload["observation"].values():
            assert not isinstance(value, np.ndarray)
    extra = make_trace(
        frame_id=0,
        tool="edge.flow_hold",
        observation={"window_title": title, "oops": pixels, "image_bgr": pixels},
        decision="HOLD",
        next_action="cool.track_and_predict",
        latency_ms=0.0,
    )
    dumped = json.dumps(trace_to_dict(extra))
    assert "image_bgr" not in dumped
    assert not isinstance(extra.observation.get("oops"), np.ndarray)


def test_http_brain_parses_mock_perceive_response() -> None:
    payload = {
        "fixture_id": "synthetic_moving_regions",
        "frame_id": 12,
        "policy_id": "balanced",
        "source": "fixture_frame",
        "image_persisted": False,
        "actions": [
            {
                "action": "BLUR",
                "reason_code": "FLAGGED_ABOVE_OVERLAY",
                "policy_version": "balanced-v2",
                "trace_id": "abc123def456",
                "region": {
                    "track_id": 1,
                    "label": "flagged",
                    "xyxy": [10.0, 20.0, 80.0, 90.0],
                    "stable_score": 0.92,
                    "mask_on": True,
                },
            }
        ],
        "trace": {
            "trace_id": "t1",
            "frame_id": 12,
            "tool": "companion.say",
            "observation": {"reason_code": "FLAGGED_ABOVE_OVERLAY"},
            "decision": "BLUR",
            "next_action": "cool.render_mask_spec",
            "latency_ms": 1.2,
            "image_persisted": False,
        },
        "traces": [
            {
                "trace_id": "t0",
                "frame_id": 12,
                "tool": "cool.reinspect_crop",
                "observation": {"score": 0.92},
                "decision": "REINSPECT",
                "next_action": "policy.evaluate",
                "latency_ms": 4.0,
                "image_persisted": False,
            }
        ],
        "tool_sequence": "DETECT→REINSPECT→MASK",
        "final_decision": "BLUR",
        "band": "mid",
        "companion_line": "Softening a flagged area.",
        "reinspect_count": 1,
        "timed_out": False,
    }
    result = parse_perceive_response(payload)
    assert result.final_decision == "BLUR"
    assert result.band == "mid"
    assert result.reinspect_count == 1
    assert result.companion_line == "Softening a flagged area."
    assert result.actions[0].action == ActionType.BLUR
    assert result.actions[0].region is not None
    assert result.actions[0].region.xyxy == (10.0, 20.0, 80.0, 90.0)
    assert result.tracks[0].track_id == 1
    assert all(trace.image_persisted is False for trace in result.traces)


def test_http_brain_error_is_not_protecting() -> None:
    class Boom:
        def perceive(self, **_kwargs):
            raise PerceiveError("HTTP 503: down", status_code=503)

    reflex = Reflex(
        "balanced",
        backend="http://127.0.0.1:9",
        http_client=Boom(),
        brain_every=1,
    )
    state = reflex.step(_frame(_magenta()))
    assert state.status == STATUS_NOT_PROTECTING
    assert state.mood == "lost"
    assert state.backend == "cloud"
    assert state.backend_error
    assert "503" in (state.backend_error or "")
    title = window_title(state)
    assert "backend=cloud" in title
    canvas = render_overlay(state)
    assert canvas.shape[0] >= 16


def test_on_mask_event_fires() -> None:
    seen: list[str] = []
    reflex = Reflex("balanced", brain_every=1, on_mask_event=lambda s: seen.append(s.status))
    reflex.step(_frame(_magenta()))
    assert seen
    assert seen[0] in (STATUS_CLOUD_LOOK, STATUS_NOT_PROTECTING)


def test_hud_shows_local_backend() -> None:
    reflex = Reflex("balanced", brain_every=1)
    state = reflex.step(_frame(_magenta()))
    assert state.backend == "local"
    assert "backend=local" in window_title(state)
