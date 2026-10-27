"""Agentic controller: visual evidence must change the next tool."""

from __future__ import annotations

import json
from dataclasses import asdict

import cv2
import numpy as np

from agent.bedrock_adapter import BedrockAdapter, BedrockUnavailable
from agent.orchestrator import Orchestrator
from vision_core.contracts import ActionType, Detection, make_trace, trace_to_dict
from vision_core.policy import load_policy
from vision_core.reinspect_crop import ReinspectResult


def _blank(w: int = 160, h: int = 120) -> np.ndarray:
    return np.full((h, w, 3), 28, dtype=np.uint8)


def _detect(score: float, label: str = "flagged"):
    def _fn(image, frame_id: int = 0, stage: str = "detect", **_kwargs):
        del image, stage
        return [
            Detection(
                label=label,
                score=score,
                xyxy=(20.0, 20.0, 100.0, 100.0),
                detector_id="test",
                frame_id=frame_id,
            )
        ]

    return _fn


def _reinspect(score: float, calls: list | None = None):
    def _fn(image, box, frame_id: int = 0, **_kwargs):
        del image
        if calls is not None:
            calls.append(frame_id)
        return ReinspectResult(
            label="flagged",
            score=score,
            xyxy=box,
            detections=[],
            crop_xyxy=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
        )

    return _fn


def _tools(result) -> list[str]:
    return [t.tool for t in result.traces]


def test_high_score_skips_reinspect() -> None:
    calls: list[int] = []
    orch = Orchestrator(
        load_policy("policies/balanced.json"),
        detect_fn=_detect(0.90),
        reinspect_fn=_reinspect(0.99, calls),
    )
    result = orch.process_frame(0, _blank(), None, 0.0)
    assert "cool.reinspect_crop" not in _tools(result)
    assert calls == []
    assert result.reinspect_count == 0
    assert any(a.action in (ActionType.BLUR, ActionType.OPAQUE_MASK) for a in result.actions)
    assert result.tool_sequence == "DETECT→MASK"


def test_mid_score_reinspects_then_changes_action() -> None:
    calls: list[int] = []
    orch = Orchestrator(
        load_policy("policies/balanced.json"),
        detect_fn=_detect(0.52),
        reinspect_fn=_reinspect(0.90, calls),
    )
    result = orch.process_frame(0, _blank(), None, 0.0)
    assert "cool.reinspect_crop" in _tools(result)
    assert calls == [0]
    assert result.reinspect_count == 1
    assert result.first_decision in {"REVIEW", "NO_ACTION"}
    assert result.final_decision == ActionType.BLUR.value
    assert result.first_decision != result.final_decision
    assert result.tool_sequence == "DETECT→REINSPECT→MASK"


def test_low_score_no_action_no_reinspect() -> None:
    calls: list[int] = []
    orch = Orchestrator(
        load_policy("policies/balanced.json"),
        detect_fn=_detect(0.20),
        reinspect_fn=_reinspect(0.90, calls),
    )
    result = orch.process_frame(0, _blank(), None, 0.0)
    assert "cool.reinspect_crop" not in _tools(result)
    assert calls == []
    assert result.reinspect_count == 0
    assert all(a.action in (ActionType.NO_ACTION, ActionType.REVIEW) for a in result.actions) or result.actions == []
    assert result.final_decision in {"NO_ACTION", "REVIEW"}
    assert result.tool_sequence == "DETECT→NO_ACTION"


def test_scene_cut_releases() -> None:
    orch = Orchestrator(
        load_policy("policies/balanced.json"),
        detect_fn=_detect(0.90),
        reinspect_fn=_reinspect(0.90),
    )
    black = np.zeros((80, 120, 3), dtype=np.uint8)
    white = np.full((80, 120, 3), 255, dtype=np.uint8)
    first = orch.process_frame(0, black, None, 0.0)
    assert first.cut is False
    assert first.tracks
    second = orch.process_frame(1, white, black, 33.0)
    assert second.cut is True
    assert any(a.action is ActionType.RELEASE for a in second.actions)
    assert "cool.scene_cut" in _tools(second)
    assert second.final_decision == "RELEASE"


def test_timeout_fallback_strict_never_unmasked() -> None:
    orch = Orchestrator(
        load_policy("policies/strict.json"),
        detect_fn=_detect(0.90),
        reinspect_fn=_reinspect(0.90),
        force_timeout=True,
    )
    result = orch.process_frame(0, _blank(), None, 0.0)
    assert result.timed_out is True
    assert any(a.action in (ActionType.HOLD, ActionType.FULL_WINDOW_MASK) for a in result.actions)
    assert all(a.action is not ActionType.NO_ACTION for a in result.actions)


def test_timeout_balanced_without_prior_mask_stands_down() -> None:
    orch = Orchestrator(
        load_policy("policies/balanced.json"),
        detect_fn=_detect(0.90),
        reinspect_fn=_reinspect(0.90),
        force_timeout=True,
    )
    result = orch.process_frame(0, _blank(), None, 0.0)
    assert result.timed_out is True
    assert result.final_decision == "NOT_PROTECTING"


def test_traces_never_contain_image_arrays() -> None:
    pixels = np.full((48, 64, 3), 77, dtype=np.uint8)
    orch = Orchestrator(
        load_policy("policies/balanced.json"),
        detect_fn=_detect(0.52),
        reinspect_fn=_reinspect(0.90),
    )
    result = orch.process_frame(0, pixels, None, 0.0)
    assert result.traces
    for trace in result.traces:
        assert trace.image_persisted is False
        payload = trace_to_dict(trace)
        dumped = json.dumps(payload)
        assert "image_bgr" not in dumped
        assert payload["image_persisted"] is False
        raw = json.dumps(asdict(trace), default=str)
        assert "image_persisted" in raw
        for value in payload["observation"].values():
            assert not isinstance(value, np.ndarray)
        # Raw pixel dumps would repeat the fill value thousands of times.
        assert dumped.count("77") < 8


def test_changing_the_image_changes_the_tool_sequence() -> None:
    """Real OpenCV detections: empty frame vs magenta blob."""
    empty = np.full((180, 240, 3), 28, dtype=np.uint8)
    magenta = empty.copy()
    cv2.circle(magenta, (120, 90), 36, (200, 40, 200), thickness=-1)

    empty_orch = Orchestrator(load_policy("policies/balanced.json"))
    empty_result = empty_orch.process_frame(0, empty, None, 0.0)
    color_orch = Orchestrator(load_policy("policies/balanced.json"))
    color_result = color_orch.process_frame(0, magenta, None, 0.0)

    assert empty_result.tool_sequence == "DETECT→NO_ACTION"
    assert "cool.reinspect_crop" not in _tools(empty_result)
    assert color_result.tool_sequence in {"DETECT→REINSPECT→MASK", "DETECT→MASK"}
    assert empty_result.tool_sequence != color_result.tool_sequence


def test_make_trace_strips_injected_pixels() -> None:
    pixels = np.full((16, 16, 3), 90, dtype=np.uint8)
    trace = make_trace(
        frame_id=1,
        tool="cool.reinspect_crop",
        observation={"score": 0.61, "oops": pixels, "image_bgr": pixels},
        decision="uncertain",
        next_action="policy.evaluate",
        latency_ms=1.2,
    )
    dumped = json.dumps(trace_to_dict(trace))
    assert "image_bgr" not in dumped
    assert not isinstance(trace.observation.get("oops"), np.ndarray)


def test_bedrock_adapter_uses_guardrail_and_falls_back_on_model_errors() -> None:
    calls = []

    class FakeClient:
        def converse(self, **req):
            calls.append(req)
            if req["modelId"] == "first":
                raise RuntimeError("AccessDenied")
            return {"output": {"message": {"content": [{"text": "OK"}]}}, "stopReason": "end_turn", "usage": {}}

    adapter = BedrockAdapter(models=("first", "second"), guardrail_id="gr1", guardrail_version="2", client=FakeClient())
    out = adapter.invoke("hello")
    assert out["text"] == "OK" and out["model_id"] == "second"
    assert all(c["guardrailConfig"]["guardrailIdentifier"] == "gr1" for c in calls)

    class Down:
        def converse(self, **_req):
            raise RuntimeError("no credentials")

    try:
        BedrockAdapter(models=("a",), client=Down()).converse("hi")
    except BedrockUnavailable as exc:
        assert "no credentials" in str(exc)
    else:
        raise AssertionError("expected BedrockUnavailable")
