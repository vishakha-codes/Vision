"""Deterministic budgeted controller for Agentic Vision.

Visual evidence (OpenCV scores / scene cuts / timeouts) changes the next tool:

    observations
      → policy.evaluate
        → high + enabled class → APPLY (no reinspect)
        → mid/uncertain + budget left → REINSPECT_CROP → redecide
        → low → NO_ACTION
        → scene_cut → RELEASE / kill tracks
        → tool timeout / missing → HOLD or NOT_PROTECTING
          (Strict never hangs unmasked)

Bedrock is optional. This controller ships the demo without invoke.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from agent.companion import say as companion_say
from agent.guardian import request_review
from vision_core.contracts import (
    FLAGGED_LABELS,
    PROTECTIVE_ACTIONS,
    ActionType,
    AgentTrace,
    Detection,
    ProtectionAction,
    TrackedRegion,
    TrackMode,
    make_trace,
    new_trace_id,
)
from vision_core.detect_regions import detect_regions
from vision_core.extract_tiles import extract_tiles
from vision_core.live.budget import TokenBucket
from vision_core.policy import (
    PolicyConfig,
    agent_settings,
    class_enabled,
    evaluate,
    load_policy,
    score_band,
)
from vision_core.reinspect_crop import ReinspectResult, reinspect_crop
from vision_core.scene_change import scene_change
from vision_core.scene_cut import scene_cut
from vision_core.track_and_predict import expire_tracks, track_and_predict

DetectFn = Callable[..., list[Detection]]
ReinspectFn = Callable[..., ReinspectResult]


class ToolTimeout(RuntimeError):
    """A vision tool exceeded the budget or was marked missing."""

    def __init__(self, tool: str, message: str | None = None) -> None:
        self.tool = tool
        super().__init__(message or f"{tool} timed out")


@dataclass
class FrameResult:
    actions: list[ProtectionAction]
    traces: list[AgentTrace]
    detections: list[Detection]
    tracks: list[TrackedRegion]
    change_score: float
    cut: bool
    reinspect_count: int
    timed_out: bool
    timings_ms: dict[str, float]
    tool_sequence: str
    companion_line: str | None
    first_decision: str
    final_decision: str
    band: str
    n_tiles: int = 0


@dataclass
class Orchestrator:
    """Frame-by-frame tool loop. State (tracks, budget, last mask) is kept here."""

    policy: PolicyConfig | str
    detect_fn: DetectFn = detect_regions
    reinspect_fn: ReinspectFn = reinspect_crop
    force_timeout: bool = False
    timeout_ms: float | None = None

    cfg: PolicyConfig = field(init=False)
    tracks: list[TrackedRegion] = field(default_factory=list)
    last_protective: list[ProtectionAction] = field(default_factory=list)
    budget_left: int = field(init=False)
    cooldown: int = field(init=False)
    last_reinspect_frame: int = -10**9
    settings: dict[str, Any] = field(init=False)
    # Reinspect budget. Policies without ``agent.reinspect_refill_per_s`` keep
    # the fixed per-run budget (synthetic replay: 40 reinspects, IoU ~0.958).
    # Live policies (``tune_policy_for_live_nsfw``) set a refill rate, which
    # turns ``reinspect_budget`` into the burst of a token bucket on ``clock``
    # (seconds), so a long live session never runs out of second looks.
    clock: Callable[[], float] = time.monotonic
    bucket: TokenBucket | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.cfg = load_policy(self.policy)
        self.settings = agent_settings(self.cfg)
        self.budget_left = int(self.settings["reinspect_budget"])
        self.cooldown = int(self.settings["reinspect_cooldown_frames"])
        if self.timeout_ms is None:
            self.timeout_ms = float(self.settings["tool_timeout_ms"])
        refill = float((self.cfg.agent or {}).get("reinspect_refill_per_s", 0.0) or 0.0)
        if refill > 0.0:
            self.bucket = TokenBucket(rate_per_s=refill, burst=float(self.budget_left))

    def can_reinspect(self, frame_id: int) -> bool:
        if self.bucket is not None:
            self.budget_left = int(self.bucket.available(self.clock()))
        if self.budget_left <= 0:
            return False
        return (frame_id - self.last_reinspect_frame) >= self.cooldown

    def _spend_reinspect(self) -> None:
        if self.bucket is not None:
            self.bucket.take(self.clock())
            self.budget_left = int(self.bucket.tokens)
        else:
            self.budget_left -= 1

    def _trace(
        self,
        *,
        frame_id: int,
        tool: str,
        observation: dict[str, Any],
        decision: str,
        next_action: str,
        latency_ms: float,
        trace_id: str | None = None,
    ) -> AgentTrace:
        return make_trace(
            frame_id=frame_id,
            tool=tool,
            observation=observation,
            decision=decision,
            next_action=next_action,
            latency_ms=latency_ms,
            trace_id=trace_id,
        )

    def _run_tool(self, name: str, fn: Callable, *args: Any, **kwargs: Any) -> tuple[Any, float]:
        if self.force_timeout:
            raise ToolTimeout(name)
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000.0
        if self.timeout_ms is not None and dt > float(self.timeout_ms):
            raise ToolTimeout(name, f"{name} took {dt:.1f}ms > {self.timeout_ms}ms")
        return result, dt

    def process_frame(
        self,
        frame_id: int,
        bgr: np.ndarray,
        prev_bgr: np.ndarray | None,
        t_ms: float,
    ) -> FrameResult:
        traces: list[AgentTrace] = []
        timings: dict[str, float] = {}
        tid = new_trace_id()
        try:
            return self._process_inner(frame_id, bgr, prev_bgr, t_ms, traces, timings, tid)
        except ToolTimeout as exc:
            return self._timeout_fallback(frame_id, traces, timings, exc.tool, tid)

    def _process_inner(
        self,
        frame_id: int,
        bgr: np.ndarray,
        prev_bgr: np.ndarray | None,
        t_ms: float,
        traces: list[AgentTrace],
        timings: dict[str, float],
        tid: str,
    ) -> FrameResult:
        if prev_bgr is None:
            change = None
            change_score = 1.0
            cut = False
            timings["scene_change"] = 0.0
        else:
            change, dt = self._run_tool("cool.scene_change", scene_change, prev_bgr, bgr)
            timings["scene_change"] = dt
            change_score = float(change.change_score)
            traces.append(
                self._trace(
                    frame_id=frame_id,
                    tool="cool.scene_change",
                    observation={"change_score": change_score, "n_boxes": len(change.boxes)},
                    decision="CHANGED" if change_score > 0.02 else "STABLE",
                    next_action="cool.extract_tiles",
                    latency_ms=dt,
                    trace_id=tid,
                )
            )
            cut, dt_cut = self._run_tool(
                "cool.scene_cut",
                scene_cut,
                prev_bgr,
                bgr,
                change_score=change_score,
            )
            timings["scene_cut"] = dt_cut

        tiles: list = []
        if change is not None:
            tiles, dt = self._run_tool(
                "cool.extract_tiles", extract_tiles, bgr, change, 12
            )
            timings["extract_tiles"] = dt
            traces.append(
                self._trace(
                    frame_id=frame_id,
                    tool="cool.extract_tiles",
                    observation={"n_tiles": len(tiles)},
                    decision=f"n={len(tiles)}",
                    next_action="cool.detect_regions",
                    latency_ms=dt,
                    trace_id=tid,
                )
            )
        else:
            timings["extract_tiles"] = 0.0

        detections, dt = self._run_tool(
            "cool.detect_regions",
            self.detect_fn,
            bgr,
            frame_id=frame_id,
            stage="detect",
        )
        timings["detect_regions"] = dt
        flagged = [d for d in detections if d.label in FLAGGED_LABELS]
        primary = max(flagged, key=lambda d: d.score) if flagged else None
        band = score_band(primary.score, self.cfg, primary.label) if primary else "low"
        traces.append(
            self._trace(
                frame_id=frame_id,
                tool="cool.detect_regions",
                observation={
                    "n_detections": len(detections),
                    "flagged_score": None if primary is None else primary.score,
                    "flagged_label": None if primary is None else primary.label,
                    "band": band,
                },
                decision=band.upper(),
                next_action="policy.evaluate",
                latency_ms=dt,
                trace_id=tid,
            )
        )

        if cut:
            released = expire_tracks(self.tracks, last_seen_ms=t_ms)
            release_actions = evaluate(released, self.cfg, frame_id=frame_id, trace_id=tid)
            if not release_actions:
                release_actions = [
                    ProtectionAction(ActionType.RELEASE, None, "SCENE_CUT", self.cfg.version, tid)
                ]
            traces.append(
                self._trace(
                    frame_id=frame_id,
                    tool="cool.scene_cut",
                    observation={"change_score": change_score, "n_expired": len(released)},
                    decision="RELEASE",
                    next_action="cool.track_and_predict",
                    latency_ms=0.0,
                    trace_id=tid,
                )
            )
            line = companion_say("SCENE_CUT")
            traces.append(
                self._trace(
                    frame_id=frame_id,
                    tool="companion.say",
                    observation={"reason_code": "SCENE_CUT", "text": line},
                    decision=line,
                    next_action="cool.track_and_predict",
                    latency_ms=0.0,
                    trace_id=tid,
                )
            )
            self.tracks = []
            self.last_protective = []
            # Fresh tracks from this frame's detections after the cut.
            self.tracks, dt = self._run_tool(
                "cool.track_and_predict",
                track_and_predict,
                self.tracks,
                detections,
                bgr,
                None,
                t_ms=t_ms,
            )
            timings["track_and_predict"] = dt
            fresh = evaluate(self.tracks, self.cfg, frame_id=frame_id, trace_id=tid)
            actions = list(release_actions) + list(fresh)
            self._sync_mask_flags(self.tracks, actions)
            sequence = "SCENE_CUT→RELEASE"
            return FrameResult(
                actions=actions,
                traces=traces,
                detections=detections,
                tracks=list(self.tracks),
                change_score=change_score,
                cut=True,
                reinspect_count=0,
                timed_out=False,
                timings_ms=timings,
                tool_sequence=sequence,
                companion_line=line,
                first_decision="RELEASE",
                final_decision="RELEASE",
                band=band,
                n_tiles=len(tiles),
            )

        first_actions = evaluate(detections, self.cfg, frame_id=frame_id, trace_id=tid)
        first_decision = _primary_action_name(first_actions, default="NO_ACTION")
        next_action = self._next_from_band(band, primary, frame_id, first_decision)
        timings["policy.evaluate"] = 0.0
        traces.append(
            self._trace(
                frame_id=frame_id,
                tool="policy.evaluate",
                observation={
                    "pass": "first",
                    "band": band,
                    "budget_left": self.budget_left,
                    "actions": [a.action.value for a in first_actions],
                    "reasons": [a.reason_code for a in first_actions],
                },
                decision=first_decision,
                next_action=next_action,
                latency_ms=0.0,
                trace_id=tid,
            )
        )

        reinspect_count = 0
        if next_action == "REINSPECT_CROP" and primary is not None:
            looked, dt = self._run_tool(
                "cool.reinspect_crop",
                self.reinspect_fn,
                bgr,
                primary.xyxy,
                frame_id=frame_id,
            )
            timings["reinspect_crop"] = dt
            reinspect_count = 1
            self._spend_reinspect()
            self.last_reinspect_frame = frame_id
            traces.append(
                self._trace(
                    frame_id=frame_id,
                    tool="cool.reinspect_crop",
                    observation={
                        "in_score": primary.score,
                        "in_label": primary.label,
                        "in_box": list(primary.xyxy),
                        "out_score": looked.score,
                        "out_label": looked.label,
                        "out_box": list(looked.xyxy),
                        "budget_left": self.budget_left,
                    },
                    decision=f"{primary.score:.2f}->{looked.score:.2f} {looked.label}",
                    next_action="policy.evaluate",
                    latency_ms=dt,
                    trace_id=tid,
                )
            )
            if looked.score > 0.0:
                primary.score = looked.score
                primary.label = looked.label
                primary.xyxy = looked.xyxy
                primary.detector_id = f"{primary.detector_id}:reinspect"
                for i, det in enumerate(detections):
                    if det is primary:
                        detections[i] = primary
                        break

        self.tracks, dt = self._run_tool(
            "cool.track_and_predict",
            track_and_predict,
            self.tracks,
            detections,
            bgr,
            prev_bgr,
            t_ms=t_ms,
        )
        timings["track_and_predict"] = dt
        traces.append(
            self._trace(
                frame_id=frame_id,
                tool="cool.track_and_predict",
                observation={"n_tracks": len(self.tracks)},
                decision=f"n={len(self.tracks)}",
                next_action="policy.evaluate",
                latency_ms=dt,
                trace_id=tid,
            )
        )

        t_eval = time.perf_counter()
        actions = evaluate(self.tracks, self.cfg, frame_id=frame_id, trace_id=tid)
        timings["policy.evaluate"] = (time.perf_counter() - t_eval) * 1000.0
        if reinspect_count:
            traces.append(
                self._trace(
                    frame_id=frame_id,
                    tool="policy.evaluate",
                    observation={
                        "pass": "post_reinspect",
                        "actions": [a.action.value for a in actions],
                        "reasons": [a.reason_code for a in actions],
                    },
                    decision=_primary_action_name(actions, default="NO_ACTION"),
                    next_action="cool.render_mask_spec",
                    latency_ms=timings["policy.evaluate"],
                    trace_id=tid,
                )
            )

        review_actions = [a for a in actions if a.action is ActionType.REVIEW]
        if review_actions:
            reviewed = [
                request_review(
                    reason_code=a.reason_code,
                    policy_version=self.cfg.version,
                    region=a.region,
                    trace_id=tid,
                )
                for a in review_actions
            ]
            traces.append(
                self._trace(
                    frame_id=frame_id,
                    tool="guardian.request_review",
                    observation={"reasons": [a.reason_code for a in reviewed]},
                    decision="REVIEW",
                    next_action="companion.say",
                    latency_ms=0.0,
                    trace_id=tid,
                )
            )

        self._sync_mask_flags(self.tracks, actions)
        protective = [a for a in actions if a.action in PROTECTIVE_ACTIONS]
        if protective:
            self.last_protective = protective

        final_decision = _primary_action_name(actions, default="NO_ACTION")
        reason = _primary_reason(actions)
        line = companion_say(reason)
        traces.append(
            self._trace(
                frame_id=frame_id,
                tool="companion.say",
                observation={"reason_code": reason, "text": line},
                decision=line,
                next_action="cool.render_mask_spec",
                latency_ms=0.0,
                trace_id=tid,
            )
        )

        sequence = _tool_sequence(reinspect_count > 0, actions)
        return FrameResult(
            actions=actions,
            traces=traces,
            detections=detections,
            tracks=list(self.tracks),
            change_score=change_score,
            cut=False,
            reinspect_count=reinspect_count,
            timed_out=False,
            timings_ms=timings,
            tool_sequence=sequence,
            companion_line=line,
            first_decision=first_decision,
            final_decision=final_decision,
            band=band,
            n_tiles=len(tiles),
        )

    def _next_from_band(
        self,
        band: str,
        primary: Detection | None,
        frame_id: int,
        first_decision: str = "NO_ACTION",
    ) -> str:
        if primary is None or band == "low":
            return "NO_ACTION"
        enabled = class_enabled(self.cfg, primary.label)
        if band == "high" and enabled:
            return "APPLY"
        if band == "mid" and self.can_reinspect(frame_id):
            return "REINSPECT_CROP"
        if band == "high":
            return "APPLY"
        if first_decision in {
            ActionType.OPAQUE_MASK.value,
            ActionType.BLUR.value,
            ActionType.PIXELATE.value,
            ActionType.HOLD.value,
            ActionType.FULL_WINDOW_MASK.value,
        }:
            return "APPLY"
        if first_decision == ActionType.REVIEW.value:
            return "REVIEW"
        return "NO_ACTION"

    def _timeout_fallback(
        self,
        frame_id: int,
        traces: list[AgentTrace],
        timings: dict[str, float],
        tool: str,
        tid: str,
    ) -> FrameResult:
        # Strict must never hang unmasked. Others HOLD last cover or stand down.
        if self.cfg.name == "strict":
            if self.last_protective:
                actions = [
                    replace(a, action=ActionType.HOLD, reason_code="TOOL_TIMEOUT", trace_id=tid)
                    for a in self.last_protective
                ]
            else:
                actions = [
                    ProtectionAction(
                        ActionType.FULL_WINDOW_MASK,
                        None,
                        "FULL_WINDOW_FALLBACK",
                        self.cfg.version,
                        tid,
                    )
                ]
            decision = actions[0].action.value
        elif self.last_protective:
            actions = [
                replace(a, action=ActionType.HOLD, reason_code="TOOL_TIMEOUT", trace_id=tid)
                for a in self.last_protective
            ]
            decision = "HOLD"
        else:
            actions = [
                ProtectionAction(
                    ActionType.NO_ACTION,
                    None,
                    "NOT_PROTECTING",
                    self.cfg.version,
                    tid,
                )
            ]
            decision = "NOT_PROTECTING"

        traces.append(
            self._trace(
                frame_id=frame_id,
                tool="agent.fallback",
                observation={"failed_tool": tool, "policy": self.cfg.name},
                decision=decision,
                next_action="HOLD" if decision != "NOT_PROTECTING" else "NOT_PROTECTING",
                latency_ms=0.0,
                trace_id=tid,
            )
        )
        reason = actions[0].reason_code
        line = companion_say(reason)
        traces.append(
            self._trace(
                frame_id=frame_id,
                tool="companion.say",
                observation={"reason_code": reason, "text": line},
                decision=line,
                next_action="cool.render_mask_spec",
                latency_ms=0.0,
                trace_id=tid,
            )
        )
        return FrameResult(
            actions=actions,
            traces=traces,
            detections=[],
            tracks=list(self.tracks),
            change_score=0.0,
            cut=False,
            reinspect_count=0,
            timed_out=True,
            timings_ms=timings,
            tool_sequence="TIMEOUT→" + decision,
            companion_line=line,
            first_decision=decision,
            final_decision=decision,
            band="low",
        )

    @staticmethod
    def _sync_mask_flags(tracks: list[TrackedRegion], actions: list[ProtectionAction]) -> None:
        on_ids = {
            a.region.track_id
            for a in actions
            if a.region is not None and a.action in PROTECTIVE_ACTIONS
        }
        for i, track in enumerate(tracks):
            tracks[i] = replace(track, mask_on=track.track_id in on_ids)


def _primary_action_name(actions: list[ProtectionAction], default: str) -> str:
    protective = [a for a in actions if a.action in PROTECTIVE_ACTIONS]
    if protective:
        return protective[0].action.value
    reviews = [a for a in actions if a.action is ActionType.REVIEW]
    if reviews:
        return ActionType.REVIEW.value
    releases = [a for a in actions if a.action is ActionType.RELEASE]
    if releases:
        return ActionType.RELEASE.value
    if actions:
        return actions[0].action.value
    return default


def _primary_reason(actions: list[ProtectionAction]) -> str:
    if not actions:
        return "NO_ACTION"
    protective = [a for a in actions if a.action in PROTECTIVE_ACTIONS]
    if protective:
        return protective[0].reason_code
    return actions[0].reason_code


def _tool_sequence(did_reinspect: bool, actions: list[ProtectionAction]) -> str:
    protective = any(a.action in PROTECTIVE_ACTIONS for a in actions)
    if did_reinspect and protective:
        return "DETECT→REINSPECT→MASK"
    if did_reinspect:
        return "DETECT→REINSPECT→NO_ACTION"
    if protective:
        return "DETECT→MASK"
    return "DETECT→NO_ACTION"
