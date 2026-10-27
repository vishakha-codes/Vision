"""Laptop reflex: local eyes, orchestrator ticks, flow-hold between them.

Every frame runs ``scene_change`` + ``extract_tiles`` locally. Detections
go through ``agent.orchestrator.Orchestrator`` (no Bedrock required).
Between brain ticks, ``track_and_predict`` warps the last mask. A slow
brain HOLDs that mask. A missing frame is ``NOT_PROTECTING``.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace

import numpy as np

from agent.orchestrator import FrameResult, Orchestrator
from edge.companion import mood_from, say
from edge.http_brain import PerceiveClient, PerceiveError, normalize_backend
from vision_core.cluster_nsfw import live_scene_changed
from vision_core.encode_tile import pack_live_frame, pick_live_tile
from vision_core.contracts import (
    PROTECTIVE_ACTIONS,
    ActionType,
    AgentTrace,
    Detection,
    Frame,
    ProtectionAction,
    TrackedRegion,
    make_trace,
    new_trace_id,
)
from vision_core.geometry import box_iou
from vision_core.policy import evaluate
from vision_core.detect_nsfw import make_nsfw_detect, tune_policy_for_live_nsfw
from vision_core.detect_regions import detect_regions
from vision_core.extract_tiles import extract_tiles
from vision_core.policy import load_policy
from vision_core.reinspect_crop import reinspect_crop
from vision_core.reinspect_nsfw import make_nsfw_reinspect
from vision_core.scene_change import scene_change
from vision_core.track_and_predict import track_and_predict

NSFW_TOOL_TIMEOUT_MS = 2000.0

STATUS_NOT_PROTECTING = "NOT_PROTECTING"
STATUS_LOCAL_HOLD = "LOCAL_HOLD"
STATUS_CLOUD_LOOK = "CLOUD_LOOK"

_POLICY_FILES = {
    "strict": "policies/strict.json",
    "balanced": "policies/balanced.json",
    "relaxed": "policies/relaxed.json",
}


def resolve_policy(policy: str) -> str:
    key = (policy or "balanced").strip()
    lowered = key.lower()
    if lowered in _POLICY_FILES:
        return _POLICY_FILES[lowered]
    if lowered.endswith(".json"):
        return key
    return f"policies/{lowered}.json"

_LOOK_LABEL = {
    STATUS_NOT_PROTECTING: "NOT_PROTECTING",
    STATUS_LOCAL_HOLD: "Local hold",
    STATUS_CLOUD_LOOK: "Cloud look",
}


def cloud_health_ok(url: str, *, timeout_s: float = 0.8) -> bool:
    """Cheap GET ``/health``. Does not own the perceive client."""
    if not url:
        return False
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= int(getattr(resp, "status", 200)) < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


@dataclass
class ReflexState:
    frame: Frame | None
    actions: list[ProtectionAction]
    traces: list[AgentTrace]
    status: str
    policy_name: str
    mood: str
    companion_line: str
    look: str
    change_score: float
    n_tiles: int
    band: str
    held: bool
    paused: bool
    protecting: bool
    frame_id: int
    tracks: list[TrackedRegion] = field(default_factory=list)
    backend: str = "local"
    backend_error: str | None = None
    detect_mode: str = "synthetic"


class Reflex:
    """Per-frame laptop loop. Policy can be swapped on the same clip."""

    def __init__(
        self,
        policy: str = "balanced",
        *,
        backend: str = "local",
        fixture_id: str = "synthetic_moving_regions",
        fixture_path: str | None = None,
        http_client: PerceiveClient | None = None,
        brain_every: int = 3,
        brain_every_ms: float = 250.0,
        brain_budget_ms: float = 80.0,
        change_gate: float = 0.08,
        detect_mode: str = "synthetic",
        reinspect_mode: str = "local",
        detect_fn=None,
        reinspect_fn=None,
        on_mask_event=None,
    ) -> None:
        self.brain_every = max(1, int(brain_every))
        self.brain_every_ms = float(brain_every_ms)
        self.brain_budget_ms = float(brain_budget_ms)
        self.change_gate = float(change_gate)
        self.detect_mode = (detect_mode or "synthetic").strip().lower()
        self.reinspect_mode = (reinspect_mode or "local").strip().lower()
        self._user_detect_fn = detect_fn
        self._user_reinspect_fn = reinspect_fn
        self.on_mask_event = on_mask_event
        self.backend_kind, self.backend_url = normalize_backend(backend)
        self.last_backend_error = None
        self.last_payload_bytes = 0
        self.fixture_id = fixture_id
        self.fixture_path = fixture_path
        self.http = http_client
        self._nsfw_reset_hold = None
        if self.http is None and self.backend_url is not None:
            self.http = PerceiveClient(self.backend_url)
            if not cloud_health_ok(self.backend_url):
                self.last_backend_error = (
                    f"cloud /health failed; local fallback ({self.backend_url})"
                )
                self.http = None
        self.prev_bgr: np.ndarray | None = None
        self.last_actions: list[ProtectionAction] = []
        self.last_tracks: list[TrackedRegion] = []
        self.last_result: FrameResult | None = None
        self.last_line = say("NO_ACTION")
        self.last_band = "low"
        self.last_reinspect = 0
        self.last_traces: list[AgentTrace] = []
        self._steps = 0
        self._last_brain_step = -10**9
        self._last_brain_t_ms = -1e12
        self._need_brain = True
        self._brain_slow = False
        self._last_local_dets: list[Detection] = []
        self._backend_used = self.backend_kind
        self.orch: Orchestrator | None = None
        self._set_policy(policy)

    @property
    def policy_name(self) -> str:
        return self.cfg.name

    def _set_policy(self, policy: str) -> None:
        self.cfg = load_policy(resolve_policy(policy))
        if self.detect_mode == "nsfw":
            self.cfg = tune_policy_for_live_nsfw(self.cfg)
        # Keep boxes so the clip does not rewind; drop mask_on so the new
        # preset is not stuck on the previous policy's hysteresis.
        old_tracks = [replace(t, mask_on=False) for t in self.last_tracks]
        detect_fn = self._user_detect_fn
        reinspect_fn = self._user_reinspect_fn
        timeout_ms = None
        if detect_fn is None and self.detect_mode == "nsfw":
            detect_fn = make_nsfw_detect(self.cfg.name)
            timeout_ms = NSFW_TOOL_TIMEOUT_MS
        elif detect_fn is None:
            detect_fn = detect_regions
        if reinspect_fn is None and self.detect_mode == "nsfw":
            reinspect_fn = make_nsfw_reinspect(mode=self.reinspect_mode, detect_fn=detect_fn)
            timeout_ms = NSFW_TOOL_TIMEOUT_MS
        elif reinspect_fn is None:
            reinspect_fn = reinspect_crop
        self._nsfw_reset_hold = getattr(detect_fn, "reset_hold", None)
        if callable(self._nsfw_reset_hold):
            self._nsfw_reset_hold()
        # Local orch is always built so a dead Graviton /health can fall back.
        self.orch = Orchestrator(
            self.cfg,
            detect_fn=detect_fn,
            reinspect_fn=reinspect_fn,
            timeout_ms=timeout_ms,
        )
        if old_tracks:
            self.orch.tracks = old_tracks
        self.last_actions = []
        self.last_tracks = old_tracks
        self._need_brain = True

    def set_policy(self, policy: str) -> None:
        """Switch Strict / Balanced / Relaxed without rewinding the clip."""
        self._set_policy(policy)

    def set_detect(self, mode: str) -> str | None:
        """Switch synthetic HSV vs on-device NSFW. Returns an error string or None."""
        mode = (mode or "").strip().lower()
        if mode not in {"synthetic", "nsfw"}:
            return "unknown detect mode"
        if mode == self.detect_mode:
            return None
        if mode == "nsfw":
            from vision_core.detect_nsfw import load_nudenet, nudenet_available

            if not nudenet_available():
                return "nudenet weights missing"
            load_nudenet()
        self.detect_mode = mode
        self._set_policy(self.policy_name)
        return None

    def _stand_down(
        self,
        frame: Frame | None,
        *,
        paused: bool,
        reason: str,
    ) -> ReflexState:
        traces = [
            make_trace(
                frame_id=-1 if frame is None else frame.frame_id,
                tool="edge.reflex",
                observation={"reason": reason, "paused": paused, "available": frame is not None},
                decision=STATUS_NOT_PROTECTING,
                next_action=STATUS_NOT_PROTECTING,
                latency_ms=0.0,
            )
        ]
        line = say("NOT_PROTECTING")
        return self._emit(ReflexState(
            frame=frame,
            actions=[],
            traces=traces,
            status=STATUS_NOT_PROTECTING,
            policy_name=self.cfg.name,
            mood="lost",
            companion_line=line,
            look=_LOOK_LABEL[STATUS_NOT_PROTECTING],
            change_score=0.0,
            n_tiles=0,
            band="low",
            held=True,
            paused=paused,
            protecting=False,
            frame_id=-1 if frame is None else frame.frame_id,
            tracks=list(self.last_tracks),
            backend=self._backend_used,
            backend_error=self.last_backend_error,
            detect_mode=self.detect_mode,
        ))

    def _emit(self, state: ReflexState) -> ReflexState:
        if self.on_mask_event is not None:
            self.on_mask_event(state)
        return state

    def _local_eyes(self, bgr: np.ndarray) -> tuple[float, int]:
        if self.prev_bgr is None:
            return 1.0, 0
        change = scene_change(self.prev_bgr, bgr)
        tiles = extract_tiles(bgr, change, 12)
        return float(change.change_score), len(tiles)

    def _should_tick(self, change_score: float, t_ms: float) -> bool:
        if self._need_brain:
            return True
        # Periodic keyframe always wins — change-gate must never go hundreds
        # of frames without a detect, even if the last brain tick was slow.
        since_frames = self._steps - self._last_brain_step
        since_ms = t_ms - self._last_brain_t_ms
        if since_frames >= self.brain_every or since_ms >= self.brain_every_ms:
            return True
        if self._brain_slow:
            return False
        return change_score >= self.change_gate

    def _warp_actions(
        self,
        actions: list[ProtectionAction],
        tracks: list[TrackedRegion],
    ) -> list[ProtectionAction]:
        by_id = {t.track_id: t for t in tracks}
        out: list[ProtectionAction] = []
        for action in actions:
            region = action.region
            if region is not None and region.track_id in by_id:
                moved = replace(by_id[region.track_id], mask_on=region.mask_on)
                out.append(replace(action, region=moved))
            else:
                out.append(action)
        return out

    def _flow_hold(self, frame: Frame) -> tuple[list[ProtectionAction], list[AgentTrace]]:
        tracks = list(self.last_tracks)
        if tracks:
            tracks = track_and_predict(
                tracks,
                [],
                frame.image_bgr,
                self.prev_bgr,
                t_ms=frame.t_ms,
                hold=True,
            )
        actions = self._warp_actions(self.last_actions, tracks)
        self.last_tracks = tracks
        self.last_actions = actions
        if self.orch is not None:
            self.orch.tracks = list(tracks)
        traces = [
            make_trace(
                frame_id=frame.frame_id,
                tool="edge.flow_hold",
                observation={
                    "n_tracks": len(tracks),
                    "n_actions": len(actions),
                    "brain_slow": self._brain_slow,
                },
                decision="HOLD",
                next_action="cool.track_and_predict",
                latency_ms=0.0,
                trace_id=new_trace_id(),
            )
        ]
        return actions, traces

    def _record_brain(self, result: FrameResult) -> None:
        self.last_result = result
        self.last_actions = list(result.actions)
        self.last_tracks = list(result.tracks)
        self.last_line = result.companion_line or say(result.final_decision)
        self.last_band = result.band
        self.last_reinspect = result.reinspect_count
        self.last_traces = list(result.traces)
        if result.detections:
            self._last_local_dets = list(result.detections)

    def _drop_stale_hold(self) -> None:
        reset = self._nsfw_reset_hold
        if callable(reset):
            reset()
        self.last_actions = []
        self.last_tracks = []
        self._last_local_dets = []
        if self.orch is not None:
            self.orch.tracks = []
            self.orch.last_protective = []
        self._need_brain = True

    def _live_brain_tick(self, frame: Frame) -> FrameResult:
        """JPEG tiles to ``/v1/live_perceive``. Never a raw desktop."""
        packed = pack_live_frame(frame.image_bgr)
        tiles = list(packed.get("tiles") or [])
        if not tiles:
            raise PerceiveError("live tile encode failed")
        tile = tiles[0]
        candidates: list[dict] = []
        self._last_local_dets = []
        detect_fn = self._user_detect_fn
        if detect_fn is None and self.orch is not None:
            detect_fn = self.orch.detect_fn
        if detect_fn is not None:
            try:
                dets = list(detect_fn(frame.image_bgr, frame_id=frame.frame_id, stage="detect") or [])
                self._last_local_dets = dets
                candidates = [
                    {
                        "xyxy": [float(v) for v in det.xyxy],
                        "label": det.label,
                        "score": float(det.score),
                    }
                    for det in dets
                ]
            except Exception:
                candidates = []
                self._last_local_dets = []
        chosen = pick_live_tile(tiles, candidates) or tile
        jpeg = chosen["jpeg"]
        self.last_payload_bytes = int(len(jpeg))
        print(
            f"live_perceive send jpeg={self.last_payload_bytes}B "
            f"tiles={len(tiles)} orig={packed['orig_width']}x{packed['orig_height']} "
            f"candidates={len(candidates)} persisted=false",
            flush=True,
        )
        return self.http.live(
            jpeg,
            frame_id=frame.frame_id,
            policy_id=self.cfg.name,
            orig_width=int(packed["orig_width"]),
            orig_height=int(packed["orig_height"]),
            tile_xyxy=chosen["xyxy"],
            candidates=candidates,
        )

    def _held_boxes(self) -> list[tuple[float, float, float, float]]:
        boxes: list[tuple[float, float, float, float]] = []
        for action in self.last_actions:
            if action.region is None:
                continue
            boxes.append(action.region.xyxy)
        for det in self._last_local_dets:
            boxes.append(det.xyxy)
        return boxes

    def _actions_from_clusters(self, dets: list[Detection]) -> list[ProtectionAction]:
        if not dets:
            return []
        return [a for a in evaluate(dets, self.cfg) if a.action in PROTECTIVE_ACTIONS]

    def _union_protective(
        self,
        primary: list[ProtectionAction],
        extra: list[ProtectionAction],
    ) -> list[ProtectionAction]:
        out = list(primary)
        for action in extra:
            if action.region is None:
                continue
            if any(
                other.region is not None and box_iou(action.region.xyxy, other.region.xyxy) >= 0.15
                for other in out
            ):
                continue
            out.append(action)
        return out

    def _stick_live_nsfw(
        self,
        result: FrameResult,
        prior: list[ProtectionAction] | None = None,
    ) -> list[ProtectionAction]:
        """Keep frost from local clusters / last boxes when cloud says NO_ACTION.

        This-frame NudeNet clusters are the overlay count. Do not union in
        leftover orchestrator tracks (that pins frost at 5 / 8 / 12).
        """
        cloud = [a for a in result.actions if a.action in PROTECTIVE_ACTIONS]
        local = self._actions_from_clusters(self._last_local_dets)
        merged = local if local else self._union_protective(cloud, local)
        if merged:
            self.last_actions = merged
            if result.final_decision not in {a.action.value for a in merged}:
                result.final_decision = merged[0].action.value
            return merged
        fallback = prior if prior is not None else self.last_actions
        held = [
            replace(action, action=ActionType.HOLD, reason_code="LOCAL_CLUSTER_HOLD")
            for action in fallback
            if action.action in PROTECTIVE_ACTIONS
        ]
        if held:
            self.last_actions = held
            result.final_decision = "HOLD"
            return held
        return list(result.actions)

    def _keep_last_nsfw(self, frame: Frame, reason: str) -> ReflexState:
        """Cloud miss / timeout: hold last boxes instead of wiping frost."""
        actions = [
            replace(action, action=ActionType.HOLD, reason_code="LOCAL_CLUSTER_HOLD")
            for action in self.last_actions
            if action.action in PROTECTIVE_ACTIONS
        ]
        self.last_actions = actions
        if self.last_result is not None:
            self.last_result.final_decision = "HOLD"
        self._backend_used = "local"
        line = say("TRACK_HELD")
        self.last_line = line
        self.prev_bgr = frame.image_bgr
        self._steps += 1
        return self._emit(ReflexState(
            frame=frame,
            actions=actions,
            traces=list(self.last_traces),
            status=STATUS_LOCAL_HOLD,
            policy_name=self.cfg.name,
            mood="covering",
            companion_line=line,
            look=_LOOK_LABEL[STATUS_LOCAL_HOLD],
            change_score=0.0,
            n_tiles=0,
            band=self.last_band,
            held=True,
            paused=False,
            protecting=True,
            frame_id=frame.frame_id,
            tracks=list(self.last_tracks),
            backend=self._backend_used,
            backend_error=reason,
            detect_mode=self.detect_mode,
        ))

    def _brain_tick(self, frame: Frame) -> FrameResult:
        if self.http is not None and self.detect_mode == "nsfw":
            try:
                result = self._live_brain_tick(frame)
                self._backend_used = "cloud"
                return result
            except PerceiveError as exc:
                if self.orch is None:
                    raise
                self.last_backend_error = f"cloud live_perceive failed; local fallback ({exc})"
                self._backend_used = "local"
                return self.orch.process_frame(frame.frame_id, frame.image_bgr, self.prev_bgr, frame.t_ms)
        if self.http is not None:
            self._backend_used = "cloud"
            return self.http.perceive(
                frame_id=frame.frame_id,
                policy_id=self.cfg.name,
                fixture_id=self.fixture_id,
                path=self.fixture_path,
            )
        if self.orch is None:
            raise PerceiveError("local orchestrator is not available")
        self._backend_used = "local"
        return self.orch.process_frame(frame.frame_id, frame.image_bgr, self.prev_bgr, frame.t_ms)

    def step(self, frame: Frame | None, *, paused: bool = False) -> ReflexState:
        if paused:
            return self._stand_down(frame, paused=True, reason="paused")
        if frame is None:
            return self._stand_down(None, paused=False, reason="unavailable")

        bgr = frame.image_bgr
        if (
            self.detect_mode == "nsfw"
            and self.prev_bgr is not None
            and live_scene_changed(self.prev_bgr, bgr, ignore_boxes=self._held_boxes())
        ):
            self._drop_stale_hold()
        change_score, n_tiles = self._local_eyes(bgr)
        tick = self._should_tick(change_score, frame.t_ms)
        held = not tick
        traces: list[AgentTrace]
        actions: list[ProtectionAction]
        status = STATUS_CLOUD_LOOK

        if tick:
            t0 = time.perf_counter()
            try:
                result = self._brain_tick(frame)
            except PerceiveError as exc:
                self.last_backend_error = str(exc)
                self._need_brain = True
                self._brain_slow = True
                if self.detect_mode == "nsfw" and any(
                    a.action in PROTECTIVE_ACTIONS for a in self.last_actions
                ):
                    return self._keep_last_nsfw(frame, f"http_error:{exc}")
                self.prev_bgr = bgr
                self._steps += 1
                return self._stand_down(frame, paused=False, reason=f"http_error:{exc}")
            if self._backend_used == "cloud":
                self.last_backend_error = None
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._brain_slow = dt_ms > self.brain_budget_ms
            self._need_brain = False
            self._last_brain_step = self._steps
            self._last_brain_t_ms = frame.t_ms
            prior_actions = list(self.last_actions)
            if (
                result.timed_out
                and result.final_decision == STATUS_NOT_PROTECTING
                and self.detect_mode == "nsfw"
                and any(a.action in PROTECTIVE_ACTIONS for a in prior_actions)
            ):
                self.last_actions = prior_actions
                return self._keep_last_nsfw(frame, "timeout")
            self._record_brain(result)
            if self.detect_mode == "nsfw":
                actions = self._stick_live_nsfw(result, prior_actions)
                if any(a.action in PROTECTIVE_ACTIONS for a in actions):
                    self.last_line = say(result.final_decision)
            else:
                actions = list(result.actions)
            traces = list(result.traces)
            if result.timed_out and result.final_decision == STATUS_NOT_PROTECTING:
                self.prev_bgr = bgr
                self._steps += 1
                return self._stand_down(frame, paused=False, reason="timeout")
            status = STATUS_CLOUD_LOOK
            band = result.band
            reinspect_count = result.reinspect_count
            line = self.last_line
        else:
            actions, traces = self._flow_hold(frame)
            status = STATUS_LOCAL_HOLD
            band = self.last_band
            reinspect_count = self.last_reinspect
            line = say("TRACK_HELD") if any(a.action in PROTECTIVE_ACTIONS for a in actions) else self.last_line
            traces = list(self.last_traces) + traces

        mood = mood_from(
            status=status,
            actions=actions,
            traces=traces,
            band=band,
            reinspect_count=reinspect_count,
        )
        if tick:
            line = self.last_line
        elif mood == "covering":
            line = say("TRACK_HELD")
        self.last_line = line

        self.prev_bgr = bgr
        self._steps += 1
        return self._emit(ReflexState(
            frame=frame,
            actions=actions,
            traces=traces,
            status=status,
            policy_name=self.cfg.name,
            mood=mood,
            companion_line=line,
            look=_LOOK_LABEL[status],
            change_score=change_score,
            n_tiles=n_tiles,
            band=band,
            held=held,
            paused=False,
            protecting=any(a.action in PROTECTIVE_ACTIONS for a in actions),
            frame_id=frame.frame_id,
            tracks=list(self.last_tracks),
            backend=self._backend_used,
            backend_error=self.last_backend_error,
            detect_mode=self.detect_mode,
        ))
