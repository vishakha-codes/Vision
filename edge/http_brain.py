"""HTTP brain client for fixture ``/v1/perceive`` and live ``/v1/live_perceive``.

Laptop reflex keeps capture / flow-hold / overlay local. Synthetic replay
sends ``fixture_id`` + ``frame_id``. Live nsfw sends JPEG tiles (or one
downscaled frame) — never an uncompressed desktop.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

DEFAULT_GRAVITON_URL = os.environ.get("SG_GRAVITON_URL", "http://127.0.0.1:8080")  # legacy :8080 API

from agent.companion import say as companion_say
from agent.orchestrator import FrameResult
from vision_core.contracts import (
    ActionType,
    AgentTrace,
    ProtectionAction,
    TrackMode,
    TrackedRegion,
)


class PerceiveError(Exception):
    """Remote ``/v1/perceive`` failed. Reflex must stand down, not crash."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


def normalize_backend(spec: str | None) -> tuple[str, str | None]:
    """Return ``(local|cloud, url_or_none)``. ``local`` is the default."""
    raw = (spec or "local").strip()
    if not raw or raw.lower() in {"local", "auto"}:
        return "local", None
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"backend must be 'local' or an http(s) URL, got {spec!r}")
    return "cloud", raw.rstrip("/")


def probe_health(url: str, *, timeout_s: float = 1.5) -> bool:
    """True when ``GET /health`` returns ok. Used for live-nsfw auto fallback."""
    kind, base = normalize_backend(url)
    if kind != "cloud" or base is None:
        return False
    req = urllib.request.Request(f"{base}/health", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return False
    return bool(isinstance(payload, dict) and payload.get("ok"))


def resolve_backend(
    spec: str | None,
    *,
    source: str = "replay",
    detect: str = "synthetic",
    graviton_url: str = DEFAULT_GRAVITON_URL,
) -> str:
    """Pick local vs Graviton. Live nsfw ``auto`` probes ``/health``."""
    raw = (spec or "auto").strip()
    if raw.lower() == "local":
        return "local"
    if raw.lower() not in {"", "auto"}:
        return raw.rstrip("/")
    if (source or "").strip().lower() == "live" and (detect or "").strip().lower() == "nsfw":
        if probe_health(graviton_url):
            return graviton_url.rstrip("/")
        return "local"
    return "local"


def _parse_region(raw: dict[str, Any] | None) -> TrackedRegion | None:
    if not raw:
        return None
    xyxy = raw.get("xyxy") or [0.0, 0.0, 0.0, 0.0]
    if len(xyxy) < 4:
        raise PerceiveError("region.xyxy must have 4 values")
    mode_raw = raw.get("mode") or TrackMode.TRACKED.value
    try:
        mode = TrackMode(mode_raw)
    except ValueError:
        mode = TrackMode.TRACKED
    return TrackedRegion(
        track_id=int(raw.get("track_id") or 0),
        xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
        label=str(raw.get("label") or "unknown"),
        stable_score=float(raw.get("stable_score") or 0.0),
        uncertainty=float(raw.get("uncertainty") or 0.0),
        last_seen_ms=float(raw.get("last_seen_ms") or 0.0),
        mode=mode,
        mask_on=bool(raw.get("mask_on", False)),
    )


def _parse_action(raw: dict[str, Any]) -> ProtectionAction:
    try:
        action = ActionType(raw["action"])
    except (KeyError, ValueError) as exc:
        raise PerceiveError(f"invalid action: {raw.get('action')!r}") from exc
    return ProtectionAction(
        action=action,
        region=_parse_region(raw.get("region") if isinstance(raw.get("region"), dict) else None),
        reason_code=str(raw.get("reason_code") or "NO_ACTION"),
        policy_version=str(raw.get("policy_version") or ""),
        trace_id=str(raw.get("trace_id") or ""),
    )


def _parse_trace(raw: dict[str, Any]) -> AgentTrace:
    return AgentTrace(
        trace_id=str(raw.get("trace_id") or ""),
        frame_id=int(raw.get("frame_id") or 0),
        tool=str(raw.get("tool") or "agent.decide"),
        observation=dict(raw.get("observation") or {}),
        decision=str(raw.get("decision") or ""),
        next_action=str(raw.get("next_action") or ""),
        latency_ms=float(raw.get("latency_ms") or 0.0),
        image_persisted=False,
    )


def parse_perceive_response(payload: dict[str, Any]) -> FrameResult:
    """Parse a ``/v1/perceive`` JSON body. No network."""
    if not isinstance(payload, dict):
        raise PerceiveError("perceive response must be a JSON object")
    actions = [_parse_action(item) for item in (payload.get("actions") or []) if isinstance(item, dict)]
    traces = [_parse_trace(item) for item in (payload.get("traces") or []) if isinstance(item, dict)]
    if not traces and isinstance(payload.get("trace"), dict):
        traces = [_parse_trace(payload["trace"])]
    tracks: list[TrackedRegion] = []
    seen: set[int] = set()
    for action in actions:
        region = action.region
        if region is not None and region.track_id not in seen:
            tracks.append(region)
            seen.add(region.track_id)
    reinspect_count = int(payload.get("reinspect_count") or 0)
    if reinspect_count <= 0:
        reinspect_count = sum(1 for trace in traces if trace.tool == "cool.reinspect_crop")
    final = str(payload.get("final_decision") or "NO_ACTION")
    line = payload.get("companion_line")
    if not line:
        reason = actions[0].reason_code if actions else final
        line = companion_say(reason)
    return FrameResult(
        actions=actions,
        traces=traces,
        detections=[],
        tracks=tracks,
        change_score=float(payload.get("change_score") or 0.0),
        cut=bool(payload.get("cut") or False),
        reinspect_count=reinspect_count,
        timed_out=bool(payload.get("timed_out") or False),
        timings_ms=dict(payload.get("timings_ms") or {}),
        tool_sequence=str(payload.get("tool_sequence") or ""),
        companion_line=str(line),
        first_decision=str(payload.get("first_decision") or final),
        final_decision=final,
        band=str(payload.get("band") or "low"),
        n_tiles=int(payload.get("n_tiles") or 0),
    )


class PerceiveClient:
    """Fixture ``/v1/perceive`` or live ``/v1/live_perceive`` (JPEG tiles)."""

    def __init__(self, base_url: str, *, timeout_s: float = 8.0) -> None:
        kind, url = normalize_backend(base_url)
        if kind != "cloud" or url is None:
            raise ValueError(f"PerceiveClient needs an http(s) URL, got {base_url!r}")
        self.base_url = url
        self.timeout_s = float(timeout_s)
        self.last_payload_bytes = 0

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"content-type": "application/json", "accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:240]
            raise PerceiveError(f"HTTP {exc.code}: {detail}", status_code=exc.code) from exc
        except urllib.error.URLError as exc:
            raise PerceiveError(f"unreachable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise PerceiveError("timeout") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PerceiveError("perceive response is not JSON") from exc
        if not isinstance(payload, dict):
            raise PerceiveError("perceive response must be a JSON object")
        return payload

    def perceive(
        self,
        *,
        frame_id: int,
        policy_id: str,
        fixture_id: str | None = None,
        path: str | None = None,
    ) -> FrameResult:
        body: dict[str, Any] = {
            "frame_id": int(frame_id),
            "policy_id": policy_id,
        }
        if fixture_id:
            body["fixture_id"] = fixture_id
        elif path:
            body["path"] = path
        else:
            raise PerceiveError("need fixture_id or an allowlisted path")
        return parse_perceive_response(self._post("/v1/perceive", body))

    def live(
        self,
        jpeg_bytes: bytes,
        *,
        frame_id: int,
        policy_id: str,
        orig_width: int,
        orig_height: int,
        tile_xyxy: list[float] | tuple[float, float, float, float],
        candidates: list[dict[str, Any]] | None = None,
    ) -> FrameResult:
        """POST a JPEG tile to ``/v1/live_perceive``. Never persist."""
        if not jpeg_bytes:
            raise PerceiveError("live tile is empty")
        self.last_payload_bytes = len(jpeg_bytes)
        body: dict[str, Any] = {
            "frame_id": int(frame_id),
            "policy_id": policy_id,
            "orig_width": int(orig_width),
            "orig_height": int(orig_height),
            "tile_xyxy": [float(v) for v in tile_xyxy],
            "jpeg_b64": base64.b64encode(jpeg_bytes).decode("ascii"),
            "payload_bytes": len(jpeg_bytes),
        }
        if candidates:
            body["candidates"] = candidates
        payload = self._post("/v1/live_perceive", body)
        return parse_perceive_response(payload)
