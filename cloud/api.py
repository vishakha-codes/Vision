"""Screen Guardian cloud API (Graviton box, COOL interpreter).

- ``POST /v1/second_look`` (L2): one JPEG picture crop → multi-scale NudeNet
  320n on the server OpenCV (COOL 4.14 on the box). Bearer token.
- ``POST /v1/review``: deep look at a crop held in RAM after an UNSURE L2
  result; called by the AgentCore agent's ``cool_second_look`` tool. Bearer.
- ``/v1/policies``, ``/v1/fixtures``, ``/v1/evaluations``, ``/v1/perceive``:
  fixture evaluation (allowlisted ids, no uploads).
- ``/v1/live_perceive``: legacy single-thread reflex (``--engine legacy``)
  only. Remove with that engine; the live engine uses ``/v1/second_look``.

No image is ever written to disk. ``image_persisted`` is always false.
"""

from __future__ import annotations

import base64
import binascii
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.orchestrator import Orchestrator
from cloud.catalog import (
    CatalogError,
    VISION_ROOT,
    list_fixtures,
    list_policies,
    reject_upload_or_url_body,
    resolve_allowlisted_relpath,
    resolve_fixture,
    resolve_policy,
)
from cloud import telemetry
from cloud.auth import expected_token, require_token
from cloud.cool_detector import opencv_build
from cloud.live_perceive import nudenet_on_server, run_live_perceive
from cloud.second_look import SecondLookError, SecondLookService
from vision_core.contracts import ProtectionAction, TrackedRegion, trace_to_dict
from vision_core.eval import evaluate_fixture
from vision_core.policy import load_policy

# Small ephemeral tile only. Never persisted.
MAX_TILE_B64_CHARS = 96_000
MAX_TILE_DECODED_BYTES = 65_536
MAX_TILE_EDGE = 256
MAX_BODY_BYTES = 600_000  # one bounded JPEG crop as base64 + a few fields

SERVICE = SecondLookService()
METRICS = telemetry.Metrics(os.environ.get("SG_METRICS_SERVICE", "l2-second-look"))


@asynccontextmanager
async def _lifespan(_app):
    if os.environ.get("SG_PRELOAD", "") == "1":
        _ = SERVICE.detector  # load + warm the net before the first request
    yield


app = FastAPI(
    title="Screen Guardian Cloud API",
    version="0.4.0",
    description=(
        "L2 COOL second look for escalated picture crops (bearer token, RAM only) "
        "and allowlisted fixture evaluation. Rejects URLs and uploads."
    ),
    lifespan=_lifespan,
)


class SecondLookRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jpeg_b64: str = Field(description="One JPEG of the picture rect. <= 400 kB, edge 16..1600 px. RAM only.")
    dhash: str = Field(default="", max_length=16, description="64-bit dHash of the picture (hex)")
    ahash: str = Field(default="", max_length=16, description="64-bit aHash of the picture (hex)")
    mode: str = "aggressive"
    profile: str = "l2"
    picture_id: str = Field(default="", max_length=64, description="Opaque region id; no titles or URLs")
    reason: str = Field(default="", max_length=32)


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dhash: str = Field(max_length=16)
    mode: str = "aggressive"


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixture_id: str
    policy_id: str


class PerceiveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixture_id: str | None = None
    frame_id: int | None = None
    policy_id: str = "balanced"
    path: str | None = Field(
        default=None,
        description="Optional relative fixtures/safe/<id>.mp4 path. Never a URL.",
    )
    tile_b64: str | None = Field(
        default=None,
        description="Optional small JPEG/PNG tile. Max 64 KiB decoded. Never persisted.",
    )
    tile_xyxy: list[float] | None = None
    tile_width: int | None = None
    tile_height: int | None = None


class LivePerceiveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str = "balanced"
    frame_id: int = 0
    orig_width: int
    orig_height: int
    jpeg_b64: str = Field(description="Allowlisted JPEG tile or downscaled frame. Never raw desktop.")
    tile_xyxy: list[float] | None = None
    candidates: list[dict[str, Any]] | None = None
    payload_bytes: int | None = None


def _http_from_catalog(exc: CatalogError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.message)


def _region_to_dict(region: TrackedRegion | None) -> dict[str, Any] | None:
    if region is None:
        return None
    return {
        "track_id": region.track_id,
        "label": region.label,
        "xyxy": [float(v) for v in region.xyxy],
        "stable_score": float(region.stable_score),
        "mask_on": bool(region.mask_on),
    }


def _action_to_dict(action: ProtectionAction) -> dict[str, Any]:
    region = _region_to_dict(action.region)
    return {
        "action": action.action.value,
        "reason_code": action.reason_code,
        "policy_version": action.policy_version,
        "trace_id": action.trace_id,
        "region": region,
    }


def _public_metrics(results: dict[str, Any], fixture_id: str, policy_id: str) -> dict[str, Any]:
    """Strip local filesystem paths from eval output."""
    skip = {"fixture", "gt"}
    metrics = {key: value for key, value in results.items() if key not in skip}
    metrics["fixture_id"] = fixture_id
    metrics["policy_id"] = policy_id
    return metrics


def _read_fixture_frames(video_path: Any, frame_id: int) -> tuple[np.ndarray, np.ndarray | None]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="fixture video could not be opened")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_id < 0 or (total and frame_id >= total):
            raise HTTPException(status_code=400, detail="frame_id out of range")
        prev: np.ndarray | None = None
        if frame_id > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_id - 1))
            ok_prev, prev_bgr = cap.read()
            prev = prev_bgr if ok_prev else None
            ok, current = cap.read()
        else:
            ok, current = cap.read()
        if not ok or current is None:
            raise HTTPException(status_code=400, detail="could not read fixture frame")
        return current, prev
    finally:
        cap.release()


def _decode_tile(tile_b64: str) -> np.ndarray:
    if len(tile_b64) > MAX_TILE_B64_CHARS:
        raise HTTPException(status_code=400, detail="tile_b64 exceeds max size")
    try:
        raw = base64.b64decode(tile_b64, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="tile_b64 is not valid base64") from exc
    if len(raw) > MAX_TILE_DECODED_BYTES:
        raise HTTPException(status_code=400, detail="tile exceeds max bytes")
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="tile_b64 is not a decodable image")
    height, width = image.shape[:2]
    if height > MAX_TILE_EDGE or width > MAX_TILE_EDGE:
        raise HTTPException(status_code=400, detail="tile exceeds max edge length")
    return image


@app.middleware("http")
async def reject_non_json_uploads(request: Request, call_next):
    if request.method == "POST" and request.url.path.startswith("/v1/"):
        try:
            length = int(request.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        if length > MAX_BODY_BYTES:
            return JSONResponse(status_code=413, content={"detail": "body too large", "image_persisted": False})
        content_type = (request.headers.get("content-type") or "").lower()
        if "multipart/form-data" in content_type or "application/octet-stream" in content_type:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "uploads are not allowed; use an allowlisted fixture_id",
                    "image_persisted": False,
                },
            )
    return await call_next(request)


@app.exception_handler(CatalogError)
async def catalog_error_handler(_request: Request, exc: CatalogError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "image_persisted": False},
    )


def _bedrock_status() -> tuple[bool, str]:
    """This process calls Bedrock only if ``SG_BEDROCK_PROBE=1`` (probe is
    cached). The box itself does not; the agent lane is AgentCore."""
    if os.environ.get("SG_BEDROCK_PROBE", "") != "1":
        return False, "not used by this process (agent lane = AgentCore Runtime)"
    from agent.bedrock_adapter import BedrockAdapter

    adapter = getattr(_bedrock_status, "_adapter", None) or BedrockAdapter(timeout_s=8.0)
    _bedrock_status._adapter = adapter  # type: ignore[attr-defined]
    ok = adapter.available()
    return ok, "" if ok else adapter.unavailable_reason[:160]


@app.get("/health")
def health() -> dict[str, Any]:
    version = str(cv2.__version__)
    build = opencv_build()
    bedrock_ok, bedrock_note = _bedrock_status()
    loaded = SERVICE._detector is not None
    return {
        "ok": True,
        "opencv_version": version,
        "opencv_build": build,
        "agent": {
            "status": "ready",
            "mode": "agentcore" if os.environ.get("SG_AGENT_RUNTIME_ARN") else "local_deterministic",
            "controller": "agent.orchestrator.Orchestrator",
        },
        "bedrock_available": bedrock_ok,
        "bedrock_note": bedrock_note,
        "region": os.environ.get("AWS_REGION", "us-east-1"),
        "image_persisted": False,
        "runtime": os.environ.get("SCREEN_GUARDIAN_RUNTIME", "local"),
        "second_look": {
            "route": "/v1/second_look",
            "auth": "bearer" if expected_token() else "unconfigured",
            "detector_loaded": loaded,
            "detector": type(SERVICE._detector).__name__ if loaded else "",
            "cache_entries": len(SERVICE.cache),
            "review_holds": len(SERVICE.holds),
        },
        "live_perceive": True,
        "nudenet": nudenet_on_server(),
    }


def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@app.post("/v1/second_look", dependencies=[Depends(require_token)])
def post_second_look(body: SecondLookRequest) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        out = SERVICE.look(
            jpeg_b64=body.jpeg_b64, dhash=body.dhash, ahash=body.ahash, mode=body.mode, profile=body.profile
        )
    except SecondLookError as exc:
        raise _bad_request(exc) from exc
    out["opencv"] = {"version": opencv_build()["version"], "cool": opencv_build()["cool"]}
    ms = (time.perf_counter() - t0) * 1000.0
    state = out["verdict"]["state"]
    METRICS.put("SecondLookRequests", 1)
    METRICS.put("SecondLookLatencyMs", ms, "Milliseconds")
    METRICS.put("SecondLookInferMs", out["timings_ms"]["infer"], "Milliseconds")
    METRICS.put("SecondLookCacheHits", 1 if out["source"] == "cloud_cache" else 0)
    METRICS.put("VerdictUnsure", 1 if state == "UNSURE" else 0)
    METRICS.put("VerdictUnsafe", 1 if state == "UNSAFE" else 0)
    telemetry.log(
        "second_look", dhash=out["dhash"], mode=out["mode"], profile=out["profile"], source=out["source"],
        state=state, score=out["verdict"]["score"], n_views=out["n_views"], bytes=out["payload_bytes"],
        infer_ms=out["timings_ms"]["infer"], ms=round(ms, 2), reason=body.reason, persisted=False,
    )
    return out


@app.post("/v1/review", dependencies=[Depends(require_token)])
def post_review(body: ReviewRequest) -> dict[str, Any]:
    try:
        out = SERVICE.review(dhash=body.dhash, mode=body.mode)
    except (SecondLookError, ValueError) as exc:
        raise _bad_request(exc) from exc
    out["opencv"] = {"version": opencv_build()["version"], "cool": opencv_build()["cool"]}
    METRICS.put("ReviewRequests", 1)
    telemetry.log("review", dhash=out["dhash"], found=out["found"],
                  state=(out.get("verdict") or {}).get("state", ""), persisted=False)
    return out


@app.get("/v1/policies")
def get_policies() -> dict[str, Any]:
    return {"policies": list_policies()}


@app.get("/v1/fixtures")
def get_fixtures() -> dict[str, Any]:
    return {"fixtures": list_fixtures()}


@app.post("/v1/evaluations")
async def post_evaluations(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="body must be JSON") from exc
    reject_upload_or_url_body(payload)
    try:
        body = EvaluationRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    try:
        video, gt = resolve_fixture(body.fixture_id)
        policy = resolve_policy(body.policy_id)
    except CatalogError as exc:
        raise _http_from_catalog(exc) from exc

    run_id = uuid.uuid4().hex[:16]
    out_dir = VISION_ROOT / "experiments" / "api_runs" / run_id
    results = evaluate_fixture(video, gt, policy, out_dir)
    return {
        "run_id": run_id,
        "fixture_id": body.fixture_id,
        "policy_id": body.policy_id,
        "image_persisted": False,
        "metrics": _public_metrics(results, body.fixture_id, body.policy_id),
    }


@app.post("/v1/perceive")
async def post_perceive(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="body must be JSON") from exc
    reject_upload_or_url_body(payload)
    try:
        body = PerceiveRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    try:
        policy_path = resolve_policy(body.policy_id)
    except CatalogError as exc:
        raise _http_from_catalog(exc) from exc

    source = "fixture_frame"
    fixture_id = body.fixture_id
    frame_id = 0 if body.frame_id is None else int(body.frame_id)
    prev: np.ndarray | None = None

    if fixture_id:
        try:
            video, _gt = resolve_fixture(fixture_id)
        except CatalogError as exc:
            raise _http_from_catalog(exc) from exc
        if body.frame_id is None:
            raise HTTPException(status_code=400, detail="frame_id is required with fixture_id")
        current, prev = _read_fixture_frames(video, frame_id)
        source = "fixture_frame"
    elif body.path:
        if body.frame_id is None:
            raise HTTPException(status_code=400, detail="frame_id is required with path")
        try:
            fixture_id, video, _gt = resolve_allowlisted_relpath(body.path)
        except CatalogError as exc:
            raise _http_from_catalog(exc) from exc
        current, prev = _read_fixture_frames(video, frame_id)
        source = "fixture_frame"
    elif body.tile_b64:
        current = _decode_tile(body.tile_b64)
        source = "ephemeral_tile"
        # Tile metadata is accepted for the edge socket; we do not persist pixels.
        _ = (body.tile_xyxy, body.tile_width, body.tile_height)
    else:
        raise HTTPException(
            status_code=400,
            detail="provide fixture_id+frame_id (preferred), an allowlisted path, or a small tile_b64",
        )

    agent = Orchestrator(load_policy(policy_path))
    fps = 30.0
    result = agent.process_frame(frame_id, current, prev, frame_id * (1000.0 / fps))
    traces = [trace_to_dict(trace) for trace in result.traces]
    step = traces[-1] if traces else {
        "tool": "agent.decide",
        "decision": result.final_decision,
        "image_persisted": False,
    }
    return {
        "fixture_id": fixture_id,
        "frame_id": frame_id,
        "policy_id": body.policy_id,
        "source": source,
        "image_persisted": False,
        "actions": [_action_to_dict(action) for action in result.actions],
        "trace": step,
        "traces": traces,
        "tool_sequence": result.tool_sequence,
        "final_decision": result.final_decision,
        "band": result.band,
        "companion_line": result.companion_line,
        "reinspect_count": result.reinspect_count,
        "timed_out": result.timed_out,
    }


@app.post("/v1/live_perceive")
async def post_live_perceive(request: Request) -> dict[str, Any]:
    """Laptop live tiles. JPEG only. Never persisted."""
    try:
        payload = await request.json()
    except Exception as extra:
        raise HTTPException(status_code=400, detail="body must be JSON") from extra
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    extra_keys = {"url", "uri", "screenshot", "image", "upload", "file"} & set(payload)
    if extra_keys:
        raise HTTPException(
            status_code=400,
            detail="URL/upload fields are not allowed; send jpeg_b64 tiles only",
        )
    try:
        body = LivePerceiveRequest.model_validate(payload)
    except ValidationError as extra:
        raise HTTPException(status_code=422, detail=extra.errors()) from extra
    try:
        policy_path = resolve_policy(body.policy_id)
    except CatalogError as extra:
        raise _http_from_catalog(extra) from extra

    ran = run_live_perceive(
        policy_path=policy_path,
        policy_id=body.policy_id,
        frame_id=int(body.frame_id),
        jpeg_b64=body.jpeg_b64,
        tile_xyxy=body.tile_xyxy,
        orig_width=int(body.orig_width),
        orig_height=int(body.orig_height),
        candidates=body.candidates,
    )
    result = ran["result"]
    traces = [trace_to_dict(trace) for trace in result.traces]
    step = traces[-1] if traces else {
        "tool": "agent.decide",
        "decision": result.final_decision,
        "image_persisted": False,
    }
    print(
        f"live_perceive frame={body.frame_id} jpeg={ran['payload_bytes']}B "
        f"tile={ran['tile_width']}x{ran['tile_height']} "
        f"detector={ran['detector_id']} persisted=false",
        flush=True,
    )
    return {
        "fixture_id": None,
        "frame_id": int(body.frame_id),
        "policy_id": body.policy_id,
        "source": "live_tile",
        "image_persisted": False,
        "payload_bytes": ran["payload_bytes"],
        "tile_width": ran["tile_width"],
        "tile_height": ran["tile_height"],
        "orig_width": int(body.orig_width),
        "orig_height": int(body.orig_height),
        "detector_id": ran["detector_id"],
        "n_candidates": ran["n_candidates"],
        "actions": [_action_to_dict(action) for action in result.actions],
        "trace": step,
        "traces": traces,
        "tool_sequence": result.tool_sequence,
        "final_decision": result.final_decision,
        "band": result.band,
        "companion_line": result.companion_line,
        "reinspect_count": result.reinspect_count,
        "timed_out": result.timed_out,
    }
