"""Ephemeral live-tile brain for ``POST /v1/live_perceive``.

Legacy single-thread reflex only (``edge.app --engine legacy``); the live
engine uses ``/v1/second_look``. Laptop eyes send JPEG tiles (or one
downscaled frame). This module decodes in RAM, runs the legacy orchestrator
with the shared live detector (``cloud.cool_detector``: every maskable class
per anchor, COOL cv2.dnn on the box), and returns boxes remapped to the
original desktop. Nothing is written to disk. Delete with the legacy engine.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import replace
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException

from agent.orchestrator import Orchestrator
from vision_core.contracts import Detection, ProtectionAction, TrackedRegion
from vision_core.detect_regions import detect_regions
from vision_core.encode_tile import remap_xyxy_from_tile, xyxy_to_tile
from vision_core.policy import load_policy
from vision_core.reinspect_crop import reinspect_crop

MAX_LIVE_B64_CHARS = 360_000
MAX_LIVE_JPEG_BYTES = 250_000
MAX_LIVE_EDGE = 960
MAX_LIVE_TILES = 4


def nudenet_on_server() -> bool:
    """NudeNet 320n loadable through this process's cv2.dnn (COOL on the box)."""
    try:
        from cloud.cool_detector import shared_detector

        shared_detector()
        return True
    except Exception:
        return False


def decode_live_jpeg(jpeg_b64: str) -> tuple[np.ndarray, int]:
    if not jpeg_b64:
        raise HTTPException(status_code=400, detail="jpeg_b64 is required")
    if len(jpeg_b64) > MAX_LIVE_B64_CHARS:
        raise HTTPException(status_code=400, detail="jpeg_b64 exceeds max size")
    try:
        raw = base64.b64decode(jpeg_b64, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="jpeg_b64 is not valid base64") from exc
    if len(raw) > MAX_LIVE_JPEG_BYTES:
        raise HTTPException(status_code=400, detail="jpeg exceeds max bytes")
    if raw[:2] != b"\xff\xd8":
        raise HTTPException(status_code=400, detail="payload must be JPEG")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="jpeg is not decodable")
    height, width = image.shape[:2]
    if height > MAX_LIVE_EDGE or width > MAX_LIVE_EDGE:
        raise HTTPException(status_code=400, detail="tile exceeds max edge length")
    return image, len(raw)


def _server_detect(policy_id: str):
    """Live detector (all maskable classes per anchor, server cv2.dnn) behind
    the legacy ``detect_nsfw`` policy logic; HSV toy only if no model loads."""
    if nudenet_on_server():
        from cloud.cool_detector import live_box_fn
        from vision_core.detect_nsfw import make_nsfw_detect

        return make_nsfw_detect(policy_id, box_fn=live_box_fn, tile=False), "nudenet_onnx_320n_v1"
    return detect_regions, "synthetic_hsv_v1"


def _as_xyxy(raw: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))


def _candidate_detections(
    candidates: list[dict[str, Any]] | None,
    tile_xyxy: tuple[float, float, float, float],
    tile_wh: tuple[int, int],
    frame_id: int,
) -> list[Detection]:
    out: list[Detection] = []
    for item in candidates or []:
        if not isinstance(item, dict):
            continue
        xyxy = _as_xyxy(item.get("xyxy"))
        if xyxy is None:
            continue
        out.append(
            Detection(
                label=str(item.get("label") or "flagged"),
                score=float(item.get("score") or 0.0),
                xyxy=xyxy_to_tile(xyxy, tile_xyxy, tile_wh),
                detector_id="edge_candidate",
                frame_id=frame_id,
            )
        )
    return out


def _wrap_detect(base_fn, extras: list[Detection]):
    def _detect(image, **kwargs):
        found = list(base_fn(image, **kwargs) or [])
        return found + extras

    return _detect


def _remap_region(
    region: TrackedRegion | None,
    tile_xyxy: tuple[float, float, float, float],
    tile_wh: tuple[int, int],
) -> TrackedRegion | None:
    if region is None:
        return None
    return replace(region, xyxy=remap_xyxy_from_tile(region.xyxy, tile_xyxy, tile_wh))


def _remap_action(
    action: ProtectionAction,
    tile_xyxy: tuple[float, float, float, float],
    tile_wh: tuple[int, int],
) -> ProtectionAction:
    return replace(action, region=_remap_region(action.region, tile_xyxy, tile_wh))


def run_live_perceive(
    *,
    policy_path,
    policy_id: str,
    frame_id: int,
    jpeg_b64: str,
    tile_xyxy: list[float] | None,
    orig_width: int,
    orig_height: int,
    candidates: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    image, nbytes = decode_live_jpeg(jpeg_b64)
    height, width = image.shape[:2]
    tile_box = _as_xyxy(tile_xyxy) or (0.0, 0.0, float(orig_width or width), float(orig_height or height))
    tile_wh = (width, height)
    base_fn, detector_id = _server_detect(policy_id)
    extras = _candidate_detections(candidates, tile_box, tile_wh, frame_id)
    cfg = load_policy(policy_path)
    if detector_id == "nudenet_onnx_320n_v1":
        from vision_core.detect_nsfw import tune_policy_for_live_nsfw
        from vision_core.reinspect_nsfw import make_nsfw_reinspect

        cfg = tune_policy_for_live_nsfw(cfg)
        reinspect_fn = make_nsfw_reinspect(mode="local", detect_fn=base_fn)
    else:
        reinspect_fn = reinspect_crop
    agent = Orchestrator(
        cfg,
        detect_fn=_wrap_detect(base_fn, extras),
        reinspect_fn=reinspect_fn,
    )
    result = agent.process_frame(frame_id, image, None, float(frame_id) * (1000.0 / 30.0))
    actions = [_remap_action(action, tile_box, tile_wh) for action in result.actions]
    tracks = [_remap_region(track, tile_box, tile_wh) for track in result.tracks]
    result.actions = actions
    result.tracks = tracks
    return {
        "result": result,
        "payload_bytes": nbytes,
        "tile_width": width,
        "tile_height": height,
        "detector_id": detector_id,
        "n_candidates": len(extras),
    }
