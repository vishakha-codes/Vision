"""JPEG-encode an in-memory BGR crop for AWS travel.

``POST /v1/perceive`` is still fixture identity (no pixels). Live nsfw
uses this helper to send **tiles / a downscaled frame** to
``POST /v1/live_perceive`` — never an uncompressed 3456×2234 desktop.

Defaults: crop max-edge 640, full-frame max-edge 960, JPEG quality 70.
Never writes disk.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

DEFAULT_MAX_EDGE = 640
LIVE_FRAME_MAX_EDGE = 960
DEFAULT_JPEG_QUALITY = 70
LIVE_MAX_TILES = 3


def resize_for_travel(
    bgr: np.ndarray,
    *,
    max_edge: int = DEFAULT_MAX_EDGE,
) -> np.ndarray:
    """Downscale so ``max(h, w) <= max_edge``. Never upscales."""
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return bgr
    height, width = bgr.shape[:2]
    longest = max(height, width)
    if longest <= max_edge or max_edge <= 0:
        return bgr
    scale = max_edge / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_tile(
    bgr: np.ndarray,
    *,
    max_edge: int = DEFAULT_MAX_EDGE,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    """Return JPEG bytes of ``bgr`` capped at ``max_edge``.

    Failed / empty input returns ``b""``. Never writes disk.
    """
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return b""
    tile = resize_for_travel(bgr, max_edge=max_edge)
    q = int(max(1, min(100, quality)))
    ok, buf = cv2.imencode(".jpg", tile, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return b""
    return bytes(buf)


def remap_xyxy_from_tile(
    xyxy: tuple[float, float, float, float],
    tile_xyxy: tuple[float, float, float, float],
    tile_wh: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Map a box from encoded-tile pixels back to the original frame."""
    tx1, ty1, tx2, ty2 = tile_xyxy
    width, height = tile_wh
    if width <= 0 or height <= 0:
        return xyxy
    sx = (tx2 - tx1) / float(width)
    sy = (ty2 - ty1) / float(height)
    x1, y1, x2, y2 = xyxy
    return (tx1 + x1 * sx, ty1 + y1 * sy, tx1 + x2 * sx, ty1 + y2 * sy)


def xyxy_to_tile(
    xyxy: tuple[float, float, float, float],
    tile_xyxy: tuple[float, float, float, float],
    tile_wh: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Map an original-frame box into encoded-tile pixels."""
    tx1, ty1, tx2, ty2 = tile_xyxy
    width, height = tile_wh
    span_x = tx2 - tx1
    span_y = ty2 - ty1
    if span_x <= 0 or span_y <= 0:
        return xyxy
    sx = width / float(span_x)
    sy = height / float(span_y)
    x1, y1, x2, y2 = xyxy
    return ((x1 - tx1) * sx, (y1 - ty1) * sy, (x2 - tx1) * sx, (y2 - ty1) * sy)


def pick_live_tile(
    tiles: list[dict[str, Any]],
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Prefer the JPEG whose crop covers the most local NSFW boxes."""
    if not tiles:
        return None
    if not candidates:
        return tiles[0]
    best = tiles[0]
    best_score = -1.0
    for tile in tiles:
        box = tile.get("xyxy") or [0.0, 0.0, 0.0, 0.0]
        if len(box) < 4:
            continue
        tx1, ty1, tx2, ty2 = (float(v) for v in box[:4])
        score = 0.0
        for item in candidates:
            xyxy = item.get("xyxy") if isinstance(item, dict) else None
            if not xyxy or len(xyxy) < 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                score += 10.0 + float(item.get("score") or 0.0)
            ix = min(tx2, x2) - max(tx1, x1)
            iy = min(ty2, y2) - max(ty1, y1)
            if ix > 0.0 and iy > 0.0:
                score += ix * iy
        if score > best_score:
            best_score = score
            best = tile
    return best


def pack_live_frame(
    bgr: np.ndarray,
    *,
    max_tiles: int = LIVE_MAX_TILES,
    crop_edge: int = DEFAULT_MAX_EDGE,
    frame_edge: int = LIVE_FRAME_MAX_EDGE,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> dict[str, Any]:
    """Pick JPEG tiles (or one downscaled frame) for ``/v1/live_perceive``.

    Prefers high-variance content crops so a SERP thumb can travel without
    the rest of the desktop. Falls back to one max-edge-960 JPEG of the
    full frame. Never returns raw capture pixels.
    """
    height, width = (0, 0) if bgr is None or bgr.size == 0 else bgr.shape[:2]
    tiles: list[dict[str, Any]] = []
    if height >= 32 and width >= 32:
        try:
            from vision_core.detect_nsfw import propose_content_tiles

            crops = propose_content_tiles(bgr)
        except Exception:
            crops = []
        for x1, y1, x2, y2 in crops[: max(1, int(max_tiles))]:
            jpeg = encode_tile(bgr[y1:y2, x1:x2], max_edge=crop_edge, quality=quality)
            if jpeg:
                tiles.append({"jpeg": jpeg, "xyxy": [float(x1), float(y1), float(x2), float(y2)]})
    if not tiles and height > 0 and width > 0:
        jpeg = encode_tile(bgr, max_edge=frame_edge, quality=quality)
        if jpeg:
            tiles.append({"jpeg": jpeg, "xyxy": [0.0, 0.0, float(width), float(height)]})
    return {
        "orig_width": int(width),
        "orig_height": int(height),
        "tiles": tiles,
        "payload_bytes": sum(len(tile["jpeg"]) for tile in tiles),
    }
