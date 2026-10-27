"""JPEG crop encoder for optional Rekognition reinspect — not live detect."""

from __future__ import annotations

import cv2
import numpy as np

from vision_core.encode_tile import encode_tile, pack_live_frame, pick_live_tile, resize_for_travel
from vision_core.reinspect_nsfw import reinspect_rekognition


def test_encode_tile_caps_long_edge_and_is_jpeg() -> None:
    image = np.full((1200, 1800, 3), 40, dtype=np.uint8)
    image[100:400, 200:700] = 180
    payload = encode_tile(image)
    assert payload[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert max(decoded.shape[:2]) == 640


def test_encode_tile_does_not_upscale() -> None:
    small = np.full((80, 120, 3), 90, dtype=np.uint8)
    resized = resize_for_travel(small)
    assert resized.shape[:2] == (80, 120)
    payload = encode_tile(small)
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[0] <= 80
    assert decoded.shape[1] <= 120


def test_encode_tile_quality_beats_default_jpeg() -> None:
    rng = np.random.default_rng(7)
    noisy = rng.integers(0, 255, size=(800, 800, 3), dtype=np.uint8)
    tight = encode_tile(noisy, max_edge=640, quality=70)
    ok, fat_buf = cv2.imencode(".jpg", resize_for_travel(noisy), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok and tight
    assert len(tight) < len(fat_buf.tobytes())


def test_encode_tile_empty_is_empty() -> None:
    assert encode_tile(np.zeros((0, 0, 3), dtype=np.uint8)) == b""


def test_pack_live_frame_never_returns_raw_desktop() -> None:
    image = np.full((800, 1200, 3), 30, dtype=np.uint8)
    image[40:300, 80:400] = 200
    packed = pack_live_frame(image)
    assert packed["orig_width"] == 1200
    assert packed["orig_height"] == 800
    assert packed["tiles"]
    assert packed["payload_bytes"] == sum(len(tile["jpeg"]) for tile in packed["tiles"])
    for tile in packed["tiles"]:
        assert tile["jpeg"][:2] == b"\xff\xd8"
        decoded = cv2.imdecode(np.frombuffer(tile["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert max(decoded.shape[:2]) <= 960


def test_pick_live_tile_prefers_candidate_crop() -> None:
    tiles = [
        {"jpeg": b"a", "xyxy": [0.0, 0.0, 100.0, 100.0]},
        {"jpeg": b"b", "xyxy": [400.0, 200.0, 700.0, 500.0]},
    ]
    candidates = [{"xyxy": [420.0, 220.0, 520.0, 320.0], "score": 0.8}]
    chosen = pick_live_tile(tiles, candidates)
    assert chosen is not None
    assert chosen["jpeg"] == b"b"


def test_rekognition_reinspect_uses_encode_tile() -> None:
    calls: list[dict] = []

    class Client:
        def detect_moderation_labels(self, **kwargs):
            calls.append(kwargs)
            return {"ModerationLabels": [{"Name": "Explicit Nudity", "Confidence": 88.0}]}

    image = np.full((900, 1400, 3), 28, dtype=np.uint8)
    result = reinspect_rekognition(image, (10.0, 10.0, 900.0, 700.0), margin=0, client=Client())
    assert result.label == "flagged"
    payload = calls[0]["Image"]["Bytes"]
    assert payload[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert max(decoded.shape[:2]) <= 640
