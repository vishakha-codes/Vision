"""Cloud evaluation API: allowlist, no uploads, no persisted images."""

from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from cloud.api import app
from cloud.catalog import allowlisted_fixture_ids, allowlisted_policy_ids


def test_health_reports_opencv_5_and_no_secrets() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    # Mac: stock OpenCV 5; the Graviton box reports COOL 4.14.0-pre.
    assert body["opencv_version"] == body["opencv_build"]["version"]
    assert body["ok"] is True
    assert body["bedrock_available"] is False  # this process does not call Bedrock unless SG_BEDROCK_PROBE=1
    assert body["agent"]["status"] == "ready"
    assert body["image_persisted"] is False
    dumped = response.text.lower()
    assert "akia" not in dumped
    assert "secret" not in dumped
    assert "aws_secret" not in dumped


def test_lists_only_allowlisted_policies_and_safe_fixtures() -> None:
    client = TestClient(app)
    policies = client.get("/v1/policies").json()["policies"]
    ids = {item["policy_id"] for item in policies}
    assert ids == {"strict", "balanced", "relaxed"}
    assert ids == set(allowlisted_policy_ids())

    fixtures = client.get("/v1/fixtures").json()["fixtures"]
    fixture_ids = {item["fixture_id"] for item in fixtures}
    assert "synthetic_moving_regions" in fixture_ids
    assert fixture_ids == set(allowlisted_fixture_ids())
    assert all(item["root"] == "fixtures/safe" for item in fixtures)
    assert all(item["allowlisted"] is True for item in fixtures)


def test_evaluations_reject_unknown_fixture_and_urls() -> None:
    client = TestClient(app)
    unknown = client.post(
        "/v1/evaluations",
        json={"fixture_id": "not_a_real_fixture", "policy_id": "balanced"},
    )
    assert unknown.status_code == 400
    assert "unknown fixture_id" in unknown.json()["detail"]

    remote = client.post(
        "/v1/evaluations",
        json={
            "fixture_id": "https://example.com/clip.mp4",
            "policy_id": "balanced",
        },
    )
    assert remote.status_code == 400

    upload = client.post(
        "/v1/evaluations",
        json={
            "fixture_id": "synthetic_moving_regions",
            "policy_id": "balanced",
            "url": "https://evil.example/x",
        },
    )
    assert upload.status_code == 400

    multipart = client.post(
        "/v1/evaluations",
        files={"file": ("shot.png", b"not-an-image", "image/png")},
    )
    assert multipart.status_code == 400
    assert multipart.json()["image_persisted"] is False


def test_perceive_allowlisted_path_does_not_persist() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/perceive",
        json={
            "path": "fixtures/safe/synthetic_moving_regions.mp4",
            "frame_id": 12,
            "policy_id": "balanced",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["image_persisted"] is False
    assert body["fixture_id"] == "synthetic_moving_regions"
    assert body["source"] == "fixture_frame"

    remote = client.post(
        "/v1/perceive",
        json={
            "path": "https://evil.example/clip.mp4",
            "frame_id": 0,
            "policy_id": "balanced",
        },
    )
    assert remote.status_code == 400


def test_perceive_fixture_frame_does_not_persist() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/perceive",
        json={
            "fixture_id": "synthetic_moving_regions",
            "frame_id": 12,
            "policy_id": "balanced",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["image_persisted"] is False
    assert body["source"] == "fixture_frame"
    assert body["trace"]["image_persisted"] is False
    assert isinstance(body["actions"], list)
    assert body["final_decision"]
    dumped = response.text
    assert "image_bgr" not in dumped


def test_perceive_rejects_huge_tile() -> None:
    client = TestClient(app)
    huge = base64.b64encode(b"x" * 80_000).decode("ascii")
    response = client.post(
        "/v1/perceive",
        json={"tile_b64": huge, "policy_id": "balanced"},
    )
    assert response.status_code == 400


def test_live_perceive_tiny_jpeg_is_ephemeral() -> None:
    from vision_core.encode_tile import encode_tile

    client = TestClient(app)
    tile = np.zeros((120, 160, 3), dtype=np.uint8)
    tile[24:96, 32:128] = (200, 40, 220)
    jpeg = encode_tile(tile, max_edge=160, quality=70)
    assert jpeg[:2] == b"\xff\xd8"
    response = client.post(
        "/v1/live_perceive",
        json={
            "policy_id": "balanced",
            "frame_id": 0,
            "orig_width": 160,
            "orig_height": 120,
            "jpeg_b64": base64.b64encode(jpeg).decode("ascii"),
            "tile_xyxy": [0, 0, 160, 120],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["image_persisted"] is False
    assert body["source"] == "live_tile"
    assert body["payload_bytes"] == len(jpeg)
    assert body["trace"]["image_persisted"] is False
    assert "image_bgr" not in response.text
    assert isinstance(body["actions"], list)


def test_live_perceive_rejects_huge_or_non_jpeg() -> None:
    client = TestClient(app)
    huge = base64.b64encode(b"\xff\xd8" + b"x" * 300_000).decode("ascii")
    too_big = client.post(
        "/v1/live_perceive",
        json={
            "policy_id": "balanced",
            "frame_id": 0,
            "orig_width": 100,
            "orig_height": 100,
            "jpeg_b64": huge,
        },
    )
    assert too_big.status_code == 400

    png = np.full((32, 32, 3), 40, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", png)
    assert ok
    not_jpeg = client.post(
        "/v1/live_perceive",
        json={
            "policy_id": "balanced",
            "frame_id": 0,
            "orig_width": 32,
            "orig_height": 32,
            "jpeg_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
        },
    )
    assert not_jpeg.status_code == 400

    sneaky = client.post(
        "/v1/live_perceive",
        json={
            "policy_id": "balanced",
            "frame_id": 0,
            "orig_width": 32,
            "orig_height": 32,
            "jpeg_b64": "aaaa",
            "screenshot": "nope",
        },
    )
    assert sneaky.status_code == 400


def test_perceive_small_tile_stays_ephemeral() -> None:
    client = TestClient(app)
    tile = np.full((32, 32, 3), 40, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", tile)
    assert ok
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    response = client.post(
        "/v1/perceive",
        json={"tile_b64": payload, "policy_id": "balanced", "frame_id": 0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "ephemeral_tile"
    assert body["image_persisted"] is False
    assert body["trace"]["image_persisted"] is False
