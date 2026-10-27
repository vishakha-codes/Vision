"""L2 /v1/second_look + /v1/review: auth, bounds, RAM-only hold, consensus."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import cloud.api as api
from cloud import auth
from cloud.second_look import SecondLookService, consensus, views_for
from vision_core.live.detector import RawDet


class StubDetector:
    """Weak hit in every view (consensus keeps it → UNSURE in AGGRESSIVE)."""

    input_size = 320

    def __init__(self, score: float = 0.2) -> None:
        self.score = score
        self.calls = 0

    def detect_batch(self, images):
        self.calls += 1
        out = []
        for img in images:
            h, w = img.shape[:2]
            out.append([RawDet("BUTTOCKS_EXPOSED", self.score, (0.3 * w, 0.3 * h, 0.7 * w, 0.7 * h))])
        return out


def _jpeg(h=300, w=400) -> str:
    img = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    return base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("SG_API_TOKEN", "tok")
    monkeypatch.setenv("SG_METRICS", "0")
    auth.reset_cache()
    stub = StubDetector()
    monkeypatch.setattr(api, "SERVICE", SecondLookService(stub))
    monkeypatch.setattr(api.METRICS, "enabled", False)
    yield TestClient(app=api.app), stub
    auth.reset_cache()


H = {"Authorization": "Bearer tok"}


def test_requires_bearer_token(client) -> None:
    c, _ = client
    body = {"jpeg_b64": _jpeg(), "dhash": "ab", "mode": "aggressive"}
    assert c.post("/v1/second_look", json=body).status_code == 401
    assert c.post("/v1/second_look", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_no_token_configured_fails_closed(monkeypatch) -> None:
    monkeypatch.delenv("SG_API_TOKEN", raising=False)
    monkeypatch.setenv("SG_NO_SECRETS_MANAGER", "1")
    auth.reset_cache()
    r = TestClient(app=api.app).post("/v1/second_look", json={"jpeg_b64": _jpeg()}, headers=H)
    assert r.status_code == 503
    auth.reset_cache()


def test_unsure_is_cached_held_then_reviewed_once(client) -> None:
    c, stub = client
    body = {"jpeg_b64": _jpeg(), "dhash": "00ff00ff00ff00ff", "ahash": "1", "mode": "aggressive"}
    r = c.post("/v1/second_look", json=body, headers=H).json()
    assert r["verdict"]["state"] == "UNSURE" and r["image_persisted"] is False
    assert r["review_hold_s"] > 0 and r["n_views"] == len(views_for(400, 300, "l2"))
    again = c.post("/v1/second_look", json=body, headers=H).json()
    assert again["source"] == "cloud_cache" and stub.calls == 1
    rev = c.post("/v1/review", json={"dhash": "00ff00ff00ff00ff", "mode": "aggressive"}, headers=H).json()
    assert rev["found"] is True and rev["profile"] == "deep"
    assert api.SERVICE.holds.pop(0x00FF00FF00FF00FF) is None  # hold dropped after the review


def test_rejects_non_jpeg_oversize_and_extra_fields(client) -> None:
    c, _ = client
    png = base64.b64encode(cv2.imencode(".png", np.zeros((50, 50, 3), np.uint8))[1].tobytes()).decode()
    assert c.post("/v1/second_look", json={"jpeg_b64": png}, headers=H).status_code == 400
    assert c.post("/v1/second_look", json={"jpeg_b64": _jpeg(), "url": "http://x"}, headers=H).status_code == 422
    big = "A" * 700_000
    assert c.post("/v1/second_look", json={"jpeg_b64": big}, headers=H).status_code == 413
    assert c.post("/v1/second_look", json={"jpeg_b64": _jpeg(2000, 100)}, headers=H).status_code == 400


def test_consensus_drops_single_view_hallucinations() -> None:
    box = (0.2, 0.2, 0.5, 0.5)
    lone = [[("BUTTOCKS_EXPOSED", 0.4, box)], [], []]
    agreed = [[("BUTTOCKS_EXPOSED", 0.4, box)], [("BUTTOCKS_EXPOSED", 0.3, (0.22, 0.2, 0.52, 0.5))], []]
    assert consensus(lone, 2) == []
    kept = consensus(agreed, 2)
    assert len(kept) == 1 and abs(kept[0].score - 0.3) < 1e-6
