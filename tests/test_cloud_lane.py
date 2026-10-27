"""L2 cloud lane against a fake HTTP server: async submit, deadline, breaker, cache merge."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from vision_core.live.cache import VerdictCache
from vision_core.live.cloud_lane import (
    AGENT_CLEAR,
    AGENT_CONFIRM,
    CloudLane,
    CloudLaneConfig,
    merge_evidence,
)
from vision_core.live.contracts import PictureRegion, ProtectionMode, VerdictState
from vision_core.live.modes import Evidence, NormDet, decide, load_mode_policy


class _Fake:
    def __init__(self) -> None:
        self.delay_s = 0.0
        self.status = 200
        self.dets: list[dict] = []
        self.state = "SAFE"
        self.requests: list[dict] = []
        self.auth: list[str] = []


def _server(fake: _Fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            fake.requests.append(body)
            fake.auth.append(self.headers.get("authorization", ""))
            time.sleep(fake.delay_s)
            payload = json.dumps({
                "source": "cloud", "verdict": {"state": fake.state, "score": 0.0, "labels": []},
                "dets": fake.dets, "n_views": 10, "timings_ms": {"total": 5.0},
                "opencv": {"version": "4.14.0-pre", "cool": True}, "image_persisted": False,
            }).encode()
            self.send_response(fake.status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture()
def fake():
    f = _Fake()
    srv = _server(f)
    f.url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield f
    srv.shutdown()


def _lane(fake, **kw) -> CloudLane:
    cfg = CloudLaneConfig(url=fake.url, token="tok", timeout_s=kw.pop("timeout_s", 0.5), **kw)
    return CloudLane(cfg)


def _crop(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (200, 300, 3), dtype=np.uint8)


def _wait(lane: CloudLane, n: int, t: float = 3.0) -> None:
    end = time.time() + t
    while time.time() < end and lane.counts["ok"] + lane.counts["failed"] + lane.counts["timeouts"] < n:
        time.sleep(0.01)


def test_submit_is_async_and_sends_crop_with_token(fake) -> None:
    fake.delay_s = 0.2
    lane = _lane(fake)
    t0 = time.perf_counter()
    assert lane.submit(region_id="r1", dhash=0xAB, ahash=0xCD, wh=(300, 200), crop_bgr=_crop(), mode="aggressive")
    assert (time.perf_counter() - t0) < 0.05  # never blocks the vision thread
    _wait(lane, 1)
    body = fake.requests[0]
    assert fake.auth[0] == "Bearer tok"
    assert body["dhash"] == "00000000000000ab" and body["mode"] == "aggressive"
    assert set(body) == {"jpeg_b64", "dhash", "ahash", "mode", "profile", "picture_id", "reason"}
    assert lane.drain()[0].state == "SAFE"
    lane.close()


def test_disabled_without_url_or_token() -> None:
    lane = CloudLane(CloudLaneConfig(url="", token=""))
    assert lane.state == "disabled"
    assert not lane.submit(region_id="r", dhash=1, ahash=1, wh=(10, 10), crop_bgr=_crop(), mode="aggressive")


def test_duplicates_and_rate_are_shed(fake) -> None:
    lane = _lane(fake, rate_per_s=0.0, burst=2.0)
    assert lane.submit(region_id="a", dhash=1, ahash=1, wh=(300, 200), crop_bgr=_crop(), mode="aggressive")
    assert not lane.submit(region_id="a", dhash=1, ahash=1, wh=(300, 200), crop_bgr=_crop(), mode="aggressive")
    assert lane.submit(region_id="b", dhash=2, ahash=2, wh=(300, 200), crop_bgr=_crop(1), mode="aggressive")
    assert not lane.submit(region_id="c", dhash=3, ahash=3, wh=(300, 200), crop_bgr=_crop(2), mode="aggressive")
    assert lane.counts["shed_dup"] == 1 and lane.counts["shed_rate"] == 1
    lane.close()


def test_timeout_opens_breaker_then_probe_recovers(fake) -> None:
    fake.delay_s = 0.3
    lane = _lane(fake, timeout_s=0.1, breaker_failures=2, breaker_open_s=0.3, rate_per_s=100, burst=100)
    for i in range(2):
        lane.submit(region_id=f"r{i}", dhash=10 + i, ahash=i, wh=(300, 200), crop_bgr=_crop(i), mode="aggressive")
        _wait(lane, i + 1)
    assert lane.counts["timeouts"] == 2
    assert lane.state == "open"
    assert not lane.submit(region_id="x", dhash=99, ahash=0, wh=(300, 200), crop_bgr=_crop(), mode="aggressive")
    assert lane.counts["shed_breaker"] == 1
    time.sleep(0.35)
    fake.delay_s = 0.0
    assert lane.state == "half_open"
    assert lane.submit(region_id="p", dhash=100, ahash=0, wh=(300, 200), crop_bgr=_crop(), mode="aggressive")
    _wait(lane, 3)
    assert lane.state == "closed"
    lane.close()


def test_server_error_counts_as_failure(fake) -> None:
    fake.status = 500
    lane = _lane(fake)
    lane.submit(region_id="r", dhash=5, ahash=5, wh=(300, 200), crop_bgr=_crop(), mode="aggressive")
    _wait(lane, 1)
    assert lane.counts["failed"] == 1 and lane.drain() == []
    lane.close()


def _unsure_evidence() -> Evidence:
    weak = NormDet("BUTTOCKS_EXPOSED", 0.2, (0.1, 0.1, 0.5, 0.5))
    return Evidence((weak,), (weak,), 10.0, 3)


def test_clean_cloud_look_clears_local_unsure_in_cache(fake) -> None:
    pol = load_mode_policy(ProtectionMode.AGGRESSIVE)
    reg = PictureRegion("r", (0, 0, 300, 200))
    cache = VerdictCache()
    cache.put(7, 8, (300, 200), _unsure_evidence(), 1000.0)
    assert decide(pol, reg, cache.get(7, 8, (300, 200), 1000.0)).state is VerdictState.UNSURE
    lane = _lane(fake)
    lane.submit(region_id="r", dhash=7, ahash=8, wh=(300, 200), crop_bgr=_crop(), mode="aggressive")
    _wait(lane, 1)
    applied = lane.apply(cache, 1100.0, decide_fn=lambda ev: decide(pol, reg, ev).state)
    assert len(applied) == 1
    assert decide(pol, reg, cache.get(7, 8, (300, 200), 1100.0)).state is VerdictState.SAFE
    assert lane.counts["flips"] == 1
    lane.close()


def test_cloud_hit_makes_unsafe_and_local_hits_stay_fail_closed(fake) -> None:
    pol = load_mode_policy(ProtectionMode.AGGRESSIVE)
    reg = PictureRegion("r", (0, 0, 300, 200))
    fake.dets = [{"label": "BUTTOCKS_EXPOSED", "score": 0.6, "box": [0.1, 0.1, 0.4, 0.4]}]
    cache = VerdictCache()
    cache.put(1, 2, (300, 200), _unsure_evidence(), 0.0)
    lane = _lane(fake)
    lane.submit(region_id="r", dhash=1, ahash=2, wh=(300, 200), crop_bgr=_crop(), mode="aggressive")
    _wait(lane, 1)
    lane.apply(cache, 10.0)
    assert decide(pol, reg, cache.get(1, 2, (300, 200), 10.0)).state is VerdictState.UNSAFE
    # A strong local first-pass hit can never be cleared by the cloud or the agent.
    strong = Evidence((NormDet("BUTTOCKS_EXPOSED", 0.9, (0, 0, 1, 1)),), None, 1.0, 1)
    lane.inject_verdict(dhash=1, ahash=2, wh=(300, 200), verdict=AGENT_CLEAR)
    res = lane.drain()[0]
    assert decide(pol, reg, merge_evidence(strong, res)).state is VerdictState.UNSAFE
    lane.close()


def test_agent_confirm_and_clear() -> None:
    pol = load_mode_policy(ProtectionMode.AGGRESSIVE)
    reg = PictureRegion("r", (0, 0, 300, 200))
    lane = CloudLane(CloudLaneConfig(url="", token=""))
    lane.inject_verdict(dhash=1, ahash=1, wh=(300, 200), verdict=AGENT_CONFIRM, labels=("BUTTOCKS_EXPOSED",))
    lane.inject_verdict(dhash=2, ahash=2, wh=(300, 200), verdict=AGENT_CLEAR)
    conf, clear = lane.drain()
    assert decide(pol, reg, merge_evidence(_unsure_evidence(), conf)).state is VerdictState.UNSAFE
    assert decide(pol, reg, merge_evidence(_unsure_evidence(), clear)).state is VerdictState.SAFE


def test_config_from_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SG_L2_URL", "https://1.2.3.4:8443/")
    monkeypatch.setenv("SG_L2_TOKEN", "abc")
    monkeypatch.setenv("SG_L2_TIMEOUT_MS", "400")
    cfg = CloudLaneConfig.from_env(tmp_path / "missing.json")
    assert cfg.url == "https://1.2.3.4:8443" and cfg.token == "abc" and cfg.timeout_s == 0.4 and cfg.enabled
