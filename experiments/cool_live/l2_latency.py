"""Measure L2 second-look latency from this Mac to the Graviton box.

Uses the real client (``vision_core.live.cloud_lane``: pinned TLS cert,
bearer token from Keychain/env, 500 ms deadline) on benign synthetic
pictures. Each picture has a unique hash (no server cache), then a second
pass re-sends the same hashes (server cache hits).

    SG_L2_URL=https://<box-ip>:8443 .venv/bin/python experiments/cool_live/l2_latency.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

VISION = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VISION))

import numpy as np  # noqa: E402

from vision_core.live.cache import hash_crop  # noqa: E402
from vision_core.live.cloud_lane import CloudLane, CloudLaneConfig  # noqa: E402
from vision_core.live.synth import photo_texture  # noqa: E402


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 1) if xs else 0.0


def main(n: int = 80) -> None:
    cfg = CloudLaneConfig.from_env()
    cfg.timeout_s = 2.0  # measure the tail instead of cutting it
    lane = CloudLane(cfg)
    rng = np.random.default_rng(11)
    rows, cached = [], []
    hashes = []
    for i in range(n):
        long_side = int(rng.choice([240, 400, 640, 900, 1200]))
        w, h = long_side, int(long_side * rng.uniform(0.55, 1.0))
        img = photo_texture(h, w, rng)
        dh, ah = hash_crop(img)
        dh ^= i  # unique per request: no server cache
        body = {"jpeg_b64": lane._encode(img), "dhash": f"{dh:016x}", "ahash": f"{ah:016x}",
                "mode": "aggressive", "profile": "l2", "picture_id": f"bench{i}", "reason": "bench"}
        t0 = time.perf_counter()
        out = lane._http_post(body, cfg.timeout_s)
        rtt = (time.perf_counter() - t0) * 1000.0
        rows.append({"long": long_side, "rtt_ms": rtt, "server_ms": out["timings_ms"]["total"],
                     "infer_ms": out["timings_ms"]["infer"], "views": out["n_views"],
                     "bytes": out["payload_bytes"], "opencv": out["opencv"]["version"]})
        hashes.append(body)
    for body in hashes[:30]:
        t0 = time.perf_counter()
        lane._http_post(body, cfg.timeout_s)
        cached.append((time.perf_counter() - t0) * 1000.0)
    rtts = [r["rtt_ms"] for r in rows]
    net = [r["rtt_ms"] - r["server_ms"] for r in rows]
    out = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": "https://<box>:8443/v1/second_look (pinned self-signed TLS, bearer token)",
        "client": "vision_core.live.cloud_lane (httpx keep-alive)",
        "n": n,
        "opencv_server": rows[0]["opencv"],
        "rtt_ms": {"p50": pct(rtts, 50), "p95": pct(rtts, 95), "max": pct(rtts, 100)},
        "server_ms": {"p50": pct([r["server_ms"] for r in rows], 50), "p95": pct([r["server_ms"] for r in rows], 95)},
        "network_plus_tls_ms": {"p50": pct(net, 50), "p95": pct(net, 95)},
        "within_500ms": sum(1 for x in rtts if x <= 500.0) / len(rtts),
        "server_cache_hit_rtt_ms": {"p50": pct(cached, 50), "p95": pct(cached, 95)},
        "by_long_side": {str(L): {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95)}
                         for L in sorted({r["long"] for r in rows})
                         for v in [[r["rtt_ms"] for r in rows if r["long"] == L]]},
        "mean_jpeg_bytes": int(np.mean([r["bytes"] for r in rows])),
        "mean_views": float(np.mean([r["views"] for r in rows])),
    }
    path = VISION / "experiments" / "cool_live" / "l2_latency_mac.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    lane.close()


if __name__ == "__main__":
    main()
