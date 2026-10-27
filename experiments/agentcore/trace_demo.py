"""End-to-end trace: OpenCV (COOL on Graviton) output changes a later agent decision.

Real path, no mocks: Mac → L2 ``/v1/second_look`` (COOL, TLS, token) →
``stubborn_unsure`` event → AgentCore Runtime → Gateway ``cool_second_look``
(Lambda in the VPC → the box ``/v1/review`` deep look on the RAM-held crop)
→ ``verdict_put`` (DynamoDB) → later ``heavy_page`` events whose policy
decision depends on that review history. Then a ``session_summary`` through
the ``sg-kids`` Guardrail.

Pictures are benign procedural textures (``vision_core.live.synth``) chosen
because NudeNet 320n gives them weak scores; any CONFIRM here is therefore a
detector false positive kept covered fail-closed, which is the behaviour the
trace documents (not a claim about real content).

    .venv/bin/python experiments/agentcore/trace_demo.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import uuid
from pathlib import Path

VISION = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VISION))

import numpy as np  # noqa: E402

from agent.agentcore_client import AgentClient  # noqa: E402
from vision_core.live.cache import hash_crop  # noqa: E402
from vision_core.live.cloud_lane import CloudLane, CloudLaneConfig  # noqa: E402
from vision_core.live.synth import photo_texture  # noqa: E402

# Seeds found by a search on this Mac (L2 UNSURE; deep review SAFE or UNSAFE).
PICTURES = {
    "photos.example.org": [(20174, 480), (20327, 640), (1002, 640), (20158, 800), (1003, 480)],
    "news.example.com": [(20154, 640), (20199, 800), (20284, 800), (20310, 800), (20324, 640)],
}


def texture(seed: int, L: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rng.choice([480, 640, 800])  # keep the generator in step with the search
    return photo_texture(int(L * rng.uniform(0.6, 1.0)), L, rng)


def strip(out: dict) -> dict:
    return {k: out.get(k) for k in ("type", "app", "dhash", "verdict", "policy", "summary", "brain", "model", "lane", "latency_ms", "note", "guardrail_action")}


def main() -> None:
    run = uuid.uuid4().hex[:8]
    home = Path(tempfile.mkdtemp(prefix=f"sg-demo-{run}-"))  # fresh device id + local caches for this run
    (home / "cloud.json").write_text((Path.home() / ".screen-guardian" / "cloud.json").read_text())
    agent = AgentClient(home=home)
    lane = CloudLane(CloudLaneConfig.from_env())
    steps: list[dict] = []
    t0 = time.time()

    def heavy(app: str, label: str) -> dict:
        ev = {"type": "heavy_page", "app": app, "mode_floor": "aggressive", "pictures_per_min": 90,
              "escalations_per_min": 8, "unsure_rate": 0.12, "shed_rate": 0.25, "device_id": agent.device_id}
        out = agent.handle_sync(ev)
        steps.append({"step": label, "event": ev, "decision": strip(out), "agent_trace": out.get("trace")})
        print(label, json.dumps(strip(out)), flush=True)
        return out

    before = {app: heavy(app, f"heavy_page BEFORE reviews: {app}") for app in PICTURES}

    for app, pics in PICTURES.items():
        for seed, L in pics:
            img = texture(seed, L)
            dh, ah = hash_crop(img)
            body = {"jpeg_b64": lane._encode(img), "dhash": f"{dh:016x}", "ahash": f"{ah:016x}", "mode": "aggressive",
                    "profile": "l2", "picture_id": f"demo-{seed}", "reason": "unsure_after_local"}
            l2 = lane._http_post(body, 2.0)
            rec = {"step": f"L2 second look ({app}, seed {seed})", "l2": {k: l2[k] for k in ("verdict", "n_views", "timings_ms", "opencv", "review_hold_s", "source")}}
            if l2["verdict"]["state"] == "UNSURE":
                ev = {"type": "stubborn_unsure", "app": app, "mode": "aggressive", "mode_floor": "aggressive",
                      "dhash": f"{dh:016x}", "tiers": [], "l1_score": l2["verdict"]["score"], "l2_score": l2["verdict"]["score"],
                      "l2_state": "UNSURE", "device_id": agent.device_id}
                from agent.agentcore_client import tiers
                ev["tiers"] = tiers(l2["verdict"]["labels"])
                out = agent.handle_sync(ev)
                rec.update(event=ev, decision=strip(out), agent_trace=out.get("trace"))
            steps.append(rec)
            print(rec["step"], json.dumps(rec["l2"]["verdict"]), json.dumps(rec.get("decision")), flush=True)

    after = {app: heavy(app, f"heavy_page AFTER reviews: {app}") for app in PICTURES}

    counts = {"UNSURE_PICTURE": 7, "PENDING_PREFROST": 40, "UNSAFE_PICTURE": 2,
              "AGENT_CONFIRM": sum(1 for s in steps if (s.get("decision") or {}).get("verdict") == "CONFIRM"),
              "AGENT_CLEAR": sum(1 for s in steps if (s.get("decision") or {}).get("verdict") == "CLEAR")}
    summ = agent.handle_sync({"type": "session_summary", "counts": counts, "minutes": 25, "mode_floor": "aggressive",
                              "device_id": agent.device_id})
    steps.append({"step": "session_summary", "decision": strip(summ), "agent_trace": summ.get("trace")})
    print("summary", json.dumps(strip(summ)), flush=True)

    result = {
        "run": run, "device_id": agent.device_id, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "runtime_arn": agent.runtime_arn,
        "claim": "The COOL deep-review output (OpenCV on Graviton) decides CONFIRM/CLEAR; those verdicts are the history that changes the later heavy_page policy for the same app.",
        "policy_before": {a: before[a].get("policy") for a in PICTURES},
        "policy_after": {a: after[a].get("policy") for a in PICTURES},
        "verdicts": [{"app": s["event"]["app"], "l2": s["l2"]["verdict"]["state"], "agent": s["decision"]["verdict"],
                      "deep_review": next((t["result"].get("verdict") for t in (s.get("agent_trace") or []) if t.get("tool") == "cool_second_look"), None)}
                     for s in steps if s.get("event", {}).get("type") == "stubborn_unsure"],
        "summary": summ.get("summary"),
        "steps": steps,
    }
    path = VISION / "experiments" / "agentcore" / f"trace_{result['time_utc'].replace(':', '')}.json"
    path.write_text(json.dumps(result, indent=1, default=str))
    print(path)
    print(json.dumps({k: result[k] for k in ("policy_before", "policy_after", "verdicts", "summary")}, indent=1))
    lane.close()


if __name__ == "__main__":
    main()
