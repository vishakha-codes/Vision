"""Old live path fixes: reinspect is hold-free, reinspect budget is a token bucket."""

from __future__ import annotations

import numpy as np

from agent.orchestrator import Orchestrator
from vision_core import detect_nsfw as dn
from vision_core.detect_nsfw import make_nsfw_detect, tune_policy_for_live_nsfw
from vision_core.policy import load_policy
from vision_core.reinspect_nsfw import make_nsfw_reinspect


def _frame(h: int = 240, w: int = 320) -> np.ndarray:
    return np.full((h, w, 3), 90, np.uint8)


def test_reinspect_does_not_touch_cluster_hold_or_last_stats() -> None:
    calls = {"n": 0}

    def box_fn(img):
        calls["n"] += 1
        # Full frame: one figure at (150..250); crops: a box near the origin.
        if img.shape[1] >= 300:
            return [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.9, "box": [150, 100, 100, 80]}]
        return [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.8, "box": [5, 5, 30, 30]}]

    detect = make_nsfw_detect("balanced", box_fn=box_fn)
    full = detect(_frame())
    assert full
    held_before = [h.xyxy for h in detect.hold._held]
    stats_before = dn.last_detect_stats()
    reinspect = make_nsfw_reinspect(mode="local", detect_fn=detect)
    looked = reinspect(_frame(), (150.0, 100.0, 250.0, 180.0), margin=10)
    assert looked.score > 0.0
    # Box is remapped into frame coords, the hold still has the full-frame box.
    assert looked.xyxy[0] >= 140.0
    assert [h.xyxy for h in detect.hold._held] == held_before
    assert dn.last_detect_stats() is stats_before


def test_replay_policy_keeps_fixed_budget() -> None:
    orch = Orchestrator(load_policy("policies/balanced.json"))
    assert orch.bucket is None and orch.budget_left == 40


def test_live_policy_uses_refilling_bucket() -> None:
    now = {"t": 0.0}
    cfg = tune_policy_for_live_nsfw(load_policy("policies/balanced.json"))
    orch = Orchestrator(cfg, clock=lambda: now["t"])
    assert orch.bucket is not None
    burst = orch.budget_left
    for i in range(burst):
        assert orch.can_reinspect(i * 100)
        orch._spend_reinspect()
    assert not orch.can_reinspect(10_000)
    now["t"] = 60.0  # a minute later the live session can reinspect again
    assert orch.can_reinspect(20_000)
    relaxed = Orchestrator(tune_policy_for_live_nsfw(load_policy("policies/relaxed.json")))
    assert relaxed.bucket is not None
