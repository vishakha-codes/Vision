"""Mac event sender: AgentCore → local Bedrock (Guardrail) → local rules; PIN mode wins."""

from __future__ import annotations

import io
import json
import time

from agent.agentcore_client import AgentClient, tiers


class FakeAgentCore:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.payloads: list[dict] = []

    def invoke_agent_runtime(self, **kw):
        self.payloads.append(json.loads(kw["payload"]))
        if self.fail:
            raise RuntimeError("AccessDenied")
        body = {"type": "heavy_page", "verdict": "NONE", "policy": {"min_mode": "aggressive", "escalation_rate_per_s": 1.0, "prefrost": False}, "brain": "bedrock"}
        return {"response": io.BytesIO(json.dumps(body).encode())}


def tools(name, args):
    if name == "cool_second_look":
        return {"found": True, "verdict": {"state": "SAFE", "score": 0.0}}
    if name == "policy_get":
        return {"found": False, "policy": {"confirmed": 0, "cleared": 0}}
    return {"ok": True}


def test_send_is_non_blocking_and_events_carry_no_pixels_or_urls(tmp_path) -> None:
    ac = FakeAgentCore()
    got = []
    c = AgentClient(runtime_arn="arn:x", home=tmp_path, agentcore=ac, on_result=lambda out, ctx: got.append((out, ctx)))
    t0 = time.perf_counter()
    assert c.stubborn_unsure(app="example.com", mode="aggressive", mode_floor="aggressive", dhash=0xAB,
                             labels=("BUTTOCKS_EXPOSED",), l1_score=0.2, l2_score=0.18, l2_state="UNSURE",
                             context={"wh": (300, 200)})
    assert time.perf_counter() - t0 < 0.05
    end = time.time() + 3
    while not got and time.time() < end:
        time.sleep(0.01)
    ev = ac.payloads[0]
    assert ev["tiers"] == ["explicit"] and "labels" not in ev and ev["dhash"] == "00000000000000ab"
    assert got[0][1] == {"wh": (300, 200)}  # context stays on the Mac


def test_fallback_to_local_rules_and_policy_cache(tmp_path) -> None:
    c = AgentClient(runtime_arn="arn:x", home=tmp_path, agentcore=FakeAgentCore(fail=True),
                    bedrock_converse=False, local_tools=tools)
    out = c.handle_sync({"type": "stubborn_unsure", "device_id": c.device_id, "app": "example.com",
                         "mode_floor": "aggressive", "dhash": "ab"})
    assert out["lane"] == "rules_local" and out["verdict"] == "CLEAR" and "AccessDenied" in out["agentcore_error"]
    out = c.handle_sync({"type": "session_summary", "device_id": c.device_id, "counts": {"UNSAFE_PICTURE": 2}})
    assert out["lane"] == "rules_local" and (tmp_path / "pending_events.jsonl").is_file()


def test_effective_mode_never_below_pin(tmp_path) -> None:
    c = AgentClient(home=tmp_path, bedrock_converse=False)
    (tmp_path / "agent_policy.json").write_text(json.dumps({"a.com": {"min_mode": "medium"}, "b.com": {"min_mode": "aggressive"}}))
    assert c.effective_mode("aggressive", "a.com") == "aggressive"
    assert c.effective_mode("medium", "b.com") == "aggressive"
    assert c.effective_mode("medium", "unknown.com") == "medium"
    assert tiers(["FACE_FEMALE", "BELLY_EXPOSED"]) == ["mild"]
