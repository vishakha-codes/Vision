"""L3 steward: safety invariants hold whatever the model asks for; rules fallback."""

from __future__ import annotations

import pytest

from agent.runtime.steward import Steward, local_summary, sanitize_event


class FakeStore:
    def __init__(self, review_state: str | None = "SAFE", confirmed: int = 0, cleared: int = 0) -> None:
        self.review_state = review_state
        self.policy = {"confirmed": confirmed, "cleared": cleared, "prefrost": False, "min_mode": "aggressive"}
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, name: str, args: dict) -> dict:
        self.calls.append((name, args))
        if name == "cool_second_look":
            if self.review_state is None:
                return {"found": False}
            return {"found": True, "verdict": {"state": self.review_state, "score": 0.3, "tiers": []}}
        if name == "policy_get":
            return {"found": True, "policy": dict(self.policy)}
        if name == "verdict_put":
            if args["verdict"] == "CONFIRM":
                self.policy["confirmed"] += 1
            return {"ok": True}
        return {"ok": True, **args}

    def names(self) -> list[str]:
        return [n for n, _ in self.calls]


def fake_model(script: list[list[tuple[str, dict]] | str]):
    """Converse stub: each turn either tool uses or final text."""
    turns = iter(script)

    def converse(**req):
        step = next(turns)
        if isinstance(step, str):
            return {"stopReason": "end_turn", "output": {"message": {"role": "assistant", "content": [{"text": step}]}}}
        content = [{"toolUse": {"toolUseId": f"t{i}", "name": n, "input": a}} for i, (n, a) in enumerate(step)]
        return {"stopReason": "tool_use", "output": {"message": {"role": "assistant", "content": content}}}

    return converse


EV = {"type": "stubborn_unsure", "device_id": "dev-1", "app": "Example.COM", "mode_floor": "aggressive",
      "dhash": "00ff00ff00ff00ff", "tiers": ["suggestive"], "l1_score": 0.2, "l2_score": 0.18}


def test_sanitize_drops_urls_titles_and_unknown_fields() -> None:
    ev = sanitize_event({**EV, "app": "https://example.com/path?q=1", "title": "secret page", "url": "x"})
    assert ev["app"] == "unknown.app" and "title" not in ev and "url" not in ev
    with pytest.raises(ValueError):
        sanitize_event({"type": "frame"})


def test_model_cannot_clear_an_unsure_review() -> None:
    store = FakeStore(review_state="UNSURE")
    model = fake_model([[("cool_second_look", {})], [("verdict_put", {"verdict": "CLEAR"})], '{"verdict":"CLEAR"}'])
    out = Steward(store, converse=model, models=("m",)).handle(EV)
    rejected = [s for s in out["trace"] if s.get("tool") == "verdict_put" and s["result"].get("error")]
    assert rejected, "CLEAR on an UNSURE review must be rejected"
    assert out["verdict"] == "KEEP_UNSURE"  # committed fail-closed by the steward


def test_unsafe_review_forces_confirm_and_tightens_policy() -> None:
    store = FakeStore(review_state="UNSAFE")
    out = Steward(store, converse=None).handle(EV)  # rules brain
    assert out["verdict"] == "CONFIRM" and out["brain"] == "rules"
    assert out["policy"]["prefrost"] is True and out["policy"]["escalation_rate_per_s"] >= 4.0


def test_policy_never_below_pin_mode_or_off() -> None:
    store = FakeStore()
    ev = {"type": "heavy_page", "device_id": "dev-1", "app": "com.apple.safari", "mode_floor": "aggressive"}
    model = fake_model([[("policy_put", {"min_mode": "off", "escalation_rate_per_s": 99, "prefrost": False})], "{}"])
    out = Steward(store, converse=model, models=("m",)).handle(ev)
    assert out["policy"]["min_mode"] == "aggressive"
    assert out["policy"]["escalation_rate_per_s"] == 6.0


def test_history_from_cool_reviews_changes_heavy_page_budget() -> None:
    ev = {"type": "heavy_page", "device_id": "dev-1", "app": "example.com", "mode_floor": "aggressive"}
    low = Steward(FakeStore(confirmed=0, cleared=4), converse=None).handle(ev)
    hit = Steward(FakeStore(confirmed=1, cleared=4), converse=None).handle(ev)
    new = Steward(FakeStore(confirmed=0, cleared=0), converse=None).handle(ev)
    assert low["policy"]["escalation_rate_per_s"] == 1.0
    assert hit["policy"]["escalation_rate_per_s"] >= 4.0 and hit["policy"]["prefrost"] is True
    # A model asking for a low budget without clean history is lifted back to the default.
    model = fake_model([[("policy_get", {})], [("policy_put", {"min_mode": "aggressive", "escalation_rate_per_s": 1, "prefrost": False})], "{}"])
    lifted = Steward(FakeStore(), converse=model, models=("m",)).handle(ev)
    assert new["policy"]["escalation_rate_per_s"] == 3.0 and lifted["policy"]["escalation_rate_per_s"] == 3.0


def test_bedrock_failure_falls_back_to_rules() -> None:
    def broken(**_):
        raise RuntimeError("AccessDenied")

    out = Steward(FakeStore(review_state="SAFE"), converse=broken, models=("a", "b")).handle(EV)
    assert out["brain"] == "rules" and out["verdict"] == "CLEAR"
    assert "audit_append" in [s.get("tool") for s in out["trace"]]


def test_session_summary_counts_only() -> None:
    store = FakeStore()
    ev = {"type": "session_summary", "device_id": "dev-1", "counts": {"UNSAFE_PICTURE": 3, "PENDING_PREFROST": 9}, "minutes": 30}
    out = Steward(store, converse=None).handle(ev)
    assert out["brain"] == "rules" and "kept 3 pictures hidden" in out["summary"]
    assert local_summary({}, 0).startswith("This session")


def test_model_cannot_confirm_a_safe_review_or_retune_policy_on_clear() -> None:
    store = FakeStore(review_state="SAFE")
    model = fake_model([[("cool_second_look", {})], [("verdict_put", {"verdict": "CONFIRM"})],
                        [("verdict_put", {"verdict": "CLEAR"}), ("policy_put", {"min_mode": "aggressive", "escalation_rate_per_s": 6, "prefrost": True})],
                        '{"verdict":"CLEAR"}'])
    out = Steward(store, converse=model, models=("m",)).handle(EV)
    errs = [s["result"]["error"] for s in out["trace"] if s.get("step") == "tool" and s["result"].get("error")]
    assert out["verdict"] == "CLEAR" and out["policy"] is None
    assert any("SAFE" in e for e in errs) and any("only a CONFIRM" in e for e in errs)
