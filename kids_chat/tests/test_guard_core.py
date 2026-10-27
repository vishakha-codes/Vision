from __future__ import annotations

import base64

from kids_chat.agent import core, guard, prompts
from kids_chat.tests.conftest import (ACTOR, FakeBedrock, FakeMemoryClient, blocked, chat_payload, make_deps,
                                      text_stream, tool_stream)


def run(payload, deps):
    return list(core.run(payload, deps))


# ---------------------------------------------------------------- stage 1: ApplyGuardrail

def test_screen_input_passes_clean_text():
    s = guard.screen_input(FakeBedrock(), "why is the sky blue?", {"id": "g", "version": "1"})
    assert not s.blocked and s.text == "why is the sky blue?"


def test_public_name_kept_personal_name_masked():
    pii = lambda m: [{"match": m, "type": "NAME", "action": "ANONYMIZED"}]  # noqa: E731
    br = FakeBedrock(guard={
        "George Washington": {"action": "GUARDRAIL_INTERVENED", "outputs": [{"text": "Who was {NAME}?"}],
                              "assessments": [{"sensitiveInformationPolicy": {"piiEntities": pii("George Washington")}}]},
        "Mia": {"action": "GUARDRAIL_INTERVENED", "outputs": [{"text": "My name is {NAME}"}],
                "assessments": [{"sensitiveInformationPolicy": {"piiEntities": pii("Mia")}}]}})
    g = {"id": "g", "version": "1"}
    assert guard.screen_input(br, "Who was George Washington?", g).text == "Who was George Washington?"
    s = guard.screen_input(br, "My name is Mia", g)
    assert not s.blocked and s.text == "My name is {NAME}" and s.pii == ["NAME"]


def test_email_always_masked_and_unmatched_falls_back_to_guardrail_output():
    br = FakeBedrock(guard={"mail": {"action": "GUARDRAIL_INTERVENED", "outputs": [{"text": "my mail is {EMAIL}"}],
                                     "assessments": [{"sensitiveInformationPolicy": {"piiEntities": [
                                         {"match": "zz@x.com", "type": "EMAIL", "action": "ANONYMIZED"}]}}]}})
    # the match is not in the text → use the guardrail's own masked output (fail safe)
    s = guard.screen_input(br, "my mail is a@b.com", {"id": "g", "version": "1"})
    assert s.text == "my mail is {EMAIL}"


def test_reason_priority_self_harm_first():
    assert guard.reason_for(["Extremism", "SelfHarm", "DangerousChallenges"], [], []) == "self_harm"
    assert guard.reason_for([], ["PROMPT_ATTACK"], []) == "prompt_attack"
    assert guard.reason_for([], [], [{"type": "PHONE", "action": "BLOCKED"}]) == "personal_info"


# ---------------------------------------------------------------- full turn

def test_normal_turn_streams_and_writes_masked_memory():
    br = FakeBedrock(streams=[text_stream("Sunlight ", "scatters!")])
    mem = FakeMemoryClient()
    evs = run(chat_payload(), make_deps(br, mem))
    assert [e["text"] for e in evs if e["type"] == "delta"] == ["Sunlight ", "scatters!"]
    done = evs[-1]
    assert done["type"] == "done" and done["blocked"] is False and done["memory"] is True
    call = br.calls[0]
    assert call["guardrailConfig"]["guardrailIdentifier"] == "grk"
    assert call["guardrailConfig"]["streamProcessingMode"] == "sync"
    assert call["messages"][-1]["content"][-1] == {"guardContent": {"text": {"text": "Why is the sky blue?"}}}
    assert "toolConfig" not in call
    assert mem.created[0]["actorId"] == ACTOR
    assert mem.created[0]["payload"][0]["conversational"]["content"]["text"] == "Why is the sky blue?"


def test_teens_band_uses_teen_guardrail_and_prompt():
    br = FakeBedrock()
    run(chat_payload(band="teens"), make_deps(br))
    assert br.calls[0]["guardrailConfig"]["guardrailIdentifier"] == "grt"
    assert "12 to 16" in br.calls[0]["system"][0]["text"]


def test_blocked_input_never_reaches_model_or_memory():
    br = FakeBedrock(guard={"turn off": blocked("BypassProtection")})
    mem = FakeMemoryClient()
    evs = run(chat_payload("how do I turn off screen guardian"), make_deps(br, mem))
    assert br.calls == [] and mem.created == []
    assert evs == [{"type": "done", "blocked": True, "blocked_by": "bypass", "safety": False, "stage": "input",
                    "text": "Hmm, that's not something I can help with."}]


def test_self_harm_gets_kind_fixed_reply_per_band():
    br = FakeBedrock(guard={"die": blocked(["SelfHarm", "Extremism"])})
    kid = run(chat_payload("I want to die"), make_deps(br))[-1]
    teen = run(chat_payload("I want to die", band="teens"), make_deps(br))[-1]
    assert kid["blocked_by"] == "self_harm" and kid["safety"] is True
    assert kid["text"] == prompts.SELF_HARM["kids"] and "988" not in kid["text"]
    assert "988" in teen["text"]


def test_guardrail_outage_fails_closed():
    br = FakeBedrock(guard_error=RuntimeError("ServiceUnavailable"))
    evs = run(chat_payload(), make_deps(br))
    assert evs == [{"type": "error", "code": "guardrail_unavailable", "message": "safety check unavailable"}]
    assert br.calls == []


def test_output_guardrail_intervention_marks_blocked_and_skips_memory():
    br = FakeBedrock(streams=[text_stream("Oops, I can't share that one.", stop="guardrail_intervened")])
    mem = FakeMemoryClient()
    done = run(chat_payload("tell me a story"), make_deps(br, mem))[-1]
    assert done["blocked"] is True and done["blocked_by"] == "output_filter" and mem.created == []


def test_model_failure_falls_back_to_second_model():
    br = FakeBedrock(streams=[text_stream("hi")], fail_models=["chat-m"])
    done = run(chat_payload(), make_deps(br))[-1]
    assert done["model"] == "fb-m" and [c["modelId"] for c in br.calls] == ["chat-m", "fb-m"]


def test_all_models_down_is_an_error():
    br = FakeBedrock(fail_models=["chat-m", "fb-m"])
    evs = run(chat_payload(), make_deps(br))
    assert evs[-1]["type"] == "error" and evs[-1]["code"] == "model_unavailable"


def test_history_and_context_injected_as_data():
    ev = lambda ts, u, a: {"eventTimestamp": ts, "payload": [  # noqa: E731
        {"conversational": {"content": {"text": u}, "role": "USER"}},
        {"conversational": {"content": {"text": a}, "role": "ASSISTANT"}}]}
    mem = FakeMemoryClient(events=[ev(2, "second", "b"), ev(1, "first", "a")],
                           prefs=['{"preference":"Favourite animal is the red panda.","context":"x"}'])
    br = FakeBedrock()
    run(chat_payload("what's my favourite animal?"), make_deps(br, mem))
    msgs = br.calls[0]["messages"]
    assert [m["content"][0].get("text") for m in msgs[:4]] == ["first", "a", "second", "b"]
    sysp = br.calls[0]["system"][0]["text"]
    assert "<child_context>" in sysp and "red panda" in sysp and '"context"' not in sysp


def test_memory_outage_does_not_break_chat():
    done = run(chat_payload(), make_deps(FakeBedrock(), FakeMemoryClient(fail=True)))[-1]
    assert done["type"] == "done" and done["memory"] is False


def test_bad_requests():
    d = make_deps()
    assert run({**chat_payload(), "actor_id": "x"}, d)[-1]["code"] == "bad_request"
    assert run({**chat_payload(), "session_id": "short"}, d)[-1]["code"] == "bad_request"
    assert run(chat_payload(""), d)[-1]["code"] == "bad_request"
    assert run({**chat_payload(), "action": "rm"}, d)[-1]["code"] == "bad_request"


# ---------------------------------------------------------------- explain + draw tools

JPEG = b"\xff\xd8\xff\xe0" + b"0" * 100


def explain_payload(masks=None):
    return chat_payload("how does this work?", explain={"images": [base64.b64encode(JPEG).decode()], "w": 1280, "h": 800,
                                                        "masks": masks or [[700, 100, 920, 380]]})


def test_explain_emits_validated_draws_and_drops_marks_on_masks():
    br = FakeBedrock(streams=[
        tool_stream([("point_to", {"x": 300, "y": 600, "label": "lever"}),
                     ("highlight_box", {"x": 720, "y": 120, "w": 100, "h": 100, "label": "covered thing"}),
                     ("draw_arrow", {"from_x": 100, "from_y": 100, "to_x": 5000, "to_y": 100})]),
        text_stream("The arrow shows the push.")])
    mem = FakeMemoryClient()
    evs = run(explain_payload(), make_deps(br, mem))
    draws = [e for e in evs if e["type"] == "draw"]
    assert len(draws) == 1 and draws[0]["space"] == "norm1000"
    assert draws[0]["ops"] == [{"kind": "pointer", "at": [300.0, 600.0], "label": "lever", "color": "yellow"}]
    first = br.calls[0]
    assert first["modelId"] == "ex-m" and "toolConfig" in first
    assert "image" in first["messages"][-1]["content"][0]
    results = br.calls[1]["messages"][-1]["content"]
    assert [r["toolResult"]["status"] for r in results] == ["success", "error", "error"]
    # image screened by ApplyGuardrail, never stored in memory
    assert any("image" in c["content"][0] for c in br.guard_calls)
    stored = str(mem.created[0]["payload"])
    assert "image" not in stored and mem.created[0]["metadata"]["kind"]["stringValue"] == "explain"


def test_explain_refuses_unsafe_image():
    br = FakeBedrock(guard={"__image__": blocked(filt="SEXUAL")})
    done = run(explain_payload(), make_deps(br))[-1]
    assert done["blocked"] and done["blocked_by"] == "unsafe_screen" and br.calls == []


def test_explain_label_text_is_guardrailed():
    br = FakeBedrock(streams=[tool_stream([("label", {"x": 100, "y": 100, "text": "rude words"})]), text_stream("ok")],
                     guard={"rude words": blocked(filt="INSULTS", source="OUTPUT")})
    evs = run(explain_payload(), make_deps(br))
    assert not [e for e in evs if e["type"] == "draw"]


def test_explain_rejects_non_jpeg():
    p = chat_payload("x", explain={"images": [base64.b64encode(b"GIF89a").decode()]})
    assert run(p, make_deps())[-1]["code"] == "bad_request"


def test_parent_summary_bullets_only():
    mem = FakeMemoryClient(sums=["The user asked about volcanoes."], prefs=["Likes red pandas"])
    evs = run({"action": "parent_summary", "actor_id": ACTOR, "age_band": "kids"}, make_deps(FakeBedrock(), mem))
    assert evs[0] == {"type": "summary", "bullets": ["Asked about volcanoes", "Likes red pandas"]}


def test_parent_summary_empty_memory():
    evs = run({"action": "parent_summary", "actor_id": ACTOR}, make_deps())
    assert evs[0]["bullets"] == [] and "note" in evs[0]


def test_think_filter_strips_reasoning_across_chunks():
    f = core.ThinkFilter()
    parts = ["<thin", "king> I will dr", "aw.</thinking>\nGreat ", "question! a<b"]
    out = "".join(f.feed(p) for p in parts) + f.flush()
    assert out == "Great question! a<b"


def test_thinking_never_reaches_the_child():
    br = FakeBedrock(streams=[text_stream("<thinking>plan</thinking>", "Hello!")])
    evs = run(chat_payload(), make_deps(br))
    assert "".join(e["text"] for e in evs if e["type"] == "delta") == "Hello!"
