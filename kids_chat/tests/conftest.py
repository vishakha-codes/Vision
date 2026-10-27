from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

VISION_ROOT = Path(__file__).resolve().parents[2]
if str(VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(VISION_ROOT))

from kids_chat.agent import core  # noqa: E402
from kids_chat.agent.memory import ChatMemory  # noqa: E402

TOKEN = "ab" * 32


# ---------------------------------------------------------------- fake AWS clients

class FakeBedrock:
    """Scripted bedrock-runtime: ``guard`` maps text substrings to ApplyGuardrail responses,
    ``streams`` is a list of ConverseStream event lists consumed in order."""

    def __init__(self, streams=None, guard=None, guard_error=None, fail_models=()):
        self.streams = list(streams or [])
        self.guard = guard or {}
        self.guard_error = guard_error
        self.fail_models = set(fail_models)
        self.calls = []
        self.guard_calls = []

    def apply_guardrail(self, **kw):
        self.guard_calls.append(kw)
        if self.guard_error:
            raise self.guard_error
        block = kw["content"][0]
        if "image" in block:
            return self.guard.get("__image__", {"action": "NONE", "outputs": [], "assessments": []})
        text = block["text"]["text"]
        for key, resp in self.guard.items():
            if key != "__image__" and key in text and (resp.get("_source", kw["source"]) == kw["source"]):
                return resp
        return {"action": "NONE", "outputs": [], "assessments": []}

    def converse_stream(self, **kw):
        self.calls.append(kw)
        if kw["modelId"] in self.fail_models:
            raise RuntimeError("ThrottlingException")
        events = self.streams.pop(0) if self.streams else text_stream("ok")
        return {"stream": iter(events)}

    def converse(self, **kw):
        self.calls.append(kw)
        return {"stopReason": "end_turn", "output": {"message": {"content": [{"text": "- Asked about volcanoes\n- Likes red pandas"}]}}}


def text_stream(*chunks, stop="end_turn"):
    ev = [{"messageStart": {"role": "assistant"}}, {"contentBlockStart": {"start": {}, "contentBlockIndex": 0}}]
    ev += [{"contentBlockDelta": {"delta": {"text": c}, "contentBlockIndex": 0}} for c in chunks]
    ev += [{"contentBlockStop": {"contentBlockIndex": 0}}, {"messageStop": {"stopReason": stop}},
           {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}}]
    return ev


def tool_stream(calls, text=""):
    ev = [{"messageStart": {"role": "assistant"}}]
    i = 0
    if text:
        ev += [{"contentBlockStart": {"start": {}, "contentBlockIndex": 0}},
               {"contentBlockDelta": {"delta": {"text": text}, "contentBlockIndex": 0}}, {"contentBlockStop": {"contentBlockIndex": 0}}]
        i = 1
    for n, (name, args) in enumerate(calls):
        idx = i + n
        ev += [{"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t{idx}", "name": name}}, "contentBlockIndex": idx}},
               {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}, "contentBlockIndex": idx}},
               {"contentBlockStop": {"contentBlockIndex": idx}}]
    ev += [{"messageStop": {"stopReason": "tool_use"}}]
    return ev


def blocked(topic=None, filt=None, pii=None, msg="Hmm, that's not something I can help with.", source=None):
    a = {}
    if topic:
        a["topicPolicy"] = {"topics": [{"name": t, "type": "DENY", "action": "BLOCKED"} for t in ([topic] if isinstance(topic, str) else topic)]}
    if filt:
        a["contentPolicy"] = {"filters": [{"type": filt, "confidence": "HIGH", "action": "BLOCKED"}]}
    if pii:
        a["sensitiveInformationPolicy"] = {"piiEntities": pii}
    r = {"action": "GUARDRAIL_INTERVENED", "outputs": [{"text": msg}], "assessments": [a]}
    if source:
        r["_source"] = source
    return r


class FakeMemoryClient:
    def __init__(self, events=None, prefs=None, sums=None, fail=False):
        self.events = list(events or [])
        self.prefs = prefs or []
        self.sums = sums or []
        self.created = []
        self.fail = fail

    def list_events(self, **kw):
        if self.fail:
            raise RuntimeError("AccessDenied")
        return {"events": self.events}

    def create_event(self, **kw):
        if self.fail:
            raise RuntimeError("AccessDenied")
        self.created.append(kw)
        return {"event": {"eventId": "e1"}}

    def _recs(self, kw):
        ns = kw.get("namespace") or kw.get("namespacePath") or ""
        src = self.prefs if ns.startswith("/preferences/") else self.sums
        return {"memoryRecordSummaries": [{"content": {"text": t}} for t in src]}

    def retrieve_memory_records(self, **kw):
        if self.fail:
            raise RuntimeError("AccessDenied")
        return self._recs(kw)

    def list_memory_records(self, **kw):
        return self._recs(kw)


GR = {"kids": {"id": "grk", "version": "3"}, "teens": {"id": "grt", "version": "3"}}
ACTOR = "0f8c3e52-7a1b-4c7e-9d2a-1234567890ab"
SESSION = f"{ACTOR}-20261026-1"


def make_deps(bedrock=None, mem=None):
    return core.Deps(bedrock=bedrock or FakeBedrock(), memory=ChatMemory(mem if mem is not None else FakeMemoryClient(), "mem-1"),
                     guardrails=GR, chat_model="chat-m", fallback_model="fb-m", explain_model="ex-m")


def chat_payload(text="Why is the sky blue?", band="kids", **kw):
    return {"action": "chat", "prompt": text, "actor_id": ACTOR, "session_id": SESSION, "age_band": band, **kw}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SG_CHAT_HOME", str(tmp_path))
    monkeypatch.delenv("SG_GUARDIAN_JSON", raising=False)
    (tmp_path / "guardian.json").write_text(json.dumps({"token_sha256": hashlib.sha256(TOKEN.encode()).hexdigest(),
                                                        "mode": "aggressive"}))
    return tmp_path
