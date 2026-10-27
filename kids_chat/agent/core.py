"""One kids-chat turn: guardrail pre-pass → memory → guarded model stream → memory write.

Shared by the AgentCore Runtime entry point (``main.py``) and the Mac
service's direct-Bedrock fallback, so both paths apply the same safety.
:func:`run` is a generator of small event dicts:

    {"type":"delta","text":"..."}
    {"type":"draw","ops":[...],"space":"norm1000"}            (explain mode)
    {"type":"done","blocked":false,"model":"...","memory":true}
    {"type":"done","blocked":true,"blocked_by":"self_harm","safety":true,"text":"<canned>"}
    {"type":"summary","bullets":[...]}                         (action parent_summary)
    {"type":"error","code":"bad_request|model_unavailable|guardrail_unavailable","message":"..."}

Fail closed: if the guardrail pre-pass cannot run, the turn is refused.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

try:  # flat layout inside the runtime zip
    import guard
    import prompts
    import screen_tools
    from memory import ChatMemory, valid_actor, valid_session
except ImportError:  # repo layout
    from kids_chat.agent import guard, prompts, screen_tools
    from kids_chat.agent.memory import ChatMemory, valid_actor, valid_session

MAX_PROMPT = 1000
MAX_IMAGES = 3
MAX_IMAGE_BYTES = 400_000
CHAT_TOKENS = 400
EXPLAIN_TOKENS = 700
MAX_TOOL_ROUNDS = 3


@dataclass
class Deps:
    bedrock: Any  # boto3 bedrock-runtime client
    memory: ChatMemory
    guardrails: dict  # {"kids": {"id","version"}, "teens": {...}}
    chat_model: str = "us.amazon.nova-2-lite-v1:0"
    fallback_model: str = "us.amazon.nova-lite-v1:0"
    explain_model: str = "us.amazon.nova-pro-v1:0"
    log: Callable[[str, dict], None] = field(default=lambda e, d: None)


def deps_from_env(bedrock: Any, agentcore: Any, log: Callable[[str, dict], None] | None = None) -> Deps:
    lg = log or (lambda e, d: None)
    return Deps(
        bedrock=bedrock,
        memory=ChatMemory(agentcore, os.environ.get("SG_MEMORY_ID", ""), log=lg),
        guardrails={"kids": {"id": os.environ.get("SG_GR_KIDS_ID", ""), "version": os.environ.get("SG_GR_KIDS_VER", "DRAFT")},
                    "teens": {"id": os.environ.get("SG_GR_TEENS_ID", ""), "version": os.environ.get("SG_GR_TEENS_VER", "DRAFT")}},
        chat_model=os.environ.get("SG_CHAT_MODEL", "us.amazon.nova-2-lite-v1:0"),
        fallback_model=os.environ.get("SG_FALLBACK_MODEL", "us.amazon.nova-lite-v1:0"),
        explain_model=os.environ.get("SG_EXPLAIN_MODEL", "us.amazon.nova-pro-v1:0"),
        log=lg,
    )


def _err(code: str, message: str) -> dict:
    return {"type": "error", "code": code, "message": message[:300]}


def run(payload: dict, deps: Deps) -> Iterator[dict]:
    action = payload.get("action") or "chat"
    if action == "ping":
        yield {"type": "pong", "t": time.time()}
        return
    actor = payload.get("actor_id")
    if not valid_actor(actor):
        yield _err("bad_request", "actor_id must be a random profile id (8-64 chars)")
        return
    band = payload.get("age_band") if payload.get("age_band") in prompts.BANDS else "kids"
    gr = deps.guardrails.get(band) or {}
    if not gr.get("id"):
        yield _err("guardrail_unavailable", "no guardrail configured")
        return
    if action == "parent_summary":
        yield from parent_summary(actor, band, deps)
        return
    if action != "chat":
        yield _err("bad_request", f"unknown action {action!r}")
        return
    session = payload.get("session_id")
    if not valid_session(session):
        yield _err("bad_request", "session_id must be 33-100 chars")
        return
    text = guard.clean(payload.get("prompt"), MAX_PROMPT)
    if not text:
        yield _err("bad_request", "empty prompt")
        return
    explain = payload.get("explain") if isinstance(payload.get("explain"), dict) else None
    images: list[bytes] = []
    if explain:
        try:
            images = [base64.b64decode(b, validate=True) for b in (explain.get("images") or [])][:MAX_IMAGES]
        except (ValueError, TypeError):
            yield _err("bad_request", "images must be base64 JPEG")
            return
        if not images or any(len(i) > MAX_IMAGE_BYTES or not i.startswith(b"\xff\xd8") for i in images):
            yield _err("bad_request", "explain needs 1-3 JPEG frames under 400 KB")
            return

    # Stage 1: ApplyGuardrail on the input (fail closed).
    t0 = time.time()
    try:
        s = guard.screen_input(deps.bedrock, text, gr)
        if not s.blocked and images:
            si = guard.screen_images(deps.bedrock, images, gr)
            if si.blocked:
                s = si
    except Exception as exc:
        deps.log("guardrail_failed", {"error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        yield _err("guardrail_unavailable", "safety check unavailable")
        return
    deps.log("prepass", {"blocked": s.blocked, "reason": s.reason, "pii": s.pii, "ms": int((time.time() - t0) * 1000)})
    if s.blocked:
        canned = prompts.SELF_HARM[band] if s.reason == "self_harm" else (
            "I can't look at that screen right now. Let's ask a grown-up!" if s.reason == "unsafe_screen" else s.canned)
        yield {"type": "done", "blocked": True, "blocked_by": s.reason, "safety": s.safety, "stage": "input", "text": canned}
        return

    # Memory (best effort): history + interests/summaries as data.
    history = deps.memory.recent_messages(actor, session)
    context = deps.memory.context(actor, s.text)
    system = prompts.system(band, name=str(payload.get("name") or prompts.DEFAULT_NAME)[:20], explain=bool(images), context=context)

    user_content: list[dict] = []
    if images:
        w, h = explain.get("w"), explain.get("h")
        user_content += [{"image": {"format": "jpeg", "source": {"bytes": b}}} for b in images]
        user_content.append({"text": f"[Screen picture{'s' if len(images) > 1 else ''}: {len(images)} frame(s), {w}x{h} px. "
                                     f"Covered regions: {len(explain.get('masks') or [])}.]"})
    # guardContent = only the latest child message is scored by the model-call guardrail
    # (history and memory context are not re-scored every turn).
    user_content.append({"guardContent": {"text": {"text": s.text}}})
    messages = history + [{"role": "user", "content": user_content}]

    result = yield from _stream(deps, messages, system, gr, images=bool(images),
                                masks=(explain or {}).get("masks") or [])
    if result.get("error"):
        yield _err("model_unavailable", result["error"])
        return
    if result["blocked"]:
        yield {"type": "done", "blocked": True, "blocked_by": "output_filter", "safety": False, "stage": "output",
               "text": result["text"] or "Oops, I can't share that one. A grown-up you trust can help!", "model": result["model"]}
        return
    answer = result["text"].strip()
    stored = deps.memory.write_turn(actor, session, s.text, answer, kind="explain" if images else "chat") if answer else False
    yield {"type": "done", "blocked": False, "model": result["model"], "memory": stored,
           "masked": s.text != text, "usage": result.get("usage", {})}


def _stream(deps: Deps, messages: list, system: str, gr: dict, *, images: bool, masks: list) -> Iterator[dict]:
    """Guarded ConverseStream (+ draw-tool loop in explain mode). Returns a summary dict."""
    models = [deps.explain_model] if images else [deps.chat_model]
    if deps.fallback_model and deps.fallback_model not in models:
        models.append(deps.fallback_model)
    last_err = ""
    for model in models:
        emitted = False
        try:
            out = {"text": "", "blocked": False, "model": model, "usage": {}}
            msgs = list(messages)
            for _round in range(MAX_TOOL_ROUNDS if images else 1):
                kw = dict(modelId=model, messages=msgs, system=[{"text": system}],
                          inferenceConfig={"maxTokens": EXPLAIN_TOKENS if images else CHAT_TOKENS, "temperature": 0.4},
                          guardrailConfig={"guardrailIdentifier": gr["id"], "guardrailVersion": str(gr["version"]),
                                           "trace": "enabled", "streamProcessingMode": "sync"})
                if images:
                    kw["toolConfig"] = screen_tools.tool_config()
                resp = deps.bedrock.converse_stream(**kw)
                text, tools, stop, usage = "", [], "", {}
                cur: dict | None = None
                think = ThinkFilter()
                for ev in resp["stream"]:
                    if "contentBlockStart" in ev:
                        tu = (ev["contentBlockStart"].get("start") or {}).get("toolUse")
                        cur = {"id": tu["toolUseId"], "name": tu["name"], "input": ""} if tu else None
                    elif "contentBlockDelta" in ev:
                        d = ev["contentBlockDelta"].get("delta") or {}
                        if "text" in d and d["text"]:
                            shown = think.feed(d["text"])
                            if shown:
                                text += shown
                                emitted = True
                                yield {"type": "delta", "text": shown}
                        elif "toolUse" in d and cur is not None:
                            cur["input"] += d["toolUse"].get("input", "")
                    elif "contentBlockStop" in ev:
                        if cur is not None:
                            tools.append(cur)
                        cur = None
                    elif "messageStop" in ev:
                        stop = ev["messageStop"].get("stopReason", "")
                        tail = think.flush()
                        if tail:
                            text += tail
                            emitted = True
                            yield {"type": "delta", "text": tail}
                    elif "metadata" in ev:
                        usage = ev["metadata"].get("usage") or {}
                out["text"] += text
                out["usage"] = usage
                if stop == "guardrail_intervened":
                    out["blocked"] = True
                    out["text"] = text
                    return out
                if not tools or stop != "tool_use":
                    return out
                # Draw tools: validate, drop marks on covered regions, guardrail the labels, emit.
                assistant_content = ([{"text": text}] if text else []) + [
                    {"toolUse": {"toolUseId": t["id"], "name": t["name"], "input": _json(t["input"])}} for t in tools]
                ops, results = [], []
                for t in tools:
                    try:
                        op = screen_tools.op_from_tool(t["name"], _json(t["input"]))
                        if screen_tools.touches_mask(op, masks):
                            raise screen_tools.DrawError("that spot is covered; never draw on covered regions")
                        if len(ops) >= screen_tools.MAX_OPS:
                            raise screen_tools.DrawError("too many drawings")
                        ops.append(op)
                        results.append((t["id"], "drawn", "success"))
                    except screen_tools.DrawError as exc:
                        results.append((t["id"], f"not drawn: {exc}", "error"))
                if ops and not guard.screen_output(deps.bedrock, screen_tools.labels_text(ops), gr):
                    ops = []
                    results = [(i, "not drawn: label not allowed", "error") for i, _, _ in results]
                if ops:
                    emitted = True
                    yield {"type": "draw", "ops": ops, "space": screen_tools.SPACE}
                msgs = msgs + [{"role": "assistant", "content": assistant_content},
                               {"role": "user", "content": [{"toolResult": {"toolUseId": i, "content": [{"text": r}], "status": st}}
                                                            for i, r, st in results]}]
            return out
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {str(exc)[:200]}"
            deps.log("model_failed", {"model": model, "error": last_err, "emitted": emitted})
            if emitted:
                return {"text": "", "blocked": False, "model": model, "error": "stream interrupted"}
    return {"text": "", "blocked": False, "model": "", "error": last_err or "no model"}


class ThinkFilter:
    """Drops ``<thinking>...</thinking>`` (Nova Pro's tool-use reasoning) from a text stream."""
    OPEN, CLOSE = "<thinking>", "</thinking>"

    def __init__(self) -> None:
        self.buf = ""
        self.inside = False
        self.started = False

    def _lead(self, out: str) -> str:
        if not self.started:
            out = out.lstrip()
            self.started = bool(out)
        return out

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        out = ""
        while True:
            tag = self.CLOSE if self.inside else self.OPEN
            i = self.buf.find(tag)
            if i >= 0:
                if not self.inside:
                    out += self.buf[:i]
                self.buf = self.buf[i + len(tag):]
                self.inside = not self.inside
                continue
            # keep a possible partial tag at the end
            keep = next((k for k in range(min(len(tag) - 1, len(self.buf)), 0, -1) if tag.startswith(self.buf[-k:])), 0)
            if not self.inside:
                out += self.buf[:len(self.buf) - keep]
            self.buf = self.buf[len(self.buf) - keep:]
            return self._lead(out)

    def flush(self) -> str:
        rest = "" if self.inside else self.buf
        self.buf = ""
        return self._lead(rest)


def _json(s: Any) -> dict:
    if isinstance(s, dict):
        return s
    try:
        v = json.loads(s or "{}")
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {}


def parent_summary(actor: str, band: str, deps: Deps) -> Iterator[dict]:
    """Up to 5 topic bullets from the session summaries. No quotes, no transcripts."""
    sums = deps.memory.summaries(actor, limit=10)
    interests = deps.memory.interests(actor, limit=5)
    if not sums and not interests:
        yield {"type": "summary", "bullets": [], "note": "No chats summarised yet (summaries appear a minute or so after a chat)."}
        yield {"type": "done", "blocked": False}
        return
    gr = deps.guardrails[band]
    src = "\n".join(f"- {x}" for x in (sums + interests))[:6000]
    try:
        r = deps.bedrock.converse(
            modelId=deps.chat_model, system=[{"text": prompts.SUMMARY}],
            messages=[{"role": "user", "content": [{"text": "Summaries:\n" + src}]}],
            inferenceConfig={"maxTokens": 200, "temperature": 0.2},
            guardrailConfig={"guardrailIdentifier": gr["id"], "guardrailVersion": str(gr["version"]), "trace": "disabled"})
    except Exception as exc:
        yield _err("model_unavailable", f"{type(exc).__name__}")
        return
    if r.get("stopReason") == "guardrail_intervened":
        bullets = ["Some chats touched topics the safety filter handled."]
    else:
        txt = "".join(c.get("text", "") for c in r["output"]["message"]["content"])
        bullets = [ln.strip()[2:].strip() for ln in txt.splitlines() if ln.strip().startswith("- ")][:5]
    yield {"type": "summary", "bullets": [b[:120] for b in bullets if b]}
    yield {"type": "done", "blocked": False}
