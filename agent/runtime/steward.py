"""Screen Guardian L3 agent: works on events from the Mac, never on frames.

Events (JSON, no pixels / titles / URLs; ``app`` is a bundle id or a bare
domain):

- ``stubborn_unsure``: a picture still UNSURE after the local and the L2
  COOL look. The agent asks the COOL tool for a deep review of the crop the
  box holds in RAM, then writes CONFIRM / CLEAR / KEEP_UNSURE into the
  verdict store, and may tighten that app's policy.
- ``heavy_page``: picture / escalation load for one app. The agent reads the
  app's history (earlier CONFIRM / CLEAR from COOL reviews) and sets the
  escalation budget and pre-frost for that app.
- ``session_summary``: reason-code counts → a short parent summary through
  Bedrock with the ``sg-kids`` Guardrail.

The model (Claude Sonnet 4.6 when the account has Anthropic access, else Nova 2 Lite, then Nova Pro) chooses and calls the
Gateway tools through a Converse tool loop. This module enforces the safety
invariants on every tool call, whatever the model says:

- The COOL deep review is the evidence: SAFE → CLEAR or KEEP_UNSURE,
  UNSAFE → CONFIRM, UNSURE (or missing) → KEEP_UNSURE or CONFIRM (the one
  judgement left to the model). Never below the mode the parent set with
  the PIN; never ``off``. On ``stubborn_unsure`` only a CONFIRM may change
  the app policy.
- Escalation budget clamped to [0.5, 6] per second; an app with a
  confirmed hit keeps prefrost and >= 4/s; the budget only drops below 3/s
  after >= 3 clean (CLEAR) reviews and no confirmed hit.

If Bedrock is unreachable the same tools run under fixed rules
(``decide_with_rules``), so a decision always comes back.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

MODE_RANK = {"off": 0, "medium": 1, "aggressive": 2}
RATE_MIN, RATE_MAX, RATE_DEFAULT = 0.5, 6.0, 3.0
CONFIRMED_RATE_MIN = 4.0
CLEARED_FOR_LOW = 3
DEFAULT_MODELS = ("us.anthropic.claude-sonnet-4-6", "us.amazon.nova-2-lite-v1:0", "us.amazon.nova-pro-v1:0")
APP_RE = re.compile(r"^(?:[A-Za-z0-9-]+\.)+[A-Za-z0-9-]+$")  # bundle id or bare domain
VERDICTS = ("CONFIRM", "CLEAR", "KEEP_UNSURE")
EVENT_TYPES = ("stubborn_unsure", "heavy_page", "session_summary")
MAX_TURNS = 6

SYSTEM = """You are the policy steward of Screen Guardian, a parental screen-protection app on a child's Mac.
You receive ONE event as JSON. You never see pixels, window titles or URLs. Picture content is described only
by severity tiers (explicit / suggestive / mild) and detector scores from OpenCV running on AWS Graviton (COOL).

Tools (call them, do not guess their results):
- cool_second_look(dhash, mode): deep multi-scale OpenCV review of the escalated picture held in RAM on the COOL box.
- policy_get(device_id, app) / policy_put(...): per-app policy for this device (min_mode, escalation_rate_per_s, prefrost).
- verdict_put(device_id, dhash, verdict, ...): store CONFIRM / CLEAR / KEEP_UNSURE for a picture.
- audit_append(device_id, kind, detail): one short audit line for the parent.

Rules:
1. stubborn_unsure: call cool_second_look first, then policy_get. Review UNSAFE -> CONFIRM. Review SAFE -> CLEAR.
   Review UNSURE or missing -> KEEP_UNSURE (stays covered), or CONFIRM only if the review tier is explicit and its score
   is within 0.03 of the mode's ON threshold (0.25 in aggressive).
   Then call verdict_put. Only if you CONFIRM, tighten the app with policy_put (prefrost true,
   escalation_rate_per_s 5). After CLEAR or KEEP_UNSURE do not call policy_put (heavy_page events set budgets).
2. heavy_page: call policy_get, then policy_put. Decide from the app's history counts:
   - confirmed >= 1: prefrost true, escalation_rate_per_s 5 (a real hit was seen on this app).
   - confirmed == 0 and cleared >= 3: escalation_rate_per_s 1 (only clean reviews; save cost), keep prefrost as it was.
   - otherwise (e.g. confirmed 0 and cleared 0, 1 or 2): escalation_rate_per_s 3, keep prefrost as it was.
   A heavy page (many escalations per minute) is not a reason to lower the budget on its own.
3. Never set min_mode below the event's mode_floor (the parent's PIN mode). Never set "off".
4. Always end with audit_append (one neutral sentence, no description of picture content).
Finish with ONLY a JSON object: {"verdict": "CONFIRM|CLEAR|KEEP_UNSURE|NONE", "policy": {...} or null, "note": "<one sentence>"}."""

SUMMARY_SYSTEM = """You write a short, calm activity summary for a parent of a child using Screen Guardian.
You get FACTS (already correct) and the raw counts. Rephrase the FACTS warmly in 2 or 3 plain sentences.
Keep every number exactly as given; do not add, merge or infer numbers or claims that are not in the FACTS.
Never describe or guess picture content, never name websites, never use body-part words. Reassure without alarm."""


class ToolError(RuntimeError):
    pass


def _rank(mode: str) -> int:
    return MODE_RANK.get(str(mode or "").lower(), MODE_RANK["aggressive"])


def sanitize_event(ev: dict[str, Any]) -> dict[str, Any]:
    """Drop anything that is not an allowed, bounded field."""
    if not isinstance(ev, dict):
        raise ValueError("event must be an object")
    etype = str(ev.get("type", ""))
    if etype not in EVENT_TYPES:
        raise ValueError(f"type must be one of {EVENT_TYPES}")
    out: dict[str, Any] = {"type": etype}
    out["device_id"] = re.sub(r"[^A-Za-z0-9-]", "", str(ev.get("device_id", "")))[:64] or "unknown"
    app = str(ev.get("app", "") or "").strip().lower()
    out["app"] = app if APP_RE.match(app) and len(app) <= 120 else "unknown.app"
    floor = str(ev.get("mode_floor", "aggressive")).lower()
    out["mode_floor"] = floor if floor in MODE_RANK else "aggressive"
    out["mode"] = str(ev.get("mode", out["mode_floor"])).lower()
    if out["mode"] not in MODE_RANK:
        out["mode"] = out["mode_floor"]
    if etype == "stubborn_unsure":
        out["dhash"] = re.sub(r"[^0-9a-f]", "", str(ev.get("dhash", "")).lower())[:16]
        out["tiers"] = [t for t in (ev.get("tiers") or []) if t in ("explicit", "suggestive", "mild")][:3]
        for k in ("l1_score", "l2_score"):
            try:
                out[k] = round(float(ev.get(k, 0.0)), 3)
            except (TypeError, ValueError):
                out[k] = 0.0
        out["l2_state"] = str(ev.get("l2_state", ""))[:12]
    elif etype == "heavy_page":
        for k in ("pictures_per_min", "escalations_per_min", "unsure_rate", "shed_rate"):
            try:
                out[k] = round(float(ev.get(k, 0.0)), 3)
            except (TypeError, ValueError):
                out[k] = 0.0
    else:
        counts = ev.get("counts") or {}
        out["counts"] = {re.sub(r"[^A-Z0-9_]", "", str(k))[:40]: int(v) for k, v in list(counts.items())[:40]
                         if isinstance(v, (int, float))}
        try:
            out["minutes"] = round(float(ev.get("minutes", 0.0)), 1)
        except (TypeError, ValueError):
            out["minutes"] = 0.0
    return out


class Steward:
    """``call_tool(name, args) -> dict`` (the Gateway), ``converse(**req) -> resp``
    (Bedrock runtime ``converse``; None → rules only), ``tool_specs`` from the
    Gateway ``tools/list`` (Converse ``toolSpec`` dicts, short names)."""

    def __init__(
        self,
        call_tool: Callable[[str, dict], dict],
        *,
        converse: Callable[..., dict] | None = None,
        models: tuple[str, ...] = DEFAULT_MODELS,
        guardrail: dict | None = None,
        tool_specs: list[dict] | None = None,
        log: Callable[[str, dict], None] | None = None,
    ) -> None:
        self._call_tool = call_tool
        self._converse = converse
        self.models = models
        self.guardrail = guardrail
        self.tool_specs = tool_specs or []
        self._log = log or (lambda _e, _d: None)
        self.trace: list[dict] = []
        self._review: dict | None = None
        self._policy: dict | None = None
        self.final_verdict: str = "NONE"
        self.final_policy: dict | None = None
        self._event: dict = {}
        self._wrote_verdict = False
        self._wrote_policy = False
        self._audited = False

    # ------------------------------------------------------------ guarded tools

    def tool(self, name: str, args: dict) -> dict:
        """Every tool call (model or rules) passes these invariants."""
        ev = self._event
        args = dict(args or {})
        args["device_id"] = ev.get("device_id", "unknown")
        if name in ("policy_get", "policy_put"):
            args["app"] = ev.get("app", "unknown.app")
        if name == "cool_second_look":
            args = {"dhash": ev.get("dhash", ""), "mode": ev.get("mode", "aggressive")}
        if name == "verdict_put":
            v = str(args.get("verdict", "")).upper()
            review_state = ((self._review or {}).get("verdict") or {}).get("state") if (self._review or {}).get("found") else None
            if v not in VERDICTS:
                return self._record(name, args, {"error": f"verdict must be one of {VERDICTS}"})
            if v == "CLEAR" and review_state != "SAFE":
                return self._record(name, args, {"error": "rejected: CLEAR needs a SAFE COOL review"})
            if review_state == "UNSAFE" and v != "CONFIRM":
                return self._record(name, args, {"error": "rejected: the COOL review was UNSAFE; use CONFIRM"})
            if v == "CONFIRM" and review_state == "SAFE":
                return self._record(name, args, {"error": "rejected: the COOL review was SAFE; use CLEAR or KEEP_UNSURE"})
            args = {"device_id": args["device_id"], "app": ev.get("app", "unknown.app"), "dhash": ev.get("dhash", ""), "verdict": v,
                    "source": "agent", "score": float(((self._review or {}).get("verdict") or {}).get("score", 0.0))}
        if name == "policy_put" and ev.get("type") == "stubborn_unsure" and self.final_verdict != "CONFIRM":
            return self._record(name, args, {"error": "rejected: on stubborn_unsure only a CONFIRM changes policy (call verdict_put first); heavy_page sets budgets"})
        if name == "policy_put":
            if self._policy is None:
                self.tool("policy_get", {})
            hist = (self._policy or {}).get("policy") or {}
            confirmed, cleared = int(hist.get("confirmed") or 0), int(hist.get("cleared") or 0)
            if self.final_verdict == "CONFIRM":
                confirmed += 1
            floor = ev.get("mode_floor", "aggressive")
            want = str(args.get("min_mode", floor)).lower()
            args["min_mode"] = want if want in MODE_RANK and _rank(want) >= _rank(floor) and want != "off" else floor
            try:
                rate = float(args.get("escalation_rate_per_s", RATE_DEFAULT))
            except (TypeError, ValueError):
                rate = RATE_DEFAULT
            prefrost = bool(args.get("prefrost", False))
            # History invariants: a confirmed hit never loses cover or budget;
            # the budget only drops below the default after >= 3 clean reviews.
            if confirmed >= 1:
                rate, prefrost = max(rate, CONFIRMED_RATE_MIN), True
            elif cleared < CLEARED_FOR_LOW:
                rate = max(rate, RATE_DEFAULT)
            args["escalation_rate_per_s"] = round(min(RATE_MAX, max(RATE_MIN, rate)), 2)
            args["prefrost"] = prefrost
            args["note"] = str(args.get("note", ""))[:200]
        if name == "audit_append":
            args = {"device_id": args["device_id"], "kind": ev.get("type", "event"),
                    "detail": str(args.get("detail", ""))[:300]}
        try:
            out = self._call_tool(name, args)
        except Exception as exc:
            out = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        if name == "cool_second_look" and isinstance(out, dict):
            self._review = out
        if name == "policy_get" and isinstance(out, dict) and not out.get("error"):
            self._policy = out
        if name == "verdict_put" and not out.get("error"):
            self._wrote_verdict = True
            self.final_verdict = args["verdict"]
        if name == "policy_put" and not out.get("error"):
            self._wrote_policy = True
            self.final_policy = {k: args[k] for k in ("min_mode", "escalation_rate_per_s", "prefrost")}
        if name == "audit_append" and not out.get("error"):
            self._audited = True
        return self._record(name, args, out)

    def _record(self, name: str, args: dict, out: dict) -> dict:
        self.trace.append({"t": round(time.time(), 3), "step": "tool", "tool": name, "args": args, "result": out})
        self._log("agent_tool", {"tool": name, "ok": not out.get("error")})
        return out

    # ------------------------------------------------------------ entry

    def handle(self, raw_event: dict) -> dict:
        t0 = time.perf_counter()
        ev = sanitize_event(raw_event)
        self._event = ev
        self.trace = [{"t": round(time.time(), 3), "step": "event", "event": ev}]
        self.final_verdict, self.final_policy = "NONE", None
        brain = "rules"
        model = ""
        note = ""
        if ev["type"] == "session_summary":
            result = self._summary(ev)
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            result["trace"] = self.trace
            return result
        if self._converse is not None:
            try:
                note, model = self._llm_loop(ev)
                brain = "bedrock"
            except Exception as exc:  # Bedrock down → same tools under rules
                self.trace.append({"t": round(time.time(), 3), "step": "llm_failed", "error": str(exc)[:300]})
        if brain == "rules":
            note = self.decide_with_rules(ev)
        self._finish(ev)
        return {
            "type": ev["type"],
            "app": ev["app"],
            "dhash": ev.get("dhash", ""),
            "verdict": self.final_verdict,
            "policy": self.final_policy,
            "note": note,
            "brain": brain,
            "model": model,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "trace": self.trace,
        }

    def _finish(self, ev: dict) -> None:
        """Commit what the model skipped (fail closed)."""
        if ev["type"] == "stubborn_unsure" and not self._wrote_verdict:
            state = ((self._review or {}).get("verdict") or {}).get("state") if (self._review or {}).get("found") else None
            v = "CONFIRM" if state == "UNSAFE" else "CLEAR" if state == "SAFE" else "KEEP_UNSURE"
            self.tool("verdict_put", {"verdict": v})
        if not self._audited:
            self.tool("audit_append", {"detail": f"{ev['type']} for {ev['app']}: verdict {self.final_verdict}."})

    # ------------------------------------------------------------ rules brain

    def decide_with_rules(self, ev: dict) -> str:
        if ev["type"] == "stubborn_unsure":
            review = self.tool("cool_second_look", {})
            pol = self.tool("policy_get", {})
            state = (review.get("verdict") or {}).get("state") if review.get("found") else None
            v = "CONFIRM" if state == "UNSAFE" else "CLEAR" if state == "SAFE" else "KEEP_UNSURE"
            self.tool("verdict_put", {"verdict": v})
            if v == "CONFIRM":
                cur = pol.get("policy") or {}
                self.tool("policy_put", {"min_mode": cur.get("min_mode", ev["mode_floor"]),
                                         "escalation_rate_per_s": 5.0, "prefrost": True,
                                         "note": "COOL review confirmed a hit"})
            return f"rules: COOL review {state or 'missing'} → {v}"
        pol = self.tool("policy_get", {})
        cur = pol.get("policy") or {}
        confirmed, cleared = int(cur.get("confirmed", 0)), int(cur.get("cleared", 0))
        if confirmed >= 1:
            rate, prefrost = 5.0, True
        elif cleared >= 3:
            rate, prefrost = 1.0, bool(cur.get("prefrost", False))
        else:
            rate, prefrost = RATE_DEFAULT, bool(cur.get("prefrost", False))
        self.tool("policy_put", {"min_mode": cur.get("min_mode", ev["mode_floor"]), "escalation_rate_per_s": rate,
                                 "prefrost": prefrost, "note": f"heavy_page; history confirmed={confirmed} cleared={cleared}"})
        return f"rules: history confirmed={confirmed} cleared={cleared} → rate {rate}/s prefrost {prefrost}"

    # ------------------------------------------------------------ model brain

    def _llm_loop(self, ev: dict) -> tuple[str, str]:
        messages = [{"role": "user", "content": [{"text": "Event:\n" + json.dumps(ev)}]}]
        last_err = None
        for model in self.models:
            try:
                return self._run_model(model, messages), model
            except ToolError:
                raise
            except Exception as exc:
                last_err = exc
                self.trace.append({"t": round(time.time(), 3), "step": "model_failed", "model": model, "error": str(exc)[:200]})
        raise RuntimeError(f"no model answered: {last_err}")

    def _run_model(self, model: str, messages: list[dict]) -> str:
        msgs = list(messages)
        for turn in range(MAX_TURNS):
            req: dict[str, Any] = {
                "modelId": model,
                "system": [{"text": SYSTEM}],
                "messages": msgs,
                "inferenceConfig": {"maxTokens": 600, "temperature": 0.0},
                "toolConfig": {"tools": [{"toolSpec": s} for s in self.tool_specs]},
            }
            if self.guardrail:
                req["guardrailConfig"] = self.guardrail
            t = time.perf_counter()
            resp = self._converse(**req)
            msg = resp["output"]["message"]
            self.trace.append({"t": round(time.time(), 3), "step": "model", "model": model, "turn": turn,
                               "stop": resp.get("stopReason"), "ms": round((time.perf_counter() - t) * 1000, 1),
                               "usage": resp.get("usage", {}),
                               "text": "".join(c.get("text", "") for c in msg.get("content", []))[:600],
                               "tool_uses": [c["toolUse"]["name"] for c in msg.get("content", []) if "toolUse" in c]})
            msgs.append(msg)
            if resp.get("stopReason") == "guardrail_intervened":
                if any(t.get("step") == "tool" for t in self.trace):
                    # Decisions were already committed through the guarded tools; the
                    # Guardrail only masked / blocked the closing note (e.g. a domain
                    # anonymised as {URL}). Keep the decisions, drop the note.
                    return "closing note withheld by the Guardrail"
                raise RuntimeError("guardrail intervened on the steward conversation")
            uses = [c["toolUse"] for c in msg.get("content", []) if "toolUse" in c]
            if not uses:
                text = "".join(c.get("text", "") for c in msg.get("content", []))
                return self._parse_final(text)
            results = []
            for use in uses:
                out = self.tool(use["name"], use.get("input") or {})
                results.append({"toolResult": {"toolUseId": use["toolUseId"], "content": [{"json": out}],
                                               "status": "error" if out.get("error") else "success"}})
            msgs.append({"role": "user", "content": results})
        return "model hit the turn limit; committed under rules"

    def _parse_final(self, text: str) -> str:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return text.strip()[:200]
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return text.strip()[:200]
        return str(obj.get("note", ""))[:200]

    # ------------------------------------------------------------ L4 summary

    def _summary(self, ev: dict) -> dict:
        counts = ev.get("counts", {})
        local = local_summary(counts, ev.get("minutes", 0.0))
        text, brain, model, gr = local, "rules", "", "NONE"
        if self._converse is not None:
            for m in self.models:
                try:
                    # FACTS go in the system prompt (our own text; the Guardrail checks the
                    # counts message and the model's answer).
                    req = {"modelId": m, "system": [{"text": SUMMARY_SYSTEM + "\nFACTS: " + local}],
                           "messages": [{"role": "user", "content": [{"text": json.dumps({"minutes": ev.get("minutes", 0.0), "counts": counts})}]}],
                           "inferenceConfig": {"maxTokens": 200, "temperature": 0.0}}
                    if self.guardrail:
                        req["guardrailConfig"] = self.guardrail
                    resp = self._converse(**req)
                    out = "".join(c.get("text", "") for c in resp["output"]["message"].get("content", [])).strip()
                    gr = "INTERVENED" if resp.get("stopReason") == "guardrail_intervened" else "NONE"
                    if gr == "INTERVENED":
                        # Blocked model text never reaches the parent: keep the rule summary.
                        self.trace.append({"t": round(time.time(), 3), "step": "guardrail_blocked_summary", "model": m})
                        model = m
                        break
                    if out:
                        text, brain, model = out, "bedrock", m
                        break
                except Exception as exc:
                    self.trace.append({"t": round(time.time(), 3), "step": "model_failed", "model": m, "error": str(exc)[:200]})
        self.tool("audit_append", {"detail": text[:300]})
        return {"type": "session_summary", "summary": text, "brain": brain, "model": model, "guardrail_action": gr}


def local_summary(counts: dict[str, int], minutes: float = 0.0) -> str:
    """Rule-based parent summary (offline fallback, and the FACTS the model rephrases). Counts only."""
    covered = sum(v for k, v in counts.items() if k.startswith("UNSAFE") or k.startswith("UNSURE"))
    pre = counts.get("PENDING_PREFROST", 0)
    conf, clr = counts.get("AGENT_CONFIRM", 0), counts.get("AGENT_CLEAR", 0)
    span = f"In the last {minutes:.0f} minutes" if minutes else "This session"
    s = f"{span}, Screen Guardian kept {covered} picture{'s' if covered != 1 else ''} hidden behind a cover"
    s += f", and briefly covered {pre} more until they were checked." if pre else "."
    if conf or clr:
        s += f" A deeper cloud check kept {conf} of them hidden and found {clr} fine to show."
    return s + " Nothing else needs your attention right now."
