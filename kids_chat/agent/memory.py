"""AgentCore Memory for the kids chat (plain boto3, no Strands session manager).

- Short-term: one ``create_event`` per finished turn, holding the *masked*
  user text and the assistant text only (never images, never blocked turns).
  Events expire after 7 days (memory resource setting).
- Long-term (built-in strategies, configured in ``infra/deploy.py``):
  ``/summaries/{actorId}/{sessionId}/`` (session summaries) and
  ``/preferences/{actorId}/`` (interests). No semantic-facts strategy.
- ``actorId`` is a random child-profile UUID made on the Mac; no names or emails.

Namespaces MUST match the strategy namespaces in ``infra/deploy.py`` or
retrieval silently returns nothing (megs ``security/identity.py`` contract).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Callable

SUMMARY_NS = "/summaries/{actorId}/{sessionId}/"
PREFERENCE_NS = "/preferences/{actorId}/"
ACTOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,63}$")
SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{32,99}$")
MAX_TURNS = 10  # history turns replayed to the model


def valid_actor(actor: Any) -> bool:
    return isinstance(actor, str) and bool(ACTOR_RE.match(actor))


def valid_session(session: Any) -> bool:
    return isinstance(session, str) and bool(SESSION_RE.match(session))


def summaries_prefix(actor: str) -> str:
    return SUMMARY_NS.split("{sessionId}")[0].replace("{actorId}", actor)


def preferences_ns(actor: str) -> str:
    return PREFERENCE_NS.replace("{actorId}", actor)


class ChatMemory:
    def __init__(self, client: Any, memory_id: str, log: Callable[[str, dict], None] | None = None) -> None:
        self.c = client
        self.memory_id = memory_id
        self.log = log or (lambda e, d: None)

    @property
    def enabled(self) -> bool:
        return bool(self.memory_id and self.c is not None)

    # ------------------------------------------------------------ short-term

    def recent_messages(self, actor: str, session: str, max_turns: int = MAX_TURNS) -> list[dict]:
        """Converse ``messages`` (text only) for the last turns of this session, oldest first."""
        if not self.enabled:
            return []
        try:
            resp = self.c.list_events(memoryId=self.memory_id, actorId=actor, sessionId=session,
                                      includePayloads=True, maxResults=max_turns * 2)
        except Exception as exc:
            self.log("memory_list_failed", {"error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            return []
        events = sorted(resp.get("events") or [], key=lambda e: _ts(e.get("eventTimestamp")))
        msgs: list[dict] = []
        for ev in events[-max_turns:]:
            for item in ev.get("payload") or []:
                conv = item.get("conversational") or {}
                text = ((conv.get("content") or {}).get("text") or "").strip()
                role = "user" if conv.get("role") == "USER" else "assistant" if conv.get("role") == "ASSISTANT" else ""
                if not text or not role:
                    continue
                if msgs and msgs[-1]["role"] == role:
                    msgs[-1]["content"][0]["text"] += "\n" + text
                else:
                    msgs.append({"role": role, "content": [{"text": text}]})
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        if msgs and msgs[-1]["role"] == "user":  # an unanswered turn: drop it
            msgs.pop()
        return msgs

    def write_turn(self, actor: str, session: str, user_text: str, assistant_text: str, *, kind: str = "chat") -> bool:
        if not self.enabled or not user_text or not assistant_text:
            return False
        try:
            self.c.create_event(
                memoryId=self.memory_id, actorId=actor, sessionId=session,
                eventTimestamp=datetime.now(timezone.utc),
                payload=[{"conversational": {"content": {"text": user_text[:2000]}, "role": "USER"}},
                         {"conversational": {"content": {"text": assistant_text[:4000]}, "role": "ASSISTANT"}}],
                metadata={"kind": {"stringValue": kind}},
            )
            return True
        except Exception as exc:
            self.log("memory_write_failed", {"error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            return False

    # ------------------------------------------------------------ long-term

    def _records(self, *, query: str | None, namespace: str | None = None, path: str | None = None, top_k: int = 4) -> list[str]:
        kw: dict = {"memoryId": self.memory_id}
        if namespace:
            kw["namespace"] = namespace
        else:
            kw["namespacePath"] = path
        try:
            if query:
                resp = self.c.retrieve_memory_records(searchCriteria={"searchQuery": query[:500], "topK": top_k}, maxResults=top_k, **kw)
            else:
                resp = self.c.list_memory_records(maxResults=top_k, **kw)
        except Exception as exc:
            self.log("memory_retrieve_failed", {"error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            return []
        out = []
        for r in resp.get("memoryRecordSummaries") or []:
            t = _unwrap(((r.get("content") or {}).get("text") or "").strip())
            if t:
                out.append(t[:600])
        return out

    def context(self, actor: str, query: str) -> str:
        """Interests + recent summaries as a short text block for the system prompt."""
        if not self.enabled:
            return ""
        prefs = self._records(query=query, namespace=preferences_ns(actor), top_k=4)
        sums = self._records(query=query, path=summaries_prefix(actor), top_k=2)
        lines = []
        if prefs:
            lines.append("Interests and preferences:")
            lines += [f"- {p}" for p in prefs]
        if sums:
            lines.append("Earlier chats (summaries):")
            lines += [f"- {s}" for s in sums]
        return "\n".join(lines)

    def summaries(self, actor: str, limit: int = 10) -> list[str]:
        if not self.enabled:
            return []
        return self._records(query=None, path=summaries_prefix(actor), top_k=limit)

    def interests(self, actor: str, limit: int = 10) -> list[str]:
        if not self.enabled:
            return []
        return self._records(query=None, namespace=preferences_ns(actor), top_k=limit)


def _unwrap(t: str) -> str:
    """userPreference records are JSON ({"preference": ..., "context": ...}); keep the sentence."""
    if t.startswith("{"):
        try:
            d = json.loads(t)
            if isinstance(d, dict) and d.get("preference"):
                return str(d["preference"])
        except ValueError:
            pass
    return t


def _ts(v: Any) -> float:
    if isinstance(v, datetime):
        return v.timestamp()
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
