"""Stage 1 of the two-stage guardrail check: ``ApplyGuardrail(source=INPUT)``.

Runs before the model and before memory. Only the *masked* text continues, so
the child's own name, email, age etc. never reach the model or AgentCore
Memory. Stage 2 is the guardrail attached to the model call itself (core.py).

Names: the guardrail anonymises every NAME, which would turn "Who was Ada
Lovelace?" into "Who was {NAME}?". We rebuild the text from the guardrail's
own matches and keep a NAME only when nothing in the message suggests it is
personal ("my", "I'm", "friend", "teacher", ...). Anything we cannot rebuild
exactly falls back to the guardrail's fully masked output (fail safe).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Topics → reason codes (short, no content). Order = priority.
TOPIC_REASONS = (
    ("SelfHarm", "self_harm"),
    ("StrangersPersonalInfo", "personal_info"),
    ("BypassProtection", "bypass"),
    ("DescribeBlockedImagery", "covered_picture"),
    ("RomanceSex", "adult_topic"),
    ("ExplicitSexualContent", "adult_topic"),
    ("DrugsAlcoholVaping", "drugs"),
    ("GettingWeapons", "weapons"),
    ("Gambling", "gambling"),
    ("Extremism", "extremism"),
    ("DangerousChallenges", "dangerous_challenge"),
)
SAFETY_REASONS = frozenset({"self_harm", "personal_info"})

PERSONAL_CUES = re.compile(
    r"\b(my|mine|me|our|i'?m|i am|im|called|named|friend|friends|brother|sister|mom|mum|mommy|dad|daddy|"
    r"grandma|grandpa|aunt|uncle|cousin|teacher|classmate|neighbou?r|coach|babysitter|boyfriend|girlfriend|crush)\b",
    re.IGNORECASE,
)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class Screened:
    blocked: bool
    text: str  # what may go to the model and memory (masked)
    reason: str = ""  # reason code when blocked
    topics: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    pii: list[str] = field(default_factory=list)
    canned: str = ""  # the guardrail's blocked message

    @property
    def safety(self) -> bool:
        return self.reason in SAFETY_REASONS


def clean(text: Any, limit: int = 1000) -> str:
    s = _CTRL.sub("", str(text or "")).strip()
    return s[:limit]


def _assess(resp: dict) -> tuple[list[str], list[str], list[dict]]:
    topics, filters, pii = [], [], []
    for a in resp.get("assessments") or []:
        for t in (a.get("topicPolicy") or {}).get("topics") or []:
            if t.get("action", "BLOCKED") == "BLOCKED":
                topics.append(t.get("name", ""))
        for f in (a.get("contentPolicy") or {}).get("filters") or []:
            if f.get("action", "BLOCKED") == "BLOCKED":
                filters.append(f.get("type", ""))
        sip = a.get("sensitiveInformationPolicy") or {}
        pii.extend(sip.get("piiEntities") or [])
        for w in ((a.get("wordPolicy") or {}).get("customWords") or []) + ((a.get("wordPolicy") or {}).get("managedWordLists") or []):
            if w.get("action", "BLOCKED") == "BLOCKED":
                filters.append("WORDS")
    return topics, filters, pii


def reason_for(topics: list[str], filters: list[str], pii: list[dict]) -> str:
    for name, code in TOPIC_REASONS:
        if name in topics:
            return code
    if "PROMPT_ATTACK" in filters:
        return "prompt_attack"
    if any(p.get("action") == "BLOCKED" for p in pii):
        return "personal_info"
    if filters:
        return "content_filter"
    return "guardrail"


def rebuild(original: str, masked: str, pii: list[dict]) -> str:
    """Original text with every PII match masked, except non-personal NAMEs."""
    personal = bool(PERSONAL_CUES.search(original))
    out = original
    for p in sorted(pii, key=lambda e: -len(str(e.get("match") or ""))):
        m, typ = str(p.get("match") or ""), str(p.get("type") or "PII")
        if not m:
            return masked
        if typ == "NAME" and not personal:
            continue
        if m not in out:
            return masked
        out = out.replace(m, "{" + typ + "}")
    return out


def screen_input(bedrock_runtime: Any, text: str, guardrail: dict) -> Screened:
    """``guardrail`` = {"id": ..., "version": ...}. Raises on AWS errors (caller fails closed)."""
    resp = bedrock_runtime.apply_guardrail(
        guardrailIdentifier=guardrail["id"], guardrailVersion=str(guardrail["version"]),
        source="INPUT", content=[{"text": {"text": text}}])
    topics, filters, pii = _assess(resp)
    outputs = resp.get("outputs") or []
    masked = (outputs[0].get("text") if outputs else None) or text
    if resp.get("action") != "GUARDRAIL_INTERVENED":
        return Screened(False, text)
    blocked = bool(topics or filters or any(p.get("action") == "BLOCKED" for p in pii))
    pii_types = sorted({str(p.get("type")) for p in pii})
    if blocked:
        return Screened(True, "", reason=reason_for(topics, filters, pii), topics=topics, filters=filters,
                        pii=pii_types, canned=masked)
    return Screened(False, rebuild(text, masked, pii), pii=pii_types)


def screen_images(bedrock_runtime: Any, images: list[bytes], guardrail: dict) -> Screened:
    """Image content filter (SEXUAL / VIOLENCE ... with IMAGE modality) on each frame."""
    for img in images:
        resp = bedrock_runtime.apply_guardrail(
            guardrailIdentifier=guardrail["id"], guardrailVersion=str(guardrail["version"]),
            source="INPUT", content=[{"image": {"format": "jpeg", "source": {"bytes": img}}}])
        if resp.get("action") == "GUARDRAIL_INTERVENED":
            topics, filters, pii = _assess(resp)
            return Screened(True, "", reason="unsafe_screen", topics=topics, filters=filters)
    return Screened(False, "")


def screen_output(bedrock_runtime: Any, text: str, guardrail: dict) -> bool:
    """True when ``text`` (e.g. drawing labels) passes the guardrail as model OUTPUT."""
    if not text.strip():
        return True
    resp = bedrock_runtime.apply_guardrail(
        guardrailIdentifier=guardrail["id"], guardrailVersion=str(guardrail["version"]),
        source="OUTPUT", content=[{"text": {"text": text}}])
    if resp.get("action") != "GUARDRAIL_INTERVENED":
        return True
    topics, filters, pii = _assess(resp)
    return not (topics or filters or any(p.get("action") == "BLOCKED" for p in pii))
