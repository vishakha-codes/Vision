"""Bedrock Converse with the Screen Guardian Guardrail (L4, and the L3 fallback).

- Model chain: ``SG_BEDROCK_MODELS`` (comma list) or Claude Sonnet 4.6 (US
  cross-region profile), then Amazon Nova Lite as the cheap fallback. Both
  were checked with Converse in account <account-id> / us-east-1 on 26 Oct.
- Guardrail: ``SG_GUARDRAIL_ID`` / ``SG_GUARDRAIL_VERSION`` (see
  ``cloud/deploy/resources-log.md``). Attached to every call when set, so
  any agent text a parent or child could see goes through it.
- Text only: callers pass reason codes and counts, never pixels, titles or
  URLs.

``BedrockUnavailable`` is raised when no model answers (no credentials, no
model access, timeouts); callers fall back to local rules.
"""

from __future__ import annotations

import os
import time
from typing import Any

DEFAULT_MODELS = ("us.anthropic.claude-sonnet-4-6", "us.amazon.nova-2-lite-v1:0", "us.amazon.nova-pro-v1:0")
GUARDRAIL_ID_ENV = "SG_GUARDRAIL_ID"
GUARDRAIL_VERSION_ENV = "SG_GUARDRAIL_VERSION"


class BedrockUnavailable(RuntimeError):
    """No model in the chain answered (credentials, access, network, timeout)."""


def _models_from_env() -> tuple[str, ...]:
    raw = os.environ.get("SG_BEDROCK_MODELS", "")
    models = tuple(m.strip() for m in raw.split(",") if m.strip())
    return models or DEFAULT_MODELS


class BedrockAdapter:
    """Thin Converse wrapper: model fallback chain + Guardrail + timeouts."""

    def __init__(
        self,
        *,
        region: str | None = None,
        model_id: str | None = None,
        models: tuple[str, ...] | None = None,
        guardrail_id: str | None = None,
        guardrail_version: str | None = None,
        timeout_s: float = 20.0,
        client: Any = None,
    ) -> None:
        self.region = region or os.environ.get("AWS_REGION") or "us-east-1"
        chain = tuple(models or _models_from_env())
        self.models = (model_id,) + tuple(m for m in chain if m != model_id) if model_id else chain
        self.model_id = self.models[0]
        self.guardrail_id = guardrail_id if guardrail_id is not None else os.environ.get(GUARDRAIL_ID_ENV, "")
        self.guardrail_version = guardrail_version or os.environ.get(GUARDRAIL_VERSION_ENV, "DRAFT")
        self.timeout_s = float(timeout_s)
        self._client = client
        self._probe: tuple[float, bool, str] | None = None

    # ------------------------------------------------------------ plumbing

    @property
    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            cfg = Config(
                connect_timeout=5,
                read_timeout=self.timeout_s,
                retries={"max_attempts": 2, "mode": "standard"},
            )
            self._client = boto3.client("bedrock-runtime", region_name=self.region, config=cfg)
        return self._client

    def guardrail_config(self) -> dict[str, str] | None:
        if not self.guardrail_id:
            return None
        return {"guardrailIdentifier": self.guardrail_id, "guardrailVersion": self.guardrail_version, "trace": "enabled"}

    # ------------------------------------------------------------ API

    def converse(
        self,
        messages: list[dict[str, Any]] | str,
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        guardrail: bool = True,
    ) -> dict[str, Any]:
        """One Converse turn. Returns ``text``, ``message``, ``stop_reason``,
        ``model_id``, ``usage``, ``latency_ms``, ``guardrail_action``."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": [{"text": messages}]}]
        req: dict[str, Any] = {
            "messages": messages,
            "inferenceConfig": {"maxTokens": int(max_tokens), "temperature": float(temperature)},
        }
        if system:
            req["system"] = [{"text": system}]
        if tools:
            req["toolConfig"] = {"tools": [{"toolSpec": t} for t in tools]}
        gr = self.guardrail_config() if guardrail else None
        if gr:
            req["guardrailConfig"] = gr
        errors: list[str] = []
        for model in self.models:
            t0 = time.perf_counter()
            try:
                resp = self.client.converse(modelId=model, **req)
            except Exception as exc:  # access denied, throttled, timeout, no creds
                errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:160]}")
                continue
            msg = resp.get("output", {}).get("message", {}) or {}
            text = "".join(part.get("text", "") for part in msg.get("content", []) if "text" in part)
            trace = resp.get("trace", {}) or {}
            return {
                "text": text.strip(),
                "message": msg,
                "stop_reason": resp.get("stopReason", ""),
                "model_id": model,
                "usage": resp.get("usage", {}),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "guardrail_action": "INTERVENED" if resp.get("stopReason") == "guardrail_intervened" else "NONE",
                "guardrail_trace": bool(trace.get("guardrail")),
            }
        raise BedrockUnavailable("; ".join(errors) or "no Bedrock model configured")

    def invoke(self, prompt: str = "", **kwargs: Any) -> dict[str, Any]:
        """Back-compat: single user prompt → ``converse``."""
        return self.converse(prompt, **kwargs)

    def apply_guardrail(self, text: str, *, source: str = "OUTPUT") -> dict[str, Any]:
        """Run text through the Guardrail without a model (``ApplyGuardrail``)."""
        if not self.guardrail_id:
            raise BedrockUnavailable("no guardrail configured")
        try:
            resp = self.client.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                source=source,
                content=[{"text": {"text": text}}],
            )
        except Exception as exc:
            raise BedrockUnavailable(f"ApplyGuardrail failed: {exc}") from exc
        outs = resp.get("outputs") or []
        return {
            "action": resp.get("action", "NONE"),
            "text": outs[0].get("text", text) if outs else text,
        }

    def available(self, *, ttl_s: float = 300.0) -> bool:
        """Cheap cached probe (one tiny Converse on the first model)."""
        now = time.monotonic()
        if self._probe is not None and now - self._probe[0] < ttl_s:
            return self._probe[1]
        ok, why = False, ""
        try:
            self.converse("Reply OK.", max_tokens=5, guardrail=False)
            ok = True
        except BedrockUnavailable as exc:
            why = str(exc)[:300]
        self._probe = (now, ok, why)
        return ok

    @property
    def unavailable_reason(self) -> str:
        return self._probe[2] if self._probe else ""
