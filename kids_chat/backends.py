"""How the Mac service reaches the agent: AgentCore Runtime first, then direct Bedrock.

Every backend's :meth:`turn` is a generator of the same event dicts that
``agent/core.py`` yields (``delta`` / ``draw`` / ``done`` / ``error`` / ``summary``).

1. :class:`AgentCoreBackend`: ``bedrock-agentcore`` ``invoke_agent_runtime``
   (SigV4 with the local AWS profile), SSE parsed line by line.
2. :class:`LocalBackend`: the same ``core.run`` in-process, calling Bedrock
   Converse directly with the same Guardrail (and AgentCore Memory when it is
   reachable). Used only when AgentCore is unreachable.
3. If both fail before anything was streamed: a friendly offline ``done``.

``ChainBackend`` only falls through when *nothing* was emitted yet, so a
child never sees a half answer followed by a different one.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Iterator

from kids_chat.agent import core, prompts

INFRA_OUTPUTS = Path(__file__).resolve().parent / "infra" / "outputs.json"


class BackendUnavailable(RuntimeError):
    pass


def load_outputs(path: Path = INFRA_OUTPUTS) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def parse_sse(lines) -> Iterator[dict]:
    """``data: {...}`` frames → dicts (comments / keepalives skipped)."""
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict):
            yield ev


class AgentCoreBackend:
    name = "agentcore"

    def __init__(self, runtime_arn: str, *, region: str = "us-east-1", client: Any = None,
                 profile: str | None = None) -> None:
        self.runtime_arn = runtime_arn
        self.region = region
        self._client = client
        self._profile = profile
        self._lock = threading.Lock()

    def client(self) -> Any:
        with self._lock:
            if self._client is None:
                import boto3
                from botocore.config import Config
                sess = boto3.Session(profile_name=self._profile) if self._profile else boto3.Session()
                self._client = sess.client("bedrock-agentcore", region_name=self.region,
                                           config=Config(read_timeout=120, connect_timeout=6, retries={"max_attempts": 1}))
            return self._client

    def turn(self, payload: dict, cancel: threading.Event | None = None) -> Iterator[dict]:
        if not self.runtime_arn:
            raise BackendUnavailable("no runtime ARN")
        session = payload.get("session_id") or payload.get("actor_id", "") + "-summary-" + "0" * 24
        try:
            resp = self.client().invoke_agent_runtime(
                agentRuntimeArn=self.runtime_arn, runtimeSessionId=session, qualifier="DEFAULT",
                contentType="application/json", accept="text/event-stream",
                payload=json.dumps(payload).encode())
        except Exception as exc:
            raise BackendUnavailable(f"{type(exc).__name__}: {str(exc)[:200]}") from exc
        body = resp.get("response")
        ctype = str(resp.get("contentType") or "")
        if body is None:
            raise BackendUnavailable("empty response")
        if "event-stream" not in ctype:
            data = body.read()
            try:
                obj = json.loads(data or b"{}")
            except ValueError:
                raise BackendUnavailable(f"unexpected response type {ctype}")
            yield obj if "type" in obj else {"type": "error", "code": "internal", "message": str(obj)[:200]}
            return
        try:
            for ev in parse_sse(body.iter_lines()):
                if cancel is not None and cancel.is_set():
                    body.close()
                    return
                yield ev
        finally:
            try:
                body.close()
            except Exception:
                pass


class LocalBackend:
    """Direct Bedrock (same guardrail) + AgentCore Memory, in-process."""
    name = "bedrock_direct"

    def __init__(self, outputs: dict | None = None, *, deps: core.Deps | None = None, region: str = "us-east-1",
                 log: Callable[[str, dict], None] | None = None) -> None:
        self._outputs = outputs or {}
        self._deps = deps
        self.region = region
        self.log = log or (lambda e, d: None)
        self._lock = threading.Lock()

    def deps(self) -> core.Deps:
        with self._lock:
            if self._deps is None:
                o = self._outputs
                if not o.get("guardrail_kids_id"):
                    raise BackendUnavailable("no guardrail ids (kids_chat/infra/outputs.json)")
                import boto3
                from botocore.config import Config
                from kids_chat.agent.memory import ChatMemory
                cfg = Config(read_timeout=60, connect_timeout=5, retries={"max_attempts": 1})
                models = o.get("models") or {}
                self._deps = core.Deps(
                    bedrock=boto3.client("bedrock-runtime", region_name=self.region, config=cfg),
                    memory=ChatMemory(boto3.client("bedrock-agentcore", region_name=self.region, config=cfg),
                                      o.get("memory_id", ""), log=self.log),
                    guardrails={"kids": {"id": o["guardrail_kids_id"], "version": o["guardrail_kids_version"]},
                                "teens": {"id": o["guardrail_teens_id"], "version": o["guardrail_teens_version"]}},
                    chat_model=models.get("chat", "us.amazon.nova-2-lite-v1:0"),
                    fallback_model=models.get("fallback", "us.amazon.nova-lite-v1:0"),
                    explain_model=models.get("explain", "us.amazon.nova-pro-v1:0"),
                    log=self.log)
            return self._deps

    def turn(self, payload: dict, cancel: threading.Event | None = None) -> Iterator[dict]:
        try:
            d = self.deps()
        except BackendUnavailable:
            raise
        except Exception as exc:
            raise BackendUnavailable(f"{type(exc).__name__}: {str(exc)[:200]}") from exc
        for ev in core.run(payload, d):
            if cancel is not None and cancel.is_set():
                return
            if ev.get("type") == "error" and ev.get("code") in ("model_unavailable", "guardrail_unavailable"):
                raise BackendUnavailable(ev.get("message", ""))
            yield ev


def offline_events(band: str = "kids") -> list[dict]:
    return [{"type": "done", "blocked": False, "offline": True, "text": prompts.OFFLINE}]


class ChainBackend:
    """Try each backend in order; fall through only if nothing was emitted yet."""

    def __init__(self, backends: list, log: Callable[[str, dict], None] | None = None) -> None:
        self.backends = [b for b in backends if b is not None]
        self.log = log or (lambda e, d: None)
        self.last_used = ""

    def turn(self, payload: dict, cancel: threading.Event | None = None) -> Iterator[dict]:
        for b in self.backends:
            emitted = False
            try:
                for ev in b.turn(payload, cancel):
                    if not emitted and ev.get("type") == "error" and ev.get("code") in ("internal", "model_unavailable", "guardrail_unavailable"):
                        raise BackendUnavailable(ev.get("message", ev.get("code", "")))
                    emitted = True
                    self.last_used = b.name
                    yield {**ev, "backend": b.name} if ev.get("type") == "done" else ev
                return
            except BackendUnavailable as exc:
                self.log("backend_unavailable", {"backend": b.name, "error": str(exc)[:200], "emitted": emitted})
                if emitted:
                    yield {"type": "error", "code": "offline", "message": "connection lost"}
                    return
            except Exception as exc:  # defensive: never crash the service
                self.log("backend_failed", {"backend": b.name, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                if emitted:
                    yield {"type": "error", "code": "offline", "message": "connection lost"}
                    return
            if cancel is not None and cancel.is_set():
                return
        self.last_used = "offline"
        for ev in offline_events(payload.get("age_band", "kids")):
            yield {**ev, "backend": "offline"}


def default_chain(log: Callable[[str, dict], None] | None = None, outputs: dict | None = None) -> ChainBackend:
    o = outputs if outputs is not None else load_outputs()
    arn = os.environ.get("SG_CHAT_RUNTIME_ARN") or o.get("runtime_arn", "")
    region = o.get("region", "us-east-1")
    chain: list = []
    if os.environ.get("SG_CHAT_FORCE_LOCAL") != "1" and arn:
        chain.append(AgentCoreBackend(arn, region=region, profile=os.environ.get("SG_CHAT_AWS_PROFILE")))
    chain.append(LocalBackend(o, region=region, log=log))
    return ChainBackend(chain, log=log)
