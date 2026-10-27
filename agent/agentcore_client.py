"""Mac-side L3/L4 event sender: AgentCore → direct Bedrock (Guardrail) → local rules.

Events only (``stubborn_unsure``, ``heavy_page``, ``session_summary``), never
frames. ``app`` is a bundle id or a bare domain; no titles, URLs or pixels.
Detector labels are reduced to severity tiers before they leave the Mac.

:meth:`AgentClient.send` returns at once; a worker thread calls
``bedrock-agentcore`` ``invoke_agent_runtime`` (SigV4, local AWS profile)
and hands the decision to ``on_result`` (worker thread: hand it to the
vision thread, e.g. ``CloudLane.inject_verdict``). Fallback chain per event:

1. AgentCore Runtime (``SG_AGENT_RUNTIME_ARN`` or ``agent_runtime_arn`` in
   ``~/.screen-guardian/cloud.json``) → Gateway tools (COOL, DynamoDB).
2. The same ``Steward`` run locally: Bedrock Converse with the ``sg-kids``
   Guardrail, tools = the L2 box ``/v1/review`` + a local policy file.
3. The same ``Steward`` under fixed rules (no model).
``session_summary`` events that fail everywhere are queued in
``~/.screen-guardian/pending_events.jsonl`` and re-sent by :meth:`flush_pending`.

Policies the agent returns are cached per app (``agent_policy.json``);
:meth:`effective_mode` never goes below the parent's PIN mode.
"""

from __future__ import annotations

import json
import os
import queue
import ssl
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

from agent.runtime.steward import MODE_RANK, Steward

HOME = Path.home() / ".screen-guardian"
EXPLICIT = {"ANUS_EXPOSED", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"}
SUGGESTIVE = {"ANUS_COVERED", "BUTTOCKS_COVERED", "FEMALE_BREAST_COVERED", "FEMALE_GENITALIA_COVERED", "MALE_GENITALIA_COVERED"}
MILD = {"BELLY_EXPOSED", "MALE_BREAST_EXPOSED"}


def tiers(labels) -> list[str]:
    out = set()
    for lab in labels or ():
        out.add("explicit" if lab in EXPLICIT else "suggestive" if lab in SUGGESTIVE else "mild" if lab in MILD else "")
    return sorted(t for t in out if t)


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def device_id(home: Path = HOME) -> str:
    path = home / "device.json"
    data = _read_json(path, {})
    if not data.get("device_id"):
        data = {"device_id": str(uuid.uuid4())}
        home.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return data["device_id"]


class LocalTools:
    """Fallback tool backend: the L2 box directly + local files."""

    def __init__(self, home: Path = HOME, l2_cfg=None) -> None:
        self.home = home
        self.l2 = l2_cfg

    def __call__(self, name: str, args: dict) -> dict:
        if name == "cool_second_look":
            return self._review(args)
        pol = _read_json(self.home / "agent_policy.json", {})
        if name == "policy_get":
            p = pol.get(args.get("app", ""))
            return {"found": bool(p), "policy": p or {"confirmed": 0, "cleared": 0, "escalation_rate_per_s": 3.0, "prefrost": False}}
        if name == "policy_put":
            cur = pol.get(args["app"], {})
            cur.update({k: args[k] for k in ("min_mode", "escalation_rate_per_s", "prefrost")})
            pol[args["app"]] = cur
            self._write("agent_policy.json", pol)
            return {"ok": True}
        if name in ("verdict_put", "audit_append"):
            if name == "verdict_put" and args.get("verdict") in ("CONFIRM", "CLEAR"):
                cur = pol.setdefault(args.get("app", "unknown.app"), {})
                key = "confirmed" if args["verdict"] == "CONFIRM" else "cleared"
                cur[key] = int(cur.get(key, 0)) + 1
                self._write("agent_policy.json", pol)
            self.home.mkdir(parents=True, exist_ok=True)
            with (self.home / "agent_local_log.jsonl").open("a") as fh:
                fh.write(json.dumps({"ts": time.time(), "tool": name, **args}) + "\n")
            return {"ok": True}
        return {"error": f"unknown tool {name}"}

    def _write(self, fname: str, obj) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / fname).write_text(json.dumps(obj, indent=1))

    def _review(self, args: dict) -> dict:
        cfg = self.l2
        if cfg is None or not cfg.enabled:
            return {"found": False, "detail": "L2 not configured"}
        ctx = None
        if cfg.url.startswith("https"):
            ctx = ssl.create_default_context(cafile=cfg.cert_path or None)
            ctx.check_hostname = False
        req = urllib.request.Request(cfg.url + "/v1/review", method="POST",
                                     data=json.dumps({"dhash": args.get("dhash", ""), "mode": args.get("mode", "aggressive")}).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.token}"})
        with urllib.request.urlopen(req, timeout=8.0, context=ctx) as resp:
            out = json.loads(resp.read())
        v = out.get("verdict") or {}
        return {"found": bool(out.get("found")), "verdict": {"state": v.get("state", ""), "score": v.get("score", 0.0),
                                                              "tiers": tiers(v.get("labels"))}}


class AgentClient:
    def __init__(
        self,
        *,
        runtime_arn: str | None = None,
        region: str = "us-east-1",
        home: Path = HOME,
        on_result: Callable[[dict, dict], None] | None = None,
        agentcore: Any = None,
        bedrock_converse: Callable[..., dict] | None | bool = True,
        local_tools: Callable[[str, dict], dict] | None = None,
        guardrail: dict | None = None,
        queue_max: int = 32,
    ) -> None:
        cfg = _read_json(home / "cloud.json", {})
        self.runtime_arn = runtime_arn or os.environ.get("SG_AGENT_RUNTIME_ARN") or cfg.get("agent_runtime_arn", "")
        self.region = region
        self.home = home
        self.device_id = device_id(home)
        self.on_result = on_result
        self._agentcore = agentcore
        self._converse = bedrock_converse
        self._local_tools = local_tools
        gid = os.environ.get("SG_GUARDRAIL_ID") or cfg.get("guardrail_id", "")
        self.guardrail = guardrail or ({"guardrailIdentifier": gid, "guardrailVersion": os.environ.get("SG_GUARDRAIL_VERSION") or str(cfg.get("guardrail_version", "1"))} if gid else None)
        self._q: queue.Queue = queue.Queue(maxsize=queue_max)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.counts = {"sent": 0, "agentcore": 0, "bedrock_local": 0, "rules_local": 0, "queued_offline": 0, "dropped": 0}

    # ------------------------------------------------------------ events

    def stubborn_unsure(self, *, app: str, mode: str, mode_floor: str, dhash: int, labels, l1_score: float,
                        l2_score: float, l2_state: str, context: dict | None = None) -> bool:
        ev = {"type": "stubborn_unsure", "app": app, "mode": mode, "mode_floor": mode_floor, "dhash": f"{int(dhash):016x}",
              "tiers": tiers(labels), "l1_score": l1_score, "l2_score": l2_score, "l2_state": l2_state}
        return self.send(ev, context=context)

    def heavy_page(self, *, app: str, mode_floor: str, pictures_per_min: float, escalations_per_min: float,
                   unsure_rate: float = 0.0, shed_rate: float = 0.0) -> bool:
        return self.send({"type": "heavy_page", "app": app, "mode_floor": mode_floor, "pictures_per_min": pictures_per_min,
                          "escalations_per_min": escalations_per_min, "unsure_rate": unsure_rate, "shed_rate": shed_rate})

    def session_summary(self, *, counts: dict[str, int], minutes: float, mode_floor: str = "aggressive") -> bool:
        return self.send({"type": "session_summary", "counts": dict(counts), "minutes": minutes, "mode_floor": mode_floor})

    def send(self, event: dict, *, context: dict | None = None) -> bool:
        """Non-blocking. ``context`` stays on the Mac (e.g. raw labels / wh for
        ``CloudLane.inject_verdict``) and is passed back to ``on_result``."""
        event = {**event, "device_id": self.device_id}
        try:
            self._q.put_nowait((event, context or {}))
        except queue.Full:
            self.counts["dropped"] += 1
            return False
        self.counts["sent"] += 1
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="sg-agent-client", daemon=True)
            self._thread.start()
        return True

    # ------------------------------------------------------------ policy

    def cached_policy(self, app: str) -> dict | None:
        return _read_json(self.home / "agent_policy.json", {}).get(app)

    def effective_mode(self, pin_mode: str, app: str) -> str:
        """The PIN mode always wins: the agent may only raise protection."""
        pol = self.cached_policy(app) or {}
        want = str(pol.get("min_mode") or pin_mode)
        return want if MODE_RANK.get(want, 0) > MODE_RANK.get(pin_mode, 0) else pin_mode

    # ------------------------------------------------------------ worker

    def handle_sync(self, event: dict) -> dict:
        """Run one event through the fallback chain (blocking; tests / CLI)."""
        if self.runtime_arn:
            try:
                out = self._invoke_agentcore(event)
                out["lane"] = "agentcore"
                self.counts["agentcore"] += 1
                return out
            except Exception as exc:
                err = f"{type(exc).__name__}: {str(exc)[:200]}"
        else:
            err = "no runtime ARN configured"
        tools = self._local_tools or LocalTools(self.home, self._l2_cfg())
        converse = self._bedrock() if self._converse else None
        out = Steward(tools, converse=converse, guardrail=self.guardrail).handle(event)
        out["lane"] = "bedrock_local" if out.get("brain") == "bedrock" else "rules_local"
        out["agentcore_error"] = err
        self.counts[out["lane"]] += 1
        if event["type"] == "session_summary" and out["lane"] == "rules_local":
            self._queue_offline(event)
        return out

    def _l2_cfg(self):
        try:
            from vision_core.live.cloud_lane import CloudLaneConfig

            return CloudLaneConfig.from_env()
        except Exception:
            return None

    def _bedrock(self):
        if callable(self._converse):
            return self._converse
        try:
            from agent.bedrock_adapter import BedrockAdapter

            return BedrockAdapter(region=self.region, timeout_s=20.0).client.converse
        except Exception:
            return None

    def _invoke_agentcore(self, event: dict) -> dict:
        if self._agentcore is None:
            import boto3
            from botocore.config import Config

            self._agentcore = boto3.client("bedrock-agentcore", region_name=self.region,
                                           config=Config(read_timeout=60, connect_timeout=5, retries={"max_attempts": 1}))
        resp = self._agentcore.invoke_agent_runtime(
            agentRuntimeArn=self.runtime_arn,
            runtimeSessionId=f"sg-{self.device_id}-{uuid.uuid4().hex}",
            payload=json.dumps(event).encode(), contentType="application/json", accept="application/json",
        )
        body = resp["response"].read()
        out = json.loads(body)
        if "error" in out and "verdict" not in out and "summary" not in out:
            raise RuntimeError(out["error"])
        return out

    def _remember_policy(self, app: str, policy: dict | None) -> None:
        if not policy or not app:
            return
        with self._lock:
            pol = _read_json(self.home / "agent_policy.json", {})
            cur = pol.get(app, {})
            cur.update(policy)
            pol[app] = cur
            self.home.mkdir(parents=True, exist_ok=True)
            (self.home / "agent_policy.json").write_text(json.dumps(pol, indent=1))

    def _queue_offline(self, event: dict) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        with (self.home / "pending_events.jsonl").open("a") as fh:
            fh.write(json.dumps(event) + "\n")
        self.counts["queued_offline"] += 1

    def flush_pending(self) -> int:
        path = self.home / "pending_events.jsonl"
        if not path.is_file():
            return 0
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        path.unlink()
        for ln in lines:
            try:
                self.send(json.loads(ln))
            except Exception:
                pass
        return len(lines)

    def _loop(self) -> None:
        while True:
            try:
                event, ctx = self._q.get(timeout=5.0)
            except queue.Empty:
                return
            try:
                out = self.handle_sync(event)
            except Exception as exc:
                out = {"type": event.get("type"), "error": str(exc)[:200], "lane": "failed"}
            if out.get("policy"):
                self._remember_policy(event.get("app", ""), out["policy"])
            if self.on_result is not None:
                try:
                    self.on_result(out, ctx)
                except Exception:
                    pass
