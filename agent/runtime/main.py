"""AgentCore Runtime entry point (direct code deploy, Python 3.12, arm64).

HTTP contract of AgentCore Runtime: ``GET /ping`` and ``POST /invocations``
on 0.0.0.0:8080. Standard library server + boto3 only (no framework), so the
zip stays small. One Steward per request (no cross-request state).

Env (set by ``vision/infra/deploy.py``): ``SG_GATEWAY_URL``,
``SG_GUARDRAIL_ID``, ``SG_GUARDRAIL_VERSION``, ``SG_BEDROCK_MODELS``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # flat layout inside the runtime zip
    from gateway_client import GatewayClient
    from steward import Steward
except ImportError:  # repo layout (tests)
    from agent.runtime.gateway_client import GatewayClient
    from agent.runtime.steward import Steward

import boto3
from botocore.config import Config

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODELS = tuple(m for m in os.environ.get("SG_BEDROCK_MODELS", "us.anthropic.claude-sonnet-4-6,us.amazon.nova-2-lite-v1:0,us.amazon.nova-pro-v1:0").split(",") if m)
_bedrock = None
_cw = None
_gateway = None
_specs: list[dict] | None = None
STARTED = time.time()


def log(event: str, data: dict) -> None:
    sys.stdout.write(json.dumps({"ts": round(time.time(), 3), "event": event, **data}, default=str) + "\n")
    sys.stdout.flush()


def clients():
    global _bedrock, _cw, _gateway, _specs
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime", region_name=REGION,
                                config=Config(read_timeout=30, connect_timeout=5, retries={"max_attempts": 2}))
        _cw = boto3.client("cloudwatch", region_name=REGION)
        _gateway = GatewayClient(os.environ["SG_GATEWAY_URL"], region=REGION)
    if _specs is None:
        _specs = _gateway.list_tools()
    return _bedrock, _cw, _gateway, _specs


def guardrail() -> dict | None:
    gid = os.environ.get("SG_GUARDRAIL_ID", "")
    if not gid:
        return None
    return {"guardrailIdentifier": gid, "guardrailVersion": os.environ.get("SG_GUARDRAIL_VERSION", "DRAFT"), "trace": "enabled"}


def metrics(cw, result: dict) -> None:
    try:
        dims = [{"Name": "Service", "Value": "agentcore-steward"}]
        data = [
            {"MetricName": "AgentEvents", "Dimensions": dims + [{"Name": "EventType", "Value": result.get("type", "?")}], "Value": 1, "Unit": "Count"},
            {"MetricName": "AgentDecisionMs", "Dimensions": dims, "Value": float(result.get("latency_ms", 0.0)), "Unit": "Milliseconds"},
            {"MetricName": "AgentToolCalls", "Dimensions": dims, "Value": float(sum(1 for s in result.get("trace", []) if s.get("step") == "tool")), "Unit": "Count"},
        ]
        v = result.get("verdict")
        if v in ("CONFIRM", "CLEAR", "KEEP_UNSURE"):
            data.append({"MetricName": f"AgentVerdict{v.title().replace('_', '')}", "Dimensions": dims, "Value": 1, "Unit": "Count"})
        if result.get("brain") == "rules":
            data.append({"MetricName": "AgentRulesFallback", "Dimensions": dims, "Value": 1, "Unit": "Count"})
        cw.put_metric_data(Namespace="ScreenGuardian", MetricData=data)
    except Exception as exc:
        log("metrics_failed", {"error": str(exc)[:200]})


def handle(event: dict) -> dict:
    bedrock, cw, gateway, specs = clients()
    st = Steward(gateway.call, converse=bedrock.converse, models=MODELS, guardrail=guardrail(),
                 tool_specs=specs, log=log)
    result = st.handle(event)
    log("agent_decision", {k: result.get(k) for k in ("type", "app", "dhash", "verdict", "policy", "brain", "model", "latency_ms")})
    metrics(cw, result)
    return result


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/ping"):
            self._send(200, {"status": "Healthy", "time_of_last_update": int(STARTED)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if not self.path.startswith("/invocations"):
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 64_000:
                self._send(413, {"error": "event too large"})
                return
            event = json.loads(self.rfile.read(n) or b"{}")
            if "prompt" in event and isinstance(event["prompt"], str):  # console / CLI test payloads
                event = json.loads(event["prompt"])
            self._send(200, handle(event))
        except ValueError as exc:
            self._send(400, {"error": str(exc)[:300]})
        except Exception as exc:
            log("agent_error", {"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            self._send(500, {"error": f"{type(exc).__name__}: {str(exc)[:200]}"})


def main() -> None:
    srv = ThreadingHTTPServer(("0.0.0.0", 8080), H)
    log("agent_start", {"models": MODELS, "guardrail": bool(guardrail())})
    srv.serve_forever()


if __name__ == "__main__":
    main()
