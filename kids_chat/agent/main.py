"""AgentCore Runtime entry point for the v2 kids chat (direct code deploy, Python 3.12, arm64).

HTTP contract of AgentCore Runtime: ``GET /ping`` and ``POST /invocations``
on 0.0.0.0:8080. Standard library + boto3 only (same shape as the v1
steward, ``vision/agent/runtime/main.py``). Replies stream as Server-Sent
Events (``text/event-stream``): one ``data: {json}`` frame per event from
:func:`core.run`, a ``: keepalive`` comment every 15 s while waiting, and a
final ``done`` or ``error`` frame.

Local run (fallback when the deploy is down):
    cd vision && SG_MEMORY_ID=... SG_GR_KIDS_ID=... .venv/bin/python -m kids_chat.agent.main
    curl -N localhost:8080/invocations -d '{"prompt":"hi","actor_id":"...","session_id":"...","age_band":"kids"}'

Env (set by ``kids_chat/infra/deploy.py``): ``SG_MEMORY_ID``,
``SG_GR_KIDS_ID/VER``, ``SG_GR_TEENS_ID/VER``, ``SG_CHAT_MODEL``,
``SG_FALLBACK_MODEL``, ``SG_EXPLAIN_MODEL``.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # flat layout inside the runtime zip
    import core
except ImportError:  # repo layout
    from kids_chat.agent import core

import boto3
from botocore.config import Config

REGION = os.environ.get("AWS_REGION", "us-east-1")
MAX_BODY = 2_000_000
KEEPALIVE_S = 15.0
STARTED = time.time()
_deps = None
_lock = threading.Lock()


def log(event: str, data: dict) -> None:
    # Never log prompt or answer text: reason codes, sizes and timings only.
    sys.stdout.write(json.dumps({"ts": round(time.time(), 3), "event": event, **data}, default=str) + "\n")
    sys.stdout.flush()


def deps() -> "core.Deps":
    global _deps
    with _lock:
        if _deps is None:
            cfg = Config(read_timeout=60, connect_timeout=5, retries={"max_attempts": 2})
            _deps = core.deps_from_env(boto3.client("bedrock-runtime", region_name=REGION, config=cfg),
                                       boto3.client("bedrock-agentcore", region_name=REGION, config=cfg), log=log)
        return _deps


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/ping"):
            self._json(200, {"status": "Healthy", "time_of_last_update": int(STARTED)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if not self.path.startswith("/invocations"):
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                self._json(413, {"error": "payload too large"})
                return
            payload = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            if isinstance(payload.get("prompt"), str) and payload["prompt"].lstrip().startswith("{") and "actor_id" not in payload:
                payload = json.loads(payload["prompt"])  # console test payloads
        except ValueError as exc:
            self._json(400, {"error": str(exc)[:200]})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        q: queue.Queue = queue.Queue()
        t0 = time.time()

        def work():
            try:
                for ev in core.run(payload, deps()):
                    q.put(ev)
            except Exception as exc:
                log("turn_error", {"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                q.put({"type": "error", "code": "internal", "message": type(exc).__name__})
            finally:
                q.put(None)

        threading.Thread(target=work, daemon=True).start()
        kinds: dict[str, int] = {}
        last = {}
        try:
            while True:
                try:
                    ev = q.get(timeout=KEEPALIVE_S)
                except queue.Empty:
                    self._chunk(b": keepalive\n\n")
                    continue
                if ev is None:
                    break
                kinds[ev.get("type", "?")] = kinds.get(ev.get("type", "?"), 0) + 1
                last = ev
                self._chunk(b"data: " + json.dumps(ev).encode() + b"\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            log("client_gone", {})
        log("turn", {"action": payload.get("action", "chat"), "band": payload.get("age_band"), "events": kinds,
                     "blocked": last.get("blocked"), "blocked_by": last.get("blocked_by"), "code": last.get("code"),
                     "ms": int((time.time() - t0) * 1000)})


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    log("kids_chat_start", {"memory": bool(os.environ.get("SG_MEMORY_ID")), "model": os.environ.get("SG_CHAT_MODEL")})
    srv.serve_forever()


if __name__ == "__main__":
    main()
