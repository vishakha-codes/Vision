"""Minimal MCP (streamable HTTP, JSON-RPC) client for AgentCore Gateway, SigV4.

The Gateway uses the ``AWS_IAM`` inbound authorizer, so the runtime's
execution role signs each call (service ``bedrock-agentcore``). Gateway tool
names are ``<target>___<tool>``; this client exposes the short tool names.
"""

from __future__ import annotations

import itertools
import json
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

_ids = itertools.count(1)


class GatewayClient:
    def __init__(self, url: str, *, region: str = "us-east-1", timeout_s: float = 15.0) -> None:
        self.url = url
        self.region = region
        self.timeout_s = timeout_s
        self._session = boto3.Session()
        self._names: dict[str, str] = {}

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        body = json.dumps({"jsonrpc": "2.0", "id": next(_ids), "method": method, "params": params or {}}).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        req = AWSRequest(method="POST", url=self.url, data=body, headers=headers)
        SigV4Auth(self._session.get_credentials().get_frozen_credentials(), "bedrock-agentcore", self.region).add_auth(req)
        http = urllib.request.Request(self.url, data=body, headers=dict(req.headers), method="POST")
        with urllib.request.urlopen(http, timeout=self.timeout_s) as resp:
            raw = resp.read().decode()
            ctype = resp.headers.get("content-type", "")
        if "text/event-stream" in ctype:  # take the last data: frame
            frames = [ln[5:].strip() for ln in raw.splitlines() if ln.startswith("data:")]
            raw = frames[-1] if frames else "{}"
        out = json.loads(raw)
        if "error" in out:
            raise RuntimeError(f"gateway {method}: {out['error']}")
        return out.get("result", {})

    def list_tools(self) -> list[dict]:
        """Converse ``toolSpec`` dicts with short names."""
        specs = []
        cursor = None
        while True:
            res = self._rpc("tools/list", {"cursor": cursor} if cursor else {})
            for t in res.get("tools", []):
                full = t["name"]
                short = full.split("___", 1)[-1]
                self._names[short] = full
                specs.append({"name": short, "description": t.get("description", "")[:500],
                              "inputSchema": {"json": t.get("inputSchema") or {"type": "object"}}})
            cursor = res.get("nextCursor")
            if not cursor:
                return specs

    def call(self, name: str, args: dict) -> dict:
        if not self._names:
            self.list_tools()
        full = self._names.get(name)
        if full is None:
            return {"error": f"unknown tool {name}"}
        res = self._rpc("tools/call", {"name": full, "arguments": args})
        if res.get("isError"):
            text = "".join(c.get("text", "") for c in res.get("content", []))
            return {"error": text[:300] or "tool error"}
        for c in res.get("content", []):
            if c.get("type") == "text":
                try:
                    return json.loads(c["text"])
                except (json.JSONDecodeError, KeyError):
                    return {"text": c.get("text", "")}
        return res.get("structuredContent") or {}
