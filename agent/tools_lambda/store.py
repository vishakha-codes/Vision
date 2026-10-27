"""Gateway Lambda target ``sg-store``: policy_get, policy_put, verdict_put, audit_append.

DynamoDB single table ``SG_TABLE`` (pk / sk, pay-per-request, TTL ``ttl``):

- ``DEV#<device>`` / ``POLICY#<app>``: min_mode, escalation_rate_per_s,
  prefrost, confirmed, cleared, note, updated
- ``DEV#<device>`` / ``VERDICT#<dhash>``: verdict, score, source, app (30 d)
- ``DEV#<device>`` / ``AUDIT#<iso>#<n>``: kind, detail (90 d)

Validation here too (defence in depth): app = bundle id or bare domain,
min_mode never ``off``, rate in [0.5, 6]. No pixels, titles or URLs exist in
any item.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from decimal import Decimal

import boto3

TABLE = os.environ.get("SG_TABLE", "screen-guardian-agent")
APP_RE = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z0-9-]+$")
DEV_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")
HASH_RE = re.compile(r"^[0-9a-f]{1,16}$")
MODES = ("medium", "aggressive")
_table = None


def table():
    global _table
    if _table is None:
        _table = boto3.resource("dynamodb").Table(TABLE)
    return _table


def _plain(obj):
    if isinstance(obj, Decimal):
        return float(obj) if obj % 1 else int(obj)
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def _dev(args: dict) -> str:
    dev = str(args.get("device_id", ""))
    if not DEV_RE.match(dev):
        raise ValueError("device_id must be [A-Za-z0-9-]{1,64}")
    return dev


def _app(args: dict) -> str:
    app = str(args.get("app", "")).lower()
    if not APP_RE.match(app) or len(app) > 120:
        raise ValueError("app must be a bundle id or bare domain (no URL, no title)")
    return app


def policy_get(args: dict) -> dict:
    dev, app = _dev(args), _app(args)
    item = table().get_item(Key={"pk": f"DEV#{dev}", "sk": f"POLICY#{app}"}).get("Item")
    if not item:
        return {"found": False, "policy": {"app": app, "min_mode": None, "escalation_rate_per_s": 3.0,
                                           "prefrost": False, "confirmed": 0, "cleared": 0}}
    pol = {k: v for k, v in _plain(item).items() if k not in ("pk", "sk")}
    return {"found": True, "policy": pol}


def policy_put(args: dict) -> dict:
    dev, app = _dev(args), _app(args)
    mode = str(args.get("min_mode", "aggressive")).lower()
    if mode not in MODES:
        raise ValueError("min_mode must be medium or aggressive (never off)")
    rate = min(6.0, max(0.5, float(args.get("escalation_rate_per_s", 3.0))))
    table().update_item(
        Key={"pk": f"DEV#{dev}", "sk": f"POLICY#{app}"},
        UpdateExpression="SET app = :a, min_mode = :m, escalation_rate_per_s = :r, prefrost = :p, note = :n, updated = :u",
        ExpressionAttributeValues={":a": app, ":m": mode, ":r": Decimal(str(round(rate, 2))),
                                   ":p": bool(args.get("prefrost", False)), ":n": str(args.get("note", ""))[:200],
                                   ":u": int(time.time())},
    )
    return {"ok": True, "app": app, "min_mode": mode, "escalation_rate_per_s": round(rate, 2),
            "prefrost": bool(args.get("prefrost", False))}


def verdict_put(args: dict) -> dict:
    dev = _dev(args)
    dhash = str(args.get("dhash", "")).lower()
    if not HASH_RE.match(dhash):
        raise ValueError("dhash must be hex")
    verdict = str(args.get("verdict", "")).upper()
    if verdict not in ("CONFIRM", "CLEAR", "KEEP_UNSURE"):
        raise ValueError("verdict must be CONFIRM, CLEAR or KEEP_UNSURE")
    app = str(args.get("app", "unknown.app")).lower()
    if not APP_RE.match(app):
        app = "unknown.app"
    now = int(time.time())
    table().put_item(Item={"pk": f"DEV#{dev}", "sk": f"VERDICT#{dhash}", "verdict": verdict, "app": app,
                           "score": Decimal(str(round(float(args.get("score", 0.0)), 4))),
                           "source": str(args.get("source", "agent"))[:16], "ts": now, "ttl": now + 30 * 86400})
    if verdict in ("CONFIRM", "CLEAR"):
        field = "confirmed" if verdict == "CONFIRM" else "cleared"
        table().update_item(
            Key={"pk": f"DEV#{dev}", "sk": f"POLICY#{app}"},
            UpdateExpression=f"ADD {field} :one SET app = :a, updated = :u",
            ExpressionAttributeValues={":one": 1, ":a": app, ":u": now},
        )
    return {"ok": True, "dhash": dhash, "verdict": verdict, "app": app}


def audit_append(args: dict) -> dict:
    dev = _dev(args)
    now = time.time()
    sk = f"AUDIT#{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(now))}#{uuid.uuid4().hex[:6]}"
    table().put_item(Item={"pk": f"DEV#{dev}", "sk": sk, "kind": str(args.get("kind", "event"))[:32],
                           "detail": str(args.get("detail", ""))[:300], "ttl": int(now) + 90 * 86400})
    return {"ok": True, "sk": sk}


TOOLS = {"policy_get": policy_get, "policy_put": policy_put, "verdict_put": verdict_put, "audit_append": audit_append}


def handler(event, context):
    name = ""
    try:
        name = context.client_context.custom["bedrockAgentCoreToolName"].split("___")[-1]
    except Exception:
        name = str((event or {}).pop("_tool", ""))
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}"}
    try:
        out = fn(dict(event or {}))
    except Exception as exc:
        out = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    print(json.dumps({"event": "tool", "tool": name, "ok": "error" not in out}))
    return out
