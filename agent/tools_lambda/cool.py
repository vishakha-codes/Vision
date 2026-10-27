"""Gateway Lambda target ``sg-cool``: cool_second_look → the Graviton COOL box.

Runs in the default VPC and calls the box on its private IP
(``SG_BOX_URL``, e.g. ``https://<box-private-ip>:8443``) over the pinned
self-signed TLS cert (``SG_BOX_CERT_PEM``) with the bearer token
(``SG_L2_TOKEN``; Lambda env is KMS-encrypted at rest). It asks
``POST /v1/review`` for a deep multi-scale look at the crop the box holds in
RAM after an UNSURE L2 result. Labels are mapped to severity tiers before they
reach the agent (no body-part words in the model conversation).
"""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.request

EXPLICIT = {"ANUS_EXPOSED", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"}
SUGGESTIVE = {"ANUS_COVERED", "BUTTOCKS_COVERED", "FEMALE_BREAST_COVERED", "FEMALE_GENITALIA_COVERED", "MALE_GENITALIA_COVERED"}
MILD = {"BELLY_EXPOSED", "MALE_BREAST_EXPOSED"}


def tier(label: str) -> str:
    if label in EXPLICIT:
        return "explicit"
    if label in SUGGESTIVE:
        return "suggestive"
    if label in MILD:
        return "mild"
    return "other"


def _ctx() -> ssl.SSLContext:
    pem = os.environ.get("SG_BOX_CERT_PEM", "")
    ctx = ssl.create_default_context(cadata=pem) if pem else ssl.create_default_context()
    ctx.check_hostname = False  # pinned cert; the box is addressed by private IP
    return ctx


def cool_second_look(args: dict) -> dict:
    dhash = str(args.get("dhash", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{1,16}", dhash):
        return {"error": "dhash must be hex"}
    mode = str(args.get("mode", "aggressive")).lower()
    body = json.dumps({"dhash": dhash, "mode": mode}).encode()
    req = urllib.request.Request(
        os.environ["SG_BOX_URL"].rstrip("/") + "/v1/review", data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['SG_L2_TOKEN']}"},
    )
    with urllib.request.urlopen(req, timeout=8.0, context=_ctx()) as resp:
        out = json.loads(resp.read())
    verdict = out.get("verdict") or {}
    dets = out.get("dets") or []
    return {
        "found": bool(out.get("found")),
        "detail": out.get("detail", ""),
        "verdict": {"state": verdict.get("state", ""), "score": verdict.get("score", 0.0),
                    "tiers": sorted({tier(lab) for lab in verdict.get("labels", [])})},
        "top": [{"tier": tier(d["label"]), "score": d["score"]} for d in dets[:5] if tier(d["label"]) != "other"],
        "n_views": out.get("n_views", 0),
        "opencv": out.get("opencv", {}),
        "timings_ms": out.get("timings_ms", {}),
        "image_persisted": False,
    }


def handler(event, context):
    try:
        name = context.client_context.custom["bedrockAgentCoreToolName"].split("___")[-1]
    except Exception:
        name = "cool_second_look"
    if name != "cool_second_look":
        return {"error": f"unknown tool {name!r}"}
    try:
        out = cool_second_look(dict(event or {}))
    except Exception as exc:
        out = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    print(json.dumps({"event": "tool", "tool": name, "ok": "error" not in out,
                      "state": (out.get("verdict") or {}).get("state", "")}))
    return out
