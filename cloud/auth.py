"""Bearer-token auth for the L2 routes.

Token source, first hit wins:

1. ``SG_API_TOKEN`` env (tests, local runs);
2. Secrets Manager secret ``SG_API_TOKEN_SECRET`` (default
   ``screen-guardian/l2-api-token``) read once with the instance role.

No token configured → the protected routes answer 503 (fail closed), never
open.
"""

from __future__ import annotations

import hmac
import os
import threading

from fastapi import HTTPException, Request

DEFAULT_SECRET_ID = "screen-guardian/l2-api-token"
_lock = threading.Lock()
_cached: str | None = None
_loaded = False


def _from_secrets_manager() -> str | None:
    secret_id = os.environ.get("SG_API_TOKEN_SECRET", DEFAULT_SECRET_ID)
    if not secret_id or os.environ.get("SG_NO_SECRETS_MANAGER"):
        return None
    try:
        import boto3

        client = boto3.client("secretsmanager", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        return client.get_secret_value(SecretId=secret_id)["SecretString"].strip() or None
    except Exception:
        return None


def expected_token() -> str | None:
    global _cached, _loaded
    env = os.environ.get("SG_API_TOKEN")
    if env:
        return env.strip()
    with _lock:
        if not _loaded:
            _cached = _from_secrets_manager()
            _loaded = True
    return _cached


def reset_cache() -> None:
    global _cached, _loaded
    with _lock:
        _cached, _loaded = None, False


def token_source() -> str:
    if os.environ.get("SG_API_TOKEN"):
        return "env"
    return "secrets_manager" if expected_token() else "none"


def require_token(request: Request) -> None:
    want = expected_token()
    if not want:
        raise HTTPException(status_code=503, detail="L2 token not configured")
    got = request.headers.get("authorization", "")
    if not got.lower().startswith("bearer ") or not hmac.compare_digest(got[7:].strip(), want):
        raise HTTPException(status_code=401, detail="bad or missing bearer token")
