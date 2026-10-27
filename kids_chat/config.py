"""Local chat settings, parent auth, child profile id, counters and opt-in transcripts.

Files (all mode 0600 in a 0700 dir; default ``~/Library/Application Support/ScreenGuardian``,
``SG_CHAT_HOME`` or the island's ``SG_ISLAND_HOME`` relocate them):

- ``chat.json``: ``{"enabled":false,"age_band":"auto","explain_screen":false,
  "transcripts":false,"daily_turns":100,"profile_id":"<uuid4>"}``. Chat is OFF by
  default; every change needs the guardian token (PIN-gated in the island).
- ``guardian.json`` is read-only here: ``token_sha256`` (written by the island) and
  ``mode`` (written by the engine). Same check as ``edge/island_bridge.GuardianStore``
  (``sha256(auth)`` vs ``token_sha256``, constant-time), re-implemented so this
  package never imports ``edge``.
- ``chat-stats.json``: per-day counters for the parent summary (no text).
- ``chat-transcripts/YYYY-MM-DD.jsonl``: only when the parent opted in.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

APP_DIR = Path.home() / "Library" / "Application Support" / "ScreenGuardian"
BANDS = ("auto", "kids", "teens")
MODE_TO_BAND = {"aggressive": "kids", "medium": "teens", "off": None}
STATS_DAYS = 30


def home_dir() -> Path:
    for key in ("SG_CHAT_HOME", "SG_ISLAND_HOME"):
        v = os.environ.get(key)
        if v:
            return Path(v).expanduser()
    return APP_DIR


def socket_path() -> Path:
    v = os.environ.get("SG_CHAT_SOCK")
    return Path(v).expanduser() if v else home_dir() / "chat.sock"


def guardian_path() -> Path:
    v = os.environ.get("SG_GUARDIAN_JSON")
    return Path(v).expanduser() if v else home_dir() / "guardian.json"


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_private(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def check_guardian_auth(auth: Any, path: Path | None = None) -> bool:
    """``sha256(auth)`` against ``token_sha256`` in guardian.json (constant time)."""
    expected = str(_read(path or guardian_path()).get("token_sha256") or "").strip().lower()
    if not expected or not isinstance(auth, str) or not auth:
        return False
    got = hashlib.sha256(auth.encode("utf-8")).hexdigest()
    return hmac.compare_digest(got, expected)


def protection_mode(path: Path | None = None) -> str:
    m = str(_read(path or guardian_path()).get("mode") or "aggressive").lower()
    return m if m in MODE_TO_BAND else "aggressive"


@dataclass
class ChatConfig:
    enabled: bool = False
    age_band: str = "auto"
    explain_screen: bool = False
    transcripts: bool = False
    daily_turns: int = 100
    profile_id: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ChatConfig":
        c = cls()
        names = {f.name for f in fields(cls)}
        for k, v in (d or {}).items():
            if k in names:
                setattr(c, k, v)
        c.enabled, c.explain_screen, c.transcripts = bool(c.enabled is True), bool(c.explain_screen is True), bool(c.transcripts is True)
        c.age_band = c.age_band if c.age_band in BANDS else "auto"
        try:
            c.daily_turns = max(0, min(1000, int(c.daily_turns)))
        except (TypeError, ValueError):
            c.daily_turns = 100
        if not re.fullmatch(r"[0-9a-f-]{36}", str(c.profile_id or "")):
            c.profile_id = ""
        return c


class ConfigStore:
    def __init__(self, home: Path | None = None, guardian: Path | None = None) -> None:
        self.home = home or home_dir()
        self.path = self.home / "chat.json"
        self.guardian = guardian or guardian_path()
        self._lock = threading.Lock()

    def load(self) -> ChatConfig:
        with self._lock:
            c = ChatConfig.from_dict(_read(self.path))
            if not c.profile_id:
                c.profile_id = str(uuid.uuid4())
                write_private(self.path, asdict(c))
            return c

    def update(self, auth: Any, changes: dict) -> ChatConfig:
        """PIN-gated change. Raises PermissionError without a valid guardian token."""
        if not check_guardian_auth(auth, self.guardian):
            raise PermissionError("guardian auth required")
        cur = self.load()
        merged = asdict(cur)
        for k in ("enabled", "age_band", "explain_screen", "transcripts", "daily_turns"):
            if k in changes:
                merged[k] = changes[k]
        if changes.get("new_profile") is True:  # "forget this child": fresh memory key
            merged["profile_id"] = str(uuid.uuid4())
        new = ChatConfig.from_dict(merged)
        new.profile_id = new.profile_id or cur.profile_id
        with self._lock:
            write_private(self.path, asdict(new))
        return new

    def band(self, cfg: ChatConfig | None = None) -> str | None:
        """Effective age band. None = chat not allowed (Adults / mode off), whatever the toggle says."""
        c = cfg or self.load()
        mode_band = MODE_TO_BAND[protection_mode(self.guardian)]
        if mode_band is None:
            return None
        return c.age_band if c.age_band in ("kids", "teens") else mode_band


class Stats:
    """Per-day counters (no text): turns, blocked reasons, safety topics, explain turns."""

    def __init__(self, home: Path | None = None) -> None:
        self.path = (home or home_dir()) / "chat-stats.json"
        self._lock = threading.Lock()

    @staticmethod
    def today() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def load(self) -> dict:
        return _read(self.path)

    def day(self, day: str | None = None) -> dict:
        return self.load().get(day or self.today(), {})

    def turns_today(self) -> int:
        return int(self.day().get("turns", 0))

    def bump(self, *, turn: bool = False, blocked_by: str = "", safety: bool = False, explain: bool = False,
             backend: str = "") -> None:
        with self._lock:
            data = self.load()
            d = data.setdefault(self.today(), {})
            if turn:
                d["turns"] = int(d.get("turns", 0)) + 1
            if blocked_by:
                b = d.setdefault("blocked", {})
                b[blocked_by] = int(b.get(blocked_by, 0)) + 1
            if safety:
                d["safety_topics"] = int(d.get("safety_topics", 0)) + 1
            if explain:
                d["explain"] = int(d.get("explain", 0)) + 1
            if backend:
                bk = d.setdefault("backend", {})
                bk[backend] = int(bk.get(backend, 0)) + 1
            for k in sorted(data)[:-STATS_DAYS]:
                data.pop(k, None)
            write_private(self.path, data)

    def week(self) -> dict:
        data = self.load()
        days = sorted(data)[-7:]
        out = {"days": len(days), "turns": 0, "blocked": 0, "safety_topics": 0, "explain": 0, "blocked_by": {}}
        for k in days:
            d = data[k]
            out["turns"] += int(d.get("turns", 0))
            out["safety_topics"] += int(d.get("safety_topics", 0))
            out["explain"] += int(d.get("explain", 0))
            for r, n in (d.get("blocked") or {}).items():
                out["blocked"] += int(n)
                out["blocked_by"][r] = out["blocked_by"].get(r, 0) + int(n)
        return out


class Transcripts:
    """Local-only, opt-in transcript log (the island shows a 'transcripts on' badge)."""

    def __init__(self, home: Path | None = None) -> None:
        self.dir = (home or home_dir()) / "chat-transcripts"

    def append(self, role: str, text: str, **extra: Any) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        path = self.dir / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as fh:
            fh.write(json.dumps({"t": round(time.time(), 1), "role": role, "text": text[:4000], **extra}) + "\n")
