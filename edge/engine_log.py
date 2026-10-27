"""Persistent JSON-lines log for the new live engine (``edge.live_engine``).

    log = EngineLog()            # ~/Library/Logs/ScreenGuardian/engine.log
    log.record({"mode": "medium", "fps": 11.8, "masks": 2, ...})

- One JSON object per line, about once a second: snapshots arriving faster
  are dropped (counted in the next line's ``"dropped"``). Snapshots that
  carry an ``"event"`` or ``"error"`` key skip the 1 Hz limit (still capped
  at ``EVENT_MAX_PER_S`` a second) so failures are never lost to rate limiting.
- Size-rotated: ``engine.log`` → ``engine.log.1`` … ``.3`` at 5 MB.
- Privacy: never pixels, window titles or URLs. Keys that look like them
  (``title``, ``url``, ``image``, ``pixels``, ``crop``, ...) are dropped at
  any depth, arrays / bytes are dropped, and strings containing ``://`` are
  replaced. The engine should not pass such fields; this is the backstop.
- Never raises into the engine: every failure is swallowed and the logger
  disables itself after repeated write errors (``error`` says why).

Override the path with ``SG_ENGINE_LOG`` (``off`` / ``-`` disables).

``edge.tick_log`` is a different job and stays for the legacy reflex: it turns
the legacy ``Reflex`` record into one human-readable stderr line for ``-v``.
This module persists the new engine's snapshots to disk.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path.home() / "Library" / "Logs" / "ScreenGuardian" / "engine.log"
MIN_INTERVAL_S = 1.0
EVENT_MAX_PER_S = 10
MAX_BYTES = 5 * 1024 * 1024
BACKUPS = 3
MAX_ERRORS = 5
MAX_DEPTH = 6
MAX_STR = 300
MAX_ITEMS = 64

# Key words that must never reach the log. A key is split into words on
# ``_`` / ``-`` / ``.`` / camelCase, so ``window_title`` and ``pageURL`` are
# dropped while ``during_scroll`` or ``tokens`` are kept.
_BANNED_WORDS = frozenset(
    {
        "pixel", "pixels", "image", "images", "img", "bgr", "bgra", "rgb", "rgba",
        "crop", "crops", "thumb", "thumbs", "thumbnail", "thumbnails", "jpeg", "jpg", "png", "screenshot",
        "title", "titles", "url", "urls", "uri", "href", "domain", "host", "hostname",
        "secret", "token", "auth", "password", "pin",
    }
)
_BANNED_PAIRS = (("window", "name"), ("app", "name"), ("tab", "name"))
_SCRUBBED = "[scrubbed]"


def default_path() -> Path | None:
    env = os.environ.get("SG_ENGINE_LOG")
    if env is not None:
        env = env.strip()
        if env.lower() in {"", "-", "off", "none", "0"}:
            return None
        return Path(env).expanduser()
    return DEFAULT_PATH


def _words(key: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key)
    return [w for w in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if w]


def _banned(key: str) -> bool:
    words = _words(key)
    if any(w in _BANNED_WORDS for w in words):
        return True
    return any(a in words and b in words for a, b in _BANNED_PAIRS)


_DROP = object()  # marker: leave this item out entirely


def scrub(value: Any) -> Any:
    """JSON-safe copy of ``value`` with pixels / titles / URLs removed."""
    out = _scrub(value, 0)
    return None if out is _DROP else out


def _scrub(value: Any, depth: int) -> Any:
    if depth > MAX_DEPTH:
        return _DROP
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, enum.Enum):
        return _scrub(value.value, depth + 1)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 4) if math.isfinite(value) else None
    if isinstance(value, str):
        if "://" in value or value.lower().startswith("www."):
            return _SCRUBBED
        return value if len(value) <= MAX_STR else value[:MAX_STR] + "…"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _DROP
    # numpy without importing it: scalars have .item(), arrays have .shape
    shape = getattr(value, "shape", None)
    if shape is not None and hasattr(value, "dtype"):
        if shape == () and hasattr(value, "item"):
            return _scrub(value.item(), depth + 1)
        size = getattr(value, "size", 0)
        if size and size <= 16 and len(shape) <= 2:  # a box / small vector, not an image
            return _scrub(value.tolist(), depth + 1)
        return _DROP
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= MAX_ITEMS:
                out["_truncated"] = True
                break
            key = str(k)
            if _banned(key):
                continue
            item = _scrub(v, depth + 1)
            if item is not _DROP:
                out[key] = item
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items = (_scrub(v, depth + 1) for v in list(value)[:MAX_ITEMS])
        return [v for v in items if v is not _DROP]
    return _scrub(str(value), depth + 1)


class EngineLog:
    """Rate-limited, size-rotated JSON-lines file. Thread-safe; never raises."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        min_interval_s: float = MIN_INTERVAL_S,
        max_bytes: int = MAX_BYTES,
        backups: int = BACKUPS,
    ) -> None:
        self.path: Path | None = Path(path).expanduser() if path is not None else default_path()
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.max_bytes = max(1024, int(max_bytes))
        self.backups = max(0, int(backups))
        self.error: str | None = None
        self.written = 0
        self.dropped = 0
        self._lock = threading.Lock()
        self._last_t: float | None = None
        self._event_window = (0.0, 0)
        self._errors = 0
        self._seq = 0
        self._dir_ready = False

    @property
    def enabled(self) -> bool:
        return self.path is not None and self._errors < MAX_ERRORS

    def record(self, snapshot: dict) -> None:
        """Append ``snapshot`` (scrubbed) if the rate limit allows. Never raises."""
        try:
            self._record(snapshot)
        except Exception as exc:  # noqa: BLE001 - the engine must never see this
            self._fail(exc)

    # ------------------------------------------------------------ internals

    def _record(self, snapshot: dict) -> None:
        if not self.enabled or not isinstance(snapshot, dict):
            return
        now = time.monotonic()
        urgent = "event" in snapshot or "error" in snapshot
        with self._lock:
            if urgent:
                start, n = self._event_window
                if now - start >= 1.0:
                    start, n = now, 0
                if n >= EVENT_MAX_PER_S:
                    self.dropped += 1
                    return
                self._event_window = (start, n + 1)
            elif self._last_t is not None and now - self._last_t < self.min_interval_s:
                self.dropped += 1
                return
            try:
                body = scrub(snapshot)
            except Exception as exc:  # noqa: BLE001 - a bad value must not count as an I/O failure
                body = {"log_error": f"unserializable snapshot ({type(exc).__name__})"}
            self._seq += 1
            line = {"t": round(time.time(), 3), "seq": self._seq}
            if self.dropped:
                line["dropped"] = self.dropped
            line.update(body if isinstance(body, dict) else {})
            text = json.dumps(line, separators=(",", ":"), ensure_ascii=False, default=str) + "\n"
            self._write(text)
            if not urgent:
                self._last_t = now
            self.dropped = 0
            self.written += 1

    def _write(self, text: str) -> None:
        path = self.path
        assert path is not None
        if not self._dir_ready:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._dir_ready = True
        data = text.encode("utf-8")
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        if size and size + len(data) > self.max_bytes:
            self._rotate()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)

    def _rotate(self) -> None:
        path = self.path
        assert path is not None
        if self.backups <= 0:
            path.unlink(missing_ok=True)
            return
        oldest = path.with_name(f"{path.name}.{self.backups}")
        oldest.unlink(missing_ok=True)
        for i in range(self.backups - 1, 0, -1):
            src = path.with_name(f"{path.name}.{i}")
            if src.exists():
                os.replace(src, path.with_name(f"{path.name}.{i + 1}"))
        if path.exists():
            os.replace(path, path.with_name(f"{path.name}.1"))

    def _fail(self, exc: BaseException) -> None:
        try:
            self._errors += 1
            self.error = f"{type(exc).__name__}: {exc}"
        except Exception:  # noqa: BLE001
            pass
