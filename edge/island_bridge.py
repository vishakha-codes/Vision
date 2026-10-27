"""Engine ↔ island Unix-socket server (``docs/island-protocol.md`` v1).

JSON lines over ``~/Library/Application Support/ScreenGuardian/island.sock``
(mode 0600, override with ``SG_ISLAND_SOCK``). The engine is the server:

- out: ``hello`` on connect, ``state`` (≤ 10 Hz and on change), ``event``,
  ``error``, and ``pong`` for ``ping``.
- in: ``set_mode`` / ``pause`` / ``quit`` (guardian actions: ``sha256(auth)``
  must match ``token_sha256`` in ``guardian.json``), ``resume``, ``ping``.

No pixels, URLs, window titles or detector class names are ever sent.
The current mode is persisted in ``guardian.json`` (``"mode"``); an explicit
``--mode`` on the command line overrides and re-saves it (with a warning).
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import json
import os
import selectors
import socket
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from vision_core.live.contracts import ProtectionMode

PROTOCOL_VERSION = 1
APP_DIR = Path.home() / "Library" / "Application Support" / "ScreenGuardian"
DEFAULT_SOCK = APP_DIR / "island.sock"
DEFAULT_GUARDIAN = APP_DIR / "guardian.json"
STATE_MIN_INTERVAL_S = 0.1  # ≤ 10 Hz
STATE_HEARTBEAT_S = 1.0
MAX_LINE = 64 * 1024
AUTH_ERROR = "guardian auth required"
GUARDIAN_ACTIONS = frozenset({"set_mode", "pause", "quit"})
MOODS = ("idle", "watching", "working", "alert", "covering", "unsure", "lost", "sleep")
STATUSES = ("PROTECTING", "NOT_PROTECTING", "PAUSED", "OFF")


def socket_path() -> Path:
    env = os.environ.get("SG_ISLAND_SOCK")
    return Path(env).expanduser() if env else DEFAULT_SOCK


def guardian_path() -> Path:
    env = os.environ.get("SG_GUARDIAN_JSON")
    return Path(env).expanduser() if env else DEFAULT_GUARDIAN


def _warn(warn: Callable[[str], None] | None, msg: str) -> None:
    if warn is not None:
        warn(msg)
    else:
        print(msg, file=sys.stderr, flush=True)


def parse_mode(value: Any) -> ProtectionMode | None:
    try:
        return ProtectionMode(str(value).strip().lower())
    except ValueError:
        return None


class GuardianStore:
    """``guardian.json``: ``token_sha256`` (written by the island) and
    ``mode`` (written by the engine). Read-modify-write keeps both."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else guardian_path()

    def load(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, self.path)
        os.chmod(self.path, 0o600)

    def mode(self) -> ProtectionMode | None:
        return parse_mode(self.load().get("mode"))

    def resolve_mode(
        self,
        cli_mode: ProtectionMode | str | None = None,
        *,
        warn: Callable[[str], None] | None = None,
    ) -> ProtectionMode:
        """Startup mode.

        - An explicit ``cli_mode`` (``--mode``) wins: it is saved, and when it
          differs from the saved mode a warning goes to ``warn`` (stderr by
          default) so the override is never silent.
        - No CLI mode: the saved mode applies; with none saved, aggressive
          (written back so the next start keeps it).

        Island ``set_mode`` still needs the guardian token; this is the local
        operator starting the engine. Raises ``OSError`` if the file cannot
        be written (the caller falls back to the in-memory mode).
        """
        held = self.mode()
        explicit = parse_mode(cli_mode) if cli_mode is not None else None
        if cli_mode is not None and explicit is None:
            _warn(warn, f"ignoring unknown --mode {cli_mode!r}")
        if explicit is None:
            if held is not None:
                return held
            explicit = ProtectionMode.AGGRESSIVE
        elif held is not None and held is not explicit:
            _warn(
                warn,
                f"WARNING: --mode {explicit.value} overrides the saved mode {held.value} "
                f"({self.path}); saving {explicit.value}.",
            )
        if held is not explicit:
            self.set_mode(explicit)
        return explicit

    def set_mode(self, mode: ProtectionMode) -> None:
        data = self.load()
        data["mode"] = mode.value
        self._write(data)

    def check_auth(self, auth: Any) -> bool:
        expected = str(self.load().get("token_sha256") or "").strip().lower()
        if not expected or not isinstance(auth, str) or not auth:
            return False
        got = hashlib.sha256(auth.encode("utf-8")).hexdigest()
        return hmac.compare_digest(got, expected)


def mood_for(
    *,
    capture_ok: bool,
    mode: ProtectionMode,
    paused: bool,
    started: bool,
    stale: int,
    unsafe: int,
    unsure: int,
    pending: int,
    alert: bool,
) -> tuple[str, str]:
    """Engine state → (status, mood) for the island character."""
    if mode is ProtectionMode.OFF:
        return "OFF", "sleep"
    if paused:
        return "PAUSED", "idle"
    if not capture_ok:
        return "NOT_PROTECTING", "lost"
    if not started or stale:
        return "PROTECTING", "working"
    if unsafe and alert:
        return "PROTECTING", "alert"
    if unsafe:
        return "PROTECTING", "covering"
    if unsure:
        return "PROTECTING", "unsure"
    if pending:
        return "PROTECTING", "working"
    return "PROTECTING", "watching"


class _Client:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.buf = b""


class IslandBridge:
    """Threaded server. Callbacks run on the bridge thread; keep them quick
    and thread-safe (the engine only flips flags)."""

    def __init__(
        self,
        *,
        sock_path: str | Path | None = None,
        guardian: GuardianStore | None = None,
        on_set_mode: Callable[[ProtectionMode], None] | None = None,
        on_pause: Callable[[], None] | None = None,
        on_resume: Callable[[], None] | None = None,
        on_quit: Callable[[], None] | None = None,
        hello: dict | None = None,
    ) -> None:
        self.sock_path = Path(sock_path) if sock_path is not None else socket_path()
        self.guardian = guardian or GuardianStore()
        self.on_set_mode = on_set_mode
        self.on_pause = on_pause
        self.on_resume = on_resume
        self.on_quit = on_quit
        self.hello = {"type": "hello", "version": PROTOCOL_VERSION, "engine": "screen-guardian"}
        self.hello.update(hello or {})
        self._lock = threading.Lock()
        self._state: dict | None = None
        self._state_dirty = False
        self._last_sent_key: str | None = None
        self._last_sent_t = 0.0
        self._outbox: list[dict] = []
        self._clients: list[_Client] = []
        self._server: socket.socket | None = None
        self._sel: selectors.BaseSelector | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._wake_r, self._wake_w = socket.socketpair()
        self.error: str | None = None

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> bool:
        try:
            self._bind()
        except OSError as exc:
            self.error = f"island socket unavailable: {exc}"
            return False
        self._thread = threading.Thread(target=self._run, name="sg-island-bridge", daemon=True)
        self._thread.start()
        return True

    def _bind(self) -> None:
        path = self.sock_path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass
        if path.exists() or path.is_symlink():
            mode = path.lstat().st_mode
            if not stat.S_ISSOCK(mode):
                raise OSError(errno.EEXIST, f"{path} exists and is not a socket")
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.settimeout(0.2)
                probe.connect(str(path))
            except OSError:
                path.unlink()  # stale socket from a crashed engine
            else:
                raise OSError(errno.EADDRINUSE, f"another engine is serving {path}")
            finally:
                probe.close()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o177)
        try:
            server.bind(str(path))
        finally:
            os.umask(old_umask)
        os.chmod(path, 0o600)
        server.listen(4)
        server.setblocking(False)
        self._server = server
        self._sel = selectors.DefaultSelector()
        self._sel.register(server, selectors.EVENT_READ, "accept")
        self._sel.register(self._wake_r, selectors.EVENT_READ, "wake")

    def stop(self) -> None:
        self._stop.set()
        self._wake()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for client in list(self._clients):
            self._drop(client)
        if self._sel is not None:
            try:
                self._sel.close()
            except Exception:
                pass
            self._sel = None
        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None
            try:
                if self.sock_path.exists():
                    self.sock_path.unlink()
            except OSError:
                pass
        for s in (self._wake_r, self._wake_w):
            try:
                s.close()
            except Exception:
                pass

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ---- engine side -----------------------------------------------------

    def publish_state(self, state: dict) -> None:
        """Latest state wins; the bridge thread rate-limits sends."""
        msg = {"type": "state"}
        msg.update(state)
        with self._lock:
            self._state = msg
            self._state_dirty = True
        self._wake()

    def send_event(self, kind: str, **fields: Any) -> None:
        msg = {"type": "event", "kind": kind}
        msg.update(fields)
        self._queue(msg)

    def send_error(self, message: str) -> None:
        self._queue({"type": "error", "message": message})

    def _queue(self, msg: dict) -> None:
        with self._lock:
            self._outbox.append(msg)
        self._wake()

    def _wake(self) -> None:
        try:
            self._wake_w.send(b"\0")
        except OSError:
            pass

    # ---- bridge thread ---------------------------------------------------

    def _run(self) -> None:
        sel = self._sel
        while not self._stop.is_set() and sel is not None:
            try:
                events = sel.select(timeout=0.05)
            except (OSError, ValueError):
                break
            for key, _mask in events:
                if key.data == "accept":
                    self._accept()
                elif key.data == "wake":
                    try:
                        self._wake_r.recv(4096)
                    except OSError:
                        pass
                else:
                    self._read(key.data)
            self._flush()

    def _accept(self) -> None:
        if self._server is None:
            return
        try:
            conn, _addr = self._server.accept()
        except OSError:
            return
        conn.setblocking(True)
        conn.settimeout(0.5)
        client = _Client(conn)
        self._clients.append(client)
        if self._sel is not None:
            self._sel.register(conn, selectors.EVENT_READ, client)
        self._send(client, self.hello)
        with self._lock:
            state = self._state
        if state is not None:
            self._send(client, dict(state, t=round(time.time(), 2)))
            if len(self._clients) == 1:  # it is the only listener: count as sent
                self._last_sent_key = self._state_key(state)
                self._last_sent_t = time.monotonic()
                with self._lock:
                    if self._state is state:
                        self._state_dirty = False

    @staticmethod
    def _state_key(state: dict) -> str:
        return json.dumps({k: v for k, v in state.items() if k != "t"}, sort_keys=True)

    def _drop(self, client: _Client) -> None:
        if client in self._clients:
            self._clients.remove(client)
        if self._sel is not None:
            try:
                self._sel.unregister(client.sock)
            except Exception:
                pass
        try:
            client.sock.close()
        except Exception:
            pass

    def _send(self, client: _Client, msg: dict) -> bool:
        line = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            client.sock.sendall(line)
            return True
        except OSError:
            self._drop(client)
            return False

    def _broadcast(self, msg: dict) -> None:
        for client in list(self._clients):
            self._send(client, msg)

    def _read(self, client: _Client) -> None:
        try:
            chunk = client.sock.recv(4096)
        except (BlockingIOError, socket.timeout):
            return
        except OSError:
            self._drop(client)
            return
        if not chunk:
            self._drop(client)
            return
        client.buf += chunk
        if len(client.buf) > MAX_LINE and b"\n" not in client.buf:
            self._drop(client)
            return
        while b"\n" in client.buf:
            line, client.buf = client.buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue
            if isinstance(msg, dict):
                self._handle(client, msg)

    def _handle(self, client: _Client, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "ping":
            self._send(client, {"type": "pong", "t": round(time.time(), 2)})
            with self._lock:
                state = self._state
            if state is not None:
                self._send(client, dict(state, t=round(time.time(), 2)))
            return
        if kind == "resume":
            if self.on_resume is not None:
                self.on_resume()
            return
        if kind not in GUARDIAN_ACTIONS:
            return  # unknown types are ignored (forward compatible)
        if not self.guardian.check_auth(msg.get("auth")):
            self._send(client, {"type": "error", "message": AUTH_ERROR})
            return
        if kind == "set_mode":
            mode = parse_mode(msg.get("mode"))
            if mode is None:
                self._send(client, {"type": "error", "message": "unknown mode"})
                return
            try:
                self.guardian.set_mode(mode)
            except OSError as exc:
                self._send(client, {"type": "error", "message": f"could not save mode: {exc}"})
            if self.on_set_mode is not None:
                self.on_set_mode(mode)
        elif kind == "pause":
            if self.on_pause is not None:
                self.on_pause()
        elif kind == "quit":
            if self.on_quit is not None:
                self.on_quit()

    def _flush(self) -> None:
        with self._lock:
            outbox, self._outbox = self._outbox, []
            state = self._state
            dirty = self._state_dirty
        for msg in outbox:
            self._broadcast(msg)
        if state is None or not self._clients:
            return
        now = time.monotonic()
        key = self._state_key(state)
        changed = key != self._last_sent_key
        since = now - self._last_sent_t
        if since < STATE_MIN_INTERVAL_S:
            return
        if (dirty and changed) or since >= STATE_HEARTBEAT_S:
            self._broadcast(dict(state, t=round(time.time(), 2)))
            self._last_sent_key = key
            self._last_sent_t = now
            with self._lock:
                if self._state is state:
                    self._state_dirty = False
