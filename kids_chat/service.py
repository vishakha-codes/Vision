"""Local kids-chat service: its own process, its own Unix socket, JSON lines.

    cd vision && .venv/bin/python -m kids_chat.service            # default socket
    SG_CHAT_SOCK=/tmp/sgc/chat.sock .venv/bin/python -m kids_chat.service --verbose

Socket ``~/Library/Application Support/ScreenGuardian/chat.sock`` (mode 0600,
``SG_CHAT_SOCK`` overrides). The island is the client. Protocol:
``kids_chat/PROTOCOL.md``. This process holds the AWS credentials (local
profile) and calls AgentCore with SigV4; the island never sees them. It does
not talk to the vision engine's ``island.sock``; for explain-my-screen it asks
the engine's ``redact.sock`` for a redacted snapshot and fails closed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import stat
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from kids_chat import draw as drawmod
from kids_chat.agent import guard
from kids_chat.backends import ChainBackend, default_chain
from kids_chat.config import ConfigStore, Stats, Transcripts, socket_path
from kids_chat.snapshot import SnapshotRefused, default_provider

PROTOCOL_VERSION = 1
MAX_LINE = 64 * 1024
MAX_TEXT = 500
MOTION = re.compile(r"\b(move|moves|moving|spin|spins|spinning|turn|turns|turning|motion|animat\w*|happen\w* next|flow\w*|going on in (the|this) video)\b", re.I)


class Conn:
    def __init__(self, sock: socket.socket | None = None, sink: Callable[[dict], None] | None = None) -> None:
        self.sock = sock
        self.sink = sink
        self.lock = threading.Lock()
        self.alive = True

    def send(self, obj: dict) -> bool:
        if self.sink is not None:
            self.sink(obj)
            return True
        data = (json.dumps(obj, separators=(",", ":")) + "\n").encode()
        with self.lock:
            if not self.alive or self.sock is None:
                return False
            try:
                self.sock.sendall(data)
                return True
            except OSError:
                self.alive = False
                return False


class ChatService:
    def __init__(self, *, store: ConfigStore | None = None, backend: ChainBackend | Any = None,
                 snapshots: Any = None, stats: Stats | None = None, transcripts: Transcripts | None = None,
                 log: Callable[[str, dict], None] | None = None, sync: bool = False) -> None:
        self.log = log or (lambda e, d: None)
        self.store = store or ConfigStore()
        self.stats = stats or Stats(self.store.home)
        self.transcripts = transcripts or Transcripts(self.store.home)
        self.backend = backend if backend is not None else default_chain(log=self.log)
        self.snapshots = snapshots if snapshots is not None else default_provider()
        self.sync = sync
        self._busy: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._session_n = 1
        self._session_day = ""
        self.conns: set[Conn] = set()

    # ------------------------------------------------------------ helpers

    def session_id(self, profile: str) -> str:
        day = datetime.now().strftime("%Y%m%d")
        if day != self._session_day:
            self._session_day, self._session_n = day, 1
        return f"{profile}-{day}-{self._session_n}"

    def state(self) -> dict:
        cfg = self.store.load()
        band = self.store.band(cfg)
        try:
            avail, why = self.snapshots.available() if (cfg.enabled and cfg.explain_screen) else (False, "off")
        except Exception:
            avail, why = False, "engine_unavailable"
        left = max(0, cfg.daily_turns - self.stats.turns_today())
        return {"type": "chat.state", "version": PROTOCOL_VERSION, "enabled": bool(cfg.enabled and band),
                "configured": cfg.enabled, "age_band": band or "off", "age_band_setting": cfg.age_band,
                "explain_screen": cfg.explain_screen, "explain_available": bool(avail),
                "explain_reason": "" if avail else why, "transcripts": cfg.transcripts, "turns_left": left,
                "daily_turns": cfg.daily_turns}

    def broadcast(self, obj: dict) -> None:
        for c in list(self.conns):
            c.send(obj)

    # ------------------------------------------------------------ dispatch

    def handle(self, msg: dict, conn: Conn) -> None:
        t = msg.get("type")
        if t in ("hello", "chat.state"):
            conn.send(self.state())
        elif t == "ping":
            conn.send({"type": "pong", "t": time.time()})
        elif t == "chat.send":
            self.on_send(msg, conn)
        elif t == "chat.cancel":
            ev = self._busy.get(str(msg.get("id")))
            if ev:
                ev.set()
        elif t == "chat.new":
            self._session_n += 1
            conn.send({"type": "chat.new", "ok": True})
        elif t == "chat.config":
            self.on_config(msg, conn)
        elif t == "chat.summary":
            self.on_summary(msg, conn)
        elif t == "draw.clear":
            self.broadcast({"type": "draw.clear"})
        # unknown types are ignored (forward compatible)

    def on_config(self, msg: dict, conn: Conn) -> None:
        try:
            self.store.update(msg.get("auth"), msg)
        except PermissionError:
            conn.send({"type": "chat.error", "code": "auth_required", "message": "guardian auth required"})
            return
        self.log("config_changed", {k: msg[k] for k in ("enabled", "age_band", "explain_screen", "transcripts") if k in msg})
        self.broadcast(self.state())

    def on_summary(self, msg: dict, conn: Conn) -> None:
        from kids_chat.config import check_guardian_auth
        if not check_guardian_auth(msg.get("auth"), self.store.guardian):
            conn.send({"type": "chat.error", "code": "auth_required", "message": "guardian auth required"})
            return
        cfg = self.store.load()
        counts = self.stats.week()

        def work():
            bullets, note = [], ""
            try:
                for ev in self.backend.turn({"action": "parent_summary", "actor_id": cfg.profile_id,
                                             "age_band": self.store.band(cfg) or "kids"}):
                    if ev.get("type") == "summary":
                        bullets, note = ev.get("bullets") or [], ev.get("note", "")
            except Exception as exc:
                self.log("summary_failed", {"error": type(exc).__name__})
            if counts["safety_topics"]:
                bullets.append(f"A safety topic came up {counts['safety_topics']}x (no details are kept).")
            conn.send({"type": "chat.summary", "bullets": bullets[:6], "note": note, "counts": counts,
                       "transcripts": cfg.transcripts})

        self._run(work)

    def on_send(self, msg: dict, conn: Conn) -> None:
        mid = str(msg.get("id") or "")[:40] or f"c{int(time.time() * 1000)}"
        cfg = self.store.load()
        band = self.store.band(cfg)

        def err(code: str, message: str) -> None:
            conn.send({"type": "chat.error", "id": mid, "code": code, "message": message})

        if not cfg.enabled or band is None:
            return err("chat_disabled", "Ask a grown-up to turn on chat")
        text = guard.clean(msg.get("text"), MAX_TEXT + 1)
        if not text:
            return err("bad_request", "empty message")
        if len(text) > MAX_TEXT:
            return err("too_long", f"Messages can be up to {MAX_TEXT} characters")
        if self.stats.turns_today() >= cfg.daily_turns:
            return err("daily_limit", "That's all the chatting for today. See you tomorrow!")
        with self._lock:
            if self._busy:
                return err("busy", "One message at a time")
            cancel = threading.Event()
            self._busy[mid] = cancel
        explain = bool(msg.get("explain"))
        payload = {"action": "chat", "prompt": text, "actor_id": cfg.profile_id, "session_id": self.session_id(cfg.profile_id),
                   "age_band": band, "name": str(msg.get("name") or "")[:20]}

        def work():
            try:
                self._turn(mid, text, payload, explain, cfg, conn, cancel)
            finally:
                with self._lock:
                    self._busy.pop(mid, None)

        self._run(work)

    def _turn(self, mid: str, text: str, payload: dict, explain: bool, cfg, conn: Conn, cancel: threading.Event) -> None:
        snap = None
        if explain:
            if not cfg.explain_screen:
                conn.send({"type": "chat.error", "id": mid, "code": "explain_disabled", "message": "Ask a grown-up to turn on explain-my-screen"})
                return
            try:
                snap = self.snapshots.get(frames=3 if MOTION.search(text) else 1)
            except SnapshotRefused as exc:
                code = "explain_unavailable" if exc.reason in ("engine_unavailable", "no_verifier") else "refused_unsafe_screen"
                self.log("snapshot_refused", {"reason": exc.reason})
                conn.send({"type": "chat.error", "id": mid, "code": code, "reason": exc.reason,
                           "message": "I can't look at the screen right now."})
                return
            payload["explain"] = snap.payload()
        self.stats.bump(turn=True, explain=explain)
        if cfg.transcripts:
            self.transcripts.append("child", text, id=mid, explain=explain)
        answer, done, drew = "", {}, False
        try:
            for ev in self.backend.turn(payload, cancel):
                if cancel.is_set():
                    break
                k = ev.get("type")
                if k == "delta":
                    answer += ev.get("text", "")
                    conn.send({"type": "chat.delta", "id": mid, "text": ev.get("text", "")})
                elif k == "draw" and snap is not None:
                    ops = drawmod.to_screen(ev.get("ops") or [], snap.display, snap.masks_norm)
                    if ops:
                        conn.send({"type": "chat.draw", "id": mid, "ops": ops, "space": "quartz_pt", "ttl_ms": drawmod.TTL_MS,
                                   "display": snap.display, "avoid": drawmod.avoid_rects(snap.display, snap.masks_norm),
                                   "replace": not drew})
                        drew = True
                elif k == "done":
                    done = ev
                elif k == "error":
                    done = {"error": ev}
        except Exception as exc:  # never crash the service on a turn
            self.log("turn_failed", {"error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            done = {"error": {"code": "offline"}}
        if cancel.is_set():
            conn.send({"type": "chat.done", "id": mid, "blocked": False, "cancelled": True})
            return
        if "error" in done:
            code = done["error"].get("code", "offline")
            code = code if code in ("bad_request", "offline") else "offline"
            conn.send({"type": "chat.error", "id": mid, "code": code, "message": "I can't chat right now. Let's try again soon!"})
            return
        out = {"type": "chat.done", "id": mid, "blocked": bool(done.get("blocked"))}
        if done.get("blocked"):
            out["blocked_by"] = done.get("blocked_by", "guardrail")
            out["safety"] = bool(done.get("safety"))
            self.stats.bump(blocked_by=out["blocked_by"], safety=out["safety"])
        if done.get("text") and (done.get("blocked") or done.get("offline") or not answer):
            out["text"] = done["text"]
        if done.get("offline"):
            out["offline"] = True
        out["backend"] = done.get("backend", "")
        if out["backend"]:
            self.stats.bump(backend=out["backend"])
        conn.send(out)
        if cfg.transcripts:
            self.transcripts.append("helper", out.get("text") or answer, id=mid, blocked=out["blocked"])
        self.log("turn_done", {"id": mid, "blocked": out["blocked"], "blocked_by": out.get("blocked_by", ""),
                               "backend": out["backend"], "explain": explain})

    def _run(self, fn: Callable[[], None]) -> None:
        if self.sync:
            fn()
        else:
            threading.Thread(target=fn, name="sg-chat-turn", daemon=True).start()


# ---------------------------------------------------------------- socket server

class Server:
    def __init__(self, service: ChatService, path: Path | None = None) -> None:
        self.service = service
        self.path = Path(path or socket_path())
        self.sock: socket.socket | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        if self.path.exists() or self.path.is_symlink():
            if not stat.S_ISSOCK(os.lstat(self.path).st_mode):
                raise RuntimeError(f"{self.path} exists and is not a socket")
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(str(self.path))
                probe.close()
                raise RuntimeError(f"another chat service is listening on {self.path}")
            except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
                if isinstance(exc, RuntimeError):
                    raise
                self.path.unlink()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            s.bind(str(self.path))
        finally:
            os.umask(old)
        os.chmod(self.path, 0o600)
        s.listen(4)
        s.settimeout(0.5)
        self.sock = s
        threading.Thread(target=self._accept, name="sg-chat-accept", daemon=True).start()

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                c, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._client, args=(c,), name="sg-chat-client", daemon=True).start()

    def _client(self, c: socket.socket) -> None:
        conn = Conn(c)
        self.service.conns.add(conn)
        conn.send(self.service.state())
        buf = b""
        try:
            while not self._stop.is_set():
                data = c.recv(16384)
                if not data:
                    break
                buf += data
                if len(buf) > MAX_LINE and b"\n" not in buf:
                    break
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(msg, dict):
                        self.service.handle(msg, conn)
        except OSError:
            pass
        finally:
            conn.alive = False
            self.service.conns.discard(conn)
            try:
                c.close()
            except OSError:
                pass

    def stop(self) -> None:
        self._stop.set()
        if self.sock is not None:
            self.sock.close()
        try:
            self.path.unlink()
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Screen Guardian kids chat service")
    ap.add_argument("--sock", default=None, help="socket path (default SG_CHAT_SOCK or the support dir)")
    ap.add_argument("--verbose", action="store_true", help="log events (never message text) to stderr")
    args = ap.parse_args(argv)

    def log(event: str, data: dict) -> None:
        if args.verbose:
            print(json.dumps({"ts": round(time.time(), 2), "event": event, **data}), file=sys.stderr, flush=True)

    service = ChatService(log=log)
    if os.environ.get("SG_CHAT_FAKE_SNAPSHOT") == "1":
        print("[kids_chat] WARNING: SG_CHAT_FAKE_SNAPSHOT=1, explain-my-screen uses a synthetic picture", file=sys.stderr)
    server = Server(service, Path(args.sock) if args.sock else None)
    server.start()
    print(f"[kids_chat] listening on {server.path}", file=sys.stderr, flush=True)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    while not stop.is_set():
        stop.wait(0.5)
    server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
