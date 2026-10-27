#!/usr/bin/env python3
"""Mock Screen Guardian engine for the native island (stdlib only).

Serves the island protocol (vision/docs/island-protocol.md) on a Unix socket,
cycles moods / status / mask counts for demos, and validates ``auth`` on the
guardian actions (set_mode, pause, quit) against guardian.json, exactly like the
real engine should.

    python3 vision/island/tools/mock_engine.py              # demo loop
    python3 vision/island/tools/mock_engine.py --static     # stays "watching"
    python3 vision/island/tools/mock_engine.py --speed 3    # faster loop
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

MODES = ("aggressive", "medium", "off")

# (seconds, state fields, optional event)
SCRIPT: list[tuple[float, dict, dict | None]] = [
    (4.0, {"mood": "watching", "masks": 0, "pending": 0, "reason": "", "warning": None}, None),
    (3.0, {"mood": "working", "pending": 1}, None),
    (1.4, {"mood": "alert", "reason": "UNSAFE_PICTURE"}, None),
    (4.0, {"mood": "covering", "masks": 2, "pending": 0}, {"type": "event", "kind": "covered", "count": 2}),
    (3.0, {"mood": "unsure", "pending": 1}, None),
    (3.0, {"mood": "covering", "masks": 9, "pending": 4, "warning": {"code": "busy", "text": "Lots to cover here"}}, None),
    (4.0, {"mood": "watching", "masks": 1, "pending": 0, "reason": "", "warning": None}, None),
    (3.0, {"mood": "idle", "masks": 0}, {"type": "event", "kind": "cleared"}),
]


def support_dir() -> Path:
    home = os.environ.get("SG_ISLAND_HOME")
    if home:
        return Path(home)
    return Path.home() / "Library" / "Application Support" / "ScreenGuardian"


def default_sock() -> str:
    return os.environ.get("SG_ISLAND_SOCK") or str(support_dir() / "island.sock")


def log(msg: str) -> None:
    print(f"[mock-engine] {msg}", file=sys.stderr, flush=True)


class MockEngine:
    def __init__(self, sock_path: str, guardian_path: Path, speed: float = 1.0,
                 cycle: bool = True, exit_on_quit: bool = True) -> None:
        self.sock_path = sock_path
        self.guardian_path = guardian_path
        self.speed = max(0.05, speed)
        self.cycle = cycle
        self.exit_on_quit = exit_on_quit
        self.lock = threading.Lock()
        self.clients: list[socket.socket] = []
        self.stop = threading.Event()
        self.paused = False
        self.mode = self._load_mode()
        self.base = {"mood": "watching", "masks": 0, "pending": 0, "reason": "", "warning": None}
        self.server: socket.socket | None = None
        self.received: list[dict] = []

    # guardian.json -----------------------------------------------------------

    def _guardian(self) -> dict:
        try:
            data = json.loads(self.guardian_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _load_mode(self) -> str:
        mode = self._guardian().get("mode")
        return mode if mode in MODES else "aggressive"

    def _save_mode(self) -> None:
        data = self._guardian()
        data["mode"] = self.mode
        self.guardian_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.guardian_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.guardian_path)

    def auth_ok(self, token: object) -> bool:
        want = self._guardian().get("token_sha256")
        if not isinstance(token, str) or not isinstance(want, str) or not token:
            return False
        got = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return hmac.compare_digest(got, want)

    # state --------------------------------------------------------------------

    def state(self) -> dict:
        with self.lock:
            s = dict(self.base)
            if self.mode == "off":
                status, s["mood"], s["masks"], s["pending"] = "OFF", "sleep", 0, 0
            elif self.paused:
                status, s["mood"], s["masks"], s["pending"] = "PAUSED", "sleep", 0, 0
            else:
                status = "PROTECTING"
            return {"type": "state", "t": round(time.time(), 2), "status": status, "mood": s["mood"],
                    "mode": self.mode, "masks": s["masks"], "pending": s["pending"],
                    "backend": "local", "fps": 11.8, "reason": s["reason"], "warning": s.get("warning")}

    def send(self, conn: socket.socket, obj: dict) -> None:
        try:
            conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))
        except OSError:
            self._drop(conn)

    def broadcast(self, obj: dict) -> None:
        with self.lock:
            clients = list(self.clients)
        for c in clients:
            self.send(c, obj)

    def _drop(self, conn: socket.socket) -> None:
        with self.lock:
            if conn in self.clients:
                self.clients.remove(conn)
        try:
            conn.close()
        except OSError:
            pass

    # protocol -------------------------------------------------------------------

    def handle(self, conn: socket.socket, msg: dict) -> None:
        kind = msg.get("type")
        self.received.append({k: v for k, v in msg.items() if k != "auth"})
        log(f"<- {kind}" + (" (auth ok)" if kind in ("set_mode", "pause", "quit") and self.auth_ok(msg.get("auth")) else ""))
        if kind in ("set_mode", "pause", "quit") and not self.auth_ok(msg.get("auth")):
            self.send(conn, {"type": "error", "message": "guardian auth required"})
            return
        if kind == "set_mode":
            mode = msg.get("mode")
            if mode not in MODES:
                self.send(conn, {"type": "error", "message": f"unknown mode {mode!r}"})
                return
            with self.lock:
                self.mode = mode
            self._save_mode()
            self.broadcast(self.state())
        elif kind == "pause":
            with self.lock:
                self.paused = True
            self.broadcast(self.state())
        elif kind == "resume":
            with self.lock:
                self.paused = False
            self.broadcast(self.state())
        elif kind == "quit":
            log("quit requested by guardian")
            if self.exit_on_quit:
                self.shutdown()
        elif kind == "ping":
            self.send(conn, self.state())

    def _reader(self, conn: socket.socket) -> None:
        buf = b""
        try:
            while not self.stop.is_set():
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(msg, dict):
                        self.handle(conn, msg)
        except OSError:
            pass
        finally:
            self._drop(conn)
            log("island disconnected")

    def _accept(self) -> None:
        assert self.server is not None
        while not self.stop.is_set():
            try:
                conn, _ = self.server.accept()
            except OSError:
                break
            log("island connected")
            with self.lock:
                self.clients.append(conn)
            self.send(conn, {"type": "hello", "version": 1, "engine": "screen-guardian",
                             "mode": self.mode, "opencv": "5.0.0-mock"})
            self.send(conn, self.state())
            threading.Thread(target=self._reader, args=(conn,), daemon=True).start()

    def _script(self) -> None:
        i = 0
        while not self.stop.is_set():
            secs, fields, event = SCRIPT[i % len(SCRIPT)]
            if self.cycle:
                with self.lock:
                    self.base.update(fields)
                if event and not self.paused and self.mode != "off":
                    self.broadcast(event)
                self.broadcast(self.state())
            end = time.time() + secs / self.speed
            while time.time() < end and not self.stop.is_set():
                time.sleep(0.05)
            i += 1

    def _heartbeat(self) -> None:
        while not self.stop.wait(1.0):
            self.broadcast(self.state())

    def start(self) -> None:
        path = Path(self.sock_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            path.unlink()
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(path))
        os.chmod(path, 0o600)
        srv.listen(4)
        self.server = srv
        for target in (self._accept, self._script, self._heartbeat):
            threading.Thread(target=target, daemon=True).start()
        log(f"listening on {path} (mode {self.mode})")

    def shutdown(self) -> None:
        self.stop.set()
        if self.server is not None:
            try:
                self.server.close()
            except OSError:
                pass
        with self.lock:
            clients, self.clients = list(self.clients), []
        for c in clients:
            try:
                c.close()
            except OSError:
                pass
        try:
            Path(self.sock_path).unlink()
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sock", default=default_sock(), help="socket path (default: SG_ISLAND_SOCK or support dir)")
    ap.add_argument("--guardian", default=None, help="guardian.json path (default: support dir)")
    ap.add_argument("--speed", type=float, default=1.0, help="script speed multiplier")
    ap.add_argument("--static", action="store_true", help="do not cycle moods")
    ap.add_argument("--duration", type=float, default=0, help="exit after N seconds (0 = run forever)")
    ap.add_argument("--keep-on-quit", action="store_true", help="ignore guardian quit (keeps serving)")
    a = ap.parse_args(argv)
    guardian = Path(a.guardian) if a.guardian else support_dir() / "guardian.json"
    eng = MockEngine(a.sock, guardian, speed=a.speed, cycle=not a.static, exit_on_quit=not a.keep_on_quit)
    eng.start()
    signal.signal(signal.SIGTERM, lambda *_: eng.stop.set())
    t0 = time.time()
    try:
        while not eng.stop.is_set():
            if a.duration and time.time() - t0 > a.duration:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        eng.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
