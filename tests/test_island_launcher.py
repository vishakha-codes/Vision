"""Island launcher + mock engine. Runs without a built GuardianIsland binary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import socket
import stat
import sys
import tempfile
import time
from pathlib import Path

import pytest

from edge import island_launcher
from edge.island_launcher import start_island, stop_island

VISION = Path(__file__).resolve().parents[1]
MOCK_PATH = VISION / "island" / "tools" / "mock_engine.py"


def _short_tmp() -> Path:
    # AF_UNIX paths are limited to ~104 bytes, pytest's tmp_path is often longer.
    return Path(tempfile.mkdtemp(prefix="sgi", dir="/tmp"))


def test_missing_binary_returns_none_with_hint(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SG_ISLAND_BIN", str(tmp_path / "nope" / "GuardianIsland"))
    assert start_island("/tmp/x.sock") is None
    err = capsys.readouterr().err
    assert "swift build -c release" in err


def test_default_binary_location():
    monkeypatch_free = island_launcher.DEFAULT_BINARY
    assert monkeypatch_free.parts[-3:] == (".build", "release", "GuardianIsland")
    assert monkeypatch_free.parents[2].name == "island"


@pytest.mark.skipif(sys.platform == "win32", reason="posix shell script")
def test_start_passes_socket_env_and_stop_terminates(monkeypatch, tmp_path):
    out = tmp_path / "env.txt"
    fake = tmp_path / "GuardianIsland"
    fake.write_text(f'#!/bin/sh\necho "$SG_ISLAND_SOCK" > "{out}"\nexec sleep 30\n')
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("SG_ISLAND_BIN", str(fake))

    proc = start_island("/tmp/sg-test.sock")
    assert proc is not None
    try:
        for _ in range(100):
            if out.exists() and out.read_text().strip():
                break
            time.sleep(0.02)
        assert out.read_text().strip() == "/tmp/sg-test.sock"
        assert proc.poll() is None
    finally:
        stop_island(proc, timeout=2.0)
    assert proc.poll() is not None


def test_stop_island_tolerates_none_and_exited():
    stop_island(None)

    class Done:
        def poll(self):
            return 0

    stop_island(Done())  # type: ignore[arg-type]


# --- mock engine -----------------------------------------------------------------


def _load_mock():
    spec = importlib.util.spec_from_file_location("sg_mock_engine", MOCK_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _Client:
    def __init__(self, path: str) -> None:
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(3)
        self.s.connect(path)
        self.buf = b""

    def send(self, obj: dict) -> None:
        self.s.sendall((json.dumps(obj) + "\n").encode())

    def next(self, pred) -> dict:
        deadline = time.time() + 3
        while time.time() < deadline:
            while b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                msg = json.loads(line)
                if pred(msg):
                    return msg
            self.buf += self.s.recv(4096)
        raise AssertionError("message not received")

    def close(self) -> None:
        self.s.close()


@pytest.fixture()
def mock_engine():
    mod = _load_mock()
    d = _short_tmp()
    token = "ab" * 32
    guardian = d / "guardian.json"
    guardian.write_text(json.dumps({"token_sha256": hashlib.sha256(token.encode()).hexdigest()}))
    eng = mod.MockEngine(str(d / "i.sock"), guardian, speed=1.0, cycle=False, exit_on_quit=True)
    eng.start()
    yield eng, token, guardian
    eng.shutdown()
    shutil.rmtree(d, ignore_errors=True)


def test_mock_engine_hello_and_state(mock_engine):
    eng, _, _ = mock_engine
    c = _Client(eng.sock_path)
    try:
        hello = c.next(lambda m: m["type"] == "hello")
        assert hello["version"] == 1 and hello["mode"] == "aggressive"
        st = c.next(lambda m: m["type"] == "state")
        assert st["status"] == "PROTECTING"
        assert {"mood", "mode", "masks", "pending", "backend", "fps", "reason"} <= st.keys()
    finally:
        c.close()


def test_mock_engine_rejects_guardian_actions_without_auth(mock_engine):
    eng, token, guardian = mock_engine
    c = _Client(eng.sock_path)
    try:
        c.next(lambda m: m["type"] == "hello")
        c.send({"type": "set_mode", "mode": "medium"})
        err = c.next(lambda m: m["type"] == "error")
        assert err["message"] == "guardian auth required"
        c.send({"type": "set_mode", "mode": "medium", "auth": "wrong"})
        c.next(lambda m: m["type"] == "error")

        c.send({"type": "set_mode", "mode": "medium", "auth": token})
        st = c.next(lambda m: m["type"] == "state" and m["mode"] == "medium")
        assert st["status"] == "PROTECTING"
        assert json.loads(guardian.read_text())["mode"] == "medium"

        c.send({"type": "pause", "auth": token})
        c.next(lambda m: m["type"] == "state" and m["status"] == "PAUSED")
        c.send({"type": "resume"})  # resume needs no auth
        c.next(lambda m: m["type"] == "state" and m["status"] == "PROTECTING")
    finally:
        c.close()


def test_mock_engine_quit_requires_auth(mock_engine):
    eng, token, _ = mock_engine
    c = _Client(eng.sock_path)
    try:
        c.next(lambda m: m["type"] == "hello")
        c.send({"type": "quit"})
        c.next(lambda m: m["type"] == "error")
        assert not eng.stop.is_set()
        c.send({"type": "quit", "auth": token})
        for _ in range(100):
            if eng.stop.is_set():
                break
            time.sleep(0.02)
        assert eng.stop.is_set()
    finally:
        c.close()
