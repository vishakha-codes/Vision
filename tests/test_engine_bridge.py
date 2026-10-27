"""Island bridge round trip over a temp Unix socket + temp guardian.json."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import stat
import tempfile
import time
from pathlib import Path

import pytest

from edge.island_bridge import AUTH_ERROR, GuardianStore, IslandBridge, mood_for
from vision_core.live.contracts import ProtectionMode

TOKEN = "test-token-not-a-secret"


@pytest.fixture()
def tmpdir_short():
    # AF_UNIX paths are capped at 104 bytes on macOS; pytest tmp_path is long.
    path = Path(tempfile.mkdtemp(prefix="sg-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class _Client:
    def __init__(self, path: Path) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(2.0)
        self.sock.connect(str(path))
        self.buf = b""

    def send(self, msg: dict) -> None:
        self.sock.sendall((json.dumps(msg) + "\n").encode())

    def recv(self, want_type: str | None = None, timeout: float = 2.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            while b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                msg = json.loads(line)
                if want_type is None or msg.get("type") == want_type:
                    return msg
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            self.buf += chunk
        raise AssertionError(f"no {want_type!r} message")

    def close(self) -> None:
        self.sock.close()


def _setup(tmp: Path, *, with_token: bool = True):
    gpath = tmp / "guardian.json"
    if with_token:
        gpath.write_text(json.dumps({"token_sha256": hashlib.sha256(TOKEN.encode()).hexdigest()}))
    store = GuardianStore(gpath)
    calls: dict[str, list] = {"mode": [], "pause": [], "resume": [], "quit": []}
    bridge = IslandBridge(
        sock_path=tmp / "island.sock",
        guardian=store,
        on_set_mode=lambda m: calls["mode"].append(m),
        on_pause=lambda: calls["pause"].append(1),
        on_resume=lambda: calls["resume"].append(1),
        on_quit=lambda: calls["quit"].append(1),
        hello={"mode": store.resolve_mode(None).value, "opencv": "5.0.0"},
    )
    assert bridge.start(), bridge.error
    return bridge, store, calls, gpath


def _wait(pred, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_bridge_round_trip_auth_and_mode_persist(tmpdir_short: Path) -> None:
    bridge, store, calls, gpath = _setup(tmpdir_short)
    try:
        sock_mode = stat.S_IMODE(os.stat(tmpdir_short / "island.sock").st_mode)
        assert sock_mode == 0o600
        bridge.publish_state({"status": "PROTECTING", "mood": "watching", "mode": "aggressive", "masks": 0})
        client = _Client(tmpdir_short / "island.sock")
        hello = client.recv("hello")
        assert hello["version"] == 1 and hello["engine"] == "screen-guardian"
        assert hello["mode"] == "aggressive"  # first-run default, now persisted
        assert json.loads(gpath.read_text())["mode"] == "aggressive"
        state = client.recv("state")
        assert state["mood"] == "watching" and "t" in state

        client.send({"type": "ping"})
        assert client.recv("pong")

        # Rejected: no auth, wrong auth.
        client.send({"type": "set_mode", "mode": "medium"})
        assert client.recv("error")["message"] == AUTH_ERROR
        client.send({"type": "quit", "auth": "wrong"})
        assert client.recv("error")["message"] == AUTH_ERROR
        assert calls["mode"] == [] and calls["quit"] == []
        assert store.mode() is ProtectionMode.AGGRESSIVE

        # Accepted.
        client.send({"type": "set_mode", "mode": "medium", "auth": TOKEN})
        assert _wait(lambda: calls["mode"] == [ProtectionMode.MEDIUM])
        assert store.mode() is ProtectionMode.MEDIUM
        data = json.loads(gpath.read_text())
        assert data["mode"] == "medium" and data["token_sha256"]  # token kept
        assert stat.S_IMODE(os.stat(gpath).st_mode) == 0o600

        client.send({"type": "set_mode", "mode": "nonsense", "auth": TOKEN})
        assert client.recv("error")["message"] == "unknown mode"

        client.send({"type": "pause", "auth": TOKEN})
        client.send({"type": "resume"})  # not a guardian action
        client.send({"type": "future_thing", "x": 1})  # ignored
        assert _wait(lambda: calls["pause"] == [1] and calls["resume"] == [1])

        # State change is pushed; events are broadcast.
        bridge.publish_state({"status": "PROTECTING", "mood": "covering", "mode": "medium", "masks": 2})
        assert client.recv("state")["mood"] == "covering"
        bridge.send_event("covered", count=2)
        ev = client.recv("event")
        assert ev == {"type": "event", "kind": "covered", "count": 2}

        client.send({"type": "quit", "auth": TOKEN})
        assert _wait(lambda: calls["quit"] == [1])
        client.close()
    finally:
        bridge.stop()
    assert not (tmpdir_short / "island.sock").exists()


def test_explicit_cli_mode_wins_warns_and_saves(tmpdir_short: Path) -> None:
    store = GuardianStore(tmpdir_short / "guardian.json")
    warnings: list[str] = []
    assert store.resolve_mode("off", warn=warnings.append) is ProtectionMode.OFF  # first run uses CLI
    assert warnings == []  # nothing saved yet: no conflict to warn about
    assert store.mode() is ProtectionMode.OFF
    # An explicit --mode beats the saved one, warns, and is saved.
    assert store.resolve_mode("medium", warn=warnings.append) is ProtectionMode.MEDIUM
    assert len(warnings) == 1 and "medium" in warnings[0] and "off" in warnings[0]
    assert store.mode() is ProtectionMode.MEDIUM
    # Same mode again: no warning.
    assert store.resolve_mode("medium", warn=warnings.append) is ProtectionMode.MEDIUM
    assert len(warnings) == 1
    # No CLI flag: the saved mode applies.
    assert store.resolve_mode(None, warn=warnings.append) is ProtectionMode.MEDIUM
    assert len(warnings) == 1
    store2 = GuardianStore(tmpdir_short / "other.json")
    assert store2.resolve_mode(None) is ProtectionMode.AGGRESSIVE
    assert store2.mode() is ProtectionMode.AGGRESSIVE


def test_cli_mode_keeps_guardian_token(tmpdir_short: Path) -> None:
    path = tmpdir_short / "guardian.json"
    path.write_text(json.dumps({"token_sha256": "ab" * 32, "mode": "aggressive"}))
    store = GuardianStore(path)
    assert store.resolve_mode("medium", warn=lambda _m: None) is ProtectionMode.MEDIUM
    data = json.loads(path.read_text())
    assert data == {"token_sha256": "ab" * 32, "mode": "medium"}


def test_app_mode_flag_defaults_to_none() -> None:
    """``--mode`` must default to None so an explicit flag can be told apart."""
    import edge.app as app
    import edge.live_engine as le

    seen = {}

    def fake_run_live(**kw):
        seen.update(kw)
        return {"frames": 0, "mode": "aggressive", "opencv": "5"}

    orig = le.run_live
    le.run_live = fake_run_live
    try:
        assert app.main(["--source", "live"]) == 0
        assert seen["mode"] is None
        assert app.main(["--source", "live", "--mode", "medium"]) == 0
        assert seen["mode"] == "medium"
    finally:
        le.run_live = orig


def test_no_token_rejects_guardian_actions(tmpdir_short: Path) -> None:
    bridge, _store, calls, _g = _setup(tmpdir_short, with_token=False)
    try:
        client = _Client(tmpdir_short / "island.sock")
        client.recv("hello")
        client.send({"type": "quit", "auth": ""})
        assert client.recv("error")["message"] == AUTH_ERROR
        client.send({"type": "quit", "auth": TOKEN})
        assert client.recv("error")["message"] == AUTH_ERROR
        assert calls["quit"] == []
        client.close()
    finally:
        bridge.stop()


def test_state_is_rate_limited(tmpdir_short: Path) -> None:
    bridge, _store, _calls, _g = _setup(tmpdir_short)
    try:
        client = _Client(tmpdir_short / "island.sock")
        client.recv("hello")
        t0 = time.monotonic()
        for i in range(200):
            bridge.publish_state({"status": "PROTECTING", "mood": "watching", "masks": i})
            time.sleep(0.002)
        time.sleep(0.25)
        elapsed = time.monotonic() - t0
        client.sock.settimeout(0.2)
        n = 0
        last = None
        try:
            while True:
                msg = client.recv("state", timeout=0.3)
                n += 1
                last = msg
        except AssertionError:
            pass
        assert 1 <= n <= int(elapsed * 10) + 2, (n, elapsed)
        assert last is not None and last["masks"] == 199  # latest wins
        client.close()
    finally:
        bridge.stop()


def test_stale_socket_is_replaced(tmpdir_short: Path) -> None:
    path = tmpdir_short / "island.sock"
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(path))
    s.close()  # leaves a dead socket file behind
    bridge = IslandBridge(sock_path=path, guardian=GuardianStore(tmpdir_short / "g.json"))
    try:
        assert bridge.start(), bridge.error
        client = _Client(path)
        assert client.recv("hello")
        client.close()
    finally:
        bridge.stop()


def test_mood_mapping() -> None:
    base = dict(capture_ok=True, mode=ProtectionMode.AGGRESSIVE, paused=False, started=True,
                stale=0, unsafe=0, unsure=0, pending=0, alert=False)
    assert mood_for(**base) == ("PROTECTING", "watching")
    assert mood_for(**{**base, "mode": ProtectionMode.OFF}) == ("OFF", "sleep")
    assert mood_for(**{**base, "paused": True}) == ("PAUSED", "idle")
    assert mood_for(**{**base, "capture_ok": False}) == ("NOT_PROTECTING", "lost")
    assert mood_for(**{**base, "stale": 1}) == ("PROTECTING", "working")
    assert mood_for(**{**base, "unsafe": 1, "alert": True})[1] == "alert"
    assert mood_for(**{**base, "unsafe": 1})[1] == "covering"
    assert mood_for(**{**base, "unsure": 1})[1] == "unsure"
    assert mood_for(**{**base, "pending": 1})[1] == "working"
