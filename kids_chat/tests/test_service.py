from __future__ import annotations

import json
import os
import socket
import threading
import time

import pytest

from kids_chat.config import ConfigStore, check_guardian_auth
from kids_chat.service import ChatService, Conn, Server
from kids_chat.snapshot import FakeSnapshotProvider, UnavailableSnapshotProvider
from kids_chat.tests.conftest import TOKEN


class ScriptBackend:
    name = "fake"

    def __init__(self, events=None):
        self.events = events
        self.payloads = []

    def turn(self, payload, cancel=None):
        self.payloads.append(payload)
        if payload.get("action") == "parent_summary":
            yield {"type": "summary", "bullets": ["Asked about space"]}
            yield {"type": "done", "blocked": False}
            return
        for ev in self.events if self.events is not None else [{"type": "delta", "text": "Hi "}, {"type": "delta", "text": "there!"},
                                                                {"type": "done", "blocked": False, "backend": "agentcore"}]:
            yield ev


def svc(home, backend=None, snapshots=None):
    out = []
    s = ChatService(store=ConfigStore(home), backend=backend or ScriptBackend(),
                    snapshots=snapshots or UnavailableSnapshotProvider(), sync=True)
    conn = Conn(sink=out.append)
    s.conns.add(conn)
    return s, conn, out


def enable(s, conn, **kw):
    s.handle({"type": "chat.config", "auth": TOKEN, "enabled": True, **kw}, conn)


def test_chat_is_off_by_default(home):
    s, conn, out = svc(home)
    s.handle({"type": "hello"}, conn)
    assert out[0]["type"] == "chat.state" and out[0]["enabled"] is False and out[0]["transcripts"] is False
    s.handle({"type": "chat.send", "id": "c1", "text": "hi"}, conn)
    assert out[-1] == {"type": "chat.error", "id": "c1", "code": "chat_disabled", "message": "Ask a grown-up to turn on chat"}
    cfg = json.loads((home / "chat.json").read_text())
    assert cfg["enabled"] is False and len(cfg["profile_id"]) == 36
    assert oct(os.stat(home / "chat.json").st_mode & 0o777) == "0o600"


def test_config_requires_guardian_token(home):
    s, conn, out = svc(home)
    s.handle({"type": "chat.config", "auth": "wrong", "enabled": True}, conn)
    assert out[-1]["code"] == "auth_required"
    s.handle({"type": "chat.config", "enabled": True}, conn)
    assert out[-1]["code"] == "auth_required"
    assert ConfigStore(home).load().enabled is False
    enable(s, conn)
    assert out[-1]["type"] == "chat.state" and out[-1]["enabled"] is True and out[-1]["age_band"] == "kids"


def test_auth_matches_island_scheme(home):
    assert check_guardian_auth(TOKEN, home / "guardian.json")
    assert not check_guardian_auth(TOKEN.upper(), home / "guardian.json")
    assert not check_guardian_auth("", home / "guardian.json")


def test_band_follows_protection_mode_and_off_disables(home):
    s, conn, out = svc(home)
    enable(s, conn)
    g = json.loads((home / "guardian.json").read_text())
    (home / "guardian.json").write_text(json.dumps({**g, "mode": "medium"}))
    s.handle({"type": "chat.state"}, conn)
    assert out[-1]["age_band"] == "teens"
    (home / "guardian.json").write_text(json.dumps({**g, "mode": "off"}))
    s.handle({"type": "chat.state"}, conn)
    assert out[-1]["enabled"] is False and out[-1]["age_band"] == "off"
    s.handle({"type": "chat.send", "id": "c1", "text": "hi"}, conn)
    assert out[-1]["code"] == "chat_disabled"


def test_explicit_band_override(home):
    s, conn, out = svc(home)
    enable(s, conn, age_band="teens")
    assert out[-1]["age_band"] == "teens"


def test_send_streams_and_counts(home):
    b = ScriptBackend()
    s, conn, out = svc(home, b)
    enable(s, conn)
    out.clear()
    s.handle({"type": "chat.send", "id": "c7", "text": "  why is the sky blue?  ", "name": "Gomi"}, conn)
    assert [o["type"] for o in out] == ["chat.delta", "chat.delta", "chat.done"]
    assert out[-1] == {"type": "chat.done", "id": "c7", "blocked": False, "backend": "agentcore"}
    p = b.payloads[0]
    assert p["prompt"] == "why is the sky blue?" and p["age_band"] == "kids" and p["name"] == "Gomi"
    assert p["actor_id"] == ConfigStore(home).load().profile_id and p["session_id"].startswith(p["actor_id"] + "-")
    assert len(p["session_id"]) >= 33
    assert s.stats.turns_today() == 1
    assert not (home / "chat-transcripts").exists()  # transcripts are opt-in


def test_blocked_turn_counted_without_text(home):
    b = ScriptBackend([{"type": "done", "blocked": True, "blocked_by": "self_harm", "safety": True, "text": "kind reply"}])
    s, conn, out = svc(home, b)
    enable(s, conn)
    s.handle({"type": "chat.send", "id": "c1", "text": "sad"}, conn)
    assert out[-1]["blocked"] is True and out[-1]["text"] == "kind reply" and out[-1]["safety"] is True
    day = s.stats.day()
    assert day["blocked"] == {"self_harm": 1} and day["safety_topics"] == 1
    assert "sad" not in (home / "chat-stats.json").read_text()


def test_transcripts_opt_in(home):
    s, conn, out = svc(home)
    enable(s, conn, transcripts=True)
    assert out[-1]["transcripts"] is True
    s.handle({"type": "chat.send", "id": "c1", "text": "hello"}, conn)
    files = list((home / "chat-transcripts").glob("*.jsonl"))
    lines = [json.loads(x) for x in files[0].read_text().splitlines()]
    assert [x["role"] for x in lines] == ["child", "helper"] and lines[1]["text"] == "Hi there!"
    assert oct(os.stat(files[0]).st_mode & 0o777) == "0o600"


def test_limits(home):
    s, conn, out = svc(home)
    enable(s, conn, daily_turns=1)
    s.handle({"type": "chat.send", "id": "c1", "text": "x" * 501}, conn)
    assert out[-1]["code"] == "too_long"
    s.handle({"type": "chat.send", "id": "c2", "text": "hi"}, conn)
    s.handle({"type": "chat.send", "id": "c3", "text": "hi"}, conn)
    assert out[-1]["code"] == "daily_limit"


def test_backend_error_becomes_friendly_offline(home):
    s, conn, out = svc(home, ScriptBackend([{"type": "error", "code": "internal", "message": "boom"}]))
    enable(s, conn)
    s.handle({"type": "chat.send", "id": "c1", "text": "hi"}, conn)
    assert out[-1]["type"] == "chat.error" and out[-1]["code"] == "offline" and "boom" not in out[-1]["message"]


def test_summary_needs_auth_and_has_no_text(home):
    s, conn, out = svc(home)
    enable(s, conn)
    s.handle({"type": "chat.summary"}, conn)
    assert out[-1]["code"] == "auth_required"
    s.handle({"type": "chat.summary", "auth": TOKEN}, conn)
    assert out[-1]["type"] == "chat.summary" and out[-1]["bullets"] == ["Asked about space"]
    assert out[-1]["counts"]["turns"] == 0


def test_new_profile_forgets(home):
    s, conn, out = svc(home)
    old = ConfigStore(home).load().profile_id
    s.handle({"type": "chat.config", "auth": TOKEN, "new_profile": True}, conn)
    assert ConfigStore(home).load().profile_id != old


# ---------------------------------------------------------------- explain

def test_explain_needs_parent_opt_in(home):
    s, conn, out = svc(home, snapshots=FakeSnapshotProvider())
    enable(s, conn)
    s.handle({"type": "chat.send", "id": "c1", "text": "how does this work?", "explain": True}, conn)
    assert out[-1]["code"] == "explain_disabled"


def test_explain_fails_closed_without_engine(home):
    s, conn, out = svc(home)
    enable(s, conn, explain_screen=True)
    assert out[-1]["explain_available"] is False
    s.handle({"type": "chat.send", "id": "c1", "text": "how does this work?", "explain": True}, conn)
    assert out[-1]["code"] == "explain_unavailable"


def test_explain_refuses_unpainted_masks(home):
    s, conn, out = svc(home, snapshots=FakeSnapshotProvider(paint_masks=False))
    enable(s, conn, explain_screen=True)
    s.handle({"type": "chat.send", "id": "c1", "text": "how does this work?", "explain": True}, conn)
    assert out[-1]["code"] == "refused_unsafe_screen" and out[-1]["reason"] == "mask_not_painted"


def test_explain_sends_redacted_frame_and_maps_draws(home):
    b = ScriptBackend([{"type": "draw", "space": "norm1000", "ops": [
        {"kind": "pointer", "at": [500, 500], "label": "pivot", "color": "yellow"},
        {"kind": "box", "rect": [720, 120, 100, 100], "label": "x", "color": "yellow"}]},  # on the mask → dropped
        {"type": "delta", "text": "See the arrow."}, {"type": "done", "blocked": False}])
    s, conn, out = svc(home, b, FakeSnapshotProvider(display={"x": 0, "y": 0, "w": 1440, "h": 900}))
    enable(s, conn, explain_screen=True)
    s.handle({"type": "chat.send", "id": "c1", "text": "how does this work?", "explain": True}, conn)
    ex = b.payloads[0]["explain"]
    assert len(ex["images"]) == 1 and ex["w"] == 1280 and ex["masks"] == [[703.1, 100.0, 921.9, 375.0]]
    draw = next(o for o in out if o["type"] == "chat.draw")
    assert draw["space"] == "quartz_pt" and draw["replace"] is True
    assert draw["ops"] == [{"kind": "pointer", "at": [720.0, 450.0], "label": "pivot", "color": "yellow", "seq": 0}]
    assert draw["avoid"] == [[1012.5, 90.0, 315.1, 247.5]]
    assert s.stats.day()["explain"] == 1


def test_motion_question_asks_for_three_frames(home):
    b = ScriptBackend()
    s, conn, out = svc(home, b, FakeSnapshotProvider())
    enable(s, conn, explain_screen=True)
    s.handle({"type": "chat.send", "id": "c1", "text": "how does this machine move?", "explain": True}, conn)
    assert len(b.payloads[0]["explain"]["images"]) == 3


# ---------------------------------------------------------------- real socket

def test_socket_roundtrip(home, tmp_path_factory):
    sock_dir = tmp_path_factory.mktemp("s")
    path = sock_dir / "c.sock"
    s = ChatService(store=ConfigStore(home), backend=ScriptBackend(), snapshots=UnavailableSnapshotProvider())
    srv = Server(s, path)
    srv.start()
    try:
        assert oct(os.stat(path).st_mode & 0o777) == "0o600"
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.settimeout(3)
        c.connect(str(path))
        f = c.makefile("rwb")
        first = json.loads(f.readline())
        assert first["type"] == "chat.state" and first["enabled"] is False
        for m in ({"type": "chat.config", "auth": TOKEN, "enabled": True}, {"type": "chat.send", "id": "c1", "text": "hi"}):
            f.write((json.dumps(m) + "\n").encode())
            f.flush()
        seen = []
        deadline = time.time() + 3
        while time.time() < deadline:
            seen.append(json.loads(f.readline()))
            if seen[-1]["type"] == "chat.done":
                break
        assert [x["type"] for x in seen] == ["chat.state", "chat.delta", "chat.delta", "chat.done"]
        c.close()
    finally:
        srv.stop()
    assert not path.exists()


def test_refuses_second_server(home, tmp_path_factory):
    path = tmp_path_factory.mktemp("s2") / "c.sock"
    s = ChatService(store=ConfigStore(home), backend=ScriptBackend(), snapshots=UnavailableSnapshotProvider())
    a = Server(s, path)
    a.start()
    try:
        with pytest.raises(RuntimeError):
            Server(s, path).start()
    finally:
        a.stop()


def test_cancel_stops_stream(home):
    gate = threading.Event()

    class Slow(ScriptBackend):
        def turn(self, payload, cancel=None):
            yield {"type": "delta", "text": "a"}
            gate.wait(2)
            yield {"type": "delta", "text": "b"}
            yield {"type": "done", "blocked": False}

    out = []
    s = ChatService(store=ConfigStore(home), backend=Slow(), snapshots=UnavailableSnapshotProvider())
    conn = Conn(sink=out.append)
    s.handle({"type": "chat.config", "auth": TOKEN, "enabled": True}, conn)
    s.handle({"type": "chat.send", "id": "c1", "text": "hi"}, conn)
    time.sleep(0.1)
    s.handle({"type": "chat.send", "id": "c2", "text": "again"}, conn)
    assert out[-1]["code"] == "busy"
    s.handle({"type": "chat.cancel", "id": "c1"}, conn)
    gate.set()
    deadline = time.time() + 2
    while time.time() < deadline and not any(o.get("cancelled") for o in out):
        time.sleep(0.02)
    assert any(o.get("type") == "chat.done" and o.get("cancelled") for o in out)
    assert not any(o.get("text") == "b" for o in out)


def test_launcher_starts_and_stops_service(home, monkeypatch):
    import tempfile
    from kids_chat import launcher
    d = tempfile.mkdtemp(prefix="sgc", dir="/private/tmp" if os.path.isdir("/private/tmp") else None)
    monkeypatch.setenv("SG_CHAT_SOCK", os.path.join(d, "c.sock"))
    proc = launcher.start_chat_service()
    try:
        deadline = time.time() + 10
        while time.time() < deadline and not launcher._listening(launcher.socket_path()):
            time.sleep(0.1)
        assert launcher._listening(launcher.socket_path())
        assert launcher.start_chat_service() is None  # already running
    finally:
        launcher.stop_chat_service(proc)
    assert proc.poll() is not None
