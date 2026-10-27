"""Snapshot verification, draw mapping, backends, the runtime HTTP/SSE server, and package isolation."""

from __future__ import annotations

import ast
import base64
import http.client
import io
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from kids_chat import draw
from kids_chat.agent import main as agent_main
from kids_chat.backends import AgentCoreBackend, BackendUnavailable, ChainBackend, parse_sse
from kids_chat.snapshot import FakeSnapshotProvider, SnapshotRefused, verify
from kids_chat.tests.conftest import FakeBedrock, chat_payload, make_deps, text_stream

PKG = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- snapshot

def test_fake_snapshot_verifies():
    snap = FakeSnapshotProvider().get()
    assert snap.w == 1280 and snap.h == 800 and len(snap.frames) == 1 and snap.masks_norm


@pytest.mark.parametrize("mutate,reason", [
    (lambda r: r.update(ok=False, reason="not_protecting"), "not_protecting"),
    (lambda r: r.update(all_checked=False), "unchecked_regions"),
    (lambda r: r.update(w=999), "size_mismatch"),
    (lambda r: r.update(frames=[]), "malformed"),
    (lambda r: r.update(masks=[[10, 10, 5, 5]]), "malformed"),
    (lambda r: r["frames"][0].update(jpeg_b64=base64.b64encode(b"\x89PNG....").decode()), "bad_image"),
    (lambda r: r.pop("display"), "malformed"),
])
def test_snapshot_fails_closed(mutate, reason):
    r = FakeSnapshotProvider().response()
    mutate(r)
    with pytest.raises(SnapshotRefused) as e:
        verify(r)
    assert e.value.reason == reason


def test_unpainted_mask_is_refused():
    with pytest.raises(SnapshotRefused) as e:
        FakeSnapshotProvider(paint_masks=False).get()
    assert e.value.reason == "mask_not_painted"


# ---------------------------------------------------------------- draw mapping

DISP = {"x": 100.0, "y": 50.0, "w": 1000.0, "h": 500.0}


def test_mapping_scales_to_quartz_points():
    ops = [{"kind": "pointer", "at": [0, 0]}, {"kind": "pointer", "at": [1000, 1000]},
           {"kind": "box", "rect": [100, 100, 200, 400]}, {"kind": "circle", "center": [500, 500], "r": 50},
           {"kind": "arrow", "from": [0, 500], "to": [1000, 500]}, {"kind": "label", "at": [10, 20], "text": "hi"},
           {"kind": "clear"}]
    out = draw.to_screen(ops, DISP)
    assert out[0]["at"] == [100.0, 50.0] and out[1]["at"] == [1100.0, 550.0]
    assert out[2]["rect"] == [200.0, 100.0, 200.0, 200.0]  # top-left origin, Quartz y-down
    assert out[3]["center"] == [600.0, 300.0] and out[3]["r"] == 50.0
    assert out[4]["from"] == [100.0, 300.0] and out[4]["to"] == [1100.0, 300.0] and out[4]["bow"] == 0.2
    assert out[5]["at"] == [110.0, 60.0] and out[6] == {"kind": "clear", "seq": 6}
    assert [o["seq"] for o in out] == list(range(7))


def test_mapping_drops_invalid_and_masked():
    ops = [{"kind": "pointer", "at": [1500, 0]}, {"kind": "teleport"}, "junk",
           {"kind": "arrow", "from": [0, 500], "to": [1000, 500]},  # crosses the mask
           {"kind": "pointer", "at": [50, 50], "label": "x" * 99}]
    out = draw.to_screen(ops, DISP, masks_norm=[[400, 400, 600, 600]])
    assert len(out) == 1 and len(out[0]["label"]) == 32


def test_mapping_caps_ops():
    out = draw.to_screen([{"kind": "pointer", "at": [i, i]} for i in range(20)], DISP)
    assert len(out) == 8


# ---------------------------------------------------------------- backends

def test_parse_sse_skips_keepalives_and_junk():
    lines = [b": keepalive", b"", b'data: {"type":"delta","text":"a"}', b"data: not json", b'{"type":"done"}']
    assert [e["type"] for e in parse_sse(lines)] == ["delta", "done"]


class FakeBody:
    def __init__(self, lines):
        self.lines = lines
        self.closed = False

    def iter_lines(self):
        return iter(self.lines)

    def read(self):
        return b"\n".join(self.lines)

    def close(self):
        self.closed = True


class FakeAgentCore:
    def __init__(self, lines=None, error=None):
        self.lines, self.error, self.kw = lines or [], error, None

    def invoke_agent_runtime(self, **kw):
        self.kw = kw
        if self.error:
            raise self.error
        return {"contentType": "text/event-stream", "response": FakeBody(self.lines)}


def test_agentcore_backend_streams():
    fc = FakeAgentCore([b'data: {"type":"delta","text":"hi"}', b"", b'data: {"type":"done","blocked":false}'])
    b = AgentCoreBackend("arn:aws:bedrock-agentcore:us-east-1:1:runtime/x", client=fc)
    evs = list(b.turn(chat_payload()))
    assert [e["type"] for e in evs] == ["delta", "done"]
    assert fc.kw["qualifier"] == "DEFAULT" and len(fc.kw["runtimeSessionId"]) >= 33
    assert json.loads(fc.kw["payload"])["prompt"] == "Why is the sky blue?"


class Raising:
    name = "down"

    def turn(self, payload, cancel=None):
        raise BackendUnavailable("no route")
        yield  # pragma: no cover


class Ok:
    name = "local"

    def turn(self, payload, cancel=None):
        yield {"type": "delta", "text": "x"}
        yield {"type": "done", "blocked": False}


class HalfThenDie:
    name = "half"

    def turn(self, payload, cancel=None):
        yield {"type": "delta", "text": "par"}
        raise BackendUnavailable("cut")


def test_chain_falls_back_when_nothing_emitted():
    ch = ChainBackend([AgentCoreBackend("arn", client=FakeAgentCore(error=RuntimeError("EndpointConnectionError"))), Ok()])
    evs = list(ch.turn(chat_payload()))
    assert evs[-1] == {"type": "done", "blocked": False, "backend": "local"} and ch.last_used == "local"


def test_chain_never_switches_mid_answer():
    evs = list(ChainBackend([HalfThenDie(), Ok()]).turn(chat_payload()))
    assert evs == [{"type": "delta", "text": "par"}, {"type": "error", "code": "offline", "message": "connection lost"}]


def test_chain_all_down_is_friendly_offline():
    evs = list(ChainBackend([Raising(), Raising()]).turn(chat_payload()))
    assert evs[-1]["offline"] is True and evs[-1]["backend"] == "offline" and "can't chat" in evs[-1]["text"]


def test_chain_skips_runtime_internal_error():
    class Internal:
        name = "agentcore"

        def turn(self, payload, cancel=None):
            yield {"type": "error", "code": "internal", "message": "KeyError"}

    assert list(ChainBackend([Internal(), Ok()]).turn(chat_payload()))[-1]["backend"] == "local"


# ---------------------------------------------------------------- runtime HTTP / SSE

def test_runtime_http_streams_sse(monkeypatch):
    monkeypatch.setattr(agent_main, "_deps", make_deps(FakeBedrock(streams=[text_stream("Hello ", "friend")])))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), agent_main.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        c.request("GET", "/ping")
        assert json.loads(c.getresponse().read())["status"] == "Healthy"
        c.request("POST", "/invocations", body=json.dumps(chat_payload()), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
        evs = list(parse_sse(io.BytesIO(r.read()).read().splitlines()))
        assert [e["type"] for e in evs] == ["delta", "delta", "done"]
        c.request("POST", "/invocations", body=b"[1,2]")
        assert c.getresponse().status == 400
        c.close()
    finally:
        srv.shutdown()


# ---------------------------------------------------------------- isolation

def test_package_never_imports_vision_pipeline():
    bad = []
    for py in PKG.rglob("*.py"):
        if "build" in py.parts:
            continue
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            bad += [f"{py.name}:{n}" for n in names if n.split(".")[0] in ("edge", "vision_core", "agent", "cloud")]
    assert bad == []


def test_agent_files_are_flat_importable():
    """The runtime zip is flat: agent modules may only import each other by bare name (with a repo fallback)."""
    from kids_chat.infra import deploy
    for f in deploy.AGENT_FILES:
        assert (PKG / "agent" / f).is_file()
