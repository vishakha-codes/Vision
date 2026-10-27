"""edge.engine_log: JSON lines, ~1 Hz, rotation, privacy scrub, never raises."""

from __future__ import annotations

import json
import os
import stat
import time

import numpy as np

from edge.engine_log import DEFAULT_PATH, EngineLog, default_path, scrub


def _lines(path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_default_path_and_env_override(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SG_ENGINE_LOG", raising=False)
    assert default_path() == DEFAULT_PATH
    assert str(DEFAULT_PATH).endswith("Library/Logs/ScreenGuardian/engine.log")
    monkeypatch.setenv("SG_ENGINE_LOG", str(tmp_path / "x.log"))
    assert EngineLog().path == tmp_path / "x.log"
    monkeypatch.setenv("SG_ENGINE_LOG", "off")
    log = EngineLog()
    assert not log.enabled
    log.record({"fps": 1})  # no-op, no error


def test_writes_json_lines_rate_limited(tmp_path) -> None:
    path = tmp_path / "logs" / "engine.log"
    log = EngineLog(path)
    for i in range(50):
        log.record({"frame": i, "mode": "medium", "masks": 2})
    rows = _lines(path)
    assert len(rows) == 1 and rows[0]["frame"] == 0 and rows[0]["mode"] == "medium"
    assert {"t", "seq"} <= set(rows[0])
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    log._last_t = time.monotonic() - 2.0  # a second later
    log.record({"frame": 50})
    rows = _lines(path)
    assert len(rows) == 2 and rows[1]["dropped"] == 49


def test_events_and_errors_skip_the_rate_limit(tmp_path) -> None:
    log = EngineLog(tmp_path / "engine.log")
    log.record({"fps": 12})
    log.record({"event": "mode_change", "mode": "medium"})
    log.record({"error": "capture thread died: RuntimeError"})
    for _ in range(30):
        log.record({"event": "spam"})
    rows = _lines(tmp_path / "engine.log")
    assert rows[1]["event"] == "mode_change" and rows[2]["error"].startswith("capture")
    assert len(rows) <= 1 + 10  # events capped per second


def test_never_writes_pixels_titles_or_urls(tmp_path) -> None:
    log = EngineLog(tmp_path / "engine.log", min_interval_s=0.0)
    log.record(
        {
            "frame": np.zeros((1117, 1728, 3), np.uint8),
            "window_title": "Some private page",
            "windows": [{"title": "Inbox", "app_name": "Mail", "bounds": (0, 0, 800, 600), "layer": 0}],
            "pageURL": "https://example.invalid/a",
            "note": "opened https://example.invalid/b",
            "crop_bgr": b"\x00\x01",
            "box": np.array([1.0, 2.0, 3.0, 4.0]),
            "score": np.float32(0.6125),
            "during_scroll": True,
            "tokens": 3.5,
            "masks": [{"reason": "UNSAFE_PARTS", "xyxy": (1, 2, 3, 4)}],
        }
    )
    text = (tmp_path / "engine.log").read_text()
    for bad in ("private", "Inbox", "Mail", "example.invalid", "window_title", "pageURL", "crop_bgr"):
        assert bad not in text
    row = _lines(tmp_path / "engine.log")[0]
    assert "frame" not in row  # an image-sized array is dropped
    assert row["box"] == [1.0, 2.0, 3.0, 4.0]
    assert row["score"] == 0.6125 and row["during_scroll"] is True and row["tokens"] == 3.5
    assert row["windows"] == [{"bounds": [0, 0, 800, 600], "layer": 0}]
    assert row["note"] == "[scrubbed]"
    assert row["masks"][0]["reason"] == "UNSAFE_PARTS"


def test_rotates_by_size(tmp_path) -> None:
    path = tmp_path / "engine.log"
    log = EngineLog(path, min_interval_s=0.0, max_bytes=2048, backups=3)
    for i in range(200):
        log.record({"i": i, "pad": "x" * 100})
    assert path.exists() and path.stat().st_size <= 2048
    assert (tmp_path / "engine.log.1").exists() and (tmp_path / "engine.log.3").exists()
    assert not (tmp_path / "engine.log.4").exists()
    assert _lines(path)[-1]["i"] == 199


def test_never_raises_into_the_engine(tmp_path) -> None:
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("file")
    log = EngineLog(blocker / "engine.log", min_interval_s=0.0)
    for _ in range(10):
        log.record({"fps": 1})  # mkdir fails every time
    assert log.error and not log.enabled

    class Bad:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    ok = EngineLog(tmp_path / "ok.log", min_interval_s=0.0)
    ok.record({"weird": Bad()})
    ok.record("not a dict")  # type: ignore[arg-type]
    ok.record({"fine": 1})
    assert _lines(tmp_path / "ok.log")[-1]["fine"] == 1


def test_scrub_is_json_safe() -> None:
    out = scrub({"a": float("nan"), "b": {1, 2}, "c": (np.int64(3),), "d": "y" * 1000})
    json.dumps(out)
    assert out["a"] is None and sorted(out["b"]) == [1, 2] and out["c"] == [3] and len(out["d"]) < 400
