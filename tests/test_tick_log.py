"""Verbose tick logger formats skip / clear reasons without pixels."""

from __future__ import annotations

from io import StringIO

from edge.tick_log import TickLogger, format_tick, skip_or_clear_reason


def test_verbose_logger_formats_skip_and_clear_reason() -> None:
    skip = {
        "tick": 4,
        "frame_id": 3,
        "source": "live",
        "detect": "nsfw",
        "backend": "local",
        "capture_ok": True,
        "capture_backend": "quartz",
        "frame_shape": (2234, 3456, 3),
        "mean_pixel": 31.2,
        "granted": True,
        "detect_called": False,
        "model_loaded": True,
        "n_raw": 0,
        "top": (),
        "max_score": 0.0,
        "threshold": 0.45,
        "n_after_policy": 0,
        "n_tiles": 4,
        "detect_reason": "nudenet raw=0 max=0.00 < balanced 0.45; model_loaded=True",
        "orch_action": "NO_ACTION",
        "overlay_n": 0,
        "gomi_state": "clear",
        "gomi_expanded": False,
    }
    reason = skip_or_clear_reason(skip)
    assert "detect skipped" in reason
    assert "change-gate" in reason
    line = format_tick(skip)
    assert "gate=skip" in line
    assert "reason=" in line
    assert "pixels" not in line.lower()

    clear = dict(skip)
    clear["detect_called"] = True
    clear["max_score"] = 0.21
    clear["top"] = (("FACE_FEMALE", 0.21),)
    clear["detect_reason"] = "nudenet max=0.21 FACE_FEMALE < balanced 0.45 (explicit max=0.00)"
    clear["n_raw"] = 1
    why = skip_or_clear_reason(clear)
    assert "0.21" in why
    assert "0.45" in why
    assert "ALL CLEAR" not in why or "nudenet" in why
    formatted = format_tick(clear)
    assert "FACE_FEMALE:0.21" in formatted
    assert "detect=ran" in formatted
    assert "clusters=0" in formatted
    assert "clusters=0 cap=" not in formatted

    capped = dict(clear)
    capped["n_clusters"] = 24
    capped["cluster_cap"] = 24
    capped["clusters_capped"] = True
    capped_line = format_tick(capped)
    assert "clusters=24 cap=24" in capped_line

    failed = {"capture_ok": False, "capture_reason": "capture failed: black frame"}
    assert skip_or_clear_reason(failed) == "capture failed: black frame"

    stream = StringIO()
    logger = TickLogger(every_s=0.0, stream=stream)
    assert logger.emit(clear) is True
    assert "nudenet max=0.21" in stream.getvalue()
