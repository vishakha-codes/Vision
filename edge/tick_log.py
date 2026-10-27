"""Stderr tick trace for ``python -m edge.app -v``.

One line that makes ALL CLEAR / skip / capture-fail obvious. No pixels.
"""

from __future__ import annotations

import sys
import time
from typing import Any, TextIO


def _clusters_field(record: dict[str, Any]) -> str:
    """``clusters=N`` normally; ``clusters=N cap=M`` only when the safety ceiling hit."""
    n = int(record.get("n_clusters", 0) or 0)
    if record.get("clusters_capped"):
        cap = int(record.get("cluster_cap", 0) or 0)
        return f"clusters={n} cap={cap}"
    return f"clusters={n}"


def skip_or_clear_reason(record: dict[str, Any]) -> str:
    """Why this tick did not cover the screen."""
    if not record.get("capture_ok", True):
        return str(record.get("capture_reason") or "capture failed")
    if record.get("detect_called") is False:
        last = record.get("detect_reason") or "no prior detect"
        return f"detect skipped (change-gate hold); last {last}"
    reason = str(record.get("detect_reason") or "")
    action = str(record.get("orch_action") or "NO_ACTION")
    n_after = int(record.get("n_after_policy") or 0)
    if n_after <= 0:
        return reason or f"ALL CLEAR: {action}"
    return reason or action


def format_tick(record: dict[str, Any]) -> str:
    """Single stderr line. Keys may be omitted; missing values become '-'."""
    shape = record.get("frame_shape")
    if isinstance(shape, (tuple, list)) and len(shape) >= 2:
        shape_s = f"{shape[1]}x{shape[0]}" if len(shape) >= 2 else str(shape)
    else:
        shape_s = "-"
    mean = record.get("mean_pixel")
    mean_s = f"{float(mean):.1f}" if mean is not None else "-"
    top = record.get("top") or ()
    if top:
        top_s = ",".join(f"{name}:{score:.2f}" for name, score in list(top)[:3])
    else:
        top_s = "-"
    gate = "skip" if record.get("detect_called") is False else "tick"
    model = "yes" if record.get("model_loaded") else "no"
    expanded = record.get("gomi_expanded")
    if expanded is True or (isinstance(expanded, (int, float)) and float(expanded) >= 0.5):
        exp_s = "yes"
    elif expanded is False or expanded == 0 or expanded == 0.0:
        exp_s = "no"
    else:
        exp_s = "-"
    reason = skip_or_clear_reason(record)
    return (
        f"[sg] tick={record.get('tick', '-')} frame={record.get('frame_id', '-')} "
        f"src={record.get('source', '-')} detect={record.get('detect', '-')} "
        f"backend={record.get('backend', '-')} "
        f"cap={'ok' if record.get('capture_ok', True) else 'FAIL'} "
        f"{record.get('capture_backend', '-')} {shape_s} mean={mean_s} "
        f"granted={record.get('granted', '-')} "
        f"gate={gate} detect={'ran' if record.get('detect_called') else 'skip'} "
        f"model={model} raw={record.get('n_raw', 0)} top={top_s} "
        f"max={float(record.get('max_score') or 0):.2f} "
        f"floor={float(record.get('threshold') or 0):.2f} "
        f"after={record.get('n_after_policy', 0)} {_clusters_field(record)} "
        f"tiles={record.get('n_tiles', 0)} "
        f"orch={record.get('orch_action', '-')} overlay={record.get('overlay_n', 0)} "
        f"gomi={record.get('gomi_state', '-')} expanded={exp_s} "
        f"reason=\"{reason}\""
    )


class TickLogger:
    """Write ``format_tick`` to stderr every tick, or at most once per second."""

    def __init__(self, *, every_s: float = 0.0, stream: TextIO | None = None) -> None:
        self.every_s = float(every_s)
        self.stream = stream or sys.stderr
        self._last = 0.0
        self._wrote = 0

    def emit(self, record: dict[str, Any], *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and self.every_s > 0 and self._wrote and (now - self._last) < self.every_s:
            return False
        line = format_tick(record)
        print(line, file=self.stream, flush=True)
        self._last = now
        self._wrote += 1
        return True
