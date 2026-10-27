"""companion.say — local kids-AI socket.

Bedrock Guardrails will replace this text later. Keep every line free of
sexual description; reason codes stay machine-only.
"""

from __future__ import annotations

_LINES: dict[str, str] = {
    "FLAGGED_ABOVE_BLOCK": "Covering that area so the screen stays appropriate.",
    "FLAGGED_STRICT_BLOCK": "Covering that area so the screen stays appropriate.",
    "FLAGGED_ABOVE_OVERLAY": "Softening a flagged area.",
    "TRACK_HELD": "Holding the last cover while I wait.",
    "TRACK_EXPIRED": "Scene changed. Covers released.",
    "SAFE_CLASS": "Watching. All clear.",
    "BELOW_THRESHOLD": "Watching. All clear.",
    "BELOW_OVERLAY": "Watching. All clear.",
    "AREA_TOO_SMALL": "Watching. All clear.",
    "CLASS_ALLOW": "Watching. All clear.",
    "UNCERTAIN_REINSPECT": "Taking a closer look.",
    "TOOL_TIMEOUT": "Look timed out. Holding cover.",
    "NOT_PROTECTING": "Not covering this frame.",
    "GUARDIAN_REVIEW": "Paused for a grown-up look.",
    "SCENE_CUT": "Scene changed. Resetting covers.",
    "FULL_WINDOW_FALLBACK": "Covering the window until I can see again.",
    "APPLY": "Cover is on.",
    "NO_ACTION": "Watching. All clear.",
    "RELEASE": "Scene changed. Covers released.",
    "HOLD": "Holding the last cover while I wait.",
    "REVIEW": "Paused for a grown-up look.",
}


def say(reason_code: str) -> str:
    """Return a short safe status line for ``reason_code``."""
    key = (reason_code or "").strip() or "NO_ACTION"
    return _LINES.get(key, "Watching the screen.")
