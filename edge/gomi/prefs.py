"""Island prefs. Local only — character, policy, detect. Not a capture."""

from __future__ import annotations

import json
from pathlib import Path

PREFS_PATH = Path(__file__).resolve().parents[2] / ".prefs" / "island.json"
DEFAULTS = {
    "character_id": "pebble",
    "policy": "balanced",
    "detect": "synthetic",
    "preview": False,
}


def load_prefs(path: Path | None = None) -> dict:
    target = path or PREFS_PATH
    data = dict(DEFAULTS)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return data
    if isinstance(raw, dict):
        for key in DEFAULTS:
            if key in raw:
                data[key] = raw[key]
    if data["character_id"] not in {"pebble", "fox", "ghost", "bear", "cloud"}:
        data["character_id"] = "pebble"
    if data["policy"] not in {"strict", "balanced", "relaxed"}:
        data["policy"] = "balanced"
    if data["detect"] not in {"nsfw", "synthetic"}:
        data["detect"] = "synthetic"
    data["preview"] = bool(data["preview"])
    return data


def save_prefs(updates: dict, path: Path | None = None) -> dict:
    target = path or PREFS_PATH
    data = load_prefs(target)
    for key in DEFAULTS:
        if key in updates:
            data[key] = updates[key]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data
