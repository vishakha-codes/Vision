"""Expanded island controls: character on the left, these options on the right."""

from __future__ import annotations

from dataclasses import dataclass

SKINS = ("pebble", "fox", "ghost", "bear", "cloud")
POLICIES = ("strict", "balanced", "relaxed")
DETECTS = ("nsfw", "synthetic")


@dataclass(frozen=True)
class Control:
    kind: str
    value: str
    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


def controls(panel_x: float, width: float) -> list[Control]:
    """Thumb row, policy row, detect row. Only shown once the island is open."""
    hits: list[Control] = []
    tw, gap = 36.0, 8.0
    row_w = len(SKINS) * tw + (len(SKINS) - 1) * gap
    x0 = min(panel_x, max(panel_x, width - row_w - 16.0))
    for i, skin in enumerate(SKINS):
        hits.append(Control("character", skin, x0 + i * (tw + gap), 70.0, tw, tw))
    pw = 78.0
    for i, name in enumerate(POLICIES):
        hits.append(Control("policy", name, panel_x + i * (pw + 6.0), 116.0, pw, 22.0))
    dw = 96.0
    for i, name in enumerate(DETECTS):
        hits.append(Control("detect", name, panel_x + i * (dw + 8.0), 144.0, dw, 22.0))
    return hits
