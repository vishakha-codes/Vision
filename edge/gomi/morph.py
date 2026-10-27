"""State morph + idle. Exact interpolation when shapes match; fade when they do not.

Idle (breathe, blink, sway, gaze) is applied after the morph so the two do not fight.
A new target interrupts: the current pose becomes the source. No tween queue.
"""

from __future__ import annotations

import math

from edge.gomi.catalog import Figure, lerp_figure
from edge.gomi.character import blink_envelope, pose_at
from edge.gomi.states import GomiState

MORPH_S = 0.32


def _ease_out(u: float) -> float:
    u = 0.0 if u < 0 else 1.0 if u > 1 else u
    return 1.0 - (1.0 - u) ** 3


def _rot(x: float, y: float, cx: float, cy: float, deg: float) -> tuple[float, float]:
    ang = math.radians(deg)
    c, s = math.cos(ang), math.sin(ang)
    dx, dy = x - cx, y - cy
    return (cx + dx * c - dy * s, cy + dx * s + dy * c)


class Morph:
    """One character. ``retarget`` may change state or skin."""

    def __init__(self, catalog: dict, skin: str = "pebble", state: str = "idle") -> None:
        self.catalog = catalog
        self.skin = skin if skin in catalog else "pebble"
        self.state = state if state in catalog[self.skin] else "idle"
        self.source: Figure | None = None
        self._u = 1.0
        self._t0 = 0.0

    @property
    def settled(self) -> bool:
        return self._u >= 1.0 and self.source is None

    def retarget(self, state: str, now: float, skin: str | None = None) -> None:
        skin_id = skin if skin in self.catalog else self.skin
        state_id = state if state in self.catalog[skin_id] else "idle"
        if skin_id == self.skin and state_id == self.state:
            return
        self.source = self.base_at(now)
        self.skin = skin_id
        self.state = state_id
        self._u = 0.0
        self._t0 = now

    def base_at(self, now: float) -> Figure:
        target = self.catalog[self.skin][self.state]
        if self.source is None:
            return target.copy()
        elapsed = max(0.0, now - self._t0)
        self._u = 1.0 if MORPH_S <= 0 else min(1.0, elapsed / MORPH_S)
        mixed = lerp_figure(self.source, target, _ease_out(self._u))
        if self._u >= 1.0:
            self.source = None
            return target.copy()
        return mixed

    def pose_at(self, now: float) -> Figure:
        """Morphed figure with idle motion. Feet stay put while breathing."""
        fig = self.base_at(now)
        try:
            gstate = GomiState(self.state)
        except ValueError:
            gstate = GomiState.IDLE
        pose = pose_at(gstate, now)
        body = fig.shapes.get("body")
        if body is None:
            return fig
        sx, sy = pose.sx, pose.sy
        feet = body.cy + body.ry
        body.rx *= sx
        body.ry *= sy
        body.cy = feet - body.ry
        anchor_x, anchor_y = body.cx, body.cy
        blink = blink_envelope(now) if gstate is not GomiState.SLEEP else 1.0
        if gstate is GomiState.ALERT:
            blink = 0.0
        eye_scale = max(0.12, 1.0 - blink)
        bob = pose.bob
        for key, shape in fig.shapes.items():
            if shape.kind == "ellipse":
                shape.cx, shape.cy = _rot(shape.cx, shape.cy, anchor_x, anchor_y, pose.sway_deg * 0.15)
                shape.cy += bob * 0.35
            elif shape.numbers:
                moved = []
                for i in range(0, len(shape.numbers) - 1, 2):
                    x, y = _rot(shape.numbers[i], shape.numbers[i + 1], anchor_x, anchor_y, pose.sway_deg * 0.15)
                    moved.extend((x, y + bob * 0.35))
                if len(shape.numbers) % 2 == 1:
                    moved.append(shape.numbers[-1])
                shape.numbers = tuple(moved)
            if key in {"left-eye", "right-eye"}:
                shape.ry *= eye_scale
            if key in {"left-pupil", "right-pupil"}:
                shape.cx += pose.look_x * 10.0
                shape.cy += pose.look_y * 8.0
                shape.ry *= eye_scale
                shape.rx *= max(0.35, eye_scale)
                if gstate is GomiState.WORKING:
                    shape.cx += math.sin(now * 2.2) * 3.0
        return fig
