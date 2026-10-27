"""Island size springs and the expanded layout. No AppKit."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from edge.gomi.panel import controls as panel_controls

COMPACT_W = 220.0
COMPACT_H = 38.0
COMPACT_R = 19.0
EXPANDED_W = 520.0
EXPANDED_H = 176.0
EXPANDED_R = 22.0
CHAR_COMPACT = (6.0, 28.0)
CHAR_EXPANDED = (20.0, 72.0)
CLOSE_S = 0.34
OPEN_RESPONSE = 0.5
OPEN_DAMPING = 0.72

# Coucou close curve: cubic-bezier(0.45, 0, 0.2, 1) over 0.34s. Monotonic.
_BEZIER = (0.45, 0.0, 0.2, 1.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def bezier_ease(u: float, p1x: float = 0.45, p1y: float = 0.0, p2x: float = 0.2, p2y: float = 1.0) -> float:
    """Y of a unit cubic bezier at x=u. Monotonic for the close curve."""
    u = 0.0 if u < 0 else 1.0 if u > 1 else u
    t = u
    for _ in range(8):
        x = 3 * (1 - t) * (1 - t) * t * p1x + 3 * (1 - t) * t * t * p2x + t * t * t
        dx = (
            3 * (1 - t) * (1 - t) * p1x
            + 6 * (1 - t) * t * (p2x - p1x)
            + 3 * t * t * (1 - p2x)
        )
        if abs(dx) < 1e-6:
            break
        t = min(1.0, max(0.0, t - (x - u) / dx))
    return 3 * (1 - t) * (1 - t) * t * p1y + 3 * (1 - t) * t * t * p2y + t * t * t


class Axis:
    """One channel. Open springs; close eases without overshoot."""

    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.vel = 0.0
        self.target = float(value)
        self.mode = "spring"
        self._from = float(value)
        self._to = float(value)
        self._age = 0.0

    def spring_to(self, target: float, *, response: float = OPEN_RESPONSE, damping: float = OPEN_DAMPING) -> None:
        self.mode = "spring"
        self.target = float(target)
        self._response = response
        self._damping = damping

    def curve_towards(self, target: float, duration: float = CLOSE_S) -> None:
        self.mode = "curve"
        self._from = self.value
        self._to = float(target)
        self.target = float(target)
        self._age = 0.0
        self._dur = max(0.05, duration)
        self.vel = 0.0

    def step(self, dt: float) -> float:
        dt = max(0.0, min(0.05, dt))
        if self.mode == "curve":
            self._age += dt
            u = bezier_ease(min(1.0, self._age / self._dur), *_BEZIER)
            self.value = _lerp(self._from, self._to, u)
            self.vel = 0.0
            return self.value
        omega = 2.0 * math.pi / max(0.05, getattr(self, "_response", OPEN_RESPONSE))
        zeta = getattr(self, "_damping", OPEN_DAMPING)
        acc = omega * omega * (self.target - self.value) - 2.0 * zeta * omega * self.vel
        self.vel += acc * dt
        self.value += self.vel * dt
        if abs(self.target - self.value) < 0.15 and abs(self.vel) < 4.0:
            self.value = self.target
            self.vel = 0.0
        return self.value


@dataclass
class Hit:
    kind: str
    value: str
    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


@dataclass
class Layout:
    width: float
    height: float
    radius: float
    char_x: float
    char_d: float
    char_y: float
    panel_x: float
    hits: list[Hit] = field(default_factory=list)
    expand: float = 0.0


def expand_amount(height: float) -> float:
    span = EXPANDED_H - COMPACT_H
    if span <= 0:
        return 0.0
    return max(0.0, min(1.0, (height - COMPACT_H) / span))


def layout_for(width: float, height: float, char_x: float, char_d: float) -> Layout:
    expand = expand_amount(height)
    radius = _lerp(COMPACT_R, EXPANDED_R, expand)
    char_y = max(4.0, (height - char_d) * 0.5)
    panel_x = char_x + char_d + 16.0
    hits: list[Hit] = []
    if expand > 0.62:
        for item in panel_controls(panel_x, width):
            hits.append(Hit(item.kind, item.value, item.x, item.y, item.w, item.h))
    return Layout(width, height, radius, char_x, char_d, char_y, panel_x, hits, expand)


class IslandGeometry:
    """Independent width, height, radius, and character slot."""

    def __init__(self) -> None:
        self.w = Axis(COMPACT_W)
        self.h = Axis(COMPACT_H)
        self.r = Axis(COMPACT_R)
        self.cx = Axis(CHAR_COMPACT[0])
        self.cd = Axis(CHAR_COMPACT[1])
        self.expanded = False

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = bool(expanded)
        if self.expanded:
            self.w.spring_to(EXPANDED_W)
            self.h.spring_to(EXPANDED_H)
            self.r.spring_to(EXPANDED_R)
            self.cx.spring_to(CHAR_EXPANDED[0])
            self.cd.spring_to(CHAR_EXPANDED[1])
        else:
            self.w.curve_towards(COMPACT_W)
            self.h.curve_towards(COMPACT_H)
            self.r.curve_towards(COMPACT_R)
            self.cx.curve_towards(CHAR_COMPACT[0])
            self.cd.curve_towards(CHAR_COMPACT[1])

    def step(self, dt: float) -> Layout:
        self.w.step(dt)
        self.h.step(dt)
        self.r.step(dt)
        self.cx.step(dt)
        self.cd.step(dt)
        return self.layout()

    def layout(self) -> Layout:
        lay = layout_for(self.w.value, self.h.value, self.cx.value, self.cd.value)
        lay.radius = self.r.value
        return lay


def hit_test(lay: Layout, x: float, y: float) -> Hit | None:
    for hit in reversed(lay.hits):
        if hit.contains(x, y):
            return hit
    return None
