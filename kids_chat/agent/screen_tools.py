"""Explain-my-screen drawing tools (client-executed: they only *queue* draw ops).

The agent runs in the cloud and the drawing happens on the Mac, so these are
Converse ``toolSpec``s whose only effect is to validate a draw op and return
"drawn". ``core.py`` streams each op back as ``{"type":"draw","op":{...}}``;
the Mac service re-validates, maps coordinates and hands them to the island.
The agent draws, it never clicks.

Coordinates: ``norm1000`` = a 0-1000 grid over the *sent* (redacted) picture,
x right, y down, origin top-left. Circle radius is in x units.
"""

from __future__ import annotations

import re
from typing import Any

SPACE = "norm1000"
GRID = 1000.0
MAX_OPS = 8
LABEL_MAX = 32
NOTE_MAX = 80
COLORS = ("yellow", "blue", "green", "pink", "white")
MASK_PAD = 12.0  # grid units added around each covered region
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def _num(desc: str) -> dict:
    return {"type": "number", "description": desc}


_LABEL = {"type": "string", "description": f"Short caption, at most {LABEL_MAX} characters (aim for 1-4 words)."}
_COLOR = {"type": "string", "enum": list(COLORS), "description": "Optional colour. Default yellow."}

TOOLS = [
    {"name": "point_to",
     "description": "Point at one spot on the picture with a friendly pointer and an optional caption. "
                    "Use it for a single thing (a button, a gear, a word). x and y are on the 0-1000 grid over the picture "
                    "(0,0 top-left; 1000,1000 bottom-right). Never point at a gray 'covered' box.",
     "schema": {"x": _num("0-1000, left to right"), "y": _num("0-1000, top to bottom"), "label": _LABEL, "color": _COLOR},
     "required": ["x", "y"]},
    {"name": "highlight_box",
     "description": "Outline a rectangular area of the picture with an optional caption. Use this for an area and point_to for a spot. "
                    "x, y is the top-left corner and w, h the size, all on the 0-1000 grid.",
     "schema": {"x": _num("left edge, 0-1000"), "y": _num("top edge, 0-1000"), "w": _num("width, 0-1000"),
                "h": _num("height, 0-1000"), "label": _LABEL, "color": _COLOR},
     "required": ["x", "y", "w", "h"]},
    {"name": "draw_circle",
     "description": "Draw a circle around something round or small. cx, cy is the centre and r the radius, on the 0-1000 grid "
                    "(r is measured along the picture width).",
     "schema": {"cx": _num("centre x, 0-1000"), "cy": _num("centre y, 0-1000"), "r": _num("radius, 5-400"),
                "label": _LABEL, "color": _COLOR},
     "required": ["cx", "cy", "r"]},
    {"name": "draw_arrow",
     "description": "Draw an arrow between two places. The head sits at the 'to' end, so use it when direction matters: "
                    "from a label to the part it names, the way something moves, or the way a force or a flow goes. "
                    "Prefer point_to for one spot and highlight_box for an area: an arrow earns its ink when the link "
                    "between two places is the thing being shown. Curved by default, like a person draws one; bow 0 is "
                    "straight, up to 1 bows right of travel, down to -1 left. All values on the 0-1000 grid.",
     "schema": {"from_x": _num("start x"), "from_y": _num("start y"), "to_x": _num("end x (arrow head)"),
                "to_y": _num("end y (arrow head)"), "bow": _num("-1 to 1, default 0.2"), "label": _LABEL, "color": _COLOR},
     "required": ["from_x", "from_y", "to_x", "to_y"]},
    {"name": "label",
     "description": f"Place a short note (at most {NOTE_MAX} characters) on the picture. Display only: the child cannot click it. "
                    "Put it beside what it explains, not on top of it. x, y is the note's top-left corner on the 0-1000 grid.",
     "schema": {"x": _num("0-1000"), "y": _num("0-1000"), "text": {"type": "string", "description": f"At most {NOTE_MAX} characters."}},
     "required": ["x", "y", "text"]},
    {"name": "clear_drawings",
     "description": "Remove every drawing from the screen. Use it before drawing something new when old drawings would confuse.",
     "schema": {}, "required": []},
]

TOOL_NAMES = frozenset(t["name"] for t in TOOLS)


def tool_config() -> dict:
    return {"tools": [{"toolSpec": {"name": t["name"], "description": t["description"],
                                    "inputSchema": {"json": {"type": "object", "properties": t["schema"],
                                                             "required": t["required"]}}}} for t in TOOLS]}


class DrawError(ValueError):
    pass


def _f(args: dict, key: str, lo: float = 0.0, hi: float = GRID) -> float:
    v = args.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise DrawError(f"{key} must be a number")
    v = float(v)
    if v != v or not lo <= v <= hi:  # NaN or out of range
        raise DrawError(f"{key} must be between {lo:g} and {hi:g}")
    return round(v, 1)


def _text(v: Any, limit: int) -> str:
    s = _CTRL.sub(" ", str(v or "")).strip()
    return s[:limit]


def _color(v: Any) -> str:
    return v if v in COLORS else "yellow"


def op_from_tool(name: str, args: dict) -> dict:
    """Tool call → validated draw op (``norm1000``). Raises :class:`DrawError`."""
    if name not in TOOL_NAMES:
        raise DrawError(f"unknown tool {name}")
    a = args if isinstance(args, dict) else {}
    if name == "point_to":
        return {"kind": "pointer", "at": [_f(a, "x"), _f(a, "y")], "label": _text(a.get("label"), LABEL_MAX), "color": _color(a.get("color"))}
    if name == "highlight_box":
        x, y, w, h = _f(a, "x"), _f(a, "y"), _f(a, "w", 1), _f(a, "h", 1)
        if x + w > GRID + 0.5 or y + h > GRID + 0.5:
            raise DrawError("box goes past the picture edge")
        return {"kind": "box", "rect": [x, y, w, h], "label": _text(a.get("label"), LABEL_MAX), "color": _color(a.get("color"))}
    if name == "draw_circle":
        return {"kind": "circle", "center": [_f(a, "cx"), _f(a, "cy")], "r": _f(a, "r", 5, 400),
                "label": _text(a.get("label"), LABEL_MAX), "color": _color(a.get("color"))}
    if name == "draw_arrow":
        fr, to = [_f(a, "from_x"), _f(a, "from_y")], [_f(a, "to_x"), _f(a, "to_y")]
        if abs(fr[0] - to[0]) + abs(fr[1] - to[1]) < 10:
            raise DrawError("arrow too short")
        bow = _f(a, "bow", -1, 1) if a.get("bow") is not None else 0.2
        return {"kind": "arrow", "from": fr, "to": to, "bow": bow, "label": _text(a.get("label"), LABEL_MAX),
                "color": _color(a.get("color"))}
    if name == "label":
        text = _text(a.get("text"), NOTE_MAX)
        if not text:
            raise DrawError("label needs text")
        return {"kind": "label", "at": [_f(a, "x"), _f(a, "y")], "text": text}
    return {"kind": "clear"}


def op_bounds(op: dict) -> tuple[float, float, float, float] | None:
    """Axis-aligned bounds (x0, y0, x1, y1) of the op's mark in grid units."""
    k = op.get("kind")
    if k == "pointer" or k == "label":
        x, y = op["at"]
        return (x - 6, y - 6, x + 6, y + 6)
    if k == "box":
        x, y, w, h = op["rect"]
        return (x, y, x + w, y + h)
    if k == "circle":
        (cx, cy), r = op["center"], op["r"]
        return (cx - r, cy - r, cx + r, cy + r)
    if k == "arrow":
        (x0, y0), (x1, y1) = op["from"], op["to"]
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    return None


def _hits(b: tuple, m: tuple, pad: float) -> bool:
    return not (b[2] < m[0] - pad or b[0] > m[2] + pad or b[3] < m[1] - pad or b[1] > m[3] + pad)


def _segment_hits(op: dict, m: tuple, pad: float, steps: int = 24) -> bool:
    (x0, y0), (x1, y1) = op["from"], op["to"]
    for i in range(steps + 1):
        t = i / steps
        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        if m[0] - pad <= x <= m[2] + pad and m[1] - pad <= y <= m[3] + pad:
            return True
    return False


def touches_mask(op: dict, masks: list, pad: float = MASK_PAD) -> bool:
    """True when the mark would sit on (or next to) a covered region. Masks: [[x0,y0,x1,y1], ...] grid units."""
    b = op_bounds(op)
    if b is None:
        return False
    for m in masks or ():
        mm = tuple(float(v) for v in m)
        if op.get("kind") == "arrow":
            if _segment_hits(op, mm, pad):
                return True
        elif op.get("kind") == "box":
            # an outline around a mask is still "drawing on" it; any overlap counts
            if _hits(b, mm, pad):
                return True
        elif _hits(b, mm, pad):
            return True
    return False


def labels_text(ops: list[dict]) -> str:
    return "\n".join(t for op in ops for t in (op.get("label", ""), op.get("text", "")) if t)
