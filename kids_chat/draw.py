"""Draw ops: agent grid (``norm1000``) → Quartz global points for the island.

Coordinate chain (end to end):

1. The engine captures ONE display (the one the child is using), paints every
   mask solid, and downscales to the *sent* picture ``w x h`` px. It reports
   that display as ``display = {x, y, w, h}`` = ``CGDisplayBounds`` in Quartz
   global points (origin top-left of the main display, y down), and the masks
   in sent pixels.
2. The model calls the draw tools on a 0-1000 grid over the picture
   (x right, y down). ``agent/screen_tools.py`` validates.
3. Here: re-validate, drop any mark touching a mask, then map
   ``X = d.x + gx/1000 * d.w`` and ``Y = d.y + gy/1000 * d.h``; sizes scale by
   ``d.w/1000`` (x, radius) and ``d.h/1000`` (y). Retina never enters: the
   picture is measured in fractions and the display in points.
4. The island draws Quartz global points (``space: "quartz_pt"``) in one
   flipped view per screen whose origin is that screen's ``CGDisplayBounds``
   origin, so multi-display setups map with no extra math.

Output op shapes (``quartz_pt``, y down):
    {"kind":"pointer","at":[x,y],"label":"..","color":".."}
    {"kind":"box","rect":[x,y,w,h],...}             rect origin = top-left corner
    {"kind":"circle","center":[x,y],"r":r,...}
    {"kind":"arrow","from":[x,y],"to":[x,y],"bow":0.2,...}   head at "to"
    {"kind":"label","at":[x,y],"text":".."}         at = the note's top-left corner
    {"kind":"clear"}
Each op also gets ``seq``: the island reveals marks one at a time in this order.
"""

from __future__ import annotations

from typing import Any

from kids_chat.agent import screen_tools as st

TTL_MS = 20_000
_TOOL_FOR = {"pointer": "point_to", "box": "highlight_box", "circle": "draw_circle", "arrow": "draw_arrow",
             "label": "label", "clear": "clear_drawings"}


def _revalidate(op: Any) -> dict:
    """Re-run the agent-side validator on an op that crossed the network."""
    if not isinstance(op, dict) or op.get("kind") not in _TOOL_FOR:
        raise st.DrawError("bad op")
    k = op["kind"]
    try:
        if k == "pointer":
            args = {"x": op["at"][0], "y": op["at"][1], "label": op.get("label"), "color": op.get("color")}
        elif k == "box":
            x, y, w, h = op["rect"]
            args = {"x": x, "y": y, "w": w, "h": h, "label": op.get("label"), "color": op.get("color")}
        elif k == "circle":
            args = {"cx": op["center"][0], "cy": op["center"][1], "r": op["r"], "label": op.get("label"), "color": op.get("color")}
        elif k == "arrow":
            args = {"from_x": op["from"][0], "from_y": op["from"][1], "to_x": op["to"][0], "to_y": op["to"][1],
                    "bow": op.get("bow"), "label": op.get("label"), "color": op.get("color")}
        elif k == "label":
            args = {"x": op["at"][0], "y": op["at"][1], "text": op.get("text")}
        else:
            args = {}
    except (KeyError, IndexError, TypeError, ValueError):
        raise st.DrawError("bad op")
    return st.op_from_tool(_TOOL_FOR[k], args)


def to_screen(ops: list, display: dict, masks_norm: list | None = None, *, limit: int = st.MAX_OPS) -> list[dict]:
    """Validated ``norm1000`` ops → ``quartz_pt`` ops. Invalid or mask-touching ops are dropped."""
    dx, dy, dw, dh = (float(display[k]) for k in ("x", "y", "w", "h"))
    sx, sy = dw / st.GRID, dh / st.GRID

    def pt(p):
        return [round(dx + p[0] * sx, 1), round(dy + p[1] * sy, 1)]

    out: list[dict] = []
    for raw in ops or []:
        try:
            op = _revalidate(raw)
        except st.DrawError:
            continue
        if st.touches_mask(op, masks_norm or []):
            continue
        k = op["kind"]
        if k == "clear":
            out.append({"kind": "clear"})
        elif k == "pointer":
            out.append({**op, "at": pt(op["at"])})
        elif k == "label":
            out.append({**op, "at": pt(op["at"])})
        elif k == "box":
            x, y, w, h = op["rect"]
            tl = pt([x, y])
            out.append({**op, "rect": [tl[0], tl[1], round(w * sx, 1), round(h * sy, 1)]})
        elif k == "circle":
            out.append({**op, "center": pt(op["center"]), "r": round(op["r"] * sx, 1)})
        elif k == "arrow":
            out.append({**op, "from": pt(op["from"]), "to": pt(op["to"])})
        out[-1]["seq"] = len(out) - 1
        if len(out) >= limit:
            break
    return out


def avoid_rects(display: dict, masks_norm: list | None) -> list[list[float]]:
    """Mask regions in Quartz points ([x, y, w, h]) so the island keeps labels off them."""
    dx, dy, dw, dh = (float(display[k]) for k in ("x", "y", "w", "h"))
    return [[round(dx + m[0] * dw / st.GRID, 1), round(dy + m[1] * dh / st.GRID, 1),
             round((m[2] - m[0]) * dw / st.GRID, 1), round((m[3] - m[1]) * dh / st.GRID, 1)] for m in masks_norm or []]
