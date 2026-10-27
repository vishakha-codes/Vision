"""Load clay SVG skins. One schema of element ids across every state."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

SVG_DIR = Path(__file__).resolve().parent / "svg"
SKIN_IDS = ("pebble", "fox", "ghost", "bear", "cloud")
STATE_IDS = ("idle", "alert", "working", "clear", "sleep")
DEFAULT_SKIN = "pebble"
REQUIRED_IDS = (
    "shadow",
    "ear-l",
    "ear-r",
    "ear-l-inner",
    "ear-r-inner",
    "puff-l",
    "puff-r",
    "body",
    "highlight",
    "blush",
    "left-eye",
    "right-eye",
    "left-pupil",
    "right-pupil",
    "mouth",
)


def _hex(value: str) -> tuple[int, int, int]:
    raw = (value or "#000000").strip().lstrip("#")
    if len(raw) < 6:
        raw = (raw + "000000")[:6]
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def _f(elem: ET.Element, key: str, default: float = 0.0) -> float:
    raw = elem.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _path_numbers(d: str) -> tuple[float, ...]:
    tokens: list[float] = []
    buf = ""
    for ch in d.replace(",", " "):
        if ch.isalpha() or ch.isspace():
            if buf.strip():
                tokens.append(float(buf))
            buf = ""
            continue
        buf += ch
    if buf.strip():
        tokens.append(float(buf))
    return tuple(tokens)


@dataclass
class Shape:
    kind: str
    cx: float = 0.0
    cy: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    opacity: float = 1.0
    fill: tuple[int, int, int] = (0, 0, 0)
    stroke: tuple[int, int, int] | None = None
    stroke_w: float = 0.0
    numbers: tuple[float, ...] = ()
    clay: bool = False

    def copy(self) -> Shape:
        return Shape(
            self.kind,
            self.cx,
            self.cy,
            self.rx,
            self.ry,
            self.opacity,
            self.fill,
            self.stroke,
            self.stroke_w,
            self.numbers,
            self.clay,
        )


@dataclass
class Figure:
    skin: str
    state: str
    top: tuple[int, int, int]
    bottom: tuple[int, int, int]
    shapes: dict[str, Shape] = field(default_factory=dict)

    def copy(self) -> Figure:
        return Figure(
            self.skin,
            self.state,
            self.top,
            self.bottom,
            {key: shape.copy() for key, shape in self.shapes.items()},
        )


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def parse_svg(text: str, *, skin: str, state: str) -> Figure:
    root = ET.fromstring(text)
    top = (255, 248, 241)
    bottom = (240, 198, 164)
    for elem in root.iter():
        if _local(elem.tag) != "stop":
            continue
        color = _hex(elem.get("stop-color", "#000000"))
        offset = (elem.get("offset") or "0").rstrip("%")
        try:
            off = float(offset)
        except ValueError:
            off = 0.0
        if off <= 1.0:
            top = color
        else:
            bottom = color
    shapes: dict[str, Shape] = {}
    for elem in root.iter():
        tag = _local(elem.tag)
        sid = elem.get("id")
        if not sid or tag not in {"ellipse", "path"}:
            continue
        opacity = _f(elem, "opacity", 1.0)
        fill_raw = elem.get("fill") or "#000000"
        clay = fill_raw.startswith("url(")
        fill = (255, 255, 255) if clay else _hex(fill_raw if fill_raw != "none" else "#000000")
        stroke_raw = elem.get("stroke")
        stroke = None if not stroke_raw or stroke_raw == "none" else _hex(stroke_raw)
        stroke_w = _f(elem, "stroke-width", 2.4 if stroke else 0.0)
        if tag == "ellipse":
            shapes[sid] = Shape(
                "ellipse",
                _f(elem, "cx"),
                _f(elem, "cy"),
                _f(elem, "rx"),
                _f(elem, "ry"),
                opacity,
                fill,
                stroke,
                stroke_w,
                (),
                clay,
            )
        else:
            shapes[sid] = Shape(
                "path",
                0.0,
                0.0,
                0.0,
                0.0,
                opacity,
                fill,
                stroke,
                stroke_w,
                _path_numbers(elem.get("d") or ""),
                clay,
            )
    missing = [sid for sid in REQUIRED_IDS if sid not in shapes]
    if missing:
        raise ValueError(f"{skin}/{state} missing ids: {', '.join(missing)}")
    return Figure(skin, state, top, bottom, shapes)


def load_skin(skin_id: str, root: Path | None = None) -> dict[str, Figure]:
    folder = (root or SVG_DIR) / skin_id
    out: dict[str, Figure] = {}
    for state in STATE_IDS:
        path = folder / f"{state}.svg"
        out[state] = parse_svg(path.read_text(encoding="utf-8"), skin=skin_id, state=state)
    return out


def load_catalog(root: Path | None = None) -> dict[str, dict[str, Figure]]:
    return {skin_id: load_skin(skin_id, root) for skin_id in SKIN_IDS}


def lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], u: float) -> tuple[int, int, int]:
    return tuple(int(round(a[i] + (b[i] - a[i]) * u)) for i in range(3))  # type: ignore[return-value]


def lerp_shape(a: Shape, b: Shape, u: float) -> Shape:
    nums: tuple[float, ...] = ()
    if a.numbers and b.numbers and len(a.numbers) == len(b.numbers):
        nums = tuple(a.numbers[i] + (b.numbers[i] - a.numbers[i]) * u for i in range(len(a.numbers)))
    elif b.numbers:
        nums = b.numbers
    else:
        nums = a.numbers
    fill = lerp_color(a.fill, b.fill, u)
    stroke = None
    if a.stroke and b.stroke:
        stroke = lerp_color(a.stroke, b.stroke, u)
    else:
        stroke = b.stroke or a.stroke
    return Shape(
        b.kind or a.kind,
        a.cx + (b.cx - a.cx) * u,
        a.cy + (b.cy - a.cy) * u,
        a.rx + (b.rx - a.rx) * u,
        a.ry + (b.ry - a.ry) * u,
        a.opacity + (b.opacity - a.opacity) * u,
        fill,
        stroke,
        a.stroke_w + (b.stroke_w - a.stroke_w) * u,
        nums,
        a.clay or b.clay,
    )


def lerp_figure(a: Figure, b: Figure, u: float) -> Figure:
    u = 0.0 if u < 0 else 1.0 if u > 1 else u
    keys = list(dict.fromkeys([*a.shapes.keys(), *b.shapes.keys()]))
    shapes: dict[str, Shape] = {}
    for key in keys:
        left = a.shapes.get(key)
        right = b.shapes.get(key)
        if left is None or right is None or left.kind != right.kind:
            src = right if u >= 0.5 and right is not None else left
            if src is None:
                continue
            faded = src.copy()
            faded.opacity = src.opacity * (u if right is not None and left is None else (1.0 - u))
            shapes[key] = faded
            continue
        if left.numbers and right.numbers and len(left.numbers) != len(right.numbers):
            src = right if u >= 0.5 else left
            faded = src.copy()
            faded.opacity = src.opacity * (u if src is right else (1.0 - u))
            shapes[key] = faded
            continue
        shapes[key] = lerp_shape(left, right, u)
    return Figure(
        b.skin if u >= 0.5 else a.skin,
        b.state if u >= 0.5 else a.state,
        lerp_color(a.top, b.top, u),
        lerp_color(a.bottom, b.bottom, u),
        shapes,
    )
