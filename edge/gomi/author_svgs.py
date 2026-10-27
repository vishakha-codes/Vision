"""Write the five original clay skins. Run: python -m edge.gomi.author_svgs

Same element ids and path commands in every file so states can interpolate.
Not a copy of any icon pack, Coucou, or Mochi.
"""

from __future__ import annotations

from pathlib import Path

STATES = ("idle", "alert", "working", "clear", "sleep")
OUT = Path(__file__).resolve().parent / "svg"

# Shared ids. Ear/puff opacity is how a skin turns parts on.
IDS = (
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


def _skin(
    *,
    top: str,
    bottom: str,
    ink: str,
    blush: str,
    body: tuple[float, float, float, float],
    ear: float,
    puff: float,
    ear_l: tuple[float, float, float, float],
    ear_r: tuple[float, float, float, float],
    inner: str,
) -> dict:
    cx, cy, rx, ry = body
    return {
        "top": top,
        "bottom": bottom,
        "ink": ink,
        "blush": blush,
        "inner": inner,
        "body": body,
        "ear": ear,
        "puff": puff,
        "ear_l": ear_l,
        "ear_r": ear_r,
        "eyes": (cx - 16, cy - 6, cx + 16, cy - 6),
        "puff_l": (cx - rx * 0.72, cy + 8, 22, 18),
        "puff_r": (cx + rx * 0.72, cy + 10, 20, 16),
    }


SKINS: dict[str, dict] = {
    "pebble": _skin(
        top="#FFF8F1",
        bottom="#F0C6A4",
        ink="#3A2A22",
        blush="#F0A090",
        body=(80, 94, 46, 48),
        ear=0.0,
        puff=0.0,
        ear_l=(58, 46, 10, 8),
        ear_r=(102, 46, 10, 8),
        inner="#F6D2C0",
    ),
    "fox": _skin(
        top="#FFE6D2",
        bottom="#E07A48",
        ink="#3A2218",
        blush="#F09878",
        body=(80, 100, 40, 42),
        ear=1.0,
        puff=0.0,
        ear_l=(56, 58, 13, 28),
        ear_r=(104, 58, 13, 28),
        inner="#FFD0B4",
    ),
    "ghost": _skin(
        top="#FBF8FF",
        bottom="#D4CBE4",
        ink="#3A3550",
        blush="#E4C0D4",
        body=(80, 92, 42, 52),
        ear=0.0,
        puff=0.35,
        ear_l=(58, 40, 8, 6),
        ear_r=(102, 40, 8, 6),
        inner="#E4D8F0",
    ),
    "bear": _skin(
        top="#FFFEFB",
        bottom="#E6DFD6",
        ink="#2C2824",
        blush="#F0B0A4",
        body=(80, 98, 48, 44),
        ear=1.0,
        puff=0.0,
        ear_l=(50, 58, 16, 12),
        ear_r=(110, 58, 16, 12),
        inner="#E8B8A8",
    ),
    "cloud": _skin(
        top="#F7FCFF",
        bottom="#C5D9EC",
        ink="#243044",
        blush="#F0C0C8",
        body=(80, 96, 36, 32),
        ear=0.0,
        puff=1.0,
        ear_l=(58, 50, 8, 6),
        ear_r=(102, 50, 8, 6),
        inner="#E4EEF6",
    ),
}


def _mouth(kind: str) -> str:
    mouths = {
        "idle": (64, 112, 80, 124, 96, 112),
        "alert": (68, 114, 80, 116, 92, 114),
        "working": (70, 116, 80, 120, 90, 116),
        "clear": (58, 108, 80, 132, 102, 108),
        "sleep": (66, 118, 80, 110, 94, 118),
    }
    x1, y1, cx, cy, x2, y2 = mouths[kind]
    return f"M {x1:.1f} {y1:.1f} Q {cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}"


def _ear_path(cx: float, base_y: float, half_w: float, height: float) -> str:
    top_y = base_y - height
    return f"M {cx:.1f} {top_y:.1f} L {cx - half_w:.1f} {base_y:.1f} L {cx + half_w:.1f} {base_y:.1f} Z"


def _state_geom(base: dict, state: str) -> dict:
    cx, cy, rx, ry = base["body"]
    lx, ly, rx_, ry_ = base["eyes"]
    eye_rx, eye_ry = 7.2, 8.4
    pupil = 2.4
    pdx = 1.2
    blush_o = 0.18
    sat = 1.0
    if state == "alert":
        eye_ry = 11.2
        eye_rx = 8.0
        blush_o = 0.42
        ry *= 1.06
        cy -= 2
    elif state == "working":
        eye_ry = 7.6
        pdx = 5.5
        blush_o = 0.12
    elif state == "clear":
        eye_ry = 3.6
        eye_rx = 8.4
        blush_o = 0.55
        ry *= 1.03
    elif state == "sleep":
        eye_ry = 1.5
        eye_rx = 7.4
        blush_o = 0.05
        cy += 5
        sat = 0.72
    return {
        "body": (cx, cy, rx, ry),
        "eye_rx": eye_rx,
        "eye_ry": eye_ry,
        "pupil": pupil,
        "pdx": pdx,
        "blush_o": blush_o,
        "sat": sat,
        "left": (lx, ly),
        "right": (rx_, ry_),
    }


def _mix(hex_color: str, sat: float) -> str:
    r, g, b = _hex(hex_color)
    gray = (r + g + b) / 3.0
    r = int(round(gray + (r - gray) * sat))
    g = int(round(gray + (g - gray) * sat))
    b = int(round(gray + (b - gray) * sat))
    return f"#{r:02X}{g:02X}{b:02X}"


def _hex(value: str) -> tuple[int, int, int]:
    raw = value.lstrip("#")
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def svg_text(skin_id: str, state: str) -> str:
    base = SKINS[skin_id]
    geo = _state_geom(base, state)
    bcx, bcy, brx, bry = geo["body"]
    top = _mix(base["top"], geo["sat"])
    bottom = _mix(base["bottom"], geo["sat"])
    elx, ely, ehw, eh = base["ear_l"]
    erx, ery, ehw2, eh2 = base["ear_r"]
    plx, ply, prx, pry = base["puff_l"]
    prcx, prcy, prrx, prry = base["puff_r"]
    lx, ly = geo["left"]
    rx, ry = geo["right"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 160" data-skin="{skin_id}" data-state="{state}">',
        "  <defs>",
        '    <radialGradient id="clay" cx="38%" cy="32%" r="72%">',
        f'      <stop offset="0%" stop-color="{top}"/>',
        f'      <stop offset="100%" stop-color="{bottom}"/>',
        "    </radialGradient>",
        "  </defs>",
        f'  <ellipse id="shadow" cx="80" cy="148" rx="34" ry="7" fill="#1A1412" opacity="0.16"/>',
        f'  <path id="ear-l" d="{_ear_path(elx, ely, ehw, eh)}" fill="url(#clay)" opacity="{base["ear"]:.2f}"/>',
        f'  <path id="ear-r" d="{_ear_path(erx, ery, ehw2, eh2)}" fill="url(#clay)" opacity="{base["ear"]:.2f}"/>',
        f'  <ellipse id="ear-l-inner" cx="{elx:.1f}" cy="{ely - eh * 0.28:.1f}" rx="{ehw * 0.42:.1f}" ry="{eh * 0.28:.1f}" fill="{base["inner"]}" opacity="{base["ear"]:.2f}"/>',
        f'  <ellipse id="ear-r-inner" cx="{erx:.1f}" cy="{ery - eh2 * 0.28:.1f}" rx="{ehw2 * 0.42:.1f}" ry="{eh2 * 0.28:.1f}" fill="{base["inner"]}" opacity="{base["ear"]:.2f}"/>',
        f'  <ellipse id="puff-l" cx="{plx:.1f}" cy="{ply:.1f}" rx="{prx:.1f}" ry="{pry:.1f}" fill="url(#clay)" opacity="{base["puff"]:.2f}"/>',
        f'  <ellipse id="puff-r" cx="{prcx:.1f}" cy="{prcy:.1f}" rx="{prrx:.1f}" ry="{prry:.1f}" fill="url(#clay)" opacity="{base["puff"]:.2f}"/>',
        f'  <ellipse id="body" cx="{bcx:.1f}" cy="{bcy:.1f}" rx="{brx:.1f}" ry="{bry:.1f}" fill="url(#clay)" opacity="1"/>',
        f'  <ellipse id="highlight" cx="{bcx - brx * 0.28:.1f}" cy="{bcy - bry * 0.36:.1f}" rx="{brx * 0.36:.1f}" ry="{bry * 0.22:.1f}" fill="#FFFFFF" opacity="0.55"/>',
        f'  <ellipse id="blush" cx="{bcx:.1f}" cy="{bcy + bry * 0.22:.1f}" rx="{brx * 0.55:.1f}" ry="8" fill="{base["blush"]}" opacity="{geo["blush_o"]:.2f}"/>',
        f'  <ellipse id="left-eye" cx="{lx:.1f}" cy="{ly:.1f}" rx="{geo["eye_rx"]:.1f}" ry="{geo["eye_ry"]:.1f}" fill="{base["ink"]}" opacity="1"/>',
        f'  <ellipse id="right-eye" cx="{rx:.1f}" cy="{ry:.1f}" rx="{geo["eye_rx"]:.1f}" ry="{geo["eye_ry"]:.1f}" fill="{base["ink"]}" opacity="1"/>',
        f'  <ellipse id="left-pupil" cx="{lx + geo["pdx"]:.1f}" cy="{ly - 1.5:.1f}" rx="{geo["pupil"]:.1f}" ry="{geo["pupil"]:.1f}" fill="#FFFFFF" opacity="0.92"/>',
        f'  <ellipse id="right-pupil" cx="{rx + geo["pdx"]:.1f}" cy="{ry - 1.5:.1f}" rx="{geo["pupil"]:.1f}" ry="{geo["pupil"]:.1f}" fill="#FFFFFF" opacity="0.92"/>',
        f'  <path id="mouth" d="{_mouth(state)}" fill="none" stroke="{base["ink"]}" stroke-width="2.6" opacity="1"/>',
        "</svg>",
        "",
    ]
    return "\n".join(lines)


def write_all(root: Path | None = None) -> list[Path]:
    dest = root or OUT
    written: list[Path] = []
    for skin_id in SKINS:
        folder = dest / skin_id
        folder.mkdir(parents=True, exist_ok=True)
        for state in STATES:
            path = folder / f"{state}.svg"
            path.write_text(svg_text(skin_id, state), encoding="utf-8")
            written.append(path)
    return written


def main() -> None:
    paths = write_all()
    print(f"wrote {len(paths)} svg files under {OUT}")


if __name__ == "__main__":
    main()
