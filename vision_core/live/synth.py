"""Synthetic screenshots for live-path tests and benchmarks (cv2 drawing only).

No real or explicit imagery: a "photo" is a procedural texture (gradient +
soft blobs + strokes at all orientations + sensor noise) in a random palette,
some low-contrast / light, some with text overlaid. A page mixes them with
``putText`` paragraphs, UI chrome, thumbnail grids (8–24 px gutters), a hero
image, and pictures clipped at the frame edge. ``marker=True`` paints a pure
magenta square inside a picture so a stub detector can "find" it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

Rect = tuple[int, int, int, int]
MARKER_BGR = (255, 0, 255)

_WORDS = (
    "the quick brown fox jumps over lazy dog screen guardian vision page news "
    "update weather market sports health science review open source local cloud "
    "frame region policy model budget cache scroll window browser privacy safety"
).split()

_PALETTES = {
    "vivid": [(40, 90, 200), (200, 150, 40), (60, 180, 90), (180, 60, 160), (30, 200, 230)],
    "muted": [(110, 120, 130), (90, 110, 100), (140, 130, 120), (120, 100, 110)],
    "warm": [(80, 140, 210), (60, 110, 190), (120, 170, 230), (50, 90, 150)],
    "cool": [(180, 120, 60), (150, 140, 90), (200, 170, 120), (120, 90, 40)],
    "dark": [(40, 30, 25), (70, 50, 40), (30, 40, 60), (60, 60, 70)],
    "light": [(236, 236, 232), (226, 230, 234), (240, 232, 226), (220, 226, 222)],
    "gray": [(60, 60, 60), (130, 130, 130), (200, 200, 200), (90, 90, 90)],
}


@dataclass
class SynthPage:
    image: np.ndarray
    pictures: list[Rect] = field(default_factory=list)  # GT picture rects
    texts: list[Rect] = field(default_factory=list)  # text block rects
    flagged: list[Rect] = field(default_factory=list)  # pictures with a marker
    markers: list[Rect] = field(default_factory=list)  # marker rects (page px)


def photo_texture(
    h: int,
    w: int,
    rng: np.random.Generator,
    *,
    palette: str | None = None,
    text_overlay: bool = False,
) -> np.ndarray:
    """A photo-like BGR texture of size ``h×w``."""
    name = palette or str(rng.choice(list(_PALETTES)))
    cols = np.array(_PALETTES[name], dtype=np.float32)
    light = name == "light"
    c0, c1 = cols[rng.integers(len(cols))], cols[rng.integers(len(cols))]
    angle = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (np.cos(angle) * xx / max(1, w) + np.sin(angle) * yy / max(1, h))
    t = (t - t.min()) / max(1e-6, float(t.max() - t.min()))
    img = c0[None, None, :] * (1 - t[..., None]) + c1[None, None, :] * t[..., None]
    layer = img.copy()
    n_blobs = int(rng.integers(4, 10))
    for _ in range(n_blobs):
        col = cols[rng.integers(len(cols))] + rng.normal(0, 18 if not light else 6, 3)
        center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        axes = (int(rng.integers(max(4, w // 10), max(6, w // 2))), int(rng.integers(max(4, h // 10), max(6, h // 2))))
        cv2.ellipse(layer, center, axes, float(rng.uniform(0, 180)), 0, 360, tuple(float(c) for c in col), -1)
    k = max(3, (min(h, w) // 12) | 1)
    layer = cv2.GaussianBlur(layer, (k, k), 0)
    img = 0.35 * img + 0.65 * layer
    # Mid-frequency strokes at every orientation (edges a photo has, text lacks).
    for _ in range(int(rng.integers(6, 16))):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        col = cols[rng.integers(len(cols))] * rng.uniform(0.7, 1.15)
        cv2.line(img, p1, p2, tuple(float(c) for c in col), int(rng.integers(2, 6)), cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = rng.normal(0, 3.0 if light else float(rng.uniform(3.0, 8.0)), img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    if light:
        img = np.clip(img.astype(np.int16), 0, 246).astype(np.uint8)
    if text_overlay:
        scale = max(0.8, min(h, w) / 300.0)
        cv2.putText(img, "SALE 50% OFF", (w // 10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), max(2, int(scale * 2)), cv2.LINE_AA)
    return img


def _sentence(rng: np.random.Generator, n: int) -> str:
    return " ".join(str(rng.choice(_WORDS)) for _ in range(n))


def draw_paragraph(
    img: np.ndarray,
    rect: Rect,
    rng: np.random.Generator,
    *,
    ink: tuple[int, int, int] = (34, 34, 34),
    scale: float = 1.0,
) -> Rect:
    """Fill ``rect`` with lines of ``putText`` (Retina-ish 13–16 pt)."""
    x1, y1, x2, y2 = rect
    thick = 2
    (tw, th), base = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    line_h = int(th * 2.0) + base
    y = y1 + th + 2
    bottom = y1
    while y + base <= y2:
        words = []
        line = ""
        while True:
            cand = (line + " " + str(rng.choice(_WORDS))).strip()
            if cv2.getTextSize(cand, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] > (x2 - x1):
                break
            line = cand
            words.append(cand)
        cv2.putText(img, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, thick, cv2.LINE_AA)
        bottom = y + base
        y += line_h
    return (x1, y1, x2, max(y1 + 1, bottom))


def draw_chrome(img: np.ndarray, rng: np.random.Generator, *, dark: bool = False) -> int:
    """Browser-like toolbar + tab strip; returns its height."""
    h, w = img.shape[:2]
    bar = int(h * 0.06)
    top = (52, 52, 56) if dark else (236, 236, 238)
    cv2.rectangle(img, (0, 0), (w, bar), top, -1)
    cv2.line(img, (0, bar), (w, bar), (200, 200, 200) if not dark else (80, 80, 80), 2)
    for i, col in enumerate(((95, 95, 255), (60, 190, 255), (90, 200, 90))):
        cv2.circle(img, (40 + i * 44, bar // 3), 12, col, -1, cv2.LINE_AA)
    for i in range(4):
        x = 200 + i * 360
        cv2.rectangle(img, (x, 10), (x + 330, bar // 2 - 4), (220, 220, 224) if not dark else (70, 70, 76), -1)
        cv2.putText(img, _sentence(rng, 2), (x + 16, bar // 2 - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (60, 60, 60) if not dark else (220, 220, 220), 2, cv2.LINE_AA)
    url = (bar // 2 + 8, bar - 12)
    cv2.rectangle(img, (220, url[0]), (w - 420, url[1]), (255, 255, 255) if not dark else (36, 36, 40), -1)
    cv2.rectangle(img, (220, url[0]), (w - 420, url[1]), (190, 190, 196), 2)
    cv2.putText(img, "https://example.test/" + _sentence(rng, 2).replace(" ", "/"), (250, url[1] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 90, 90) if not dark else (200, 200, 200), 2, cv2.LINE_AA)
    for i in range(5):
        cv2.rectangle(img, (w - 380 + i * 70, url[0]), (w - 330 + i * 70, url[1]), (200, 120, 40), -1)
    # Scrollbar
    cv2.rectangle(img, (w - 22, bar + 20), (w - 8, bar + 420), (180, 180, 180), -1)
    return bar


def _place(img: np.ndarray, rect: Rect, rng: np.random.Generator, page: SynthPage, *,
           palette: str | None = None, text_overlay: bool = False, marker: bool = False) -> None:
    x1, y1, x2, y2 = rect
    H, W = img.shape[:2]
    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        return
    tex = photo_texture(y2 - y1, x2 - x1, rng, palette=palette, text_overlay=text_overlay)
    img[cy1:cy2, cx1:cx2] = tex[cy1 - y1 : cy2 - y1, cx1 - x1 : cx2 - x1]
    page.pictures.append((cx1, cy1, cx2, cy2))
    if marker:
        mw, mh = max(16, (cx2 - cx1) // 4), max(16, (cy2 - cy1) // 4)
        mx = cx1 + (cx2 - cx1 - mw) // 2
        my = cy1 + (cy2 - cy1 - mh) // 2
        cv2.rectangle(img, (mx, my), (mx + mw - 1, my + mh - 1), MARKER_BGR, -1)
        page.flagged.append((cx1, cy1, cx2, cy2))
        page.markers.append((mx, my, mx + mw, my + mh))


def make_page(
    width: int = 3456,
    height: int = 2234,
    *,
    seed: int = 0,
    dark: bool = False,
    gutter: int | None = None,
    flagged_every: int = 0,
) -> SynthPage:
    """A ~20-picture desktop page at Retina scale.

    Layout: chrome bar; left text column; a hero image; a thumbnail grid with
    ``gutter`` px gutters; a right rail of mixed pictures/text; one picture
    clipped at the right edge and one at the bottom edge.
    ``flagged_every=k`` puts a marker in every k-th picture.
    """
    rng = np.random.default_rng(seed)
    bg = (30, 30, 32) if dark else (250, 250, 250)
    ink = (225, 225, 225) if dark else (34, 34, 34)
    img = np.full((height, width, 3), bg, dtype=np.uint8)
    page = SynthPage(img)
    sx = width / 3456.0
    sy = height / 2234.0
    top = draw_chrome(img, rng, dark=dark)
    g = int(gutter if gutter is not None else rng.integers(8, 25))
    count = [0]

    def pic(rect: Rect, **kw) -> None:
        count[0] += 1
        mark = bool(flagged_every) and count[0] % flagged_every == 0
        _place(img, rect, rng, page, marker=mark, **kw)

    def S(x: float, y: float) -> tuple[int, int]:
        return int(x * sx), int(y * sy)

    # Left text column with a heading and paragraphs.
    lx1, ly = S(60, 0)[0], top + int(60 * sy)
    lx2 = S(980, 0)[0]
    cv2.putText(img, _sentence(rng, 3).title(), (lx1, ly + 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, ink, 4, cv2.LINE_AA)
    ly += 110
    for i in range(3):
        rect = draw_paragraph(img, (lx1, ly, lx2, ly + int(300 * sy)), rng, ink=ink)
        page.texts.append(rect)
        ly = rect[3] + int(50 * sy)
        if i == 0:
            pic((lx1, ly, lx1 + int(450 * sx), ly + int(500 * sy)), text_overlay=True)
            rect = draw_paragraph(img, (lx1 + int(500 * sx), ly, lx2, ly + int(500 * sy)), rng, ink=ink)
            page.texts.append(rect)
            ly += int(550 * sy)

    # Hero image.
    hx1, hy1 = S(1060, 0)[0], top + int(60 * sy)
    hx2, hy2 = S(2460, 0)[0], top + int(660 * sy)
    pic((hx1, hy1, hx2, hy2))
    cap = draw_paragraph(img, (hx1, hy2 + 20, hx2, hy2 + 100), rng, ink=ink, scale=0.9)
    page.texts.append(cap)

    # Thumbnail grid under the hero.
    gx, gy = hx1, hy2 + 140
    cols, rows = 6, 2
    tw = (hx2 - hx1 - (cols - 1) * g) // cols
    th = int(tw * 0.8)
    palettes = list(_PALETTES)
    for r in range(rows):
        for c in range(cols):
            x = gx + c * (tw + g)
            y = gy + r * (th + g)
            pic((x, y, x + tw, y + th), palette=palettes[(r * cols + c) % len(palettes)])
    gy2 = gy + rows * (th + g)
    rect = draw_paragraph(img, (hx1, gy2 + 30, hx2, min(height - 40, gy2 + 420)), rng, ink=ink)
    page.texts.append(rect)

    # Two pictures under the left column (clipped by nothing, gutter g).
    by = max(ly, height - int(400 * sy))
    if by + int(280 * sy) < height - 20:
        pic((lx1, by, lx1 + int(420 * sx), by + int(300 * sy)), palette="dark" if not dark else "light")
        pic((lx1 + int(420 * sx) + g, by, lx1 + int(840 * sx) + g, by + int(300 * sy)))

    # Right rail: small cards (picture + text), last one clipped at the right edge.
    rx1 = S(2560, 0)[0]
    ry = top + int(60 * sy)
    for i in range(3):
        pic((rx1, ry, rx1 + int(380 * sx), ry + int(280 * sy)), palette="light" if i == 1 else None)
        t = draw_paragraph(img, (rx1 + int(410 * sx), ry, width - int(60 * sx), ry + int(280 * sy)), rng,
                           ink=ink, scale=0.9)
        page.texts.append(t)
        ry += int(330 * sy)
    pic((width - int(300 * sx), ry, width + int(200 * sx), ry + int(420 * sy)))
    # Clipped at the bottom edge.
    pic((rx1, height - int(260 * sy), rx1 + int(640 * sx), height + int(240 * sy)), palette="gray")
    return page


def scroll_view(page: np.ndarray, y0: int, height: int) -> np.ndarray:
    """A viewport into a taller page (content moves up by ``y0``)."""
    return np.ascontiguousarray(page[y0 : y0 + height])


def cartoon_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Flat-colour illustration (no noise): ellipses / polygons, curved edges."""
    cols = _PALETTES["vivid"] + _PALETTES["warm"]
    img = np.full((h, w, 3), cols[int(rng.integers(len(cols)))], dtype=np.uint8)
    for _ in range(int(rng.integers(6, 12))):
        col = tuple(int(c) for c in cols[int(rng.integers(len(cols)))])
        center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        axes = (int(rng.integers(w // 10, w // 3)), int(rng.integers(h // 10, h // 3)))
        cv2.ellipse(img, center, axes, float(rng.uniform(0, 180)), 0, 360, col, -1, cv2.LINE_AA)
        cv2.ellipse(img, center, axes, float(rng.uniform(0, 180)), 0, 360, (20, 20, 20), 3, cv2.LINE_AA)
    return img


def make_ui_page(width: int = 3456, height: int = 2234, *, seed: int = 0, dark: bool = False) -> SynthPage:
    """Hard negatives (headings, dense text, code, table, banner, form, chart,
    icons) next to a few positives (photos, a cartoon, text-on-photo)."""
    rng = np.random.default_rng(seed + 1000)
    bg = (30, 30, 32) if dark else (250, 250, 250)
    ink = (225, 225, 225) if dark else (34, 34, 34)
    img = np.full((height, width, 3), bg, dtype=np.uint8)
    page = SynthPage(img)
    top = draw_chrome(img, rng, dark=dark)
    y = top + 60
    # Big heading (taller than the minimum picture side).
    cv2.putText(img, "Breaking News Today", (60, y + 120), cv2.FONT_HERSHEY_SIMPLEX, 4.5, ink, 10, cv2.LINE_AA)
    page.texts.append((60, y, 1500, y + 150))
    y += 200
    # Dense paragraph: tight leading.
    x1, x2 = 60, 1100
    yy = y
    for _ in range(14):
        cv2.putText(img, _sentence(rng, 9), (x1, yy + 26), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ink, 2, cv2.LINE_AA)
        yy += 34
    page.texts.append((x1, y, x2, yy))
    y = yy + 50
    # Code block on a dark panel.
    panel = (40, 44, 52) if not dark else (16, 16, 18)
    cv2.rectangle(img, (60, y), (1100, y + 420), panel, -1)
    for i in range(10):
        cv2.putText(img, "def f(x): return x * %d  # %s" % (i, _sentence(rng, 2)), (90, y + 50 + i * 38),
                    cv2.FONT_HERSHEY_PLAIN, 2.2, (180, 220, 160) if i % 2 else (220, 180, 120), 2, cv2.LINE_AA)
    page.texts.append((60, y, 1100, y + 420))
    y += 470
    # Table with grid lines.
    tx1, ty1 = 60, y
    for r in range(7):
        for c in range(4):
            cx, cy = tx1 + c * 260, ty1 + r * 60
            cv2.rectangle(img, (cx, cy), (cx + 260, cy + 60), (200, 200, 200), 2)
            cv2.putText(img, _sentence(rng, 1), (cx + 14, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ink, 2, cv2.LINE_AA)
    page.texts.append((tx1, ty1, tx1 + 1040, ty1 + 420))

    # Middle column: banner, buttons, form, chart.
    mx = 1250
    y = top + 60
    cv2.rectangle(img, (mx, y), (mx + 1100, y + 260), (200, 110, 30), -1)
    cv2.putText(img, "Join the Summer Program", (mx + 50, y + 150), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (255, 255, 255), 6, cv2.LINE_AA)
    page.texts.append((mx, y, mx + 1100, y + 260))
    y += 320
    for i in range(4):
        bx = mx + i * 270
        cv2.rectangle(img, (bx, y), (bx + 240, y + 90), ((60, 160, 60), (60, 60, 200), (120, 120, 120), (200, 140, 40))[i], -1)
        cv2.putText(img, "Button", (bx + 40, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    y += 150
    for i in range(4):
        cv2.putText(img, _sentence(rng, 2).title(), (mx, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ink, 2, cv2.LINE_AA)
        cv2.rectangle(img, (mx, y + 45), (mx + 1100, y + 120), (255, 255, 255) if not dark else (44, 44, 48), -1)
        cv2.rectangle(img, (mx, y + 45), (mx + 1100, y + 120), (180, 180, 186), 2)
        y += 150
    page.texts.append((mx, y - 600, mx + 1100, y))
    # Bar chart (axis-aligned UI graphic).
    cy0 = y + 40
    cv2.line(img, (mx, cy0 + 400), (mx + 1100, cy0 + 400), ink, 3)
    cv2.line(img, (mx, cy0), (mx, cy0 + 400), ink, 3)
    for i in range(9):
        hgt = int(rng.integers(60, 380))
        cv2.rectangle(img, (mx + 30 + i * 118, cy0 + 400 - hgt), (mx + 120 + i * 118, cy0 + 398), (200, 130, 60), -1)
    page.texts.append((mx, cy0, mx + 1100, cy0 + 400))
    # Icon row (below min size).
    for i in range(12):
        ix = mx + i * 90
        cv2.rectangle(img, (ix, cy0 + 460), (ix + 60, cy0 + 520), _PALETTES["vivid"][i % 5], -1)

    # Small panels (< 1 % of the frame each, so not "background"): a promo
    # card with text, a code snippet, a gradient button, a chip row.
    sy0 = cy0 + 560
    cv2.rectangle(img, (mx, sy0), (mx + 470, sy0 + 140), (90, 40, 160), -1)
    cv2.putText(img, "New: Dark mode", (mx + 24, sy0 + 88), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.rectangle(img, (mx + 520, sy0), (mx + 1100, sy0 + 120), (58, 40, 40), -1)
    cv2.putText(img, "$ pip install guardian", (mx + 540, sy0 + 75), cv2.FONT_HERSHEY_PLAIN, 2.4, (120, 230, 120), 2, cv2.LINE_AA)
    grad = np.linspace(0, 1, 400, dtype=np.float32)[None, :, None]
    btn = (np.array([200, 120, 40], np.float32) * (1 - grad) + np.array([240, 180, 90], np.float32) * grad)
    img[sy0 + 170 : sy0 + 280, mx : mx + 400] = np.repeat(btn, 110, axis=0).astype(np.uint8)
    cv2.putText(img, "Subscribe", (mx + 90, sy0 + 245), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
    # The gradient button is deliberately not listed as a negative: a smooth
    # colour ramp is tonally picture-like and may be proposed (the detector
    # then clears it). Known gap, see regions.py.
    page.texts.extend([(mx, sy0, mx + 470, sy0 + 140), (mx + 520, sy0, mx + 1100, sy0 + 120)])

    # Right column: positives.
    rx = 2500
    y = top + 60
    _place(img, (rx, y, rx + 880, y + 520), rng, page, text_overlay=True)
    y += 560
    cart = cartoon_texture(440, 420, rng)
    img[y : y + 440, rx : rx + 420] = cart
    page.pictures.append((rx, y, rx + 420, y + 440))
    _place(img, (rx + 460, y, rx + 880, y + 440), rng, page, palette="light")
    y += 480
    _place(img, (rx, y, rx + 880, min(height, y + 600)), rng, page, palette="dark")
    return page
