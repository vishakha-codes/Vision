"""Procedural benign "web page" corpus for the live-path eval.

Everything is drawn with cv2 + numpy from a seed. Nothing is downloaded and
there is no real or explicit imagery: a "picture" is a photo-like texture
(smooth gradients, soft blobs, mid-frequency texture, sensor noise) and a
``flagged`` picture is just a deterministic subset of those textures. The
ground truth is every picture rectangle in page pixels, plus stand-in
"part" boxes inside flagged pictures (what a body-part detector would
return), so MEDIUM masks can be scored too.

Layout units are CSS px; ``scale = width_px / 1728`` maps them to device
px, so 1728×1117 is 1× and 3456×2234 (Retina) is 2×.
"""

from __future__ import annotations

import argparse
import json
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

RETINA = (3456, 2234)
HALF = (1728, 1117)
RESOLUTIONS = {"retina": RETINA, "half": HALF}
CSS_WIDTH = 1728.0

LAYOUTS = ("news", "image_search", "shopping", "video", "social")

Rect = tuple[int, int, int, int]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BG = (250, 250, 250)
_INK = (34, 34, 34)
_MUTED = (120, 120, 120)
_RULE = (226, 226, 226)
_ACCENTS = ((210, 120, 20), (60, 60, 200), (40, 150, 60), (160, 60, 150), (20, 140, 220))

# Palettes in BGR. "light" is the low-contrast case (product shots on white,
# pale skies); the rest give varied photo colour statistics.
_PALETTES = {
    "vivid": [(40, 90, 200), (200, 150, 40), (60, 180, 90), (180, 60, 160), (30, 200, 230)],
    "muted": [(110, 120, 130), (90, 110, 100), (140, 130, 120), (120, 100, 110), (100, 120, 140)],
    "warm": [(80, 140, 210), (60, 110, 190), (120, 170, 230), (90, 120, 160), (50, 90, 150)],
    "cool": [(180, 120, 60), (150, 140, 90), (200, 170, 120), (120, 90, 40), (170, 160, 150)],
    "dark": [(40, 30, 25), (70, 50, 40), (30, 40, 60), (60, 60, 70), (20, 25, 30)],
    "light": [(240, 240, 238), (232, 236, 240), (244, 238, 232), (226, 230, 228), (236, 236, 244)],
}


@dataclass
class Picture:
    """One picture on a page. ``xyxy`` is in page device pixels."""

    pic_id: str
    xyxy: Rect
    kind: str = "picture"  # "picture" | "video"
    flagged: bool = False
    style: str = "vivid"
    tags: tuple[str, ...] = ()
    parts: tuple[Rect, ...] = ()  # stand-in body-part boxes (flagged only)

    def to_dict(self) -> dict:
        return {
            "id": self.pic_id,
            "xyxy": list(self.xyxy),
            "kind": self.kind,
            "flagged": self.flagged,
            "style": self.style,
            "tags": list(self.tags),
            "parts": [list(p) for p in self.parts],
        }


@dataclass
class Page:
    page_id: str
    layout: str
    seed: int
    scale: float
    image: np.ndarray
    pictures: list[Picture] = field(default_factory=list)

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def picture(self, pic_id: str) -> Picture:
        for pic in self.pictures:
            if pic.pic_id == pic_id:
                return pic
        raise KeyError(pic_id)

    def gt(self) -> dict:
        return {
            "page_id": self.page_id,
            "layout": self.layout,
            "seed": self.seed,
            "scale": self.scale,
            "width": self.width,
            "height": self.height,
            "pictures": [p.to_dict() for p in self.pictures],
        }


def stable_seed(*parts: object) -> int:
    """Seed that does not depend on PYTHONHASHSEED."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


# --------------------------------------------------------------------------- textures


def photo_texture(h: int, w: int, rng: np.random.Generator, style: str = "vivid") -> np.ndarray:
    """Photo-like BGR texture: gradient + soft blobs + texture + noise.

    Built at about 1/12 resolution and upscaled, so a Retina hero costs a few
    ms of drawing plus one noise pass.
    """
    h = max(2, int(h))
    w = max(2, int(w))
    pal = np.array(_PALETTES[style], np.float32)
    jitter = 18.0 if style != "light" else 6.0
    pal = np.clip(pal + rng.normal(0.0, jitter, pal.shape), 0, 255)
    sh = max(4, h // 12)
    sw = max(4, w // 12)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    xx /= max(1, sw - 1)
    yy /= max(1, sh - 1)
    ang = rng.uniform(0, np.pi)
    t = np.cos(ang) * xx + np.sin(ang) * yy
    t = (t - t.min()) / max(1e-6, float(t.max() - t.min()))
    c0, c1 = pal[rng.choice(len(pal), 2, replace=False)]
    img = c0[None, None, :] * (1 - t[..., None]) + c1[None, None, :] * t[..., None]
    if rng.random() < 0.5:  # horizon / subject-background split
        hy = rng.uniform(0.35, 0.7)
        band = 1.0 / (1.0 + np.exp(-(yy - hy) * 30.0))
        c2 = pal[rng.integers(len(pal))]
        img = img * (1 - 0.6 * band[..., None]) + c2[None, None, :] * 0.6 * band[..., None]
    for _ in range(int(rng.integers(3, 9))):
        cx, cy = rng.uniform(0, 1), rng.uniform(0, 1)
        rx, ry = rng.uniform(0.06, 0.35), rng.uniform(0.06, 0.35)
        g = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
        col = pal[rng.integers(len(pal))] + rng.normal(0, 20, 3)
        a = rng.uniform(0.3, 0.8) * g[..., None]
        img = img * (1 - a) + col[None, None, :] * a
    tex = rng.normal(0.0, 10.0 if style != "light" else 3.0, (sh * 2, sw * 2)).astype(np.float32)
    tex = cv2.resize(cv2.GaussianBlur(tex, (0, 0), 1.2), (sw, sh), interpolation=cv2.INTER_AREA)
    img = img + tex[..., None]
    out = cv2.resize(np.clip(img, 0, 255).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    noise = rng.standard_normal((h, w), dtype=np.float32) * (4.0 if style != "light" else 1.5)
    out += noise[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


@lru_cache(maxsize=8)
def _video_base(h: int, w: int, seed: int, scene: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed("video", seed, scene))
    return photo_texture(max(8, h // 4) + 40, max(8, w // 4) + 80, rng, style=("vivid", "warm", "cool")[scene % 3])


def video_texture(h: int, w: int, seed: int, t: int, *, scene: int = 0) -> np.ndarray:
    """Animated photo-like content: a slow pan + drift, scene cut on ``scene``."""
    big = _video_base(h, w, seed, scene)
    dx = int((t * 3) % 80)
    dy = int(20 + 15 * np.sin(t / 5.0))
    crop = big[dy : dy + max(8, h // 4), dx : dx + max(8, w // 4)]
    out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    shade = 0.9 + 0.1 * np.sin(t / 3.0)
    noise = np.random.default_rng(stable_seed("vnoise", seed, t)).standard_normal((h, w), dtype=np.float32) * 3.0
    out = out * shade + noise[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- drawing


_WORDS = (
    "the of and to in is for on with as by at from that this city new report data team "
    "market season local weather school river garden project travel coffee music design "
    "week update photo guide review open early light green summer winter island morning"
).split()


class Canvas:
    """A page being drawn. All public coordinates are CSS px."""

    def __init__(self, page_id: str, layout: str, seed: int, width_px: int, height_css: float, scale: float):
        self.page_id = page_id
        self.layout = layout
        self.seed = seed
        self.s = scale
        self.w_css = width_px / scale
        h_px = int(round(height_css * scale))
        self.img = np.full((h_px, width_px, 3), _BG, np.uint8)
        self.rng = np.random.default_rng(seed)
        self.pictures: list[Picture] = []

    @property
    def h_css(self) -> float:
        return self.img.shape[0] / self.s

    def px(self, v: float) -> int:
        return int(round(v * self.s))

    def r(self, x1: float, y1: float, x2: float, y2: float) -> Rect:
        H, W = self.img.shape[:2]
        return (
            max(0, min(W, self.px(x1))),
            max(0, min(H, self.px(y1))),
            max(0, min(W, self.px(x2))),
            max(0, min(H, self.px(y2))),
        )

    def rect(self, x1, y1, x2, y2, color, *, filled: bool = True, radius: float = 0.0) -> None:
        a, b, c, d = self.r(x1, y1, x2, y2)
        if c <= a or d <= b:
            return
        th = -1 if filled else max(1, self.px(1))
        rad = self.px(radius)
        if rad <= 1 or not filled:
            cv2.rectangle(self.img, (a, b), (c - 1, d - 1), color, th)
            return
        rad = min(rad, (c - a) // 2, (d - b) // 2)
        cv2.rectangle(self.img, (a + rad, b), (c - 1 - rad, d - 1), color, -1)
        cv2.rectangle(self.img, (a, b + rad), (c - 1, d - 1 - rad), color, -1)
        for cx, cy in ((a + rad, b + rad), (c - 1 - rad, b + rad), (a + rad, d - 1 - rad), (c - 1 - rad, d - 1 - rad)):
            cv2.circle(self.img, (cx, cy), rad, color, -1, cv2.LINE_AA)

    def text(self, x, y, s: str, *, size: float = 14.0, color=_INK, bold: bool = False) -> None:
        scale = size * self.s / 30.0
        th = max(1, int(round(scale * (2.4 if bold else 1.4))))
        cv2.putText(self.img, s, (self.px(x), self.px(y)), _FONT, scale, color, th, cv2.LINE_AA)

    def words(self, n: int) -> str:
        return " ".join(self.rng.choice(_WORDS, n))

    def paragraph(self, x, y, width, lines: int, *, size: float = 14.0, color=_INK) -> float:
        """Draw ``lines`` of pseudo-text; return the y below the paragraph."""
        line_h = size * 1.6
        chars = max(8, int(width / (size * 0.62)))
        for i in range(lines):
            n = max(2, int(chars / 6.2 * (0.55 if i == lines - 1 else self.rng.uniform(0.85, 1.0))))
            line = self.words(n)[:chars]
            self.text(x, y + size + i * line_h, line, size=size, color=color)
        return y + lines * line_h + size * 0.8

    def button(self, x, y, w, h, label: str, *, color=None, fg=(255, 255, 255)) -> None:
        color = color or _ACCENTS[int(self.rng.integers(len(_ACCENTS)))]
        self.rect(x, y, x + w, y + h, color, radius=h / 2.5)
        self.text(x + h * 0.45, y + h * 0.68, label, size=h * 0.45, color=fg, bold=True)

    def icon(self, x, y, size: float) -> None:
        kind = int(self.rng.integers(4))
        col = _MUTED
        cx, cy, r = self.px(x + size / 2), self.px(y + size / 2), max(2, self.px(size / 2.6))
        if kind == 0:
            cv2.circle(self.img, (cx, cy), r, col, max(1, self.px(1.6)), cv2.LINE_AA)
        elif kind == 1:
            for i in range(3):
                yy = self.px(y + size * (0.3 + 0.2 * i))
                cv2.line(self.img, (self.px(x + size * 0.2), yy), (self.px(x + size * 0.8), yy), col, max(1, self.px(1.6)))
        elif kind == 2:
            pts = np.array([[cx, cy - r], [cx + r, cy + r], [cx - r, cy + r]], np.int32)
            cv2.polylines(self.img, [pts], True, col, max(1, self.px(1.6)), cv2.LINE_AA)
        else:
            cv2.rectangle(self.img, (cx - r, cy - r), (cx + r, cy + r), col, max(1, self.px(1.6)))

    def avatar(self, x, y, d: float) -> None:
        """UI chrome, not a picture: flat circle with an initial."""
        col = tuple(int(v) for v in self.rng.integers(90, 220, 3))
        cv2.circle(self.img, (self.px(x + d / 2), self.px(y + d / 2)), self.px(d / 2), col, -1, cv2.LINE_AA)
        self.text(x + d * 0.3, y + d * 0.68, chr(65 + int(self.rng.integers(26))), size=d * 0.5, color=(255, 255, 255), bold=True)

    def picture(
        self,
        x1,
        y1,
        x2,
        y2,
        *,
        style: str | None = None,
        kind: str = "picture",
        tags: tuple[str, ...] = (),
        overlay_text: bool | None = None,
    ) -> Picture | None:
        rect = self.r(x1, y1, x2, y2)
        a, b, c, d = rect
        if c - a < 8 or d - b < 8:
            return None
        style = style or str(self.rng.choice(["vivid", "muted", "warm", "cool", "dark", "vivid", "warm"]))
        # Texture draws use their own stream so the layout (self.rng) is the
        # same at every resolution.
        tex_rng = np.random.default_rng(stable_seed("tex", self.seed, len(self.pictures)))
        tex = photo_texture(d - b, c - a, tex_rng, style)
        tags = tuple(tags)
        if style == "light":
            tags += ("low_contrast",)
        if overlay_text is None:
            overlay_text = bool(self.rng.random() < 0.15)
        if overlay_text:
            h_css = (y2 - y1)
            size = max(10.0, min(36.0, h_css * 0.09))
            band_top = (d - b) - int(size * self.s * 2.0)
            tex[band_top:] = (tex[band_top:].astype(np.float32) * 0.45).astype(np.uint8)
            scale = size * self.s / 30.0
            cv2.putText(tex, self.words(3).upper(), (int(size * self.s * 0.6), (d - b) - int(size * self.s * 0.6)),
                        _FONT, scale, (255, 255, 255), max(1, int(scale * 2.4)), cv2.LINE_AA)
            tags += ("text_overlay",)
        W = self.img.shape[1]
        if a == 0 or c == W:
            tags += ("edge_touch",)
        self.img[b:d, a:c] = tex
        pic = Picture(pic_id=f"p{len(self.pictures):03d}", xyxy=rect, kind=kind, style=style, tags=tags)
        self.pictures.append(pic)
        return pic

    def hrule(self, x1, x2, y) -> None:
        self.rect(x1, y, x2, y + 1, _RULE)

    def finish(self, flagged_rate: float = 0.3) -> Page:
        _assign_flags(self.pictures, np.random.default_rng(stable_seed("flags", self.seed)), flagged_rate)
        return Page(self.page_id, self.layout, self.seed, self.s, self.img, self.pictures)


def _parts_for(rect: Rect, rng: np.random.Generator) -> tuple[Rect, ...]:
    """1–3 non-overlapping "body-part sized" boxes inside a flagged picture."""
    x1, y1, x2, y2 = rect
    w, h = x2 - x1, y2 - y1
    n = int(rng.integers(1, 4))
    out: list[Rect] = []
    rows = np.linspace(y1 + 0.12 * h, y2 - 0.12 * h, n + 1)
    for i in range(n):
        pw = int(w * rng.uniform(0.18, 0.34))
        ph = int(min(rows[i + 1] - rows[i], h * rng.uniform(0.14, 0.26)))
        if pw < 4 or ph < 4:
            continue
        cx = x1 + int(w * rng.uniform(0.3, 0.7))
        py = int(rows[i])
        out.append((max(x1, cx - pw // 2), py, min(x2, cx + pw // 2), py + ph))
    return tuple(out)


def _assign_flags(pictures: list[Picture], rng: np.random.Generator, rate: float) -> None:
    if not pictures:
        return
    flags = rng.random(len(pictures)) < rate
    if len(pictures) >= 2:
        if not flags.any():
            flags[int(rng.integers(len(pictures)))] = True
        if flags.all():
            flags[int(rng.integers(len(pictures)))] = False
    else:
        flags[:] = True
    for pic, flag in zip(pictures, flags):
        pic.flagged = bool(flag)
        pic.parts = _parts_for(pic.xyxy, rng) if pic.flagged else ()


# --------------------------------------------------------------------------- layouts


def _nav(cv: Canvas, *, h: float = 64.0, search: bool = False) -> float:
    W = cv.w_css
    cv.rect(0, 0, W, h, (255, 255, 255))
    cv.hrule(0, W, h - 1)
    cv.text(28, h * 0.62, cv.words(1).capitalize() + " " + cv.words(1).capitalize(), size=22, bold=True, color=_ACCENTS[0])
    if search:
        cv.rect(260, h * 0.2, min(W - 240, 980), h * 0.8, (241, 243, 244), radius=h * 0.3)
        cv.text(290, h * 0.6, cv.words(3), size=15, color=_MUTED)
    x = W - 230
    for _ in range(4):
        cv.icon(x, h * 0.3, h * 0.4)
        x += h * 0.6
    cv.avatar(W - 64, h * 0.18, h * 0.64)
    return h


def _news(cv: Canvas) -> None:
    W = cv.w_css
    y = _nav(cv)
    x = 120
    for _ in range(7):
        cv.text(x, y + 30, cv.words(1).capitalize(), size=14, color=_MUTED)
        x += 120
    y += 48
    cv.hrule(0, W, y)
    # full-bleed hero touching both page edges
    hero_h = 560
    cv.picture(0, y, W, y + hero_h, overlay_text=True)
    y += hero_h + 32
    col_x, col_w = 200, 860
    side_x = col_x + col_w + 60
    cv.text(col_x, y + 34, cv.words(7).capitalize(), size=34, bold=True)
    cv.text(col_x, y + 66, cv.words(5), size=14, color=_MUTED)
    cv.avatar(col_x, y + 80, 36)
    cv.text(col_x + 48, y + 104, cv.words(2).title(), size=14)
    ys = y + 140
    y_body = ys
    while y_body < cv.h_css - 400:
        y_body = cv.paragraph(col_x, y_body, col_w, int(cv.rng.integers(4, 9)), size=15)
        if cv.rng.random() < 0.45:
            ih = float(cv.rng.uniform(280, 480))
            inset = 0 if cv.rng.random() < 0.6 else 120
            style = "light" if cv.rng.random() < 0.25 else None
            cv.picture(col_x + inset, y_body, col_x + col_w - inset, y_body + ih, style=style)
            cv.text(col_x + inset, y_body + ih + 22, cv.words(8), size=12, color=_MUTED)
            y_body += ih + 44
    # sidebar: "most read" with small thumbs + an ad box
    sy = ys
    cv.text(side_x, sy, "Most read", size=18, bold=True)
    sy += 20
    while sy < cv.h_css - 500:
        cv.picture(side_x, sy, side_x + 120, sy + 84)
        cv.paragraph(side_x + 134, sy - 4, 260, 3, size=13)
        sy += 108
        if cv.rng.random() < 0.15:
            cv.rect(side_x, sy, side_x + 300, sy + 250, (238, 238, 238))
            cv.text(side_x + 120, sy + 130, "AD", size=20, color=_MUTED)
            sy += 270
    fy = cv.h_css - 220
    cv.rect(0, fy, W, cv.h_css, (40, 40, 44))
    for i in range(4):
        cv.paragraph(120 + i * 380, fy + 40, 300, 4, size=13, color=(200, 200, 200))


def _image_search(cv: Canvas) -> None:
    W = cv.w_css
    y = _nav(cv, h=72, search=True)
    x = 260
    for _ in range(8):
        label = cv.words(1)
        cv.rect(x, y + 14, x + 18 + len(label) * 9, y + 46, (235, 238, 242), radius=14)
        cv.text(x + 10, y + 36, label, size=14)
        x += 36 + len(label) * 9
    y += 64
    # justified rows with varied thumbnail aspect, gutters 8–24 device px
    gutter_px = int(cv.rng.integers(8, 25))
    gut = gutter_px / cv.s
    margin = 16
    first = True
    while y < cv.h_css - 260:
        row_h = float(cv.rng.uniform(150, 280))
        aspects = []
        total = 0.0
        while total < (W - 2 * margin) or len(aspects) < 2:
            a = float(cv.rng.choice([0.66, 0.75, 1.0, 1.33, 1.5, 1.78]))
            aspects.append(a)
            total += a * row_h + gut
        widths = np.array(aspects) * row_h
        widths *= (W - 2 * margin - gut * (len(widths) - 1)) / widths.sum()
        x = margin if not first else 0  # first row bleeds to the page edge
        for w in widths:
            style = "light" if cv.rng.random() < 0.15 else None
            cv.picture(x, y, x + w, y + row_h, style=style)
            cv.text(x + 2, y + row_h + 16, cv.words(2)[: max(4, int(w / 9))], size=12, color=_MUTED)
            cv.text(x + 2, y + row_h + 32, cv.words(1) + ".com", size=11, color=(90, 140, 90))
            x += w + gut
        first = False
        y += row_h + 40 + gut
    cv.button(W / 2 - 90, cv.h_css - 120, 180, 44, "More results", color=(232, 234, 237), fg=_INK)


def _shopping(cv: Canvas) -> None:
    W = cv.w_css
    y = _nav(cv, search=True)
    cv.rect(0, y, W, y + 44, (35, 47, 62))
    for i in range(9):
        cv.text(24 + i * 150, y + 28, cv.words(1).capitalize(), size=14, color=(255, 255, 255))
    y += 64
    side_w = 260
    sy = y
    for _ in range(5):
        cv.text(24, sy + 18, cv.words(1).title(), size=16, bold=True)
        sy += 30
        for _ in range(5):
            cv.rect(24, sy, 40, sy + 16, (120, 120, 120), filled=False)
            cv.text(50, sy + 14, cv.words(2), size=13)
            sy += 26
        sy += 14
    cols = 4
    gx = side_w + 20
    card_w = (W - gx - 20 - 24 * (cols - 1)) / cols
    card_h = card_w + 150
    row = 0
    while y + card_h < cv.h_css - 100:
        for c in range(cols):
            x = gx + c * (card_w + 24)
            cv.rect(x, y, x + card_w, y + card_h, (255, 255, 255))
            style = "light" if cv.rng.random() < 0.4 else None
            cv.picture(x + 12, y + 12, x + card_w - 12, y + card_w - 12, style=style)
            ty = y + card_w + 6
            cv.text(x + 12, ty + 18, cv.words(4)[:30], size=14)
            for k in range(5):
                cv.text(x + 12 + k * 16, ty + 44, "*", size=18, color=(0, 160, 240))
            cv.text(x + 12, ty + 76, f"${int(cv.rng.integers(5, 400))}.{int(cv.rng.integers(0, 99)):02d}", size=20, bold=True)
            cv.button(x + 12, ty + 92, 130, 34, "Add to cart", color=(0, 200, 255), fg=_INK)
        y += card_h + 24
        row += 1


def _video(cv: Canvas) -> None:
    W = cv.w_css
    y = _nav(cv, h=56, search=True)
    y += 24
    px1 = 24
    pw = W * 0.66
    ph = pw * 9 / 16
    cv.rect(px1, y, px1 + pw, y + ph, (0, 0, 0))
    pic = cv.picture(px1, y, px1 + pw, y + ph - 40, style="vivid", kind="video", overlay_text=False)
    # controls bar (chrome, below the picture area)
    cv.rect(px1, y + ph - 40, px1 + pw, y + ph, (20, 20, 20))
    cv.rect(px1 + 60, y + ph - 22, px1 + pw - 200, y + ph - 18, (90, 90, 90))
    cv.rect(px1 + 60, y + ph - 22, px1 + pw * 0.35, y + ph - 18, (40, 40, 230))
    cv.text(px1 + 16, y + ph - 12, ">", size=20, color=(255, 255, 255), bold=True)
    if pic is not None:
        pic.tags += ("player",)
    ty = y + ph + 16
    cv.text(px1, ty + 28, cv.words(8).capitalize(), size=22, bold=True)
    cv.avatar(px1, ty + 48, 44)
    cv.text(px1 + 56, ty + 76, cv.words(2).title(), size=15, bold=True)
    cv.button(px1 + 260, ty + 50, 120, 40, "Subscribe", color=(15, 15, 15))
    for i in range(3):
        cv.button(px1 + pw - 420 + i * 140, ty + 50, 120, 40, cv.words(1), color=(240, 240, 240), fg=_INK)
    dy = cv.paragraph(px1, ty + 110, pw, 4, size=14)
    while dy < cv.h_css - 120:
        cv.avatar(px1, dy, 40)
        dy = cv.paragraph(px1 + 56, dy - 4, pw - 60, int(cv.rng.integers(1, 4)), size=14) + 12
    sx = px1 + pw + 24
    sy = y
    while sy < cv.h_css - 140:
        cv.picture(sx, sy, sx + 168, sy + 94, overlay_text=False)
        cv.rect(sx + 128, sy + 74, sx + 164, sy + 90, (0, 0, 0))
        cv.paragraph(sx + 180, sy - 4, W - sx - 200, 2, size=13)
        cv.text(sx + 180, sy + 70, cv.words(2), size=12, color=_MUTED)
        sy += 110


def _social(cv: Canvas) -> None:
    W = cv.w_css
    y0 = _nav(cv, h=60)
    for i in range(8):
        cv.icon(40, y0 + 30 + i * 56, 26)
        cv.text(80, y0 + 50 + i * 56, cv.words(1).capitalize(), size=17)
    fx, fw = 420, 640
    y = y0 + 20
    while y < cv.h_css - 300:
        cv.rect(fx, y, fx + fw, y + 4, _RULE)
        y += 16
        cv.avatar(fx + 12, y, 48)
        cv.text(fx + 72, y + 22, cv.words(2).title(), size=15, bold=True)
        cv.text(fx + 72, y + 42, "@" + cv.words(1) + " . 2h", size=13, color=_MUTED)
        y = cv.paragraph(fx + 72, y + 52, fw - 90, int(cv.rng.integers(1, 4)), size=15)
        if cv.rng.random() < 0.75:
            ih = float(cv.rng.uniform(260, 520))
            style = "light" if cv.rng.random() < 0.2 else None
            if cv.rng.random() < 0.3:
                half = (fw - 90 - 6) / 2
                cv.picture(fx + 72, y, fx + 72 + half, y + ih, style=style)
                cv.picture(fx + 72 + half + 6, y, fx + fw - 18, y + ih)
            else:
                cv.picture(fx + 72, y, fx + fw - 18, y + ih, style=style, overlay_text=bool(cv.rng.random() < 0.3))
            y += ih + 12
        for k in range(4):
            cv.icon(fx + 72 + k * 130, y, 22)
            cv.text(fx + 100 + k * 130, y + 17, str(int(cv.rng.integers(1, 999))), size=13, color=_MUTED)
        y += 44
    rx = fx + fw + 40
    cv.rect(rx, y0 + 20, W - 40, y0 + 520, (245, 248, 250), radius=16)
    cv.text(rx + 20, y0 + 56, "Who to follow", size=19, bold=True)
    for i in range(6):
        yy = y0 + 80 + i * 70
        cv.avatar(rx + 20, yy, 48)
        cv.text(rx + 80, yy + 22, cv.words(2).title(), size=14, bold=True)
        cv.button(W - 160, yy + 6, 96, 34, "Follow", color=(15, 15, 15))


_BUILDERS = {
    "news": (_news, 3000.0),
    "image_search": (_image_search, 3000.0),
    "shopping": (_shopping, 2600.0),
    "video": (_video, 2000.0),
    "social": (_social, 3200.0),
}


def make_page(
    layout: str,
    seed: int = 0,
    *,
    width_px: int = HALF[0],
    scale: float | None = None,
    height_css: float | None = None,
    variant: int = 0,
    flagged_rate: float = 0.3,
) -> Page:
    """Render one page. Same ``(layout, seed, width_px, variant)`` → same pixels and GT."""
    if layout not in _BUILDERS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")
    builder, default_h = _BUILDERS[layout]
    scale = float(scale) if scale else width_px / CSS_WIDTH
    page_seed = stable_seed(layout, seed, variant)
    cv = Canvas(f"{layout}-{seed}-{variant}", layout, page_seed, width_px, height_css or default_h, scale)
    builder(cv)
    return cv.finish(flagged_rate)


# --------------------------------------------------------------------------- desktop scenes
#
# A macOS-like desktop around the pages: wallpaper (photo, gradient or solid),
# a frosted menu bar with a clock, a frosted Dock with icons (and a magnified
# frame), app windows with title bars and soft drop shadows. Everything is
# drawn with cv2 + numpy; window bounds are what the window server would
# report, so the eval can hand the engine a window list with every frame.

WALLPAPERS = ("photo", "gradient", "solid")
MENU_PT = 37  # menu bar height (points)
DOCK_PT = (728, 60, 3)  # Dock width, height, gap above the screen bottom (points)
VISIBLE_BOTTOM_PT = 63  # NSScreen.visibleFrame stops this far above the bottom (Dock + gap)
TITLE_PT = 52  # window title bar + toolbar (points)


def desktop_scale(width: int) -> float:
    return width / CSS_WIDTH


@lru_cache(maxsize=6)
def _wallpaper(kind: str, width: int, height: int, seed: int) -> np.ndarray:
    if kind == "solid":
        return np.full((height, width, 3), (118, 72, 38), np.uint8)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    yy /= max(1, height - 1)
    xx /= max(1, width - 1)
    if kind == "gradient":
        b = 170 - 90 * yy + 20 * xx
        g = 110 - 40 * yy + 30 * xx
        r = 60 + 90 * yy + 10 * xx
        return np.clip(np.stack([b, g, r], -1), 0, 255).astype(np.uint8)
    if kind != "photo":
        raise ValueError(f"unknown wallpaper {kind!r}; expected one of {WALLPAPERS}")
    rng = np.random.default_rng(stable_seed("wallpaper", seed, width))
    img = photo_texture(height, width, rng, style="cool").astype(np.float32)
    sky = np.stack([200 - 60 * yy, 150 - 40 * yy, 90 + 30 * yy], -1)
    return np.clip(0.5 * img + 0.5 * sky, 0, 255).astype(np.uint8)


def desktop_wallpaper(kind: str, width: int, height: int, seed: int = 7) -> np.ndarray:
    """A fresh copy of the wallpaper (cached render)."""
    return _wallpaper(kind, int(width), int(height), int(seed)).copy()


def desktop_rects(width: int, height: int, *, magnify: bool = False) -> dict[str, Rect]:
    """Menu bar, Dock window and visible frame (``NSScreen.visibleFrame``) in px."""
    s = desktop_scale(width)
    menu = (0, 0, width, int(round(MENU_PT * s)))
    dw, dh, gap = DOCK_PT
    if magnify:
        dw, dh = dw + 120, dh + 56  # magnified icons: the Dock window grows up and out
    cx = width // 2
    dock = (cx - int(round(dw * s / 2)), height - int(round((dh + gap) * s)), cx + int(round(dw * s / 2)), height - int(round(gap * s)))
    visible = (0, menu[3], width, height - int(round(VISIBLE_BOTTOM_PT * s)))
    return {"menu": menu, "dock": dock, "visible": visible}


def _frost(img: np.ndarray, rect: Rect, lighten: float) -> None:
    """Frosted-glass strip: what is under it, blurred hard and lightened."""
    x1, y1, x2, y2 = rect
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return
    h, w = crop.shape[:2]
    small = cv2.resize(crop, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small.astype(np.float32), (0, 0), 3.0)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    img[y1:y2, x1:x2] = np.clip(big * (1 - lighten) + 255 * lighten, 0, 255).astype(np.uint8)


def draw_shadow(img: np.ndarray, rect: Rect, s: float, strength: float = 0.45) -> None:
    """Soft drop shadow around (mostly below) a window, computed at 1/8 scale."""
    H, W = img.shape[:2]
    r = int(round(30 * s))
    x1, y1, x2, y2 = rect
    gx1, gy1, gx2, gy2 = max(0, x1 - 2 * r), max(0, y1 - 2 * r), min(W, x2 + 2 * r), min(H, y2 + 3 * r)
    if gx2 <= gx1 or gy2 <= gy1:
        return
    k = 8
    mh, mw = max(1, (gy2 - gy1) // k), max(1, (gx2 - gx1) // k)
    m = np.zeros((mh, mw), np.float32)
    m[max(0, (y1 + r // 3 - gy1) // k) : max(0, (y2 + r // 2 - gy1) // k), max(0, (x1 - gx1) // k) : max(0, (x2 - gx1) // k)] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), max(0.5, r / k / 2))
    m = cv2.resize(m, (gx2 - gx1, gy2 - gy1), interpolation=cv2.INTER_LINEAR)
    roi = img[gy1:gy2, gx1:gx2].astype(np.float32)
    img[gy1:gy2, gx1:gx2] = (roi * (1 - strength * m[..., None])).astype(np.uint8)


def draw_window_chrome(img: np.ndarray, rect: Rect, s: float, *, shadow: bool = True) -> Rect:
    """Shadow, title bar / toolbar and border. Returns the content rect."""
    H, W = img.shape[:2]
    if shadow:
        draw_shadow(img, rect, s)
    x1, y1, x2, y2 = rect
    th = int(round(TITLE_PT * s))
    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y1 + th)
    if cx2 > cx1 and cy2 > cy1:
        img[cy1:cy2, cx1:cx2] = (236, 236, 236)
        img[max(0, y1 + th - max(1, int(s))) : cy2, cx1:cx2] = (210, 210, 210)
    for i, c in enumerate(((95, 95, 255), (60, 190, 255), (80, 200, 40))):
        cv2.circle(img, (x1 + int((18 + 20 * i) * s), y1 + int(14 * s)), max(2, int(6 * s)), c, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1 + int(160 * s), y1 + int(26 * s)), (x2 - int(160 * s), y1 + int(44 * s)), (250, 250, 250), -1)
    cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (200, 200, 200), 1)
    return (x1, y1 + th, x2, y2)


def draw_menubar(img: np.ndarray, clock: str) -> Rect:
    W = img.shape[1]
    s = desktop_scale(W)
    menu = desktop_rects(W, img.shape[0])["menu"]
    _frost(img, menu, 0.25)
    x = int(60 * s)
    sc = 0.5 * s
    th = max(1, int(round(s)))
    for word in ("Safari", "File", "Edit", "View", "History", "Bookmarks", "Window", "Help"):
        cv2.putText(img, word, (x, int(25 * s)), _FONT, sc, (20, 20, 20), th, cv2.LINE_AA)
        x += int((20 + 11 * len(word)) * s)
    for i in range(6):
        cv2.rectangle(img, (W - int((450 - 55 * i) * s), int(10 * s)), (W - int((430 - 55 * i) * s), int(28 * s)), (30, 30, 30), th)
    cv2.putText(img, clock, (W - int(165 * s), int(25 * s)), _FONT, sc, (20, 20, 20), th, cv2.LINE_AA)
    return menu


def draw_dock(img: np.ndarray, *, magnify: bool = False, seed: int = 11) -> Rect:
    H, W = img.shape[:2]
    s = desktop_scale(W)
    dock = desktop_rects(W, H, magnify=magnify)["dock"]
    x1, y1, x2, y2 = dock
    base_top = desktop_rects(W, H)["dock"][1]
    _frost(img, (x1, base_top, x2, y2), 0.4)
    n = 12
    rng = np.random.default_rng(seed)
    styles = ("vivid", "warm", "cool", "muted")
    size = int(round(48 * s))
    gap = ((x2 - x1) - n * size) // (n + 1)
    mid = n // 2
    for i in range(n):
        sz = size
        if magnify:
            sz = int(round(size * (1.0 + 0.9 * max(0.0, 1.0 - abs(i - mid) / 3.0))))
        ix = x1 + gap + i * (size + gap) - (sz - size) // 2
        iy = y2 - int(round(6 * s)) - sz
        icon = photo_texture(sz, sz, np.random.default_rng(int(rng.integers(1 << 30))), styles[i % 4])
        m = np.zeros((sz, sz), np.uint8)
        rr = max(2, sz // 5)
        cv2.rectangle(m, (rr, 0), (sz - rr, sz), 255, -1)
        cv2.rectangle(m, (0, rr), (sz, sz - rr), 255, -1)
        for cx, cy in ((rr, rr), (sz - rr, rr), (rr, sz - rr), (sz - rr, sz - rr)):
            cv2.circle(m, (cx, cy), rr, 255, -1)
        ax1, ay1, ax2, ay2 = max(0, ix), max(0, iy), min(W, ix + sz), min(H, iy + sz)
        if ax2 <= ax1 or ay2 <= ay1:
            continue
        roi = img[ay1:ay2, ax1:ax2]
        mm = m[ay1 - iy : ay2 - iy, ax1 - ix : ax2 - ix] > 0
        roi[mm] = icon[ay1 - iy : ay2 - iy, ax1 - ix : ax2 - ix][mm]
    return dock


@lru_cache(maxsize=8)
def _text_app(kind: str, w: int, h: int, seed: int, s: float) -> np.ndarray:
    """Content of a text-only app window: ``notes`` (sidebar + paragraphs) or
    ``finder`` (a file list with small icons — UI, not pictures)."""
    rng = np.random.default_rng(stable_seed(kind, seed, w, h))
    img = np.full((h, w, 3), (250, 250, 250), np.uint8)
    side = int(150 * s)
    img[:, :side] = (240, 238, 236)
    sc = 0.45 * s
    th = max(1, int(round(s)))
    for i in range(max(1, int(h / (34 * s)))):
        y = int((30 + 34 * i) * s)
        cv2.putText(img, f"{kind.title()} {i}", (int(16 * s), y), _FONT, sc, (40, 40, 40), th, cv2.LINE_AA)
    if kind == "finder":
        for i in range(max(1, int(h / (28 * s)))):
            y = int((14 + 28 * i) * s)
            cv2.rectangle(img, (side + int(14 * s), y), (side + int(30 * s), y + int(16 * s)), (200, 150, 60), -1)
            cv2.putText(img, " ".join(rng.choice(_WORDS, 3)), (side + int(40 * s), y + int(13 * s)), _FONT, sc, (30, 30, 30), th, cv2.LINE_AA)
    else:
        y = int(40 * s)
        while y < h - int(30 * s):
            x = side + int(20 * s)
            line = " ".join(rng.choice(_WORDS, int(rng.integers(6, 12))))
            cv2.putText(img, line, (x, y), _FONT, sc, (30, 30, 30), th, cv2.LINE_AA)
            y += int(24 * s) if rng.random() > 0.15 else int(48 * s)
    return img


def text_app(kind: str, w: int, h: int, seed: int, s: float) -> np.ndarray:
    return _text_app(kind, int(w), int(h), int(seed), float(s))


def page_with_flags(page: Page, flagged: set[str], page_id: str | None = None) -> Page:
    """Same pixels, ``flagged`` = exactly these picture ids (parts regenerated)."""
    rng = np.random.default_rng(stable_seed("reflag", page.page_id, sorted(flagged)))
    pics = []
    for p in page.pictures:
        f = p.pic_id in flagged
        pics.append(Picture(p.pic_id, p.xyxy, p.kind, f, p.style, p.tags, _parts_for(p.xyxy, rng) if f else ()))
    return Page(page_id or page.page_id, page.layout, page.seed, page.scale, page.image, pics)


def retexture(page: Page, seed: int, page_id: str | None = None) -> Page:
    """Same layout, new picture contents, nothing flagged (a clean page that
    looks like the flagged one around its pictures)."""
    img = page.image.copy()
    pics = []
    for i, p in enumerate(page.pictures):
        x1, y1, x2, y2 = p.xyxy
        rng = np.random.default_rng(stable_seed("retex", seed, page.page_id, i))
        img[y1:y2, x1:x2] = photo_texture(y2 - y1, x2 - x1, rng, p.style)
        pics.append(Picture(p.pic_id, p.xyxy, p.kind, False, p.style, p.tags, ()))
    return Page(page_id or f"{page.page_id}-clean{seed}", page.layout, page.seed, page.scale, img, pics)


def build_corpus(res: str = "half", seed: int = 0, variants: int = 1) -> list[Page]:
    width = RESOLUTIONS[res][0]
    return [make_page(layout, seed, width_px=width, variant=v) for layout in LAYOUTS for v in range(variants)]


def corpus_gt(pages: list[Page]) -> dict:
    return {"pages": [p.gt() for p in pages]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the synthetic benign page corpus + GT JSON")
    parser.add_argument("--out", required=True)
    parser.add_argument("--res", choices=sorted(RESOLUTIONS), default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--png", action="store_true", help="also write page PNGs (synthetic pixels only)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pages = build_corpus(args.res, args.seed, args.variants)
    (out / "corpus_gt.json").write_text(json.dumps(corpus_gt(pages), indent=2) + "\n")
    if args.png:
        for page in pages:
            img = page.image.copy()
            for pic in page.pictures:
                x1, y1, x2, y2 = pic.xyxy
                col = (0, 0, 255) if pic.flagged else (0, 200, 0)
                cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), col, max(2, int(page.scale * 2)))
            cv2.imwrite(str(out / f"{page.page_id}.png"), img)
    n = sum(len(p.pictures) for p in pages)
    f = sum(1 for p in pages for pic in p.pictures if pic.flagged)
    print(f"wrote {len(pages)} pages, {n} pictures ({f} flagged) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
