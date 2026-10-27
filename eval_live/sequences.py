"""Screen sequences over the synthetic corpus: scroll, navigate, sub-pane, video.

A sequence is a per-frame *plan* (which page each pane shows, at what scroll
offset, video time) plus per-frame ground truth in **screen** coordinates.
Frames are rendered lazily from the plan, so a Retina sequence never has to
sit in memory, and re-rendering (for the annotated mp4) gives identical pixels.

Screen = static browser chrome (menu bar + tab strip + address bar) on top and
one or two panes below it that show a window onto a taller page.

``desk_*`` sequences (layout ``desktop``) render a whole macOS-like desktop
instead: wallpaper, menu bar with a ticking clock, Dock (with a magnified
frame), app windows with shadows, overlapping, dragged, re-ordered, opened
and closed. Each of their frames carries the **window list** the window
server would report (``FrameGT.windows``, a ``ScreenLayout``), which the
harness passes to the engine exactly as the live capture does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import cv2
import numpy as np

from eval_live.corpus import (
    HALF,
    RESOLUTIONS,
    Page,
    desktop_rects,
    desktop_scale,
    desktop_wallpaper,
    draw_dock,
    draw_menubar,
    draw_window_chrome,
    make_page,
    page_with_flags,
    retexture,
    stable_seed,
    text_app,
    video_texture,
)
from eval_live.corpus import TITLE_PT
from vision_core.live.contracts import ScreenLayout, WindowInfo

Rect = tuple[int, int, int, int]

FPS = 12.0  # live capture rate on the Mac (edge/app.py live default)
NAV_WINDOW = 12  # frames after a navigation that count as "after navigation" (1 s)


@dataclass
class Pane:
    rect: Rect  # screen rect the pane occupies
    page: str  # page key in Sequence.pages
    scroll_y: int = 0


@dataclass
class DeskWindow:
    wid: int
    rect: Rect  # window bounds in screen px (title bar included)
    content: str  # a page key in Sequence.pages, or "notes" / "finder" (text apps)
    scroll_y: int = 0
    owner: str = "Safari"
    alpha: float = 1.0  # < 1: drawn half-faded (open / close animation)


@dataclass
class DesktopPlan:
    wallpaper: str
    windows: list[DeskWindow]  # front to back, as drawn
    clock: str = "Mon 26 Oct 10:41"
    magnify: bool = False  # the Dock is magnified (its window grows)
    listed: tuple[int, ...] | None = None  # ids the window list reports (None = every drawn window)
    stale: bool = False  # the window list may not match the pixels on this frame
    event: bool = False  # drag / z-order / open / close / navigation on this frame


@dataclass
class FramePlan:
    panes: list[Pane]
    video_t: int | None = None  # drives every "video" picture on visible pages
    video_scene: int = 0
    desktop: DesktopPlan | None = None


@dataclass
class GTPicture:
    gid: str  # "<page_id>/<pic_id>", stable across frames
    xyxy: Rect  # visible part, clipped to its pane (screen px)
    full_xyxy: Rect  # unclipped rect in screen px (may extend off the pane)
    clip: Rect  # pane rect that clips it
    flagged: bool
    kind: str
    parts: tuple[Rect, ...] = ()  # visible part boxes (clipped)
    full_parts: tuple[Rect, ...] = ()
    tags: tuple[str, ...] = ()
    hidden: tuple[Rect, ...] = ()  # desktop: front windows / menu bar / Dock covering part of ``xyxy``
    window_id: int | None = None  # desktop: owning window

    @property
    def full_area(self) -> int:
        x1, y1, x2, y2 = self.full_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def visible_area(self) -> int:
        x1, y1, x2, y2 = self.xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)

    def to_dict(self) -> dict:
        return {
            "gid": self.gid,
            "xyxy": list(self.xyxy),
            "full_xyxy": list(self.full_xyxy),
            "flagged": self.flagged,
            "kind": self.kind,
            "parts": [list(p) for p in self.parts],
            "tags": list(self.tags),
        }


@dataclass
class FrameGT:
    index: int
    t_ms: float
    layout: str
    pictures: list[GTPicture]
    scroll_dy: int = 0  # largest |dy| of any pane vs the previous frame
    nav: bool = False  # a pane switched page on this frame
    frames_since_nav: int | None = None
    video_changed: bool = False
    windows: ScreenLayout | None = None  # desktop: the window list handed to the engine
    desktop: dict | None = None  # desktop: menu / dock / visible rects and the drawn layout (truth)
    frames_since_event: int | None = None  # desktop: frames since a drag / z-order / open / close / nav

    @property
    def moving(self) -> bool:
        return self.scroll_dy != 0

    @property
    def after_nav(self) -> bool:
        return self.frames_since_nav is not None and self.frames_since_nav < NAV_WINDOW

    def flagged(self) -> list[GTPicture]:
        return [p for p in self.pictures if p.flagged]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "t_ms": self.t_ms,
            "layout": self.layout,
            "scroll_dy": self.scroll_dy,
            "nav": self.nav,
            "frames_since_nav": self.frames_since_nav,
            "pictures": [p.to_dict() for p in self.pictures],
        }


@dataclass
class Sequence:
    name: str
    kind: str  # scroll_smooth | scroll_jumpy | scroll_momentum | navigation | subpane | video
    width: int
    height: int
    pages: dict[str, Page]
    plans: list[FramePlan]
    layout: str
    video_flag_from: int | None = None  # frame index from which video pictures are flagged
    gt: list[FrameGT] = field(default_factory=list)
    pages_seed: int = 0  # text-app content seed (desktop)

    def __post_init__(self) -> None:
        if not self.gt:
            self.gt = _build_gt(self)

    def __len__(self) -> int:
        return len(self.plans)

    def frames(self) -> Iterator[np.ndarray]:
        chrome = None if self.layout == "desktop" else browser_chrome(self.width, self.height)
        for plan in self.plans:
            yield render_frame(self, plan, chrome)

    def frame(self, i: int) -> np.ndarray:
        chrome = None if self.layout == "desktop" else browser_chrome(self.width, self.height)
        return render_frame(self, self.plans[i], chrome)

    def windows(self) -> list[ScreenLayout | None]:
        """Per-frame window list (None for the web-page sequences)."""
        return [g.windows for g in self.gt]


# --------------------------------------------------------------------------- rendering


def chrome_height(width: int) -> int:
    return int(round(108 * width / 1728.0))


def browser_chrome(width: int, height: int) -> np.ndarray:
    """Static macOS-ish menu bar + tab strip + address bar (UI chrome only)."""
    s = width / 1728.0
    ch = chrome_height(width)
    img = np.full((height, width, 3), 250, np.uint8)
    img[: int(26 * s)] = (236, 236, 236)
    img[int(26 * s) : int(62 * s)] = (222, 224, 228)
    img[int(62 * s) : ch] = (255, 255, 255)
    img[ch - max(1, int(s)) : ch] = (210, 210, 210)
    font = cv2.FONT_HERSHEY_SIMPLEX
    sc = 0.45 * s
    for i, word in enumerate(("File", "Edit", "View", "History", "Window", "Help")):
        cv2.putText(img, word, (int((40 + 70 * i) * s), int(18 * s)), font, sc, (30, 30, 30), max(1, int(s)), cv2.LINE_AA)
    for i in range(4):
        x = int((90 + 230 * i) * s)
        col = (255, 255, 255) if i == 0 else (232, 233, 236)
        cv2.rectangle(img, (x, int(32 * s)), (x + int(220 * s), int(62 * s)), col, -1)
        cv2.putText(img, f"Tab {i + 1}", (x + int(14 * s), int(52 * s)), font, sc, (60, 60, 60), max(1, int(s)), cv2.LINE_AA)
    for i, col in enumerate(((95, 95, 255), (70, 190, 255), (80, 200, 40))):
        cv2.circle(img, (int((22 + 20 * i) * s), int(46 * s)), int(6 * s), col, -1, cv2.LINE_AA)
    cv2.rectangle(img, (int(160 * s), int(68 * s)), (width - int(160 * s), int(100 * s)), (241, 243, 244), -1)
    cv2.putText(img, "https://example.test/", (int(180 * s), int(90 * s)), font, sc, (90, 90, 90), max(1, int(s)), cv2.LINE_AA)
    return img


def _blit_page(dst: np.ndarray, pane: Pane, page: Page) -> None:
    x1, y1, x2, y2 = pane.rect
    h = y2 - y1
    w = min(x2 - x1, page.width)
    src = page.image[pane.scroll_y : pane.scroll_y + h, :w]
    dst[y1 : y1 + src.shape[0], x1 : x1 + w] = src


def render_frame(seq: Sequence, plan: FramePlan, chrome: np.ndarray | None) -> np.ndarray:
    if plan.desktop is not None:
        return render_desktop(seq, plan.desktop)
    frame = chrome.copy()
    for pane in plan.panes:
        page = seq.pages[pane.page]
        _blit_page(frame, pane, page)
        if plan.video_t is None:
            continue
        for pic in page.pictures:
            if pic.kind != "video":
                continue
            full = _to_screen(pic.xyxy, pane)
            vis = _clip(full, pane.rect)
            if vis is None:
                continue
            fw, fh = full[2] - full[0], full[3] - full[1]
            tex = video_texture(fh, fw, page.seed, plan.video_t, scene=plan.video_scene)
            ox, oy = vis[0] - full[0], vis[1] - full[1]
            frame[vis[1] : vis[3], vis[0] : vis[2]] = tex[oy : oy + vis[3] - vis[1], ox : ox + vis[2] - vis[0]]
    return frame


# --------------------------------------------------------------------------- ground truth


def _to_screen(r: Rect, pane: Pane) -> Rect:
    ox, oy = pane.rect[0], pane.rect[1] - pane.scroll_y
    return (r[0] + ox, r[1] + oy, r[2] + ox, r[3] + oy)


def _clip(r: Rect, c: Rect) -> Rect | None:
    x1, y1 = max(r[0], c[0]), max(r[1], c[1])
    x2, y2 = min(r[2], c[2]), min(r[3], c[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def frame_gt(seq: Sequence, plan: FramePlan, index: int) -> list[GTPicture]:
    out: list[GTPicture] = []
    for pane in plan.panes:
        page = seq.pages[pane.page]
        clip_rect = (pane.rect[0], pane.rect[1], min(pane.rect[2], pane.rect[0] + page.width), pane.rect[3])
        for pic in page.pictures:
            full = _to_screen(pic.xyxy, pane)
            vis = _clip(full, clip_rect)
            if vis is None:
                continue
            flagged = pic.flagged
            if pic.kind == "video" and seq.video_flag_from is not None:
                flagged = index >= seq.video_flag_from
            parts_full = tuple(_to_screen(p, pane) for p in pic.parts) if flagged else ()
            if pic.kind == "video" and flagged and not parts_full:
                parts_full = _video_parts(full)
            parts_vis = tuple(c for c in (_clip(p, clip_rect) for p in parts_full) if c is not None)
            out.append(
                GTPicture(
                    gid=f"{page.page_id}/{pic.pic_id}",
                    xyxy=vis,
                    full_xyxy=full,
                    clip=clip_rect,
                    flagged=flagged,
                    kind=pic.kind,
                    parts=parts_vis,
                    full_parts=parts_full,
                    tags=pic.tags,
                )
            )
    return out


def _video_parts(full: Rect) -> tuple[Rect, ...]:
    x1, y1, x2, y2 = full
    w, h = x2 - x1, y2 - y1
    return ((x1 + int(0.38 * w), y1 + int(0.3 * h), x1 + int(0.62 * w), y1 + int(0.55 * h)),)


def _build_gt(seq: Sequence) -> list[FrameGT]:
    if seq.layout == "desktop":
        return _build_desktop_gt(seq)
    out: list[FrameGT] = []
    prev: FramePlan | None = None
    since: int | None = None
    for i, plan in enumerate(seq.plans):
        dy = 0
        nav = False
        if prev is not None:
            for a, b in zip(prev.panes, plan.panes):
                if a.page != b.page:
                    nav = True
                else:
                    dy = max(dy, abs(b.scroll_y - a.scroll_y), key=abs)
            if len(prev.panes) != len(plan.panes):
                nav = True
        if nav:
            since = 0
        elif since is not None:
            since += 1
        main = seq.pages[plan.panes[-1].page]
        layout = "subpane" if seq.kind == "subpane" else main.layout
        out.append(
            FrameGT(
                index=i,
                t_ms=i * 1000.0 / FPS,
                layout=layout,
                pictures=frame_gt(seq, plan, i),
                scroll_dy=int(dy),
                nav=nav,
                frames_since_nav=since,
                video_changed=plan.video_t is not None and (prev is None or prev.video_t != plan.video_t),
            )
        )
        prev = plan
    return out


# --------------------------------------------------------------------------- scroll profiles


def scroll_profile(kind: str, n: int, rng: np.random.Generator, max_scroll: int, scale: float = 1.0) -> list[int]:
    """Per-frame scroll offsets (device px). Speeds are CSS px/frame × scale.

    smooth: 10–40 px/frame with short pauses; jumpy: 80–200 px steps between
    holds; momentum: flick → exponential decay → rest, sometimes reversed.
    """
    ys = [0]
    y = 0.0
    if kind == "smooth":
        v = float(rng.uniform(10, 40))
        for i in range(1, n):
            if i < 3 or rng.random() < 0.12:
                step = 0.0
            else:
                v = float(np.clip(v + rng.normal(0, 5), 10, 40))
                step = v
            y += step * scale
            ys.append(int(round(y)))
    elif kind == "jumpy":
        for i in range(1, n):
            step = float(rng.uniform(80, 200)) if (i >= 2 and rng.random() < 0.5) else 0.0
            if step and rng.random() < 0.15 and y > 400 * scale:
                step = -step
            y += step * scale
            ys.append(int(round(y)))
    elif kind == "momentum":
        v = 0.0
        direction = 1.0
        rest = 2
        for _ in range(1, n):
            if abs(v) < 2.0:
                v = 0.0
                if rest > 0:
                    rest -= 1
                else:
                    direction = -1.0 if (y > 800 * scale and rng.random() < 0.25) else 1.0
                    v = direction * float(rng.uniform(120, 220))
                    rest = int(rng.integers(2, 5))
            y += v * scale
            v *= float(rng.uniform(0.78, 0.88))
            ys.append(int(round(y)))
    elif kind == "static":
        ys = [0] * n
    else:
        raise ValueError(kind)
    return [int(np.clip(v, 0, max_scroll)) for v in ys[:n]]


# --------------------------------------------------------------------------- builders


def _viewport(width: int, height: int) -> Rect:
    return (0, chrome_height(width), width, height)


def _single(name, kind, layout, profile, *, width, height, seed, n, variant=0) -> Sequence:
    page = make_page(layout, seed, width_px=width, variant=variant)
    vp = _viewport(width, height)
    rng = np.random.default_rng(stable_seed("seq", name, seed))
    ys = scroll_profile(profile, n, rng, page.height - (vp[3] - vp[1]), scale=width / 1728.0)
    plans = [FramePlan([Pane(vp, page.page_id, y)]) for y in ys]
    return Sequence(name, kind, width, height, {page.page_id: page}, plans, layout)


def _navigation(name, layout_a, layout_b, *, width, height, seed, n) -> Sequence:
    a = make_page(layout_a, seed, width_px=width)
    b = make_page(layout_b, seed, width_px=width, variant=1)
    vp = _viewport(width, height)
    vh = vp[3] - vp[1]
    rng = np.random.default_rng(stable_seed("seq", name, seed))
    cut = n // 2
    ys_a = scroll_profile("smooth", cut, rng, a.height - vh, scale=width / 1728.0)
    ys_b = [0, 0, 0] + scroll_profile("smooth", max(0, n - cut - 3), rng, b.height - vh, scale=width / 1728.0)
    plans = [FramePlan([Pane(vp, a.page_id, y)]) for y in ys_a]
    plans += [FramePlan([Pane(vp, b.page_id, y)]) for y in ys_b[: n - cut]]
    return Sequence(name, "navigation", width, height, {a.page_id: a, b.page_id: b}, plans, f"{layout_a}->{layout_b}")


def _subpane(name, *, width, height, seed, n) -> Sequence:
    """Static article on the left; an image-grid pane on the right scrolls."""
    vp = _viewport(width, height)
    split = int(width * 0.58)
    left = make_page("news", seed, width_px=split, scale=width / 1728.0, variant=2)
    right = make_page("image_search", seed, width_px=width - split - int(8 * width / 1728.0),
                      scale=width / 1728.0, variant=2)
    gap = width - split - right.width
    lp = Pane((0, vp[1], split, vp[3]), left.page_id, int(left.height * 0.18))
    rng = np.random.default_rng(stable_seed("seq", name, seed))
    ys = scroll_profile("smooth", n, rng, right.height - (vp[3] - vp[1]), scale=width / 1728.0)
    plans = [FramePlan([lp, Pane((split + gap, vp[1], width, vp[3]), right.page_id, y)]) for y in ys]
    return Sequence(name, "subpane", width, height, {left.page_id: left, right.page_id: right}, plans, "subpane")


def _video(name, *, width, height, seed, n) -> Sequence:
    """Playing video: content changes every frame; a scene cut at 2/3 makes it flagged."""
    page = make_page("video", seed, width_px=width)
    vp = _viewport(width, height)
    cut = (2 * n) // 3
    scroll_from = n - max(3, n // 8)
    s = width / 1728.0
    plans = []
    for i in range(n):
        y = 0 if i < scroll_from else int(round((i - scroll_from + 1) * 24 * s))
        plans.append(FramePlan([Pane(vp, page.page_id, y)], video_t=i, video_scene=0 if i < cut else 1))
    return Sequence(name, "video", width, height, {page.page_id: page}, plans, "video", video_flag_from=cut)


# --------------------------------------------------------------------------- desktop


def _content_rect(w: DeskWindow, s: float) -> Rect:
    x1, y1, x2, y2 = w.rect
    return (x1, y1 + int(round(TITLE_PT * s)), x2, y2)


def render_desktop(seq: Sequence, d: DesktopPlan) -> np.ndarray:
    W, H = seq.width, seq.height
    s = desktop_scale(W)
    img = desktop_wallpaper(d.wallpaper, W, H)
    for w in reversed(d.windows):
        before = img.copy() if w.alpha < 1.0 else None
        c = draw_window_chrome(img, w.rect, s)
        vis = _clip(c, (0, 0, W, H))
        if vis is not None:
            cw, ch = c[2] - c[0], c[3] - c[1]
            if w.content in seq.pages:
                page = seq.pages[w.content]
                src = np.zeros((ch, cw, 3), np.uint8)
                part = page.image[w.scroll_y : w.scroll_y + ch, :cw]
                src[: part.shape[0], : part.shape[1]] = part
            else:
                src = text_app(w.content, cw, ch, seq.pages_seed, s)
            img[vis[1] : vis[3], vis[0] : vis[2]] = src[vis[1] - c[1] : vis[3] - c[1], vis[0] - c[0] : vis[2] - c[0]]
        if before is not None:
            r = _clip(w.rect, (0, 0, W, H))
            if r is not None:
                a = float(w.alpha)
                img[r[1] : r[3], r[0] : r[2]] = (a * img[r[1] : r[3], r[0] : r[2]] + (1 - a) * before[r[1] : r[3], r[0] : r[2]]).astype(np.uint8)
    draw_menubar(img, d.clock)
    draw_dock(img, magnify=d.magnify)
    return img


def desktop_layout(seq: Sequence, d: DesktopPlan, *, truth: bool = False, t_ms: float = 0.0) -> ScreenLayout:
    """The window list for one desktop frame, front to back: menu bar (layer
    24), Dock (20), app windows (0), desktop picture (very negative layer).
    ``truth``: every drawn window (the pixels); else what the window server
    reports on that frame (``listed`` may lag an animation)."""
    W, H = seq.width, seq.height
    rects = desktop_rects(W, H, magnify=d.magnify)
    wins = [
        WindowInfo(9001, tuple(float(v) for v in rects["menu"]), layer=24, owner_pid=1, owner="Window Server"),
        WindowInfo(9002, tuple(float(v) for v in rects["dock"]), layer=20, owner_pid=2, owner="Dock"),
    ]
    listed = None if truth else d.listed
    for w in d.windows:
        if listed is not None and w.wid not in listed:
            continue
        wins.append(WindowInfo(w.wid, tuple(float(v) for v in w.rect), layer=0, owner_pid=1000 + w.wid, owner=w.owner, alpha=1.0 if truth else w.alpha))
    wins.append(WindowInfo(9003, (0.0, 0.0, float(W), float(H)), layer=-2147483624, owner_pid=2, owner="Dock"))
    return ScreenLayout(frame_wh=(W, H), windows=tuple(wins), visible_xyxy=tuple(float(v) for v in rects["visible"]),
                        t_ms=t_ms, stale=False if truth else bool(d.stale))


def desktop_gt(seq: Sequence, d: DesktopPlan, index: int) -> list[GTPicture]:
    W, H = seq.width, seq.height
    s = desktop_scale(W)
    rects = desktop_rects(W, H, magnify=d.magnify)
    screen = (0, 0, W, H)
    fronts: list[Rect] = [rects["menu"], rects["dock"]]
    out: list[GTPicture] = []
    for w in d.windows:
        if w.content in seq.pages:
            page = seq.pages[w.content]
            c = _content_rect(w, s)
            clip_rect = _clip(c, screen)
            if clip_rect is not None:
                pane = Pane(c, page.page_id, w.scroll_y)
                for pic in page.pictures:
                    full = _to_screen(pic.xyxy, pane)
                    vis = _clip(full, clip_rect)
                    if vis is None:
                        continue
                    parts_full = tuple(_to_screen(p, pane) for p in pic.parts) if pic.flagged else ()
                    parts_vis = tuple(cc for cc in (_clip(p, clip_rect) for p in parts_full) if cc is not None)
                    hidden = tuple(h for h in (_clip(f, vis) for f in fronts) if h is not None)
                    out.append(GTPicture(
                        gid=f"{page.page_id}/{pic.pic_id}", xyxy=vis, full_xyxy=full, clip=clip_rect,
                        flagged=pic.flagged, kind=pic.kind, parts=parts_vis, full_parts=parts_full,
                        tags=pic.tags, hidden=hidden, window_id=w.wid,
                    ))
        r = _clip(w.rect, screen)
        if r is not None:
            fronts.append(r)
    return out


def _build_desktop_gt(seq: Sequence) -> list[FrameGT]:
    out: list[FrameGT] = []
    prev: DesktopPlan | None = None
    since_nav: int | None = None
    since_event: int | None = None
    for i, plan in enumerate(seq.plans):
        d = plan.desktop
        t = i * 1000.0 / FPS
        dy = 0
        nav = False
        if prev is not None:
            old = {w.wid: w for w in prev.windows}
            for w in d.windows:
                o = old.get(w.wid)
                if o is None:
                    continue
                if o.content != w.content:
                    nav = True
                else:
                    dy = max(dy, abs(w.scroll_y - o.scroll_y), abs(w.rect[0] - o.rect[0]), abs(w.rect[1] - o.rect[1]))
        if nav:
            since_nav = 0
        elif since_nav is not None:
            since_nav += 1
        if d.event or nav:
            since_event = 0
        elif since_event is not None:
            since_event += 1
        rects = desktop_rects(seq.width, seq.height, magnify=d.magnify)
        out.append(FrameGT(
            index=i, t_ms=t, layout="desktop", pictures=desktop_gt(seq, d, i), scroll_dy=int(dy), nav=nav,
            frames_since_nav=since_nav,
            windows=desktop_layout(seq, d, t_ms=t),
            desktop={**rects, "truth": desktop_layout(seq, d, truth=True, t_ms=t)},
            frames_since_event=since_event,
        ))
        prev = d
    return out


def _pt(v: tuple[float, float, float, float], s: float) -> Rect:
    return tuple(int(round(x * s)) for x in v)  # type: ignore[return-value]


BROWSER_PT = (80, 60, 1300, 1000)
NOTES_PT = (1040, 330, 1640, 840)
FINDER_PT = (420, 640, 1000, 1010)
SECOND_PT = (700, 120, 1600, 760)
CLOCKS = ("Mon 26 Oct 10:41", "Mon 26 Oct 10:42", "Mon 26 Oct 10:43")


def _browser_page(width: int, s: float, seed: int, layout: str = "news", variant: int = 0) -> Page:
    cw = int(round((BROWSER_PT[2] - BROWSER_PT[0]) * s))
    return make_page(layout, seed, width_px=cw, variant=variant)


def _pick_flagged(page: Page, content: Rect, scroll: int, screen_area: float, *, avoid: Rect | None = None,
                  prefer_rows: tuple[int, int] | None = None) -> str | None:
    """A mid-size picture fully visible in the window (the one to flag)."""
    pane = Pane(content, page.page_id, scroll)
    best, best_key = None, None
    for pic in page.pictures:
        full = _to_screen(pic.xyxy, pane)
        vis = _clip(full, content)
        if vis != full:
            continue
        a = (full[2] - full[0]) * (full[3] - full[1])
        if a > 0.08 * screen_area or a < 0.004 * screen_area:
            continue
        if avoid is not None and _clip(full, avoid) is not None:
            continue
        rows_ok = prefer_rows is None or (full[1] < prefer_rows[1] and full[3] > prefer_rows[0])
        key = (rows_ok, a)
        if best_key is None or key > best_key:
            best, best_key = pic.pic_id, key
    return best


def _desk_base(width: int, height: int, seed: int, *, flag: bool = True, layout: str = "news",
               prefer_rows: tuple[int, int] | None = None, avoid: Rect | None = None, maximised: bool = False):
    s = desktop_scale(width)
    if maximised:
        vis = desktop_rects(width, height)["visible"]
        brect = vis
        cw = vis[2] - vis[0]
        page = make_page(layout, seed, width_px=cw)
    else:
        brect = _pt(BROWSER_PT, s)
        page = _browser_page(width, s, seed, layout)
    content = (brect[0], brect[1] + int(round(TITLE_PT * s)), brect[2], min(brect[3], height))
    # Start scrolled to where a mid-size picture is fully in view (flag that
    # one), preferring a scroll where no picture is cut by the window edges.
    scroll, chosen, best = 0, None, None
    for sc in range(0, int(900 * s), max(1, int(10 * s))):
        got = _pick_flagged(page, content, sc, float(width * height), avoid=avoid, prefer_rows=prefer_rows)
        if got is None:
            continue
        pane = Pane(content, page.page_id, sc)
        cut = 0
        for pic in page.pictures:
            full = _to_screen(pic.xyxy, pane)
            vis = _clip(full, content)
            if vis is not None and vis != full:
                cut += 1
        if best is None or cut < best:
            scroll, chosen, best = sc, got, cut
        if cut == 0:
            break
    if not flag:
        chosen = None
    page = page_with_flags(page, {chosen} if chosen else set(), page_id=f"{page.page_id}-desk{int(flag)}")
    return s, brect, page, scroll


def _desktop(name: str, *, width: int, height: int, seed: int, n: int, wallpaper: str = "photo", flag: bool = True,
             variant: str = "static") -> Sequence:
    s = desktop_scale(width)
    notes = _pt(NOTES_PT, s)
    plans: list[FramePlan] = []
    pages: dict[str, Page] = {}

    def clock(i: int) -> str:
        return CLOCKS[min(len(CLOCKS) - 1, (3 * i) // max(1, n))]

    if variant == "static":
        _s, brect, page, y0 = _desk_base(width, height, seed, flag=flag, avoid=notes)
        pages[page.page_id] = page
        for i in range(n):
            wins = [DeskWindow(102, notes, "notes", owner="Notes"), DeskWindow(101, brect, page.page_id, scroll_y=y0)]
            mag = n // 2 <= i < n // 2 + 2
            # The Dock magnifying over the window changes what is visible: an event.
            plans.append(FramePlan([], desktop=DesktopPlan(wallpaper, wins, clock(i), magnify=mag, event=i in (n // 2, n // 2 + 2))))
    elif variant == "drag":
        # Notes is dragged left across the flagged picture, then back.
        _s, brect, page, y0 = _desk_base(width, height, seed, flag=True, prefer_rows=(notes[1], notes[3]))
        pages[page.page_id] = page
        step = int(round(60 * s))
        half = n // 2
        for i in range(n):
            k = i if i < half else max(0, 2 * half - i)
            dx = -step * k
            r = (notes[0] + dx, notes[1], notes[2] + dx, notes[3])
            wins = [DeskWindow(102, r, "notes", owner="Notes"), DeskWindow(101, brect, page.page_id, scroll_y=y0)]
            plans.append(FramePlan([], desktop=DesktopPlan(wallpaper, wins, clock(i), event=i > 0 and k != (i - 1 if i - 1 < half else max(0, 2 * half - i + 1)))))
    elif variant == "cmdtab":
        # Notes in front of the flagged picture, then Cmd-Tab: browser in front, then Finder on top.
        _s, brect, page, y0 = _desk_base(width, height, seed, flag=True, prefer_rows=(notes[1], notes[3]))
        pages[page.page_id] = page
        finder = _pt(FINDER_PT, s)
        third = n // 3
        for i in range(n):
            nwin = DeskWindow(102, (notes[0] - int(300 * s), notes[1], notes[2] - int(300 * s), notes[3]), "notes", owner="Notes")
            bwin = DeskWindow(101, brect, page.page_id, scroll_y=y0)
            if i < third:
                wins = [nwin, bwin]
            elif i < 2 * third:
                wins = [bwin, nwin]
            else:
                wins = [DeskWindow(103, finder, "finder", owner="Finder"), bwin, nwin]
            plans.append(FramePlan([], desktop=DesktopPlan(wallpaper, wins, clock(i), event=i in (third, 2 * third))))
    elif variant == "scroll":
        # Browser scrolls with the wallpaper visible around it; then the clock ticks.
        _s, brect, page, y0 = _desk_base(width, height, seed, flag=True, avoid=notes)
        pages[page.page_id] = page
        rng = np.random.default_rng(stable_seed("deskscroll", seed))
        vh = brect[3] - brect[1] - int(round(TITLE_PT * s))
        cut = (2 * n) // 3
        ys = [y0 + v for v in scroll_profile("smooth", cut, rng, page.height - vh - y0, scale=s)] + [None] * (n - cut)
        last = 0
        for i in range(n):
            y = last if ys[i] is None else ys[i]
            last = y
            tick = CLOCKS[0] if i < cut + 2 else (CLOCKS[1] if i < cut + 5 else CLOCKS[2])
            wins = [DeskWindow(102, notes, "notes", owner="Notes"), DeskWindow(101, brect, page.page_id, scroll_y=int(y))]
            plans.append(FramePlan([], desktop=DesktopPlan(wallpaper, wins, tick)))
    elif variant == "nav_clean":
        # Flagged page, then a clean page with the same layout (new picture contents).
        _s, brect, page, y0 = _desk_base(width, height, seed, flag=True, avoid=notes)
        clean = retexture(page, seed + 1)
        pages[page.page_id] = page
        pages[clean.page_id] = clean
        for i in range(n):
            pid = page.page_id if i < n // 2 else clean.page_id
            wins = [DeskWindow(102, notes, "notes", owner="Notes"), DeskWindow(101, brect, pid, scroll_y=y0)]
            plans.append(FramePlan([], desktop=DesktopPlan(wallpaper, wins, clock(i))))
    elif variant == "open_close":
        # A second browser window with one flagged picture opens (fading in,
        # window list ahead of the pixels), stays, then closes (list drops it
        # one frame before the pixels do).
        _s, brect, page, y0 = _desk_base(width, height, seed, flag=False)
        y0 = 0
        second = _pt(SECOND_PT, s)
        pg2 = make_page("image_search", seed, width_px=second[2] - second[0], variant=3)
        c2 = (second[0], second[1] + int(round(TITLE_PT * s)), second[2], second[3])
        chosen = _pick_flagged(pg2, c2, 0, float(width * height))
        pg2 = page_with_flags(pg2, {chosen} if chosen else set(), page_id=f"{pg2.page_id}-desk1")
        pages[page.page_id] = page
        pages[pg2.page_id] = pg2
        a, b = n // 4, (3 * n) // 4
        for i in range(n):
            bwin = DeskWindow(101, brect, page.page_id, scroll_y=y0)
            if i < a or i > b:
                d = DesktopPlan(wallpaper, [bwin], clock(i), event=i == b + 1)
            elif i == a:
                d = DesktopPlan(wallpaper, [DeskWindow(104, second, pg2.page_id, alpha=0.5), bwin], clock(i), stale=True, event=True)
            elif i == b:
                d = DesktopPlan(wallpaper, [DeskWindow(104, second, pg2.page_id, alpha=0.5), bwin], clock(i),
                                listed=(101,), stale=True, event=True)
            else:
                d = DesktopPlan(wallpaper, [DeskWindow(104, second, pg2.page_id), bwin], clock(i))
            plans.append(FramePlan([], desktop=d))
    elif variant == "maximised":
        _s, brect, page, y0 = _desk_base(width, height, seed, flag=True, maximised=True)
        pages[page.page_id] = page
        rng = np.random.default_rng(stable_seed("deskmax", seed))
        vh = brect[3] - brect[1] - int(round(TITLE_PT * s))
        ys = [y0 + v for v in scroll_profile("smooth", n, rng, page.height - vh - y0, scale=s)]
        for i in range(n):
            wins = [DeskWindow(101, brect, page.page_id, scroll_y=int(ys[i]))]
            plans.append(FramePlan([], desktop=DesktopPlan(wallpaper, wins, clock(i))))
    else:
        raise ValueError(variant)
    seq = Sequence(name, f"desktop_{variant}", width, height, pages, plans, "desktop")
    return seq


DESKTOP_SEQUENCE_NAMES = (
    "desk_static_photo",
    "desk_static_gradient",
    "desk_static_solid",
    "desk_clean",
    "desk_drag",
    "desk_cmdtab",
    "desk_scroll_clock",
    "desk_nav_clean",
    "desk_open_close",
    "desk_maximised",
)


SEQUENCE_NAMES = (
    "news_smooth",
    "search_jumpy",
    "shop_momentum",
    "social_momentum",
    "search_smooth",
    "nav_search_to_news",
    "nav_social_to_shop",
    "subpane_grid",
    "video_play",
)


def make_sequence(name: str, *, res: str = "half", seed: int = 0, n: int = 40,
                  size: tuple[int, int] | None = None) -> Sequence:
    width, height = size or RESOLUTIONS[res]
    kw = dict(width=width, height=height, seed=seed, n=n)
    if name == "news_smooth":
        return _single(name, "scroll_smooth", "news", "smooth", **kw)
    if name == "search_smooth":
        return _single(name, "scroll_smooth", "image_search", "smooth", **kw)
    if name == "search_jumpy":
        return _single(name, "scroll_jumpy", "image_search", "jumpy", **kw)
    if name == "shop_momentum":
        return _single(name, "scroll_momentum", "shopping", "momentum", **kw)
    if name == "social_momentum":
        return _single(name, "scroll_momentum", "social", "momentum", **kw)
    if name == "nav_search_to_news":
        return _navigation(name, "image_search", "news", **kw)
    if name == "nav_social_to_shop":
        return _navigation(name, "social", "shopping", **kw)
    if name == "subpane_grid":
        return _subpane(name, **kw)
    if name == "video_play":
        return _video(name, **kw)
    desk = {
        "desk_static_photo": dict(variant="static", wallpaper="photo"),
        "desk_static_gradient": dict(variant="static", wallpaper="gradient"),
        "desk_static_solid": dict(variant="static", wallpaper="solid"),
        "desk_clean": dict(variant="static", wallpaper="photo", flag=False),
        "desk_drag": dict(variant="drag", wallpaper="photo"),
        "desk_cmdtab": dict(variant="cmdtab", wallpaper="gradient"),
        "desk_scroll_clock": dict(variant="scroll", wallpaper="photo"),
        "desk_nav_clean": dict(variant="nav_clean", wallpaper="photo"),
        "desk_open_close": dict(variant="open_close", wallpaper="solid"),
        "desk_maximised": dict(variant="maximised", wallpaper="photo"),
    }.get(name)
    if desk is not None:
        return _desktop(name, **kw, **desk)
    raise ValueError(f"unknown sequence {name!r}; expected one of {SEQUENCE_NAMES + DESKTOP_SEQUENCE_NAMES}")


def make_suite(names=SEQUENCE_NAMES, *, res: str = "half", seed: int = 0, n: int = 40,
               size: tuple[int, int] | None = None) -> Iterator[Sequence]:
    """Lazily build sequences one at a time (pages are freed between them)."""
    for name in names:
        yield make_sequence(name, res=res, seed=seed, n=n, size=size)


__all__ = [
    "DESKTOP_SEQUENCE_NAMES",
    "DeskWindow",
    "DesktopPlan",
    "FPS",
    "HALF",
    "NAV_WINDOW",
    "FrameGT",
    "FramePlan",
    "GTPicture",
    "Pane",
    "SEQUENCE_NAMES",
    "Sequence",
    "make_sequence",
    "make_suite",
    "scroll_profile",
]
