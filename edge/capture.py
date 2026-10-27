"""Capture adapters. Replay of the safe fixture is the judged path.

``--source live`` grabs the primary laptop display (mss and/or Quartz).
That is permissioned desktop capture, not ScreenCaptureKit, and it is
**after** pixels are already on screen. Raw frames are never written.

macOS Screen Recording must be granted for Terminal / iTerm / Python /
Cursor (System Settings → Privacy & Security → Screen Recording).

Every Quartz grab also keeps the **window list** it already fetches
(``ScreenLayout``: id, owner pid / process name — never a window title —,
layer, bounds in capture px, alpha, front-to-back order) plus
``NSScreen.visibleFrame`` (the screen minus the menu bar and the Dock), so
the engine knows which window owns which pixels. ``WindowWatcher`` fetches
the same list on the main thread for the 60 Hz overlay clip (bounds and
layers need no Screen Recording permission).
"""

from __future__ import annotations

import os
import runpy
import time
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from edge.coords import DisplayGeom, cocoa_rect_to_capture, points_xyxy_to_capture, scale_from_sizes
from vision_core.contracts import Frame
from vision_core.live.contracts import ScreenLayout, WindowInfo

_VISION_ROOT = Path(__file__).resolve().parents[1]
SAFE_DIR = _VISION_ROOT / "fixtures" / "safe"
SAFE_FIXTURE = SAFE_DIR / "synthetic_moving_regions.mp4"
SAFE_GENERATOR = SAFE_DIR / "generate_synthetic.py"

EXCLUDE_TITLES = frozenset(
    {
        "Screen Guardian",
        "Screen Guardian Mask",
        "Screen Guardian Status",
        "Gomi",
    }
)

# Owner (process) names whose windows are never captured: the native
# SwiftUI island (``vision/island``, product GuardianIsland).
EXCLUDE_OWNERS = frozenset({"GuardianIsland", "Screen Guardian", "ScreenGuardian"})

PERMISSION_HELP = (
    "Screen Guardian needs macOS Screen Recording permission.\n"
    "\n"
    "  System Settings → Privacy & Security → Screen Recording\n"
    "  Enable the app that launched this process (Terminal, iTerm, Python, or Cursor).\n"
    "  Quit and reopen that app after granting access, then re-run.\n"
    "\n"
    "Capture starts after the desktop is already visible — this is not\n"
    "zero-exposure and first pixels can appear before any mask."
)


class CaptureAdapter(Protocol):
    """Eyes. ``read`` returns a Frame or None when capture is unavailable."""

    available: bool

    def read(self) -> Frame | None: ...

    def release(self) -> None: ...


def ensure_safe_fixture() -> Path:
    """Return the safe replay clip, generating it locally if needed."""
    if SAFE_FIXTURE.is_file():
        return SAFE_FIXTURE
    if not SAFE_GENERATOR.is_file():
        raise FileNotFoundError(f"missing generator {SAFE_GENERATOR}")
    runpy.run_path(str(SAFE_GENERATOR), run_name="__main__")
    if not SAFE_FIXTURE.is_file():
        raise FileNotFoundError(f"generator did not write {SAFE_FIXTURE}")
    return SAFE_FIXTURE


def screen_recording_granted() -> bool | None:
    """True / False when Quartz can answer; None if the API is missing."""
    try:
        from Quartz import CGPreflightScreenCaptureAccess

        return bool(CGPreflightScreenCaptureAccess())
    except Exception:
        return None


def preferred_live_backend() -> str:
    """Backend the live grabber will try first. Not ``none`` once Quartz/mss import."""
    try:
        from Quartz import CGMainDisplayID  # noqa: F401

        return "quartz"
    except Exception:
        pass
    try:
        import mss  # noqa: F401

        return "mss"
    except Exception:
        return "none"


def request_screen_recording() -> bool | None:
    """Show the system prompt. Do not call from pytest (it can hang)."""
    try:
        from Quartz import CGRequestScreenCaptureAccess

        return bool(CGRequestScreenCaptureAccess())
    except Exception:
        return None


def _in_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def _primary_screen():
    """The primary display's NSScreen (the one with the menu bar, which
    ``CGMainDisplayID`` grabs). ``NSScreen.mainScreen`` is the screen with
    the key window, which differs on a multi-display setup."""
    from AppKit import NSScreen

    screens = NSScreen.screens()
    if screens is not None and len(screens):
        return screens[0]
    return NSScreen.mainScreen()


def _cocoa_primary_frame() -> tuple[float, float, float]:
    """Return (origin_x, origin_y, height) in Cocoa points for the primary screen."""
    try:
        screen = _primary_screen()
        if screen is None:
            raise RuntimeError("no main screen")
        frame = screen.frame()
        return float(frame.origin.x), float(frame.origin.y), float(frame.size.height)
    except Exception:
        return 0.0, 0.0, 0.0


def _cocoa_visible_frame() -> tuple[float, float, float, float] | None:
    """``visibleFrame`` (x, y, w, h) of the primary screen in Cocoa points:
    the screen minus the menu bar and the Dock (an auto-hidden Dock leaves
    only a thin strip). None when AppKit is unavailable."""
    try:
        screen = _primary_screen()
        if screen is None:
            return None
        f = screen.visibleFrame()
        return float(f.origin.x), float(f.origin.y), float(f.size.width), float(f.size.height)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Window list → ScreenLayout
# ---------------------------------------------------------------------------

WINDOW_CHANGE_STALE_MS = 300.0  # after a window appears / disappears / resizes, the list may lead the pixels
MISSION_CONTROL_FRAC = 0.9  # a new Dock-owned overlay this big = Mission Control / Launchpad / App Exposé
DOCK_OVERLAY_STEADY_S = 10.0  # a big Dock overlay present this long (or at start) is the Dock itself


def _num(d, key: str, default: float = 0.0) -> float:
    try:
        v = d.get(key)
        return float(v) if v is not None else float(default)
    except Exception:
        return float(default)


def window_layout_from_info(
    info,
    geom: DisplayGeom,
    *,
    exclude_ids: set[int] | frozenset[int] = frozenset(),
    exclude_pids: set[int] | frozenset[int] = frozenset(),
    exclude_titles: frozenset[str] = EXCLUDE_TITLES,
    visible_cocoa: tuple[float, float, float, float] | None = None,
    t_ms: float = 0.0,
    steady_dock_ids: set[int] | frozenset[int] = frozenset(),
) -> ScreenLayout:
    """``CGWindowListCopyWindowInfo`` dicts (front to back) → ``ScreenLayout``
    in capture px. Our own windows (by id, pid, owner or title) are left
    out; titles are only compared, never kept. Windows on other displays
    (outside the grabbed frame) are dropped. ``steady_dock_ids``: big Dock
    overlays that are always there (recent macOS keeps a full-screen Dock
    window at the Dock level), so they do not mean Mission Control."""
    W, H = int(geom.width_px), int(geom.height_px)
    wins: list[WindowInfo] = []
    stale = False
    for d in info or []:
        try:
            wid = int(d.get("kCGWindowNumber") or 0)
        except Exception:
            continue
        if not wid or wid in exclude_ids:
            continue
        owner = str(d.get("kCGWindowOwnerName") or "")
        pid = int(_num(d, "kCGWindowOwnerPID"))
        title = str(d.get("kCGWindowName") or "")
        if pid in exclude_pids or title in exclude_titles or title.startswith("Screen Guardian"):
            continue  # our own overlay: never an owner, never an occluder
        own_ui = owner in EXCLUDE_OWNERS  # the island: hides what is behind it, owns no pictures
        b = d.get("kCGWindowBounds") or {}
        x, y, w, h = _num(b, "X"), _num(b, "Y"), _num(b, "Width"), _num(b, "Height")
        if w <= 0 or h <= 0:
            continue
        xyxy = points_xyxy_to_capture((x, y, x + w, y + h), geom)
        if xyxy[2] <= 0 or xyxy[3] <= 0 or xyxy[0] >= W or xyxy[1] >= H:
            continue  # another display
        layer = int(_num(d, "kCGWindowLayer"))
        if own_ui:
            layer = max(1, layer)
        alpha = _num(d, "kCGWindowAlpha", 1.0)
        if layer == 0 and 0.05 < alpha < 0.99:
            stale = True  # fading in / out: the list leads or lags the pixels
        if _big_dock_overlay(owner, layer, xyxy, W, H) and wid not in steady_dock_ids:
            stale = True  # Mission Control / Launchpad: every window is rescaled
        wins.append(WindowInfo(wid, xyxy, layer=layer, owner_pid=pid, owner=owner, alpha=alpha))
    visible = cocoa_rect_to_capture(visible_cocoa, geom) if visible_cocoa is not None else None
    return ScreenLayout(frame_wh=(W, H), windows=tuple(wins), visible_xyxy=visible, t_ms=float(t_ms), stale=stale)


def _big_dock_overlay(owner: str, layer: int, xyxy, W: int, H: int) -> bool:
    if owner != "Dock" or layer <= 0:
        return False
    return (min(W, xyxy[2]) - max(0, xyxy[0])) * (min(H, xyxy[3]) - max(0, xyxy[1])) >= MISSION_CONTROL_FRAC * W * H


class WindowLister:
    """Builds layouts and marks them stale for a short while after the set
    of app windows (or one's size) changes: open / close / minimise / zoom
    animations run ~250 ms, during which the list and the pixels disagree.
    Big Dock overlays seen on the first build, or present for
    ``DOCK_OVERLAY_STEADY_S``, are the Dock itself, not Mission Control."""

    def __init__(self) -> None:
        self._last_sig: dict[int, tuple[float, float]] | None = None
        self._changed_at: float = -1e9
        self._dock_seen: dict[int, float] | None = None  # big Dock overlay id → first seen
        self.exclude_window_ids: list[int] = []

    def _steady_dock_ids(self, info, geom: DisplayGeom, now: float) -> frozenset[int]:
        W, H = int(geom.width_px), int(geom.height_px)
        ids = set()
        for d in info or []:
            b = d.get("kCGWindowBounds") or {}
            x, y, w, h = _num(b, "X"), _num(b, "Y"), _num(b, "Width"), _num(b, "Height")
            xyxy = points_xyxy_to_capture((x, y, x + w, y + h), geom)
            if _big_dock_overlay(str(d.get("kCGWindowOwnerName") or ""), int(_num(d, "kCGWindowLayer")), xyxy, W, H):
                ids.add(int(_num(d, "kCGWindowNumber")))
        first = self._dock_seen is None
        seen = {} if first else self._dock_seen
        self._dock_seen = {i: (-1e9 if first else seen.get(i, now)) for i in ids}
        return frozenset(i for i, t in self._dock_seen.items() if now - t >= DOCK_OVERLAY_STEADY_S)

    def build(self, info, geom: DisplayGeom, *, t_ms: float | None = None, now: float | None = None) -> ScreenLayout:
        now = time.perf_counter() if now is None else now
        layout = window_layout_from_info(
            info,
            geom,
            exclude_ids=frozenset(int(i) for i in self.exclude_window_ids),
            exclude_pids=frozenset({os.getpid()}),
            visible_cocoa=_cocoa_visible_frame(),
            t_ms=0.0 if t_ms is None else t_ms,
            steady_dock_ids=self._steady_dock_ids(info, geom, now),
        )
        sig = {w.window_id: (round(w.xyxy[2] - w.xyxy[0]), round(w.xyxy[3] - w.xyxy[1])) for w in layout.windows if w.is_app}
        if self._last_sig is not None and sig != self._last_sig:
            self._changed_at = now
        self._last_sig = sig
        if (now - self._changed_at) * 1000.0 < WINDOW_CHANGE_STALE_MS and not layout.stale:
            layout = ScreenLayout(layout.frame_wh, layout.windows, layout.visible_xyxy, layout.t_ms, stale=True)
        return layout


def _quartz_window_info():
    from Quartz import CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionOnScreenOnly

    return CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID) or []


class WindowWatcher:
    """Fresh window list on the main thread for the 60 Hz overlay clip
    (~1–3 ms per call). ``poll`` returns the cached layout if called again
    within ``min_interval_s``; None when Quartz is unavailable."""

    def __init__(self, *, min_interval_s: float = 1.0 / 90.0) -> None:
        self.min_interval_s = float(min_interval_s)
        self.exclude_window_ids: list[int] = []
        self._lister = WindowLister()
        self._t = 0.0
        self._layout: ScreenLayout | None = None
        self._key: tuple | None = None
        self.last_ms = 0.0
        try:
            _quartz_window_info  # noqa: B018
            from Quartz import CGWindowListCopyWindowInfo  # noqa: F401

            self.available = True
        except Exception:
            self.available = False

    def poll(self, geom: DisplayGeom | None) -> ScreenLayout | None:
        if not self.available or geom is None:
            return None
        now = time.perf_counter()
        key = (geom.width_px, geom.height_px, geom.scale_x, geom.scale_y)
        if self._layout is not None and self._key == key and now - self._t < self.min_interval_s:
            return self._layout
        t0 = now
        try:
            info = _quartz_window_info()
        except Exception:
            return None
        self._lister.exclude_window_ids = self.exclude_window_ids
        self._layout = self._lister.build(info, geom, now=now)
        self._t, self._key = now, key
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return self._layout


def _cgimage_to_bgr(cg_image) -> np.ndarray | None:
    if cg_image is None:
        return None
    try:
        from Quartz import (
            CGDataProviderCopyData,
            CGImageGetBitsPerPixel,
            CGImageGetBytesPerRow,
            CGImageGetDataProvider,
            CGImageGetHeight,
            CGImageGetWidth,
        )
    except Exception:
        return None
    width = int(CGImageGetWidth(cg_image))
    height = int(CGImageGetHeight(cg_image))
    if width <= 0 or height <= 0:
        return None
    row_bytes = int(CGImageGetBytesPerRow(cg_image))
    bpp = max(1, int(CGImageGetBitsPerPixel(cg_image)) // 8)
    if bpp < 3:
        return None
    provider = CGImageGetDataProvider(cg_image)
    data = CGDataProviderCopyData(provider)
    # Zero-copy view of the CFData (rows may be padded), then ONE copy into
    # a contiguous BGR array. The old path copied the Retina frame 3×
    # (bytes(), .copy(), ascontiguousarray).
    try:
        buf = np.frombuffer(memoryview(data), dtype=np.uint8)
    except TypeError:
        buf = np.frombuffer(bytes(data), dtype=np.uint8)
    if buf.size < (height - 1) * row_bytes + width * bpp:
        return None
    view = np.lib.stride_tricks.as_strided(
        buf,
        shape=(height, width, bpp),
        strides=(row_bytes, bpp, 1),
        writeable=False,
    )
    if bpp == 4:
        return cv2.cvtColor(view, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(view[:, :, :3])


def _sample(bgr: np.ndarray) -> np.ndarray:
    """Sparse view for stats: mean/std over a full Retina grab costs ~25 ms."""
    step = max(1, min(bgr.shape[0], bgr.shape[1]) // 256) if bgr.ndim >= 2 else 1
    return bgr[::step, ::step]


def _looks_like_denied(bgr: np.ndarray) -> bool:
    if bgr.size == 0:
        return True
    sample = _sample(bgr)
    return float(sample.mean()) < 1.0 and float(sample.std()) < 1.5


class QuartzGrabber:
    """Primary-display grab via Quartz, excluding Screen Guardian windows.

    ScreenCaptureKit is not used. Window exclusion is title / id based so
    the overlay does not feed back into the detector (hall-of-mirrors).
    """

    def __init__(self) -> None:
        self.last_reason: str | None = None
        self.lister = WindowLister()

    def grab(
        self,
        exclude_window_ids: list[int] | None = None,
        exclude_titles: frozenset[str] = EXCLUDE_TITLES,
    ) -> tuple[np.ndarray, DisplayGeom, ScreenLayout | None] | None:
        try:
            from Quartz import (
                CGDisplayBounds,
                CGMainDisplayID,
                CGWindowListCopyWindowInfo,
                CGWindowListCreateImageFromArray,
                kCGNullWindowID,
                kCGWindowImageDefault,
                kCGWindowListOptionOnScreenOnly,
            )
        except Exception as exc:
            self.last_reason = f"capture unavailable: Quartz import failed ({exc})"
            return None

        display_id = CGMainDisplayID()
        bounds = CGDisplayBounds(display_id)
        left = float(bounds.origin.x)
        top = float(bounds.origin.y)
        width_pt = float(bounds.size.width)
        height_pt = float(bounds.size.height)
        exclude = {int(i) for i in (exclude_window_ids or [])}
        me = os.getpid()

        try:
            info = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID) or []
            keep = []
            for window in info:
                wid = int(window.get("kCGWindowNumber") or 0)
                title = str(window.get("kCGWindowName") or "")
                owner = str(window.get("kCGWindowOwnerName") or "")
                if wid and wid in exclude:
                    continue
                if title in exclude_titles or title.startswith("Screen Guardian"):
                    continue
                if owner in EXCLUDE_OWNERS:
                    continue
                try:
                    if int(window.get("kCGWindowOwnerPID") or 0) == me:
                        continue  # our own overlay / status windows
                except Exception:
                    pass
                if wid:
                    keep.append(wid)
            image = CGWindowListCreateImageFromArray(bounds, keep, kCGWindowImageDefault)
        except Exception as exc:
            self.last_reason = f"capture unavailable: Quartz grab failed ({exc})"
            return None

        bgr = _cgimage_to_bgr(image)
        if bgr is None:
            self.last_reason = "capture unavailable: Quartz returned no image"
            return None
        if _looks_like_denied(bgr):
            self.last_reason = "capture unavailable: Screen Recording permission denied (black frame)"
            return None
        px_h, px_w = bgr.shape[:2]
        scale_x, scale_y = scale_from_sizes(px_w, px_h, width_pt, height_pt)
        ox, oy, cocoa_h = _cocoa_primary_frame()
        if cocoa_h <= 0:
            cocoa_h = height_pt
        geom = DisplayGeom(
            left_pt=left,
            top_pt=top,
            width_pt=width_pt,
            height_pt=height_pt,
            width_px=px_w,
            height_px=px_h,
            scale_x=scale_x,
            scale_y=scale_y,
            cocoa_origin_x=ox,
            cocoa_origin_y=oy,
            cocoa_height_pt=cocoa_h,
        )
        self.last_reason = None
        try:
            self.lister.exclude_window_ids = sorted(exclude)
            layout = self.lister.build(info, geom)
        except Exception:  # noqa: BLE001 - the engine falls back to "no window list"
            layout = None
        return bgr, geom, layout


class MssGrabber:
    """Fallback primary-display grab. Cannot exclude overlay windows."""

    def __init__(self) -> None:
        self.last_reason: str | None = None
        self._sct = None

    def grab(
        self,
        exclude_window_ids: list[int] | None = None,
        exclude_titles: frozenset[str] = EXCLUDE_TITLES,
    ) -> tuple[np.ndarray, DisplayGeom] | None:
        del exclude_window_ids, exclude_titles
        try:
            from mss import mss
        except Exception as exc:
            self.last_reason = f"capture unavailable: mss import failed ({exc})"
            return None
        if self._sct is None:
            self._sct = mss()
        monitors = self._sct.monitors
        if len(monitors) < 2:
            self.last_reason = "capture unavailable: mss found no display"
            return None
        mon = monitors[1]
        shot = self._sct.grab(mon)
        bgra = np.array(shot, dtype=np.uint8)
        if bgra.ndim != 3 or bgra.shape[2] < 3:
            self.last_reason = "capture unavailable: mss returned an empty frame"
            return None
        bgr = np.ascontiguousarray(bgra[:, :, :3])
        if _looks_like_denied(bgr):
            self.last_reason = "capture unavailable: Screen Recording permission denied (black frame)"
            return None
        px_h, px_w = bgr.shape[:2]
        width_pt = float(mon.get("width") or px_w)
        height_pt = float(mon.get("height") or px_h)
        scale_x, scale_y = scale_from_sizes(px_w, px_h, width_pt, height_pt)
        ox, oy, cocoa_h = _cocoa_primary_frame()
        if cocoa_h <= 0:
            cocoa_h = height_pt
        geom = DisplayGeom(
            left_pt=float(mon.get("left") or 0.0),
            top_pt=float(mon.get("top") or 0.0),
            width_pt=width_pt,
            height_pt=height_pt,
            width_px=px_w,
            height_px=px_h,
            scale_x=scale_x,
            scale_y=scale_y,
            cocoa_origin_x=ox,
            cocoa_origin_y=oy,
            cocoa_height_pt=cocoa_h,
        )
        self.last_reason = None
        return bgr, geom

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None


class ReplayCapture:
    """Read ``fixtures/safe/synthetic_moving_regions.mp4`` only.

    The demo path never opens another file. Tests may pass an explicit
    path under ``fixtures/safe/``.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        loop: bool = True,
    ) -> None:
        resolved = Path(path) if path is not None else ensure_safe_fixture()
        resolved = resolved.resolve()
        if path is not None:
            try:
                resolved.relative_to(SAFE_DIR.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"ReplayCapture only opens files under {SAFE_DIR}"
                ) from exc
        elif resolved != SAFE_FIXTURE.resolve():
            raise ValueError("default replay must be the safe fixture")

        self.path = resolved
        self.loop = loop
        self.cap = cv2.VideoCapture(str(resolved))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"capture unavailable: failed to open {resolved}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._index = 0
        self.available = True
        self.fixture_id = resolved.stem
        self.geom: DisplayGeom | None = None
        self.backend = "replay"
        self.granted = True
        self.last_ok = False
        self.last_mean = 0.0
        self.last_std = 0.0
        self.last_shape: tuple[int, ...] | None = None

    def read(self) -> Frame | None:
        ok, bgr = self.cap.read()
        if not ok:
            if not self.loop:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = self.cap.read()
            if not ok:
                return None
        height, width = bgr.shape[:2]
        self.last_ok = True
        self.last_shape = bgr.shape
        self.last_mean = float(bgr.mean())
        self.last_std = float(bgr.std())
        t_ms = self._index * (1000.0 / self.fps if self.fps > 0 else 33.333)
        frame = Frame(
            frame_id=self._index,
            t_ms=t_ms,
            image_bgr=bgr,
            width=width,
            height=height,
        )
        self._index += 1
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.available = False

    def __enter__(self) -> ReplayCapture:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


GrabFn = Callable[..., tuple[np.ndarray, DisplayGeom] | None]


class LiveCapture:
    """Primary-display capture for ``--source live``.

    Target 8–15 FPS so the laptop stays usable. Overlay / HUD windows are
    excluded when Quartz can see their titles. pytest will not grab a real
    display unless a ``grabber`` is injected or ``allow_real=True``.
    """

    def __init__(
        self,
        *,
        target_fps: float = 12.0,
        grabber: GrabFn | object | None = None,
        exclude_window_ids: list[int] | None = None,
        allow_real: bool | None = None,
        request_permission: bool = False,
    ) -> None:
        self.target_fps = max(4.0, min(15.0, float(target_fps)))
        self.fps = self.target_fps
        self.exclude_window_ids = list(exclude_window_ids or [])
        self.geom: DisplayGeom | None = None
        self.windows: ScreenLayout | None = None  # window list of the last grab (None = unknown)
        self.reason = "capture unavailable"
        self.available = False
        self.backend = "none"
        self._index = 0
        self._last_t = 0.0
        self._t0: float | None = None
        self.last_grab_ms = 0.0
        self._grabber = grabber
        self._owns_mss: MssGrabber | None = None
        self.width = 0
        self.height = 0
        self.fixture_id = "live_desktop"
        self.path = None
        self.granted: bool | None = None
        self.last_ok = False
        self.last_mean = 0.0
        self.last_std = 0.0
        self.last_shape: tuple[int, ...] | None = None

        if allow_real is None:
            allow_real = os.environ.get("SCREEN_GUARDIAN_ALLOW_LIVE") == "1" or not _in_pytest()
        self._allow_real = bool(allow_real)

        if grabber is not None:
            self.available = True
            self.backend = "mock"
            self.reason = ""
            return

        if _in_pytest() and not self._allow_real:
            self.reason = "capture unavailable: live grab disabled under pytest (inject grabber)"
            return

        granted = screen_recording_granted()
        self.granted = granted
        if granted is False:
            if request_permission and not _in_pytest():
                granted = request_screen_recording()
            if granted is False:
                self.reason = PERMISSION_HELP
                return

        quartz = QuartzGrabber()
        mss_grab = MssGrabber()
        self._owns_mss = mss_grab

        def _real_grab(*, exclude_window_ids=None, exclude_titles=EXCLUDE_TITLES):
            ids = exclude_window_ids if exclude_window_ids is not None else self.exclude_window_ids
            got = quartz.grab(ids, exclude_titles)
            if got is not None:
                self.backend = "quartz"
                return got
            got = mss_grab.grab(ids, exclude_titles)
            if got is not None:
                self.backend = "mss"
                return got
            self.reason = quartz.last_reason or mss_grab.last_reason or PERMISSION_HELP
            return None

        self._grabber = _real_grab
        self.available = True
        self.reason = ""
        self.backend = preferred_live_backend()

    def read(self) -> Frame | None:
        if not self.available or self._grabber is None:
            return None
        min_dt = 1.0 / self.target_fps
        now = time.perf_counter()
        wait = self._last_t + min_dt - now
        if self._last_t > 0 and wait > 0:
            time.sleep(wait)
        self._last_t = time.perf_counter()
        if self._t0 is None:
            self._t0 = self._last_t

        try:
            if callable(self._grabber):
                got = self._grabber(
                    exclude_window_ids=self.exclude_window_ids,
                    exclude_titles=EXCLUDE_TITLES,
                )
            else:
                got = self._grabber.grab(
                    exclude_window_ids=self.exclude_window_ids,
                    exclude_titles=EXCLUDE_TITLES,
                )
        except Exception as exc:
            self.available = False
            self.reason = f"capture unavailable: {exc}"
            return None

        if got is None:
            self.available = False
            self.last_ok = False
            if not self.reason:
                self.reason = PERMISSION_HELP
            return None
        bgr, geom = got[0], got[1]
        layout = got[2] if len(got) > 2 else None
        self.last_grab_ms = (time.perf_counter() - self._last_t) * 1000.0
        self.geom = geom
        self.width = geom.width_px
        self.height = geom.height_px
        self.last_ok = True
        self.last_shape = bgr.shape
        sample = _sample(bgr) if bgr.size else bgr
        self.last_mean = float(sample.mean()) if bgr.size else 0.0
        self.last_std = float(sample.std()) if bgr.size else 0.0
        # Monotonic ms since the first grab (start of this grab), not
        # index × 1000/fps, so keyframe rules and velocities survive slow loops.
        t_ms = (self._last_t - self._t0) * 1000.0
        if layout is not None and tuple(layout.frame_wh) == (bgr.shape[1], bgr.shape[0]):
            self.windows = ScreenLayout(layout.frame_wh, layout.windows, layout.visible_xyxy, t_ms, layout.stale)
        else:
            self.windows = None
        frame = Frame(
            frame_id=self._index,
            t_ms=t_ms,
            image_bgr=bgr,
            width=geom.width_px,
            height=geom.height_px,
        )
        self._index += 1
        return frame

    def release(self) -> None:
        self.available = False
        if self._owns_mss is not None:
            self._owns_mss.close()
            self._owns_mss = None

    def require(self) -> None:
        if self.available:
            return
        raise RuntimeError(self.reason or PERMISSION_HELP)

    def __enter__(self) -> LiveCapture:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def open_source(source: str, **kwargs) -> CaptureAdapter:
    key = (source or "replay").strip().lower()
    if key == "replay":
        return ReplayCapture()
    if key == "live":
        return LiveCapture(**kwargs)
    raise ValueError(f"unknown source {source!r}; use replay or live")
