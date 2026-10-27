"""Always-on-top macOS mask overlay for live protection.

ONE borderless, transparent, click-through window covers the primary
display. Every mask is a subview of it, keyed by a stable id, so a mask
that follows a scroll is a frame move (implicit animations off), not a new
window. The window:

- ignores mouse events (scrolling and clicking pass through to the page),
- sits above normal, fullscreen and menu windows
  (``CanJoinAllSpaces | FullScreenAuxiliary | Stationary | IgnoresCycle``),
- sets ``NSWindowSharingNone`` and a title starting with "Screen Guardian",
  which ``edge.capture`` excludes (no hall-of-mirrors feedback).
  ``SG_CAPTURABLE=1`` makes it read-only shareable so a screen recorder can
  film the frost; the title/pid exclusion still keeps it out of our capture.

FROST = the strongest behind-window ``NSVisualEffectView`` material plus a
tint layer, so text is not readable through it. OPAQUE = fully opaque fill
with rounded corners and a faint stripe pattern.

Capture-pixel boxes are mapped with ``edge.coords``. AppKit can fail in a
sandbox / unsigned agent; callers keep working with the memory backend.

The live engine hands ``MaskOverlay.update`` masks that are already padded
and **cut to their window's visible region** (``LiveEngine.tick`` →
``vision_core.live.windows.clip_specs``), and passes ``pad_pt=0`` so no pad
is added after the cut: a mask never spills onto the menu bar, the Dock, a
window in front, or the island (an occluder in the window list). The window
level stays above menus so full-screen apps are covered; the cut, not the
z-order, keeps system UI visible.
"""

from __future__ import annotations

import os

from typing import TYPE_CHECKING, Protocol

from edge.coords import DisplayGeom, capture_xyxy_to_cocoa
from vision_core.cluster_nsfw import MAX_CLUSTERS, cluster_xyxys

if TYPE_CHECKING:
    from edge.reflex import ReflexState
    from vision_core.live.contracts import MaskSpec

# Same SERP-grid safety ceiling as clustering (legacy reflex path).
MAX_OVERLAY_WINDOWS = MAX_CLUSTERS
from vision_core.contracts import PROTECTIVE_ACTIONS, ActionType
from vision_core.geometry import box_iou

OVERLAY_TITLE = "Screen Guardian Overlay"
MASK_TITLE = "Screen Guardian Mask"
STATUS_TITLE = "Screen Guardian Status"
MASK_PAD_PT = 4.0
CORNER_RADIUS_PT = 10.0
FADE_IN_S = 0.0  # appear instantly (fail closed); > 0 = quick fade-in
FROST_TINT_RGBA = (0.20, 0.23, 0.28, 0.62)
OPAQUE_RGBA = (0.15, 0.17, 0.21, 1.0)
STRIPE_RGBA = (1.0, 1.0, 1.0, 0.05)

# (x, y, w, h) Cocoa global points, bottom-left origin.
Rect = tuple[float, float, float, float]

def _cocoa_to_xyxy(rect: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = rect
    return (x, y, x + w, y + h)


def match_rects(
    previous: list[tuple[float, float, float, float]],
    incoming: list[tuple[float, float, float, float]],
    *,
    iou_min: float = 0.08,
) -> list[int | None]:
    """For each incoming Cocoa rect, index of a previous rect to reuse."""
    assigned_prev: set[int] = set()
    assigned_new: set[int] = set()
    out: list[int | None] = [None] * len(incoming)
    pairs: list[tuple[float, int, int]] = []
    for ni, nr in enumerate(incoming):
        nbox = _cocoa_to_xyxy(nr)
        for pi, pr in enumerate(previous):
            iou = box_iou(nbox, _cocoa_to_xyxy(pr))
            if iou >= iou_min:
                pairs.append((iou, ni, pi))
    pairs.sort(reverse=True)
    for _iou, ni, pi in pairs:
        if ni in assigned_new or pi in assigned_prev:
            continue
        assigned_new.add(ni)
        assigned_prev.add(pi)
        out[ni] = pi
    return out


class OverlayBackend(Protocol):
    def set_rects(self, rects: list[tuple[float, float, float, float]], style: str) -> None: ...

    def set_status(self, text: str) -> None: ...

    def mask_count(self) -> int: ...

    def window_ids(self) -> list[int]: ...

    def pump(self) -> None: ...

    def close(self) -> None: ...


class MemoryOverlayBackend:
    """Test double. Records rects; no display."""

    def __init__(self) -> None:
        self.rects: list[tuple[float, float, float, float]] = []
        self.style: str = "frost"
        self.status: str = ""
        self.closed = False

    def set_rects(self, rects: list[tuple[float, float, float, float]], style: str) -> None:
        self.rects = list(rects)
        self.style = style

    def set_status(self, text: str) -> None:
        self.status = text

    def mask_count(self) -> int:
        return 0 if self.closed else len(self.rects)

    def window_ids(self) -> list[int]:
        return list(range(1, len(self.rects) + 1))

    def pump(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True
        self.rects = []


def overlay_level() -> int:
    """Above normal, fullscreen and pop-up menu windows; below the screen
    saver level so the island can sit on top (use ≥ NSScreenSaverWindowLevel
    there if it must stay visible over masks)."""
    try:
        from AppKit import NSPopUpMenuWindowLevel

        return int(NSPopUpMenuWindowLevel) + 1
    except Exception:
        return 102


def pump_appkit_events(max_events: int = 64) -> None:
    """Drain pending AppKit events without blocking (legacy loop only; the
    new engine runs ``NSApplication.run`` with a timer)."""
    try:
        from AppKit import NSApplication
        from Foundation import NSDate, NSDefaultRunLoopMode

        app = NSApplication.sharedApplication()
        for _ in range(max_events):
            event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                2**64 - 1,
                NSDate.dateWithTimeIntervalSinceNow_(0.0),
                NSDefaultRunLoopMode,
                True,
            )
            if event is None:
                break
            app.sendEvent_(event)
    except Exception:
        return


def _cg_rgba(rgba: tuple[float, float, float, float]):
    from Quartz import CGColorCreateGenericRGB

    return CGColorCreateGenericRGB(*rgba)


def _stripe_layer(size: float = 4096.0, step: float = 22.0):
    """Faint diagonal stripes, drawn once big and clipped by the mask."""
    from Quartz import (
        CAShapeLayer,
        CGPathAddLineToPoint,
        CGPathCreateMutable,
        CGPathMoveToPoint,
    )

    path = CGPathCreateMutable()
    x = -size
    while x < size:
        CGPathMoveToPoint(path, None, x, 0.0)
        CGPathAddLineToPoint(path, None, x + size, size)
        x += step
    layer = CAShapeLayer.layer()
    layer.setFrame_(((0.0, 0.0), (size, size)))
    layer.setPath_(path)
    layer.setStrokeColor_(_cg_rgba(STRIPE_RGBA))
    layer.setFillColor_(None)
    layer.setLineWidth_(6.0)
    return layer


def _make_mask_view(ns_rect, style: str):
    from AppKit import NSView, NSViewHeightSizable, NSViewWidthSizable

    container = NSView.alloc().initWithFrame_(ns_rect)
    container.setWantsLayer_(True)
    layer = container.layer()
    layer.setCornerRadius_(CORNER_RADIUS_PT)
    layer.setMasksToBounds_(True)
    layer.setBorderWidth_(1.0)
    layer.setBorderColor_(_cg_rgba((1.0, 1.0, 1.0, 0.16)))
    sizable = NSViewWidthSizable | NSViewHeightSizable
    if style == "opaque":
        layer.setBackgroundColor_(_cg_rgba(OPAQUE_RGBA))
        try:
            layer.addSublayer_(_stripe_layer())
        except Exception:
            pass
        return container
    try:
        from AppKit import (
            NSAppearance,
            NSAppearanceNameDarkAqua,
            NSVisualEffectBlendingModeBehindWindow,
            NSVisualEffectStateActive,
            NSVisualEffectView,
        )

        try:
            from AppKit import NSVisualEffectMaterialFullScreenUI as material
        except Exception:
            from AppKit import NSVisualEffectMaterialHUDWindow as material
        fx = NSVisualEffectView.alloc().initWithFrame_(container.bounds())
        fx.setMaterial_(material)
        fx.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        fx.setState_(NSVisualEffectStateActive)
        try:
            fx.setAppearance_(NSAppearance.appearanceNamed_(NSAppearanceNameDarkAqua))
        except Exception:
            pass
        fx.setAutoresizingMask_(sizable)
        container.addSubview_(fx)
    except Exception:
        pass
    tint = NSView.alloc().initWithFrame_(container.bounds())
    tint.setWantsLayer_(True)
    tint.layer().setBackgroundColor_(_cg_rgba(FROST_TINT_RGBA))
    tint.setAutoresizingMask_(sizable)
    container.addSubview_(tint)
    return container


class AppKitMaskWindow:
    """The single overlay window. Main thread only."""

    def __init__(self, title: str = OVERLAY_TITLE) -> None:
        from AppKit import (
            NSApplication,
            NSApplicationActivationPolicyAccessory,
            NSBackingStoreBuffered,
            NSColor,
            NSMakeRect,
            NSScreen,
            NSView,
            NSWindow,
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorFullScreenAuxiliary,
            NSWindowCollectionBehaviorIgnoresCycle,
            NSWindowCollectionBehaviorStationary,
            NSWindowSharingNone,
            NSWindowSharingReadOnly,
            NSWindowStyleMaskBorderless,
        )

        self._app = NSApplication.sharedApplication()
        try:
            self._app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        except Exception:
            pass
        screen = NSScreen.mainScreen()
        if screen is None:
            raise RuntimeError("no main screen")
        frame = screen.frame()
        self.origin = (float(frame.origin.x), float(frame.origin.y))
        self.size = (float(frame.size.width), float(frame.size.height))
        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False,
        )
        window.setTitle_(title)
        window.setOpaque_(False)
        window.setHasShadow_(False)
        window.setBackgroundColor_(NSColor.clearColor())
        window.setLevel_(overlay_level())
        window.setIgnoresMouseEvents_(True)
        window.setHidesOnDeactivate_(False)
        window.setReleasedWhenClosed_(False)
        try:
            capturable = os.environ.get("SG_CAPTURABLE") == "1"
            window.setSharingType_(NSWindowSharingReadOnly if capturable else NSWindowSharingNone)
        except Exception:
            pass
        window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorIgnoresCycle
        )
        content = NSView.alloc().initWithFrame_(NSMakeRect(0.0, 0.0, self.size[0], self.size[1]))
        content.setWantsLayer_(True)
        window.setContentView_(content)
        window.orderFrontRegardless()
        self._window = window
        self._content = content
        self._views: dict[str, tuple[object, str]] = {}
        self._status_label = None

    def _local(self, rect: Rect):
        from AppKit import NSMakeRect

        x, y, w, h = rect
        return NSMakeRect(x - self.origin[0], y - self.origin[1], max(4.0, w), max(4.0, h))

    def apply(self, items: dict[str, tuple[Rect, str]]) -> None:
        """Make the subviews match ``items`` (key → (Cocoa rect, style))."""
        from Quartz import CATransaction

        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        try:
            for key in [k for k in self._views if k not in items]:
                view, _style = self._views.pop(key)
                view.removeFromSuperview()
            for key, (rect, style) in items.items():
                ns_rect = self._local(rect)
                held = self._views.get(key)
                if held is not None and held[1] == style:
                    held[0].setFrame_(ns_rect)
                    continue
                if held is not None:
                    held[0].removeFromSuperview()
                view = _make_mask_view(ns_rect, style)
                if FADE_IN_S > 0:
                    view.setAlphaValue_(0.0)
                self._content.addSubview_(view)
                self._views[key] = (view, style)
                if FADE_IN_S > 0:
                    self._fade_in(view)
        finally:
            CATransaction.commit()
        if items:
            try:
                if not self._window.isVisible():
                    self._window.orderFrontRegardless()
            except Exception:
                pass

    @staticmethod
    def _fade_in(view) -> None:
        try:
            from AppKit import NSAnimationContext

            def _anim(ctx):
                ctx.setDuration_(FADE_IN_S)
                view.animator().setAlphaValue_(1.0)

            NSAnimationContext.runAnimationGroup_completionHandler_(_anim, None)
        except Exception:
            view.setAlphaValue_(1.0)

    def set_status(self, text: str) -> None:
        """Small top-right status chip (legacy reflex HUD only)."""
        try:
            if self._status_label is None:
                from AppKit import NSColor, NSFont, NSMakeRect, NSTextField

                width, height = 300.0, 22.0
                label = NSTextField.alloc().initWithFrame_(
                    NSMakeRect(self.size[0] - width - 16.0, self.size[1] - height - 30.0, width, height)
                )
                label.setBezeled_(False)
                label.setDrawsBackground_(True)
                label.setBackgroundColor_(NSColor.blackColor().colorWithAlphaComponent_(0.55))
                label.setEditable_(False)
                label.setSelectable_(False)
                label.setTextColor_(NSColor.whiteColor())
                label.setFont_(NSFont.systemFontOfSize_(11.0))
                self._content.addSubview_(label)
                self._status_label = label
            self._status_label.setStringValue_(text[:80])
        except Exception:
            self._status_label = None

    def mask_count(self) -> int:
        return len(self._views)

    def window_ids(self) -> list[int]:
        try:
            return [int(self._window.windowNumber())]
        except Exception:
            return []

    def close(self) -> None:
        for view, _style in self._views.values():
            try:
                view.removeFromSuperview()
            except Exception:
                pass
        self._views = {}
        try:
            self._window.orderOut_(None)
        except Exception:
            pass
        try:
            self._window.close()
        except Exception:
            pass


class MaskBackend(Protocol):
    def apply(self, items: dict[str, tuple[Rect, str]]) -> None: ...

    def mask_count(self) -> int: ...

    def window_ids(self) -> list[int]: ...

    def close(self) -> None: ...


class MemoryMaskBackend:
    """Test double for the single-window overlay. Counts view churn so tests
    can assert that a moving mask is moved, not recreated."""

    def __init__(self) -> None:
        self.items: dict[str, tuple[Rect, str]] = {}
        self.created = 0
        self.moved = 0
        self.removed = 0
        self.applies = 0
        self.closed = False

    def apply(self, items: dict[str, tuple[Rect, str]]) -> None:
        self.applies += 1
        for key in self.items:
            if key not in items:
                self.removed += 1
        for key, (rect, style) in items.items():
            held = self.items.get(key)
            if held is None or held[1] != style:
                self.created += 1
            elif held[0] != rect:
                self.moved += 1
        self.items = dict(items)

    def mask_count(self) -> int:
        return 0 if self.closed else len(self.items)

    def window_ids(self) -> list[int]:
        return [1]

    def close(self) -> None:
        self.closed = True
        self.items = {}


class MaskOverlay:
    """Live-engine overlay controller: ``MaskSpec`` (capture px) → keyed
    Cocoa rects → backend. Call ``update`` from the main thread."""

    def __init__(self, backend: MaskBackend | None = None, *, pad_pt: float = MASK_PAD_PT) -> None:
        self.backend = backend
        self.pad = float(pad_pt)
        self.error: str | None = None
        self._last: dict[str, tuple[Rect, str]] | None = None
        if self.backend is None:
            try:
                self.backend = AppKitMaskWindow()
            except Exception as exc:
                self.error = f"on-screen overlay unavailable: {exc}"
                self.backend = None
        self.available = self.backend is not None

    def rects_for(self, masks: list[MaskSpec], geom: DisplayGeom, pad_pt: float | None = None) -> dict[str, tuple[Rect, str]]:
        items: dict[str, tuple[Rect, str]] = {}
        p = self.pad if pad_pt is None else float(pad_pt)
        for mask in masks:
            x, y, w, h = capture_xyxy_to_cocoa(tuple(float(v) for v in mask.xyxy), geom)
            style = getattr(mask.style, "value", str(mask.style))
            items[str(mask.region_id)] = ((x - p, y - p, w + 2 * p, h + 2 * p), style)
        return items

    def update(self, masks: list[MaskSpec], geom: DisplayGeom | None, pad_pt: float | None = None) -> bool:
        """Returns True when the backend was touched. ``pad_pt``: override
        the pad (0 when the masks were padded before being window-clipped)."""
        if self.backend is None:
            return False
        if geom is None:
            items: dict[str, tuple[Rect, str]] = {}
        else:
            items = self.rects_for(masks, geom, pad_pt)
        if items == self._last:
            return False
        self.backend.apply(items)
        self._last = items
        return True

    def mask_count(self) -> int:
        return 0 if self.backend is None else int(self.backend.mask_count())

    def window_ids(self) -> list[int]:
        return [] if self.backend is None else list(self.backend.window_ids())

    def close(self) -> None:
        if self.backend is not None:
            try:
                self.backend.close()
            except Exception:
                pass
            self.backend = None
        self.available = False
        self._last = None


class AppKitOverlayBackend:
    """Legacy reflex backend, now drawn into the single overlay window.

    Rect identity across calls comes from ``match_rects`` so a moving box
    moves its view instead of recreating it.
    """

    def __init__(self) -> None:
        self._window = AppKitMaskWindow()
        self._placed: list[tuple[float, float, float, float]] = []
        self._keys: list[str] = []
        self._next_key = 0

    def set_rects(self, rects: list[tuple[float, float, float, float]], style: str) -> None:
        assignment = match_rects(self._placed, rects)
        keys: list[str] = []
        for oi in assignment:
            if oi is not None and 0 <= oi < len(self._keys):
                keys.append(self._keys[oi])
            else:
                self._next_key += 1
                keys.append(f"legacy-{self._next_key}")
        self._window.apply({key: (rect, style) for key, rect in zip(keys, rects)})
        self._placed = list(rects)
        self._keys = keys

    def set_status(self, text: str) -> None:
        self._window.set_status(text)

    def mask_count(self) -> int:
        return self._window.mask_count()

    def window_ids(self) -> list[int]:
        return self._window.window_ids()

    def pump(self) -> None:
        pump_appkit_events()

    def close(self) -> None:
        self._window.close()
        self._placed = []
        self._keys = []


def _geom_from_state(state: ReflexState) -> DisplayGeom | None:
    frame = state.frame
    if frame is None:
        return None
    width = float(frame.width or 0)
    height = float(frame.height or 0)
    if width <= 0 or height <= 0:
        return None
    return DisplayGeom(
        left_pt=0.0,
        top_pt=0.0,
        width_pt=width,
        height_pt=height,
        width_px=int(width),
        height_px=int(height),
        scale_x=1.0,
        scale_y=1.0,
        cocoa_origin_x=0.0,
        cocoa_origin_y=0.0,
        cocoa_height_pt=height,
    )


def _action_style(state: ReflexState) -> str:
    for action in state.actions:
        if action.action is ActionType.OPAQUE_MASK:
            return "opaque"
        if action.action is ActionType.FULL_WINDOW_MASK:
            return "opaque"
    return "frost"


def _mask_boxes(state: ReflexState) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    frame = state.frame
    for action in state.actions:
        if action.action not in PROTECTIVE_ACTIONS:
            continue
        if action.action is ActionType.FULL_WINDOW_MASK and frame is not None:
            boxes.append((0.0, 0.0, float(frame.width), float(frame.height)))
            continue
        if action.region is None:
            continue
        boxes.append(tuple(float(v) for v in action.region.xyxy))
    return boxes


class ScreenOverlay:
    """Place frosted windows on detected regions. Safe to no-op in tests."""

    def __init__(self, backend: OverlayBackend | None = None) -> None:
        self.backend = backend
        self.available = backend is not None
        self.error: str | None = None
        if self.backend is None:
            try:
                self.backend = AppKitOverlayBackend()
                self.available = True
            except Exception as exc:
                self.error = f"on-screen overlay unavailable: {exc}"
                self.available = False

    def sync(self, state: ReflexState, geom: DisplayGeom | None) -> None:
        if self.backend is None:
            return
        placed = geom if geom is not None else _geom_from_state(state)
        if placed is None:
            return
        pad = 8.0
        rects = []
        for box in cluster_xyxys(_mask_boxes(state), max_clusters=MAX_OVERLAY_WINDOWS):
            x, y, w, h = capture_xyxy_to_cocoa(box, placed)
            rects.append((x - pad, y - pad, w + 2 * pad, h + 2 * pad))
        self.backend.set_rects(rects, _action_style(state))
        extra = f" · {state.backend_error}" if state.backend_error else ""
        self.backend.set_status(
            f"SG {state.policy_name} · {state.mood} · {state.status}{extra}"
        )

    def mask_count(self) -> int:
        if self.backend is None:
            return 0
        counter = getattr(self.backend, "mask_count", None)
        if callable(counter):
            return int(counter())
        return 0

    def window_ids(self) -> list[int]:
        if self.backend is None:
            return []
        return self.backend.window_ids()

    def pump(self) -> None:
        if self.backend is not None:
            self.backend.pump()

    def close(self) -> None:
        if self.backend is not None:
            try:
                self.backend.close()
            except Exception:
                pass
            self.backend = None
        self.available = False
