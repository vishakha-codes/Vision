"""Always-on-top Gomi island. AppKit window on macOS; memory backend in tests."""

from __future__ import annotations

import sys
import time
from typing import Protocol

from edge.gomi.catalog import DEFAULT_SKIN, SKIN_IDS
from edge.gomi.geometry import (
    COMPACT_H,
    COMPACT_W,
    EXPANDED_H,
    EXPANDED_W,
    Hit,
    IslandGeometry,
    hit_test,
)
from edge.gomi.morph import Morph
from edge.gomi.prefs import load_prefs, save_prefs
from edge.gomi.render import catalog, paint_ns
from edge.gomi.states import GomiBrain, IslandSnapshot, event_from_reflex

WINDOW_TITLE = "Gomi"
SPRING_OMEGA = 2.0 * 3.141592653589793 / 0.28
SPRING_ZETA = 0.82


class IslandBackend(Protocol):
    def apply_snapshot(self, snap: IslandSnapshot) -> None: ...

    def set_expand_amount(self, amount: float) -> None: ...

    def window_ids(self) -> list[int]: ...

    def pump(self) -> None: ...

    def close(self) -> None: ...


class MemoryIsland:
    """Test double. Records snapshots; no display."""

    def __init__(self) -> None:
        self.snapshots: list[IslandSnapshot] = []
        self.expand = 0.0
        self.closed = False

    def apply_snapshot(self, snap: IslandSnapshot) -> None:
        self.snapshots.append(snap)

    def set_expand_amount(self, amount: float) -> None:
        self.expand = float(amount)

    def window_ids(self) -> list[int]:
        return []

    def front(self) -> None:
        return None

    def pump(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _lerp(a: float, b: float, t: float) -> float:
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return a + (b - a) * t


def _label(text: str, x: float, y: float, size: float, rgba: tuple[float, float, float, float]) -> None:
    from AppKit import NSColor, NSFont, NSFontAttributeName, NSForegroundColorAttributeName
    from Foundation import NSAttributedString

    font = NSFont.systemFontOfSize_(size)
    color = NSColor.colorWithCalibratedRed_green_blue_alpha_(*rgba)
    attrs = {NSFontAttributeName: font, NSForegroundColorAttributeName: color}
    ns = NSAttributedString.alloc().initWithString_attributes_(text[:72], attrs)
    ns.drawAtPoint_((x, y))


def step_spring(value: float, vel: float, target: float, dt: float) -> tuple[float, float]:
    """Underdamped spring (response ~0.28s, damping ~0.82). dt-based so extra pumps are safe."""
    dt = max(0.0, min(0.05, dt))
    acc = SPRING_OMEGA * SPRING_OMEGA * (target - value) - 2.0 * SPRING_ZETA * SPRING_OMEGA * vel
    vel = vel + acc * dt
    value = value + vel * dt
    if abs(target - value) < 0.002 and abs(vel) < 0.02:
        return target, 0.0
    return value, vel


def _paint_ops_ns(ops) -> None:
    from AppKit import NSAffineTransform, NSBezierPath, NSColor, NSGraphicsContext, NSMakeRect

    ctx = NSGraphicsContext.currentContext()
    for op in ops:
        if isinstance(op, EllipseOp):
            rect = NSMakeRect(op.cx - op.rx, op.cy - op.ry, max(0.8, op.rx * 2.0), max(0.8, op.ry * 2.0))
            path = NSBezierPath.bezierPathWithOvalInRect_(rect)
            r, g, b, a = op.fill.ns()
            if ctx is not None and abs(op.angle) > 0.05:
                ctx.saveGraphicsState()
                xf = NSAffineTransform.transform()
                xf.translateXBy_yBy_(op.cx, op.cy)
                xf.rotateByDegrees_(op.angle)
                xf.translateXBy_yBy_(-op.cx, -op.cy)
                xf.concat()
                NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a).setFill()
                path.fill()
                ctx.restoreGraphicsState()
            else:
                NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a).setFill()
                path.fill()
        elif isinstance(op, PolyOp):
            path = NSBezierPath.bezierPath()
            first = True
            for x, y in op.points:
                if first:
                    path.moveToPoint_((x, y))
                    first = False
                else:
                    path.lineToPoint_((x, y))
            path.closePath()
            r, g, b, a = op.fill.ns()
            NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a).setFill()
            path.fill()
            if op.stroke is not None and op.stroke_w > 0:
                sr, sg, sb, sa = op.stroke.ns()
                NSColor.colorWithCalibratedRed_green_blue_alpha_(sr, sg, sb, sa).setStroke()
                path.setLineWidth_(op.stroke_w)
                path.stroke()
        elif isinstance(op, LineOp):
            path = NSBezierPath.bezierPath()
            path.moveToPoint_((op.x1, op.y1))
            path.lineToPoint_((op.x2, op.y2))
            path.setLineWidth_(op.w)
            path.setLineCapStyle_(1)
            r, g, b, a = op.color.ns()
            NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a).setStroke()
            path.stroke()


def _draw_label(text: str, x: float, y: float, size: float, rgba: tuple[float, float, float, float]) -> None:
    from AppKit import NSColor, NSFont, NSForegroundColorAttributeName, NSFontAttributeName
    from Foundation import NSAttributedString

    font = NSFont.systemFontOfSize_(size)
    color = NSColor.colorWithCalibratedRed_green_blue_alpha_(*rgba)
    attrs = {NSFontAttributeName: font, NSForegroundColorAttributeName: color}
    ns = NSAttributedString.alloc().initWithString_attributes_(text[:72], attrs)
    ns.drawAtPoint_((x, y))


def _round_rect(x: float, y: float, w: float, h: float, r: float):
    from AppKit import NSBezierPath, NSMakeRect

    return NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
        NSMakeRect(x, y, max(2.0, w), max(2.0, h)),
        min(r, h * 0.5, w * 0.5),
        min(r, h * 0.5, w * 0.5),
    )


def _paint_hit(hit: Hit, selected: bool, fade: float) -> None:
    from AppKit import NSColor

    path = _round_rect(hit.x, hit.y, hit.w, hit.h, 8.0)
    if selected:
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.93, 0.78, 0.62, 0.95 * fade).setFill()
        path.fill()
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.18, 0.14, 0.12, fade).setStroke()
    else:
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.08 * fade).setFill()
        path.fill()
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.28 * fade).setStroke()
    path.setLineWidth_(1.0)
    path.stroke()
    ink = (0.16, 0.12, 0.10, fade) if selected else (0.90, 0.92, 0.90, fade)
    _label(hit.value, hit.x + 8.0, hit.y + 5.0, 11.0, ink)


def paint_island_view(view, runtime: GomiRuntime) -> None:
    from AppKit import NSBezierPath, NSColor, NSGraphicsContext, NSMakeRect

    bounds = view.bounds()
    w = float(bounds.size.width)
    h = float(bounds.size.height)
    lay = runtime.current_layout()
    snap = runtime.brain.last_snap
    radius = min(lay.radius, h * 0.5)
    ctx = NSGraphicsContext.currentContext()
    if ctx is not None:
        ctx.saveGraphicsState()
    shell = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
        NSMakeRect(1.0, 1.0, max(2.0, w - 2.0), max(2.0, h - 2.0)),
        radius,
        radius,
    )
    shell.addClip()
    NSColor.colorWithCalibratedRed_green_blue_alpha_(0.07, 0.08, 0.10, 0.94).setFill()
    shell.fill()
    # Faint top sheen so the capsule is not a flat slab.
    sheen = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
        NSMakeRect(8.0, 3.0, max(4.0, w - 16.0), max(3.0, min(10.0, h * 0.22))),
        6.0,
        6.0,
    )
    NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.06).setFill()
    sheen.fill()

    fig = runtime.figure
    if fig is not None:
        paint_ns(fig, lay.char_x, lay.char_y, lay.char_d)

    fade = max(0.0, (lay.expand - 0.35) / 0.65)
    text_x = lay.char_x + lay.char_d + 12.0
    _label(snap.headline, text_x, 10.0, 13.0, (0.94, 0.95, 0.93, 1.0))
    _label(snap.scene_line, text_x, 28.0, 11.0, (0.72, 0.78, 0.76, 1.0))
    if fade > 0.04:
        _label(f"backend  {snap.backend}", text_x, 48.0, 11.0, (0.78, 0.82, 0.80, fade))
        for hit in lay.hits:
            if hit.kind == "character":
                selected = hit.value == runtime.character_id
                _paint_hit(hit, selected, fade)
                try:
                    still = catalog()[hit.value]["idle"]
                    paint_ns(still, hit.x + 6.0, hit.y + 1.0, hit.h - 14.0)
                except Exception:
                    pass
                _label(hit.value, hit.x + 4.0, hit.y + hit.h - 12.0, 8.0, (0.95, 0.94, 0.9, fade))
            else:
                current = runtime.policy if hit.kind == "policy" else runtime.detect
                _paint_hit(hit, hit.value == current, fade)
        if runtime.note:
            _label(runtime.note, text_x, 168.0, 10.0, (0.95, 0.72, 0.55, fade))

    NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.22).setStroke()
    shell.setLineWidth_(1.25)
    shell.stroke()
    if ctx is not None:
        ctx.restoreGraphicsState()


_VIEW_CLS = None
_TICK_CLS = None
_WINDOW_CLS = None


def _appkit_classes():
    """Build PyObjC subclasses once, after AppKit is imported."""
    global _VIEW_CLS, _TICK_CLS, _WINDOW_CLS
    if _VIEW_CLS is not None:
        return _VIEW_CLS, _TICK_CLS, _WINDOW_CLS

    import objc
    from AppKit import (
        NSTrackingActiveAlways,
        NSTrackingInVisibleRect,
        NSTrackingMouseEnteredAndExited,
        NSView,
        NSWindow,
    )
    from Foundation import NSObject

    class GomiIslandWindow(NSWindow):
        def canBecomeKeyWindow(self):
            return True

        def canBecomeMainWindow(self):
            return False

    class GomiIslandView(NSView):
        def initWithFrame_(self, frame):
            self = objc.super(GomiIslandView, self).initWithFrame_(frame)
            if self is None:
                return None
            self.runtime = None
            self._drag_origin = None
            self._mouse_origin = None
            self._dragged = False
            return self

        def isOpaque(self):
            return False

        def isFlipped(self):
            return True

        def acceptsFirstMouse_(self, _event):
            return True

        def acceptsFirstResponder(self):
            return True

        def keyDown_(self, event):
            runtime = getattr(self, "runtime", None)
            chars = ""
            try:
                chars = str(event.charactersIgnoringModifiers() or "")
            except Exception:
                chars = ""
            if runtime is not None and chars == "\x1b":
                runtime.collapse()
                return
            objc.super(GomiIslandView, self).keyDown_(event)

        def drawRect_(self, _rect):
            runtime = getattr(self, "runtime", None)
            if runtime is not None:
                paint_island_view(self, runtime)

        def updateTrackingAreas(self):
            objc.super(GomiIslandView, self).updateTrackingAreas()
            from AppKit import NSTrackingArea

            for area in list(self.trackingAreas() or []):
                self.removeTrackingArea_(area)
            opts = NSTrackingMouseEnteredAndExited | NSTrackingActiveAlways | NSTrackingInVisibleRect
            area = NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
                self.bounds(), opts, self, None
            )
            self.addTrackingArea_(area)

        def mouseEntered_(self, _event):
            runtime = getattr(self, "runtime", None)
            if runtime is not None:
                runtime.set_hover(True)

        def mouseExited_(self, _event):
            runtime = getattr(self, "runtime", None)
            if runtime is not None:
                runtime.set_hover(False)
            # Hover must not resize. Resize-under-cursor was the expand/shrink loop.

        def mouseDown_(self, event):
            from AppKit import NSEvent

            win = self.window()
            if win is None:
                return
            try:
                win.makeKeyAndOrderFront_(None)
            except Exception:
                pass
            self._drag_origin = (float(win.frame().origin.x), float(win.frame().origin.y))
            loc = NSEvent.mouseLocation()
            self._mouse_origin = (float(loc.x), float(loc.y))
            self._dragged = False

        def mouseDragged_(self, _event):
            from AppKit import NSEvent

            if self._drag_origin is None or self._mouse_origin is None:
                return
            loc = NSEvent.mouseLocation()
            dx = float(loc.x) - self._mouse_origin[0]
            dy = float(loc.y) - self._mouse_origin[1]
            if abs(dx) + abs(dy) > 4.0:
                self._dragged = True
            win = self.window()
            if win is not None:
                win.setFrameOrigin_((self._drag_origin[0] + dx, self._drag_origin[1] + dy))

        def mouseUp_(self, event):
            runtime = getattr(self, "runtime", None)
            if runtime is not None and not self._dragged:
                try:
                    loc = self.convertPoint_fromView_(event.locationInWindow(), None)
                    hit = runtime.hit_at(float(loc.x), float(loc.y))
                except Exception:
                    hit = None
                if hit is not None:
                    runtime.activate(hit)
                else:
                    runtime.toggle_pin()
            self._drag_origin = None
            self._mouse_origin = None
            self._dragged = False

    class GomiTick(NSObject):
        def initWithRuntime_(self, runtime):
            self = objc.super(GomiTick, self).init()
            if self is None:
                return None
            self.runtime = runtime
            return self

        def tick_(self, _timer):
            runtime = getattr(self, "runtime", None)
            if runtime is not None:
                runtime.pump()

    _VIEW_CLS = GomiIslandView
    _TICK_CLS = GomiTick
    _WINDOW_CLS = GomiIslandWindow
    return _VIEW_CLS, _TICK_CLS, _WINDOW_CLS


class AppKitIsland:
    """Borderless always-on-top capsule. Only this window is interactive."""

    def __init__(self) -> None:
        from AppKit import (
            NSApplication,
            NSApplicationActivationPolicyAccessory,
            NSBackingStoreBuffered,
            NSColor,
            NSMakeRect,
            NSScreen,
            NSStatusWindowLevel,
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorIgnoresCycle,
            NSWindowCollectionBehaviorStationary,
            NSWindowSharingNone,
            NSWindowStyleMaskBorderless,
        )
        from Foundation import NSRunLoop, NSRunLoopCommonModes, NSTimer

        self.snap = IslandSnapshot.empty()
        self.expand = 0.0
        self.runtime: GomiRuntime | None = None
        self._timer = None
        self._tick = None
        self._closed = False

        app = NSApplication.sharedApplication()
        try:
            app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        except Exception:
            pass

        view_cls, tick_cls, window_cls = _appkit_classes()
        x, y = self._default_origin()
        rect = NSMakeRect(x, y, COMPACT_W, COMPACT_H)
        window = window_cls.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False,
        )
        window.setTitle_(WINDOW_TITLE)
        window.setOpaque_(False)
        window.setBackgroundColor_(NSColor.clearColor())
        window.setLevel_(NSStatusWindowLevel + 2)
        window.setIgnoresMouseEvents_(False)
        window.setHidesOnDeactivate_(False)
        window.setHasShadow_(True)
        window.setAcceptsMouseMovedEvents_(True)
        try:
            window.setSharingType_(NSWindowSharingNone)
        except Exception:
            pass
        window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorIgnoresCycle
            | 256  # NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        view = view_cls.alloc().initWithFrame_(window.contentView().bounds())
        view.setAutoresizingMask_(18)  # width + height
        window.setContentView_(view)
        window.orderFrontRegardless()
        self.window = window
        self.view = view
        self._tick = tick_cls.alloc().initWithRuntime_(None)
        self._timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / 60.0, self._tick, "tick:", None, True
        )
        NSRunLoop.currentRunLoop().addTimer_forMode_(self._timer, NSRunLoopCommonModes)

    def bind(self, runtime: GomiRuntime) -> None:
        self.runtime = runtime
        self.view.runtime = runtime
        if self._tick is not None:
            self._tick.runtime = runtime

    @staticmethod
    def _default_origin() -> tuple[float, float]:
        from AppKit import NSScreen

        screen = NSScreen.mainScreen()
        if screen is None:
            return (80.0, 640.0)
        vis = screen.visibleFrame()
        x = float(vis.origin.x + (vis.size.width - COMPACT_W) * 0.5)
        y = float(vis.origin.y + vis.size.height - COMPACT_H - 8.0)
        return (x, y)

    def apply_snapshot(self, snap: IslandSnapshot) -> None:
        self.snap = snap
        try:
            self.view.setNeedsDisplay_(True)
        except Exception:
            pass

    def set_expand_amount(self, amount: float) -> None:
        self.expand = float(amount)
        self._resize()

    def _resize(self) -> None:
        from AppKit import NSMakeRect

        runtime = self.runtime
        if runtime is not None and getattr(runtime, "geom", None) is not None:
            lay = runtime.geom.layout()
            w, h = lay.width, lay.height
        else:
            w = _lerp(COMPACT_W, EXPANDED_W, self.expand)
            h = _lerp(COMPACT_H, EXPANDED_H, self.expand)
        frame = self.window.frame()
        cur_w = float(frame.size.width)
        cur_h = float(frame.size.height)
        if abs(cur_w - w) < 0.6 and abs(cur_h - h) < 0.6:
            return
        top = float(frame.origin.y) + cur_h
        x = float(frame.origin.x)
        try:
            self.window.setFrame_display_(NSMakeRect(x, top - h, w, h), True)
        except Exception:
            pass

    def front(self) -> None:
        try:
            self.window.orderFrontRegardless()
        except Exception:
            pass

    def window_ids(self) -> list[int]:
        try:
            return [int(self.window.windowNumber())]
        except Exception:
            return []

    def pump(self) -> None:
        try:
            from AppKit import NSApplication
            from Foundation import NSDate, NSDefaultRunLoopMode

            app = NSApplication.sharedApplication()
            event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                2**64 - 1,
                NSDate.dateWithTimeIntervalSinceNow_(0.0),
                NSDefaultRunLoopMode,
                True,
            )
            if event is not None:
                app.sendEvent_(event)
            if self.view is not None:
                self.view.setNeedsDisplay_(True)
        except Exception:
            return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._timer is not None:
            try:
                self._timer.invalidate()
            except Exception:
                pass
            self._timer = None
        try:
            self.window.orderOut_(None)
            self.window.close()
        except Exception:
            pass


class GomiRuntime:
    """State machine + island backend. Safe to construct with MemoryIsland."""

    def __init__(
        self,
        backend: IslandBackend,
        *,
        character: str | None = None,
        policy: str = "balanced",
        detect: str = "synthetic",
        prefs_path=None,
    ) -> None:
        self.backend = backend
        self.brain = GomiBrain()
        self._t0 = time.monotonic()
        self.expand = 0.0
        self.expand_vel = 0.0
        self.expand_target = 0.0
        self.pinned = False
        self.hover = False
        self._last_pump = self._t0
        self._prefs_path = prefs_path
        prefs = load_prefs(prefs_path)
        self.character_id = character if character in SKIN_IDS else prefs.get("character_id", DEFAULT_SKIN)
        if self.character_id not in SKIN_IDS:
            self.character_id = DEFAULT_SKIN
        self.policy = policy if policy in {"strict", "balanced", "relaxed"} else "balanced"
        self.detect = detect if detect in {"nsfw", "synthetic"} else "synthetic"
        self.note = ""
        self.on_setting = None
        self.geom = IslandGeometry()
        self.morph = Morph(catalog(), self.character_id, "clear")
        self.figure = self.morph.pose_at(0.0)
        if hasattr(backend, "bind"):
            backend.bind(self)

    def on_state(self, state) -> None:
        event = event_from_reflex(state)
        if event.policy_name in {"strict", "balanced", "relaxed"}:
            self.policy = event.policy_name
        snap = self.brain.apply(event, time.monotonic() - self._t0)
        self.backend.apply_snapshot(snap)

    def set_hover(self, hover: bool) -> None:
        self.hover = bool(hover)
        # Hover is visual-only. Expand is click (pin) only — state
        # flicker and mouse-enter from a resize must not toggle size.

    def toggle_pin(self) -> None:
        self.pinned = not self.pinned
        self._sync_expand()

    def collapse(self) -> None:
        if not self.pinned:
            return
        self.pinned = False
        self._sync_expand()

    def _sync_expand(self) -> None:
        self.expand_target = 1.0 if self.pinned else 0.0
        self.geom.set_expanded(self.pinned)

    def current_layout(self):
        return self.geom.layout()

    def hit_at(self, x: float, y: float):
        lay = self.current_layout()
        if lay.expand < 0.62:
            return None
        return hit_test(lay, x, y)

    def activate(self, hit: Hit) -> None:
        if hit.kind == "character" and hit.value in SKIN_IDS:
            self.character_id = hit.value
            self.note = ""
        elif hit.kind == "policy":
            self.policy = hit.value
        elif hit.kind == "detect":
            self.detect = hit.value
        else:
            return
        save_prefs(
            {"character_id": self.character_id, "policy": self.policy, "detect": self.detect},
            self._prefs_path,
        )
        if callable(self.on_setting):
            try:
                note = self.on_setting(hit.kind, hit.value)
            except Exception as exc:
                note = str(exc)
            self.note = "" if not note else str(note)
        now = time.monotonic() - self._t0
        self.morph.retarget(self.brain.state.value, now, skin=self.character_id)
        self.figure = self.morph.pose_at(now)

    def pump(self) -> None:
        wall = time.monotonic()
        dt = wall - self._last_pump
        self._last_pump = wall
        now = wall - self._t0
        snap = self.brain.tick(now)
        self.morph.retarget(snap.state.value, now, skin=self.character_id)
        self.figure = self.morph.pose_at(now)
        lay = self.geom.step(dt)
        prev = self.expand
        self.expand = lay.expand
        self.backend.apply_snapshot(snap)
        if abs(self.expand - prev) > 1e-4 or abs(self.expand - self.expand_target) > 1e-3:
            self.backend.set_expand_amount(self.expand)
        self.backend.pump()

    def front(self) -> None:
        fn = getattr(self.backend, "front", None)
        if callable(fn):
            fn()

    def window_ids(self) -> list[int]:
        return self.backend.window_ids()

    def stop(self) -> None:
        self.backend.close()


def open_island(
    *,
    headless: bool = False,
    character: str | None = None,
    policy: str = "balanced",
    detect: str = "synthetic",
) -> GomiRuntime:
    if headless:
        return GomiRuntime(MemoryIsland(), character=character, policy=policy, detect=detect)
    try:
        return GomiRuntime(AppKitIsland(), character=character, policy=policy, detect=detect)
    except Exception as exc:
        print(f"Gomi island unavailable ({exc}); continuing without a window.", file=sys.stderr)
        return GomiRuntime(MemoryIsland(), character=character, policy=policy, detect=detect)
