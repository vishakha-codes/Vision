import AppKit
import SwiftUI

/// Borderless, non-activating panel that may sit over the menu bar and notch.
final class IslandPanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
    override func constrainFrameRect(_ frameRect: NSRect, to screen: NSScreen?) -> NSRect { frameRect }
}

/// The first click on the island should act immediately, not just focus the panel.
final class FirstMouseHostingView<Content: View>: NSHostingView<Content> {
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
}

/// Places the island at the top centre (hugging the notch when there is one), keeps
/// it click-through outside the island silhouette, and wires hover, click-outside,
/// Esc and mouse gaze.
@MainActor
final class IslandWindowController {
    /// The engine's screen capture excludes windows whose title starts with "Screen Guardian".
    static let windowTitle = "Screen Guardian Island"

    let model: IslandModel
    let panel: IslandPanel
    private var timer: Timer?
    private var pollInterval: TimeInterval = 0
    private var monitors: [Any] = []

    init(model: IslandModel) {
        self.model = model
        let size = IslandLayout.panelSize
        panel = IslandPanel(contentRect: NSRect(origin: .zero, size: size),
                            styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        panel.title = Self.windowTitle
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.mainMenuWindow)) + 3)
        panel.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary, .ignoresCycle]
        // Never in screenshots or recordings. SG_CAPTURABLE=1 is for recording the demo only.
        panel.sharingType = Paths.env["SG_CAPTURABLE"] == "1" ? .readOnly : .none
        panel.isMovable = false
        panel.hidesOnDeactivate = false
        panel.isReleasedWhenClosed = false
        panel.ignoresMouseEvents = true
        panel.animationBehavior = .none

        let host = FirstMouseHostingView(rootView: IslandRootView(model: model))
        host.frame = NSRect(origin: .zero, size: size)
        host.autoresizingMask = [.width, .height]
        panel.contentView = host

        relocate()
        panel.orderFrontRegardless()

        model.onExpandChange = { [weak self] e in self?.expandChanged(e) }
        NotificationCenter.default.addObserver(forName: NSApplication.didChangeScreenParametersNotification,
                                               object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.relocate() }
        }
        installMonitors()
        schedulePoll(1.0 / 30.0)
    }

    // MARK: Geometry

    static func targetScreen() -> NSScreen? { NSScreen.screens.first ?? NSScreen.main }

    static func layout(for screen: NSScreen) -> IslandLayout {
        var L = IslandLayout()
        let top = screen.safeAreaInsets.top
        L.hasNotch = top > 0
        L.notchHeight = top > 0 ? top : 24
        if let l = screen.auxiliaryTopLeftArea?.width, let r = screen.auxiliaryTopRightArea?.width {
            let w = screen.frame.width - l - r
            L.notchWidth = (w > 60 && w < screen.frame.width) ? w : 185
        }
        let mb = screen.frame.maxY - screen.visibleFrame.maxY
        L.menuBarHeight = mb > 0 ? mb : NSStatusBar.system.thickness
        return L
    }

    func relocate() {
        guard let screen = Self.targetScreen() else { return }
        let L = Self.layout(for: screen)
        if model.layout != L { model.layout = L }
        let size = IslandLayout.panelSize
        let sf = screen.frame
        panel.setFrame(NSRect(x: sf.midX - size.width / 2, y: sf.maxY - size.height,
                              width: size.width, height: size.height), display: true)
    }

    /// The island silhouette in screen coordinates.
    private func islandRect() -> NSRect {
        let L = model.layout
        let s = model.collapsedSize
        let pf = panel.frame
        return NSRect(x: pf.midX - s.width / 2, y: pf.maxY - L.topInset - s.height, width: s.width, height: s.height)
    }

    // MARK: Polling (hover, click-through, gaze)

    private func schedulePoll(_ interval: TimeInterval) {
        guard interval != pollInterval else { return }
        pollInterval = interval
        timer?.invalidate()
        let t = Timer(timeInterval: interval, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.poll() }
        }
        t.tolerance = interval * 0.2
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    private func poll() {
        let mouse = NSEvent.mouseLocation
        let rect = islandRect()
        let slack: CGFloat = model.expanded ? 0 : 4
        let inside = rect.insetBy(dx: -slack, dy: -slack).contains(mouse)

        if panel.ignoresMouseEvents == inside { panel.ignoresMouseEvents = !inside }
        let hover = inside && !model.expanded
        if model.hovering != hover { model.hovering = hover }

        if model.expanded {
            let L = model.layout
            let cx = rect.minX + (L.hasNotch ? 14 : 0) + 20 + 98
            let cy = rect.maxY - (L.headerHeight + 6 + 88)
            let dx = max(-1, min(1, (mouse.x - cx) / 320))
            let dy = max(-1, min(1, -(mouse.y - cy) / 240))
            model.gaze.point = CGPoint(x: dx, y: dy)
        } else if model.gaze.point != nil {
            model.gaze.point = nil
        }

        // Cheap when the pointer is far away.
        let near = rect.insetBy(dx: -160, dy: -120).contains(mouse)
        schedulePoll(model.expanded || near ? 1.0 / 30.0 : 1.0 / 8.0)
    }

    // MARK: Events

    private func installMonitors() {
        if let m = NSEvent.addLocalMonitorForEvents(matching: .leftMouseDown, handler: { [weak self] event in
            guard let self else { return event }
            let inPanel = event.window === self.panel
            let consumed = MainActor.assumeIsolated { () -> Bool in
                guard inPanel && !self.model.expanded else { return false }
                self.model.setExpanded(true)
                return true
            }
            return consumed ? nil : event
        }) { monitors.append(m) }

        if let m = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown], handler: { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, self.model.expanded else { return }
                if !self.islandRect().contains(NSEvent.mouseLocation) { self.model.setExpanded(false) }
            }
        }) { monitors.append(m) }

        if let m = NSEvent.addLocalMonitorForEvents(matching: .keyDown, handler: { [weak self] event in
            guard let self else { return event }
            nonisolated(unsafe) let e = event
            let consumed = MainActor.assumeIsolated { () -> Bool in
                self.model.expanded && self.model.handleKey(e)
            }
            return consumed ? nil : event
        }) { monitors.append(m) }
    }

    private func expandChanged(_ e: Bool) {
        if e {
            panel.ignoresMouseEvents = false
            panel.makeKeyAndOrderFront(nil)
        } else {
            panel.orderFrontRegardless()
        }
        poll()
    }
}
