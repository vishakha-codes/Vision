import AppKit
import SwiftUI

/// A click-through overlay panel. It never becomes key, so it never steals focus or clicks.
final class DrawingPanel: NSPanel {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}

/// One drawing panel per screen, for explain-my-screen marks.
///
/// - Level `.statusBar` (25): above apps, below the island (main menu + 3) and well below the
///   frost windows (`NSPopUpMenuWindowLevel + 1`), so a mask always paints over a drawing.
/// - `ignoresMouseEvents`: the drawings never take a click. The agent draws; it never clicks.
/// - `sharingType = .none` and a "Screen Guardian…" title: drawings never appear in screen
///   capture, so they can't feed back into the vision engine or the next snapshot.
@MainActor
final class DrawingOverlayController {
    static let title = "Screen Guardian Drawing"

    let store: DrawingStore
    private var panels: [DrawingPanel] = []
    private var screensKey = ""
    private var observer: NSObjectProtocol?

    init(store: DrawingStore) {
        self.store = store
        observer = NotificationCenter.default.addObserver(forName: NSApplication.didChangeScreenParametersNotification,
                                                          object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.screensKey = ""
                self?.store.clear()   // marks were measured on the old layout
                self?.ensurePanels()
            }
        }
    }

    /// Builds (or rebuilds after a display change) one panel per screen.
    func ensurePanels() {
        let key = NSScreen.screens.map { "\($0.frame)" }.joined(separator: "|")
        guard key != screensKey else { panels.forEach { $0.orderFrontRegardless() }; return }
        screensKey = key
        panels.forEach { $0.close() }
        panels.removeAll()
        for screen in NSScreen.screens {
            let id = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID ?? CGMainDisplayID()
            let origin = CGDisplayBounds(id).origin   // Quartz global, y down
            let panel = DrawingPanel(contentRect: screen.frame, styleMask: [.borderless, .nonactivatingPanel],
                                     backing: .buffered, defer: false)
            panel.title = Self.title
            panel.isReleasedWhenClosed = false
            panel.isOpaque = false
            panel.backgroundColor = .clear
            panel.hasShadow = false
            panel.ignoresMouseEvents = true
            panel.hidesOnDeactivate = false
            panel.level = .statusBar
            panel.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary, .ignoresCycle]
            panel.sharingType = Paths.env["SG_CAPTURABLE"] == "1" ? .readOnly : .none
            panel.animationBehavior = .none
            let host = NSHostingView(rootView: DrawingCanvasView(store: store, origin: origin))
            host.frame = NSRect(origin: .zero, size: screen.frame.size)
            host.autoresizingMask = [.width, .height]
            panel.contentView = host
            panel.setFrame(screen.frame, display: false)
            panel.orderFrontRegardless()
            panels.append(panel)
        }
    }

    func close() {
        panels.forEach { $0.close() }
        panels.removeAll()
        screensKey = ""
        if let observer { NotificationCenter.default.removeObserver(observer) }
    }
}
