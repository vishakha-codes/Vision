import AppKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var model: IslandModel?
    var controller: IslandWindowController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let m = IslandModel()
        model = m
        controller = IslandWindowController(model: m)
        m.start()
        if CommandLine.arguments.contains("--selftest") { SelfTest.run(m) }
    }

    func applicationWillTerminate(_ notification: Notification) {
        model?.stop()
    }
}

let args = CommandLine.arguments

if args.contains("--help") || args.contains("-h") {
    print("""
    GuardianIsland: the Screen Guardian companion island.
      GuardianIsland                  run (connects to the engine socket)
      GuardianIsland --snapshot DIR   render PNG snapshots offscreen, then exit
    Env: SG_ISLAND_SOCK (socket path), SG_ISLAND_HOME (support dir override)
    """)
    exit(0)
}

if let i = args.firstIndex(of: "--snapshot") {
    let dir = i + 1 < args.count ? args[i + 1] : "snapshots"
    MainActor.assumeIsolated {
        _ = NSApplication.shared
        Snapshots.run(into: URL(fileURLWithPath: dir, isDirectory: true))
    }
    exit(0)
}

MainActor.assumeIsolated {
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    let delegate = AppDelegate()
    app.delegate = delegate
    signal(SIGTERM, SIG_DFL)
    app.run()
    _ = delegate
}
