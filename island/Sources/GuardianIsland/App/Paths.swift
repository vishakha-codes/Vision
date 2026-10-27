import Foundation

/// File locations shared with the engine (island-protocol.md).
enum Paths {
    static let env = ProcessInfo.processInfo.environment

    /// `SG_ISLAND_HOME` relocates everything (tests and demos); the engine reads the default.
    static var supportDir: URL {
        if let h = env["SG_ISLAND_HOME"], !h.isEmpty { return URL(fileURLWithPath: h, isDirectory: true) }
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        return base.appendingPathComponent("ScreenGuardian", isDirectory: true)
    }

    static var socket: String {
        if let s = env["SG_ISLAND_SOCK"], !s.isEmpty { return s }
        return supportDir.appendingPathComponent("island.sock").path
    }

    static var prefs: URL { supportDir.appendingPathComponent("island.json") }
    static var guardian: URL { supportDir.appendingPathComponent("guardian.json") }
    static var lockState: URL { supportDir.appendingPathComponent("island-lock.json") }

    static var keychainService: String { env["SG_ISLAND_KEYCHAIN_SERVICE"] ?? "ScreenGuardian" }

    static func ensureSupportDir() {
        try? FileManager.default.createDirectory(at: supportDir, withIntermediateDirectories: true,
                                                 attributes: [.posixPermissions: 0o700])
    }

    /// Writes `data` atomically with mode 0600.
    @discardableResult
    static func writePrivate(_ data: Data, to url: URL) -> Bool {
        ensureSupportDir()
        do {
            try data.write(to: url, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
            return true
        } catch {
            NSLog("Screen Guardian island: could not write \(url.lastPathComponent): \(error)")
            return false
        }
    }
}
