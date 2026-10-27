import AppKit
import SwiftUI

enum EngineStatus: String {
    case protecting = "PROTECTING"
    case notProtecting = "NOT_PROTECTING"
    case paused = "PAUSED"
    case off = "OFF"
}

enum ProtectionMode: String, CaseIterable, Identifiable {
    case aggressive, medium, off
    var id: String { rawValue }

    var title: String {
        switch self {
        case .aggressive: return "Aggressive"
        case .medium: return "Medium"
        case .off: return "Off"
        }
    }

    var audience: String {
        switch self {
        case .aggressive: return "Kids"
        case .medium: return "Teens"
        case .off: return "Adults"
        }
    }

    var blurb: String {
        switch self {
        case .aggressive: return "Covers whole pictures."
        case .medium: return "Covers only body regions, so health and sex-ed stay readable."
        case .off: return "Nothing is covered."
        }
    }

    var icon: String {
        switch self {
        case .aggressive: return "shield.fill"
        case .medium: return "shield.lefthalf.filled"
        case .off: return "shield.slash"
        }
    }
}

enum PanelTab: String, CaseIterable, Identifiable {
    case protection, character, settings, chat
    var id: String { rawValue }

    var title: String {
        switch self {
        case .protection: return "Protection"
        case .character: return "Character"
        case .settings: return "Settings"
        case .chat: return "Chat"
        }
    }

    var icon: String {
        switch self {
        case .protection: return "shield.lefthalf.filled"
        case .character: return "face.smiling"
        case .settings: return "gearshape.fill"
        case .chat: return "bubble.left.and.bubble.right.fill"
        }
    }
}

enum StatusTone { case green, amber, red, grey }

enum GuardedAction: Equatable {
    case setMode(ProtectionMode)
    case pause
    case quit
    case changePassword

    var verb: String {
        switch self {
        case .setMode(let m): return "switch to \(m.title)"
        case .pause: return "pause protection"
        case .quit: return "quit Screen Guardian"
        case .changePassword: return "change the PIN"
        }
    }
}

enum LockFlow: Equatable {
    case setup(confirming: Bool)
    case unlock(GuardedAction)
    case change(confirming: Bool)
}

/// Island geometry for the current screen. All sizes in points.
struct IslandLayout: Equatable {
    var hasNotch = true
    var notchWidth: CGFloat = 185
    var notchHeight: CGFloat = 32
    var menuBarHeight: CGFloat = 24

    static let panelSize = CGSize(width: 820, height: 460)
    static let expandedWidth: CGFloat = 680
    static let bodyHeight: CGFloat = 292

    /// Gap between the screen top and the island (0 when it hugs the notch).
    var topInset: CGFloat { hasNotch ? 0 : 5 }
    var headerHeight: CGFloat { hasNotch ? max(notchHeight, 30) : 36 }
    var earRadius: CGFloat { hasNotch ? 7 : 0 }

    /// Wing width beside the notch. `wide` is the "live activity" look used when
    /// something worth reading is happening (covering, checking, not protecting).
    func wing(wide: Bool) -> CGFloat { hasNotch ? (wide ? 118 : 58) : 0 }

    func collapsed(wide: Bool = false) -> CGSize {
        hasNotch ? CGSize(width: notchWidth + 2 * wing(wide: wide) + 2 * earRadius, height: notchHeight)
                 : CGSize(width: wide ? 206 : 176, height: 30)
    }

    func hover(wide: Bool = false) -> CGSize {
        let c = collapsed(wide: wide)
        return hasNotch ? CGSize(width: c.width + 16, height: c.height + 4)
                        : CGSize(width: c.width + 10, height: c.height + 3)
    }

    var expanded: CGSize { CGSize(width: Self.expandedWidth, height: headerHeight + Self.bodyHeight) }

    func size(expanded e: Bool, hovering: Bool, wide: Bool = false) -> CGSize {
        e ? expanded : (hovering ? hover(wide: wide) : collapsed(wide: wide))
    }
}

@MainActor
final class IslandModel: ObservableObject {
    // Engine
    @Published var connected = false
    @Published var status: EngineStatus = .notProtecting
    @Published var engineMood: Mood = .watching
    @Published var mode: ProtectionMode = .aggressive
    @Published var masks = 0
    @Published var pending = 0
    @Published var backend = ""
    @Published var fps = 0.0
    @Published var reason = ""
    /// Engine warning (protocol `state.warning`), e.g. code "busy": the screen
    /// has more to cover than a sane share of it, so the engine warns instead
    /// of blurring everything.
    @Published var warningCode: String?
    @Published var warningText: String?
    @Published var engineError: String?
    @Published var engineVersion: Int?
    @Published var opencv: String?
    @Published var lastError: String?
    @Published var requestedMode: ProtectionMode?

    // UI
    @Published var expanded = false
    @Published var hovering = false
    @Published var tab: PanelTab = .protection
    @Published var prefs: Prefs
    @Published var layout = IslandLayout()
    @Published var systemReduceMotion = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
    @Published var poke = 0
    @Published var now = Date()
    /// Snapshot-only: 0 = collapsed, 1 = expanded, values in between (and slight
    /// overshoot) render the spring mid-flight for motion previews.
    @Published var previewProgress: CGFloat?

    // Lock
    @Published var lockFlow: LockFlow?
    @Published var entry = ""
    @Published var lockMessage: String?
    @Published var lockBusy = false
    @Published var shake = 0
    private var firstEntry = ""
    private var pendingAfterSetup: GuardedAction?

    let lock: GuardianLock
    lazy var chat = ChatModel(lock: lock, snapshot: snapshotMode)
    let gaze = GazeSource()
    var snapshotMode = false
    private var client: EngineClient?
    private var token: String?
    private var timers: [Timer] = []
    private var errorClear: DispatchWorkItem?
    var onExpandChange: ((Bool) -> Void)?

    init(snapshot: Bool = false) {
        snapshotMode = snapshot
        prefs = snapshot ? Prefs() : Prefs.load()
        lock = GuardianLock(probe: !snapshot)
    }

    // MARK: Derived

    var spec: CharacterSpec { Catalog.byID(prefs.character) }
    var palette: Palette { spec.palette(prefs.variant) }
    var reduceMotion: Bool { prefs.reduceMotion || systemReduceMotion }

    var accent: Color {
        let p = palette
        let base = p.tile.luminance < 0.16 ? p.accent : p.tile
        return base.lighter(0.18).color
    }

    var mood: Mood {
        guard connected else { return .lost }
        if status == .paused || status == .off { return engineMood == .lost ? .lost : .sleep }
        return engineMood
    }

    /// Short word for the collapsed island, only when it matters.
    var liveWord: String? {
        guard connected else { return "Not protecting" }
        switch status {
        case .notProtecting: return "Not protecting"
        case .paused: return "Paused"
        case .off, .protecting: break
        }
        guard status == .protecting else { return nil }
        if warningCode != nil, engineMood != .alert { return warningCode == "busy" ? "Busy" : "Heads up" }
        switch engineMood {
        case .covering: return masks > 0 ? "Covering \(masks)" : "Covering"
        case .alert: return "Hold on"
        case .unsure, .working: return "Checking"
        default: return nil
        }
    }

    var collapsedSize: CGSize { layout.size(expanded: expanded, hovering: hovering, wide: liveWord != nil) }

    var tone: StatusTone {
        guard connected else { return .red }
        switch status {
        case .protecting: return (engineMood == .unsure || engineMood == .lost || warningCode != nil) ? .amber : .green
        case .notProtecting: return .red
        case .paused, .off: return .grey
        }
    }

    var statusTitle: String {
        guard connected else { return "Not protecting" }
        switch status {
        case .protecting: return engineMood == .unsure ? "Protecting · checking" : "Protecting"
        case .notProtecting: return "Not protecting"
        case .paused: return "Paused"
        case .off: return "Off"
        }
    }

    var statusDetail: String {
        if let e = lastError { return e }
        guard connected else { return "Waiting for the Screen Guardian engine…" }
        var parts: [String] = []
        switch status {
        case .protecting:
            parts.append(masks == 1 ? "1 picture covered" : "\(masks) pictures covered")
        case .notProtecting:
            parts.append(reasonText ?? "The screen isn't being checked")
        case .paused:
            parts.append("Paused by a grown-up")
        case .off:
            parts.append("Adult mode · nothing is covered")
        }
        if status == .protecting, pending > 0 { parts.append("\(pending) checking") }
        if !backend.isEmpty, status == .protecting { parts.append(backend) }
        if fps > 0, status == .protecting { parts.append(String(format: "%.0f fps", fps)) }
        return parts.joined(separator: " · ")
    }

    var reasonText: String? {
        switch reason {
        case "": return nil
        case "UNSAFE_PICTURE": return "Covered an unsafe picture"
        case "CAPTURE_UNAVAILABLE", "NO_CAPTURE": return "Screen capture unavailable"
        case "ANALYZER_UNAVAILABLE": return "The picture checker didn't start"
        case "PERMISSION", "NO_PERMISSION": return "Needs Screen Recording permission"
        default:
            return reason.replacingOccurrences(of: "_", with: " ").lowercased().capitalized(with: nil)
        }
    }

    var speech: String {
        let n = spec.name
        if !connected { return "I can't reach the guardian engine." }
        switch mood {
        case .idle: return "All clear!"
        case .watching: return "\(n) is keeping watch."
        case .working: return "Taking a closer look…"
        case .alert: return "Whoa, hold on!"
        case .covering: return "I covered that for you."
        case .unsure: return "Hmm, not sure about that one…"
        case .lost: return "I can't see the screen right now."
        case .sleep: return status == .off ? "Napping. Nothing is covered." : "Napping until a grown-up resumes."
        }
    }

    // MARK: Lifecycle

    func start() {
        Paths.ensureSupportDir()
        token = GuardianToken.ensure()
        let c = EngineClient(path: Paths.socket)
        c.onConnection = { [weak self] up in self?.setConnected(up) }
        c.onMessage = { [weak self] msg in self?.handle(msg) }
        client = c
        c.start()
        chat.start()

        timers.append(Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { _ = self?.client?.send(["type": "ping"]) }
        })
        timers.append(Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, self.expanded else { return }
                self.now = Date()
                if self.lock.unlockedUntil != nil && !self.lock.isUnlocked { self.lock.lockNow() }
            }
        })
        NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.accessibilityDisplayOptionsDidChangeNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.systemReduceMotion = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
            }
        }

        if !lock.hasPassword {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.9) { [weak self] in
                guard let self, !self.lock.hasPassword else { return }
                self.tab = .protection
                self.setExpanded(true)
                self.beginFlow(.setup(confirming: false))
            }
        }
    }

    func stop() {
        client?.stop()
        chat.stop()
        timers.forEach { $0.invalidate() }
    }

    private func setConnected(_ up: Bool) {
        connected = up
        if !up {
            status = .notProtecting
            requestedMode = nil
        }
    }

    func handle(_ m: [String: Any]) {
        guard let type = m["type"] as? String else { return }
        switch type {
        case "hello":
            engineVersion = m["version"] as? Int
            opencv = m["opencv"] as? String
            if let md = (m["mode"] as? String).flatMap(ProtectionMode.init(rawValue:)) { mode = md }
        case "state":
            if let s = (m["status"] as? String).flatMap(EngineStatus.init(rawValue:)) { status = s }
            if let md = (m["mood"] as? String).flatMap(Mood.init(rawValue:)) {
                if md != engineMood && (md == .alert || md == .covering) { poke += 1 }
                engineMood = md
            }
            if let md = (m["mode"] as? String).flatMap(ProtectionMode.init(rawValue:)) {
                mode = md
                if requestedMode == md { requestedMode = nil }
            }
            if let v = m["masks"] as? Int { masks = v }
            if let v = m["pending"] as? Int { pending = v }
            if let v = m["backend"] as? String { backend = v }
            if let v = m["fps"] as? Double { fps = v } else if let v = m["fps"] as? Int { fps = Double(v) }
            reason = (m["reason"] as? String) ?? ""
            if let w = m["warning"] as? [String: Any], let code = w["code"] as? String, !code.isEmpty {
                warningCode = code
                warningText = (w["text"] as? String) ?? "Lots to cover here"
            } else {
                warningCode = nil
                warningText = nil
            }
            if let e = m["error"] as? String, !e.isEmpty { engineError = e } else { engineError = nil }
        case "event":
            if (m["kind"] as? String) == "covered" {
                if let c = m["count"] as? Int { masks = c }
                poke += 1
            } else if (m["kind"] as? String) == "cleared" {
                masks = 0
            }
        case "error":
            showError(m["message"] as? String ?? "Engine error")
        default:
            break
        }
    }

    private func showError(_ text: String) {
        lastError = text.prefix(1).uppercased() + text.dropFirst()
        requestedMode = nil
        errorClear?.cancel()
        let w = DispatchWorkItem { [weak self] in self?.lastError = nil }
        errorClear = w
        DispatchQueue.main.asyncAfter(deadline: .now() + 6, execute: w)
    }

    // MARK: UI actions

    func setExpanded(_ e: Bool) {
        guard expanded != e else { return }
        expanded = e
        if !e {
            lockFlow = nil
            entry = ""
            lockMessage = nil
        } else {
            now = Date()
        }
        onExpandChange?(e)
    }

    func pokeCharacter() { poke += 1 }

    func choose(character id: String) {
        guard prefs.character != id else { pokeCharacter(); return }
        prefs.character = id
        prefs.variant = 0
        savePrefs()
        pokeCharacter()
    }

    func choose(variant: Int) {
        prefs.variant = variant
        savePrefs()
        pokeCharacter()
    }

    func toggleReduceMotion() { prefs.reduceMotion.toggle(); savePrefs() }
    func toggleMaskCount() { prefs.showMaskCount.toggle(); savePrefs() }

    private func savePrefs() { if !snapshotMode { prefs.save() } }

    func resume() {
        guard connected else { return }
        client?.send(["type": "resume"])
    }

    // MARK: Guarded actions

    func request(_ action: GuardedAction) {
        if !lock.hasPassword {
            pendingAfterSetup = action
            beginFlow(.setup(confirming: false))
        } else if lock.isUnlocked {
            perform(action)
        } else {
            beginFlow(.unlock(action))
        }
    }

    private func perform(_ action: GuardedAction) {
        switch action {
        case .setMode(let m):
            guard connected else { showError("The engine isn't running"); return }
            requestedMode = m
            sendGuarded(["type": "set_mode", "mode": m.rawValue])
        case .pause:
            sendGuarded(["type": "pause"])
        case .quit:
            sendGuarded(["type": "quit"])
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) { NSApp.terminate(nil) }
        case .changePassword:
            beginFlow(.change(confirming: false))
        }
    }

    private func sendGuarded(_ obj: [String: Any]) {
        guard let token else { showError("No guardian token: Keychain unavailable"); return }
        var o = obj
        o["auth"] = token
        if client?.send(o) != true, case .some(let t) = obj["type"] as? String, t != "quit" {
            showError("The engine isn't running")
        }
    }

    func beginFlow(_ flow: LockFlow) {
        lockFlow = flow
        entry = ""
        firstEntry = ""
        lockBusy = false
        switch flow {
        case .setup: lockMessage = nil
        case .unlock: lockMessage = nil
        case .change: lockMessage = nil
        }
        if lock.throttleRemaining() > 0, case .unlock = flow { lockMessage = throttleText() }
    }

    func cancelFlow() {
        lockFlow = nil
        entry = ""
        lockMessage = nil
        pendingAfterSetup = nil
    }

    func type(_ ch: String) {
        guard lockFlow != nil, !lockBusy, entry.count < 32 else { return }
        entry += ch
        lockMessage = nil
    }

    func backspace() {
        guard !entry.isEmpty, !lockBusy else { return }
        entry.removeLast()
    }

    func throttleText() -> String {
        let s = Int(lock.throttleRemaining().rounded(.up))
        return String(format: "Too many tries. Wait %d:%02d", s / 60, s % 60)
    }

    func submit() {
        guard let flow = lockFlow, !lockBusy else { return }
        switch flow {
        case .setup(let confirming), .change(let confirming):
            let isSetup: Bool = { if case .setup = flow { return true } else { return false } }()
            if !confirming {
                guard entry.count >= 4 else { lockMessage = "Use at least 4 digits or letters"; shake += 1; entry = ""; return }
                firstEntry = entry
                entry = ""
                lockFlow = isSetup ? .setup(confirming: true) : .change(confirming: true)
                lockMessage = nil
            } else {
                guard entry == firstEntry else {
                    lockMessage = "Those didn't match. Try again."
                    shake += 1
                    entry = ""; firstEntry = ""
                    lockFlow = isSetup ? .setup(confirming: false) : .change(confirming: false)
                    return
                }
                lockBusy = true
                let pw = entry
                DispatchQueue.main.async {
                    let ok = self.lock.setPassword(pw)
                    self.lockBusy = false
                    self.entry = ""; self.firstEntry = ""
                    if ok {
                        self.lockFlow = nil
                        if let a = self.pendingAfterSetup { self.pendingAfterSetup = nil; self.perform(a) }
                    } else {
                        self.lockMessage = "Couldn't save to the Keychain"
                    }
                }
            }
        case .unlock(let action):
            guard !entry.isEmpty else { return }
            lockBusy = true
            lock.verify(entry) { [weak self] result in
                guard let self else { return }
                self.lockBusy = false
                self.entry = ""
                switch result {
                case .ok:
                    self.lockFlow = nil
                    self.lockMessage = nil
                    self.perform(action)
                case .wrong(let left):
                    self.shake += 1
                    self.lockMessage = left > 0 ? "Not quite. \(left) \(left == 1 ? "try" : "tries") left" : "Not quite."
                case .throttled:
                    self.shake += 1
                    self.lockMessage = self.throttleText()
                }
            }
        }
    }

    /// Keyboard input while the panel is key. Returns true when consumed.
    func handleKey(_ event: NSEvent) -> Bool {
        if event.keyCode == 53 {  // Esc
            if lockFlow != nil { cancelFlow() } else { setExpanded(false) }
            return true
        }
        guard lockFlow != nil else { return false }
        switch event.keyCode {
        case 51, 117: backspace(); return true
        case 36, 76: submit(); return true
        default:
            if let chars = event.characters, !event.modifierFlags.contains(.command) {
                let printable = chars.filter { !$0.isNewline && $0.unicodeScalars.allSatisfy { $0.value >= 32 && $0.value != 127 } }
                if !printable.isEmpty { type(String(printable)); return true }
            }
            return false
        }
    }
}
