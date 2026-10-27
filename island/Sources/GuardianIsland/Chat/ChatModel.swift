import AppKit
import Combine
import SwiftUI

/// Where the kids chat service listens (kids_chat/PROTOCOL.md). Same support dir as the engine socket.
enum ChatPaths {
    static var socket: String {
        if let s = Paths.env["SG_CHAT_SOCK"], !s.isEmpty { return s }
        return Paths.supportDir.appendingPathComponent("chat.sock").path
    }
}

struct ChatMessage: Identifiable, Equatable {
    enum Role { case child, helper, notice }
    let id: String
    var role: Role
    var text: String
    var streaming = false
    var blocked = false
    var safety = false
}

/// Chat settings as the service reports them (`chat.state`).
struct ChatState: Equatable {
    var enabled = false
    var configured = false
    var ageBand = "kids"          // effective band: kids | teens | off
    var ageBandSetting = "auto"   // auto | kids | teens
    var explainScreen = false
    var explainAvailable = false
    var explainReason = ""
    var transcripts = false
    var turnsLeft = 0
}

/// Kids chat: talks to `kids_chat.service` over its own socket (never the engine's),
/// streams replies into bubbles, and forwards draw ops to the drawing overlay.
/// History lives in RAM only; the service keeps transcripts only if the parent opted in.
@MainActor
final class ChatModel: ObservableObject {
    @Published private(set) var connected = false
    @Published private(set) var state = ChatState()
    @Published private(set) var messages: [ChatMessage] = []
    @Published private(set) var pendingID: String?
    @Published var draft = ""
    @Published var explainNext = false
    @Published var notice: String?

    // Parent settings sheet (PIN-gated)
    @Published var parentOpen = false
    @Published var parentUnlocked = false
    @Published var pinEntry = ""
    @Published var pinMessage: String?
    @Published var pinBusy = false
    @Published private(set) var summary: [String] = []
    @Published private(set) var summaryCounts = ""

    let lock: GuardianLock
    let drawings = DrawingStore()
    private var client: EngineClient?
    private var overlay: DrawingOverlayController?
    private var counter = 0
    private var snapshotMode: Bool
    /// Offscreen snapshots can't render ScrollView or TextField; the views swap in plain stand-ins.
    var isSnapshot: Bool { snapshotMode }
    private var noticeClear: DispatchWorkItem?
    private var relay: AnyCancellable?

    static let maxChars = 500

    init(lock: GuardianLock, snapshot: Bool = false) {
        self.lock = lock
        snapshotMode = snapshot
        // The Chat tab shows a "clear drawings" button while marks are up.
        relay = drawings.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
    }

    // MARK: Lifecycle

    func start() {
        guard !snapshotMode, client == nil else { return }
        // EngineClient is a generic JSON-lines Unix-socket client with reconnect back-off.
        let c = EngineClient(path: ChatPaths.socket)
        c.onConnection = { [weak self] up in
            guard let self else { return }
            self.connected = up
            if up { _ = self.client?.send(["type": "hello"]) } else { self.endPending(notice: "Chat went offline.") }
        }
        c.onMessage = { [weak self] m in self?.handle(m) }
        client = c
        c.start()
    }

    func stop() {
        client?.stop()
        overlay?.close()
    }

    // MARK: Derived

    var canSend: Bool {
        connected && state.enabled && pendingID == nil && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && draft.count <= Self.maxChars && state.turnsLeft > 0
    }

    var statusLine: String {
        if !connected { return "Chat isn't running" }
        if !state.enabled { return state.ageBand == "off" ? "Chat is off in Adults mode" : "Ask a grown-up to turn on chat" }
        return state.ageBand == "teens" ? "Teen mode" : "Kids mode"
    }

    // MARK: Sending

    func send(name: String) {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard canSend, let client else { return }
        counter += 1
        let id = "c\(counter)-\(Int(Date().timeIntervalSince1970) % 100000)"
        var msg: [String: Any] = ["type": "chat.send", "id": id, "text": text, "name": name]
        if explainNext && state.explainScreen { msg["explain"] = true }
        guard client.send(msg) else { flash("Chat isn't running"); return }
        messages.append(ChatMessage(id: id + "-q", role: .child, text: text))
        messages.append(ChatMessage(id: id, role: .helper, text: "", streaming: true))
        trim()
        pendingID = id
        draft = ""
        explainNext = false
    }

    func cancel() {
        guard let id = pendingID else { return }
        _ = client?.send(["type": "chat.cancel", "id": id])
    }

    func newChat() {
        messages.removeAll()
        drawings.clear()
        _ = client?.send(["type": "chat.new"])
    }

    func clearDrawings() {
        drawings.clear()
        _ = client?.send(["type": "draw.clear"])
    }

    // MARK: Incoming

    func handle(_ m: [String: Any]) {
        guard let type = m["type"] as? String else { return }
        let id = m["id"] as? String
        switch type {
        case "chat.state":
            var s = ChatState()
            s.enabled = m["enabled"] as? Bool ?? false
            s.configured = m["configured"] as? Bool ?? false
            s.ageBand = m["age_band"] as? String ?? "kids"
            s.ageBandSetting = m["age_band_setting"] as? String ?? "auto"
            s.explainScreen = m["explain_screen"] as? Bool ?? false
            s.explainAvailable = m["explain_available"] as? Bool ?? false
            s.explainReason = m["explain_reason"] as? String ?? ""
            s.transcripts = m["transcripts"] as? Bool ?? false
            s.turnsLeft = m["turns_left"] as? Int ?? 0
            state = s
            if !s.explainScreen { explainNext = false }
        case "chat.delta":
            guard let id, let text = m["text"] as? String, let i = messages.lastIndex(where: { $0.id == id }) else { return }
            messages[i].text += text
        case "chat.done":
            guard let id, let i = messages.lastIndex(where: { $0.id == id }) else { return }
            messages[i].streaming = false
            if m["blocked"] as? Bool == true {
                messages[i].blocked = true
                messages[i].safety = m["safety"] as? Bool ?? false
                messages[i].text = m["text"] as? String ?? "Let's talk about something else!"
            } else if let t = m["text"] as? String, messages[i].text.isEmpty {
                messages[i].text = t
            } else if m["cancelled"] as? Bool == true, messages[i].text.isEmpty {
                messages[i].text = "(stopped)"
            }
            if pendingID == id { pendingID = nil }
            state.turnsLeft = max(0, state.turnsLeft - 1)
        case "chat.error":
            let text = m["message"] as? String ?? "Something went wrong."
            if let id, let i = messages.lastIndex(where: { $0.id == id }) {
                messages[i].streaming = false
                messages[i].role = .notice
                messages[i].text = text
                if pendingID == id { pendingID = nil }
            } else if parentOpen {
                pinMessage = (m["code"] as? String) == "auth_required" ? "The engine didn't accept the guardian key." : text
            } else {
                flash(text)
            }
        case "chat.draw":
            showDrawing(m)
        case "draw.clear":
            drawings.clear()
        case "chat.summary":
            summary = (m["bullets"] as? [String]) ?? []
            if let c = m["counts"] as? [String: Any] {
                let turns = c["turns"] as? Int ?? 0, blocked = c["blocked"] as? Int ?? 0, safety = c["safety_topics"] as? Int ?? 0
                summaryCounts = "Last 7 days: \(turns) messages · \(blocked) handled by the safety filter · \(safety) safety topics"
            }
            if summary.isEmpty { summary = [(m["note"] as? String) ?? "Nothing to summarise yet."] }
        default:
            break
        }
    }

    private func showDrawing(_ m: [String: Any]) {
        guard (m["space"] as? String) == "quartz_pt", let raw = m["ops"] as? [[String: Any]] else { return }
        let marks = raw.compactMap(DrawMark.init(json:))
        let avoid = ((m["avoid"] as? [[Double]]) ?? []).compactMap { a -> CGRect? in
            a.count == 4 ? CGRect(x: a[0], y: a[1], width: a[2], height: a[3]) : nil
        }
        let ttl = Double(m["ttl_ms"] as? Int ?? 20000) / 1000
        if overlay == nil { overlay = DrawingOverlayController(store: drawings) }
        overlay?.ensurePanels()
        drawings.show(marks, avoid: avoid, ttl: ttl, replace: m["replace"] as? Bool ?? true,
                      reduceMotion: NSWorkspace.shared.accessibilityDisplayShouldReduceMotion)
    }

    private func endPending(notice: String) {
        if let id = pendingID, let i = messages.lastIndex(where: { $0.id == id }) {
            messages[i].streaming = false
            if messages[i].text.isEmpty { messages[i].role = .notice; messages[i].text = notice }
        }
        pendingID = nil
    }

    private func trim() {
        if messages.count > 80 { messages.removeFirst(messages.count - 80) }
    }

    private func flash(_ text: String) {
        notice = text
        noticeClear?.cancel()
        let w = DispatchWorkItem { [weak self] in self?.notice = nil }
        noticeClear = w
        DispatchQueue.main.asyncAfter(deadline: .now() + 5, execute: w)
    }

    // MARK: Parent settings (PIN-gated)

    func openParent() {
        parentOpen = true
        pinEntry = ""
        pinMessage = lock.hasPassword ? nil : "Set a parent PIN in the Protection tab first."
        parentUnlocked = lock.isUnlocked
        if parentUnlocked { requestSummary() }
    }

    func closeParent() {
        parentOpen = false
        pinEntry = ""
        pinMessage = nil
    }

    func submitPIN() {
        guard lock.hasPassword, !pinEntry.isEmpty, !pinBusy else { return }
        pinBusy = true
        let pw = pinEntry
        pinEntry = ""
        lock.verify(pw) { [weak self] result in
            guard let self else { return }
            self.pinBusy = false
            switch result {
            case .ok:
                self.parentUnlocked = true
                self.pinMessage = nil
                self.requestSummary()
            case .wrong(let left):
                self.pinMessage = left > 0 ? "Not quite. \(left) \(left == 1 ? "try" : "tries") left" : "Not quite."
            case .throttled(let until):
                let s = max(0, Int(until.timeIntervalSinceNow.rounded(.up)))
                self.pinMessage = String(format: "Too many tries. Wait %d:%02d", s / 60, s % 60)
            }
        }
    }

    /// Sends a guarded `chat.config`; only while the parent lock is open.
    func setConfig(_ changes: [String: Any]) {
        guard parentUnlocked, lock.isUnlocked else { parentUnlocked = false; pinMessage = "Unlock again to change settings."; return }
        guard let token = GuardianToken.ensure() else { pinMessage = "No guardian key: Keychain unavailable"; return }
        var msg = changes
        msg["type"] = "chat.config"
        msg["auth"] = token
        if client?.send(msg) != true { pinMessage = "Chat isn't running" }
    }

    func requestSummary() {
        guard parentUnlocked, let token = GuardianToken.ensure() else { return }
        summary = ["Loading…"]
        _ = client?.send(["type": "chat.summary", "auth": token])
    }

    // MARK: Snapshots

    func fakeForSnapshot(enabled: Bool) {
        connected = true
        state = ChatState(enabled: enabled, configured: enabled, ageBand: "kids", explainScreen: true, explainAvailable: true,
                          transcripts: false, turnsLeft: 99)
        if enabled {
            messages = [
                ChatMessage(id: "1", role: .child, text: "My favourite animal is the red panda!"),
                ChatMessage(id: "2", role: .helper, text: "Red pandas are so fluffy! They use their bushy tails like a blanket when they sleep."),
                ChatMessage(id: "3", role: .child, text: "Why is the sky blue?"),
                ChatMessage(id: "4", role: .helper, text: "Sunlight is made of many colours. Blue light bounces around the air the most", streaming: true),
            ]
        }
    }
}
