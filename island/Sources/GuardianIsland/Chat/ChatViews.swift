import SwiftUI

/// The island's Chat tab: bubbles, a message field, the explain-my-screen chip, and a
/// PIN-gated "Grown-ups" sheet for the chat settings. Fits the 424 x 226 tab area.
struct KidsChatTab: View {
    @ObservedObject var model: IslandModel
    @ObservedObject var chat: ChatModel

    var body: some View {
        Group {
            if chat.parentOpen {
                ParentChatSheet(model: model, chat: chat)
            } else if chat.connected && chat.state.enabled {
                conversation
            } else {
                disabled
            }
        }
        .animation(Theme.quick, value: chat.parentOpen)
    }

    // MARK: Conversation

    private var conversation: some View {
        VStack(alignment: .leading, spacing: 8) {
            topBar
            ChatBubbleList(model: model, chat: chat)
            if let n = chat.notice {
                Text(n).font(Theme.font(10.5, .medium)).foregroundColor(Theme.tone(.amber)).lineLimit(1)
            }
            ChatInputBar(model: model, chat: chat)
        }
    }

    private var topBar: some View {
        HStack(spacing: 6) {
            StatusDot(tone: chat.pendingID == nil ? .green : .amber, size: 6)
            Text(chat.statusLine).font(Theme.font(10.5, .semibold)).foregroundColor(Theme.dim)
            if chat.state.transcripts {
                ChatBadge(text: "TRANSCRIPTS ON", tint: Theme.tone(.amber))
                    .help("A grown-up can read this chat on this Mac")
            }
            Spacer(minLength: 4)
            if !chat.messages.isEmpty {
                ChatIconButton(icon: "square.and.pencil", help: "New chat") { chat.newChat() }
            }
            if !chat.drawings.isEmpty {
                ChatIconButton(icon: "eraser", help: "Clear drawings") { chat.clearDrawings() }
            }
            ChatIconButton(icon: "lock.fill", help: "Grown-ups: chat settings") { chat.openParent() }
        }
        .frame(height: 20)
    }

    // MARK: Disabled

    private var disabled: some View {
        VStack(spacing: 10) {
            Spacer(minLength: 0)
            HStack(alignment: .bottom, spacing: 8) {
                CharacterCanvas(spec: model.spec, variant: model.prefs.variant, zoom: 1.6, corner: 0.5, overlays: false)
                    .frame(width: 30, height: 30)
                Text(disabledText)
                    .font(Theme.font(12, .medium))
                    .foregroundColor(Theme.text)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 12).padding(.vertical, 9)
                    .background(BubbleShape(tailLeft: true).fill(Color.white.opacity(0.1)))
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Spacer(minLength: 0)
            HStack {
                Text(chat.connected ? "Chat is off" : "Chat service not running")
                    .font(Theme.font(11, .semibold)).foregroundColor(Theme.faint)
                Spacer()
                if chat.connected {
                    PillButton(title: "Grown-ups", icon: "lock.fill", tint: model.accent) { chat.openParent() }
                }
            }
        }
    }

    private var disabledText: String {
        if !chat.connected { return "I can't reach the chat helper right now. A grown-up can start it." }
        if chat.state.ageBand == "off" { return "Chat is turned off in Adults mode." }
        return "Hi! I can chat, answer questions and help with homework. Ask a grown-up to turn me on."
    }
}

// MARK: - Bubbles

struct ChatBubbleList: View {
    @ObservedObject var model: IslandModel
    @ObservedObject var chat: ChatModel

    var body: some View {
        if chat.isSnapshot {
            VStack(alignment: .leading, spacing: 7) {
                Spacer(minLength: 0)
                ForEach(chat.messages.suffix(4)) { m in row(m) }
            }
            .frame(minHeight: 0, maxHeight: .infinity, alignment: .bottom)
            .clipped()
        } else {
            live
        }
    }

    private var live: some View {
        ScrollViewReader { proxy in
            ScrollView(.vertical, showsIndicators: false) {
                LazyVStack(alignment: .leading, spacing: 7) {
                    if chat.messages.isEmpty {
                        helperBubble(ChatMessage(id: "hello", role: .helper,
                                                 text: "Hi! Ask me anything: why the sky is blue, a tricky word, or how something works."))
                    }
                    ForEach(chat.messages) { m in
                        row(m).id(m.id)
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .padding(.trailing, 2)
            }
            .onChange(of: chat.messages) { _, _ in
                withAnimation(.easeOut(duration: 0.15)) { proxy.scrollTo("bottom", anchor: .bottom) }
            }
        }
    }

    @ViewBuilder private func row(_ m: ChatMessage) -> some View {
        switch m.role {
        case .child:
            HStack {
                Spacer(minLength: 40)
                Text(m.text)
                    .font(Theme.font(12, .medium))
                    .foregroundColor(.black.opacity(0.78))
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
                    .padding(.horizontal, 11).padding(.vertical, 7)
                    .background(BubbleShape(tailLeft: false).fill(model.accent.opacity(0.88)))
            }
        case .helper:
            helperBubble(m)
        case .notice:
            HStack(spacing: 5) {
                Image(systemName: "exclamationmark.circle.fill").font(.system(size: 10, weight: .bold))
                Text(m.text).font(Theme.font(11, .medium))
            }
            .foregroundColor(Theme.faint)
            .frame(maxWidth: .infinity, alignment: .center)
        }
    }

    private func helperBubble(_ m: ChatMessage) -> some View {
        HStack(alignment: .bottom, spacing: 6) {
            CharacterCanvas(spec: model.spec, variant: model.prefs.variant, zoom: 1.6, corner: 0.5, overlays: false)
                .frame(width: 22, height: 22)
            Group {
                if m.streaming && m.text.isEmpty {
                    TypingDots()
                } else {
                    VStack(alignment: .leading, spacing: 3) {
                        if m.blocked {
                            Label(m.safety ? "Let's get help" : "Let's change the topic",
                                  systemImage: m.safety ? "heart.fill" : "hand.raised.fill")
                                .font(Theme.font(9.5, .heavy))
                                .foregroundColor(m.safety ? Theme.tone(.red) : Theme.tone(.amber))
                        }
                        Text(m.text + (m.streaming ? " ▍" : ""))
                            .font(Theme.font(12, .medium))
                            .foregroundColor(Theme.text)
                            .fixedSize(horizontal: false, vertical: true)
                            .textSelection(.enabled)
                    }
                }
            }
            .padding(.horizontal, 11).padding(.vertical, 7)
            .background(BubbleShape(tailLeft: true).fill(Color.white.opacity(m.blocked ? 0.07 : 0.1)))
            Spacer(minLength: 30)
        }
    }
}

struct TypingDots: View {
    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 20)) { tl in
            let t = tl.date.timeIntervalSinceReferenceDate
            HStack(spacing: 4) {
                ForEach(0..<3) { i in
                    Circle().fill(Theme.dim)
                        .frame(width: 5, height: 5)
                        .opacity(0.35 + 0.65 * max(0, sin(t * 5 - Double(i) * 0.7)))
                }
            }
            .frame(height: 14)
        }
    }
}

// MARK: - Input

struct ChatInputBar: View {
    @ObservedObject var model: IslandModel
    @ObservedObject var chat: ChatModel
    @FocusState private var focused: Bool

    var body: some View {
        HStack(spacing: 6) {
            if chat.state.explainScreen {
                Button {
                    chat.explainNext.toggle()
                } label: {
                    Image(systemName: chat.explainNext ? "viewfinder.circle.fill" : "viewfinder")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundColor(chat.explainNext ? model.accent : Theme.faint)
                }
                .buttonStyle(PressStyle())
                .help(chat.state.explainAvailable ? "Explain my screen: send one covered-safe picture with this question"
                                                  : "Explain my screen isn't available right now")
                .disabled(!chat.state.explainAvailable)
                .opacity(chat.state.explainAvailable ? 1 : 0.4)
            }
            if chat.isSnapshot {
                Text("Message \(model.spec.name)…").font(Theme.font(12, .medium)).foregroundColor(Theme.faint)
                Spacer(minLength: 0)
            } else {
                TextField(chat.explainNext ? "Ask about your screen…" : "Message \(model.spec.name)…", text: $chat.draft)
                    .textFieldStyle(.plain)
                    .font(Theme.font(12, .medium))
                    .foregroundColor(Theme.text)
                    .focused($focused)
                    .onSubmit { chat.send(name: model.spec.name) }
                    .disabled(chat.pendingID != nil)
            }
            if chat.draft.count > ChatModel.maxChars - 80 {
                Text("\(ChatModel.maxChars - chat.draft.count)")
                    .font(Theme.font(10, .bold))
                    .foregroundColor(chat.draft.count > ChatModel.maxChars ? Theme.tone(.red) : Theme.faint)
            }
            if chat.pendingID != nil {
                Button { chat.cancel() } label: {
                    Image(systemName: "stop.circle.fill").font(.system(size: 20)).foregroundColor(Theme.dim)
                }
                .buttonStyle(PressStyle())
                .help("Stop")
            } else {
                Button { chat.send(name: model.spec.name) } label: {
                    Image(systemName: "arrow.up.circle.fill").font(.system(size: 20))
                        .foregroundColor(chat.canSend ? model.accent : Theme.faint)
                }
                .buttonStyle(PressStyle())
                .disabled(!chat.canSend)
            }
        }
        .padding(.leading, 12).padding(.trailing, 5)
        .frame(height: 34)
        .background(Capsule().fill(Theme.card))
        .overlay(Capsule().stroke(chat.explainNext ? model.accent.opacity(0.5) : Theme.stroke, lineWidth: 1))
        .onAppear { DispatchQueue.main.async { focused = true } }
    }
}

// MARK: - Parent sheet (PIN-gated)

struct ParentChatSheet: View {
    @ObservedObject var model: IslandModel
    @ObservedObject var chat: ChatModel
    @FocusState private var pinFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "lock.open.fill").font(.system(size: 10.5, weight: .bold)).foregroundColor(model.accent)
                Text("Chat settings for grown-ups").font(Theme.font(12.5, .bold)).foregroundColor(Theme.text)
                Spacer()
                ChatIconButton(icon: "xmark", help: "Close") { chat.closeParent() }
            }
            if chat.parentUnlocked {
                settings
            } else {
                pinEntry
            }
        }
    }

    private var pinEntry: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Enter the parent PIN to change chat settings.")
                .font(Theme.font(11.5, .medium)).foregroundColor(Theme.dim)
            HStack(spacing: 8) {
                SecureField("PIN", text: $chat.pinEntry)
                    .textFieldStyle(.plain)
                    .font(Theme.font(13, .semibold))
                    .foregroundColor(Theme.text)
                    .focused($pinFocused)
                    .onSubmit { chat.submitPIN() }
                    .padding(.horizontal, 12)
                    .frame(height: 30)
                    .background(Capsule().fill(Theme.card))
                    .overlay(Capsule().stroke(Theme.stroke, lineWidth: 1))
                    .disabled(!model.lock.hasPassword || chat.pinBusy)
                PillButton(title: chat.pinBusy ? "Checking…" : "Unlock", icon: "lock.open.fill", tint: model.accent, filled: true) {
                    chat.submitPIN()
                }
            }
            if let m = chat.pinMessage {
                Text(m).font(Theme.font(11, .semibold)).foregroundColor(Theme.tone(.amber))
            }
            Spacer(minLength: 0)
        }
        .onAppear { DispatchQueue.main.async { pinFocused = true } }
    }

    private var settings: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(alignment: .leading, spacing: 6) {
                toggleRow("bubble.left.and.bubble.right.fill", "Chat", "Off by default. Kid-safe AI with safety filters.",
                          on: chat.state.configured) { chat.setConfig(["enabled": !chat.state.configured]) }
                ageRow
                toggleRow("viewfinder", "Explain my screen", "Sends one picture with covered parts painted over.",
                          on: chat.state.explainScreen) { chat.setConfig(["explain_screen": !chat.state.explainScreen]) }
                toggleRow("doc.text.fill", "Keep transcripts", "Saved on this Mac only. The child sees a badge.",
                          on: chat.state.transcripts) { chat.setConfig(["transcripts": !chat.state.transcripts]) }
                summaryCard
                HStack {
                    Spacer()
                    PillButton(title: "Forget chats", icon: "trash", tint: Theme.tone(.red)) {
                        chat.setConfig(["new_profile": true])
                        chat.newChat()
                    }
                    .help("Start a fresh profile: earlier chat memory is no longer used")
                }
                if let m = chat.pinMessage {
                    Text(m).font(Theme.font(11, .semibold)).foregroundColor(Theme.tone(.amber))
                }
            }
        }
    }

    private var ageRow: some View {
        HStack(spacing: 10) {
            rowIcon("person.2.fill")
            VStack(alignment: .leading, spacing: 1) {
                Text("Age").font(Theme.font(12.5, .semibold)).foregroundColor(Theme.text)
                Text("Auto follows the protection mode").font(Theme.font(10.5, .medium)).foregroundColor(Theme.faint).lineLimit(1)
            }
            Spacer(minLength: 6)
            HStack(spacing: 2) {
                ForEach(["auto", "kids", "teens"], id: \.self) { b in
                    let sel = chat.state.ageBandSetting == b
                    Button { chat.setConfig(["age_band": b]) } label: {
                        Text(b.capitalized).font(Theme.font(11, .semibold))
                            .foregroundColor(sel ? .black.opacity(0.85) : Theme.dim)
                            .padding(.horizontal, 9).frame(height: 22)
                            .background(Capsule().fill(sel ? model.accent : Color.white.opacity(0.08)))
                    }
                    .buttonStyle(PressStyle())
                }
            }
        }
        .padding(.horizontal, 10).frame(height: 42)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Theme.card))
    }

    private var summaryCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("What they chatted about").font(Theme.font(11.5, .bold)).foregroundColor(Theme.text)
                Spacer()
                ChatIconButton(icon: "arrow.clockwise", help: "Refresh") { chat.requestSummary() }
            }
            ForEach(Array(chat.summary.enumerated()), id: \.offset) { _, b in
                Text("• " + b).font(Theme.font(11, .medium)).foregroundColor(Theme.dim).fixedSize(horizontal: false, vertical: true)
            }
            if !chat.summaryCounts.isEmpty {
                Text(chat.summaryCounts).font(Theme.font(10, .medium)).foregroundColor(Theme.faint)
            }
            Text("Topics only, never the child's words, unless transcripts are on.")
                .font(Theme.font(9.5, .medium)).foregroundColor(Theme.faint)
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Theme.card))
    }

    private func rowIcon(_ icon: String) -> some View {
        Image(systemName: icon)
            .font(.system(size: 11, weight: .bold))
            .foregroundColor(model.accent)
            .frame(width: 26, height: 26)
            .background(Circle().fill(model.accent.opacity(0.15)))
    }

    private func toggleRow(_ icon: String, _ title: String, _ detail: String, on: Bool, action: @escaping () -> Void) -> some View {
        HStack(spacing: 10) {
            rowIcon(icon)
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(Theme.font(12.5, .semibold)).foregroundColor(Theme.text)
                Text(detail).font(Theme.font(10.5, .medium)).foregroundColor(Theme.faint).lineLimit(1)
            }
            Spacer(minLength: 6)
            ToggleSwitch(on: on, tint: model.accent, action: action)
        }
        .padding(.horizontal, 10).frame(height: 42)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Theme.card))
    }
}

// MARK: - Small pieces

struct ChatBadge: View {
    var text: String
    var tint: Color
    var body: some View {
        Text(text).font(Theme.font(8.5, .heavy)).tracking(0.6)
            .foregroundColor(tint)
            .padding(.horizontal, 6).padding(.vertical, 2.5)
            .background(Capsule().fill(tint.opacity(0.15)))
    }
}

struct ChatIconButton: View {
    var icon: String
    var help: String
    var action: () -> Void
    @State private var hover = false

    var body: some View {
        Button(action: action) {
            Image(systemName: icon).font(.system(size: 10.5, weight: .bold))
                .foregroundColor(hover ? Theme.text : Theme.faint)
                .frame(width: 22, height: 20)
                .background(Capsule().fill(Color.white.opacity(hover ? 0.12 : 0.06)))
        }
        .buttonStyle(PressStyle())
        .onHover { hover = $0 }
        .help(help)
    }
}
