import SwiftUI

/// Expanded island: header in the notch band, character column on the left,
/// tabbed options on the right. Laid out at its final size so nothing reflows
/// while the silhouette springs open.
struct ExpandedPanel: View {
    @ObservedObject var model: IslandModel
    @ObservedObject var lock: GuardianLock
    @Namespace private var tabNS

    init(model: IslandModel) {
        self.model = model
        self.lock = model.lock
    }

    static let rightWidth: CGFloat = 424

    var body: some View {
        let L = model.layout
        let ear: CGFloat = L.hasNotch ? 14 : 0
        VStack(spacing: 0) {
            header
                .frame(height: L.headerHeight)
                .padding(.horizontal, ear + 22)
            HStack(alignment: .top, spacing: 18) {
                leftColumn.frame(width: 196)
                rightColumn.frame(width: Self.rightWidth)
            }
            .padding(.top, 6)
            .padding(.horizontal, ear + 20)
            Spacer(minLength: 0)
        }
    }

    private var header: some View {
        let L = model.layout
        return HStack(spacing: 7) {
            Image(systemName: "shield.lefthalf.filled")
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(model.accent)
            Text("Screen Guardian")
                .font(Theme.font(12.5, .bold))
                .foregroundColor(Theme.text)
            Spacer(minLength: L.hasNotch ? L.notchWidth + 20 : 10)
            HStack(spacing: 6) {
                StatusDot(tone: model.tone, size: 7)
                Text(model.statusTitle)
                    .font(Theme.font(11.5, .semibold))
                    .foregroundColor(Theme.dim)
                    .lineLimit(1)
            }
        }
    }

    private var leftColumn: some View {
        VStack(spacing: 6) {
            Color.clear.frame(width: 176, height: 176)  // the CharacterPlacement sits here
            Text(model.spec.name)
                .font(Theme.font(17, .bold))
                .foregroundColor(Theme.text)
                .padding(.top, 6)
            Text(model.speech)
                .font(Theme.font(12, .medium))
                .foregroundColor(Theme.dim)
                .multilineTextAlignment(.center)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
                .frame(width: 190)
                .id(model.speech)
                .transition(.opacity)
                .animation(.easeInOut(duration: 0.2), value: model.speech)
        }
    }

    private var rightColumn: some View {
        ZStack(alignment: .top) {
            VStack(spacing: 12) {
                tabBar
                Group {
                    switch model.tab {
                    case .protection: ProtectionTab(model: model)
                    case .character: CharacterTab(model: model)
                    case .settings: SettingsTab(model: model)
                    case .chat: KidsChatTab(model: model, chat: model.chat)
                    }
                }
                .frame(width: Self.rightWidth, height: 226, alignment: .top)
                .transition(.opacity)
            }
            .blur(radius: model.lockFlow == nil ? 0 : 6)
            .opacity(model.lockFlow == nil ? 1 : 0.35)

            if model.lockFlow != nil {
                LockOverlay(model: model)
                    .frame(width: Self.rightWidth, height: 268)
                    .transition(.opacity.combined(with: .scale(scale: 0.97)))
            }
        }
        .animation(Theme.quick, value: model.lockFlow)
    }

    private var tabBar: some View {
        HStack(spacing: 2) {
            ForEach(PanelTab.allCases) { t in
                let sel = model.tab == t
                Button {
                    withAnimation(Theme.quick) { model.tab = t }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: t.icon).font(.system(size: 10.5, weight: .bold))
                        Text(t.title).font(Theme.font(12, .semibold))
                    }
                    .foregroundColor(sel ? Theme.text : Theme.faint)
                    .frame(maxWidth: .infinity)
                    .frame(height: 26)
                    .background {
                        if sel {
                            Capsule().fill(Color.white.opacity(0.13))
                                .overlay(Capsule().stroke(Color.white.opacity(0.06), lineWidth: 1))
                                .matchedGeometryEffect(id: "tab", in: tabNS)
                        }
                    }
                    .contentShape(Capsule())
                }
                .buttonStyle(PressStyle())
            }
        }
        .padding(3)
        .background(Capsule().fill(Color.white.opacity(0.05)))
    }
}

// MARK: - Protection

struct ProtectionTab: View {
    @ObservedObject var model: IslandModel
    @ObservedObject var lock: GuardianLock

    init(model: IslandModel) {
        self.model = model
        self.lock = model.lock
    }

    var body: some View {
        VStack(spacing: 10) {
            statusCard
            HStack(spacing: 8) {
                ForEach(ProtectionMode.allCases) { m in ModeCard(model: model, mode: m) }
            }
            lockRow
        }
    }

    private var statusCard: some View {
        HStack(spacing: 11) {
            StatusDot(tone: model.tone, size: 10)
                .padding(.leading, 3)
            VStack(alignment: .leading, spacing: 2) {
                Text(model.statusTitle).font(Theme.font(14, .bold)).foregroundColor(Theme.text)
                Text(model.statusDetail)
                    .font(Theme.font(11, .medium))
                    .foregroundColor(model.lastError != nil ? Theme.tone(.amber) : Theme.dim)
                    .lineLimit(1)
                if let line = model.warningText ?? model.engineError {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 9, weight: .bold))
                        Text(line).font(Theme.font(11, .semibold)).lineLimit(1)
                    }
                    .foregroundColor(Theme.tone(.amber))
                }
            }
            Spacer(minLength: 6)
            if model.connected {
                if model.status == .paused {
                    PillButton(title: "Resume", icon: "play.fill", tint: Theme.tone(.green), filled: true) { model.resume() }
                } else if model.status == .protecting {
                    PillButton(title: "Pause", icon: "pause.fill", locked: !lock.isUnlocked) { model.request(.pause) }
                }
            }
        }
        .padding(.horizontal, 12)
        .frame(height: (model.warningText ?? model.engineError) != nil ? 68 : 56)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Theme.card))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Theme.stroke, lineWidth: 1))
    }

    private var lockRow: some View {
        HStack(spacing: 6) {
            Image(systemName: lock.isUnlocked ? "lock.open.fill" : "lock.fill")
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(lock.isUnlocked ? model.accent : Theme.faint)
            Text(lockText)
                .font(Theme.font(11, .medium))
                .foregroundColor(Theme.faint)
                .lineLimit(1)
            Spacer()
            if lock.isUnlocked {
                Button { lock.lockNow() } label: {
                    Text("Lock now").font(Theme.font(11, .bold)).foregroundColor(model.accent)
                }
                .buttonStyle(PressStyle())
            }
        }
        .padding(.horizontal, 4)
        .frame(height: 18)
    }

    private var lockText: String {
        if !lock.hasPassword { return "No guardian PIN yet. Set one to lock these options." }
        if lock.isUnlocked {
            let s = Int(lock.remainingUnlock(now: model.now))
            return String(format: "Grown-up mode · locks again in %d:%02d", s / 60, s % 60)
        }
        return "Locked · a grown-up's PIN is needed to change these"
    }
}

struct ModeCard: View {
    @ObservedObject var model: IslandModel
    var mode: ProtectionMode
    @State private var hover = false

    var body: some View {
        let selected = model.mode == mode
        let pending = model.requestedMode == mode && !selected
        let tint = mode == .off ? Theme.tone(.grey) : model.accent
        Button { model.request(.setMode(mode)) } label: {
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Image(systemName: mode.icon)
                        .font(.system(size: 11.5, weight: .bold))
                        .foregroundColor(selected ? .black.opacity(0.8) : tint)
                        .frame(width: 26, height: 26)
                        .background(Circle().fill(selected ? tint : tint.opacity(0.16)))
                    Spacer()
                    if selected {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundColor(tint)
                            .transition(.scale.combined(with: .opacity))
                    } else if pending {
                        Text("…").font(Theme.font(14, .bold)).foregroundColor(Theme.dim)
                    }
                }
                .padding(.bottom, 3)
                Text(mode.title).font(Theme.font(13.5, .bold)).foregroundColor(Theme.text)
                Text(mode.audience.uppercased())
                    .font(Theme.font(9, .heavy))
                    .tracking(0.8)
                    .foregroundColor(selected ? tint : Theme.faint)
                Text(mode.blurb)
                    .font(Theme.font(10.5, .medium))
                    .foregroundColor(Theme.dim)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 1)
                Spacer(minLength: 0)
            }
            .padding(10)
            .frame(width: (ExpandedPanel.rightWidth - 16) / 3, height: 128, alignment: .topLeading)
            .background(RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(selected ? tint.opacity(0.13) : (hover ? Theme.cardHi : Theme.card)))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(selected ? tint.opacity(0.75) : Theme.stroke, lineWidth: selected ? 1.5 : 1))
            .opacity(model.connected ? 1 : 0.55)
            .contentShape(RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(PressStyle())
        .onHover { hover = $0 }
        .animation(Theme.quick, value: selected)
    }
}

// MARK: - Character

struct CharacterTab: View {
    @ObservedObject var model: IslandModel
    static let cols = 8
    static let tile: CGFloat = 44
    static let gap: CGFloat = 8.5

    var body: some View {
        let rows = stride(from: 0, to: Catalog.all.count, by: Self.cols).map {
            Array(Catalog.all[$0..<min($0 + Self.cols, Catalog.all.count)])
        }
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: Self.gap) {
                ForEach(rows.indices, id: \.self) { r in
                    HStack(spacing: Self.gap) {
                        ForEach(rows[r]) { c in
                            CharacterTileButton(spec: c, selected: c.id == model.prefs.character,
                                                accent: model.accent) { model.choose(character: c.id) }
                        }
                    }
                }
            }
            .padding(.leading, 2)
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Colour").font(Theme.font(10, .heavy)).tracking(0.8).foregroundColor(Theme.faint)
                    Text(model.spec.blurb).font(Theme.font(11, .medium)).foregroundColor(Theme.dim)
                        .lineLimit(2).fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 4)
                ForEach(model.spec.variants.indices, id: \.self) { i in
                    let sel = i == model.prefs.variant
                    Button { model.choose(variant: i) } label: {
                        VStack(spacing: 3) {
                            CharacterCanvas(spec: model.spec, variant: i, zoom: 1.6, corner: 0.5, overlays: false)
                                .frame(width: 30, height: 30)
                                .overlay(Circle().stroke(sel ? Color.white : Color.white.opacity(0.1), lineWidth: sel ? 2 : 1).padding(-2.5))
                            Text(model.spec.variants[i].name).font(Theme.font(9.5, .semibold))
                                .foregroundColor(sel ? Theme.text : Theme.faint)
                        }
                        .frame(width: 52)
                    }
                    .buttonStyle(PressStyle())
                }
            }
            .padding(10)
            .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Theme.card))
        }
    }
}

struct CharacterTileButton: View {
    var spec: CharacterSpec
    var selected: Bool
    var accent: Color
    var action: () -> Void
    @State private var hover = false

    var body: some View {
        Button(action: action) {
            CharacterCanvas(spec: spec, overlays: false)
                .frame(width: CharacterTab.tile, height: CharacterTab.tile)
                .overlay(
                    RoundedRectangle(cornerRadius: CharacterTab.tile * 0.235 + 3, style: .continuous)
                        .stroke(Color.white, lineWidth: selected ? 2.2 : 0)
                        .padding(-3.5)
                )
                .scaleEffect(hover ? 1.08 : 1)
                .shadow(color: .black.opacity(hover ? 0.4 : 0.2), radius: hover ? 6 : 2, y: hover ? 4 : 1)
        }
        .buttonStyle(PressStyle())
        .onHover { h in withAnimation(Theme.hoverSpring) { hover = h } }
        .help(spec.name)
    }
}

// MARK: - Settings

struct SettingsTab: View {
    @ObservedObject var model: IslandModel
    @ObservedObject var lock: GuardianLock

    init(model: IslandModel) {
        self.model = model
        self.lock = model.lock
    }

    var body: some View {
        VStack(spacing: 6) {
            row("figure.walk.motion", "Reduce motion",
                model.systemReduceMotion ? "Also on in System Settings" : "Calmer, simpler animations") {
                ToggleSwitch(on: model.reduceMotion, tint: model.accent) { model.toggleReduceMotion() }
            }
            row("number", "Show mask count", "How many pictures are covered right now") {
                ToggleSwitch(on: model.prefs.showMaskCount, tint: model.accent) { model.toggleMaskCount() }
            }
            row("key.fill", "Guardian PIN",
                lock.hasPassword ? "Needed to change mode, pause or quit" : "Not set yet") {
                PillButton(title: lock.hasPassword ? "Change" : "Set PIN", locked: lock.hasPassword && !lock.isUnlocked) {
                    model.request(.changePassword)
                }
            }
            row("power", "Quit Screen Guardian", "Stops protection on this Mac") {
                PillButton(title: "Quit", tint: Theme.tone(.red), locked: !lock.isUnlocked) { model.request(.quit) }
            }
            HStack(spacing: 6) {
                Image(systemName: "info.circle").font(.system(size: 10, weight: .bold))
                Text(aboutText).lineLimit(2).fixedSize(horizontal: false, vertical: true)
            }
            .font(Theme.font(10.5, .medium))
            .foregroundColor(Theme.faint)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 6)
            .padding(.top, 4)
        }
    }

    private var aboutText: String {
        var s = "Screen Guardian island 1.0"
        if let v = model.engineVersion { s += " · protocol v\(v)" }
        if let cv = model.opencv { s += " · OpenCV \(cv)" }
        return s + ". The island never sees your screen; it only shows the engine's status."
    }

    private func row<C: View>(_ icon: String, _ title: String, _ detail: String,
                              @ViewBuilder trailing: () -> C) -> some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(model.accent)
                .frame(width: 26, height: 26)
                .background(Circle().fill(model.accent.opacity(0.15)))
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(Theme.font(12.5, .semibold)).foregroundColor(Theme.text)
                Text(detail).font(Theme.font(10.5, .medium)).foregroundColor(Theme.faint).lineLimit(1)
            }
            Spacer(minLength: 6)
            trailing()
        }
        .padding(.horizontal, 10)
        .frame(height: 42)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Theme.card))
    }
}

// MARK: - Chat bubbles (used by Chat/ChatViews.swift)

struct BubbleShape: Shape {
    var tailLeft: Bool
    func path(in r: CGRect) -> Path {
        var p = Path(roundedRect: r, cornerRadius: 14, style: .continuous)
        let x = tailLeft ? r.minX + 2 : r.maxX - 2
        let dir: CGFloat = tailLeft ? -1 : 1
        var tail = Path()
        tail.move(to: CGPoint(x: x - dir * 10, y: r.maxY - 1))
        tail.addQuadCurve(to: CGPoint(x: x + dir * 5, y: r.maxY + 1), control: CGPoint(x: x, y: r.maxY + 1))
        tail.addQuadCurve(to: CGPoint(x: x - dir * 2, y: r.maxY - 10), control: CGPoint(x: x - dir * 1, y: r.maxY - 2))
        tail.closeSubpath()
        p.addPath(tail)
        return p
    }
}
