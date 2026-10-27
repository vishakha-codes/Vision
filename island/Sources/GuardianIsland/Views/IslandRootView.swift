import SwiftUI

private func mix(_ a: CGFloat, _ b: CGFloat, _ t: CGFloat) -> CGFloat { a + (b - a) * t }

/// The whole island: one animated silhouette, collapsed and expanded content, and a
/// single long-lived character that springs between the notch and the panel.
/// Geometry is a function of a progress value `p` (0 collapsed, 1 expanded). Live,
/// `p` is 0 or 1 and SwiftUI springs between them; snapshots set in-between values.
struct IslandRootView: View {
    @ObservedObject var model: IslandModel
    @ObservedObject var lock: GuardianLock

    init(model: IslandModel) {
        self.model = model
        self.lock = model.lock
    }

    var body: some View {
        let L = model.layout
        let e = model.expanded
        let preview = model.previewProgress
        let p = preview ?? (e ? 1 : 0)
        let rest = L.size(expanded: false, hovering: model.hovering, wide: model.liveWord != nil)
        let size = CGSize(width: mix(rest.width, L.expanded.width, p),
                          height: max(rest.height, mix(rest.height, L.expanded.height, p)))
        let shape = islandShape(L, p, rest)
        let showPanel = preview.map { $0 > 0.12 } ?? e
        let panelFade: Double = preview.map { Double(min(1, max(0, ($0 - 0.1) / 0.5))) } ?? 1

        Color.clear
        .frame(width: size.width, height: size.height)
        .overlay(alignment: .top) {
          ZStack(alignment: .top) {
            IslandBackground(expanded: p > 0.5, snapshot: model.snapshotMode, notch: L.hasNotch)
                .frame(width: size.width, height: size.height)

            CollapsedContent(model: model, size: rest)
                .frame(width: size.width, height: rest.height)
                .opacity(preview.map { Double(max(0, 1 - $0 * 4)) } ?? (e ? 0.0 : 1.0))
                .animation(.easeOut(duration: e ? 0.1 : 0.25).delay(e ? 0 : 0.12), value: e)

            if showPanel {
                ExpandedPanel(model: model)
                    .frame(width: L.expanded.width, height: L.expanded.height)
                    .frame(width: size.width, height: size.height, alignment: .top)
                    .opacity(panelFade)
                    .scaleEffect(preview.map { 0.5 + 0.5 * $0 } ?? 1, anchor: .top)
                    .transition(.asymmetric(
                        // Content grows with the silhouette (about its width ratio) instead of being cropped.
                        insertion: .opacity.animation(.easeOut(duration: 0.24))
                            .combined(with: .scale(scale: 0.5, anchor: .top)),
                        removal: .opacity.animation(.easeOut(duration: 0.12))))
            }

            CharacterPlacement(model: model, size: size, rest: rest, p: p)
                .frame(width: size.width, height: size.height)
          }
          .frame(width: size.width, height: size.height, alignment: .top)
        }
        .clipShape(shape)
        .background(
            shape.fill(Color.black)
                .shadow(color: .black.opacity(mix(0.28, 0.5, min(1, p))), radius: mix(8, 26, min(1, p)),
                        x: 0, y: mix(3, 12, min(1, p)))
        )
        .contentShape(shape)
        .onTapGesture { if !e { model.setExpanded(true) } }
        .frame(width: IslandLayout.panelSize.width, height: IslandLayout.panelSize.height, alignment: .top)
        .offset(y: L.topInset)
        .animation(model.reduceMotion ? .easeInOut(duration: 0.2) : (e ? Theme.openSpring : Theme.closeCurve), value: e)
        .animation(model.reduceMotion ? .easeInOut(duration: 0.15) : Theme.hoverSpring, value: model.hovering)
        .animation(model.reduceMotion ? .easeInOut(duration: 0.2) : Theme.openSpring, value: model.liveWord != nil)
        .environment(\.colorScheme, .dark)
    }

    private func islandShape(_ L: IslandLayout, _ p: CGFloat, _ rest: CGSize) -> IslandShape {
        let t = min(1, max(0, p))
        if L.hasNotch {
            return IslandShape(bottomRadius: mix(13, 30, t), ear: mix(L.earRadius, 14, t), topRadius: 0)
        }
        return IslandShape(bottomRadius: mix(rest.height / 2, 28, t), ear: 0, topRadius: mix(rest.height / 2, 22, t))
    }
}

struct IslandBackground: View {
    var expanded: Bool
    var snapshot: Bool
    var notch: Bool

    var body: some View {
        ZStack {
            if !snapshot { VisualEffectBlur().opacity(expanded ? 1 : 0) }
            LinearGradient(colors: [Color.black, Color(red: 0.055, green: 0.055, blue: 0.075).opacity(expanded ? 0.94 : 1)],
                           startPoint: .top, endPoint: .bottom)
            RadialGradient(colors: [Color.white.opacity(expanded ? 0.05 : 0), .clear],
                           center: UnitPoint(x: 0.25, y: 0.35), startRadius: 0, endRadius: 380)
        }
    }
}

/// The one character instance. Position, size, zoom and corner follow `p`.
struct CharacterPlacement: View {
    @ObservedObject var model: IslandModel
    var size: CGSize
    var rest: CGSize
    var p: CGFloat

    var body: some View {
        let L = model.layout
        let e = model.expanded
        let t = p
        let ear = L.hasNotch ? mix(L.earRadius, 14, min(1, t)) : 0
        let d0 = max(18, rest.height - (L.hasNotch ? 5 : 6))
        let x0 = ear + (L.hasNotch ? 12 : 4) + d0 / 2
        let x1 = ear + 20 + 98
        let d = mix(d0, 176, t)
        let x = mix(x0, x1, t)
        let y = mix(rest.height / 2, L.headerHeight + 6 + 88, t)

        LiveCharacter(spec: model.spec, variant: model.prefs.variant, mood: model.mood,
                      reduceMotion: model.reduceMotion, gaze: e ? model.gaze : nil, poke: model.poke,
                      zoom: mix(1.6, 1, min(1, max(0, t))), corner: mix(0.32, 0.235, min(1, max(0, t))),
                      overlays: t > 0.6, fps: e ? 60 : 30)
            .id(model.spec.id)
            .transition(.scale(scale: 0.6).combined(with: .opacity))
            .frame(width: max(1, d), height: max(1, d))
            .shadow(color: .black.opacity(0.35 * Double(min(1, max(0, t)))), radius: 14, y: 8)
            .scaleEffect(model.hovering && !e ? 1.06 : 1)
            .contentShape(Rectangle())
            .onTapGesture { if e { model.pokeCharacter() } else { model.setExpanded(true) } }
            .position(x: x, y: y)
            .animation(Theme.quick, value: model.spec.id)
    }
}

struct CollapsedContent: View {
    @ObservedObject var model: IslandModel
    var size: CGSize

    var body: some View {
        let L = model.layout
        let ear = L.hasNotch ? L.earRadius : 0
        let word = model.liveWord
        HStack(spacing: 6) {
            Spacer().frame(width: ear + (L.hasNotch ? 12 : 4) + size.height - 2)
            if !L.hasNotch {
                Text(word ?? model.statusTitle)
                    .font(Theme.font(11.5, .semibold))
                    .foregroundColor(word != nil ? Theme.tone(model.tone) : Theme.dim)
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
            HStack(spacing: 6) {
                if L.hasNotch, let word {
                    Text(word)
                        .font(Theme.font(11.5, .bold))
                        .foregroundColor(Theme.tone(model.tone))
                        .lineLimit(1)
                        .fixedSize()
                        .transition(.opacity.combined(with: .scale(scale: 0.8, anchor: .trailing)))
                } else if model.prefs.showMaskCount && model.masks > 0 && model.connected {
                    HStack(spacing: 3) {
                        Image(systemName: "eye.slash.fill").font(.system(size: 9, weight: .bold))
                        Text("\(model.masks)").font(Theme.font(12, .bold)).monospacedDigit()
                    }
                    .foregroundColor(Theme.text)
                    .transition(.scale.combined(with: .opacity))
                }
                StatusDot(tone: model.tone, size: 8)
            }
            .padding(.trailing, ear + (L.hasNotch ? 14 : 11))
            .animation(Theme.quick, value: model.masks)
            .animation(Theme.quick, value: word)
        }
    }
}
