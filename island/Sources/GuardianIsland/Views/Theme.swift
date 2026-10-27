import AppKit
import SwiftUI

enum Theme {
    static let text = Color.white.opacity(0.95)
    static let dim = Color.white.opacity(0.6)
    static let faint = Color.white.opacity(0.38)
    static let card = Color.white.opacity(0.055)
    static let cardHi = Color.white.opacity(0.09)
    static let stroke = Color.white.opacity(0.08)

    static func font(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .rounded)
    }

    static func tone(_ t: StatusTone) -> Color {
        switch t {
        case .green: return Color(red: 0.25, green: 0.86, blue: 0.56)
        case .amber: return Color(red: 0.99, green: 0.76, blue: 0.24)
        case .red: return Color(red: 0.98, green: 0.42, blue: 0.42)
        case .grey: return Color(red: 0.62, green: 0.64, blue: 0.69)
        }
    }

    static let openSpring = Animation.spring(response: 0.48, dampingFraction: 0.74)
    static let closeCurve = Animation.timingCurve(0.4, 0, 0.2, 1, duration: 0.34)
    static let hoverSpring = Animation.spring(response: 0.3, dampingFraction: 0.68)
    static let quick = Animation.spring(response: 0.28, dampingFraction: 0.82)
}

/// The island silhouette. With `ear > 0` the top corners flare outward like the
/// hardware notch (concave); otherwise the top corners are convex like a pill.
struct IslandShape: Shape {
    var bottomRadius: CGFloat
    var ear: CGFloat
    var topRadius: CGFloat

    var animatableData: AnimatablePair<AnimatablePair<CGFloat, CGFloat>, CGFloat> {
        get { AnimatablePair(AnimatablePair(bottomRadius, ear), topRadius) }
        set { bottomRadius = newValue.first.first; ear = newValue.first.second; topRadius = newValue.second }
    }

    func path(in r: CGRect) -> Path {
        let w = r.width, h = r.height
        var p = Path()
        if ear > 0.5 {
            let e = min(ear, h / 2)
            let br = min(bottomRadius, (h - e) / 2, (w - 2 * e) / 2)
            p.move(to: CGPoint(x: 0, y: 0))
            p.addLine(to: CGPoint(x: w, y: 0))
            p.addQuadCurve(to: CGPoint(x: w - e, y: e), control: CGPoint(x: w - e, y: 0))
            p.addLine(to: CGPoint(x: w - e, y: h - br))
            p.addQuadCurve(to: CGPoint(x: w - e - br, y: h), control: CGPoint(x: w - e, y: h))
            p.addLine(to: CGPoint(x: e + br, y: h))
            p.addQuadCurve(to: CGPoint(x: e, y: h - br), control: CGPoint(x: e, y: h))
            p.addLine(to: CGPoint(x: e, y: e))
            p.addQuadCurve(to: CGPoint(x: 0, y: 0), control: CGPoint(x: e, y: 0))
            p.closeSubpath()
        } else {
            let tr = min(topRadius, h / 2, w / 2)
            let br = min(bottomRadius, h / 2, w / 2)
            p.move(to: CGPoint(x: tr, y: 0))
            p.addLine(to: CGPoint(x: w - tr, y: 0))
            p.addQuadCurve(to: CGPoint(x: w, y: tr), control: CGPoint(x: w, y: 0))
            p.addLine(to: CGPoint(x: w, y: h - br))
            p.addQuadCurve(to: CGPoint(x: w - br, y: h), control: CGPoint(x: w, y: h))
            p.addLine(to: CGPoint(x: br, y: h))
            p.addQuadCurve(to: CGPoint(x: 0, y: h - br), control: CGPoint(x: 0, y: h))
            p.addLine(to: CGPoint(x: 0, y: tr))
            p.addQuadCurve(to: CGPoint(x: tr, y: 0), control: CGPoint(x: 0, y: 0))
            p.closeSubpath()
        }
        return p
    }
}

/// Behind-window blur for the expanded panel (live only; snapshots skip it).
struct VisualEffectBlur: NSViewRepresentable {
    func makeNSView(context: Context) -> NSVisualEffectView {
        let v = NSVisualEffectView()
        v.material = .hudWindow
        v.blendingMode = .behindWindow
        v.state = .active
        v.appearance = NSAppearance(named: .darkAqua)
        return v
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {}
}

// MARK: - Controls (pure SwiftUI so they also render offscreen)

struct HoverHighlight: ViewModifier {
    @State private var hover = false
    var radius: CGFloat = 10
    var base: Color = Theme.card
    var hi: Color = Theme.cardHi

    func body(content: Content) -> some View {
        content
            .background(RoundedRectangle(cornerRadius: radius, style: .continuous).fill(hover ? hi : base))
            .onHover { h in withAnimation(.easeOut(duration: 0.12)) { hover = h } }
    }
}

struct PressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
            .animation(.spring(response: 0.2, dampingFraction: 0.7), value: configuration.isPressed)
    }
}

struct PillButton: View {
    var title: String
    var icon: String?
    var tint: Color = .white
    var filled = false
    var locked = false
    var action: () -> Void

    @State private var hover = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                if let icon { Image(systemName: icon).font(.system(size: 10.5, weight: .bold)) }
                Text(title).font(Theme.font(12, .semibold))
                if locked {
                    Image(systemName: "lock.fill").font(.system(size: 8.5, weight: .bold)).opacity(0.7)
                }
            }
            .foregroundColor(filled ? .black.opacity(0.85) : tint)
            .padding(.horizontal, 12)
            .frame(height: 28)
            .background(Capsule().fill(filled ? tint.opacity(hover ? 1 : 0.92) : Color.white.opacity(hover ? 0.14 : 0.09)))
            .overlay(Capsule().stroke(Color.white.opacity(filled ? 0 : 0.06), lineWidth: 1))
            .contentShape(Capsule())
        }
        .buttonStyle(PressStyle())
        .onHover { hover = $0 }
    }
}

struct ToggleSwitch: View {
    var on: Bool
    var tint: Color
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack(alignment: on ? .trailing : .leading) {
                Capsule().fill(on ? tint : Color.white.opacity(0.16))
                Circle().fill(.white).shadow(color: .black.opacity(0.25), radius: 1.5, y: 1)
                    .padding(2.5)
            }
            .frame(width: 38, height: 22)
            .animation(Theme.quick, value: on)
        }
        .buttonStyle(PressStyle())
    }
}

struct StatusDot: View {
    var tone: StatusTone
    var size: CGFloat = 8
    var body: some View {
        let c = Theme.tone(tone)
        Circle().fill(c)
            .frame(width: size, height: size)
            .background(Circle().fill(c.opacity(0.45)).blur(radius: size * 0.55).frame(width: size * 2, height: size * 2))
    }
}
