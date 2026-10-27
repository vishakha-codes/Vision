import SwiftUI

/// Guardian PIN entry: type on the keyboard or tap the keypad. Letters work too,
/// so a guardian can use a password instead of digits.
struct LockOverlay: View {
    @ObservedObject var model: IslandModel

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 7) {
                Image(systemName: icon)
                    .font(.system(size: 17, weight: .bold))
                    .foregroundColor(model.accent)
                    .frame(width: 38, height: 38)
                    .background(Circle().fill(model.accent.opacity(0.16)))
                Text(title).font(Theme.font(16, .bold)).foregroundColor(Theme.text)
                Text(subtitle)
                    .font(Theme.font(11.5, .medium))
                    .foregroundColor(Theme.dim)
                    .fixedSize(horizontal: false, vertical: true)
                    .lineLimit(3)
                PinDots(count: model.entry.count, busy: model.lockBusy, accent: model.accent)
                    .modifier(Shake(amount: CGFloat(model.shake)))
                    .animation(.linear(duration: 0.4), value: model.shake)
                    .padding(.top, 4)
                Text(model.lockMessage ?? (model.lockBusy ? "Checking…" : "Type on the keyboard or tap the keys"))
                    .font(Theme.font(10.5, .semibold))
                    .foregroundColor(model.lockMessage != nil ? Theme.tone(.amber) : Theme.faint)
                    .lineLimit(2)
                Spacer(minLength: 0)
                HStack(spacing: 8) {
                    PillButton(title: "Cancel") { model.cancelFlow() }
                    if case .setup = model.lockFlow {} else {
                        Text("Esc to close").font(Theme.font(10, .medium)).foregroundColor(Theme.faint)
                    }
                }
            }
            .frame(width: 196, alignment: .leading)
            Keypad(model: model)
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RoundedRectangle(cornerRadius: 18, style: .continuous)
            .fill(Color(red: 0.085, green: 0.085, blue: 0.105)))
        .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(Color.white.opacity(0.09), lineWidth: 1))
        .shadow(color: .black.opacity(0.4), radius: 18, y: 8)
    }

    private var icon: String {
        switch model.lockFlow {
        case .setup, .change: return "key.fill"
        default: return "lock.shield.fill"
        }
    }

    private var title: String {
        switch model.lockFlow {
        case .setup(let c): return c ? "Confirm the PIN" : "Create a guardian PIN"
        case .change(let c): return c ? "Confirm new PIN" : "Choose a new PIN"
        case .unlock: return "Grown-ups only"
        case .none: return ""
        }
    }

    private var subtitle: String {
        switch model.lockFlow {
        case .setup(let c):
            return c ? "Type it once more." : "Grown-ups use it to change protection, pause or quit. 4+ digits or letters."
        case .change(let c):
            return c ? "Type the new PIN again." : "4 or more digits or letters."
        case .unlock(let a):
            return "Enter the guardian PIN to \(a.verb)."
        case .none:
            return ""
        }
    }
}

struct PinDots: View {
    var count: Int
    var busy: Bool
    var accent: Color

    var body: some View {
        let slots = max(4, min(12, count + (count >= 4 ? 0 : 0)))
        HStack(spacing: 7) {
            ForEach(0..<slots, id: \.self) { i in
                Circle()
                    .fill(i < count ? accent : Color.white.opacity(0.12))
                    .frame(width: 10, height: 10)
                    .scaleEffect(i < count ? 1 : 0.8)
                    .animation(.spring(response: 0.22, dampingFraction: 0.6), value: count)
            }
        }
        .opacity(busy ? 0.5 : 1)
        .frame(height: 16)
    }
}

struct Shake: GeometryEffect {
    var amount: CGFloat
    var animatableData: CGFloat {
        get { amount }
        set { amount = newValue }
    }

    func effectValue(size: CGSize) -> ProjectionTransform {
        ProjectionTransform(CGAffineTransform(translationX: 7 * sin(amount * .pi * 4), y: 0))
    }
}

struct Keypad: View {
    @ObservedObject var model: IslandModel
    private let rows: [[String]] = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], ["⌫", "0", "✓"]]

    var body: some View {
        VStack(spacing: 6) {
            ForEach(rows, id: \.self) { row in
                HStack(spacing: 6) {
                    ForEach(row, id: \.self) { k in KeyCap(label: k, accent: model.accent) { press(k) } }
                }
            }
        }
    }

    private func press(_ k: String) {
        switch k {
        case "⌫": model.backspace()
        case "✓": model.submit()
        default: model.type(k)
        }
    }
}

struct KeyCap: View {
    var label: String
    var accent: Color
    var action: () -> Void
    @State private var hover = false

    var body: some View {
        let isOK = label == "✓"
        Button(action: action) {
            Group {
                if label == "⌫" {
                    Image(systemName: "delete.left.fill").font(.system(size: 13, weight: .bold))
                } else if isOK {
                    Image(systemName: "checkmark").font(.system(size: 13, weight: .heavy))
                } else {
                    Text(label).font(Theme.font(16, .semibold)).monospacedDigit()
                }
            }
            .foregroundColor(isOK ? .black.opacity(0.85) : Theme.text)
            .frame(width: 56, height: 46)
            .background(RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(isOK ? accent : Color.white.opacity(hover ? 0.14 : 0.08)))
            .contentShape(RoundedRectangle(cornerRadius: 12))
        }
        .buttonStyle(PressStyle())
        .onHover { hover = $0 }
    }
}
