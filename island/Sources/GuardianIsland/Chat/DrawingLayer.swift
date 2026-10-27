import AppKit
import SwiftUI

/// One explain-my-screen mark, in Quartz global points (origin top-left of the main display, y down).
struct DrawMark: Equatable {
    enum Kind: String { case pointer, box, circle, arrow, label }
    var kind: Kind
    var at = CGPoint.zero        // pointer tip / label top-left
    var rect = CGRect.zero       // box
    var center = CGPoint.zero    // circle
    var radius: CGFloat = 0
    var from = CGPoint.zero      // arrow
    var to = CGPoint.zero
    var bow: CGFloat = 0.2
    var text = ""                // caption or note
    var color = "yellow"
    var seq = 0

    init(kind: Kind) { self.kind = kind }

    /// From a `chat.draw` op. `clear` and unknown kinds return nil (the caller handles `replace`).
    init?(json o: [String: Any]) {
        func num(_ v: Any?) -> CGFloat? {
            if let d = v as? Double, d.isFinite, abs(d) < 100_000 { return CGFloat(d) }
            if let i = v as? Int, abs(i) < 100_000 { return CGFloat(i) }
            return nil
        }
        func pt(_ v: Any?) -> CGPoint? {
            guard let a = v as? [Any], a.count == 2, let x = num(a[0]), let y = num(a[1]) else { return nil }
            return CGPoint(x: x, y: y)
        }
        guard let k = (o["kind"] as? String).flatMap(Kind.init(rawValue:)) else { return nil }
        kind = k
        text = String(((o["label"] as? String) ?? (o["text"] as? String) ?? "").prefix(k == .label ? 80 : 32))
        color = o["color"] as? String ?? "yellow"
        seq = o["seq"] as? Int ?? 0
        switch k {
        case .pointer, .label:
            guard let p = pt(o["at"]) else { return nil }
            at = p
            if k == .label && text.isEmpty { return nil }
        case .box:
            guard let a = o["rect"] as? [Any], a.count == 4, let x = num(a[0]), let y = num(a[1]),
                  let w = num(a[2]), let h = num(a[3]), w > 0, h > 0 else { return nil }
            rect = CGRect(x: x, y: y, width: w, height: h)
        case .circle:
            guard let c = pt(o["center"]), let r = num(o["r"]), r > 0 else { return nil }
            center = c
            radius = r
        case .arrow:
            guard let f = pt(o["from"]), let t = pt(o["to"]) else { return nil }
            from = f
            to = t
            bow = max(-1, min(1, num(o["bow"]) ?? 0.2))
        }
    }

    /// What the mark covers (its target), for label placement.
    var bounds: CGRect {
        switch kind {
        case .pointer: return CGRect(x: at.x - 4, y: at.y - 4, width: 30, height: 30)
        case .label: return CGRect(origin: at, size: CGSize(width: 1, height: 1))
        case .box: return rect
        case .circle: return CGRect(x: center.x - radius, y: center.y - radius, width: 2 * radius, height: 2 * radius)
        case .arrow:
            return CGRect(x: min(from.x, to.x), y: min(from.y, to.y), width: abs(from.x - to.x), height: abs(from.y - to.y))
        }
    }

    var inkLength: CGFloat {
        switch kind {
        case .box: return 2 * (rect.width + rect.height)
        case .circle: return 2 * .pi * radius
        case .arrow: return hypot(to.x - from.x, to.y - from.y) * (1 + abs(bow) * 0.3)
        case .pointer, .label: return 0
        }
    }
}

/// A mark on the timeline: one pen at a time, so each mark starts when the previous one has settled.
struct PlacedMark: Identifiable {
    let id = UUID()
    var mark: DrawMark
    var start: Date
    var duration: TimeInterval
    var flightFrom: CGPoint?   // pointer: where it flies in from
    var settle: Date { start.addingTimeInterval(duration) }
}

@MainActor
final class DrawingStore: ObservableObject {
    @Published private(set) var marks: [PlacedMark] = []
    @Published private(set) var avoid: [CGRect] = []
    @Published private(set) var expiresAt: Date?
    var reduceMotion = false
    private var lastPointer: CGPoint?
    private var expiry: DispatchWorkItem?

    static let penLift: TimeInterval = 0.018
    static let queueLimit: TimeInterval = 3
    static let landing: TimeInterval = 0.55
    static let fadeOut: TimeInterval = 0.45

    var isEmpty: Bool { marks.isEmpty }

    func clear() {
        marks = []
        avoid = []
        expiresAt = nil
        lastPointer = nil
        expiry?.cancel()
    }

    func show(_ new: [DrawMark], avoid: [CGRect], ttl: TimeInterval, replace: Bool, reduceMotion: Bool, now: Date = Date()) {
        self.reduceMotion = reduceMotion
        if replace { marks = []; lastPointer = nil }
        self.avoid = replace ? avoid : self.avoid + avoid
        var t = max(now, marks.map(\.settle).max() ?? now)
        if t.timeIntervalSince(now) > Self.queueLimit { t = now }  // a backlog is drawn at once
        for m in new.sorted(by: { $0.seq < $1.seq }).prefix(12) {
            var p = PlacedMark(mark: m, start: t, duration: 0)
            if reduceMotion {
                p.duration = 0
            } else if m.kind == .pointer {
                let origin = lastPointer ?? Self.pointerHome(near: m.at)
                p.flightFrom = origin
                p.duration = min(max(hypot(m.at.x - origin.x, m.at.y - origin.y) / 900, 0.45), 1.1)
                lastPointer = m.at
            } else if m.kind == .label {
                p.duration = 0.25
            } else {
                p.duration = min(max(TimeInterval(m.inkLength) / 1400, 0.3), 0.9)
            }
            marks.append(p)
            t = p.settle.addingTimeInterval(Self.penLift + (m.text.isEmpty ? 0 : 0.12))
        }
        let end = (marks.map(\.settle).max() ?? now).addingTimeInterval(ttl)
        expiresAt = end
        expiry?.cancel()
        let w = DispatchWorkItem { [weak self] in self?.clear() }
        expiry = w
        DispatchQueue.main.asyncAfter(deadline: .now() + end.timeIntervalSince(now) + Self.fadeOut, execute: w)
    }

    /// The pointer flies in from just under the island (top centre of the screen holding the target).
    static func pointerHome(near p: CGPoint) -> CGPoint {
        let top = NSScreen.screens.first?.frame.maxY ?? 0
        for s in NSScreen.screens {
            let q = CGRect(x: s.frame.minX, y: top - s.frame.maxY, width: s.frame.width, height: s.frame.height)
            if q.contains(p) { return CGPoint(x: q.midX, y: q.minY + 60) }
        }
        return CGPoint(x: p.x, y: max(0, p.y - 300))
    }
}

/// Colour names allowed by the agent's tools.
enum DrawPalette {
    static func color(_ name: String) -> Color {
        switch name {
        case "blue": return Color(red: 0.36, green: 0.62, blue: 1.0)
        case "green": return Color(red: 0.25, green: 0.86, blue: 0.56)
        case "pink": return Color(red: 1.0, green: 0.45, blue: 0.7)
        case "white": return .white
        default: return Color(red: 1.0, green: 0.8, blue: 0.2)
        }
    }
}

/// The overlay for one screen. `origin` is that screen's `CGDisplayBounds` origin, so a global
/// Quartz point p is drawn at p - origin.
struct DrawingCanvasView: View {
    @ObservedObject var store: DrawingStore
    let origin: CGPoint

    static let gap: CGFloat = 6

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 60, paused: store.isEmpty)) { tl in
            Canvas { ctx, size in
                draw(ctx, size: size, now: tl.date)
            }
        }
        .allowsHitTesting(false)
        .ignoresSafeArea()
    }

    private func local(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x - origin.x, y: p.y - origin.y) }
    private func local(_ r: CGRect) -> CGRect { r.offsetBy(dx: -origin.x, dy: -origin.y) }

    private func draw(_ ctx: GraphicsContext, size: CGSize, now: Date) {
        let screen = CGRect(origin: .zero, size: size)
        var fade: Double = 1
        if let end = store.expiresAt {
            let left = end.timeIntervalSince(now)
            if left < 0 { fade = max(0, 1 + left / DrawingStore.fadeOut) }
        }
        guard fade > 0 else { return }
        var ctx = ctx
        ctx.opacity = fade
        // Captions keep off every other mark, every placed caption and every covered region.
        let markRects = store.marks.map { local($0.mark.bounds).insetBy(dx: -Self.gap, dy: -Self.gap) }
        let covered = store.avoid.map(local)
        var placed: [CGRect] = []
        for (i, pm) in store.marks.enumerated() {
            let keepOff = covered + placed + markRects.enumerated().filter { $0.offset != i }.map(\.element)
            func caption(anchor: CGPoint, target: CGRect, since: Date) {
                if let box = self.caption(ctx, pm.mark, anchor: anchor, target: target, since: since, now: now, screen: screen, keepOff: keepOff) {
                    placed.append(box)
                }
            }
            let m = pm.mark
            let raw = pm.duration <= 0 ? 1 : now.timeIntervalSince(pm.start) / pm.duration
            guard raw >= 0 else { continue }
            let p = CGFloat(min(1, raw))
            let color = DrawPalette.color(m.color)
            switch m.kind {
            case .box:
                let r = local(m.rect).insetBy(dx: -(Self.gap + 1.5), dy: -(Self.gap + 1.5))
                stroke(ctx, Path(roundedRect: r, cornerRadius: 8).trimmedPath(from: 0, to: p), color)
                if raw >= 1 { caption(anchor: CGPoint(x: r.maxX, y: r.minY), target: r, since: pm.settle) }
            case .circle:
                let c = local(m.center), r = m.radius + Self.gap + 1.5
                let path = Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r))
                stroke(ctx, path.trimmedPath(from: 0, to: p), color)
                if raw >= 1 {
                    let box = CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)
                    caption(anchor: CGPoint(x: box.maxX, y: box.minY), target: box, since: pm.settle)
                }
            case .arrow:
                let a = local(m.from), b = local(m.to)
                let ctrl = Self.bowControl(a, b, m.bow)
                var path = Path()
                path.move(to: a)
                path.addQuadCurve(to: b, control: ctrl)
                stroke(ctx, path.trimmedPath(from: 0, to: p), color)
                if raw >= 1 {
                    stroke(ctx, Self.arrowHead(tip: b, from: ctrl), color)
                    caption(anchor: a, target: CGRect(origin: a, size: .zero), since: pm.settle)
                }
            case .pointer:
                let tip = local(m.at)
                let pos: CGPoint
                var scale: CGFloat = 1
                if let f = pm.flightFrom, raw < 1 {
                    let t = Self.smootherstep(p)
                    pos = Self.flight(from: local(f), to: tip, t: t)
                    scale = 1 + 0.3 * sin(.pi * t)
                } else {
                    pos = tip
                    let since = now.timeIntervalSince(pm.settle)
                    scale = 1 + 0.035 * CGFloat(0.5 - 0.5 * cos(since * 2 * .pi / 2.8))  // breathing
                }
                pointer(ctx, at: pos, scale: scale, color: color)
                if raw >= 1 {
                    caption(anchor: CGPoint(x: tip.x + 26, y: tip.y + 26), target: CGRect(x: tip.x - 2, y: tip.y - 2, width: 30, height: 30),
                            since: pm.settle.addingTimeInterval(DrawingStore.landing - 0.3))
                }
            case .label:
                note(ctx, m, opacity: Double(p), screen: screen)
            }
        }
    }

    // MARK: Pieces

    private func stroke(_ ctx: GraphicsContext, _ path: Path, _ color: Color) {
        var shadow = ctx
        shadow.addFilter(.shadow(color: .black.opacity(0.45), radius: 2))
        shadow.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: 3.5, lineCap: .round, lineJoin: .round))
    }

    static func bowControl(_ a: CGPoint, _ b: CGPoint, _ bow: CGFloat) -> CGPoint {
        let dx = b.x - a.x, dy = b.y - a.y
        // y is down, so (-dy, dx) is to the right of travel
        return CGPoint(x: (a.x + b.x) / 2 - dy * bow, y: (a.y + b.y) / 2 + dx * bow)
    }

    static func arrowHead(tip: CGPoint, from c: CGPoint) -> Path {
        let angle = atan2(tip.y - c.y, tip.x - c.x)
        var p = Path()
        for delta in [-0.5, 0.5] {
            p.move(to: tip)
            p.addLine(to: CGPoint(x: tip.x - 14 * cos(angle + delta), y: tip.y - 14 * sin(angle + delta)))
        }
        return p
    }

    static func smootherstep(_ x: CGFloat) -> CGFloat { x * x * x * (x * (x * 6 - 15) + 10) }

    /// Cubic arc between two points, bowed to one side.
    static func flight(from s: CGPoint, to e: CGPoint, t: CGFloat) -> CGPoint {
        let dx = e.x - s.x, dy = e.y - s.y, d = max(1, hypot(dx, dy))
        let ux = dx / d, uy = dy / d, nx = -uy, ny = ux
        let dir: CGFloat = s.x <= e.x ? -1 : 1
        let cd = min(max(d * 0.38, 70), 220), ah = min(max(d * 0.08, 18), 58) * dir
        let c1 = CGPoint(x: s.x + ux * cd + nx * ah, y: s.y + uy * cd + ny * ah)
        let c2 = CGPoint(x: e.x - ux * cd - nx * ah * 0.65, y: e.y - uy * cd - ny * ah * 0.65)
        let o = 1 - t
        return CGPoint(x: o * o * o * s.x + 3 * o * o * t * c1.x + 3 * o * t * t * c2.x + t * t * t * e.x,
                       y: o * o * o * s.y + 3 * o * o * t * c1.y + 3 * o * t * t * c2.y + t * t * t * e.y)
    }

    private func pointer(_ ctx: GraphicsContext, at tip: CGPoint, scale: CGFloat, color: Color) {
        let size: CGFloat = 44 * scale
        // TipTourPointerMark's tip sits at (4.04, 4.69) of its 24-unit box
        let k = size / 24
        let rect = CGRect(x: tip.x - 4.04 * k, y: tip.y - 4.69 * k, width: size, height: size)
        let shape = PointerMark().path(in: rect)
        var glow = ctx
        glow.addFilter(.blur(radius: 6))
        glow.fill(shape, with: .color(color.opacity(0.45)))
        ctx.fill(shape, with: .color(.white))
        ctx.stroke(shape, with: .color(color), style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
    }

    /// Draws the mark's caption chip; returns where it went.
    private func caption(_ ctx: GraphicsContext, _ m: DrawMark, anchor: CGPoint, target: CGRect, since: Date, now: Date,
                         screen: CGRect, keepOff: [CGRect]) -> CGRect? {
        guard !m.text.isEmpty else { return nil }
        let age = now.timeIntervalSince(since)
        guard age >= 0 else { return nil }
        let light = !["blue", "pink"].contains(m.color)
        let text = ctx.resolve(Text(m.text).font(.system(size: 15, weight: .semibold, design: .rounded))
            .foregroundColor(light ? .black.opacity(0.85) : .white))
        let ts = text.measure(in: CGSize(width: 260, height: 60))
        let size = CGSize(width: ts.width + 18, height: ts.height + 10)
        let box = Self.labelBox(size, at: anchor, target: target, keepOff: keepOff, in: screen.insetBy(dx: 6, dy: 6))
        var c = ctx
        c.opacity *= min(1, age / 0.2)
        let chip = Path(roundedRect: box, cornerRadius: 7)
        var sh = c
        sh.addFilter(.shadow(color: .black.opacity(0.35), radius: 4, y: 1))
        sh.fill(chip, with: .color(DrawPalette.color(m.color)))
        c.draw(text, in: box.insetBy(dx: 9, dy: 5))
        return box
    }

    /// Beside the anchor or the target, clear of other marks and covered regions; else clamped on screen.
    static func labelBox(_ size: CGSize, at p: CGPoint, target: CGRect, keepOff: [CGRect], in area: CGRect) -> CGRect {
        var spots = [CGPoint(x: p.x + 12, y: p.y - size.height - 4), CGPoint(x: p.x + 12, y: p.y + 8),
                     CGPoint(x: p.x - 12 - size.width, y: p.y - size.height - 4), CGPoint(x: p.x - 12 - size.width, y: p.y + 8)]
        if target.width > 2 {
            spots += [CGPoint(x: target.maxX + 6, y: target.midY - size.height / 2), CGPoint(x: target.minX - 6 - size.width, y: target.midY - size.height / 2),
                      CGPoint(x: target.midX - size.width / 2, y: target.minY - 6 - size.height), CGPoint(x: target.midX - size.width / 2, y: target.maxY + 6)]
        }
        for s in spots {
            let r = CGRect(origin: s, size: size)
            if area.contains(r) && !keepOff.contains(where: { $0.intersects(r) }) && !r.intersects(target) { return r }
        }
        let s = spots[0]
        return CGRect(x: max(area.minX, min(area.maxX - size.width, s.x)), y: max(area.minY, min(area.maxY - size.height, s.y)),
                      width: size.width, height: size.height)
    }

    private func note(_ ctx: GraphicsContext, _ m: DrawMark, opacity: Double, screen: CGRect) {
        let text = ctx.resolve(Text(m.text).font(.system(size: 14, weight: .medium, design: .rounded)).foregroundColor(.white))
        let ts = text.measure(in: CGSize(width: 280, height: 200))
        var box = CGRect(origin: local(m.at), size: CGSize(width: ts.width + 24, height: ts.height + 16))
        box.origin.x = max(screen.minX + 6, min(screen.maxX - box.width - 6, box.origin.x))
        box.origin.y = max(screen.minY + 6, min(screen.maxY - box.height - 6, box.origin.y))
        guard !store.avoid.map(local).contains(where: { $0.intersects(box) }) else { return }
        var c = ctx
        c.opacity *= opacity
        let card = Path(roundedRect: box, cornerRadius: 12)
        var sh = c
        sh.addFilter(.shadow(color: .black.opacity(0.4), radius: 8))
        sh.fill(card, with: .color(Color(white: 0.08).opacity(0.94)))
        c.stroke(card, with: .color(DrawPalette.color(m.color).opacity(0.6)), lineWidth: 1)
        c.draw(text, in: box.insetBy(dx: 12, dy: 8))
    }
}

/// The "look here" pointer: a rounded cursor-like arrow, tip at the top-left.
struct PointerMark: Shape {
    func path(in rect: CGRect) -> Path {
        let scale = min(rect.width, rect.height) / 24
        let ox = rect.midX - 12 * scale, oy = rect.midY - 12 * scale
        func pt(_ x: CGFloat, _ y: CGFloat) -> CGPoint { CGPoint(x: ox + x * scale, y: oy + y * scale) }
        var p = Path()
        p.move(to: pt(4.037, 4.688))
        p.addCurve(to: pt(4.688, 4.037), control1: pt(3.90, 3.90), control2: pt(3.90, 3.90))
        p.addLine(to: pt(20.688, 10.537))
        p.addCurve(to: pt(20.625, 11.484), control1: pt(21.42, 10.84), control2: pt(21.42, 10.84))
        p.addLine(to: pt(14.501, 13.064))
        p.addCurve(to: pt(13.063, 14.499), control1: pt(13.43, 13.34), control2: pt(13.43, 13.34))
        p.addLine(to: pt(11.484, 20.625))
        p.addCurve(to: pt(10.537, 20.688), control1: pt(11.17, 21.42), control2: pt(11.17, 21.42))
        p.closeSubpath()
        return p
    }
}
