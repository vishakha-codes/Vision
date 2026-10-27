import SwiftUI

enum MouthStyle { case curve, cat, none }
enum CoverStyle { case paws, shield }

/// Where the shared face parts sit on a character, in unit tile coordinates (y down).
struct FaceSpec {
    var center: CGPoint
    var eyeDX = 0.12
    var eyeDY = -0.03
    var eyeSize = 0.046
    var mouthDY = 0.075
    var mouthW = 0.07
    var mouth: MouthStyle = .curve
    var cheekDX = 0.2
    var cheekDY = 0.055
    var cheekR = 0.045
    var cheeks = true
    var glowEyes = false

    var eyeY: Double { center.y + eyeDY }
}

/// Drawing toolkit handed to each character. All coordinates are unit tile
/// coordinates: (0,0) top-left, (1,1) bottom-right. Bodies may run past the edge;
/// the tile crops them like an app icon.
struct Sketch {
    var ctx: GraphicsContext
    let s: CGFloat
    let pal: Palette
    let f: RigFrame
    let face: FaceSpec

    var pose: Pose { f.pose }

    // MARK: Geometry

    func pt(_ x: Double, _ y: Double) -> CGPoint { CGPoint(x: x * s, y: y * s) }

    func ellipse(_ cx: Double, _ cy: Double, _ rx: Double, _ ry: Double) -> Path {
        Path(ellipseIn: CGRect(x: (cx - rx) * s, y: (cy - ry) * s, width: 2 * rx * s, height: 2 * ry * s))
    }

    func circle(_ cx: Double, _ cy: Double, _ r: Double) -> Path { ellipse(cx, cy, r, r) }

    func rrect(_ x0: Double, _ y0: Double, _ x1: Double, _ y1: Double, _ r: Double) -> Path {
        Path(roundedRect: CGRect(x: x0 * s, y: y0 * s, width: (x1 - x0) * s, height: (y1 - y0) * s),
             cornerRadius: r * s, style: .continuous)
    }

    /// A rounded bar from `a` to `b`.
    func capsule(_ ax: Double, _ ay: Double, _ bx: Double, _ by: Double, _ w: Double) -> Path {
        let dx = (bx - ax), dy = (by - ay)
        let len = max(0.0001, (dx * dx + dy * dy).squareRoot())
        let base = Path(roundedRect: CGRect(x: -w / 2 * s, y: -w / 2 * s, width: (len + w) * s, height: w * s),
                        cornerRadius: w / 2 * s)
        let tf = CGAffineTransform(translationX: ax * s, y: ay * s).rotated(by: atan2(dy, dx))
        return base.applying(tf)
    }

    func rotated(_ path: Path, _ angle: Double, _ cx: Double, _ cy: Double) -> Path {
        let tf = CGAffineTransform(translationX: cx * s, y: cy * s).rotated(by: angle)
            .translatedBy(x: -cx * s, y: -cy * s)
        return path.applying(tf)
    }

    func offset(_ path: Path, _ dx: Double, _ dy: Double) -> Path {
        path.applying(CGAffineTransform(translationX: dx * s, y: dy * s))
    }

    /// A smooth closed curve through unit points (Catmull-Rom).
    func smooth(_ raw: [(Double, Double)]) -> Path {
        let pts = raw.map { pt($0.0, $0.1) }
        var p = Path()
        let n = pts.count
        guard n > 2 else { return p }
        p.move(to: pts[0])
        for i in 0..<n {
            let p0 = pts[(i - 1 + n) % n], p1 = pts[i], p2 = pts[(i + 1) % n], p3 = pts[(i + 2) % n]
            let c1 = CGPoint(x: p1.x + (p2.x - p0.x) / 6, y: p1.y + (p2.y - p0.y) / 6)
            let c2 = CGPoint(x: p2.x - (p3.x - p1.x) / 6, y: p2.y - (p3.y - p1.y) / 6)
            p.addCurve(to: p2, control1: c1, control2: c2)
        }
        p.closeSubpath()
        return p
    }

    func union(_ paths: Path...) -> Path {
        guard var acc = paths.first else { return Path() }
        for p in paths.dropFirst() { acc = acc.union(p) }
        return acc
    }

    /// The common head-on-body silhouette.
    func headBody(_ cx: Double = 0.5, _ cy: Double, _ rx: Double, _ ry: Double,
                  bodyTop: Double = 0.72, bodyInset: Double = 0.12) -> Path {
        union(ellipse(cx, cy, rx, ry),
              rrect(bodyInset, bodyTop, 1 - bodyInset, 1.4, 0.3))
    }

    // MARK: Paint

    func flat(_ path: Path, _ c: RGB, _ alpha: Double = 1) {
        ctx.fill(path, with: .color(c.color(alpha)))
    }

    /// Matte clay shading: soft drop shadow, top-left light, inner occlusion
    /// toward the bottom-right and a faint rim light.
    func clay(_ path: Path, _ base: RGB, shadow: Double = 1, light: Double = 1) {
        let b = path.boundingRect
        guard b.width > 0, b.height > 0 else { return }
        if shadow > 0 {
            var sc = ctx
            sc.addFilter(.blur(radius: s * 0.03))
            sc.translateBy(x: s * 0.004, y: s * 0.022)
            sc.fill(path, with: .color(RGB.ink.color(0.28 * shadow)))
        }
        ctx.fill(path, with: .color(base.color))

        var c = ctx
        c.clip(to: path)
        let span = max(b.width, b.height)
        c.fill(path, with: .radialGradient(
            Gradient(colors: [base.lighter(0.32).color(0.85 * light), base.lighter(0.1).color(0)]),
            center: CGPoint(x: b.minX + b.width * 0.34, y: b.minY + b.height * 0.2),
            startRadius: 0, endRadius: span * 0.7))

        let big = CGRect(x: b.minX - s, y: b.minY - s, width: b.width + 2 * s, height: b.height + 2 * s)
        // Inner occlusion (bottom-right).
        var hole = Path(big)
        hole.addPath(path.applying(CGAffineTransform(translationX: -s * 0.025, y: -s * 0.035)))
        var occ = c
        occ.addFilter(.blur(radius: s * 0.035))
        occ.fill(hole, with: .color(base.darker(0.5).color(0.42)), style: FillStyle(eoFill: true))
        // Rim light (top-left).
        var rim = Path(big)
        rim.addPath(path.applying(CGAffineTransform(translationX: s * 0.012, y: s * 0.016)))
        var rl = c
        rl.addFilter(.blur(radius: s * 0.012))
        rl.fill(rim, with: .color(base.lighter(0.6).color(0.35 * light)), style: FillStyle(eoFill: true))
    }

    func stroke(_ path: Path, _ c: RGB, _ w: Double, _ alpha: Double = 1) {
        ctx.stroke(path, with: .color(c.color(alpha)),
                   style: StrokeStyle(lineWidth: w * s, lineCap: .round, lineJoin: .round))
    }

    func soft(_ path: Path, _ c: RGB, _ alpha: Double, blur: Double) {
        var g = ctx
        g.addFilter(.blur(radius: blur * s))
        g.fill(path, with: .color(c.color(alpha)))
    }

    func gloss(_ cx: Double, _ cy: Double, _ rx: Double, _ ry: Double, _ alpha: Double = 0.55) {
        soft(ellipse(cx, cy, rx, ry), .white, alpha, blur: min(rx, ry) * 0.5)
    }
}

// MARK: - Shared face

extension Sketch {
    var eyeShiftX: Double { f.gazeX * face.eyeDX * 0.3 }
    var eyeShiftY: Double { f.gazeY * 0.024 }

    func drawFace() {
        let fs = face
        let p = pose
        let es = fs.eyeSize * p.eyeScale
        let cy = fs.eyeY + eyeShiftY
        let open = max(0, p.eyeOpen) * (1 - f.blink)

        if fs.cheeks {
            for side in [-1.0, 1.0] {
                soft(ellipse(fs.center.x + side * fs.cheekDX, fs.center.y + fs.cheekDY,
                             fs.cheekR * 1.25, fs.cheekR * 0.78),
                     pal.cheek, min(0.95, 0.55 + 0.4 * p.blush), blur: 0.012)
            }
        }

        for side in [-1.0, 1.0] {
            let cx = fs.center.x + side * fs.eyeDX + eyeShiftX
            if p.eyeOpen < 0.38 {
                // Closed: a soft downward curve.
                var arc = Path()
                arc.move(to: pt(cx - es * 0.62, cy - es * 0.08))
                arc.addQuadCurve(to: pt(cx + es * 0.62, cy - es * 0.08), control: pt(cx, cy + es * 0.6))
                stroke(arc, pal.eye, es * 0.34)
            } else {
                let h = es * 1.12 * max(0.12, open)
                if fs.glowEyes { soft(ellipse(cx, cy, es * 0.9, max(h, es * 0.3) * 0.9), pal.eye, 0.55, blur: es * 0.5) }
                flat(ellipse(cx, cy, es / 2, h / 2), pal.eye)
                let sparkle = min(1, max(0, (p.eyeScale - 1.18) / 0.35)) * min(1, open * 1.5)
                if sparkle > 0.01 {
                    flat(circle(cx - es * 0.16, cy - es * 0.2, es * 0.17), .white, 0.9 * sparkle)
                }
            }
            if p.worry > 0.03 {
                let by = cy - es * 1.5
                var brow = Path()
                brow.move(to: pt(cx + side * es * 0.75, by + es * 0.22))
                brow.addLine(to: pt(cx - side * es * 0.6, by - es * 0.3))
                stroke(brow, pal.eye, es * 0.3, min(1, p.worry))
            }
        }

        drawMouth()

        if p.worry > 0.3 {
            let dx = fs.center.x + fs.eyeDX + 0.1, dy = fs.eyeY - 0.07
            var drop = Path()
            drop.move(to: pt(dx, dy - 0.04))
            drop.addQuadCurve(to: pt(dx, dy + 0.025), control: pt(dx + 0.045, dy + 0.01))
            drop.addQuadCurve(to: pt(dx, dy - 0.04), control: pt(dx - 0.045, dy + 0.01))
            flat(drop, RGB(0xBFE6FF), min(0.95, (p.worry - 0.3) * 1.4))
            flat(circle(dx - 0.008, dy + 0.002, 0.006), .white, 0.8 * p.worry)
        }
    }

    func drawMouth() {
        let fs = face
        let p = pose
        let w = fs.mouthW
        let mx = fs.center.x + eyeShiftX * 0.5
        let my = fs.center.y + fs.mouthDY + eyeShiftY * 0.4
        let lw = fs.eyeSize * 0.3
        if p.mouthOpen > 0.15 {
            let mo = min(1, p.mouthOpen)
            let o = ellipse(mx, my + 0.008, w * 0.3 * (0.7 + 0.3 * mo), w * 0.36 * mo)
            flat(o, pal.eye.darker(0.15))
            var inner = ctx
            inner.clip(to: o)
            inner.fill(ellipse(mx, my + 0.008 + w * 0.3 * mo, w * 0.25, w * 0.16), with: .color(pal.cheek.color(0.9)))
            return
        }
        switch fs.mouth {
        case .none:
            return
        case .curve:
            var m = Path()
            m.move(to: pt(mx - w / 2, my))
            m.addQuadCurve(to: pt(mx + w / 2, my), control: pt(mx, my + p.smile * w * 0.75))
            stroke(m, pal.eye, lw)
        case .cat:
            var m = Path()
            let d = max(-0.3, p.smile) * w * 0.55
            m.move(to: pt(mx - w / 2, my - 0.004))
            m.addQuadCurve(to: pt(mx, my - 0.004), control: pt(mx - w / 4, my + d))
            m.addQuadCurve(to: pt(mx + w / 2, my - 0.004), control: pt(mx + w / 4, my + d))
            stroke(m, pal.eye, lw)
        }
    }
}

// MARK: - Covering (paws or shield)

extension Sketch {
    func drawCover(_ style: CoverStyle, fromSide: Bool = false) {
        let c = min(1.05, max(0, pose.cover))
        guard c > 0.01 else { return }
        let e = 1 - pow(1 - min(1, c), 3)
        let fc = face.center
        let ey = face.eyeY + 0.012
        switch style {
        case .paws:
            for side in [-1.0, 1.0] {
                let tx = fc.x + side * max(0.1, face.eyeDX * 1.02)
                let sx = fc.x + side * (face.eyeDX + (fromSide ? 0.24 : 0.16))
                let sy = fromSide ? fc.y + 0.2 : 1.3
                let x = sx + (tx - sx) * e
                let y = sy + (ey - sy) * e
                if fromSide {
                    if e < 0.15 { continue }
                } else {
                    clay(capsule(sx + side * 0.05, 1.35, x, y + 0.02, 0.14), pal.paw, shadow: 0.7)
                }
                let paw = rotated(ellipse(x, y, 0.1, 0.088), -side * 0.3 * e, x, y)
                clay(paw, pal.paw, shadow: 0.8)
                for i in [-1.0, 1.0] {
                    var toe = Path()
                    toe.move(to: pt(x + i * 0.028, y - 0.085))
                    toe.addLine(to: pt(x + i * 0.024, y - 0.05))
                    stroke(toe, pal.paw.darker(0.35), 0.009, 0.7)
                }
            }
        case .shield:
            let x = fc.x
            let y = 1.45 + (ey + 0.04 - 1.45) * e
            let w = 0.46, h = 0.42
            var sh = Path()
            sh.move(to: pt(x - w / 2, y - h / 2 + 0.05))
            sh.addQuadCurve(to: pt(x, y - h / 2), control: pt(x - w / 2, y - h / 2))
            sh.addQuadCurve(to: pt(x + w / 2, y - h / 2 + 0.05), control: pt(x + w / 2, y - h / 2))
            sh.addCurve(to: pt(x, y + h / 2), control1: pt(x + w / 2, y + 0.08), control2: pt(x + 0.1, y + h / 2 - 0.05))
            sh.addCurve(to: pt(x - w / 2, y - h / 2 + 0.05), control1: pt(x - 0.1, y + h / 2 - 0.05), control2: pt(x - w / 2, y + 0.08))
            sh.closeSubpath()
            clay(sh, pal.accent, shadow: 1)
            stroke(sh, pal.accent.lighter(0.55), 0.018, 0.8)
            // Star emblem.
            var star = Path()
            for i in 0..<10 {
                let a = -Double.pi / 2 + Double(i) * .pi / 5
                let r = i % 2 == 0 ? 0.085 : 0.038
                let q = pt(x + cos(a) * r, y - 0.02 + sin(a) * r)
                if i == 0 { star.move(to: q) } else { star.addLine(to: q) }
            }
            star.closeSubpath()
            flat(star, .white, 0.92)
            for side in [-1.0, 1.0] {
                clay(ellipse(x + side * w / 2, y - 0.02, 0.07, 0.065), pal.paw, shadow: 0.6)
            }
        }
    }
}

// MARK: - Overlays ("?" and "z")

extension Sketch {
    func drawOverlays() {
        let p = pose
        if p.question > 0.02 {
            let q = min(1, p.question)
            var g = ctx
            g.opacity = q
            let bob = sin(f.t * 2.4) * 0.012 * f.motion
            g.translateBy(x: 0.8 * s, y: (0.24 + bob) * s)
            g.rotate(by: .radians(0.22))
            g.scaleBy(x: 0.6 + 0.4 * q, y: 0.6 + 0.4 * q)
            var sh = g
            sh.addFilter(.blur(radius: s * 0.01))
            sh.draw(Text("?").font(.system(size: s * 0.24, weight: .black, design: .rounded))
                .foregroundColor(RGB.ink.color(0.35)), at: CGPoint(x: s * 0.006, y: s * 0.012))
            g.draw(Text("?").font(.system(size: s * 0.24, weight: .black, design: .rounded))
                .foregroundColor(.white), at: .zero)
        }
        if p.sleepy > 0.02 {
            for i in 0..<3 {
                let ph = (f.t * 0.33 + Double(i) / 3).truncatingRemainder(dividingBy: 1)
                var g = ctx
                g.opacity = sin(.pi * ph) * min(1, p.sleepy)
                g.translateBy(x: (0.74 + 0.13 * ph) * s, y: (0.38 - 0.24 * ph) * s)
                g.rotate(by: .radians(-0.18))
                g.draw(Text("z").font(.system(size: s * (0.08 + 0.07 * ph), weight: .heavy, design: .rounded))
                    .foregroundColor(.white), at: .zero)
            }
        }
    }
}
