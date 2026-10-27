import SwiftUI

// Original characters for Screen Guardian. Every shape is drawn in code; none is
// traced from or derived from any third-party artwork.

private func pal(_ name: String, _ tile: UInt32, _ body: UInt32, _ belly: UInt32, _ accent: UInt32,
                 _ accent2: UInt32, _ eye: UInt32, _ cheek: UInt32, alt: Bool = false) -> Palette {
    Palette(name: name, tile: RGB(tile), body: RGB(body), belly: RGB(belly), accent: RGB(accent),
            accent2: RGB(accent2), eye: RGB(eye), cheek: RGB(cheek), alt: alt)
}

extension Sketch {
    func roundTri(_ a: (Double, Double), _ b: (Double, Double), _ c: (Double, Double), _ r: Double) -> Path {
        let A = pt(a.0, a.1), B = pt(b.0, b.1), C = pt(c.0, c.1)
        var p = Path()
        p.move(to: CGPoint(x: (A.x + B.x) / 2, y: (A.y + B.y) / 2))
        p.addArc(tangent1End: B, tangent2End: C, radius: r * s)
        p.addArc(tangent1End: C, tangent2End: A, radius: r * s)
        p.addArc(tangent1End: A, tangent2End: B, radius: r * s)
        p.closeSubpath()
        return p
    }

    func clipped(_ p: Path) -> Sketch {
        var n = self
        n.ctx.clip(to: p)
        return n
    }

    /// Small bear-style nose with a glint.
    func nose(_ x: Double, _ y: Double, _ w: Double, _ c: RGB) {
        clay(roundTri((x - w / 2, y - w * 0.3), (x + w / 2, y - w * 0.3), (x, y + w * 0.4), w * 0.28), c, shadow: 0, light: 0.5)
        flat(ellipse(x - w * 0.12, y - w * 0.12, w * 0.13, w * 0.07), .white, 0.55)
    }

    var wave: Double { f.motion }

    /// A water drop: round end at (cx, cy), pointing back along `angle` (0 = pointing down).
    func drop(_ cx: Double, _ cy: Double, _ r: Double, _ angle: Double) -> Path {
        var p = Path()
        p.move(to: pt(cx, cy + r * 2.3))
        p.addCurve(to: pt(cx - r, cy), control1: pt(cx - r * 0.2, cy + r * 1.6), control2: pt(cx - r, cy + r * 0.8))
        p.addArc(center: pt(cx, cy), radius: r * s, startAngle: .degrees(180), endAngle: .degrees(360), clockwise: false)
        p.addCurve(to: pt(cx, cy + r * 2.3), control1: pt(cx + r, cy + r * 0.8), control2: pt(cx + r * 0.2, cy + r * 1.6))
        p.closeSubpath()
        return rotated(p, angle, cx, cy)
    }
}

enum Catalog {
    static let all: [CharacterSpec] = [
        pip, ember, biscuit, nova, mallow, sprout, button, waddle, bloop, boo,
        jelly, ribbit, bun, miso, pebble, toasty, axo, honey, leo, zip,
    ]

    static func byID(_ id: String) -> CharacterSpec { all.first { $0.id == id } ?? pip }

    // MARK: Pip — polar bear cub

    static let pip = CharacterSpec(
        id: "pip", name: "Pip", blurb: "A polar bear cub who keeps a calm watch.",
        variants: [
            pal("Cobalt", 0x3E63B6, 0xF4EDE1, 0xFFF9F0, 0xE9C3B4, 0x2B2622, 0x2B2622, 0xF2A39C),
            pal("Navy", 0x27355E, 0xF1ECE4, 0xFFFFFF, 0xDCC6C0, 0x2B2622, 0x2B2622, 0xF0A0A0),
            pal("Teal", 0x2F7F79, 0xF6EEDF, 0xFFF8EC, 0xEAC6B0, 0x2B2622, 0x2B2622, 0xF2A090),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.6), eyeDX: 0.125, eyeDY: -0.035, mouthDY: 0.105,
                       mouthW: 0.05, cheekDX: 0.215, cheekDY: 0.05),
        back: { k in
            for side in [-1.0, 1.0] {
                let x = 0.5 + side * 0.235
                k.clay(k.circle(x, 0.33, 0.095), k.pal.body)
                k.soft(k.circle(x, 0.335, 0.048), k.pal.accent, 0.95, blur: 0.006)
            }
        },
        body: { k in k.clay(k.headBody(0.5, 0.6, 0.36, 0.31, bodyTop: 0.78), k.pal.body) },
        decor: { k in
            k.clay(k.ellipse(0.5, 0.675, 0.12, 0.085), k.pal.belly, shadow: 0.25, light: 0.5)
            k.nose(0.5, 0.652, 0.07, k.pal.accent2)
        }
    )

    // MARK: Ember — fox

    static let ember = CharacterSpec(
        id: "ember", name: "Ember", blurb: "A quick little fox with sharp eyes.",
        variants: [
            pal("Teal", 0x2E7D78, 0xE27A3E, 0xF8EDDF, 0xF8EDDF, 0x3A2A24, 0x2A1E1A, 0xF0A08A),
            pal("Navy", 0x26355E, 0xE5793A, 0xF8EDDF, 0xF8EDDF, 0x3A2A24, 0x2A1E1A, 0xF0A08A),
            pal("Arctic", 0x6A3F6B, 0xF1EDE7, 0xFFFFFF, 0xFFFFFF, 0xB9AEC6, 0x2A2230, 0xF2A6B6),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.6), eyeDX: 0.12, eyeDY: -0.045, mouthDY: 0.085,
                       mouthW: 0.06, mouth: .cat, cheekDX: 0.22, cheekDY: 0.06),
        back: { k in
            for side in [-1.0, 1.0] {
                let m = { (x: Double) in 0.5 + side * (x - 0.5) }
                let ear = k.roundTri((m(0.16), 0.48), (m(0.43), 0.33), (m(0.2), 0.09), 0.045)
                k.clay(ear, k.pal.body)
                k.clipped(ear).flat(k.circle(m(0.2), 0.1, 0.1), k.pal.accent2, 0.95)
                k.soft(k.roundTri((m(0.23), 0.43), (m(0.38), 0.35), (m(0.235), 0.2), 0.03), k.pal.accent, 0.85, blur: 0.006)
            }
        },
        body: { k in k.clay(k.headBody(0.5, 0.58, 0.37, 0.29, bodyTop: 0.72, bodyInset: 0.13), k.pal.body) },
        decor: { k in
            let head = k.headBody(0.5, 0.58, 0.37, 0.29, bodyTop: 0.72, bodyInset: 0.13)
            let mask = k.union(k.ellipse(0.38, 0.71, 0.165, 0.12), k.ellipse(0.62, 0.71, 0.165, 0.12),
                               k.ellipse(0.5, 0.86, 0.22, 0.16))
            k.clipped(head).clay(mask, k.pal.belly, shadow: 0, light: 0.4)
            k.clay(k.ellipse(0.5, 0.645, 0.032, 0.023), k.pal.accent2, shadow: 0, light: 0.5)
            k.flat(k.ellipse(0.49, 0.638, 0.01, 0.005), .white, 0.6)
        }
    )

    // MARK: Biscuit — floppy-eared puppy

    static let biscuit = CharacterSpec(
        id: "biscuit", name: "Biscuit", blurb: "A floppy-eared pup who loves a cuddle.",
        variants: [
            pal("Rust", 0xB4513A, 0xF3E7D3, 0xFFF7EA, 0x8B5A3C, 0x2B211C, 0x2B211C, 0xEE9C8A),
            pal("Mustard", 0xD9A23D, 0xF5EADA, 0xFFF9EF, 0xA0663F, 0x2B211C, 0x2B211C, 0xEE9C8A),
            pal("Sage", 0x8AA27A, 0xF4ECDF, 0xFFFAF2, 0x3D3432, 0x2B211C, 0x2B211C, 0xEFA294),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.59), eyeDX: 0.12, eyeDY: -0.04, mouthDY: 0.11,
                       mouthW: 0.06, cheekDX: 0.205, cheekDY: 0.055),
        body: { k in
            k.clay(k.headBody(0.5, 0.57, 0.34, 0.3, bodyTop: 0.76), k.pal.body)
            let flop = sin(k.f.t * 1.9) * 0.05 * k.wave
            for side in [-1.0, 1.0] {
                let x = 0.5 + side * 0.32, px = 0.5 + side * 0.26
                k.clay(k.rotated(k.ellipse(x, 0.5, 0.098, 0.2), side * (0.25 + flop), px, 0.33), k.pal.accent, shadow: 0.8)
            }
        },
        decor: { k in
            k.soft(k.ellipse(0.62, 0.545, 0.08, 0.072), k.pal.accent, 0.8, blur: 0.006)
            k.clay(k.ellipse(0.5, 0.68, 0.13, 0.09), k.pal.belly, shadow: 0.2, light: 0.5)
            k.clay(k.ellipse(0.5, 0.643, 0.04, 0.026), k.pal.eye, shadow: 0, light: 0.5)
            k.flat(k.ellipse(0.488, 0.636, 0.012, 0.006), .white, 0.6)
        }
    )

    // MARK: Nova — astronaut robot

    static let nova = CharacterSpec(
        id: "nova", name: "Nova", blurb: "A tiny astronaut robot with a shiny visor.",
        variants: [
            pal("Navy", 0x243259, 0xF2F2EF, 0xD9DFE8, 0xFF8A5B, 0x1B2240, 0x86E9FF, 0xFF9EB0),
            pal("Night", 0x1C1C21, 0xF0EEEA, 0xD6D9DF, 0xFFCE5C, 0x16182A, 0x9CF0C4, 0xFF9EB0),
            pal("Coral", 0xE5735C, 0xF6F4F0, 0xDDE2EA, 0x3E63B6, 0x1B2240, 0x86E9FF, 0xFFB3C1),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.555), eyeDX: 0.1, eyeDY: -0.015, eyeSize: 0.052,
                       mouthDY: 0.07, mouthW: 0.06, cheekDX: 0.17, cheekDY: 0.05, cheekR: 0.03, glowEyes: true),
        cover: .shield,
        back: { k in
            k.stroke(k.capsule(0.5, 0.21, 0.5, 0.11, 0.001), k.pal.belly.darker(0.25), 0.022)
            k.soft(k.circle(0.5, 0.095, 0.06), k.pal.accent, 0.5 + 0.3 * sin(k.f.t * 3), blur: 0.03)
            k.clay(k.circle(0.5, 0.095, 0.04), k.pal.accent, shadow: 0.5)
            for side in [-1.0, 1.0] {
                let x = 0.5 + side * 0.355
                k.clay(k.rrect(x - 0.045, 0.44, x + 0.045, 0.62, 0.04), k.pal.belly)
            }
        },
        body: { k in
            k.clay(k.union(k.circle(0.5, 0.53, 0.35), k.rrect(0.15, 0.8, 0.85, 1.4, 0.24)), k.pal.body)
            k.clay(k.rrect(0.26, 0.85, 0.74, 0.93, 0.04), k.pal.belly, shadow: 0.4)
            k.soft(k.circle(0.5, 1.0, 0.035), k.pal.accent, 0.9, blur: 0.006)
        },
        decor: { k in
            let visor = k.rrect(0.245, 0.4, 0.755, 0.7, 0.15)
            k.ctx.fill(visor, with: .linearGradient(
                Gradient(colors: [k.pal.accent2.lighter(0.2).color, k.pal.accent2.color]),
                startPoint: k.pt(0.5, 0.4), endPoint: k.pt(0.5, 0.7)))
            k.stroke(visor, k.pal.accent2.darker(0.4), 0.012, 0.8)
            var streak = Path()
            streak.move(to: k.pt(0.3, 0.52))
            streak.addQuadCurve(to: k.pt(0.42, 0.435), control: k.pt(0.31, 0.44))
            k.stroke(streak, .white, 0.022, 0.4)
            k.flat(k.circle(0.69, 0.45, 0.012), .white, 0.4)
        }
    )

    // MARK: Mallow — cloud

    static let mallow = CharacterSpec(
        id: "mallow", name: "Mallow", blurb: "A soft little cloud that floats nearby.",
        variants: [
            pal("Sky", 0x4F7FD6, 0xFBFAF7, 0xFFFFFF, 0xDDE6F5, 0x2C3550, 0x2C3550, 0xF6A9B5),
            pal("Lavender", 0x9B8FD4, 0xFCFAF8, 0xFFFFFF, 0xE6E0F5, 0x3A3152, 0x3A3152, 0xF6A9B5),
            pal("Sunset", 0xE98A6C, 0xFFF8F0, 0xFFFFFF, 0xFFE4D2, 0x4A2E2A, 0x4A2E2A, 0xF59AA0),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.66), eyeDX: 0.1, eyeDY: -0.02, mouthDY: 0.06,
                       mouthW: 0.06, cheekDX: 0.19, cheekDY: 0.045),
        floats: true,
        back: { k in
            k.soft(k.ellipse(0.5, 0.96, 0.3, 0.035), .ink, 0.22, blur: 0.02)
        },
        body: { k in
            k.clay(k.union(k.circle(0.3, 0.63, 0.19), k.circle(0.5, 0.52, 0.25), k.circle(0.71, 0.6, 0.2),
                           k.rrect(0.1, 0.6, 0.9, 0.87, 0.135)), k.pal.body)
        }
    )

    // MARK: Sprout — cactus (cowboy variant)

    static let sprout = CharacterSpec(
        id: "sprout", name: "Sprout", blurb: "A prickly-looking cactus with a soft heart.",
        variants: [
            pal("Mustard", 0xDDA640, 0x5E9E6E, 0x8CC596, 0xF28BA8, 0xFFE08A, 0x1F2A22, 0xF39B93),
            pal("Coral", 0xE47C64, 0x5A9A6A, 0x8CC596, 0xFFF1C9, 0xF2B33D, 0x1F2A22, 0xF5A6A0),
            pal("Cowboy", 0x8F86C9, 0x63A273, 0x8CC596, 0x8A5A3B, 0xD9573A, 0x1F2A22, 0xF39B93, alt: true),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.6), eyeDX: 0.085, eyeDY: -0.03, eyeSize: 0.042,
                       mouthDY: 0.07, mouthW: 0.05, cheekDX: 0.14, cheekDY: 0.05, cheekR: 0.035),
        back: { k in
            k.clay(k.union(k.rrect(0.1, 0.64, 0.32, 0.77, 0.065), k.rrect(0.1, 0.46, 0.225, 0.77, 0.0625)), k.pal.body)
            k.clay(k.union(k.rrect(0.68, 0.56, 0.9, 0.69, 0.065), k.rrect(0.775, 0.38, 0.9, 0.69, 0.0625)), k.pal.body)
        },
        body: { k in
            k.clay(k.rrect(0.29, 0.26, 0.71, 1.4, 0.21), k.pal.body)
            for (x, y) in [(0.36, 0.4), (0.64, 0.42), (0.35, 0.84), (0.66, 0.88), (0.16, 0.56), (0.84, 0.48), (0.5, 0.95)] {
                var v = Path()
                v.move(to: k.pt(x - 0.012, y - 0.01)); v.addLine(to: k.pt(x, y)); v.addLine(to: k.pt(x + 0.012, y - 0.01))
                k.stroke(v, k.pal.belly.lighter(0.3), 0.008, 0.85)
            }
        },
        front: { k in
            if k.pal.alt {
                k.clay(k.rrect(0.36, 0.1, 0.64, 0.3, 0.08), k.pal.accent)
                k.flat(k.rrect(0.36, 0.24, 0.64, 0.28, 0.005), k.pal.accent2)
                k.clay(k.ellipse(0.5, 0.29, 0.3, 0.045), k.pal.accent, shadow: 0.6)
            } else {
                for i in 0..<5 {
                    let a = Double(i) * 2 * .pi / 5 - .pi / 2
                    k.clay(k.circle(0.5 + cos(a) * 0.04, 0.255 + sin(a) * 0.04, 0.036), k.pal.accent, shadow: 0.35)
                }
                k.clay(k.circle(0.5, 0.255, 0.024), k.pal.accent2, shadow: 0)
            }
        }
    )

    // MARK: Button — mushroom

    static let button = CharacterSpec(
        id: "button", name: "Button", blurb: "A cheerful mushroom from the mossy woods.",
        variants: [
            pal("Sage", 0x8DA67D, 0xF4EBDC, 0xFFF8EE, 0xD9573A, 0xFBF3E6, 0x2B231F, 0xF0A091),
            pal("Navy", 0x2A3A63, 0xF4EBDC, 0xFFF8EE, 0xE1A73E, 0xFFF6E0, 0x2B231F, 0xF0A091),
            pal("Lilac", 0xA597D6, 0xF4EBDC, 0xFFF8EE, 0x7B3F74, 0xF7E6F0, 0x2B231F, 0xF0A0A8),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.75), eyeDX: 0.1, eyeDY: -0.03, mouthDY: 0.07,
                       mouthW: 0.055, cheekDX: 0.17, cheekDY: 0.045),
        body: { k in
            let stem = k.rrect(0.25, 0.5, 0.75, 1.4, 0.2)
            k.clay(stem, k.pal.body)
            k.clipped(stem).soft(k.ellipse(0.5, 0.56, 0.32, 0.06), .ink, 0.25, blur: 0.03)
            var cap = Path()
            cap.move(to: k.pt(0.04, 0.55))
            cap.addCurve(to: k.pt(0.5, 0.1), control1: k.pt(0.04, 0.25), control2: k.pt(0.25, 0.1))
            cap.addCurve(to: k.pt(0.96, 0.55), control1: k.pt(0.75, 0.1), control2: k.pt(0.96, 0.25))
            cap.addQuadCurve(to: k.pt(0.04, 0.55), control: k.pt(0.5, 0.64))
            cap.closeSubpath()
            k.clay(cap, k.pal.accent)
            let spots = k.clipped(cap)
            for (x, y, r) in [(0.3, 0.33, 0.065), (0.58, 0.22, 0.055), (0.76, 0.42, 0.05), (0.47, 0.44, 0.04), (0.15, 0.49, 0.035)] {
                spots.clay(spots.circle(x, y, r), k.pal.accent2, shadow: 0, light: 0.6)
            }
        }
    )

    // MARK: Waddle — penguin

    static let waddle = CharacterSpec(
        id: "waddle", name: "Waddle", blurb: "A tidy penguin who never misses a thing.",
        variants: [
            pal("Mustard", 0xE2AE48, 0x2A2B33, 0xF6EFE2, 0xF08A3C, 0x2A2B33, 0x23242A, 0xF4A3A0),
            pal("Coral", 0xE17D68, 0x2A2B33, 0xF6EFE2, 0xF5A13A, 0x2A2B33, 0x23242A, 0xF4A3A0),
            pal("Teal", 0x2F807C, 0x26304A, 0xF6EFE2, 0xF08A3C, 0x26304A, 0x23242A, 0xF4A3A0),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.6), eyeDX: 0.1, eyeDY: -0.035, mouthDY: 0.12,
                       mouthW: 0.06, mouth: .none, cheekDX: 0.175, cheekDY: 0.055, cheekR: 0.038),
        back: { k in
            let w = sin(k.f.t * 2.2) * 0.06 * k.wave
            k.clay(k.rotated(k.ellipse(0.13, 0.86, 0.075, 0.18), 0.35 + w, 0.13, 0.72), k.pal.body)
            k.clay(k.rotated(k.ellipse(0.87, 0.86, 0.075, 0.18), -0.35 - w, 0.87, 0.72), k.pal.body)
        },
        body: { k in k.clay(k.headBody(0.5, 0.57, 0.36, 0.33, bodyTop: 0.7), k.pal.body) },
        decor: { k in
            let head = k.headBody(0.5, 0.57, 0.36, 0.33, bodyTop: 0.7)
            let mask = k.union(k.circle(0.4, 0.6, 0.15), k.circle(0.6, 0.6, 0.15), k.ellipse(0.5, 0.98, 0.27, 0.32))
            k.clipped(head).clay(mask, k.pal.belly, shadow: 0, light: 0.4)
            k.clay(k.roundTri((0.45, 0.645), (0.55, 0.645), (0.5, 0.705), 0.016), k.pal.accent, shadow: 0.35)
        }
    )

    // MARK: Bloop — whale

    static let bloop = CharacterSpec(
        id: "bloop", name: "Bloop", blurb: "A gentle whale who blows happy bubbles.",
        variants: [
            pal("Coral", 0xE7836A, 0x4C79C9, 0xC5D8F3, 0xDCEEFF, 0xDCEEFF, 0x17213A, 0xF4A1B0),
            pal("Mustard", 0xE0AA45, 0x2E4B8F, 0xBFD0EE, 0xE4F1FF, 0xE4F1FF, 0x111A33, 0xF4A1B0),
            pal("Lavender", 0x9A8DD0, 0x2F8F8A, 0xBFE6DF, 0xE2FFF9, 0xE2FFF9, 0x13302D, 0xF4A1B0),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.72), eyeDX: 0.15, eyeDY: -0.03, mouthDY: 0.065,
                       mouthW: 0.09, cheekDX: 0.25, cheekDY: 0.04),
        back: { k in
            let up = (0.5 + 0.5 * k.f.breathe) * 0.018
            for (a, d, r) in [(-0.62, 0.15, 0.036), (0.0, 0.19, 0.042), (0.62, 0.15, 0.036)] {
                let cx = 0.5 + sin(a) * d, cy = 0.43 - cos(a) * d - up
                k.clay(k.drop(cx, cy, r, a), k.pal.accent2, shadow: 0.3)
            }
        },
        body: { k in
            k.clay(k.union(k.ellipse(0.5, 0.86, 0.48, 0.44), k.rrect(0.02, 0.86, 0.98, 1.4, 0.2)), k.pal.body)
        },
        decor: { k in
            let body = k.union(k.ellipse(0.5, 0.86, 0.48, 0.44), k.rrect(0.02, 0.86, 0.98, 1.4, 0.2))
            let b = k.clipped(body)
            b.clay(k.ellipse(0.5, 1.07, 0.38, 0.2), k.pal.belly, shadow: 0, light: 0.4)
            for x in [0.42, 0.5, 0.58] {
                b.stroke(b.capsule(x, 0.93, x, 1.02, 0.001), k.pal.belly.darker(0.15), 0.01, 0.7)
            }
        }
    )

    // MARK: Boo — ghost

    static let boo = CharacterSpec(
        id: "boo", name: "Boo", blurb: "A shy ghost who is never scary.",
        variants: [
            pal("Plum", 0x6C3F69, 0xF6F1EA, 0xFFFFFF, 0xE4DCEB, 0x2B2230, 0x2B2230, 0xF2A0B5),
            pal("Navy", 0x24315A, 0xF5F2EC, 0xFFFFFF, 0xDCE3F0, 0x232838, 0x232838, 0xF2A0B5),
            pal("Midnight", 0x1E1D22, 0xEDE7FF, 0xFFFFFF, 0xD9D0F5, 0x2A2240, 0x2A2240, 0xF5A3C0),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.55), eyeDX: 0.1, eyeDY: -0.03, eyeSize: 0.05,
                       mouthDY: 0.07, mouthW: 0.055, cheekDX: 0.17, cheekDY: 0.05),
        floats: true,
        back: { k in
            k.soft(k.ellipse(0.5, 0.985, 0.22, 0.03), .ink, 0.25, blur: 0.02)
            let w = sin(k.f.t * 2.4) * 0.08 * k.wave
            k.clay(k.rotated(k.ellipse(0.2, 0.67, 0.055, 0.085), 0.5 + w, 0.2, 0.62), k.pal.body, shadow: 0.5)
            k.clay(k.rotated(k.ellipse(0.8, 0.67, 0.055, 0.085), -0.5 - w, 0.8, 0.62), k.pal.body, shadow: 0.5)
        },
        body: { k in
            var g = Path()
            g.move(to: k.pt(0.2, 0.53))
            g.addArc(center: k.pt(0.5, 0.53), radius: 0.3 * k.s, startAngle: .degrees(180), endAngle: .degrees(360), clockwise: false)
            g.addLine(to: k.pt(0.8, 0.88))
            for i in 0..<3 {
                let x0 = 0.8 - Double(i) * 0.2
                let dip = 0.975 + 0.015 * sin(k.f.t * 3 + Double(i) * 1.7) * k.wave
                g.addQuadCurve(to: k.pt(x0 - 0.2, 0.88), control: k.pt(x0 - 0.1, dip))
            }
            g.closeSubpath()
            k.clay(g, k.pal.body)
        }
    )

    // MARK: Jelly — jellyfish

    static let jelly = CharacterSpec(
        id: "jelly", name: "Jelly", blurb: "A drifting jellyfish with a glow of its own.",
        variants: [
            pal("Navy", 0x23345F, 0xF4B6CF, 0xFFE3EE, 0xE9A0C1, 0xFFFFFF, 0x3A1F33, 0xF07FA5),
            pal("Teal", 0x2E7E7B, 0xCBBAF2, 0xEDE6FF, 0xB6A3EA, 0xFFFFFF, 0x2D2450, 0xF08FB5),
            pal("Midnight", 0x1C1B21, 0xF6EBD7, 0xFFFFFF, 0xEADBC0, 0xFFFFFF, 0x2E2620, 0xF2A090),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.5), eyeDX: 0.1, eyeDY: 0, mouthDY: 0.065,
                       mouthW: 0.05, cheekDX: 0.17, cheekDY: 0.05),
        floats: true,
        back: { k in
            for i in 0..<5 {
                let x = 0.3 + Double(i) * 0.1
                var p = Path()
                p.move(to: k.pt(x, 0.58))
                for j in 1...6 {
                    let y = 0.58 + Double(j) * 0.075
                    let xx = x + sin(Double(j) * 0.9 + Double(i)) * 0.02 + sin(k.f.t * 2.2 + Double(j) * 0.9 + Double(i)) * 0.022 * k.wave
                    p.addLine(to: k.pt(xx, y))
                }
                k.stroke(p, k.pal.accent, 0.028, 0.9)
            }
        },
        body: { k in
            var b = Path()
            b.move(to: k.pt(0.18, 0.56))
            b.addArc(center: k.pt(0.5, 0.56), radius: 0.32 * k.s, startAngle: .degrees(180), endAngle: .degrees(360), clockwise: false)
            for i in 0..<5 {
                let x0 = 0.82 - Double(i) * 0.128
                b.addQuadCurve(to: k.pt(x0 - 0.128, 0.56), control: k.pt(x0 - 0.064, 0.625))
            }
            b.closeSubpath()
            k.clay(b, k.pal.body)
        },
        decor: { k in
            for (x, y, r) in [(0.35, 0.36, 0.024), (0.62, 0.31, 0.019), (0.71, 0.43, 0.014), (0.27, 0.47, 0.012)] {
                k.flat(k.circle(x, y, r), k.pal.belly, 0.75)
            }
        }
    )

    // MARK: Ribbit — frog

    static let ribbit = CharacterSpec(
        id: "ribbit", name: "Ribbit", blurb: "A grinning frog who sits very still.",
        variants: [
            pal("Lavender", 0x9C8FD2, 0x76B062, 0xEAF3CB, 0x5E9450, 0x2E4A2A, 0x1F2A1C, 0xF2989A),
            pal("Coral", 0xE5806A, 0x7BB566, 0xEAF3CB, 0x5E9450, 0x2E4A2A, 0x1F2A1C, 0xF2989A),
            pal("Navy", 0x24345E, 0x8DC26B, 0xF0F6D2, 0x6A9F55, 0x2E4A2A, 0x1F2A1C, 0xF2989A),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.56), eyeDX: 0.2, eyeDY: -0.11, eyeSize: 0.05,
                       mouthDY: 0.15, mouthW: 0.2, cheekDX: 0.27, cheekDY: 0.11),
        body: { k in
            k.clay(k.union(k.ellipse(0.5, 0.68, 0.42, 0.28), k.circle(0.3, 0.45, 0.125), k.circle(0.7, 0.45, 0.125),
                           k.rrect(0.08, 0.76, 0.92, 1.4, 0.3)), k.pal.body)
        },
        decor: { k in
            for x in [0.3, 0.7] { k.clay(k.circle(x, 0.45, 0.075), k.pal.belly, shadow: 0, light: 0.3) }
            k.clay(k.ellipse(0.5, 1.06, 0.3, 0.2), k.pal.belly, shadow: 0, light: 0.3)
        }
    )

    // MARK: Bun — bunny

    static let bun = CharacterSpec(
        id: "bun", name: "Bun", blurb: "A bunny with tall ears that hear everything.",
        variants: [
            pal("Coral", 0xE27E6B, 0xF6EFE6, 0xFFFFFF, 0xF3B2B8, 0x2C2422, 0x2C2422, 0xF4A7A9),
            pal("Cobalt", 0x3F63B5, 0xF6EFE6, 0xFFFFFF, 0xF3B2B8, 0x2C2422, 0x2C2422, 0xF4A7A9),
            pal("Sage", 0x8BA47C, 0xD9C3A9, 0xF2E6D6, 0xF0B7B4, 0x2C2422, 0x2C2422, 0xF0A09A),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.62), eyeDX: 0.12, eyeDY: -0.03, mouthDY: 0.075,
                       mouthW: 0.05, mouth: .cat, cheekDX: 0.2, cheekDY: 0.05),
        back: { k in
            let w = sin(k.f.t * 1.3) * 0.045 * k.wave
            let l = k.rotated(k.rrect(0.3, -0.05, 0.44, 0.44, 0.07), -0.16, 0.37, 0.42)
            k.clay(l, k.pal.body)
            k.soft(k.rotated(k.rrect(0.337, 0.02, 0.403, 0.38, 0.033), -0.16, 0.37, 0.42), k.pal.accent, 0.9, blur: 0.006)
            let r = k.rotated(k.rrect(0.56, -0.03, 0.7, 0.44, 0.07), 0.2 + w, 0.63, 0.42)
            k.clay(r, k.pal.body)
            k.soft(k.rotated(k.rrect(0.597, 0.04, 0.663, 0.38, 0.033), 0.2 + w, 0.63, 0.42), k.pal.accent, 0.9, blur: 0.006)
        },
        body: { k in k.clay(k.headBody(0.5, 0.62, 0.34, 0.29, bodyTop: 0.78), k.pal.body) },
        decor: { k in
            k.clay(k.ellipse(0.5, 0.66, 0.024, 0.017), k.pal.accent.darker(0.1), shadow: 0, light: 0.4)
        }
    )

    // MARK: Miso — cat

    static let miso = CharacterSpec(
        id: "miso", name: "Miso", blurb: "A curious cat who sees in the dark.",
        variants: [
            pal("Coral", 0xE5826B, 0x2D2C33, 0x4A4852, 0xF0A3B0, 0xEDE6DA, 0xF5E9C8, 0xE8798C),
            pal("Mustard", 0xDDA544, 0x2D2C33, 0x4A4852, 0xF0A3B0, 0xEDE6DA, 0xF5E9C8, 0xE8798C),
            pal("Ginger", 0x2F7E7A, 0xF0A35E, 0xF8E3C8, 0xF5B5B5, 0xFFF4E4, 0x2B1F19, 0xF08A80),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.6), eyeDX: 0.125, eyeDY: -0.035, mouthDY: 0.078,
                       mouthW: 0.055, mouth: .cat, cheekDX: 0.21, cheekDY: 0.055),
        back: { k in
            for side in [-1.0, 1.0] {
                let m = { (x: Double) in 0.5 + side * (x - 0.5) }
                k.clay(k.roundTri((m(0.15), 0.46), (m(0.41), 0.33), (m(0.19), 0.13), 0.035), k.pal.body)
                k.soft(k.roundTri((m(0.21), 0.41), (m(0.36), 0.35), (m(0.225), 0.22), 0.02), k.pal.accent, 0.8, blur: 0.005)
            }
        },
        body: { k in k.clay(k.headBody(0.5, 0.6, 0.37, 0.3, bodyTop: 0.78), k.pal.body) },
        decor: { k in
            k.clay(k.roundTri((0.477, 0.642), (0.523, 0.642), (0.5, 0.666), 0.007), k.pal.accent, shadow: 0, light: 0.4)
            for side in [-1.0, 1.0] {
                for dy in [-0.014, 0.014] {
                    var w = Path()
                    w.move(to: k.pt(0.5 + side * 0.17, 0.665 + dy))
                    w.addLine(to: k.pt(0.5 + side * 0.31, 0.655 + dy * 2.6))
                    k.stroke(w, k.pal.accent2, 0.006, 0.55)
                }
            }
        }
    )

    // MARK: Pebble — rock

    static let pebble = CharacterSpec(
        id: "pebble", name: "Pebble", blurb: "A steady little rock with a sprout on top.",
        variants: [
            pal("Teal", 0x5F9C90, 0xC9C3B8, 0xE2DDD3, 0x7DBA6A, 0x8F887C, 0x2B2A28, 0xEE9F96),
            pal("Coral", 0xD97C62, 0xCFC9BF, 0xE6E1D8, 0x7DBA6A, 0x948D82, 0x2B2A28, 0xEE9F96),
            pal("Navy", 0x2B3A62, 0xE3D9C6, 0xF1EADD, 0x8CC77A, 0xA59A86, 0x2B2A28, 0xEE9F96),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.64), eyeDX: 0.11, eyeDY: -0.03, mouthDY: 0.07,
                       mouthW: 0.055, cheekDX: 0.2, cheekDY: 0.05),
        body: { k in
            let rock = k.smooth([(0.04, 1.25), (0.06, 0.72), (0.18, 0.47), (0.42, 0.36), (0.64, 0.35),
                                 (0.85, 0.46), (0.96, 0.7), (0.97, 1.25)])
            k.clay(rock, k.pal.body)
            for (x, y, r) in [(0.26, 0.58, 0.012), (0.79, 0.57, 0.01), (0.71, 0.86, 0.014), (0.22, 0.9, 0.011),
                              (0.41, 0.44, 0.008), (0.85, 0.77, 0.008), (0.12, 0.74, 0.009)] {
                k.flat(k.circle(x, y, r), k.pal.accent2, 0.5)
            }
        },
        front: { k in
            let w = sin(k.f.t * 1.6) * 0.08 * k.wave
            k.stroke(k.capsule(0.6, 0.37, 0.615, 0.28, 0.001), k.pal.accent.darker(0.15), 0.018)
            k.clay(k.rotated(k.ellipse(0.565, 0.26, 0.048, 0.022), 0.5 + w, 0.61, 0.28), k.pal.accent, shadow: 0.3)
            k.clay(k.rotated(k.ellipse(0.665, 0.25, 0.052, 0.024), -0.5 + w, 0.61, 0.28), k.pal.accent, shadow: 0.3)
        }
    )

    // MARK: Toasty — toaster bot

    static let toasty = CharacterSpec(
        id: "toasty", name: "Toasty", blurb: "A toaster bot. Pops up when surprised!",
        variants: [
            pal("Cobalt", 0x3D62B4, 0xF1E8D8, 0xDCCFB9, 0xE0614A, 0xE7B46C, 0x2B2420, 0xF29A8C),
            pal("Teal", 0x2E7F79, 0xF4C6B8, 0xE7AFA0, 0xF2D16B, 0xE7B46C, 0x2B2420, 0xF08C80),
            pal("Mustard", 0xDFA845, 0x9FC9D8, 0x84B3C4, 0xE0614A, 0xEDBE78, 0x1F2A30, 0xF29A8C),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.66), eyeDX: 0.13, eyeDY: -0.03, mouthDY: 0.075,
                       mouthW: 0.06, cheekDX: 0.23, cheekDY: 0.05),
        cover: .shield,
        back: { k in
            let pop = max(k.pose.hop, k.pose.mouthOpen) * 0.07 + max(0, -k.f.hopY) * 1.2
            for (x0, x1, top) in [(0.26, 0.47, 0.13), (0.53, 0.74, 0.16)] {
                let slice = k.rrect(x0, top - pop, x1, 0.46, 0.075)
                k.clay(slice, k.pal.accent2, shadow: 0.5)
                k.clipped(slice).stroke(slice, k.pal.accent2.darker(0.35), 0.03, 0.6)
            }
            k.clay(k.rrect(0.86, 0.52, 0.975, 0.59, 0.03), k.pal.accent, shadow: 0.6)
        },
        body: { k in
            let box = k.rrect(0.08, 0.34, 0.92, 1.25, 0.2)
            k.clay(box, k.pal.body)
            k.clipped(box).flat(k.rrect(0, 0.9, 1, 0.97, 0), k.pal.belly, 0.85)
            for (x0, x1) in [(0.27, 0.46), (0.54, 0.73)] {
                k.flat(k.rrect(x0, 0.37, x1, 0.395, 0.012), k.pal.body.darker(0.5), 0.9)
            }
        }
    )

    // MARK: Axo — axolotl

    static let axo = CharacterSpec(
        id: "axo", name: "Axo", blurb: "A smiley axolotl with frilly gills.",
        variants: [
            pal("Teal", 0x2B7C7F, 0xF6B3C2, 0xFBD3DB, 0xE76F8E, 0xFFD4DE, 0x3A1E27, 0xEE7F9A),
            pal("Navy", 0x24335C, 0xF6B3C2, 0xFBD3DB, 0xE76F8E, 0xFFD4DE, 0x3A1E27, 0xEE7F9A),
            pal("Mustard", 0xE0A846, 0xF3EDE2, 0xFFFFFF, 0xF29AA8, 0xFFE0E6, 0x3A1E27, 0xF29AA8),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.62), eyeDX: 0.17, eyeDY: -0.04, mouthDY: 0.06,
                       mouthW: 0.13, cheekDX: 0.245, cheekDY: 0.04),
        back: { k in
            for side in [-1.0, 1.0] {
                for (i, y) in [0.43, 0.53, 0.63].enumerated() {
                    let a = -0.6 + Double(i) * 0.5 + sin(k.f.t * 2.6 + Double(i)) * 0.07 * k.wave
                    let ax = 0.5 + side * 0.28
                    let bx = ax + side * cos(a) * 0.2, by = y + sin(a) * 0.2
                    k.clay(k.capsule(ax, y, bx, by, 0.075), k.pal.accent, shadow: 0.6)
                    k.flat(k.circle(bx, by, 0.022), k.pal.accent2, 0.6)
                }
            }
        },
        body: { k in k.clay(k.headBody(0.5, 0.6, 0.38, 0.29, bodyTop: 0.78), k.pal.body) }
    )

    // MARK: Honey — honey bear

    static let honey = CharacterSpec(
        id: "honey", name: "Honey", blurb: "A sleepy bear with a sticky secret.",
        variants: [
            pal("Teal", 0x2F7C78, 0xA06B45, 0xE6C69E, 0xF4B73A, 0x24170F, 0x24170F, 0xE88C7A),
            pal("Plum", 0x6E4069, 0xA87249, 0xEACBA4, 0xF4B73A, 0x24170F, 0x24170F, 0xE88C7A),
            pal("Cobalt", 0x3E63B4, 0x6B4A39, 0xD8B48C, 0xF7C04A, 0x1E130D, 0x1E130D, 0xE88C7A),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.6), eyeDX: 0.125, eyeDY: -0.035, mouthDY: 0.105,
                       mouthW: 0.05, cheekDX: 0.215, cheekDY: 0.05),
        back: { k in
            for side in [-1.0, 1.0] {
                let x = 0.5 + side * 0.24
                k.clay(k.circle(x, 0.34, 0.09), k.pal.body)
                k.soft(k.circle(x, 0.345, 0.045), k.pal.belly, 0.9, blur: 0.006)
            }
        },
        body: { k in
            k.clay(k.headBody(0.5, 0.6, 0.36, 0.31, bodyTop: 0.78), k.pal.body)
            let drip = k.union(k.ellipse(0.5, 0.31, 0.17, 0.055), k.capsule(0.415, 0.32, 0.415, 0.395, 0.042),
                               k.capsule(0.565, 0.32, 0.565, 0.365, 0.036))
            k.clay(drip, k.pal.accent, shadow: 0.35)
            k.gloss(0.45, 0.295, 0.05, 0.012, 0.7)
        },
        decor: { k in
            k.clay(k.ellipse(0.5, 0.675, 0.125, 0.09), k.pal.belly, shadow: 0.25, light: 0.5)
            k.nose(0.5, 0.652, 0.07, k.pal.accent2)
        }
    )

    // MARK: Leo — lion

    static let leo = CharacterSpec(
        id: "leo", name: "Leo", blurb: "A brave lion cub with a fluffy mane.",
        variants: [
            pal("Plum", 0x6B3D68, 0xF2B54A, 0xFBE6C0, 0xC9582F, 0x4A2A1C, 0x2E1C12, 0xF08A70),
            pal("Navy", 0x24345E, 0xF2B54A, 0xFBE6C0, 0xD76A2E, 0x4A2A1C, 0x2E1C12, 0xF08A70),
            pal("Teal", 0x2E7D78, 0xF4C35E, 0xFCEBC8, 0x9C4B2C, 0x4A2A1C, 0x2E1C12, 0xF08A70),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.6), eyeDX: 0.1, eyeDY: -0.04, mouthDY: 0.1,
                       mouthW: 0.045, cheekDX: 0.17, cheekDY: 0.04, cheekR: 0.035),
        back: { k in
            var mane = k.circle(0.5, 0.57, 0.3)
            for i in 0..<12 {
                let a = Double(i) * 2 * .pi / 12
                mane = mane.union(k.circle(0.5 + cos(a) * 0.3, 0.57 + sin(a) * 0.3, 0.115))
            }
            k.clay(mane, k.pal.accent)
            k.clay(k.rrect(0.17, 0.82, 0.83, 1.4, 0.3), k.pal.body)
            for side in [-1.0, 1.0] { k.clay(k.circle(0.5 + side * 0.2, 0.37, 0.06), k.pal.body, shadow: 0.5) }
        },
        body: { k in k.clay(k.circle(0.5, 0.59, 0.27), k.pal.body) },
        decor: { k in
            k.clay(k.union(k.ellipse(0.455, 0.67, 0.07, 0.055), k.ellipse(0.545, 0.67, 0.07, 0.055)), k.pal.belly, shadow: 0.2, light: 0.4)
            k.nose(0.5, 0.642, 0.06, k.pal.accent2)
        }
    )

    // MARK: Zip — ninja

    static let zip = CharacterSpec(
        id: "zip", name: "Zip", blurb: "A tiny ninja, super fast and super quiet.",
        variants: [
            pal("Coral", 0xE4795F, 0x2A2F45, 0xF6E6D3, 0xD94040, 0x2A2F45, 0x23202A, 0xF09C90),
            pal("Mustard", 0xDEA644, 0x2A2F45, 0xF6E6D3, 0x3E7BD6, 0x2A2F45, 0x23202A, 0xF09C90),
            pal("Sage", 0x87A27A, 0x3A2F48, 0xF3E2CE, 0xF2C14E, 0x3A2F48, 0x23202A, 0xF09C90),
        ],
        face: FaceSpec(center: CGPoint(x: 0.5, y: 0.58), eyeDX: 0.12, eyeDY: -0.02, mouthDY: 0.1,
                       mouthW: 0.05, mouth: .none, cheekDX: 0.2, cheekDY: 0.035, cheekR: 0.032),
        back: { k in
            let w = sin(k.f.t * 3.1) * 0.03 * k.wave
            k.clay(k.capsule(0.8, 0.42, 0.98, 0.36 + w, 0.05), k.pal.accent, shadow: 0.6)
            k.clay(k.capsule(0.8, 0.44, 0.96, 0.52 - w, 0.045), k.pal.accent, shadow: 0.6)
        },
        body: { k in k.clay(k.headBody(0.5, 0.58, 0.36, 0.32, bodyTop: 0.74), k.pal.body) },
        decor: { k in
            let head = k.headBody(0.5, 0.58, 0.36, 0.32, bodyTop: 0.74)
            let h = k.clipped(head)
            h.clay(k.rrect(0.15, 0.475, 0.85, 0.645, 0.085), k.pal.belly, shadow: 0.3, light: 0.4)
            h.clay(k.rrect(0.1, 0.37, 0.9, 0.435, 0.02), k.pal.accent, shadow: 0.4)
            k.clay(k.circle(0.8, 0.405, 0.04), k.pal.accent, shadow: 0.4)
        }
    )
}
