import SwiftUI

/// A character: named parts sharing one face, one cover rig and one animation engine.
struct CharacterSpec: Identifiable {
    let id: String
    let name: String
    let blurb: String
    let variants: [Palette]
    var face: FaceSpec
    var cover: CoverStyle = .paws
    /// Floating characters bob in place instead of breathing from the ground.
    var floats = false
    var back: (Sketch) -> Void = { _ in }
    var body: (Sketch) -> Void
    var decor: (Sketch) -> Void = { _ in }
    var front: (Sketch) -> Void = { _ in }

    func palette(_ i: Int) -> Palette { variants[max(0, min(variants.count - 1, i))] }
}

enum CharacterRenderer {
    static func tilePath(_ rect: CGRect, corner: CGFloat) -> Path {
        Path(roundedRect: rect, cornerRadius: rect.width * corner, style: .continuous)
    }

    static func draw(_ ctx: GraphicsContext, size: CGSize, spec: CharacterSpec, palette pal: Palette,
                     frame f: RigFrame, zoom: CGFloat = 1, corner: CGFloat = 0.235,
                     overlays: Bool = true) {
        let s = min(size.width, size.height)
        var base = ctx
        base.translateBy(x: (size.width - s) / 2, y: (size.height - s) / 2)
        let rect = CGRect(x: 0, y: 0, width: s, height: s)
        let tile = tilePath(rect, corner: corner)

        // Tile: flat, saturated, a touch of light from the top.
        base.fill(tile, with: .linearGradient(
            Gradient(colors: [pal.tile.lighter(0.07).color, pal.tile.darker(0.08).color]),
            startPoint: CGPoint(x: 0, y: 0), endPoint: CGPoint(x: 0, y: s)))
        var glow = base
        glow.clip(to: tile)
        glow.fill(Path(rect), with: .radialGradient(
            Gradient(colors: [pal.tile.lighter(0.16).color(0.7), pal.tile.color(0)]),
            center: CGPoint(x: s * 0.45, y: s * 0.38), startRadius: 0, endRadius: s * 0.7))

        var c = base
        c.clip(to: tile)
        if zoom > 1.001 {
            // Zoom toward the face (collapsed island) and back to the full tile.
            let t = min(1, (zoom - 1) / 0.6)
            let fx = 0.5 + (spec.face.center.x - 0.5) * t
            let fy = 0.5 + (spec.face.center.y - 0.01 - 0.5) * t
            c.translateBy(x: s / 2, y: s / 2)
            c.scaleBy(x: zoom, y: zoom)
            c.translateBy(x: -fx * s, y: -fy * s)
        }
        let overlayCtx = c

        let p = f.pose
        let pivotY: CGFloat = spec.floats ? 0.62 : 1.02
        var bob = 0.0
        if spec.floats { bob = sin(f.t * 1.7) * 0.014 * f.motion }
        let tremble = sin(f.t * 38) * 0.0028 * min(1, max(0, p.cover)) * f.motion
        c.translateBy(x: 0.5 * s, y: pivotY * s)
        c.rotate(by: .radians(f.sway + p.tilt))
        let br = f.breathe * (spec.floats ? 0.006 : 0.011) * (1 + 0.5 * p.sleepy)
        let sq = max(-0.12, min(0.12, p.squash * 0.035))
        c.scaleBy(x: 1 - br * 0.45 + sq, y: 1 + br - sq)
        c.translateBy(x: -0.5 * s, y: -pivotY * s)
        c.translateBy(x: tremble * s, y: (f.hopY + bob) * s)

        let sk = Sketch(ctx: c, s: s, pal: pal, f: f, face: spec.face)
        spec.back(sk)
        spec.body(sk)

        var fctx = c
        fctx.translateBy(x: f.gazeX * 0.014 * s, y: f.gazeY * 0.01 * s)
        let fk = Sketch(ctx: fctx, s: s, pal: pal, f: f, face: spec.face)
        spec.decor(fk)
        fk.drawFace()
        spec.front(sk)
        sk.drawCover(spec.cover, fromSide: spec.floats)

        if overlays {
            Sketch(ctx: overlayCtx, s: s, pal: pal, f: f, face: spec.face).drawOverlays()
        }
    }
}
