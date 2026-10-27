import SwiftUI

/// A plain sRGB colour that can be mixed, so every character's shading is derived
/// from its own palette instead of hard-coded greys.
struct RGB: Hashable {
    var r: Double
    var g: Double
    var b: Double

    init(_ hex: UInt32) {
        r = Double((hex >> 16) & 0xFF) / 255
        g = Double((hex >> 8) & 0xFF) / 255
        b = Double(hex & 0xFF) / 255
    }

    init(r: Double, g: Double, b: Double) {
        self.r = r; self.g = g; self.b = b
    }

    static let white = RGB(r: 1, g: 1, b: 1)
    /// Shadows are tinted toward a warm aubergine rather than pure black: it reads as clay.
    static let ink = RGB(r: 0.10, g: 0.06, b: 0.12)

    func mix(_ o: RGB, _ t: Double) -> RGB {
        RGB(r: r + (o.r - r) * t, g: g + (o.g - g) * t, b: b + (o.b - b) * t)
    }

    func lighter(_ t: Double) -> RGB { mix(.white, t) }
    func darker(_ t: Double) -> RGB { mix(.ink, t) }

    var color: Color { Color(.sRGB, red: r, green: g, blue: b, opacity: 1) }
    func color(_ alpha: Double) -> Color { Color(.sRGB, red: r, green: g, blue: b, opacity: alpha) }

    var luminance: Double { 0.2126 * r + 0.7152 * g + 0.0722 * b }
}

/// One colour variant of a character. `alt` lets a variant switch on an accessory
/// (for example the cactus's cowboy hat) without a separate character.
struct Palette {
    var name: String
    var tile: RGB
    var body: RGB
    var belly: RGB
    var accent: RGB
    var accent2: RGB
    var eye: RGB
    var cheek: RGB
    var alt: Bool = false

    var paw: RGB { body.luminance < 0.3 ? body.lighter(0.08) : body.darker(0.05) }
}
