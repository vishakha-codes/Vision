import Foundation

/// Child-adjustable display prefs (island.json). Nothing here weakens protection.
struct Prefs: Codable, Equatable {
    var character = "pip"
    var variant = 0
    var showMaskCount = true
    var reduceMotion = false

    enum CodingKeys: String, CodingKey {
        case character, variant
        case showMaskCount = "show_mask_count"
        case reduceMotion = "reduce_motion"
    }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        character = (try? c.decodeIfPresent(String.self, forKey: .character)) ?? "pip"
        variant = (try? c.decodeIfPresent(Int.self, forKey: .variant)) ?? 0
        showMaskCount = (try? c.decodeIfPresent(Bool.self, forKey: .showMaskCount)) ?? true
        reduceMotion = (try? c.decodeIfPresent(Bool.self, forKey: .reduceMotion)) ?? false
        sanitize()
    }

    mutating func sanitize() {
        if !Catalog.all.contains(where: { $0.id == character }) { character = "pip" }
        let n = Catalog.byID(character).variants.count
        variant = max(0, min(n - 1, variant))
    }

    static func load(from url: URL = Paths.prefs) -> Prefs {
        guard let data = try? Data(contentsOf: url),
              var p = try? JSONDecoder().decode(Prefs.self, from: data) else { return Prefs() }
        p.sanitize()
        return p
    }

    func save(to url: URL = Paths.prefs) {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? enc.encode(self) { Paths.writePrivate(data, to: url) }
    }
}
