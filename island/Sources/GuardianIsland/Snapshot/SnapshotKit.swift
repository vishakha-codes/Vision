import AppKit
import SwiftUI

@MainActor
enum SnapshotKit {
    @discardableResult
    static func save<V: View>(_ view: V, size: CGSize, to url: URL, scale: CGFloat = 2) -> Bool {
        let r = ImageRenderer(content: view.frame(width: size.width, height: size.height)
            .environment(\.colorScheme, .dark))
        r.scale = scale
        r.proposedSize = ProposedViewSize(size)
        guard let cg = r.cgImage else {
            FileHandle.standardError.write("snapshot: could not render \(url.lastPathComponent)\n".data(using: .utf8)!)
            return false
        }
        let rep = NSBitmapImageRep(cgImage: cg)
        guard let data = rep.representation(using: .png, properties: [:]) else { return false }
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url)
            return true
        } catch {
            FileHandle.standardError.write("snapshot: \(error)\n".data(using: .utf8)!)
            return false
        }
    }
}

/// Every mood of every character in one sheet.
struct MoodSheet: View {
    var characters: [CharacterSpec] = Catalog.all
    var cell: CGFloat = 112

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Text("").frame(width: 70)
                ForEach(Mood.allCases, id: \.self) { m in
                    Text(m.label).font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundColor(.white.opacity(0.7)).frame(width: cell)
                }
            }
            ForEach(characters) { c in
                HStack(spacing: 10) {
                    Text(c.name).font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundColor(.white.opacity(0.85)).frame(width: 70, alignment: .leading)
                    ForEach(Mood.allCases, id: \.self) { m in
                        CharacterCanvas(spec: c, frame: .still(m)).frame(width: cell, height: cell)
                    }
                }
            }
        }
        .padding(20)
        .background(Color(red: 0.09, green: 0.09, blue: 0.11))
    }
}
