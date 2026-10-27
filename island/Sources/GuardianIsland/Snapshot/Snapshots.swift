import AppKit
import SwiftUI

/// `GuardianIsland --snapshot DIR`: renders the island and every character offscreen
/// with ImageRenderer so visuals can be reviewed without Screen Recording permission.
@MainActor
enum Snapshots {
    static func makeModel(notch: Bool = true) -> IslandModel {
        let m = IslandModel(snapshot: true)
        m.connected = true
        m.status = .protecting
        m.engineMood = .watching
        m.mode = .aggressive
        m.masks = 2
        m.backend = "local"
        m.fps = 11.8
        m.engineVersion = 1
        m.opencv = "5.0.0"
        var L = IslandLayout()
        L.hasNotch = notch
        L.notchWidth = 185
        L.notchHeight = notch ? 32 : 24
        m.layout = L
        m.lock.fakeForSnapshot(hasPassword: true, unlocked: false)
        return m
    }

    static func island(_ m: IslandModel, height: CGFloat, name: String, in dir: URL) {
        SnapshotKit.save(DesktopScene(notch: m.layout.hasNotch, height: height) { IslandRootView(model: m) },
                         size: CGSize(width: IslandLayout.panelSize.width, height: height),
                         to: dir.appendingPathComponent(name))
    }

    static func run(into dir: URL) {
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // Collapsed and hover.
        island(makeModel(), height: 84, name: "collapsed_notch.png", in: dir)
        let hov = makeModel(); hov.hovering = true
        island(hov, height: 84, name: "hover_notch.png", in: dir)
        island(makeModel(notch: false), height: 84, name: "collapsed_pill.png", in: dir)
        let cov = makeModel(); cov.engineMood = .covering
        island(cov, height: 84, name: "collapsed_covering.png", in: dir)
        let busy = makeModel(); busy.engineMood = .covering; busy.masks = 9
        busy.warningCode = "busy"; busy.warningText = "Lots to cover here"
        island(busy, height: 84, name: "collapsed_busy.png", in: dir)
        let busyX = makeModel(); busyX.engineMood = .covering; busyX.masks = 9; busyX.expanded = true
        busyX.warningCode = "busy"; busyX.warningText = "Lots to cover here"
        island(busyX, height: 430, name: "expanded_busy.png", in: dir)
        SnapshotKit.save(CollapsedStates(), size: CGSize(width: 820, height: 412), to: dir.appendingPathComponent("collapsed_states.png"))

        // Expanded, one per tab.
        for tab in PanelTab.allCases {
            let m = makeModel()
            m.expanded = true
            m.tab = tab
            if tab == .settings { m.lock.fakeForSnapshot(hasPassword: true, unlocked: true) }
            if tab == .chat { m.chat.fakeForSnapshot(enabled: true) }
            island(m, height: 430, name: "expanded_\(tab.rawValue).png", in: dir)
        }
        let lockM = makeModel()
        lockM.expanded = true
        lockM.beginFlow(.unlock(.setMode(.medium)))
        lockM.entry = "123"
        island(lockM, height: 430, name: "expanded_lock.png", in: dir)

        let setupM = makeModel()
        setupM.expanded = true
        setupM.lock.fakeForSnapshot(hasPassword: false, unlocked: false)
        setupM.beginFlow(.setup(confirming: false))
        island(setupM, height: 430, name: "expanded_setup.png", in: dir)

        let lost = makeModel()
        lost.connected = false
        lost.status = .notProtecting
        lost.masks = 0
        lost.expanded = true
        lost.prefs.character = "boo"
        island(lost, height: 430, name: "expanded_not_protecting.png", in: dir)

        let unlocked = makeModel()
        unlocked.expanded = true
        unlocked.mode = .medium
        unlocked.engineMood = .covering
        unlocked.prefs.character = "ember"
        unlocked.lock.fakeForSnapshot(hasPassword: true, unlocked: true)
        island(unlocked, height: 430, name: "expanded_medium_unlocked.png", in: dir)

        let pill = makeModel(notch: false)
        pill.expanded = true
        pill.prefs.character = "nova"
        island(pill, height: 430, name: "expanded_pill.png", in: dir)

        // Characters.
        SnapshotKit.save(CharacterGrid(), size: CGSize(width: 900, height: 820), to: dir.appendingPathComponent("character_grid.png"))
        SnapshotKit.save(VariantSheet(), size: CGSize(width: 1010, height: CGFloat(Catalog.all.count / 2) * 140 + 40),
                         to: dir.appendingPathComponent("variants.png"), scale: 1.5)
        let sheetH = CGFloat(Catalog.all.count) * 122 + 70
        SnapshotKit.save(MoodSheet(), size: CGSize(width: 1060, height: sheetH), to: dir.appendingPathComponent("all_moods.png"), scale: 1)
        for c in Catalog.all {
            for mood in Mood.allCases {
                SnapshotKit.save(CharacterCanvas(spec: c, frame: .still(mood)), size: CGSize(width: 256, height: 256),
                                 to: dir.appendingPathComponent("characters/\(c.id)/\(mood.rawValue).png"), scale: 1)
            }
        }
        MotionPreviews.run(into: dir)
        print("snapshots written to \(dir.path)")
    }
}

/// A slice of a Mac desktop: wallpaper, menu bar and (optionally) the notch.
struct DesktopScene<Content: View>: View {
    var notch: Bool
    var height: CGFloat
    @ViewBuilder var content: () -> Content

    var body: some View {
        ZStack(alignment: .top) {
            LinearGradient(colors: [Color(red: 0.20, green: 0.27, blue: 0.47), Color(red: 0.47, green: 0.35, blue: 0.55),
                                    Color(red: 0.85, green: 0.57, blue: 0.48)],
                           startPoint: .topLeading, endPoint: .bottomTrailing)
            VStack(spacing: 0) {
                HStack(spacing: 18) {
                    Circle().fill(Color.white.opacity(0.85)).frame(width: 11, height: 11)
                    ForEach([46.0, 30, 34, 40, 52], id: \.self) { w in
                        Capsule().fill(Color.white.opacity(0.55)).frame(width: w, height: 7)
                    }
                    Spacer()
                    ForEach([16.0, 16, 22, 60], id: \.self) { w in
                        Capsule().fill(Color.white.opacity(0.55)).frame(width: w, height: 7)
                    }
                }
                .padding(.horizontal, 16)
                .frame(height: notch ? 32 : 24)
                .background(Color.black.opacity(0.22))
                Spacer()
            }
            if notch {
                UnevenRoundedRectangle(bottomLeadingRadius: 10, bottomTrailingRadius: 10)
                    .fill(Color.black).frame(width: 185, height: 32)
            }
            content()
        }
        .frame(width: IslandLayout.panelSize.width, height: height, alignment: .top)
        .clipped()
    }
}

/// The four status looks of the collapsed island.
@MainActor
struct CollapsedStates: View {
    var body: some View {
        let rows: [(String, IslandModel)] = [
            ("Protecting (quiet)", Snapshots.makeModel()),
            ("Covering", { let m = Snapshots.makeModel(); m.engineMood = .covering; m.prefs.character = "bun"; return m }()),
            ("Unsure (checking)", { let m = Snapshots.makeModel(); m.engineMood = .unsure; m.masks = 1; m.prefs.character = "ember"; return m }()),
            ("Not protecting (engine absent)", { let m = Snapshots.makeModel(); m.connected = false; m.prefs.character = "boo"; return m }()),
            ("Paused", { let m = Snapshots.makeModel(); m.status = .paused; m.masks = 0; m.prefs.character = "waddle"; return m }()),
        ]
        VStack(spacing: 0) {
            ForEach(rows.indices, id: \.self) { i in
                ZStack(alignment: .bottomLeading) {
                    DesktopScene(notch: true, height: 82) { IslandRootView(model: rows[i].1) }
                    Text(rows[i].0).font(Theme.font(12, .semibold)).foregroundColor(.white.opacity(0.9)).padding(10)
                }
            }
        }
    }
}

struct CharacterGrid: View {
    var body: some View {
        let cols = 5
        let rows = stride(from: 0, to: Catalog.all.count, by: cols).map {
            Array(Catalog.all[$0..<min($0 + cols, Catalog.all.count)])
        }
        VStack(spacing: 22) {
            ForEach(rows.indices, id: \.self) { r in
                HStack(spacing: 26) {
                    ForEach(rows[r]) { c in
                        VStack(spacing: 8) {
                            CharacterCanvas(spec: c, frame: .still(.idle)).frame(width: 140, height: 140)
                                .shadow(color: .black.opacity(0.3), radius: 8, y: 5)
                            Text(c.name).font(Theme.font(14, .semibold)).foregroundColor(.white.opacity(0.85))
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(red: 0.95, green: 0.94, blue: 0.92).opacity(0.0))
        .background(Color(red: 0.09, green: 0.09, blue: 0.11))
    }
}

struct VariantSheet: View {
    var body: some View {
        let pairs = stride(from: 0, to: Catalog.all.count, by: 2).map {
            Array(Catalog.all[$0..<min($0 + 2, Catalog.all.count)])
        }
        VStack(alignment: .leading, spacing: 12) {
            ForEach(pairs.indices, id: \.self) { r in
                HStack(spacing: 30) {
                    ForEach(pairs[r]) { c in
                        HStack(spacing: 10) {
                            Text(c.name).font(Theme.font(13, .semibold)).foregroundColor(.white.opacity(0.8)).frame(width: 64, alignment: .leading)
                            ForEach(c.variants.indices, id: \.self) { v in
                                CharacterCanvas(spec: c, variant: v, frame: .still(.idle)).frame(width: 116, height: 116)
                            }
                        }
                    }
                }
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Color(red: 0.09, green: 0.09, blue: 0.11))
    }
}
