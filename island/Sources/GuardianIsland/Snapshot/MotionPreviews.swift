import SwiftUI

/// Frame strips of the real motion: the island's open spring, mood blends from the
/// live RigDriver, blinking, breathing and gaze.
@MainActor
enum MotionPreviews {
    /// The same spring as Theme.openSpring (response 0.48, damping 0.74), from 0 to 1.
    static func spring(_ t: Double, response: Double = 0.48, damping: Double = 0.74) -> Double {
        let w = 2 * Double.pi / response
        let wd = w * (1 - damping * damping).squareRoot()
        return 1 - exp(-damping * w * t) * (cos(wd * t) + damping * w / wd * sin(wd * t))
    }

    static func run(into dir: URL) {
        let m = dir.appendingPathComponent("motion")

        // (a) collapse -> expand spring, 8 evenly spaced frames over 0.56 s.
        let times = (0..<8).map { Double($0) * 0.08 }
        var frames: [(String, IslandModel)] = []
        for (i, t) in times.enumerated() {
            let model = Snapshots.makeModel()
            let p = spring(t)
            model.previewProgress = CGFloat(p)
            model.expanded = p > 0.5
            frames.append((String(format: "t = %.2fs   progress %.2f", t, p), model))
            Snapshots.island(model, height: 380, name: "motion/expand_\(i).png", in: dir)
        }
        SnapshotKit.save(ExpandSheet(frames: frames), size: CGSize(width: 1660, height: 4 * 404 + 70),
                         to: m.appendingPathComponent("expand_spring_sheet.png"), scale: 1)

        // (b) mood transitions from the live driver, (c) blink, breathing, gaze.
        SnapshotKit.save(MoodMotionSheet(), size: CGSize(width: 14 * 118 + 40, height: 6 * 168 + 60),
                         to: m.appendingPathComponent("mood_blink_breath_sheet.png"), scale: 1.5)
    }
}

struct ExpandSheet: View {
    var frames: [(String, IslandModel)]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Collapse → expand (open spring: response 0.48 s, damping 0.74), 8 frames, 80 ms apart")
                .font(Theme.font(16, .bold)).foregroundColor(.white)
            ForEach(0..<4, id: \.self) { r in
                HStack(spacing: 10) {
                    ForEach(0..<2, id: \.self) { c in
                        let f = frames[r * 2 + c]
                        ZStack(alignment: .bottomLeading) {
                            DesktopScene(notch: true, height: 380) { IslandRootView(model: f.1) }
                                .frame(width: 820, height: 380)
                            Text("\(r * 2 + c + 1).  \(f.0)").font(Theme.font(13, .semibold).monospacedDigit())
                                .foregroundColor(.white).padding(8)
                                .background(Capsule().fill(Color.black.opacity(0.45))).padding(8)
                        }
                    }
                }
            }
        }
        .padding(15)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Color(red: 0.09, green: 0.09, blue: 0.11))
    }
}

/// Runs a RigDriver through a mood script at 120 Hz and samples it.
@MainActor
func sampleDriver(_ script: [(Double, Mood)], at samples: [Double], mouse: ((Double) -> CGPoint?)? = nil) -> [RigFrame] {
    let d = RigDriver()
    var out: [RigFrame] = []
    var t = 0.0
    var next = 0
    while next < samples.count {
        let mood = script.last { $0.0 <= t }?.1 ?? script[0].1
        let f = d.frame(at: t, mood: mood, reduceMotion: false, mouse: mouse?(t), poke: 0)
        if t + 1e-9 >= samples[next] { out.append(f); next += 1 }
        t += 1.0 / 120.0
    }
    return out
}

struct MoodMotionSheet: View {
    struct Row { var title: String; var spec: CharacterSpec; var frames: [RigFrame]; var labels: [String] }

    var rows: [Row] {
        let s1: [(Double, Mood)] = [(0, .idle), (0.5, .alert), (1.7, .covering)]
        let t1 = [0.3, 0.55, 0.62, 0.7, 0.8, 0.95, 1.2, 1.5, 1.75, 1.82, 1.9, 2.0, 2.2, 2.7]
        let s2: [(Double, Mood)] = [(0, .watching), (0.4, .unsure), (1.5, .lost), (2.4, .sleep)]
        let t2 = [0.3, 0.45, 0.55, 0.7, 1.0, 1.45, 1.55, 1.65, 1.8, 2.3, 2.45, 2.6, 2.8, 3.4]
        func label(_ s: [(Double, Mood)], _ t: [Double]) -> [String] {
            t.map { x in String(format: "%.2fs ", x) + (s.last { $0.0 <= x }?.1.label ?? "") }
        }
        // Blink: the driver's 0.14 s sine, sampled every 20 ms.
        let blinkFrames: [RigFrame] = (0..<14).map { i in
            var f = RigFrame.still(.idle); f.gazeX = 0; f.gazeY = 0
            let ms = Double(i) * 0.02 - 0.04
            f.blink = (ms >= 0 && ms <= 0.14) ? sin(.pi * ms / 0.14) : 0
            return f
        }
        // Breathing: one 3.6 s breath; the guide line marks the top at rest.
        let breathFrames: [RigFrame] = (0..<14).map { i in
            var f = RigFrame.still(.idle); f.gazeX = 0; f.gazeY = 0
            f.breathe = sin(2 * .pi * Double(i) / 13); f.t = Double(i) * 3.6 / 13
            f.sway = 0.021 * sin(2 * .pi * f.t / 5.3)
            return f
        }
        let gazeTimes = (0..<14).map { Double($0) * 0.12 + 0.2 }
        let gaze = sampleDriver([(0, .idle)], at: gazeTimes) { t in
            CGPoint(x: cos(t * 3.2), y: sin(t * 3.2) * 0.6)
        }
        let work = sampleDriver([(0, .working)], at: (0..<14).map { Double($0) * 0.25 + 0.3 })
        return [
            Row(title: "Idle → Alert (hop) → Covering: live RigDriver springs, 120 Hz", spec: Catalog.pip,
                frames: sampleDriver(s1, at: t1), labels: label(s1, t1)),
            Row(title: "Watching → Unsure (tilt + ?) → Lost → Sleep", spec: Catalog.ember,
                frames: sampleDriver(s2, at: t2), labels: label(s2, t2)),
            Row(title: "Blink (0.14 s, sampled every 20 ms)", spec: Catalog.bun, frames: blinkFrames,
                labels: (0..<14).map { "\(Int(Double($0) * 20 - 40)) ms" }),
            Row(title: "Breathing (~2 % scale, 3.6 s) + sway. Line = resting top", spec: Catalog.nova, frames: breathFrames,
                labels: breathFrames.map { String(format: "%.1fs", $0.t) }),
            Row(title: "Gaze following the mouse (expanded panel)", spec: Catalog.miso, frames: gaze,
                labels: gazeTimes.map { String(format: "%.2fs", $0) }),
            Row(title: "Working: looking around", spec: Catalog.toasty, frames: work,
                labels: (0..<14).map { String(format: "%.2fs", Double($0) * 0.25 + 0.3) }),
        ]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            ForEach(rows.indices, id: \.self) { r in
                let row = rows[r]
                VStack(alignment: .leading, spacing: 5) {
                    Text(row.title).font(Theme.font(13, .bold)).foregroundColor(.white.opacity(0.9))
                    HStack(spacing: 8) {
                        ForEach(row.frames.indices, id: \.self) { i in
                            VStack(spacing: 3) {
                                ZStack(alignment: .top) {
                                    CharacterCanvas(spec: row.spec, frame: row.frames[i]).frame(width: 110, height: 110)
                                    if r == 3 {
                                        Rectangle().fill(Color.white.opacity(0.6)).frame(width: 110, height: 0.5)
                                            .offset(y: 110 * 0.188)
                                    }
                                }
                                Text(row.labels[i]).font(Theme.font(9.5, .semibold).monospacedDigit())
                                    .foregroundColor(.white.opacity(0.6)).lineLimit(1)
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
