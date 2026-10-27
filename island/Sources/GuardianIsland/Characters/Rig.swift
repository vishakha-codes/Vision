import Foundation
import CoreGraphics

/// Moods the engine drives (island-protocol.md, `mood`).
enum Mood: String, CaseIterable, Codable {
    case idle, watching, working, alert, covering, unsure, lost, sleep

    var label: String {
        switch self {
        case .idle: return "Idle"
        case .watching: return "Watching"
        case .working: return "Working"
        case .alert: return "Alert"
        case .covering: return "Covering"
        case .unsure: return "Unsure"
        case .lost: return "Lost"
        case .sleep: return "Sleep"
        }
    }
}

/// The shared pose every character is drawn from. All fields are springs, so a mood
/// change mid-transition simply retargets from wherever the face currently is.
struct Pose {
    var eyeOpen = 1.0      // 1 = open dots, 0 = closed arcs
    var eyeScale = 1.0
    var gazeX = 0.0        // -1 ... 1
    var gazeY = 0.0
    var smile = 0.55       // -1 frown ... 1 grin
    var mouthOpen = 0.0
    var tilt = 0.0         // radians
    var hop = 0.0          // amplitude of the alert hop
    var cover = 0.0        // 0 ... 1 paws / shield raised
    var worry = 0.0
    var question = 0.0
    var sleepy = 0.0
    var squash = 0.0       // poke impulse
    var scan = 0.0         // watching: slow tracking sweep
    var look = 0.0         // working: looking around
    var blush = 0.0

    static let keys: [WritableKeyPath<Pose, Double>] = [
        \.eyeOpen, \.eyeScale, \.gazeX, \.gazeY, \.smile, \.mouthOpen, \.tilt, \.hop,
        \.cover, \.worry, \.question, \.sleepy, \.squash, \.scan, \.look, \.blush,
    ]

    static var zero: Pose {
        var p = Pose()
        for k in keys { p[keyPath: k] = 0 }
        return p
    }

    static func target(_ mood: Mood) -> Pose {
        var p = Pose()
        switch mood {
        case .idle:
            p.smile = 0.6
        case .watching:
            p.smile = 0.35; p.scan = 1; p.eyeScale = 1.05
        case .working:
            p.smile = 0.05; p.look = 1; p.gazeY = -0.15
        case .alert:
            p.eyeScale = 1.55; p.mouthOpen = 0.85; p.smile = 0; p.hop = 1
        case .covering:
            p.cover = 1; p.eyeOpen = 0; p.smile = 0.15; p.blush = 0.5
        case .unsure:
            p.tilt = 0.16; p.question = 1; p.smile = -0.12; p.gazeX = 0.45; p.gazeY = -0.55
        case .lost:
            p.worry = 1; p.smile = -0.6; p.eyeScale = 1.12; p.tilt = -0.04; p.gazeY = 0.2
        case .sleep:
            p.eyeOpen = 0; p.sleepy = 1; p.smile = 0.2; p.tilt = 0.05; p.gazeY = 0.3
        }
        return p
    }
}

/// Everything the drawing code needs for one frame.
struct RigFrame {
    var pose: Pose
    var t: Double
    var blink: Double        // 0 open ... 1 shut
    var breathe: Double      // -1 ... 1
    var sway: Double         // radians
    var hopY: Double         // unit offset (negative = up)
    var gazeX: Double
    var gazeY: Double
    var motion: Double       // 0 with Reduce Motion, 1 otherwise

    /// A representative, deterministic frame for snapshots.
    static func still(_ mood: Mood, t: Double = 0.9) -> RigFrame {
        let p = Pose.target(mood)
        var gx = p.gazeX, gy = p.gazeY
        if mood == .watching { gx = 0.55 }
        if mood == .working { gx = -0.6; gy = -0.35 }
        return RigFrame(pose: p, t: t, blink: 0, breathe: 0.4, sway: 0,
                        hopY: mood == .alert ? -0.03 : 0, gazeX: gx, gazeY: gy, motion: 1)
    }
}

/// Integrates the pose springs plus the idle layers (breathing, blinking, sway, gaze).
/// One driver per on-screen character; it lives in a reference type so the Canvas
/// closure can advance it each frame without invalidating SwiftUI state.
final class RigDriver {
    private(set) var pose = Pose.target(.idle)
    private var vel = Pose.zero
    private var target = Pose.target(.idle)
    private var mood: Mood = .idle
    private var moodStart: Double = 0
    private var last: Double?

    private var nextBlink: Double = 0
    private var blinkStart: Double = -10
    private var secondBlink = false

    private var lookX = 0.0, lookY = 0.0
    private var lookSlot = -1
    private var lookTargetX = 0.0, lookTargetY = 0.0
    private var mouseX = 0.0, mouseY = 0.0
    private var pokeSeen = 0
    private var rng = SystemRandomNumberGenerator()

    func frame(at t: Double, mood newMood: Mood, reduceMotion: Bool,
               mouse: CGPoint?, poke: Int) -> RigFrame {
        let dt = min(max(t - (last ?? t), 0), 1.0 / 20.0)
        if last == nil {
            pose = Pose.target(newMood); target = pose; mood = newMood
            nextBlink = t + Double.random(in: 1.5...4, using: &rng)
            pokeSeen = poke
        }
        last = t

        if newMood != mood {
            mood = newMood
            moodStart = t
            target = Pose.target(newMood)
            if newMood == .alert && !reduceMotion { vel.squash += 4 }
        }
        if poke != pokeSeen {
            pokeSeen = poke
            if !reduceMotion { vel.squash -= 7; vel.smile += 4 }
        }

        // Springs: slightly under-damped so moods blend with a little life.
        let response = reduceMotion ? 0.18 : 0.42
        let damping = reduceMotion ? 1.0 : 0.66
        let k = pow(2 * Double.pi / response, 2)
        let c = 4 * Double.pi * damping / response
        var remaining = dt
        while remaining > 0 {
            let h = min(remaining, 1.0 / 120.0)
            for key in Pose.keys {
                let x = pose[keyPath: key]
                var v = vel[keyPath: key]
                let goal = key == \Pose.squash ? 0 : target[keyPath: key]
                v += (k * (goal - x) - c * v) * h
                vel[keyPath: key] = v
                pose[keyPath: key] = x + v * h
            }
            remaining -= h
        }

        let motion = reduceMotion ? 0.0 : 1.0

        // Blink every 2-6 s, sometimes twice.
        var blink = 0.0
        let closed = pose.eyeOpen < 0.4 || pose.cover > 0.5
        if t >= nextBlink && !closed {
            blinkStart = t
            secondBlink = Double.random(in: 0...1, using: &rng) < 0.22
            nextBlink = t + Double.random(in: 2...6, using: &rng)
        }
        let bd = 0.14
        let since = t - blinkStart
        if since < bd {
            blink = sin(.pi * since / bd)
        } else if secondBlink && since > bd + 0.08 && since < 2 * bd + 0.08 {
            blink = sin(.pi * (since - bd - 0.08) / bd)
        }

        // Breathing (about 2 % scale) and sway; slower and deeper asleep.
        let breathPeriod = 3.6 + 1.6 * pose.sleepy
        let breathe = sin(2 * .pi * t / breathPeriod) * motion
        let sway = (0.021 * sin(2 * .pi * t / 5.3) + 0.006 * sin(2 * .pi * t / 2.1)) * motion
            * (1 - 0.6 * pose.sleepy)

        // Alert hop: short hops with a rest between them.
        var hopY = 0.0
        if pose.hop > 0.02 && motion > 0 {
            let ph = (t - moodStart).truncatingRemainder(dividingBy: 1.15)
            if ph < 0.34 { hopY = -sin(.pi * ph / 0.34) * 0.045 * pose.hop }
        }

        // Gaze layers.
        var gx = pose.gazeX
        var gy = pose.gazeY
        gx += pose.scan * 0.75 * sin(t * 0.55) * motion
        gy += pose.scan * 0.12 * sin(t * 0.31) * motion
        if pose.look > 0.02 {
            let slot = Int(floor(t / 0.85))
            if slot != lookSlot {
                lookSlot = slot
                lookTargetX = Double.random(in: -1...1, using: &rng)
                lookTargetY = Double.random(in: -0.7...0.3, using: &rng)
            }
            let a = min(1, dt * 14)
            lookX += (lookTargetX - lookX) * a
            lookY += (lookTargetY - lookY) * a
            gx += pose.look * lookX * motion
            gy += pose.look * lookY * motion
        }
        if let m = mouse, mood == .idle || mood == .watching || mood == .alert {
            let a = min(1, dt * 9)
            mouseX += (Double(m.x) - mouseX) * a
            mouseY += (Double(m.y) - mouseY) * a
            gx = mouseX
            gy = mouseY
        }

        return RigFrame(pose: pose, t: t, blink: blink, breathe: breathe, sway: sway,
                        hopY: hopY, gazeX: max(-1.2, min(1.2, gx)), gazeY: max(-1, min(1, gy)),
                        motion: motion)
    }
}
