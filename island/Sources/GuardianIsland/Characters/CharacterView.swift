import SwiftUI

let rigEpoch = Date().timeIntervalSinceReferenceDate

/// A single still frame of a character (snapshots, grid tiles).
struct CharacterCanvas: View {
    let spec: CharacterSpec
    var variant: Int = 0
    var frame: RigFrame = .still(.idle)
    var zoom: CGFloat = 1
    var corner: CGFloat = 0.235
    var overlays = true

    var body: some View {
        Canvas { ctx, size in
            CharacterRenderer.draw(ctx, size: size, spec: spec, palette: spec.palette(variant), frame: frame,
                                   zoom: zoom, corner: corner, overlays: overlays)
        }
    }
}

/// Mouse gaze shared with the window controller without invalidating SwiftUI:
/// the rig reads it every frame. `nil` means "no mouse following".
final class GazeSource {
    var point: CGPoint?
}

/// The animated character: TimelineView drives the shared RigDriver each frame.
struct LiveCharacter: View, Animatable {
    let spec: CharacterSpec
    var variant: Int = 0
    var mood: Mood
    var reduceMotion = false
    var gaze: GazeSource?
    var poke = 0
    var zoom: CGFloat = 1
    var corner: CGFloat = 0.235
    var overlays = true
    var fps: Double = 60
    var paused = false

    var animatableData: AnimatablePair<CGFloat, CGFloat> {
        get { AnimatablePair(zoom, corner) }
        set { zoom = newValue.first; corner = newValue.second }
    }

    @State private var driver = RigDriver()

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / (reduceMotion ? min(fps, 20) : fps), paused: paused)) { tl in
            Canvas { ctx, size in
                let t = tl.date.timeIntervalSinceReferenceDate - rigEpoch
                let fr = driver.frame(at: t, mood: mood, reduceMotion: reduceMotion,
                                      mouse: gaze?.point, poke: poke)
                CharacterRenderer.draw(ctx, size: size, spec: spec, palette: spec.palette(variant), frame: fr,
                                       zoom: zoom, corner: corner, overlays: overlays)
            }
        }
    }
}
