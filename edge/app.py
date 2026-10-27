"""Laptop reflex: replay fixture or live desktop protection.

Replay (judged / eval path)::

    python -m edge.app --policy balanced --source replay --detect synthetic

Live laptop screen, new threaded engine (``edge.live_engine``)::

    python -m edge.app --source live [--mode aggressive|medium|off]

Capture, vision (``vision_core.live`` analyzer) and the overlay run on
separate threads; masks follow scrolling every frame; one click-through
overlay window; the SwiftUI island talks over ``edge.island_bridge``.
An explicit ``--mode`` wins over the mode saved in ``guardian.json`` (it warns
and saves); without it the saved mode applies.

Old single-thread reflex (eyes local, OpenCV brain on Graviton)::

    python -m edge.app --source live --engine legacy --detect nsfw --backend http://<box-ip>:8080

``--engine legacy --source live`` captures the primary display. ``--detect nsfw`` runs the
bundled NudeNet ONNX detector — not HSV, not a fake score. Masks are
drawn as always-on-top macOS windows in screen space. The OpenCV HUD is
optional (``--preview`` / ``--hud``) and is not the protection surface.

This is not zero-exposure: capture happens after pixels are visible, so
the first frame of a page can appear before a mask. Quit with q / Esc
(HUD or terminal) or Ctrl-C; overlay windows are torn down.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import tty
from pathlib import Path

import cv2

from edge.capture import (
    PERMISSION_HELP,
    CaptureAdapter,
    open_source,
    preferred_live_backend,
    screen_recording_granted,
)
from edge.gomi.prefs import load_prefs
from edge.gomi_boot import (
    add_cli_flags,
    chain_mask_listeners,
    front_gomi,
    island_backend_name,
    pump_gomi,
    start_gomi,
    stop_gomi,
    window_ids as gomi_window_ids,
)
from edge.overlay import close_window, render_overlay, show, try_open_window
from edge.http_brain import resolve_backend
from edge.reflex import Reflex, ReflexState
from edge.screen_overlay import ScreenOverlay
from edge.tick_log import TickLogger
from vision_core.contracts import Frame
from vision_core.detect_nsfw import last_detect_stats, load_nudenet, nudenet_available

_VISION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREVIEW = _VISION_ROOT / "experiments" / "edge_replay_preview.mp4"
POLICY_KEYS = {
    ord("1"): "strict",
    ord("s"): "strict",
    ord("S"): "strict",
    ord("2"): "balanced",
    ord("b"): "balanced",
    ord("B"): "balanced",
    ord("3"): "relaxed",
    ord("r"): "relaxed",
    ord("R"): "relaxed",
}


def default_detect(source: str) -> str:
    return "nsfw" if (source or "").strip().lower() == "live" else "synthetic"


def _apply_key(key: int, reflex: Reflex, paused: bool) -> tuple[bool, bool]:
    """Handle a key. Returns (paused, quit)."""
    if key in (ord("q"), ord("Q"), 27):
        return paused, True
    if key == ord(" "):
        return (not paused), False
    policy = POLICY_KEYS.get(key)
    if policy is not None:
        reflex.set_policy(policy)
    return paused, False


def _open_writer(path: Path, canvas, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = canvas.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open VideoWriter for {path}")
    return writer


def _cycle_policy(index: int, period: int) -> str | None:
    if period <= 0:
        return None
    slot = (index // period) % 3
    return ("strict", "balanced", "relaxed")[slot]


class _RawStdin:
    """Single-key reads without Enter, restored on exit."""

    def __init__(self) -> None:
        self.fd: int | None = None
        self.old = None

    def __enter__(self) -> _RawStdin:
        if not sys.stdin.isatty():
            return self
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.fd is not None and self.old is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            except Exception:
                pass

    def poll_key(self) -> int | None:
        if self.fd is None:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        ch = sys.stdin.read(1)
        if not ch:
            return None
        if ch == "\x1b":
            return 27
        return ord(ch)


def _tick_record(
    *,
    tick: int,
    source_key: str,
    detect_mode: str,
    cap,
    last_frame: Frame | None,
    state: ReflexState,
    reflex: Reflex,
    overlay: ScreenOverlay | None,
    gomi,
) -> dict:
    stats = last_detect_stats()
    result = reflex.last_result
    orch = None if result is None else result.final_decision
    if orch is None:
        orch = state.actions[0].action.value if state.actions else "NO_ACTION"
    shape = getattr(cap, "last_shape", None)
    if shape is None and last_frame is not None:
        shape = last_frame.image_bgr.shape
    gomi_state = "-"
    gomi_exp: float | bool = False
    if gomi is not None:
        gomi_state = getattr(getattr(gomi, "brain", None), "state", None)
        gomi_state = getattr(gomi_state, "value", gomi_state) or "-"
        gomi_exp = bool(getattr(gomi, "pinned", False)) or float(getattr(gomi, "expand", 0.0)) >= 0.5
    overlay_n = 0
    if overlay is not None and overlay.available:
        overlay_n = overlay.mask_count()
    return {
        "tick": tick,
        "frame_id": state.frame_id,
        "source": source_key,
        "detect": detect_mode,
        "backend": state.backend,
        "capture_ok": last_frame is not None and bool(getattr(cap, "last_ok", True)),
        "capture_reason": getattr(cap, "reason", None) or state.backend_error,
        "capture_backend": getattr(cap, "backend", "-"),
        "frame_shape": shape,
        "mean_pixel": getattr(cap, "last_mean", None),
        "granted": getattr(cap, "granted", None),
        "detect_called": not state.held,
        "model_loaded": stats.model_loaded,
        "n_raw": stats.n_raw,
        "top": stats.top,
        "max_score": stats.max_score,
        "threshold": stats.threshold,
        "n_after_policy": stats.n_after_policy,
        "n_clusters": stats.n_clusters,
        "cluster_cap": stats.cluster_cap,
        "clusters_capped": stats.clusters_capped,
        "n_tiles": stats.n_tiles,
        "payload_bytes": getattr(reflex, "last_payload_bytes", 0),
        "detect_reason": (
            stats.reason
            if detect_mode == "nsfw"
            else f"synthetic HSV; orch={orch}"
        ),
        "orch_action": orch,
        "overlay_n": overlay_n,
        "gomi_state": gomi_state,
        "gomi_expanded": gomi_exp,
    }


def run(
    *,
    policy: str = "balanced",
    source: str = "replay",
    backend: str = "auto",
    max_frames: int | None = None,
    preview: Path | None = None,
    headless: bool = False,
    cycle_policies: bool = False,
    capture: CaptureAdapter | None = None,
    detect: str | None = None,
    hud: bool = False,
    screen_overlay: bool | None = None,
    reinspect: str = "local",
    companion: bool = True,
    on_mask_event=None,
    verbose: bool = False,
    character: str | None = None,
) -> dict:
    """Drive the reflex. Replay may write a preview mp4; live never persists frames."""
    source_key = (source or "replay").strip().lower()
    detect_mode = (detect or default_detect(source_key)).strip().lower()
    use_overlay = bool(screen_overlay) if screen_overlay is not None else source_key == "live"
    owns_capture = capture is None
    cap = capture if capture is not None else open_source(source_key)

    if source_key == "live" and not getattr(cap, "available", False):
        reason = getattr(cap, "reason", None) or PERMISSION_HELP
        print(reason, file=sys.stderr)
        if owns_capture:
            cap.release()
        return {
            "frames": 0,
            "display": False,
            "preview": None,
            "log": None,
            "policy": policy,
            "status": "NOT_PROTECTING",
            "mood": "lost",
            "backend": "local",
            "backend_url": None,
            "source": source_key,
            "detect": detect_mode,
            "overlay": False,
            "error": "capture unavailable",
            "opencv": cv2.__version__,
        }

    backend = resolve_backend(backend, source=source_key, detect=detect_mode)
    if detect_mode == "nsfw":
        if nudenet_available():
            print("loading NudeNet 320n on the laptop (candidates + local fallback)...", flush=True)
            load_nudenet()
            print("laptop detector ready", flush=True)
        elif backend == "local":
            raise SystemExit(
                "nudenet weights are missing. Install into vision/.venv:\n"
                "  python -m pip install nudenet onnxruntime"
            )
        else:
            print(
                "laptop NudeNet missing; cloud /v1/live_perceive will detect if installed",
                file=sys.stderr,
            )
        if backend != "local":
            print(
                f"live nsfw brain → {backend} (JPEG tiles / downscaled frame, not raw desktop)",
                flush=True,
            )

    fixture_id = getattr(cap, "fixture_id", None) or (
        "live_desktop" if source_key == "live" else "synthetic_moving_regions"
    )
    cap_path = getattr(cap, "path", None)
    fixture_path = None
    if cap_path is not None:
        try:
            rel = Path(cap_path).resolve().relative_to(_VISION_ROOT)
            fixture_path = str(rel)
        except ValueError:
            fixture_path = None

    env_headless = headless or os.environ.get("SCREEN_GUARDIAN_HEADLESS") == "1"
    paused = False
    display = False
    # Open the HUD first so OpenCV HighGUI owns NSApp. Creating the Gomi
    # pill before namedWindow orphaned the always-on-top window on replay.
    want_hud = hud or (
        source_key == "replay"
        and not headless
        and preview is None
        and os.environ.get("SCREEN_GUARDIAN_HEADLESS") != "1"
    )
    if want_hud:
        display = try_open_window()

    gomi = start_gomi(
        enabled=companion,
        headless=env_headless,
        character=character,
        policy=policy,
        detect=detect_mode,
    )
    on_mask_event = chain_mask_listeners(on_mask_event, None if gomi is None else gomi.on_state)
    if gomi is not None and not env_headless:
        front_gomi(gomi)

    reflex = Reflex(
        policy,
        backend=backend,
        fixture_id=str(fixture_id),
        fixture_path=fixture_path,
        detect_mode=detect_mode,
        reinspect_mode=reinspect,
        on_mask_event=on_mask_event,
    )
    if gomi is not None:
        def _island_setting(kind: str, value: str) -> str | None:
            if kind == "policy":
                reflex.set_policy(value)
                return None
            if kind == "detect":
                return reflex.set_detect(value)
            return None

        gomi.on_setting = _island_setting
        gomi.policy = reflex.policy_name
        gomi.detect = reflex.detect_mode

    writer: cv2.VideoWriter | None = None
    preview_path = preview
    if source_key == "live" and preview_path is not None:
        print(
            "refusing to write a live-desktop preview mp4 (raw capture persist)",
            file=sys.stderr,
        )
        preview_path = None
    if source_key == "replay" and not display:
        preview_path = preview_path or DEFAULT_PREVIEW
        if cycle_policies or preview is None:
            cycle_policies = True
        if max_frames is None:
            max_frames = 90

    written = 0
    last_state: ReflexState | None = None
    last_frame: Frame | None = None
    fps = float(getattr(cap, "fps", 12.0 if source_key == "live" else 30.0) or 12.0)
    log_rows: list[dict] = []
    overlay: ScreenOverlay | None = None
    if use_overlay:
        overlay = ScreenOverlay()
        if overlay.error:
            print(overlay.error, file=sys.stderr)
        elif overlay.available:
            print(
                "on-screen overlay is active (frosted windows on detected regions; "
                "the rest of the desktop stays clickable). q / Esc / Ctrl-C quits.",
                flush=True,
            )
    if source_key == "live":
        granted = getattr(cap, "granted", None)
        if granted is None:
            granted = screen_recording_granted()
        backend_name = getattr(cap, "backend", None) or "none"
        if backend_name in ("", "none", "-") and getattr(cap, "available", False):
            backend_name = preferred_live_backend()
        print(
            f"screen recording granted={granted} "
            f"capture_backend={backend_name} "
            f"(quartz preferred, mss fallback; black frame = deny)",
            file=sys.stderr,
            flush=True,
        )

    if gomi is not None and not env_headless:
        gomi_ids = gomi_window_ids(gomi)
        backend_name = island_backend_name(gomi)
        if gomi_ids:
            print(
                f"Gomi companion island is on (always-on-top pill, "
                f"{backend_name} id={gomi_ids[0]}). "
                "It is a separate window at the top of the display — not the "
                "tiny HUD sprite inside the OpenCV fixture window. "
                "Click the pill to expand or collapse; drag to move. "
                "--no-companion hides it.",
                flush=True,
            )
        else:
            print(
                f"Gomi island did not create a window ({backend_name}); "
                "continuing without the pill.",
                file=sys.stderr,
                flush=True,
            )

    tick_logger: TickLogger | None = None
    if verbose or source_key == "live":
        tick_logger = TickLogger(every_s=0.0 if verbose else 1.0)
    gomi_fronted = False

    stdin = _RawStdin()
    try:
        stdin.__enter__()
        while True:
            ids = []
            if overlay is not None and overlay.available:
                ids.extend(overlay.window_ids())
            ids.extend(gomi_window_ids(gomi))
            if ids and hasattr(cap, "exclude_window_ids"):
                cap.exclude_window_ids = ids

            if cycle_policies and max_frames:
                period = max(1, max_frames // 3)
                nxt = _cycle_policy(written, period)
                if nxt is not None and nxt != reflex.policy_name:
                    reflex.set_policy(nxt)

            if not paused:
                last_frame = cap.read()
            state = reflex.step(last_frame, paused=paused)
            last_state = state
            if overlay is not None and overlay.available:
                overlay.sync(state, getattr(cap, "geom", None))
                overlay.pump()
            pump_gomi(gomi)

            if display:
                canvas = render_overlay(state)
                key = show(state, canvas, delay_ms=max(1, int(1000.0 / fps)))
                if key < 0:
                    display = False
                    if source_key == "replay":
                        preview_path = preview_path or DEFAULT_PREVIEW
                        if max_frames is None:
                            max_frames = 90
                        cycle_policies = True
                else:
                    paused, should_quit = _apply_key(key, reflex, paused)
                    if should_quit:
                        break
            else:
                if source_key == "replay":
                    canvas = render_overlay(state)
                    if writer is None and preview_path is not None:
                        writer = _open_writer(Path(preview_path), canvas, fps)
                    if writer is not None:
                        writer.write(canvas)
                # live: LiveCapture.read already rate-limits; no extra sleep
                key = stdin.poll_key()
                if key is not None:
                    paused, should_quit = _apply_key(key, reflex, paused)
                    if should_quit:
                        break

            written += 1
            if display and gomi is not None and not gomi_fronted:
                front_gomi(gomi)
                gomi_fronted = True
            if tick_logger is not None:
                tick_logger.emit(
                    _tick_record(
                        tick=written,
                        source_key=source_key,
                        detect_mode=detect_mode,
                        cap=cap,
                        last_frame=last_frame,
                        state=state,
                        reflex=reflex,
                        overlay=overlay,
                        gomi=gomi,
                    ),
                    force=(written == 1),
                )
            log_rows.append(
                {
                    "i": written,
                    "frame_id": state.frame_id,
                    "status": state.status,
                    "mood": state.mood,
                    "policy": state.policy_name,
                    "backend": state.backend,
                    "detect": state.detect_mode,
                    "actions": [a.action.value for a in state.actions],
                    "error": state.backend_error,
                }
            )
            if max_frames is not None and written >= max_frames:
                break
            if last_frame is None and not getattr(cap, "available", False):
                if written >= (max_frames or 8):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stdin.__exit__(None, None, None)
        except Exception:
            pass
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        if display:
            try:
                close_window()
            except Exception:
                pass
        if overlay is not None:
            try:
                overlay.close()
            except Exception:
                pass
        try:
            stop_gomi(gomi)
        except Exception:
            pass
        if owns_capture:
            try:
                cap.release()
            except Exception:
                pass

    log_path = None
    if source_key == "replay" and preview_path is not None and not display:
        log_path = Path(preview_path).with_suffix(".jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("".join(json.dumps(row) + "\n" for row in log_rows))

    return {
        "frames": written,
        "display": display,
        "preview": None if display or source_key == "live" else str(preview_path) if preview_path else None,
        "log": None if log_path is None else str(log_path),
        "policy": reflex.policy_name,
        "status": None if last_state is None else last_state.status,
        "mood": None if last_state is None else last_state.mood,
        "backend": reflex.backend_kind,
        "backend_url": reflex.backend_url,
        "source": source_key,
        "detect": detect_mode,
        "overlay": bool(use_overlay and overlay is not None and overlay.available),
        "opencv": cv2.__version__,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Screen Guardian laptop reflex (OpenCV 5). "
            "Replay uses synthetic HSV on the safe fixture. "
            "Live uses on-device NudeNet + macOS overlay windows. "
            "Not a zero-exposure or Canopy-accuracy claim."
        )
    )
    parser.add_argument("--policy", default="balanced", choices=("strict", "balanced", "relaxed"))
    parser.add_argument("--source", default="replay", choices=("replay", "live"))
    parser.add_argument(
        "--detect",
        default=None,
        choices=("synthetic", "nsfw"),
        help="synthetic = HSV fixture blobs (default for replay). "
        "nsfw = on-device NudeNet boxes (default for live). "
        "Not a production NSFW accuracy claim.",
    )
    parser.add_argument("--frames", type=int, default=None, help="stop after N frames")
    parser.add_argument(
        "--preview",
        nargs="?",
        const="hud",
        default=None,
        help="OpenCV debug HUD (flag) or write annotated mp4 (path, replay only). "
        "Live protection is the on-screen overlay even without this.",
    )
    parser.add_argument(
        "--hud",
        action="store_true",
        help="same as --preview with no path: open the OpenCV debug window",
    )
    parser.add_argument("--headless", action="store_true", help="do not open an OpenCV window")
    parser.add_argument(
        "--backend",
        default="auto",
        help="auto (live nsfw → Graviton if /health ok, else local), "
        "local, or http://HOST:8080. Live nsfw POSTs JPEG tiles to /v1/live_perceive.",
    )
    parser.add_argument(
        "--reinspect",
        default="local",
        choices=("local", "rekognition"),
        help="MID-band second look. rekognition sends crops only, never the desktop, off by default",
    )
    parser.add_argument(
        "--overlay",
        dest="screen_overlay",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="macOS on-screen mask windows. Default on for --source live, off for replay.",
    )
    parser.add_argument(
        "--engine",
        default="new",
        choices=("new", "legacy"),
        help="live runtime: new = threaded edge.live_engine (default), "
        "legacy = old single-thread reflex loop. Replay always uses the reflex.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=("aggressive", "medium", "off"),
        help="protection mode for the new live engine. Given explicitly, it overrides "
        "(with a warning) and re-saves the mode in guardian.json; omitted, the saved "
        "mode (set from the island) applies, else aggressive.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="new live engine capture rate (clamped to 4-15)",
    )
    add_cli_flags(parser)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print a full pipeline tick line to stderr every frame "
        "(live already logs once per second without this)",
    )
    supplied = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    if args.source == "live" and args.engine == "new":
        from edge.live_engine import run_live

        try:
            result = run_live(
                mode=args.mode,
                fps=args.fps,
                overlay=args.screen_overlay is not False,
                island=args.companion,
            )
        except KeyboardInterrupt:
            print("stopped", file=sys.stderr)
            return 0
        if result.get("error") == "capture unavailable":
            return 2
        print(
            f"edge live engine frames={result['frames']} results={result.get('results')} "
            f"mode={result['mode']} stopped={result.get('stopped')} "
            f"follow_ms_max={result.get('follow_ms_max')} cv={result['opencv']}"
        )
        return 0
    prefs = load_prefs()
    policy = args.policy
    if "--policy" not in supplied and prefs.get("policy"):
        policy = str(prefs["policy"])
    detect = args.detect
    if "--detect" not in supplied and args.source == "live" and prefs.get("detect"):
        detect = str(prefs["detect"])
    character = str(prefs.get("character_id") or "pebble")

    preview_path: Path | None = None
    hud = bool(args.hud)
    if args.preview == "hud":
        hud = True
    elif args.preview is not None:
        preview_path = Path(args.preview)

    try:
        result = run(
            policy=policy,
            source=args.source,
            backend=args.backend,
            max_frames=args.frames,
            preview=preview_path,
            headless=args.headless or (preview_path is not None),
            cycle_policies=(args.headless or preview_path is not None) and args.source == "replay",
            detect=detect,
            hud=hud,
            screen_overlay=args.screen_overlay,
            reinspect=args.reinspect,
            companion=args.companion,
            verbose=args.verbose,
            character=character,
        )
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 0
    if result.get("error") == "capture unavailable":
        return 2
    print(
        f"edge {result['source']} detect={result.get('detect')} frames={result['frames']} "
        f"backend={result['backend']} display={result['display']} "
        f"preview={result['preview']} cv={result['opencv']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
