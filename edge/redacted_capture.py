"""Redacted snapshots for kids chat ("explain my screen").

Serves redacted snapshots of the protected display to the kids chat service
over its own Unix socket, ``~/Library/Application Support/ScreenGuardian/redact.sock``
(mode 0600; ``SG_REDACT_SOCK`` overrides). ``island.sock`` stays pixel-free.
Protocol: ``kids_chat/PROTOCOL.md`` → "Engine redacted-snapshot socket".

Fail closed. A snapshot is refused (``{"ok": false, "reason": ...}``) unless:
- the parent turned explain-my-screen on (``chat.json`` ``explain_screen``),
- the engine is protecting (not paused, not mode off, capture OK, results > 0),
- the latest frame is < 500 ms old,
- every region is checked: no stale track, no deferred or pending change, no
  PENDING mask (before the busy cap, which may drop speculative masks), no
  busy warning, and the last analyzed frame is at or after the last frame
  where anything changed (``change_seq``), so the frame being sent matches
  what was judged (waits up to 1.5 s for that, then refuses). Requiring the
  newest frame itself to be analyzed never held live: capture runs ahead of
  the analyzer by a frame or two,
- the redacted copy passes one more local NSFW pass (exposed intimate
  classes, whole frame plus a 3×3 grid).
Every drawn mask (any verdict) is painted SOLID gray with padding, never
blurred: ``EngineCore.visible()``, i.e. capture-pixel masks already cut to
their window's visible region (so the front window and system UI stay
readable, and every covered picture stays covered). Then the copy is downscaled (longest edge ≤ 1280) and JPEG-encoded
(≤ 250 KB). Nothing is written to disk.

Wired in ``edge.live_engine``: ``LiveEngine.on_frame`` stamps
``last_frame_mono``; ``run_live`` starts / stops ``RedactedCaptureServer``.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import stat
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

APP_DIR = Path.home() / "Library" / "Application Support" / "ScreenGuardian"
FRESH_S = 0.5
CATCH_UP_S = 1.5
PAD_PX = 6
GRAY = (128, 128, 128)
MAX_JPEG = 250_000
MAX_FRAMES = 3


def _home() -> Path:
    for key in ("SG_CHAT_HOME", "SG_ISLAND_HOME"):
        if os.environ.get(key):
            return Path(os.environ[key]).expanduser()
    return APP_DIR


def socket_path() -> Path:
    v = os.environ.get("SG_REDACT_SOCK")
    return Path(v).expanduser() if v else _home() / "redact.sock"


def parent_opted_in() -> bool:
    try:
        cfg = json.loads((_home() / "chat.json").read_text())
    except (OSError, ValueError):
        return False
    return bool(cfg.get("enabled") is True and cfg.get("explain_screen") is True)


RECHECK_SCORE = 0.25  # exposed intimate classes only: covered ones are already masked per mode
RECHECK_GRID = 3  # whole frame plus a 3×3 grid, so small pictures are seen at a usable scale


def default_recheck() -> Callable[[np.ndarray], bool] | None:
    """True = the redacted copy still shows exposed intimate content. None = no detector (→ refuse).

    The legacy ``detect_nsfw`` balanced policy flagged covered classes too and
    refused harmless diagrams (a gear read as BUTTOCKS_COVERED 0.36)."""
    try:
        from vision_core.live.detector import NudeNetDetector
        from vision_core.nudenet_labels import EXPLICIT_CLASSES

        det = NudeNetDetector()
    except Exception:
        return None
    g = RECHECK_GRID

    def recheck(bgr: np.ndarray) -> bool:
        h, w = bgr.shape[:2]
        views = [bgr] + [bgr[i * h // g:(i + 1) * h // g, j * w // g:(j + 1) * w // g] for i in range(g) for j in range(g)]
        return any(d.label in EXPLICIT_CLASSES and d.score >= RECHECK_SCORE
                   for dets in det.detect_batch(views) for d in dets)

    return recheck


class Refused(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def paint(frame_bgr: np.ndarray, boxes: list[tuple[float, float, float, float]], pad: int = PAD_PX) -> tuple[np.ndarray, list[list[float]]]:
    """Copy of the frame with every box filled solid gray. Returns (copy, padded boxes in px)."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    painted = []
    for x1, y1, x2, y2 in boxes:
        a, b = max(0, int(x1) - pad), max(0, int(y1) - pad)
        c, d = min(w, int(np.ceil(x2)) + pad), min(h, int(np.ceil(y2)) + pad)
        if c <= a or d <= b:
            continue
        cv2.rectangle(out, (a, b), (c - 1, d - 1), GRAY, thickness=-1)
        if c - a > 120 and d - b > 40:
            cv2.putText(out, "covered", (a + 10, b + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2, cv2.LINE_AA)
        painted.append([float(a), float(b), float(c), float(d)])
    return out, painted


def encode(bgr: np.ndarray, max_edge: int, quality: int) -> tuple[bytes, int, int, float]:
    h, w = bgr.shape[:2]
    s = min(1.0, float(max_edge) / float(max(w, h)))
    small = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else bgr
    for q in (quality, 65, 55):
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
        if ok and len(buf) <= MAX_JPEG:
            return buf.tobytes(), small.shape[1], small.shape[0], s
    raise Refused("too_large")


class RedactedCapture:
    def __init__(self, engine: Any, *, recheck: Callable[[np.ndarray], bool] | None | bool = True,
                 opted_in: Callable[[], bool] = parent_opted_in) -> None:
        self.engine = engine
        self.recheck = default_recheck() if recheck is True else (recheck or None)
        self.opted_in = opted_in

    def _grab(self) -> tuple[np.ndarray, list, Any]:
        e = self.engine
        deadline = time.monotonic() + CATCH_UP_S
        while True:
            with e.lock:
                core = e.core
                if e.idle or not e.capture_ok:
                    raise Refused("not_protecting")
                if core.results <= 0 or core._last_frame is None or e.geom is None:
                    raise Refused("no_frame")
                last = getattr(e, "last_frame_mono", None)
                if last is None or time.monotonic() - last > FRESH_S:
                    raise Refused("stale_frame")
                masks = core.visible()
                specs = core.specs() if hasattr(core, "specs") else masks  # before the busy cap
                checked = (core.masks.stale_count() == 0 and not core._deferred and core.pending_change is None
                           and not core.urgent and core.last_result_seq >= core.change_seq
                           and not getattr(core, "warning", None)
                           and not any(getattr(m.verdict, "name", str(m.verdict)) == "PENDING" for m in list(masks) + list(specs)))
                if checked:
                    return core._last_frame.copy(), [tuple(m.xyxy) for m in masks], e.geom
            if time.monotonic() > deadline:
                raise Refused("unchecked_regions")
            time.sleep(0.05)

    def one(self, max_edge: int, quality: int) -> tuple[dict, int, int, list[list[float]], Any]:
        frame, boxes, geom = self._grab()
        red, painted = paint(frame, boxes)
        if self.recheck is None:
            raise Refused("no_recheck")
        if self.recheck(red):
            raise Refused("nsfw_recheck")
        jpg, w, h, s = encode(red, max_edge, quality)
        return ({"jpeg_b64": base64.b64encode(jpg).decode(), "t": time.time()}, w, h,
                [[round(v * s, 1) for v in m] for m in painted], geom)

    def snapshot(self, frames: int = 1, interval_ms: int = 700, max_edge: int = 1280, quality: int = 75) -> dict:
        try:
            if not self.opted_in():
                raise Refused("disabled")
            out, masks, size, geom = [], [], None, None
            for i in range(max(1, min(MAX_FRAMES, int(frames)))):
                if i:
                    time.sleep(max(0.2, min(1.5, interval_ms / 1000.0)))
                f, w, h, m, geom = self.one(min(1280, int(max_edge)), int(quality))
                if size is not None and size != (w, h):
                    raise Refused("display_changed")
                size = (w, h)
                out.append(f)
                masks.extend(m)  # union over frames: a mark may not sit on any frame's mask
            return {"type": "redacted_snapshot", "ok": True, "frames": out, "w": size[0], "h": size[1],
                    "display": {"x": geom.left_pt, "y": geom.top_pt, "w": geom.width_pt, "h": geom.height_pt},
                    "masks": masks, "all_checked": True, "status": "PROTECTING"}
        except Refused as exc:
            return {"type": "redacted_snapshot", "ok": False, "reason": exc.reason}
        except Exception as exc:  # anything unexpected: refuse
            return {"type": "redacted_snapshot", "ok": False, "reason": f"error:{type(exc).__name__}"}

    def status(self) -> dict:
        e = self.engine
        if not self.opted_in():
            return {"type": "snapshot_status", "available": False, "reason": "disabled"}
        if self.recheck is None:
            return {"type": "snapshot_status", "available": False, "reason": "no_recheck"}
        with e.lock:
            ok = not e.idle and e.capture_ok and e.core.results > 0
        return {"type": "snapshot_status", "available": ok, "reason": "" if ok else "not_protecting"}


class RedactedCaptureServer:
    def __init__(self, engine: Any, path: Path | None = None, **kw: Any) -> None:
        self.cap = RedactedCapture(engine, **kw)
        self.path = Path(path or socket_path())
        self.sock: socket.socket | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        if self.path.exists() and stat.S_ISSOCK(os.lstat(self.path).st_mode):
            self.path.unlink()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            s.bind(str(self.path))
        finally:
            os.umask(old)
        os.chmod(self.path, 0o600)
        s.listen(2)
        s.settimeout(0.5)
        self.sock = s
        threading.Thread(target=self._loop, name="sg-redact", daemon=True).start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                c, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _serve(self, c: socket.socket) -> None:
        try:
            c.settimeout(5)
            buf = b""
            while b"\n" not in buf and len(buf) < 4096:
                d = c.recv(4096)
                if not d:
                    return
                buf += d
            req = json.loads(buf.split(b"\n", 1)[0] or b"{}")
            if req.get("type") == "snapshot_status":
                resp = self.cap.status()
            elif req.get("type") == "redacted_snapshot":
                resp = self.cap.snapshot(req.get("frames", 1), req.get("interval_ms", 700),
                                         req.get("max_edge", 1280), req.get("quality", 75))
            else:
                resp = {"type": "error", "reason": "unknown request"}
            c.sendall(json.dumps(resp).encode() + b"\n")
        except (OSError, ValueError):
            pass
        finally:
            c.close()

    def stop(self) -> None:
        self._stop.set()
        if self.sock is not None:
            self.sock.close()
        try:
            self.path.unlink()
        except OSError:
            pass
