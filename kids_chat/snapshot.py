"""Redacted screenshots for explain-my-screen: the narrow interface to the vision engine.

The chat service never captures the screen itself. It asks the engine, which
owns capture and the masks, over a separate Unix socket (``redact.sock``;
``island.sock`` stays pixel-free). Spec: ``kids_chat/PROTOCOL.md`` section
"Engine redacted-snapshot socket"; the engine side is
``edge/redacted_capture.py``.

Fail closed, twice:
1. The engine refuses unless it is PROTECTING, the frame is fresh and every
   region is checked (``ok:false, reason``).
2. This client re-verifies what it got: JPEG decodes at the stated size, every
   mask region is painted *solid* in every frame, ``all_checked`` is true. Any
   doubt → :class:`SnapshotRefused`.
"""

from __future__ import annotations

import base64
import io
import json
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kids_chat.config import home_dir

MAX_FRAMES = 3
MAX_JPEG = 400_000
SOLID_SPREAD = 28  # max per-channel spread inside a painted mask (JPEG ringing)


class SnapshotRefused(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def redact_socket_path() -> Path:
    v = os.environ.get("SG_REDACT_SOCK")
    return Path(v).expanduser() if v else home_dir() / "redact.sock"


@dataclass
class Snapshot:
    frames: list[bytes]  # JPEG bytes, already redacted by the engine
    w: int
    h: int
    display: dict  # captured area in Cocoa global points: {"x","y","w","h"}
    masks_px: list[list[float]] = field(default_factory=list)  # [[x0,y0,x1,y1]] in sent pixels

    @property
    def masks_norm(self) -> list[list[float]]:
        return [[round(m[0] * 1000 / self.w, 1), round(m[1] * 1000 / self.h, 1),
                 round(m[2] * 1000 / self.w, 1), round(m[3] * 1000 / self.h, 1)] for m in self.masks_px]

    def payload(self) -> dict:
        return {"images": [base64.b64encode(f).decode() for f in self.frames], "w": self.w, "h": self.h,
                "masks": self.masks_norm}


def verify(resp: dict) -> Snapshot:
    """Engine response → verified :class:`Snapshot`, or raise :class:`SnapshotRefused`."""
    if not isinstance(resp, dict) or resp.get("ok") is not True:
        raise SnapshotRefused(str((resp or {}).get("reason") or "refused"))
    if resp.get("all_checked") is not True:
        raise SnapshotRefused("unchecked_regions")
    try:
        w, h = int(resp["w"]), int(resp["h"])
        disp = {k: float(resp["display"][k]) for k in ("x", "y", "w", "h")}
        masks = [[float(v) for v in m] for m in resp.get("masks") or []]
        frames = [base64.b64decode(f["jpeg_b64"], validate=True) for f in resp["frames"]]
    except (KeyError, TypeError, ValueError):
        raise SnapshotRefused("malformed")
    if not 1 <= len(frames) <= MAX_FRAMES or not (64 <= w <= 2048 and 64 <= h <= 2048) or disp["w"] <= 0 or disp["h"] <= 0:
        raise SnapshotRefused("malformed")
    if any(len(m) != 4 or m[2] <= m[0] or m[3] <= m[1] for m in masks):
        raise SnapshotRefused("malformed")
    try:
        from PIL import Image
    except ImportError:
        raise SnapshotRefused("no_verifier")
    for jpg in frames:
        if len(jpg) > MAX_JPEG or not jpg.startswith(b"\xff\xd8"):
            raise SnapshotRefused("bad_image")
        try:
            im = Image.open(io.BytesIO(jpg)).convert("RGB")
        except Exception:
            raise SnapshotRefused("bad_image")
        if im.size != (w, h):
            raise SnapshotRefused("size_mismatch")
        for m in masks:
            if not _solid(im, m):
                raise SnapshotRefused("mask_not_painted")
    return Snapshot(frames=frames, w=w, h=h, display=disp, masks_px=masks)


def _solid(im: Any, m: list[float], grid: int = 7) -> bool:
    x0, y0 = max(0, int(m[0]) + 3), max(0, int(m[1]) + 3)
    x1, y1 = min(im.width - 1, int(m[2]) - 3), min(im.height - 1, int(m[3]) - 3)
    if x1 <= x0 or y1 <= y0:
        return True  # sliver: nothing visible to leak
    px = [im.getpixel((x0 + (x1 - x0) * i // (grid - 1), y0 + (y1 - y0) * j // (grid - 1)))
          for i in range(grid) for j in range(grid)]
    return all(max(c) - min(c) <= SOLID_SPREAD for c in zip(*px))


class EngineSnapshotProvider:
    """Talks to the engine's ``redact.sock`` (JSON line request → one JSON line reply)."""

    def __init__(self, path: Path | None = None, timeout: float = 4.0) -> None:
        self.path = path or redact_socket_path()
        self.timeout = timeout

    def _call(self, req: dict) -> dict:
        if not self.path.exists():
            raise SnapshotRefused("engine_unavailable")
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        try:
            s.connect(str(self.path))
            s.sendall(json.dumps(req).encode() + b"\n")
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(1 << 16)
                if not chunk:
                    break
                buf += chunk
                if len(buf) > 4 * MAX_JPEG * MAX_FRAMES:
                    raise SnapshotRefused("too_large")
            return json.loads(buf.split(b"\n", 1)[0] or b"{}")
        except (OSError, ValueError):
            raise SnapshotRefused("engine_unavailable")
        finally:
            s.close()

    def available(self) -> tuple[bool, str]:
        try:
            r = self._call({"type": "snapshot_status"})
        except SnapshotRefused as exc:
            return False, exc.reason
        return bool(r.get("available") is True), str(r.get("reason") or "")

    def get(self, frames: int = 1) -> Snapshot:
        n = max(1, min(MAX_FRAMES, int(frames)))
        return verify(self._call({"type": "redacted_snapshot", "frames": n, "interval_ms": 700,
                                  "max_edge": 1280, "quality": 75}))


class UnavailableSnapshotProvider:
    def available(self) -> tuple[bool, str]:
        return False, "engine_unavailable"

    def get(self, frames: int = 1) -> Snapshot:
        raise SnapshotRefused("engine_unavailable")


class FakeSnapshotProvider:
    """Tests and the stage fallback: a synthetic diagram (drawn here, no media files) with one
    painted 'covered' box, returned through the same :func:`verify` path.
    Enable for a demo with ``SG_CHAT_FAKE_SNAPSHOT=1`` (logged loudly; say so on stage)."""

    def __init__(self, display: dict | None = None, *, paint_masks: bool = True, all_checked: bool = True) -> None:
        self.display = display or {"x": 0.0, "y": 0.0, "w": 1440.0, "h": 900.0}
        self.paint_masks = paint_masks
        self.all_checked = all_checked

    def available(self) -> tuple[bool, str]:
        return True, "fake"

    def response(self, frames: int = 1) -> dict:
        from PIL import Image, ImageDraw
        w, h = 1280, 800
        mask = [900.0, 80.0, 1180.0, 300.0]
        out = []
        for i in range(max(1, min(MAX_FRAMES, frames))):
            im = Image.new("RGB", (w, h), (246, 244, 238))
            d = ImageDraw.Draw(im)
            # a simple lever-and-pulley diagram
            d.line([(200, 600), (760, 470)], fill=(60, 60, 70), width=14)
            d.polygon([(470, 540), (420, 640), (520, 640)], fill=(200, 120, 60))
            d.rectangle([180, 520, 260, 600], fill=(80, 140, 220))
            d.ellipse([900 - 60, 420 + i * 10, 900 + 60, 540 + i * 10], outline=(60, 60, 70), width=10)
            d.line([(900, 540), (900, 700)], fill=(60, 60, 70), width=6)
            d.rectangle([860, 700, 940, 760], fill=(90, 170, 110))
            d.text((60, 40), "How a lever works", fill=(20, 20, 30))
            # a picture region the engine covered: tiled colour, then painted solid if masked
            for yy in range(int(mask[1]), int(mask[3]), 20):
                for xx in range(int(mask[0]), int(mask[2]), 20):
                    d.rectangle([xx, yy, xx + 19, yy + 19], fill=((xx * 7) % 255, (yy * 5) % 255, 120))
            if self.paint_masks:
                d.rectangle(mask, fill=(128, 128, 128))
                d.text((mask[0] + 12, mask[1] + 12), "covered", fill=(150, 150, 150))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=75)
            out.append({"jpeg_b64": base64.b64encode(buf.getvalue()).decode(), "t": time.time()})
        return {"type": "redacted_snapshot", "ok": True, "frames": out, "w": w, "h": h, "display": self.display,
                "masks": [mask], "all_checked": self.all_checked, "status": "PROTECTING", "age_ms": 0}

    def get(self, frames: int = 1) -> Snapshot:
        return verify(self.response(frames))


def default_provider() -> Any:
    if os.environ.get("SG_CHAT_FAKE_SNAPSHOT") == "1":
        return FakeSnapshotProvider()
    return EngineSnapshotProvider()
