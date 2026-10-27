"""Contracts for the rebuilt live path (picture regions → verdicts → masks).

Shared by three workstreams, so change these only additively:

- ``vision_core.live`` (regions, cache, detector, modes) implements ``LiveAnalyzer``.
- ``edge.live_engine`` (threads, scroll follow, mask persistence, overlay) calls it.
- ``eval_live`` (synthetic screen corpus) scores both.

All boxes are ``(x1, y1, x2, y2)`` floats in **capture pixels** of the frame
passed in. Nothing here touches the display, AppKit, or the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np

Box = tuple[float, float, float, float]


class ProtectionMode(str, Enum):
    """Guardian-chosen mode, changed only behind the parental lock.

    AGGRESSIVE (kids): cover the **whole picture** that contains any flagged
        part, include suggestive classes, and pre-frost new pictures until
        they are cleared ("blur until safe").
    MEDIUM (teens): cover **body-part regions** (padded) for explicit classes
        only; pictures stay visible otherwise (sex-ed / health content).
    OFF (adults): no masks. Status still visible; capture may idle.
    """

    AGGRESSIVE = "aggressive"
    MEDIUM = "medium"
    OFF = "off"


class VerdictState(str, Enum):
    PENDING = "PENDING"  # proposed, not classified yet
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    UNSURE = "UNSURE"  # fail-closed per mode; may be escalated (cloud / agent)


class MaskStyle(str, Enum):
    FROST = "frost"  # strong blur material
    OPAQUE = "opaque"  # fully opaque cover


@dataclass
class PictureRegion:
    """A picture-like rectangle on screen (photo, thumbnail, video frame)."""

    region_id: str
    xyxy: Box
    kind: str = "picture"  # "picture" | "video" | "tile" (fallback crop, not a picture rect)
    picture_score: float = 0.0  # 0..1, how picture-like (vs text / UI)
    content_hash: int = 0  # perceptual hash of the region content (cache key)
    # Added (additive): 64-bit aHash of the same 32×32 gray (second cache key;
    # ``content_hash`` is the dHash).
    ahash: int = 0
    # Added (additive): sides cut by the frame / a pane edge ("t", "b", "l", "r"
    # combined, "" = the whole picture is visible).
    clipped: str = ""
    # Added (additive): for ``kind == "tile"``, the picture-like area the tile
    # belongs to (AGGRESSIVE covers the whole area when any tile hits).
    group_xyxy: Box | None = None
    # Added (additive): best estimate of the unclipped picture rect in capture
    # px (may extend off-screen); equals ``xyxy`` when nothing is clipped.
    full_xyxy: Box | None = None
    # Added (additive): id of the app window the picture sits in (from the
    # window list passed to ``analyze``); None when no window info is known.
    window_id: int | None = None


@dataclass
class Verdict:
    state: VerdictState
    score: float = 0.0  # max flagged-class score that drove the state
    labels: tuple[str, ...] = ()  # raw detector classes (never shown to the child)
    parts: tuple[Box, ...] = ()  # body-part boxes in capture pixels (MEDIUM masks)
    source: str = "local"  # "local" | "cache" | "cloud" | "agent"
    latency_ms: float = 0.0


@dataclass
class MaskSpec:
    """One rectangle the overlay must cover."""

    region_id: str
    xyxy: Box
    style: MaskStyle = MaskStyle.FROST
    reason: str = ""  # reason code, e.g. "UNSAFE_PICTURE", "PENDING_PREFROST"
    verdict: VerdictState = VerdictState.UNSAFE
    # Added (additive): owning app window (masks are clipped to its visible
    # region on every overlay tick); None = no window info (screen-wide).
    window_id: int | None = None


@dataclass
class AnalyzeResult:
    regions: list[PictureRegion] = field(default_factory=list)
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    masks: list[MaskSpec] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    # Added (additive): integer counters for logs / eval, e.g. ``n_regions``,
    # ``n_cache_hits``, ``n_classified``, ``n_pending``, ``n_second_look``, ``n_crops``.
    counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class WindowInfo:
    """One on-screen window from the window server (no titles, no pixels).

    ``xyxy`` is in **capture pixels** of the frame it came with (Retina
    correct), unclipped (a window may extend off-screen).
    """

    window_id: int
    xyxy: Box
    layer: int = 0  # 0 = normal app window; anything else is system UI / panels
    owner_pid: int = 0
    owner: str = ""  # process name (e.g. "Safari"); never the window title
    alpha: float = 1.0

    @property
    def is_app(self) -> bool:
        return self.layer == 0


@dataclass
class ScreenLayout:
    """Window list that belongs to one frame.

    ``windows`` is **front to back** (window-server order). ``visible_xyxy``
    is ``NSScreen.visibleFrame`` in capture px (the screen minus the menu
    bar and the Dock); None = unknown (treated as the whole frame).
    ``stale``: the list may not match the pixels (open / close / minimise
    animation, Mission Control): keep existing masks, add none over system UI.
    """

    frame_wh: tuple[int, int]
    windows: tuple[WindowInfo, ...] = ()
    visible_xyxy: Box | None = None
    t_ms: float = 0.0
    stale: bool = False

    def window(self, window_id: int | None) -> WindowInfo | None:
        if window_id is None:
            return None
        for w in self.windows:
            if w.window_id == window_id:
                return w
        return None


class LiveAnalyzer(Protocol):
    """Implemented in ``vision_core.live.analyzer``. Thread-confined: one
    vision worker thread owns an instance; it is not called concurrently."""

    mode: ProtectionMode

    def set_mode(self, mode: ProtectionMode) -> None: ...

    def analyze(
        self,
        frame_bgr: np.ndarray,
        *,
        changed: Box | None = None,
        t_ms: float | None = None,
        windows: ScreenLayout | None = None,
    ) -> AnalyzeResult:
        """Propose picture regions (optionally only inside ``changed``),
        classify new ones (cache hits are free), and return the masks this
        mode wants for the **whole** frame.

        ``t_ms``: frame time (None = the analyzer's own monotonic clock).
        ``windows`` (added, optional): the window list captured with this
        frame. Regions are then only proposed inside app windows, never
        across a window border, and every mask carries its ``window_id``."""
        ...
