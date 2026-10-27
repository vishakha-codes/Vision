"""Synthetic screens for the live-engine tests. cv2-drawn only: textured
abstract "pictures" (noise + gradients + shapes) and text lines. No real
or explicit imagery anywhere."""

from __future__ import annotations

import cv2
import numpy as np

from vision_core.live.contracts import (
    AnalyzeResult,
    MaskSpec,
    PictureRegion,
    ProtectionMode,
    Verdict,
    VerdictState,
)

Box = tuple[float, float, float, float]


def textured_picture(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack(
        [
            127 + 100 * np.sin(xx / (11 + seed % 7) + seed),
            127 + 100 * np.cos(yy / (13 + seed % 5)),
            127 + 100 * np.sin((xx + yy) / 17.0),
        ],
        axis=-1,
    )
    img = np.clip(base + rng.normal(0, 18, (h, w, 3)), 0, 255).astype(np.uint8)
    for _ in range(8):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(6, max(8, min(h, w) // 4)))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(img, (cx, cy), r, color, -1)
    return img


def make_page(
    height: int,
    width: int,
    rects: list[tuple[int, int, int, int]],
    *,
    seed: int = 0,
    bg: int = 244,
) -> np.ndarray:
    """A long page: text lines everywhere, pictures at ``rects`` (x1,y1,x2,y2)."""
    rng = np.random.default_rng(seed)
    page = np.full((height, width, 3), bg, dtype=np.uint8)
    words = ["lorem", "ipsum", "dolor", "sit", "amet", "screen", "guardian", "opencv", "frame"]
    for y in range(28, height, 26):
        x = 16
        while x < width - 80:
            word = words[int(rng.integers(0, len(words)))]
            cv2.putText(page, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
            x += 12 * len(word) + int(rng.integers(6, 20))
    for i, (x1, y1, x2, y2) in enumerate(rects):
        page[y1:y2, x1:x2] = textured_picture(y2 - y1, x2 - x1, seed * 100 + i + 1)
    return page


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def clip(box: Box, w: int, h: int) -> Box:
    return (max(0.0, box[0]), max(0.0, box[1]), min(float(w), box[2]), min(float(h), box[3]))


class KnownRectsAnalyzer:
    """Stub analyzer that "knows" where the pictures are on each frame.

    ``truth(index)`` → list of (region_id, box) visible on frame ``index``;
    the frame index is recovered from ``t_ms`` (``index * frame_ms``).
    """

    def __init__(
        self,
        truth,
        *,
        frame_ms: float,
        verdict: VerdictState = VerdictState.UNSAFE,
        mode: ProtectionMode = ProtectionMode.AGGRESSIVE,
        safe=None,
    ) -> None:
        self.truth = truth
        self.frame_ms = frame_ms
        self.verdict = verdict
        self.mode = mode
        self.safe = safe  # optional truth(index) → (rid, box) of SAFE pictures
        self.calls: list[tuple[int, object]] = []

    def set_mode(self, mode: ProtectionMode) -> None:
        self.mode = mode

    def analyze(self, frame_bgr, *, changed=None, t_ms: float = 0.0) -> AnalyzeResult:
        index = int(round(t_ms / self.frame_ms))
        self.calls.append((index, changed))
        h, w = frame_bgr.shape[:2]
        masks = []
        for rid, box in self.truth(index):
            vis = clip(box, w, h)
            if vis[2] - vis[0] < 8 or vis[3] - vis[1] < 8:
                continue
            masks.append(MaskSpec(region_id=rid, xyxy=vis, reason="UNSAFE_PICTURE", verdict=self.verdict))
        regions = [PictureRegion(m.region_id, m.xyxy) for m in masks]
        verdicts = {m.region_id: Verdict(self.verdict) for m in masks}
        for rid, box in (self.safe(index) if self.safe else []):
            vis = clip(box, w, h)
            if vis[2] - vis[0] >= 8 and vis[3] - vis[1] >= 8:
                regions.append(PictureRegion(rid, vis))
                verdicts[rid] = Verdict(VerdictState.SAFE)
        return AnalyzeResult(regions=regions, verdicts=verdicts, masks=masks)


class NullAnalyzer:
    """Test-only analyzer that finds nothing (there is no stand-in in the
    live path: a missing analyzer makes the engine report NOT_PROTECTING)."""

    def __init__(self, mode: ProtectionMode = ProtectionMode.AGGRESSIVE) -> None:
        self.mode = mode

    def set_mode(self, mode: ProtectionMode) -> None:
        self.mode = mode

    def analyze(self, frame_bgr, *, changed=None, t_ms=None, **_kw) -> AnalyzeResult:
        return AnalyzeResult()
