"""Raw-tensor fake NudeNet: the real ``NudeNetDetector`` minus the model.

``FakeRawNet`` subclasses ``vision_core.live.detector.NudeNetDetector`` and
replaces only the ONNX forward pass. Letterboxing, ``blobFromImages``
batching, YOLOv8 decoding (``decode_yolov8``) and per-class NMS are the
production code, so tests and the eval exercise the same decode the live
engine runs — unlike injecting ``RawDet`` objects directly.

The synthetic output has NudeNet 320n's shape ``(B, 4 + 18, 2100)``: a few
overlapping anchors per target (as a real YOLO head fires), each anchor
carrying **every** class score of the target on the same column (so "covered
on top, exposed runner-up" is representable), plus low background noise.

Targets come from ``targets_fn(image) -> list[FakeTarget]`` in that image's
pixels. The default finds the magenta markers drawn by
``vision_core.live.synth`` (no imagery involved). ``eval_live.oracle`` plugs
in GT part boxes instead.

No weights are loaded and nothing is downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from vision_core.live.detector import NudeNetDetector
from vision_core.nudenet_labels import NUDENET_LABELS

N_ANCHORS = 2100  # 320n: 40² + 20² + 10²
FAKE_MODEL = Path("fake-320n.onnx")  # never opened; the name sets input_size=320
MARKER_BGR = (255, 0, 255)


@dataclass(frozen=True)
class FakeTarget:
    """One object for the fake head: a box (image px) and its class scores."""

    xyxy: tuple[float, float, float, float]
    scores: Mapping[str, float] = field(default_factory=dict)


TargetsFn = Callable[[np.ndarray], Sequence[FakeTarget]]


def marker_boxes(image: np.ndarray, min_px: int = 9) -> list[tuple[float, float, float, float]]:
    """Bounding boxes of magenta (255, 0, 255) marker blobs, image px."""
    import cv2

    if image.ndim != 3 or image.shape[2] < 3:
        return []
    b, g, r = image[..., 0], image[..., 1], image[..., 2]
    m = ((b > 240) & (g < 15) & (r > 240)).astype(np.uint8)
    if int(m.sum()) < min_px:
        return []
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area >= min_px:
            out.append((float(x), float(y), float(x + w), float(y + h)))
    return out


def marker_targets(scores: Mapping[str, float]) -> TargetsFn:
    """``targets_fn`` firing ``scores`` on every synth marker."""

    frozen = dict(scores)

    def fn(image: np.ndarray) -> list[FakeTarget]:
        return [FakeTarget(box, frozen) for box in marker_boxes(image)]

    return fn


class FakeRawNet(NudeNetDetector):
    """``NudeNetDetector`` whose ``forward()`` synthesises YOLOv8 output.

    ``scores``: class → score fired on each target (default: the old oracle,
    ``FEMALE_BREAST_EXPOSED`` 0.9). ``targets_fn`` overrides where targets
    are (default: synth markers). ``min_model_px``: targets smaller than this
    in the 320-px model input are invisible, as for the real model.
    """

    def __init__(
        self,
        scores: Mapping[str, float] | None = None,
        *,
        targets_fn: TargetsFn | None = None,
        anchors_per_target: int = 3,
        background: float = 0.02,
        min_model_px: float = 4.0,
        seed: int = 0,
        **kw,
    ) -> None:
        self.fake_scores = dict(scores or {"FEMALE_BREAST_EXPOSED": 0.9})
        for lab in self.fake_scores:
            if lab not in NUDENET_LABELS:
                raise ValueError(f"not a NudeNet label: {lab!r}")
        self.targets_fn: TargetsFn = targets_fn or marker_targets(self.fake_scores)
        self.anchors_per_target = max(1, int(anchors_per_target))
        self.background = float(background)
        self.min_model_px = float(min_model_px)
        self._rng = np.random.default_rng(seed)
        self._pending: list[list[FakeTarget]] = []
        self.calls = 0
        self.crops = 0
        self.fired = 0
        kw.setdefault("warmup", False)
        kw.setdefault("backend", "cv2")
        super().__init__(kw.pop("model_path", FAKE_MODEL), **kw)

    # -- no model ----------------------------------------------------------
    def _load_cv2(self) -> None:
        self.backend = "fake"

    def _load_ort(self) -> None:
        self.backend = "fake"

    # -- targets are found on the original images, then letterbox-scaled ---
    def detect_batch(self, images):
        items = list(images)
        size = float(self.input_size)
        for img in items:
            h, w = img.shape[:2]
            scale = size / float(max(1, max(h, w)))
            model = []
            for t in self.targets_fn(img):
                x1, y1, x2, y2 = (float(v) * scale for v in t.xyxy)
                if min(x2 - x1, y2 - y1) < self.min_model_px:
                    continue
                model.append(FakeTarget((x1, y1, x2, y2), t.scores))
            self._pending.append(model)
        try:
            return super().detect_batch(items)
        finally:
            self._pending.clear()

    def forward(self, blob: np.ndarray) -> np.ndarray:
        self.calls += 1
        B = int(blob.shape[0])
        self.crops += B
        C = len(NUDENET_LABELS)
        out = np.zeros((B, 4 + C, N_ANCHORS), np.float32)
        # Junk background anchors: tiny boxes, low scores on every class.
        out[:, 0:2, :] = self._rng.uniform(0, self.input_size, (B, 2, N_ANCHORS))
        out[:, 2:4, :] = 8.0
        if self.background > 0:
            out[:, 4:, :] = self._rng.uniform(0, self.background, (B, C, N_ANCHORS))
        batch, self._pending = self._pending[:B], self._pending[B:]
        for b, targets in enumerate(batch):
            if targets:
                self.fired += 1
            for j, t in enumerate(targets):
                x1, y1, x2, y2 = t.xyxy
                cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                for k in range(self.anchors_per_target):
                    a = 64 + j * 8 + k
                    if a >= N_ANCHORS:
                        break
                    jit = (0.0, 1.0, -1.0, 2.0, -2.0)[k % 5]
                    out[b, 0:4, a] = (cx + jit, cy + jit, w, h)
                    out[b, 4:, a] = 0.0
                    for lab, sc in t.scores.items():
                        out[b, 4 + NUDENET_LABELS.index(lab), a] = max(0.0, float(sc) - 0.02 * abs(jit))
        return out
