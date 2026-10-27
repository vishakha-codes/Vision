"""NudeNet (YOLOv8) part detector on **OpenCV 5 DNN**, onnxruntime fallback.

Primary path: ``cv2.dnn.readNetFromONNX`` (the new OpenCV 5 engine; the
classic importer cannot load this graph). The graph has a dynamic batch, so
crops are letterboxed and run in one ``blobFromImages`` batch. Decoding is
vectorised YOLOv8 (``(B, 4 + C, N)``: cx, cy, w, h in input pixels, then C
class scores, no objectness) + per-class ``cv2.dnn.NMSBoxesBatched``. Each
anchor yields its top class **and** every maskable class column at or above
the floor, so "covered 0.55 / exposed 0.50" keeps the exposed score.

Preprocessing matches NudeNet's reader: pad **right / bottom** with black to
a square, resize to the model input, RGB, ``1/255``. Small crops are
upscaled into the input, large ones area-downscaled.

Model path is pluggable (a future ``640m.onnx``: input size is read from the
file name or passed in). Nothing is downloaded here.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from vision_core.nudenet_labels import MASKABLE_LABELS, NUDENET_LABELS  # noqa: F401  (re-export)

DEFAULT_SCORE_FLOOR = 0.12  # below every mode's UNSURE floor
DEFAULT_NMS_IOU = 0.45
MAX_CANDIDATES = 512  # (anchor, class) pairs per image kept for NMS, best first
MAX_BATCH = 16  # ~4 ms / crop at 16, ~14 ms at 1 (cv2.dnn, 320n, 26 Oct)


@dataclass(frozen=True)
class RawDet:
    """One detector box in the pixel coords of the image that was passed in."""

    label: str
    score: float
    xyxy: tuple[float, float, float, float]


class PartDetector(Protocol):
    """What the analyzer needs. Tests inject a stub with the same shape."""

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[RawDet]]: ...


@dataclass(frozen=True)
class Letterbox:
    scale: float  # model px per input px
    width: int
    height: int


def letterbox(bgr: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """Top-left letterbox into ``size×size`` (NudeNet pads right / bottom)."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    elif bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
    h0, w0 = bgr.shape[:2]
    scale = float(size) / float(max(1, max(h0, w0)))
    nw = max(1, min(size, int(round(w0 * scale))))
    nh = max(1, min(size, int(round(h0 * scale))))
    src = bgr
    k = int(1.0 / scale) if scale < 0.5 else 1
    if k >= 2:
        # INTER_AREA is ~10× faster on an exact integer factor; the trimmed
        # remainder (< k px) is negligible, then a linear resize finishes.
        src = cv2.resize(
            bgr[: (h0 // k) * k, : (w0 // k) * k],
            (w0 // k, h0 // k),
            interpolation=cv2.INTER_AREA,
        )
    if (nw, nh) != (src.shape[1], src.shape[0]):
        src = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = cv2.copyMakeBorder(src, 0, size - nh, 0, size - nw, cv2.BORDER_CONSTANT, value=0)
    return out, Letterbox(scale, w0, h0)


def make_blob(images: Sequence[np.ndarray], size: int) -> tuple[np.ndarray, list[Letterbox]]:
    boxed: list[np.ndarray] = []
    metas: list[Letterbox] = []
    for img in images:
        lb, meta = letterbox(img, size)
        boxed.append(lb)
        metas.append(meta)
    blob = cv2.dnn.blobFromImages(boxed, 1.0 / 255.0, (size, size), (0, 0, 0), swapRB=True, crop=False)
    return blob, metas


def decode_yolov8(
    output: np.ndarray,
    metas: Sequence[Letterbox],
    labels: Sequence[str],
    *,
    score_floor: float = DEFAULT_SCORE_FLOOR,
    nms_iou: float = DEFAULT_NMS_IOU,
    extra_labels: Sequence[str] | None = MASKABLE_LABELS,
    max_candidates: int = MAX_CANDIDATES,
) -> list[list[RawDet]]:
    """``(B, 4 + C, N)`` → per-image boxes in original image pixels.

    Candidates per anchor: its top class (any label, so faces still show in
    probes) plus every ``extra_labels`` column with a score ≥ ``score_floor``
    (YOLOv8 heads are independent sigmoids, so a covered top score does not
    hide an exposed runner-up). ``extra_labels=None`` is the old top-class-only
    decode. NMS is per class.
    """
    out = np.asarray(output, dtype=np.float32)
    if out.ndim == 2:
        out = out[None]
    floor = float(score_floor)
    wanted = frozenset(extra_labels or ())
    extra_cols = np.asarray([i for i, lab in enumerate(labels) if lab in wanted], dtype=np.intp)
    results: list[list[RawDet]] = []
    for b, meta in enumerate(metas):
        pred = out[b]  # (4 + C, N)
        cls = pred[4:]  # (C, N)
        n = cls.shape[1]
        top = cls.argmax(axis=0)
        top_conf = cls[top, np.arange(n)]
        a_top = np.flatnonzero(top_conf >= floor)
        c_top = top[a_top]
        if extra_cols.size and a_top.size:
            # An extra column ≥ floor implies the anchor's top score is too.
            sub = cls[np.ix_(extra_cols, a_top)]  # (E, A)
            ei, ej = np.nonzero(sub >= floor)
            ea = a_top[ej]
            ec = extra_cols[ei]
            fresh = ec != top[ea]  # the top class is already a candidate
            anchors = np.concatenate([a_top, ea[fresh]])
            cids = np.concatenate([c_top, ec[fresh]])
        else:
            anchors, cids = a_top, c_top
        if anchors.size == 0:
            results.append([])
            continue
        conf = cls[cids, anchors]
        if conf.size > max_candidates > 0:
            sel = np.argpartition(-conf, max_candidates - 1)[:max_candidates]
            anchors, cids, conf = anchors[sel], cids[sel], conf[sel]
        p = pred[:4, anchors]
        inv = 1.0 / max(meta.scale, 1e-9)
        x1 = np.clip((p[0] - p[2] * 0.5) * inv, 0, meta.width)
        y1 = np.clip((p[1] - p[3] * 0.5) * inv, 0, meta.height)
        x2 = np.clip((p[0] + p[2] * 0.5) * inv, 0, meta.width)
        y2 = np.clip((p[1] + p[3] * 0.5) * inv, 0, meta.height)
        xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), conf.tolist(), cids.tolist(), floor, float(nms_iou))
        dets: list[RawDet] = []
        for i in np.asarray(idx).reshape(-1):
            c = int(cids[i])
            name = labels[c] if c < len(labels) else f"class_{c}"
            dets.append(RawDet(name, float(conf[i]), (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]))))
        dets.sort(key=lambda d: -d.score)
        results.append(dets)
    return results


def _infer_input_size(path: Path, default: int = 320) -> int:
    m = re.search(r"(\d{3,4})[a-z]?\.onnx$", path.name)
    if m:
        size = int(m.group(1))
        if 160 <= size <= 1280:
            return size
    return default


def default_model_path() -> Path:
    from vision_core.detect_nsfw import resolve_model_path

    return resolve_model_path()


class NudeNetDetector:
    """Letterbox + batch + decode. ``backend``: ``"cv2"``, ``"ort"`` or ``"auto"``
    (cv2.dnn, falling back to onnxruntime if OpenCV cannot load / run it)."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        backend: str = "auto",
        input_size: int | None = None,
        labels: Sequence[str] = NUDENET_LABELS,
        score_floor: float = DEFAULT_SCORE_FLOOR,
        nms_iou: float = DEFAULT_NMS_IOU,
        max_batch: int = MAX_BATCH,
        warmup: bool = True,
    ) -> None:
        self.model_path = Path(model_path) if model_path else default_model_path()
        self.input_size = int(input_size or _infer_input_size(self.model_path))
        self.labels = tuple(labels)
        self.score_floor = float(score_floor)
        self.nms_iou = float(nms_iou)
        self.max_batch = max(1, int(max_batch))
        self.backend = ""
        self.fallback_reason = ""
        self._net = None
        self._ort = None
        self._ort_input = ""
        want = (backend or "auto").strip().lower()
        if want in {"auto", "cv2", "opencv", "dnn"}:
            try:
                self._load_cv2()
            except Exception as exc:  # pragma: no cover - depends on the wheel
                if want != "auto":
                    raise
                self.fallback_reason = f"cv2.dnn load failed: {exc}"
        if not self.backend:
            self._load_ort()
        if warmup:
            self.warmup()

    def _load_cv2(self) -> None:
        self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.backend = "cv2"

    def _load_ort(self) -> None:
        import onnxruntime as ort

        self._ort = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self._ort_input = self._ort.get_inputs()[0].name
        self.backend = "ort"

    def warmup(self) -> float:
        """First forward allocates the graph (~20 ms). Returns its cost."""
        t0 = time.perf_counter()
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        self.forward(dummy)
        return (time.perf_counter() - t0) * 1000.0

    def forward(self, blob: np.ndarray) -> np.ndarray:
        """Raw model output for an NCHW float blob."""
        if self.backend == "cv2":
            try:
                self._net.setInput(blob)
                return np.asarray(self._net.forward())
            except cv2.error as exc:  # pragma: no cover - depends on the wheel
                self.fallback_reason = f"cv2.dnn forward failed: {exc}"
                self._load_ort()
        return np.asarray(self._ort.run(None, {self._ort_input: blob})[0])

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[RawDet]]:
        out: list[list[RawDet]] = []
        items = list(images)
        for start in range(0, len(items), self.max_batch):
            chunk = items[start : start + self.max_batch]
            blob, metas = make_blob(chunk, self.input_size)
            raw = self.forward(blob)
            out.extend(
                decode_yolov8(
                    raw, metas, self.labels, score_floor=self.score_floor, nms_iou=self.nms_iou
                )
            )
        return out

    def detect(self, bgr: np.ndarray) -> list[RawDet]:
        return self.detect_batch([bgr])[0]
