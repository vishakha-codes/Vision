"""NudeNet 320n on whatever ``cv2`` this process imported (COOL on the box).

COOL on the Graviton4 AMI is OpenCV 4.14.0-pre. Its classic ONNX importer
cannot load the stock dynamic-shape ``320n.onnx``, but it loads the
static-shape copies made by :mod:`cloud.cool_model` (same weights). Those have
a fixed batch, so :class:`StaticBatchDetector` runs full chunks on the
``b{B}`` net and the remainder one by one on the ``b1`` net.

Letterbox, blob, YOLOv8 decode and ``NMSBoxesBatched`` are the shared
``vision_core.live.detector`` code, so the Mac (stock OpenCV 5) and the box
(COOL 4.14) decode identically.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from vision_core.live.detector import NudeNetDetector

VISION_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = VISION_ROOT / ".models"


@lru_cache(maxsize=1)
def opencv_build() -> dict:
    """Which OpenCV build serves requests (reported in every response)."""
    info = cv2.getBuildInformation()

    def grab(key: str) -> str:
        m = re.search(rf"^\s*{re.escape(key)}:\s*(.+)$", info, re.MULTILINE)
        return m.group(1).strip() if m else ""

    path = str(getattr(cv2, "__file__", ""))
    return {
        "version": str(cv2.__version__),
        "cool": path.startswith("/opt/cool"),
        "module": path,
        "custom_hal": grab("Custom HAL"),
        "cpu_baseline": grab("Baseline"),
        "parallel": grab("Parallel framework"),
        "lapack": "armpl" if "armpl" in info else grab("Lapack")[:40],
        "threads": int(cv2.getNumThreads()),
    }


class StaticBatchDetector(NudeNetDetector):
    """``NudeNetDetector`` over fixed-batch ONNX copies (classic importer)."""

    def __init__(self, model_dir: str | Path | None = None, *, batch: int = 4, **kw) -> None:
        self.static_batch = max(1, int(batch))
        self.model_dir = Path(model_dir or os.environ.get("SG_MODEL_DIR") or MODELS_DIR)
        self._net_b = None
        self._net_1 = None
        kw.setdefault("max_batch", 16)
        super().__init__(
            self.model_dir / f"320n_static_b{self.static_batch}.onnx",
            backend="cv2",
            input_size=320,
            **kw,
        )

    def _load_cv2(self) -> None:
        one = self.model_dir / "320n_static_b1.onnx"
        self._net_1 = cv2.dnn.readNetFromONNX(str(one))
        if self.static_batch > 1:
            self._net_b = cv2.dnn.readNetFromONNX(str(self.model_path))
        self._net = self._net_1
        self.backend = "cv2"

    def forward(self, blob: np.ndarray) -> np.ndarray:
        n = int(blob.shape[0])
        outs: list[np.ndarray] = []
        i = 0
        B = self.static_batch
        while self._net_b is not None and n - i >= B:
            self._net_b.setInput(np.ascontiguousarray(blob[i : i + B]))
            outs.append(np.asarray(self._net_b.forward()))
            i += B
        while i < n:
            self._net_1.setInput(np.ascontiguousarray(blob[i : i + 1]))
            outs.append(np.asarray(self._net_1.forward()))
            i += 1
        return np.concatenate(outs, axis=0)

    def warmup(self) -> float:
        ms = super().warmup()
        if self._net_b is not None:
            self._net_b.setInput(np.zeros((self.static_batch, 3, self.input_size, self.input_size), np.float32))
            self._net_b.forward()
        return ms


def make_server_detector() -> NudeNetDetector:
    """COOL / classic importer → static copies; OpenCV 5 → the stock graph.

    ``SG_DETECTOR=static|stock`` forces one (benchmarks)."""
    want = os.environ.get("SG_DETECTOR", "").strip().lower()
    static_ok = (MODELS_DIR / "320n_static_b1.onnx").is_file() or bool(os.environ.get("SG_MODEL_DIR"))
    if want == "static" or (want != "stock" and not str(cv2.__version__).startswith("5.") and static_ok):
        return StaticBatchDetector(batch=int(os.environ.get("SG_STATIC_BATCH", "4")))
    return NudeNetDetector(backend="cv2")


_SHARED: NudeNetDetector | None = None


def shared_detector() -> NudeNetDetector:
    """One net per process (L2 second look + the legacy live_perceive route)."""
    global _SHARED
    if _SHARED is None:
        _SHARED = make_server_detector()
    return _SHARED


def live_box_fn(bgr: np.ndarray) -> list[dict]:
    """``detect_nsfw`` ``box_fn`` on the shared live detector: every maskable
    class per anchor (``vision_core.live.detector.decode_yolov8``), NudeNet
    ``[x, y, w, h]`` boxes. Replaces the wheel's top-class ``_postprocess``."""
    return [
        {"class": d.label, "score": float(d.score), "box": [d.xyxy[0], d.xyxy[1], d.xyxy[2] - d.xyxy[0], d.xyxy[3] - d.xyxy[1]]}
        for d in shared_detector().detect(bgr)
    ]
