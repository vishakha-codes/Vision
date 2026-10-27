"""L2 "COOL second look": multi-scale NudeNet 320n on one picture crop.

The Mac (L1) escalates a picture it could not settle (UNSURE after its own
second look, a large picture with weak scores, or no local detector). It
sends one JPEG of the picture rect plus the picture's dHash / aHash. This
module, in RAM only:

1. decodes the JPEG (bounded size / edge);
2. builds the views: whole, mirrored whole, a 2×2 grid at 0.6 × the long
   side (``deep``, for the agent review, adds a 3×3 grid at 0.4 ×);
3. runs them as one letterboxed batch on the server detector (COOL);
4. keeps a class box only when several views agree (consensus, k=2 for
   ``l2``, k=3 for ``deep``), merged with ``NMSBoxesBatched`` in
   picture-relative coords (0..1), so the Mac can drop them straight into
   its evidence cache;
5. returns the mode verdict (same ``vision_core.live.modes.decide``).

Evidence is cached by (dHash, aHash, profile) in an LRU (no pixels). A
picture that is still UNSURE is held in RAM for ``REVIEW_HOLD_S`` so the
AgentCore agent can ask for a deeper review; the hold is dropped after the
review or the TTL. Nothing is written to disk.
"""

from __future__ import annotations

import base64
import binascii
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np

from vision_core.live.contracts import PictureRegion, ProtectionMode, VerdictState
from vision_core.live.detector import NudeNetDetector
from vision_core.live.modes import Evidence, NormDet, decide, load_mode_policy
from vision_core.live.regions import tile_grid

MAX_JPEG_BYTES = 400_000
MAX_B64_CHARS = 540_000
MAX_EDGE = 1600
MIN_EDGE = 16
CONSENSUS_K = {"l2": 2, "deep": 3}  # views that must agree on a class box
CONSENSUS_IOU = 0.2
CACHE_ENTRIES = 2048
CACHE_TTL_S = 15 * 60.0
REVIEW_HOLD_S = 120.0
REVIEW_HOLD_MAX = 32
PROFILES = ("l2", "deep")


class SecondLookError(ValueError):
    """Bad input (maps to HTTP 400)."""


def decode_jpeg_b64(jpeg_b64: str) -> tuple[np.ndarray, int]:
    if not jpeg_b64:
        raise SecondLookError("jpeg_b64 is required")
    if len(jpeg_b64) > MAX_B64_CHARS:
        raise SecondLookError("jpeg_b64 exceeds max size")
    try:
        raw = base64.b64decode(jpeg_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SecondLookError("jpeg_b64 is not valid base64") from exc
    if len(raw) > MAX_JPEG_BYTES:
        raise SecondLookError("jpeg exceeds max bytes")
    if raw[:2] != b"\xff\xd8":
        raise SecondLookError("payload must be JPEG")
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SecondLookError("jpeg is not decodable")
    h, w = img.shape[:2]
    if max(h, w) > MAX_EDGE or min(h, w) < MIN_EDGE:
        raise SecondLookError(f"crop edge must be {MIN_EDGE}..{MAX_EDGE} px")
    return img, len(raw)


def views_for(w: int, h: int, profile: str) -> list[tuple[tuple[float, float, float, float], bool]]:
    """(box in crop px, mirrored) per view.

    ``l2``: whole, mirrored whole, 2×2 grid at 0.6 × the long side (≤ 6 views).
    ``deep``: the same plus a 3×3 grid at 0.4 × (≤ 15 views)."""
    whole = (0.0, 0.0, float(w), float(h))
    long_side = float(max(w, h))
    out: list[tuple[tuple[float, float, float, float], bool]] = [(whole, False), (whole, True)]
    out += [(b, False) for b in tile_grid(whole, max(160.0, 0.6 * long_side), 0.25)]
    if profile == "deep":
        out += [(b, False) for b in tile_grid(whole, max(128.0, 0.4 * long_side), 0.2)]
    # tile_grid returns the whole box when a side fits in one tile: dedupe.
    seen: set = set()
    uniq = []
    for box, flip in out:
        key = (tuple(round(v) for v in box), flip)
        if key not in seen:
            seen.add(key)
            uniq.append((box, flip))
    return uniq


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _nms_norm(dets: list[tuple[str, float, tuple[float, float, float, float]]], iou: float = 0.45) -> list[NormDet]:
    if not dets:
        return []
    labels = sorted({d[0] for d in dets})
    ids = [labels.index(d[0]) for d in dets]
    xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for _, _, b in dets]
    keep = cv2.dnn.NMSBoxesBatched(xywh, [d[1] for d in dets], ids, 0.0, iou)
    out = [NormDet(dets[int(i)][0], float(dets[int(i)][1]), dets[int(i)][2]) for i in np.asarray(keep).reshape(-1)]
    out.sort(key=lambda d: -d.score)
    return out


def consensus(per_view: list[list[tuple[str, float, tuple]]], k: int, iou: float = CONSENSUS_IOU) -> list[NormDet]:
    """Keep a class box only if ``k`` views see it (same class, IoU > ``iou``
    in picture coords); its score is the k-th best supporting score.

    Why: on benign procedural textures, max-over-views of fine tiles flags
    55% of pictures in AGGRESSIVE (3×3 tiles hallucinate weak parts);
    whole + mirror + 2×2 with k=2 flags 12%, k=3 2.7%
    (``experiments/cool_live/consensus_fp.json``)."""
    found: list[tuple[str, float, tuple]] = []
    for i, dets in enumerate(per_view):
        for label, score, box in dets:
            support = [score]
            for j, other in enumerate(per_view):
                if j == i:
                    continue
                best = max((s for lab, s, b in other if lab == label and _iou(box, b) > iou), default=None)
                if best is not None:
                    support.append(best)
            if len(support) >= k:
                found.append((label, sorted(support, reverse=True)[k - 1], box))
    return _nms_norm(found)


@dataclass
class LookResult:
    dets: tuple[NormDet, ...]
    n_views: int
    infer_ms: float
    prep_ms: float
    raw_top: float = 0.0  # best single-view score before consensus (logs)


def run_views(detector: NudeNetDetector, img: np.ndarray, profile: str) -> LookResult:
    t0 = time.perf_counter()
    h, w = img.shape[:2]
    views = views_for(w, h, profile)
    crops: list[np.ndarray] = []
    for (x1, y1, x2, y2), flip in views:
        c = img[int(math.floor(y1)) : int(math.ceil(y2)), int(math.floor(x1)) : int(math.ceil(x2))]
        crops.append(cv2.flip(c, 1) if flip else c)
    t1 = time.perf_counter()
    raw = detector.detect_batch(crops)
    t2 = time.perf_counter()
    per_view: list[list[tuple[str, float, tuple[float, float, float, float]]]] = []
    for ((x1, y1, x2, y2), flip), dets in zip(views, raw):
        vw = x2 - x1
        lst = []
        for d in dets:
            bx1, by1, bx2, by2 = d.xyxy
            if flip:
                bx1, bx2 = vw - bx2, vw - bx1
            lst.append((d.label, float(d.score), (
                min(1.0, max(0.0, (bx1 + x1) / w)), min(1.0, max(0.0, (by1 + y1) / h)),
                min(1.0, max(0.0, (bx2 + x1) / w)), min(1.0, max(0.0, (by2 + y1) / h)))))
        per_view.append(lst)
    k = CONSENSUS_K[profile] if len(views) >= CONSENSUS_K[profile] else max(1, len(views))
    dets = tuple(consensus(per_view, k))
    top = max((s for lst in per_view for _, s, _ in lst), default=0.0)
    return LookResult(dets, len(views), (t2 - t1) * 1000.0, (t1 - t0) * 1000.0, top)


class _TTLCache:
    def __init__(self, max_entries: int, ttl_s: float) -> None:
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return None
            t, value = item
            if time.monotonic() - t > self.ttl_s:
                self._d.pop(key, None)
                return None
            self._d.move_to_end(key)
            return value

    def put(self, key, value) -> None:
        with self._lock:
            self._d[key] = (time.monotonic(), value)
            self._d.move_to_end(key)
            while len(self._d) > self.max_entries:
                self._d.popitem(last=False)

    def pop(self, key):
        with self._lock:
            item = self._d.pop(key, None)
        if item is None or time.monotonic() - item[0] > self.ttl_s:
            return None
        return item[1]

    def __len__(self) -> int:
        return len(self._d)


def parse_hash(value: str | int | None) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, int):
        return value & ((1 << 64) - 1)
    try:
        return int(str(value), 16) & ((1 << 64) - 1)
    except ValueError as exc:
        raise SecondLookError("hashes must be hex strings") from exc


class SecondLookService:
    """One per process. The detector is not re-entrant: calls are serialised
    (the box has 2 vCPUs and OpenCV already uses both inside a forward)."""

    def __init__(self, detector: NudeNetDetector | None = None) -> None:
        self._detector = detector
        self._lock = threading.Lock()
        self.cache = _TTLCache(CACHE_ENTRIES, CACHE_TTL_S)
        self.holds = _TTLCache(REVIEW_HOLD_MAX, REVIEW_HOLD_S)
        self._policies: dict[ProtectionMode, object] = {}

    @property
    def detector(self) -> NudeNetDetector:
        if self._detector is None:
            from cloud.cool_detector import shared_detector

            self._detector = shared_detector()
        return self._detector

    def policy(self, mode: ProtectionMode):
        if mode not in self._policies:
            self._policies[mode] = load_mode_policy(mode)
        return self._policies[mode]

    def _verdict(self, mode: ProtectionMode, dets: tuple[NormDet, ...]) -> dict:
        reg = PictureRegion("cloud", (0.0, 0.0, 1.0, 1.0))
        ev = Evidence((), dets, 0.0, 0)  # the cloud look is a second look
        v = decide(self.policy(mode), reg, ev, source="cloud")
        return {"state": v.state.value, "score": round(float(v.score), 4), "labels": list(v.labels)}

    def look(self, *, jpeg_b64: str, dhash: str | int | None, ahash: str | int | None, mode: str, profile: str = "l2") -> dict:
        t0 = time.perf_counter()
        if profile not in PROFILES:
            raise SecondLookError(f"profile must be one of {PROFILES}")
        try:
            pmode = ProtectionMode(mode)
        except ValueError as exc:
            raise SecondLookError("mode must be aggressive|medium|off") from exc
        dh, ah = parse_hash(dhash), parse_hash(ahash)
        key = (dh, ah, profile)
        cached = self.cache.get(key) if dh else None
        if cached is not None:
            dets, n_views, infer_ms = cached
            payload_bytes = 0
            source = "cloud_cache"
            img = None
        else:
            img, payload_bytes = decode_jpeg_b64(jpeg_b64)
            with self._lock:
                res = run_views(self.detector, img, profile)
            dets, n_views, infer_ms = res.dets, res.n_views, res.infer_ms
            if dh:
                self.cache.put(key, (dets, n_views, infer_ms))
            source = "cloud"
        verdict = self._verdict(pmode, dets)
        held = False
        if img is not None and dh and verdict["state"] == VerdictState.UNSURE.value and profile == "l2":
            self.holds.put(dh, img)  # RAM only, for an agent review
            held = True
        return {
            "dhash": f"{dh:016x}",
            "ahash": f"{ah:016x}",
            "mode": pmode.value,
            "profile": profile,
            "source": source,
            "verdict": verdict,
            "dets": [{"label": d.label, "score": round(d.score, 4), "box": [round(v, 4) for v in d.box]} for d in dets],
            "n_views": n_views,
            "payload_bytes": payload_bytes,
            "review_hold_s": REVIEW_HOLD_S if held else 0.0,
            "image_persisted": False,
            "timings_ms": {"infer": round(infer_ms, 2), "total": round((time.perf_counter() - t0) * 1000.0, 2)},
        }

    def review(self, *, dhash: str | int, mode: str) -> dict:
        """Deep look at a held crop (called by the agent's COOL tool)."""
        t0 = time.perf_counter()
        dh = parse_hash(dhash)
        pmode = ProtectionMode(mode)
        cached = self.cache.get((dh, 0, "deep"))
        if cached is None:
            img = self.holds.pop(dh)
            if img is None:
                return {"dhash": f"{dh:016x}", "found": False, "image_persisted": False,
                        "detail": "no held crop (expired or never UNSURE); decide from the event scores"}
            with self._lock:
                res = run_views(self.detector, img, "deep")
            del img
            cached = (res.dets, res.n_views, res.infer_ms)
            self.cache.put((dh, 0, "deep"), cached)
        dets, n_views, infer_ms = cached
        return {
            "dhash": f"{dh:016x}",
            "found": True,
            "mode": pmode.value,
            "profile": "deep",
            "verdict": self._verdict(pmode, dets),
            "dets": [{"label": d.label, "score": round(d.score, 4)} for d in dets[:8]],
            "n_views": n_views,
            "image_persisted": False,
            "timings_ms": {"infer": round(infer_ms, 2), "total": round((time.perf_counter() - t0) * 1000.0, 2)},
        }
