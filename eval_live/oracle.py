"""Oracle classifier: a region is UNSAFE iff it overlaps a ``flagged`` GT picture.

The corpus is benign, so the NSFW model's accuracy **cannot** be measured
here. The oracle replaces the model with ground truth, which isolates what
*can* be measured: picture proposals, mask semantics (whole picture vs part
boxes), scroll follow, persistence and the engine's timing.

Adapters, in order of preference:

- ``OracleDetector`` — a ``PartDetector`` (``detect_batch(crops)``) injected
  into the real analyzer. It locates each crop on the current frame (numpy
  view offset, else template match) and returns part boxes of flagged
  pictures inside it, so the analyzer's own proposals, cache, decide and
  mask code all run.
- ``OracleWrappedAnalyzer`` — when injection is unsupported: the real
  analyzer's proposals, oracle verdicts, contract mask semantics.
- ``GTProposalAnalyzer`` — perfect proposals (GT rects) + oracle verdicts.
  Isolates the engine from proposal quality.
- ``LegacyOracle.box_fn`` — NudeNet-shaped raw boxes for
  ``make_nsfw_detect(box_fn=...)`` (the v1 baseline).

Score variants (``ORACLE_VARIANTS``, ``eval_live.run --oracle``):

- ``classic`` — the original: ``OracleDetector`` injects ``RawDet``
  ``FEMALE_BREAST_EXPOSED`` 0.9, skipping the decoder.
- ``realistic`` — ``oracle_raw_net``: a raw YOLOv8 tensor through the real
  letterbox / decode / NMS (``eval_live.fakenet``), EXPOSED 0.77–0.87, parts
  under 4 model px invisible.
- ``confusion`` — same anchor scores ``FEMALE_BREAST_COVERED`` 0.55 (top) and
  ``FEMALE_BREAST_EXPOSED`` 0.50 (runner-up): only a multi-class decode sees
  the exposed hit.
- ``covered`` — covered classes only, as on the user's real screen
  (``FEMALE_BREAST_COVERED`` 0.72, ``FEMALE_GENITALIA_COVERED`` 0.64).
- ``unsure`` — EXPOSED 0.38: UNSURE in medium (0.30–0.45), so every flagged
  picture needs a token-bucket second look.
- ``unsure_shift`` — ``unsure`` with the analyzer's clock shifted by +1 s:
  the control for the cold start at ``t_ms = 0`` that every sequence has
  (``eval_live.sequences`` stamps frame 0 at 0 ms).

Variants apply to the injected-detector path (``--analyzer auto|real``) and
the legacy ``box_fn``; ``gt`` / ``regions`` keep oracle verdicts.
"""

from __future__ import annotations

import inspect
import time
import weakref
from typing import Iterable, Iterator, Sequence as Seq

import cv2
import numpy as np

from eval_live.sequences import FrameGT, GTPicture
from vision_core.live.contracts import (
    AnalyzeResult,
    Box,
    MaskSpec,
    MaskStyle,
    PictureRegion,
    ProtectionMode,
    Verdict,
    VerdictState,
)

ORACLE_LABEL = "FEMALE_BREAST_EXPOSED"  # an explicit class: masks in every mode/policy
ORACLE_SCORE = 0.9
REGION_OVERLAP = 0.5  # region ∩ picture ≥ 50% of either → that picture's verdict

# variant → per-part class scores cycled over a picture's parts (None = classic RawDet oracle)
ORACLE_VARIANTS: dict[str, tuple[dict[str, float], ...] | None] = {
    "classic": None,
    "realistic": (
        {"FEMALE_BREAST_EXPOSED": 0.87, "FEMALE_BREAST_COVERED": 0.20},
        {"FEMALE_BREAST_EXPOSED": 0.77},
        {"FEMALE_GENITALIA_EXPOSED": 0.82, "FEMALE_GENITALIA_COVERED": 0.25},
    ),
    "confusion": ({"FEMALE_BREAST_COVERED": 0.55, "FEMALE_BREAST_EXPOSED": 0.50},),
    "covered": ({"FEMALE_BREAST_COVERED": 0.72}, {"FEMALE_GENITALIA_COVERED": 0.64}),
    "unsure": ({"FEMALE_BREAST_EXPOSED": 0.38},),
    "unsure_shift": ({"FEMALE_BREAST_EXPOSED": 0.38},),
}
TIME_SHIFT_MS = {"unsure_shift": 1000.0}
_VARIANT = "classic"


def set_oracle_variant(name: str) -> None:
    """Select the score variant used by ``make_live_analyzer`` / ``LegacyOracle``."""
    global _VARIANT
    if name not in ORACLE_VARIANTS:
        raise ValueError(f"unknown oracle variant {name!r}; expected one of {sorted(ORACLE_VARIANTS)}")
    _VARIANT = name


def oracle_variant() -> str:
    return _VARIANT


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _ptr(a: np.ndarray) -> int:
    return int(a.__array_interface__["data"][0])


class OracleTruth:
    """Per-sequence GT, keyed by the frame objects the harness hands out."""

    def __init__(self, gts: Seq[FrameGT], width: int, height: int) -> None:
        self.gts = list(gts)
        self.width = int(width)
        self.height = int(height)
        self.index = 0
        self._by_ptr: dict[int, tuple[int, weakref.ref]] = {}
        self._latest: np.ndarray | None = None
        self._latest_small: np.ndarray | None = None
        self.stats = {"located_view": 0, "located_full": 0, "located_match": 0, "unlocated": 0}

    # -- registration -------------------------------------------------------
    def register(self, frame: np.ndarray, index: int) -> None:
        self._by_ptr[_ptr(frame)] = (int(index), weakref.ref(frame))
        self.index = int(index)
        self._latest = frame
        self._latest_small = None

    def iter_frames(self, frames: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        for i, frame in enumerate(frames):
            self.register(frame, i)
            yield frame

    def set_index(self, index: int) -> None:
        self.index = int(index)

    @property
    def current(self) -> FrameGT:
        return self.gts[self.index]

    # -- locating images ----------------------------------------------------
    def _registered(self, arr: np.ndarray) -> tuple[int, np.ndarray] | None:
        hit = self._by_ptr.get(_ptr(arr))
        if hit is None:
            return None
        frame = hit[1]()
        if frame is None or frame.shape != arr.shape:
            return None
        return hit[0], frame

    def locate(self, image: np.ndarray) -> tuple[int, float, float, float, float] | None:
        """``(gt_index, x0, y0, sx, sy)``: image pixel (u, v) ↦ screen (x0 + u·sx, y0 + v·sy)."""
        # 1) a registered frame, or a numpy view into one
        node = image
        while isinstance(node, np.ndarray):
            hit = self._registered(node)
            if hit is not None:
                idx, frame = hit
                off = _ptr(image) - _ptr(frame)
                s0, s1 = frame.strides[0], frame.strides[1]
                if off == 0 and image.shape == frame.shape:
                    self.stats["located_view"] += 1
                    return idx, 0.0, 0.0, 1.0, 1.0
                if frame.flags.c_contiguous and image.strides[:2] == frame.strides[:2] and 0 <= off:
                    y0, rem = divmod(off, s0)
                    x0 = rem // s1
                    self.stats["located_view"] += 1
                    return idx, float(x0), float(y0), 1.0, 1.0
                break
            node = node.base
        latest = self._latest
        if latest is None or image.ndim < 2 or image.size == 0:
            self.stats["unlocated"] += 1
            return None
        H, W = latest.shape[:2]
        h, w = image.shape[:2]
        # 2) a resized copy of the whole latest frame
        if abs(w / h - W / H) < 0.01 and w <= W:
            a = cv2.resize(_gray(latest), (64, 40), interpolation=cv2.INTER_AREA).astype(np.float32)
            b = cv2.resize(_gray(image), (64, 40), interpolation=cv2.INTER_AREA).astype(np.float32)
            if float(np.mean(np.abs(a - b))) < 4.0:
                self.stats["located_full"] += 1
                return self.index, 0.0, 0.0, W / w, H / h
        # 3) an unscaled crop copy of the latest frame: template match at 1/4
        if 16 <= w <= W and 16 <= h <= H:
            q = 4
            if self._latest_small is None:
                self._latest_small = cv2.resize(_gray(latest), (W // q, H // q), interpolation=cv2.INTER_AREA)
            tpl = cv2.resize(_gray(image), (max(4, w // q), max(4, h // q)), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(self._latest_small, tpl, cv2.TM_CCOEFF_NORMED)
            _, best, _, loc = cv2.minMaxLoc(res)
            if best >= 0.9:
                self.stats["located_match"] += 1
                return self.index, float(loc[0] * q), float(loc[1] * q), 1.0, 1.0
        self.stats["unlocated"] += 1
        return None

    def gt_for(self, frame: np.ndarray) -> tuple[FrameGT, float, float]:
        """GT for a whole frame + scale (frame px → screen px)."""
        loc = self.locate(frame)
        if loc is None:
            h, w = frame.shape[:2]
            return self.current, self.width / w, self.height / h
        return self.gts[loc[0]], loc[3], loc[4]

    # -- verdicts -----------------------------------------------------------
    @staticmethod
    def flagged_under(gt: FrameGT, box) -> list[GTPicture]:
        out = []
        for p in gt.flagged():
            inter = _inter(box, p.xyxy)
            if inter is None:
                continue
            a = _area(inter)
            if a >= REGION_OVERLAP * _area(box) or a >= REGION_OVERLAP * max(1.0, p.visible_area):
                out.append(p)
        return out

    def verdict(self, gt: FrameGT, box) -> Verdict:
        t0 = time.perf_counter()
        hits = self.flagged_under(gt, box)
        if not hits:
            return Verdict(VerdictState.SAFE, score=0.0, source="oracle", latency_ms=(time.perf_counter() - t0) * 1000)
        parts = tuple(tuple(map(float, b)) for p in hits for b in p.parts) or tuple(tuple(map(float, p.xyxy)) for p in hits)
        return Verdict(
            VerdictState.UNSAFE,
            score=ORACLE_SCORE,
            labels=(ORACLE_LABEL,),
            parts=parts,
            source="oracle",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


# --------------------------------------------------------------------------- injected detector


class OracleDetector:
    """``PartDetector``-shaped oracle (``vision_core.live.detector``).

    Each crop is located on the current frame; it returns one RawDet per
    flagged-picture part inside the crop, in crop pixels.
    """

    def __init__(self, truth: OracleTruth, label: str = ORACLE_LABEL, score: float = ORACLE_SCORE) -> None:
        self.truth = truth
        self.label = label
        self.score = score
        self.calls = 0
        self.crops = 0

    def _raw(self, label: str, score: float, xyxy):
        try:
            from vision_core.live.detector import RawDet

            return RawDet(label=label, score=float(score), xyxy=tuple(float(v) for v in xyxy))
        except ImportError:  # pragma: no cover - detector module not landed yet
            return {"label": label, "score": float(score), "xyxy": tuple(float(v) for v in xyxy)}

    def detect(self, bgr: np.ndarray):
        return self.detect_batch([bgr])[0]

    def detect_batch(self, images: Seq[np.ndarray]):
        self.calls += 1
        out = []
        for img in images:
            self.crops += 1
            loc = self.truth.locate(img)
            if loc is None:
                out.append([])
                continue
            idx, x0, y0, sx, sy = loc
            h, w = img.shape[:2]
            screen = (x0, y0, x0 + w * sx, y0 + h * sy)
            dets = []
            for p in self.truth.gts[idx].flagged():
                for part in p.parts:
                    inter = _inter(part, screen)
                    if inter is None or _area(inter) < 0.5 * _area(part):
                        continue
                    dets.append(
                        self._raw(
                            self.label,
                            self.score,
                            ((inter[0] - x0) / sx, (inter[1] - y0) / sy, (inter[2] - x0) / sx, (inter[3] - y0) / sy),
                        )
                    )
            out.append(dets)
        return out

    # some detector wrappers expose these
    def warmup(self) -> float:
        return 0.0


def oracle_raw_net(truth: OracleTruth, profile: Seq[dict[str, float]], **kw):
    """``FakeRawNet`` whose targets are the GT parts under each crop, scored
    with ``profile`` (cycled over parts). Real letterbox / decode / NMS."""
    from eval_live.fakenet import FakeRawNet, FakeTarget

    def targets(img: np.ndarray):
        loc = truth.locate(img)
        if loc is None:
            return []
        idx, x0, y0, sx, sy = loc
        h, w = img.shape[:2]
        screen = (x0, y0, x0 + w * sx, y0 + h * sy)
        out = []
        k = 0
        for p in truth.gts[idx].flagged():
            for part in p.parts:
                inter = _inter(part, screen)
                if inter is None or _area(inter) < 0.5 * _area(part):
                    continue
                box = ((inter[0] - x0) / sx, (inter[1] - y0) / sy, (inter[2] - x0) / sx, (inter[3] - y0) / sy)
                out.append(FakeTarget(box, profile[k % len(profile)]))
                k += 1
        return out

    return FakeRawNet(profile[0], targets_fn=targets, **kw)


def make_oracle_detector(truth: OracleTruth, variant: str | None = None):
    """``PartDetector`` for ``variant`` (default: the selected one)."""
    profile = ORACLE_VARIANTS[variant or _VARIANT]
    if profile is None:
        return OracleDetector(truth)
    return oracle_raw_net(truth, profile)


class TimeShifted:
    """Forwards to an analyzer, adding ``shift_ms`` to every ``t_ms``."""

    def __init__(self, inner, shift_ms: float) -> None:
        self._inner = inner
        self.shift_ms = float(shift_ms)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def set_mode(self, mode: ProtectionMode) -> None:
        self._inner.set_mode(mode)

    def analyze(self, frame_bgr: np.ndarray, *, changed: Box | None = None, t_ms: float = 0.0) -> AnalyzeResult:
        return self._inner.analyze(frame_bgr, changed=changed, t_ms=float(t_ms) + self.shift_ms)


# --------------------------------------------------------------------------- analyzers


def masks_for(mode: ProtectionMode, regions: Seq[PictureRegion], verdicts: dict[str, Verdict], *, pad: float = 0.0) -> list[MaskSpec]:
    """Contract semantics: AGGRESSIVE = whole picture; MEDIUM = padded parts."""
    if mode == ProtectionMode.OFF:
        return []
    out: list[MaskSpec] = []
    for r in regions:
        v = verdicts.get(r.region_id)
        if v is None or v.state == VerdictState.SAFE:
            continue
        if mode == ProtectionMode.AGGRESSIVE:
            out.append(MaskSpec(r.region_id, r.xyxy, MaskStyle.FROST, f"{v.state.value}_PICTURE", v.state))
            continue
        if v.state != VerdictState.UNSAFE:
            continue
        for k, b in enumerate(v.parts):
            out.append(MaskSpec(f"{r.region_id}#{k}", (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad),
                                MaskStyle.FROST, "UNSAFE_PARTS", v.state))
    return out


class GTProposalAnalyzer:
    """Perfect proposals (visible GT picture rects) + oracle verdicts."""

    name = "gt_proposals+oracle"

    def __init__(self, truth: OracleTruth, mode: ProtectionMode = ProtectionMode.AGGRESSIVE) -> None:
        self.truth = truth
        self.mode = mode

    def set_mode(self, mode: ProtectionMode) -> None:
        self.mode = mode

    def analyze(self, frame_bgr: np.ndarray, *, changed: Box | None = None, t_ms: float = 0.0) -> AnalyzeResult:
        del changed, t_ms
        t0 = time.perf_counter()
        gt, sx, sy = self.truth.gt_for(frame_bgr)
        regions = []
        verdicts = {}
        for p in gt.pictures:
            box = (p.xyxy[0] / sx, p.xyxy[1] / sy, p.xyxy[2] / sx, p.xyxy[3] / sy)
            regions.append(PictureRegion(p.gid, box, kind=p.kind, picture_score=1.0))
            v = self.truth.verdict(gt, p.xyxy)
            if sx != 1.0 or sy != 1.0:
                v.parts = tuple((b[0] / sx, b[1] / sy, b[2] / sx, b[3] / sy) for b in v.parts)
            verdicts[p.gid] = v
        res = AnalyzeResult(regions, verdicts, masks_for(self.mode, regions, verdicts))
        res.timings_ms["total"] = (time.perf_counter() - t0) * 1000
        return res


class OracleWrappedAnalyzer:
    """Real analyzer proposals; verdicts + masks recomputed with the oracle."""

    def __init__(self, inner, truth: OracleTruth, mode: ProtectionMode = ProtectionMode.AGGRESSIVE) -> None:
        self.inner = inner
        self.truth = truth
        self.mode = mode
        self.name = f"{type(inner).__name__}_proposals+oracle_wrap"

    def set_mode(self, mode: ProtectionMode) -> None:
        self.mode = mode
        if hasattr(self.inner, "set_mode"):
            self.inner.set_mode(mode)

    def analyze(self, frame_bgr: np.ndarray, *, changed: Box | None = None, t_ms: float = 0.0) -> AnalyzeResult:
        res = self.inner.analyze(frame_bgr, changed=changed, t_ms=t_ms)
        t0 = time.perf_counter()
        gt, sx, sy = self.truth.gt_for(frame_bgr)
        verdicts = {}
        for r in res.regions:
            box = (r.xyxy[0] * sx, r.xyxy[1] * sy, r.xyxy[2] * sx, r.xyxy[3] * sy)
            v = self.truth.verdict(gt, box)
            v.parts = tuple((b[0] / sx, b[1] / sy, b[2] / sx, b[3] / sy) for b in v.parts)
            verdicts[r.region_id] = v
        out = AnalyzeResult(list(res.regions), verdicts, masks_for(self.mode, res.regions, verdicts), dict(res.timings_ms))
        out.timings_ms["oracle"] = (time.perf_counter() - t0) * 1000
        return out


class ProposerAnalyzer:
    """Bare ``vision_core.live.regions`` proposals (no verdicts, no masks).

    Used under ``OracleWrappedAnalyzer`` before the full analyzer lands.
    """

    def __init__(self, mode: ProtectionMode = ProtectionMode.AGGRESSIVE) -> None:
        from vision_core.live.regions import RegionProposer

        self.mode = mode
        self.proposer = RegionProposer()

    def set_mode(self, mode: ProtectionMode) -> None:
        self.mode = mode

    def analyze(self, frame_bgr: np.ndarray, *, changed: Box | None = None, t_ms: float = 0.0) -> AnalyzeResult:
        del t_ms
        t0 = time.perf_counter()
        regions = self.proposer.propose(frame_bgr, changed)
        return AnalyzeResult(regions=list(regions), timings_ms={"regions": (time.perf_counter() - t0) * 1000})


def _regions_wrap(mode: ProtectionMode, truth: OracleTruth, why: str):
    inner = ProposerAnalyzer(mode)
    return (
        OracleWrappedAnalyzer(inner, truth, mode),
        "regions_proposals+oracle_wrap",
        f"{why}; vision_core.live.regions.RegionProposer proposals + oracle verdicts + contract mask semantics",
    )


class NullAnalyzer:
    """Eval baseline: finds no pictures (never used by the live engine)."""

    def __init__(self, mode: ProtectionMode = ProtectionMode.AGGRESSIVE) -> None:
        self.mode = mode

    def set_mode(self, mode: ProtectionMode) -> None:
        self.mode = mode

    def analyze(self, frame_bgr, *, changed=None, t_ms=None, **_kw) -> AnalyzeResult:
        return AnalyzeResult()


def make_live_analyzer(mode: ProtectionMode, truth: OracleTruth, impl: str = "auto"):
    """Return ``(analyzer, impl_label, note)``.

    ``impl``: ``auto`` (real analyzer with the oracle injected, else wrapped;
    if the analyzer has not landed, ``regions`` proposals; else the stub),
    ``real`` (raise if the analyzer is not importable), ``regions``
    (``RegionProposer`` + oracle), ``gt`` (perfect proposals), ``stub``
    (``NullAnalyzer``: no pictures at all; eval only).
    """
    if impl == "regions":
        return _regions_wrap(mode, truth, "requested")
    if impl == "gt":
        return GTProposalAnalyzer(truth, mode), GTProposalAnalyzer.name, "perfect proposals (GT rects) + oracle verdicts"
    if impl == "stub":
        return NullAnalyzer(mode), "stub", "NullAnalyzer (eval only): returns no regions/masks"
    try:
        from vision_core.live.analyzer import make_analyzer
    except Exception as exc:  # noqa: BLE001 - any import failure means "not landed yet"
        if impl == "real":
            raise
        why = f"vision_core.live.analyzer not importable ({type(exc).__name__}: {exc})"
        try:
            return _regions_wrap(mode, truth, why)
        except Exception:  # noqa: BLE001
            pass
        return NullAnalyzer(mode), "stub", f"vision_core.live.analyzer not importable ({type(exc).__name__}: {exc}); using NullAnalyzer"
    det = make_oracle_detector(truth)
    shift = TIME_SHIFT_MS.get(_VARIANT, 0.0)
    try:
        params = inspect.signature(make_analyzer).parameters
    except (TypeError, ValueError):
        params = {}
    accepts_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    for key in ("detector", "part_detector", "detector_fn"):
        if key in params or accepts_kw:
            try:
                an = make_analyzer(mode, **{key: det})
            except TypeError:
                continue
            an._eval_oracle = det  # type: ignore[attr-defined]
            label = f"real+oracle_detector({key})" + ("" if _VARIANT == "classic" else f"[{_VARIANT}]")
            note = f"{type(det).__name__} ({_VARIANT}) injected via make_analyzer({key}=...)"
            if shift:
                return TimeShifted(an, shift), label, f"{note}; t_ms + {shift:.0f}"
            return an, label, note
    an = make_analyzer(mode)
    return OracleWrappedAnalyzer(an, truth, mode), "real_proposals+oracle_wrap", "detector injection unsupported; oracle wraps proposals"


# --------------------------------------------------------------------------- legacy baseline


class LegacyOracle:
    """``box_fn`` for ``make_nsfw_detect``: NudeNet-shaped part boxes on flagged pictures.

    ``detect_nsfw`` calls ``box_fn`` with a 960-px downscale of the frame
    (or a reinspect crop). Boxes are returned in original pixels
    (``"space": "orig"``), so the downscale does not matter.
    """

    def __init__(self, truth: OracleTruth) -> None:
        self.truth = truth
        self.calls = 0
        # The wheel decodes top-class-only: keep each variant's top class.
        profile = ORACLE_VARIANTS[_VARIANT]
        self.profile = (
            ((ORACLE_LABEL, ORACLE_SCORE),)
            if profile is None
            else tuple(max(d.items(), key=lambda kv: kv[1]) for d in profile)
        )

    def box_fn(self, image: np.ndarray) -> list[dict]:
        self.calls += 1
        loc = self.truth.locate(image)
        if loc is None:
            return []
        idx, x0, y0, sx, sy = loc
        h, w = image.shape[:2]
        screen = (x0, y0, x0 + w * sx, y0 + h * sy)
        full = (x0, y0) == (0.0, 0.0) and sx >= 1.0 and abs(w * sx - self.truth.width) < 2
        out = []
        k = 0
        for p in self.truth.gts[idx].flagged():
            for part in p.parts:
                inter = _inter(part, screen)
                if inter is None:
                    continue
                label, score = self.profile[k % len(self.profile)]
                k += 1
                if full:
                    # orig = full-frame pixels; detect_nsfw adds no offset
                    x, y = inter[0], inter[1]
                else:
                    # crop path: work-space pixels of this crop
                    x, y = (inter[0] - x0) / sx, (inter[1] - y0) / sy
                bw, bh = (inter[2] - inter[0]), (inter[3] - inter[1])
                item = {"class": label, "score": score, "box": [x, y, bw, bh]}
                if full:
                    item["space"] = "orig"
                else:
                    item["box"] = [x, y, bw / sx, bh / sy]
                out.append(item)
        return out
