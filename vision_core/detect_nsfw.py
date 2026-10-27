"""On-device adult-imagery detector for ``--detect nsfw``.

This is **not** a production NSFW system and is **not** a Canopy
replacement. It runs NudeNet's bundled 320n ONNX detector in-process
(bounding boxes + class scores). Skin-color heuristics are not used as
the classifier.

Honest limits:
- First pixels of a page can appear before any mask (capture is after
  the desktop is already visible).
- Text, cartoons, and news faces are not masked. FACE_* / FEET_* /
  ARMPITS_* / BELLY_COVERED never emit.
- Balanced and strict mask suggestive / intimate classes NudeNet
  already returns (covered genitalia / breast, belly exposed, buttocks
  covered, plus every exposed intimate class). Relaxed stays
  explicit-only. Live floors prefer over-blur (balanced ~0.30, strict
  ~0.22). Art, sculpture, and medical imagery can still false-hit.
- This is **not** a Canopy replacement and **not** a production accuracy
  claim. Illustrated / thumbnail grids can still score below the floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from vision_core.cluster_nsfw import MAX_CLUSTERS, ClusterHold, apply_cluster_cap
from vision_core.contracts import Detection, Frame
from vision_core.geometry import as_int_box, box_area

# NudeNet 320n labels and class groups live in ``vision_core.nudenet_labels``
# (shared with the live path); re-exported here for the legacy / cloud API.
# FACE_* is never a mask, even at 0.99.
from vision_core.nudenet_labels import (  # noqa: F401  (re-exports)
    EXPLICIT_CLASSES,
    INTIMATE_CLASSES,
    NEVER_MASK_CLASSES,
    SUGGESTIVE_CLASSES,
    is_never_mask,
    label_bucket,
)

DETECTOR_ID = "nudenet_onnx_320n_v1"
MODELS_DIR = Path(__file__).resolve().parents[1] / ".models"
BUNDLED_NAME = "320n.onnx"

# Extra floor before a box is emitted. Prefer over-blur on live: balanced
# 0.30 so EXPOSED 0.39 / BUTTOCKS 0.44 stay on. Relaxed stays explicit-ish.
# Policy JSON / synthetic eval are not retuned — live overlay floors live
# in ``tune_policy_for_live_nsfw``.
POLICY_MIN_SCORE = {
    "strict": 0.22,
    "balanced": 0.30,
    "relaxed": 0.55,
}

LIVE_HOLD_SCORE = 0.20
LIVE_OVERLAY_ON = {
    "strict": 0.22,
    "balanced": 0.30,
    "relaxed": 0.55,
}

# Live reinspect token bucket (see ``agent.orchestrator.Orchestrator``).
LIVE_REINSPECT_REFILL_PER_S = 2.0
LIVE_REINSPECT_BURST = 8

_DEFAULT_MIN_SCORE = 0.30
_MAX_LONG_EDGE = 960
_MIN_AREA = 400
_DESKTOP_MIN_W = 1400
_DESKTOP_MIN_H = 800
_CONTENT_TILE = 640
_MAX_CONTENT_TILES = 12

BoxFn = Callable[[np.ndarray], list[dict]]
ScoreFn = Callable[[np.ndarray], float]

_DETECTOR = None


@dataclass
class DetectStats:
    """Last ``detect_nsfw`` call. Safe to read from the tick logger."""

    model_loaded: bool = False
    n_raw: int = 0
    top: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    max_score: float = 0.0
    max_explicit: float = 0.0
    max_intimate: float = 0.0
    emitted: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    threshold: float = _DEFAULT_MIN_SCORE
    policy_name: str = ""
    n_after_policy: int = 0
    n_clusters: int = 0
    cluster_cap: int = MAX_CLUSTERS
    clusters_capped: bool = False
    n_tiles: int = 0
    tiled: bool = False
    reason: str = "detect not called"


LAST_STATS = DetectStats()


def last_detect_stats() -> DetectStats:
    return LAST_STATS


def _normalize_policy(policy_name: str | None) -> str:
    key = (policy_name or "").strip().lower()
    if key in POLICY_MIN_SCORE:
        return key
    return "balanced"


def tune_policy_for_live_nsfw(cfg):
    """Lower live overlay / off floors. Does not rewrite policy JSON."""
    from dataclasses import replace

    name = str(getattr(cfg, "name", "") or "balanced")
    agent = dict(getattr(cfg, "agent", None) or {})
    # Live reinspects refill over time (token bucket in the orchestrator)
    # instead of a 40-per-process budget. Replay / synthetic eval never calls
    # this, so its 40-reinspect / IoU numbers are unchanged.
    agent.setdefault("reinspect_refill_per_s", LIVE_REINSPECT_REFILL_PER_S)
    agent["reinspect_budget"] = int(agent.get("live_reinspect_burst", LIVE_REINSPECT_BURST))
    if name == "relaxed":
        return replace(cfg, agent=agent)
    th = dict(getattr(cfg, "thresholds", None) or {})
    th["flagged_class_min_confidence"] = LIVE_HOLD_SCORE
    th["overlay_trigger_score"] = float(LIVE_OVERLAY_ON.get(name, 0.30))
    return replace(cfg, thresholds=th, agent=agent)


def emit_classes_for_policy(policy_name: str | None) -> frozenset[str]:
    """Labels that may become a mask box for this preset."""
    if _normalize_policy(policy_name) == "relaxed":
        return EXPLICIT_CLASSES
    return INTIMATE_CLASSES


def label_should_emit(label: str, policy_name: str | None) -> bool:
    if is_never_mask(label):
        return False
    return str(label or "") in emit_classes_for_policy(policy_name)


def _set_stats(stats: DetectStats) -> DetectStats:
    global LAST_STATS
    LAST_STATS = stats
    return stats


def _bgr(image: Frame | np.ndarray) -> np.ndarray:
    if isinstance(image, Frame):
        return image.image_bgr
    return image


def resolve_model_path() -> Path:
    """Prefer ``vision/.models/320n.onnx``, else the NudeNet wheel copy."""
    local = MODELS_DIR / BUNDLED_NAME
    if local.is_file():
        return local
    try:
        import nudenet

        bundled = Path(nudenet.__file__).resolve().parent / BUNDLED_NAME
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "nudenet is not installed in this venv. "
            "Use vision/.venv and pip install -r requirements.txt."
        ) from exc
    if not bundled.is_file():
        raise RuntimeError(
            f"NudeNet weights missing ({bundled}). "
            "Reinstall nudenet in vision/.venv or copy 320n.onnx to vision/.models/."
        )
    return bundled


def load_nudenet():
    """Load the ONNX session once. Safe to call at process start."""
    global _DETECTOR
    if _DETECTOR is None:
        from nudenet import NudeDetector

        _DETECTOR = NudeDetector(model_path=str(resolve_model_path()))
    return _DETECTOR


def nudenet_available() -> bool:
    try:
        return resolve_model_path().is_file()
    except RuntimeError:
        return False


def nudenet_color_input(bgr: np.ndarray) -> np.ndarray:
    """NudeNet's reader runs ``COLOR_RGBA2BGR`` then ``blobFromImage(swapRB)``.

    A 3-channel BGR array through ``COLOR_RGBA2BGR`` swaps R/B on OpenCV 5,
    so the ONNX session sees inverted chroma and misses real photographs.
    Feed true RGBA.
    """
    if bgr.ndim == 3 and bgr.shape[2] == 3:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        return cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGBA)
    return bgr


def _nudenet_boxes(bgr: np.ndarray) -> list[dict]:
    detector = load_nudenet()
    raw = detector.detect(nudenet_color_input(bgr))
    out: list[dict] = []
    for item in raw or []:
        out.append(
            {
                "class": str(item.get("class") or ""),
                "score": float(item.get("score") or 0.0),
                "box": list(item.get("box") or []),
            }
        )
    return out


def _downscale(bgr: np.ndarray, max_long: int = _MAX_LONG_EDGE) -> tuple[np.ndarray, float]:
    height, width = bgr.shape[:2]
    long_edge = max(width, height)
    if long_edge <= max_long:
        return bgr, 1.0
    scale = max_long / float(long_edge)
    resized = cv2.resize(
        bgr,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _xywh_to_xyxy(
    box: Iterable[float],
    width: int,
    height: int,
    *,
    inv_scale: float,
    offset: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    vals = list(box)
    if len(vals) != 4:
        return None
    x, y, w, h = (float(v) for v in vals)
    if w <= 1.0 and h <= 1.0 and x <= 1.0 and y <= 1.0:
        # normalized xywh
        x, y, w, h = x * width, y * height, w * width, h * height
    x1 = (x * inv_scale) + offset[0]
    y1 = (y * inv_scale) + offset[1]
    x2 = ((x + w) * inv_scale) + offset[0]
    y2 = ((y + h) * inv_scale) + offset[1]
    return tuple(float(v) for v in as_int_box((x1, y1, x2, y2), width, height))


def _tile_proposals(
    bgr: np.ndarray,
    score_fn: ScoreFn,
    *,
    tile: int = 256,
    stride: int = 192,
    min_score: float,
) -> list[dict]:
    """Image-level classifier fallback: sliding windows, not the whole display."""
    height, width = bgr.shape[:2]
    hits: list[dict] = []
    if height < 8 or width < 8:
        return hits
    y = 0
    while y < height:
        x = 0
        y2 = min(height, y + tile)
        while x < width:
            x2 = min(width, x + tile)
            crop = bgr[y:y2, x:x2]
            score = float(score_fn(crop))
            if score >= min_score:
                hits.append(
                    {
                        "class": "FEMALE_GENITALIA_EXPOSED",
                        "score": score,
                        "box": [float(x), float(y), float(x2 - x), float(y2 - y)],
                    }
                )
            if x2 >= width:
                break
            x += stride
        if y2 >= height:
            break
        y += stride
    return hits


def _should_tile(height: int, width: int) -> bool:
    return width >= _DESKTOP_MIN_W and height >= _DESKTOP_MIN_H


def _grid_origins(length: int, tile: int, max_steps: int = 4) -> list[int]:
    if length <= tile:
        return [0]
    last = max(0, length - tile)
    steps = min(max(2, int(max_steps)), 4)
    if last == 0:
        return [0]
    return sorted({int(round(i * last / (steps - 1))) for i in range(steps)})


def propose_content_tiles(
    bgr: np.ndarray,
    *,
    tile: int = _CONTENT_TILE,
    max_tiles: int = _MAX_CONTENT_TILES,
) -> list[tuple[int, int, int, int]]:
    """High-variance 640px crops so SERP thumbs / a large photo can fire."""
    height, width = bgr.shape[:2]
    tw = min(tile, width)
    th = min(tile, height)
    if tw < 32 or th < 32:
        return []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    scored: list[tuple[float, int, int, int, int]] = []
    for y in _grid_origins(height, th):
        for x in _grid_origins(width, tw):
            crop = gray[y : y + th, x : x + tw]
            if crop.size == 0:
                continue
            scored.append((float(crop.std()), x, y, x + tw, y + th))
    scored.sort(reverse=True)
    chosen: list[tuple[int, int, int, int]] = []
    area = float(tw * th)
    for std, x1, y1, x2, y2 in scored:
        if std < 8.0:
            continue
        overlap = False
        for ox1, oy1, ox2, oy2 in chosen:
            ix = max(0, min(x2, ox2) - max(x1, ox1))
            iy = max(0, min(y2, oy2) - max(y1, oy1))
            if (ix * iy) > 0.55 * area:
                overlap = True
                break
        if overlap:
            continue
        chosen.append((x1, y1, x2, y2))
        if len(chosen) >= max_tiles:
            break
    return chosen


def _content_tile_boxes(bgr: np.ndarray, box_fn: BoxFn | None = None) -> tuple[list[dict], int]:
    tiles = propose_content_tiles(bgr)
    runner = box_fn or _nudenet_boxes
    raw: list[dict] = []
    for x1, y1, x2, y2 in tiles:
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        for item in runner(crop):
            box = list(item.get("box") or [])
            if len(box) == 4:
                box = [float(box[0]) + x1, float(box[1]) + y1, float(box[2]), float(box[3])]
            raw.append(
                {
                    "class": str(item.get("class") or item.get("label") or ""),
                    "score": float(item.get("score") or 0.0),
                    "box": box,
                    "space": "orig",
                }
            )
    return raw, len(tiles)


def _top_raw(raw: list[dict], n: int = 3) -> tuple[tuple[str, float], ...]:
    ranked = sorted(raw, key=lambda item: -float(item.get("score") or 0.0))
    out: list[tuple[str, float]] = []
    for item in ranked[:n]:
        out.append((str(item.get("class") or "-"), float(item.get("score") or 0.0)))
    return tuple(out)


def _fmt_hits(hits: list[tuple[str, float]] | tuple[tuple[str, float], ...], n: int = 4) -> str:
    ranked = sorted(hits, key=lambda item: -float(item[1]))
    return " ".join(f"{name}:{score:.2f}" for name, score in ranked[:n])


def _stats_reason(
    *,
    policy_name: str,
    floor: float,
    raw: list[dict],
    n_after: int,
    emitted: list[tuple[str, float]] | tuple[tuple[str, float], ...],
) -> str:
    policy = policy_name or "balanced"
    if n_after > 0 and emitted:
        return f"mask={_fmt_hits(emitted)}"
    if not raw:
        return f"nudenet raw=0 max=0.00 < {policy} {floor:.2f}; model_loaded={_DETECTOR is not None}"

    faces = [item for item in raw if str(item.get("class") or "").startswith("FACE_")]
    intimate = [
        item
        for item in raw
        if str(item.get("class") or "") in INTIMATE_CLASSES and not is_never_mask(str(item.get("class") or ""))
    ]
    bits: list[str] = []
    if faces:
        top_face = max(faces, key=lambda item: float(item.get("score") or 0.0))
        bits.append(f"{top_face.get('class') or 'FACE'} ignored")
    if intimate:
        top_i = max(intimate, key=lambda item: float(item.get("score") or 0.0))
        lab = str(top_i.get("class") or "-")
        score = float(top_i.get("score") or 0.0)
        bits.append(
            f"no intimate/explicit above floor ({lab}:{score:.2f} < {policy} {floor:.2f})"
        )
    else:
        bits.append(f"no intimate/explicit above floor ({policy} {floor:.2f})")
    return "; ".join(bits)


def detect_nsfw(
    image: Frame | np.ndarray,
    gt_boxes: Iterable[dict] | None = None,
    frame_id: int = 0,
    *,
    stage: str = "detect",
    origin: tuple[int, int] = (0, 0),
    min_score: float | None = None,
    policy_name: str | None = None,
    box_fn: BoxFn | None = None,
    score_fn: ScoreFn | None = None,
    tile: bool | None = None,
    hold: ClusterHold | None = None,
    record_stats: bool = True,
) -> list[Detection]:
    """Return ``flagged`` boxes for policy-allowed NudeNet classes.

    Faces / feet / armpits / covered belly never emit. Balanced and strict
    also emit suggestive intimate classes (covered genitalia / breast,
    belly exposed, …). Relaxed stays explicit-only.

    ``gt_boxes`` is ignored (eval GT is synthetic-only). ``box_fn`` / ``score_fn``
    are injectable so tests never download media or touch a display.
    ``record_stats=False`` leaves ``LAST_STATS`` alone (second looks on crops).
    """
    del gt_boxes  # live path does not overlay fixture GT
    policy = _normalize_policy(policy_name) if policy_name else "balanced"
    floor = (
        float(min_score)
        if min_score is not None
        else float(POLICY_MIN_SCORE.get(policy, _DEFAULT_MIN_SCORE))
    )
    bgr = _bgr(image)
    if bgr is None or bgr.size == 0:
        if not record_stats:
            return []
        _set_stats(
            DetectStats(
                model_loaded=_DETECTOR is not None,
                threshold=floor,
                policy_name=policy,
                reason="empty frame",
            )
        )
        return []
    height, width = bgr.shape[:2]
    use_tiles = (
        tile
        if tile is not None
        else (box_fn is None and score_fn is None and _should_tile(height, width))
    )

    used, scale = _downscale(bgr)
    inv = 1.0 / scale if scale else 1.0
    n_tiles = 0
    extra: list[dict] = []
    if box_fn is not None:
        raw = list(box_fn(used))
        if use_tiles:
            extra, n_tiles = _content_tile_boxes(bgr, box_fn=box_fn)
    elif score_fn is not None:
        raw = _tile_proposals(used, score_fn, min_score=floor)
    else:
        raw = _nudenet_boxes(used)
        if use_tiles:
            extra, n_tiles = _content_tile_boxes(bgr)

    detections: list[Detection] = []
    emitted_hits: list[tuple[str, float]] = []
    ox, oy = origin
    emit_policy = policy or "balanced"
    hold_floor = min(floor, LIVE_HOLD_SCORE)
    n_after_on = 0
    for item in list(raw) + extra:
        label = str(item.get("class") or item.get("label") or "")
        if not label_should_emit(label, emit_policy):
            continue
        score = float(item.get("score") or 0.0)
        if score < hold_floor:
            continue
        space = str(item.get("space") or "work")
        xyxy = _xywh_to_xyxy(
            item.get("box") or item.get("xyxy") or (),
            width,
            height,
            inv_scale=1.0 if space == "orig" else inv,
            offset=(ox, oy),
        )
        if xyxy is None:
            continue
        if box_area(xyxy) < _MIN_AREA:
            continue
        detections.append(
            Detection(
                label="flagged",
                score=max(0.0, min(1.0, score)),
                xyxy=xyxy,
                detector_id=DETECTOR_ID,
                frame_id=frame_id,
            )
        )
        if score >= floor:
            n_after_on += 1
            emitted_hits.append((label, score))
        elif hold is not None:
            emitted_hits.append((label, score))
    detections.sort(key=lambda d: (d.score, box_area(d.xyxy)), reverse=True)
    emitted_hits.sort(key=lambda item: -item[1])
    from vision_core.cluster_nsfw import cluster_detections

    if score_fn is None:
        detections = cluster_detections(detections, max_clusters=None)
    if hold is not None:
        hold.on_score = floor
        hold.hold_score = LIVE_HOLD_SCORE
        detections = hold.apply(detections)
    else:
        detections = [d for d in detections if d.score >= floor]
    detections, clusters_capped = apply_cluster_cap(detections)
    if hold is not None:
        hold.replace(detections)
    if not record_stats:
        return detections
    combined = list(raw) + extra
    top = _top_raw(combined)
    explicit = [item for item in combined if str(item.get("class") or "") in EXPLICIT_CLASSES]
    intimate = [item for item in combined if str(item.get("class") or "") in INTIMATE_CLASSES]
    if n_after_on == 0 and detections:
        reason = f"cluster_hold={len(detections)} (after=0; keep frost)"
    else:
        reason = _stats_reason(
            policy_name=policy,
            floor=floor,
            raw=combined,
            n_after=n_after_on,
            emitted=emitted_hits,
        )
    _set_stats(
        DetectStats(
            model_loaded=_DETECTOR is not None or box_fn is not None or score_fn is not None,
            n_raw=len(combined),
            top=top,
            max_score=top[0][1] if top else 0.0,
            max_explicit=max((float(item.get("score") or 0.0) for item in explicit), default=0.0),
            max_intimate=max((float(item.get("score") or 0.0) for item in intimate), default=0.0),
            emitted=tuple(emitted_hits),
            threshold=floor,
            policy_name=policy,
            n_after_policy=n_after_on,
            n_clusters=len(detections),
            cluster_cap=MAX_CLUSTERS,
            clusters_capped=clusters_capped,
            n_tiles=n_tiles,
            tiled=bool(use_tiles and n_tiles),
            reason=reason,
        )
    )
    return detections


def make_nsfw_detect(
    policy_name: str = "balanced",
    *,
    box_fn: BoxFn | None = None,
    score_fn: ScoreFn | None = None,
    min_score: float | None = None,
    tile: bool | None = None,
):
    """Orchestrator-compatible ``detect_fn`` closed over the live policy."""

    hold = ClusterHold(
        on_score=float(min_score) if min_score is not None else float(POLICY_MIN_SCORE.get(_normalize_policy(policy_name), _DEFAULT_MIN_SCORE)),
        hold_score=LIVE_HOLD_SCORE,
    )

    def _detect(
        image: Frame | np.ndarray,
        gt_boxes: Iterable[dict] | None = None,
        frame_id: int = 0,
        *,
        stage: str = "detect",
        origin: tuple[int, int] = (0, 0),
    ) -> list[Detection]:
        return detect_nsfw(
            image,
            gt_boxes,
            frame_id,
            stage=stage,
            origin=origin,
            min_score=min_score,
            policy_name=policy_name,
            box_fn=box_fn,
            score_fn=score_fn,
            tile=tile,
            hold=hold,
        )

    def _detect_once(
        image: Frame | np.ndarray,
        gt_boxes: Iterable[dict] | None = None,
        frame_id: int = 0,
        *,
        stage: str = "reinspect",
        origin: tuple[int, int] = (0, 0),
    ) -> list[Detection]:
        """Same detector, no ``ClusterHold`` and no ``LAST_STATS`` write.

        Reinspect runs on a crop: going through ``_detect`` would replace the
        held full-frame clusters with crop-coordinate boxes (v1 bug #12).
        """
        return detect_nsfw(
            image,
            gt_boxes,
            frame_id,
            stage=stage,
            origin=origin,
            min_score=min_score,
            policy_name=policy_name,
            box_fn=box_fn,
            score_fn=score_fn,
            tile=tile,
            hold=None,
            record_stats=False,
        )

    _detect.reset_hold = hold.reset  # type: ignore[attr-defined]
    _detect.hold = hold  # type: ignore[attr-defined]
    _detect.hold_free = _detect_once  # type: ignore[attr-defined]
    return _detect
