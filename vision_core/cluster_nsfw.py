"""Merge overlapping / nearby NSFW part-boxes into figure clusters.

NudeNet emits breast + belly + genitalia as separate labels. Those must
become **one** frost window per figure. Distant Google-grid thumbs stay
separate — we only union boxes that overlap or sit a few pixels apart
(same body), never the whole desktop.

Live hold: once a cluster is masked, keep it while its score stays at or
above ``HOLD_SCORE`` until ``reset`` (scene change). Synthetic HSV / eval
does not import this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from vision_core.contracts import Detection
from vision_core.geometry import box_area, box_iou

Box = tuple[float, float, float, float]

# Safety ceiling so a SERP grid is not clipped to a toy count, and so we
# never spawn hundreds of frost windows. Typical frames are 0–N unique
# figures after NMS — this is not a target or a default.
MAX_CLUSTERS = 24
# Tiny pad so breast/belly that miss by a couple of pixels still union.
# A typical SERP gutter is wider than this, so adjacent thumbs stay split.
CLUSTER_PAD_PX = 8.0
CLUSTER_IOU_MIN = 0.02
HOLD_SCORE = 0.20
HOLD_MATCH_IOU = 0.15
LIVE_MEAN_JUMP = 14.0
LIVE_FP_L1 = 16.0
LIVE_HIST_L1 = 0.42


def apply_cluster_cap(
    items: list,
    *,
    max_clusters: int = MAX_CLUSTERS,
) -> tuple[list, bool]:
    """Keep ``items`` as-is unless it exceeds the safety ceiling.

    Returns ``(kept, capped)``. Empty stays empty — never invents boxes.
    """
    cap = max(0, int(max_clusters))
    if len(items) <= cap:
        return list(items), False
    return list(items[:cap]), True


def union_xyxy(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def boxes_related(
    a: Box,
    b: Box,
    *,
    iou_min: float = CLUSTER_IOU_MIN,
    pad: float = CLUSTER_PAD_PX,
) -> bool:
    """True when two boxes are the same figure (overlap or almost-touch)."""
    if box_iou(a, b) >= iou_min:
        return True
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = min(ax2 + pad, bx2 + pad) - max(ax1 - pad, bx1 - pad)
    iy = min(ay2 + pad, by2 + pad) - max(ay1 - pad, by1 - pad)
    return ix > 0.0 and iy > 0.0


def cluster_detections(
    detections: list[Detection],
    *,
    max_clusters: int | None = MAX_CLUSTERS,
    pad: float = CLUSTER_PAD_PX,
) -> list[Detection]:
    """Union overlapping / nearby boxes. Count is 0–N unique figures.

    ``max_clusters`` is a safety ceiling (SERP grid), not a default. Only
    the highest-score extras are dropped, and only when over the cap.
    Pass ``None`` to return every unique figure (caller applies the cap).
    """
    n = len(detections)
    if n <= 1:
        return list(detections)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if boxes_related(detections[i].xyxy, detections[j].xyxy, pad=pad):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out: list[Detection] = []
    for idxs in groups.values():
        xyxy = detections[idxs[0]].xyxy
        best = idxs[0]
        best_score = detections[best].score
        for i in idxs[1:]:
            xyxy = union_xyxy(xyxy, detections[i].xyxy)
            if detections[i].score > best_score:
                best = i
                best_score = detections[i].score
        out.append(replace(detections[best], xyxy=xyxy, score=best_score))
    out.sort(key=lambda d: (d.score, box_area(d.xyxy)), reverse=True)
    if max_clusters is None:
        return out
    kept, _capped = apply_cluster_cap(out, max_clusters=max_clusters)
    return kept


def cluster_xyxys(
    boxes: list[tuple[float, float, float, float]],
    *,
    max_clusters: int = MAX_CLUSTERS,
) -> list[tuple[float, float, float, float]]:
    """Union nearby overlay rects (already in capture xyxy)."""
    if len(boxes) <= 1:
        return list(boxes)
    dets = [
        Detection(label="flagged", score=1.0, xyxy=tuple(box), detector_id="overlay", frame_id=0)
        for box in boxes
    ]
    return [d.xyxy for d in cluster_detections(dets, max_clusters=max_clusters)]


@dataclass
class HeldCluster:
    xyxy: Box
    score: float


class ClusterHold:
    """Keep a masked cluster while score >= hold floor, until ``reset``."""

    def __init__(self, *, on_score: float = 0.30, hold_score: float = HOLD_SCORE) -> None:
        self.on_score = float(on_score)
        self.hold_score = float(hold_score)
        self._held: list[HeldCluster] = []

    def reset(self) -> None:
        self._held = []

    def replace(self, clusters: list[Detection]) -> None:
        self._held = [HeldCluster(d.xyxy, d.score) for d in clusters]

    def apply(self, clusters: list[Detection]) -> list[Detection]:
        used: set[int] = set()
        out: list[Detection] = []
        for det in clusters:
            matched: int | None = None
            best_iou = 0.0
            for i, old in enumerate(self._held):
                if i in used:
                    continue
                iou = box_iou(det.xyxy, old.xyxy)
                related = iou >= HOLD_MATCH_IOU or boxes_related(det.xyxy, old.xyxy)
                if related and iou >= best_iou:
                    best_iou = iou
                    matched = i
            keep = det.score >= self.on_score
            if not keep and matched is not None and det.score >= self.hold_score:
                keep = True
            if not keep:
                continue
            if matched is not None:
                used.add(matched)
            out.append(det)
        # Frost cover / after=0 must not drop a masked figure. Keep unmatched
        # holds only when this frame produced no clusters — not on top of a
        # live set (that invents stale SERP thumbs). Reset() still clears on
        # a real scene change.
        if not out:
            for i, old in enumerate(self._held):
                if i in used:
                    continue
                out.append(
                    Detection(
                        label="flagged",
                        score=old.score,
                        xyxy=old.xyxy,
                        detector_id="cluster_hold",
                        frame_id=0,
                    )
                )
        self.replace(out)
        return out


def luma_fingerprint(bgr, grid: int = 6):
    """Cheap 6×6 luminance grid. No pixels leave the process."""
    import cv2
    import numpy as np

    if bgr is None or getattr(bgr, "size", 0) == 0:
        return np.zeros((grid, grid), dtype=np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return cv2.resize(gray, (grid, grid), interpolation=cv2.INTER_AREA).astype(np.float32)


def _hist_l1(prev, cur) -> float:
    import cv2
    import numpy as np

    pg = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) if prev.ndim == 3 else prev
    cg = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY) if cur.ndim == 3 else cur
    hp = cv2.calcHist([pg], [0], None, [16], [0, 256]).flatten()
    hc = cv2.calcHist([cg], [0], None, [16], [0, 256]).flatten()
    hp = hp / (float(hp.sum()) + 1e-6)
    hc = hc / (float(hc.sum()) + 1e-6)
    return float(np.abs(hp - hc).sum()) * 0.5


def _paint_ignore(bgr, boxes, value=0):
    """Zero frost rects so overlay pixels cannot look like a page change."""
    import numpy as np

    if bgr is None or getattr(bgr, "size", 0) == 0:
        return bgr
    out = np.array(bgr, copy=True)
    height, width = out.shape[:2]
    for box in boxes or []:
        if box is None or len(box) < 4:
            continue
        x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
        x1, y1 = max(0, min(width, x1)), max(0, min(height, y1))
        x2, y2 = max(0, min(width, x2)), max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = value
    return out


def live_scene_changed(
    prev,
    cur,
    *,
    mean_jump: float = LIVE_MEAN_JUMP,
    fp_l1: float = LIVE_FP_L1,
    hist_l1: float = LIVE_HIST_L1,
    ignore_boxes=None,
) -> bool:
    """True when the live desktop is no longer the same shot.

    Tuned so a mean 72→92 (or a 6×6 fingerprint jump) drops stale frost.
    Overlay / frost rects in ``ignore_boxes`` are painted out so a mask
    appearing or disappearing is not treated as navigation.
    Does **not** change ``scene_cut`` defaults used by synthetic eval.
    """
    import cv2
    import numpy as np

    if prev is None or cur is None:
        return False
    if getattr(prev, "size", 0) == 0 or getattr(cur, "size", 0) == 0:
        return False
    if ignore_boxes:
        prev = _paint_ignore(prev, ignore_boxes)
        cur = _paint_ignore(cur, ignore_boxes)
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) if prev.ndim == 3 else prev
    cur_g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY) if cur.ndim == 3 else cur
    if abs(float(prev_g.mean()) - float(cur_g.mean())) >= mean_jump:
        return True
    delta = float(np.mean(np.abs(luma_fingerprint(prev) - luma_fingerprint(cur))))
    if delta >= fp_l1:
        return True
    return _hist_l1(prev, cur) >= hist_l1
