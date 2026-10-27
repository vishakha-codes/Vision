"""track_and_predict — IoU assign, EMA / Kalman smooth, optional DIS flow."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from vision_core.contracts import Detection, Frame, TrackedRegion, TrackMode
from vision_core.geometry import as_int_box, blend_boxes, box_center, box_iou, clip_box, shift_box

_DIS = None
_EMA = 0.65
_IOU_MIN = 0.25
_MAX_MISSES = 8
_SCORE_EMA = 0.55


def _bgr(image: Frame | np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if isinstance(image, Frame):
        return image.image_bgr
    return image


def _dis() -> cv2.DISOpticalFlow:
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    return _DIS


def _mean_flow(
    flow: np.ndarray | None,
    xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float]:
    if flow is None:
        return 0.0, 0.0
    x1, y1, x2, y2 = as_int_box(xyxy, width, height)
    patch = flow[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0, 0.0
    return float(np.mean(patch[..., 0])), float(np.mean(patch[..., 1]))


def _next_track_id(tracks: list[TrackedRegion]) -> int:
    if not tracks:
        return 1
    return max(t.track_id for t in tracks) + 1


def _predict_box(
    track: TrackedRegion,
    flow: np.ndarray | None,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    # Flow is the motion measurement. Velocity is the fallback when there is
    # no previous frame. Adding both double-counts and overshoots on hold.
    if flow is not None:
        fx, fy = _mean_flow(flow, track.xyxy, width, height)
        return shift_box(track.xyxy, fx, fy, width, height)
    dx, dy = track.velocity
    return shift_box(track.xyxy, dx, dy, width, height)


def expire_tracks(tracks: list[TrackedRegion], last_seen_ms: float | None = None) -> list[TrackedRegion]:
    """Mark every track EXPIRED (scene-cut / hard reset)."""
    out: list[TrackedRegion] = []
    for track in tracks:
        ts = track.last_seen_ms if last_seen_ms is None else last_seen_ms
        out.append(replace(track, mode=TrackMode.EXPIRED, last_seen_ms=ts, misses=track.misses + 1))
    return out


def _mode_for(score: float, hits: int, uncertain_hi: float = 0.75, uncertain_lo: float = 0.40) -> TrackMode:
    if hits <= 1:
        return TrackMode.NEW
    if uncertain_lo <= score < uncertain_hi:
        return TrackMode.REINSPECT
    return TrackMode.TRACKED


def _velocity_from_shift(
    track: TrackedRegion,
    box: tuple[float, float, float, float],
) -> tuple[float, float]:
    ocx, ocy = box_center(track.xyxy)
    ncx, ncy = box_center(box)
    return (
        0.5 * track.velocity[0] + 0.5 * (ncx - ocx),
        0.5 * track.velocity[1] + 0.5 * (ncy - ocy),
    )


def track_and_predict(
    tracks: list[TrackedRegion],
    detections: list[Detection],
    frame: Frame | np.ndarray,
    prev_frame: Frame | np.ndarray | None = None,
    *,
    t_ms: float | None = None,
    iou_min: float = _IOU_MIN,
    hold: bool = False,
) -> list[TrackedRegion]:
    """Associate detections to tracks; predict with flow + EMA when unmatched.

    ``hold=True`` is an intentional skip of detection (edge change-gate).
    Warp with DIS flow / velocity, refresh the Kalman-EMA velocity, and do
    **not** count the frame as a miss. Change-gate skips detect, not track.
    """
    image = _bgr(frame)
    if image is None:
        return list(tracks)
    height, width = image.shape[:2]
    if t_ms is None:
        t_ms = float(frame.t_ms) if isinstance(frame, Frame) else 0.0

    flow = None
    prev = _bgr(prev_frame)
    if prev is not None and prev.shape[:2] == image.shape[:2]:
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flow = _dis().calc(prev_gray, cur_gray, None)

    predicted = [_predict_box(track, flow, width, height) for track in tracks]
    assigned_tracks: set[int] = set()
    assigned_dets: set[int] = set()
    pairs: list[tuple[float, int, int]] = []
    for ti, pred_box in enumerate(predicted):
        if tracks[ti].mode == TrackMode.EXPIRED:
            continue
        for di, det in enumerate(detections):
            iou = box_iou(pred_box, det.xyxy)
            if iou >= iou_min:
                pairs.append((iou, ti, di))
    pairs.sort(reverse=True)

    updated: list[TrackedRegion] = []
    for _iou, ti, di in pairs:
        if ti in assigned_tracks or di in assigned_dets:
            continue
        assigned_tracks.add(ti)
        assigned_dets.add(di)
        track = tracks[ti]
        det = detections[di]
        pred_box = predicted[ti]
        meas = clip_box(det.xyxy, width, height)
        box = blend_boxes(pred_box, meas, _EMA)
        vx, vy = _velocity_from_shift(track, box)
        stable = _SCORE_EMA * det.score + (1.0 - _SCORE_EMA) * track.stable_score
        hits = track.hits + 1
        updated.append(
            TrackedRegion(
                track_id=track.track_id,
                xyxy=box,
                label=det.label,
                stable_score=stable,
                uncertainty=max(0.0, min(1.0, 1.0 - stable)),
                last_seen_ms=t_ms,
                mode=_mode_for(stable, hits),
                velocity=(vx, vy),
                hits=hits,
                misses=0,
                mask_on=track.mask_on,
            )
        )

    next_id = _next_track_id(tracks + updated)
    for di, det in enumerate(detections):
        if di in assigned_dets:
            continue
        box = clip_box(det.xyxy, width, height)
        updated.append(
            TrackedRegion(
                track_id=next_id,
                xyxy=box,
                label=det.label,
                stable_score=det.score,
                uncertainty=max(0.0, min(1.0, 1.0 - det.score)),
                last_seen_ms=t_ms,
                mode=TrackMode.NEW,
                velocity=(0.0, 0.0),
                hits=1,
                misses=0,
                mask_on=False,
            )
        )
        next_id += 1

    for ti, track in enumerate(tracks):
        if ti in assigned_tracks or track.mode == TrackMode.EXPIRED:
            continue
        box = clip_box(predicted[ti], width, height)
        vx, vy = _velocity_from_shift(track, box)
        if hold:
            updated.append(
                replace(
                    track,
                    xyxy=box,
                    mode=TrackMode.HELD,
                    misses=0,
                    velocity=(vx, vy),
                    last_seen_ms=t_ms,
                )
            )
            continue
        misses = track.misses + 1
        if misses > _MAX_MISSES:
            updated.append(replace(track, mode=TrackMode.EXPIRED, misses=misses))
            continue
        updated.append(
            replace(
                track,
                xyxy=box,
                mode=TrackMode.HELD,
                misses=misses,
                velocity=(vx, vy),
                uncertainty=min(1.0, track.uncertainty + 0.08),
            )
        )

    return [t for t in updated if t.mode != TrackMode.EXPIRED]
