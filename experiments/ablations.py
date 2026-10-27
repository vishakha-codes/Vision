"""B0–B5 ablations on the safe synthetic fixture.

B0–B2 are thinner local loops (honestly approximate vs the full agent).
B3 reuses ``vision_core.eval.evaluate_fixture`` (current Balanced).
B4 is the shipping ``edge.reflex`` loop (change-gate skips detect; flow-hold
still tracks; periodic keyframes).
B5 is a placeholder: COOL vs stock OpenCV 5 on Graviton is not run.

Same hardware label for every local row: this Mac, arm64, OpenCV 5.0.0.
Do not invent Graviton / COOL timings.
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from edge.reflex import Reflex
from vision_core.contracts import (
    FLAGGED_LABELS,
    PROTECTIVE_ACTIONS,
    Frame,
    ProtectionAction,
    TrackedRegion,
)
from vision_core.detect_regions import detect_regions
from vision_core.eval import _best_flagged, _gt_flagged_box, evaluate_fixture
from vision_core.geometry import box_iou
from vision_core.policy import evaluate, load_policy
from vision_core.track_and_predict import track_and_predict

_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = _ROOT / "fixtures" / "safe" / "synthetic_moving_regions.mp4"
GT_PATH = _ROOT / "fixtures" / "safe" / "synthetic_moving_regions.gt.json"
POLICY_PATH = _ROOT / "policies" / "balanced.json"
OUT_DIR = _ROOT / "experiments" / "ablations"
CHANGE_GATE = 0.08  # edge.reflex.Reflex default
B4_BRAIN_EVERY = 3  # edge.reflex.Reflex default keyframe
B4_BRAIN_EVERY_MS = 250.0  # edge.reflex.Reflex default time keyframe

B5_REASON = (
    "NOT RUN. Cloud Optimized OpenCV (COOL) vs stock OpenCV 5 on Graviton "
    "requires a Marketplace subscribe + wizard launch of "
    "https://aws.amazon.com/marketplace/pp/prodview-fdvbfiewzuehs. "
    "cli-user cannot resolve the AMI: org SCP <scp-id> (management account "
    "<account-id>) explicitly denies ec2:DescribeImages. This repo does not "
    "launch EC2. No stock-vs-COOL P50/P95, fps, CPU, or $/1k-frames numbers "
    "exist. Mark B5 pending/user."
)


def hardware_label() -> dict[str, Any]:
    return {
        "label": f"this Mac, {platform.machine()}, OpenCV {cv2.__version__}",
        "system": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "cool_running": False,
        "graviton": False,
        "bedrock_invoke": False,
    }


def _metrics(
    ious: list[float],
    mask_series: list[bool],
    exposure_gap: int,
    reinspect_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    flicker = sum(1 for prev, cur in zip(mask_series, mask_series[1:]) if prev != cur)
    row = {
        "mean_iou_flagged": float(statistics.fmean(ious)) if ious else 0.0,
        "exposure_gap_frames": int(exposure_gap),
        "flicker_on_off_count": int(flicker),
        "reinspect_count": int(reinspect_count),
        "masked_frames": int(sum(1 for x in mask_series if x)),
        "frame_count": len(mask_series),
        "iou_n": len(ious),
    }
    if extra:
        row.update(extra)
    return row


def _iter_frames(
    fixture: Path, gt_frames: dict[int, dict]
) -> Iterable[tuple[int, np.ndarray, float, tuple[float, float, float, float] | None]]:
    cap = cv2.VideoCapture(str(fixture))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open fixture {fixture}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_id = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t_ms = frame_id * (1000.0 / fps)
            gt_box = _gt_flagged_box(gt_frames.get(frame_id) or {})
            yield frame_id, bgr, t_ms, gt_box
            frame_id += 1
    finally:
        cap.release()


def _sync_mask_flags(tracks: list[TrackedRegion], actions: list[ProtectionAction]) -> None:
    on_ids = {
        a.region.track_id
        for a in actions
        if a.region is not None and a.action in PROTECTIVE_ACTIONS
    }
    for i, track in enumerate(tracks):
        tracks[i] = replace(track, mask_on=track.track_id in on_ids)


def _score_frame(
    gt_box: tuple[float, float, float, float] | None,
    pred_box: tuple[float, float, float, float] | None,
    protective: bool,
    ious: list[float],
    mask_series: list[bool],
) -> int:
    """Return 1 when the GT flagged box is present and uncovered."""
    if gt_box is not None and pred_box is not None:
        ious.append(box_iou(gt_box, pred_box))
    elif gt_box is not None:
        ious.append(0.0)
    mask_series.append(protective)
    return 1 if (gt_box is not None and not protective) else 0


def run_b0(gt_frames: dict[int, dict]) -> dict[str, Any]:
    """Whole-frame score → whole-window cover when a flagged blob is present."""
    ious: list[float] = []
    mask_series: list[bool] = []
    exposure = 0
    t0 = time.perf_counter()
    width = height = 0
    for _fid, bgr, _t_ms, gt_box in _iter_frames(FIXTURE, gt_frames):
        height, width = bgr.shape[:2]
        dets = detect_regions(bgr, frame_id=_fid, stage="detect")
        flagged = [d for d in dets if d.label in FLAGGED_LABELS]
        present = bool(flagged)
        pred = (0.0, 0.0, float(width), float(height)) if present else None
        exposure += _score_frame(gt_box, pred, present, ious, mask_series)
    return _metrics(
        ious,
        mask_series,
        exposure,
        0,
        {
            "elapsed_s": time.perf_counter() - t0,
            "approximate": True,
            "notes": (
                "Thinner loop. Max flagged HSV score is treated as a whole-frame "
                "score; if any flagged blob is present the predicted cover is the "
                "entire 640x360 window. No track, hold, reinspect, or localization."
            ),
        },
    )


def run_b1(cfg, gt_frames: dict[int, dict]) -> dict[str, Any]:
    """Per-frame detect + policy mask. No track/hold, no reinspect."""
    ious: list[float] = []
    mask_series: list[bool] = []
    exposure = 0
    t0 = time.perf_counter()
    for fid, bgr, _t_ms, gt_box in _iter_frames(FIXTURE, gt_frames):
        dets = detect_regions(bgr, frame_id=fid, stage="detect")
        actions = evaluate(dets, cfg, frame_id=fid)
        protective = any(a.action in PROTECTIVE_ACTIONS for a in actions)
        pred = _best_flagged(dets, [])
        exposure += _score_frame(gt_box, pred, protective, ious, mask_series)
    return _metrics(
        ious,
        mask_series,
        exposure,
        0,
        {
            "elapsed_s": time.perf_counter() - t0,
            "approximate": True,
            "notes": (
                "Thinner loop. synthetic_hsv_v1 + Balanced policy.evaluate on "
                "detections only. No track_and_predict, no hysteresis carry, no "
                "reinspect_crop."
            ),
        },
    )


def run_b2(cfg, gt_frames: dict[int, dict]) -> dict[str, Any]:
    """B1 + IoU/DIS track, flow-hold, hysteresis. No reinspect."""
    ious: list[float] = []
    mask_series: list[bool] = []
    exposure = 0
    tracks: list[TrackedRegion] = []
    prev: np.ndarray | None = None
    t0 = time.perf_counter()
    for fid, bgr, t_ms, gt_box in _iter_frames(FIXTURE, gt_frames):
        dets = detect_regions(bgr, frame_id=fid, stage="detect")
        tracks = track_and_predict(tracks, dets, bgr, prev, t_ms=t_ms)
        actions = evaluate(tracks, cfg, frame_id=fid)
        _sync_mask_flags(tracks, actions)
        protective = any(a.action in PROTECTIVE_ACTIONS for a in actions)
        pred = _best_flagged(dets, tracks)
        exposure += _score_frame(gt_box, pred, protective, ious, mask_series)
        prev = bgr
    return _metrics(
        ious,
        mask_series,
        exposure,
        0,
        {
            "elapsed_s": time.perf_counter() - t0,
            "approximate": True,
            "notes": (
                "Thinner loop. B1 plus cool.track_and_predict (IoU assign + "
                "DISOpticalFlow ULTRAFAST + EMA) and policy hysteresis. "
                "Orchestrator reinspect is off."
            ),
        },
    )


def run_b3() -> dict[str, Any]:
    """Current Balanced agent: track + budgeted reinspect."""
    t0 = time.perf_counter()
    results = evaluate_fixture(FIXTURE, GT_PATH, POLICY_PATH, OUT_DIR / "b3")
    return {
        "mean_iou_flagged": float(results["mean_iou_flagged"]),
        "exposure_gap_frames": int(results["exposure_gap_frames"]),
        "flicker_on_off_count": int(results["flicker_on_off_count"]),
        "reinspect_count": int(results["reinspect_crop_count"]),
        "masked_frames": int(results["masked_frames"]),
        "frame_count": int(results["frame_count"]),
        "elapsed_s": time.perf_counter() - t0,
        "approximate": False,
        "tool_sequences": results.get("tool_sequences"),
        "source": "vision_core.eval.evaluate_fixture + agent.orchestrator (Balanced)",
        "notes": (
            "Full current pipeline. Balanced mid-band score (~0.64) calls "
            "cool.reinspect_crop when budget/cooldown allow (40 budget, 12-frame "
            "cooldown), then BLUR. Same metric definitions as B0–B2."
        ),
    }


def run_b4(_cfg, gt_frames: dict[int, dict]) -> dict[str, Any]:
    """Shipping edge reflex: change-gate skips detect; flow-hold still tracks."""
    reflex = Reflex(
        "balanced",
        brain_every=B4_BRAIN_EVERY,
        brain_every_ms=B4_BRAIN_EVERY_MS,
        change_gate=CHANGE_GATE,
    )
    ious: list[float] = []
    mask_series: list[bool] = []
    exposure = 0
    reinspect = 0
    skipped = 0
    ticked = 0
    t0 = time.perf_counter()
    for fid, bgr, t_ms, gt_box in _iter_frames(FIXTURE, gt_frames):
        height, width = bgr.shape[:2]
        state = reflex.step(
            Frame(frame_id=fid, t_ms=t_ms, image_bgr=bgr, width=width, height=height)
        )
        if state.held:
            skipped += 1
        else:
            ticked += 1
            reinspect += reflex.last_reinspect
        # Score the live (flow-warped) tracks, not a stale last-detect box.
        pred = _best_flagged([], state.tracks)
        exposure += _score_frame(gt_box, pred, state.protecting, ious, mask_series)
    return _metrics(
        ious,
        mask_series,
        exposure,
        reinspect,
        {
            "elapsed_s": time.perf_counter() - t0,
            "approximate": True,
            "change_gate": CHANGE_GATE,
            "brain_every": B4_BRAIN_EVERY,
            "brain_every_ms": B4_BRAIN_EVERY_MS,
            "brain_ticks": ticked,
            "skipped_frames": skipped,
            "notes": (
                f"Shipping edge.reflex: change-gate {CHANGE_GATE} skips detect/"
                f"reinspect; DIS flow-hold still warps the last mask; keyframe "
                f"every {B4_BRAIN_EVERY} frames or {B4_BRAIN_EVERY_MS:.0f} ms so "
                "the loop never goes blind. Approximate vs B3, which detects "
                "every frame."
            ),
        },
    )


def run_b5_placeholder() -> dict[str, Any]:
    return {
        "mean_iou_flagged": None,
        "exposure_gap_frames": None,
        "flicker_on_off_count": None,
        "reinspect_count": None,
        "masked_frames": None,
        "frame_count": None,
        "elapsed_s": None,
        "status": "pending_user",
        "approximate": False,
        "cool_running": False,
        "notes": B5_REASON,
    }


def main() -> int:
    if not FIXTURE.is_file() or not GT_PATH.is_file():
        raise SystemExit(f"missing fixture or GT under {FIXTURE.parent}")
    if str(cv2.__version__) != "5.0.0":
        raise SystemExit(f"expected cv2 5.0.0, got {cv2.__version__}")

    cfg = load_policy(POLICY_PATH)
    gt_doc = json.loads(GT_PATH.read_text())
    gt_frames = {int(item["frame"]): item for item in gt_doc.get("frames") or []}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    rows = {
        "B0": run_b0(gt_frames),
        "B1": run_b1(cfg, gt_frames),
        "B2": run_b2(cfg, gt_frames),
        "B3": run_b3(),
        "B4": run_b4(cfg, gt_frames),
        "B5": run_b5_placeholder(),
    }
    names = {
        "B0": "whole-frame score → whole-window cover",
        "B1": "per-frame detect + mask (no track, no reinspect)",
        "B2": "B1 + track / flow-hold / hysteresis (no reinspect)",
        "B3": "B2 + budgeted orchestrator reinspect (current Balanced)",
        "B4": "B3 + edge change-gate (skip detect, keep track + keyframes)",
        "B5": "COOL vs stock OpenCV 5 on Graviton (NOT RUN)",
    }
    table = []
    for key, name in names.items():
        row = dict(rows[key])
        row["id"] = key
        row["name"] = name
        table.append(row)

    summary = {
        "generated_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "fixture": str(FIXTURE),
        "gt": str(GT_PATH),
        "policy": "balanced",
        "policy_path": str(POLICY_PATH),
        "detector": "synthetic_hsv_v1 (color blobs, not NSFW)",
        "hardware": hardware_label(),
        "change_gate_b4": CHANGE_GATE,
        "brain_every_b4": B4_BRAIN_EVERY,
        "brain_every_ms_b4": B4_BRAIN_EVERY_MS,
        "honesty": {
            "live_nsfw_accuracy": False,
            "zero_exposure_claim": False,
            "cool_running": False,
            "bedrock_invoke": False,
            "b5_timings_invented": False,
        },
        "baselines": table,
    }
    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
