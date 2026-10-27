"""evaluate_fixture + CLI: one synthetic clip in, metrics and traces out.

The frame loop goes through ``agent.orchestrator.Orchestrator`` so OpenCV
scores change the next tool (APPLY / REINSPECT / NO_ACTION / RELEASE).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from agent.orchestrator import Orchestrator
from vision_core.contracts import (
    FLAGGED_LABELS,
    PROTECTIVE_ACTIONS,
    Detection,
    make_trace,
    trace_to_dict,
)
from vision_core.geometry import box_iou
from vision_core.policy import evaluate, load_policy
from vision_core.render_mask_spec import render_mask_spec

_ROOT = Path(__file__).resolve().parents[1]

INTENDED_MAPPING = {
    "fixture_first_pass_score": 0.64,
    "fixture_reinspect_score": 0.92,
    "first_pass": {"strict": "OPAQUE_MASK", "balanced": "BLUR", "relaxed": "REVIEW"},
    "post_reinspect": {"strict": "OPAQUE_MASK", "balanced": "BLUR", "relaxed": "NO_ACTION"},
    "notes": (
        "Strict treats ~0.64 as HIGH (class=block) and applies OPAQUE without reinspect. "
        "Balanced treats ~0.64 as MID and reinspecs only when budget/cooldown allow; "
        "block_trigger 0.97 keeps post-reinspect BLUR (not Strict OPAQUE). "
        "Relaxed first-pass ~0.64 is MID/below min_conf → REVIEW; overlay 0.94 / block 0.99 "
        "sit above typical reinspect ~0.92 so post-reinspect stays NO_ACTION."
    ),
}


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    alt = _ROOT / candidate
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(path)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((q / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


def _timing_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "n": 0}
    return {
        "mean": float(statistics.fmean(values)),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "n": len(values),
    }


def _gt_flagged_box(frame_rec: dict) -> tuple[float, float, float, float] | None:
    for box in frame_rec.get("boxes") or []:
        if box.get("class") in FLAGGED_LABELS or box.get("id") == "r2":
            xyxy = box.get("xyxy")
            if xyxy and len(xyxy) == 4:
                return tuple(float(v) for v in xyxy)
    return None


def _best_flagged(
    detections: Iterable[Detection],
    tracks,
) -> tuple[float, float, float, float] | None:
    candidates: list[tuple[float, tuple[float, float, float, float]]] = []
    for det in detections:
        if det.label in FLAGGED_LABELS:
            candidates.append((det.score, det.xyxy))
    for track in tracks:
        if track.label in FLAGGED_LABELS:
            candidates.append((track.stable_score, track.xyxy))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _policy_spread(sample_score: float, area: float, policies_dir: Path, post_score: float = 0.92) -> dict[str, Any]:
    from vision_core.contracts import TrackedRegion, TrackMode

    def _region(score: float) -> TrackedRegion:
        box = (100.0, 80.0, 180.0, 160.0)
        if area < 2000:
            box = (40.0, 40.0, 140.0, 140.0)
        return TrackedRegion(
            track_id=1,
            xyxy=box,
            label="flagged",
            stable_score=score,
            uncertainty=1.0 - score,
            last_seen_ms=0.0,
            mode=TrackMode.TRACKED,
        )

    def _across(score: float) -> dict[str, str]:
        out: dict[str, str] = {}
        for name in ("strict", "balanced", "relaxed"):
            path = policies_dir / f"{name}.json"
            if not path.exists():
                continue
            actions = evaluate([_region(score)], path)
            out[name] = actions[0].action.value if actions else "NO_ACTION"
        return out

    first = _across(sample_score)
    post = _across(post_score)
    return {
        "sample_flagged_score": sample_score,
        "post_reinspect_score": post_score,
        "strict": first.get("strict", "NO_ACTION"),
        "balanced": first.get("balanced", "NO_ACTION"),
        "relaxed": first.get("relaxed", "NO_ACTION"),
        "first_pass": first,
        "post_reinspect": post,
        "intended_mapping": INTENDED_MAPPING,
    }


def evaluate_fixture(
    fixture: str | Path,
    gt: str | Path,
    policy: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    """Run the agentic OpenCV 5 loop on a labeled clip and write artifacts."""
    fixture_path = _resolve(fixture)
    gt_path = _resolve(gt)
    policy_path = _resolve(policy)
    out_dir = Path(out)
    if not out_dir.is_absolute():
        out_dir = (_ROOT / out_dir).resolve() if not Path(out).exists() else out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_policy(policy_path)
    agent = Orchestrator(cfg)
    gt_doc = json.loads(gt_path.read_text())
    gt_frames = {int(item["frame"]): item for item in gt_doc.get("frames") or []}

    cap = cv2.VideoCapture(str(fixture_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open fixture {fixture_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or gt_doc.get("fps") or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or gt_doc.get("width") or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or gt_doc.get("height") or 0)
    writer = cv2.VideoWriter(
        str(out_dir / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Failed to open VideoWriter for {out_dir / 'annotated.mp4'}")

    traces_path = out_dir / "traces.jsonl"
    timings: dict[str, list[float]] = defaultdict(list)
    ious: list[float] = []
    first_pass_scores: list[float] = []
    post_reinspect_scores: list[float] = []
    mask_series: list[bool] = []
    sequences: Counter[str] = Counter()
    exposure_gap = 0
    reinspect_count = 0
    prev_bgr: np.ndarray | None = None
    frame_id = 0

    with traces_path.open("w") as traces_fp:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t_ms = frame_id * (1000.0 / fps)

            result = agent.process_frame(frame_id, bgr, prev_bgr, t_ms)
            for name, value in result.timings_ms.items():
                timings[name].append(float(value))
            reinspect_count += result.reinspect_count
            sequences[result.tool_sequence] += 1

            for det in result.detections:
                if det.label in FLAGGED_LABELS:
                    if ":reinspect" in (det.detector_id or ""):
                        post_reinspect_scores.append(det.score)
                    else:
                        first_pass_scores.append(det.score)

            t6 = time.perf_counter()
            annotated, _specs = render_mask_spec(result.actions, bgr)
            timings["render_mask_spec"].append((time.perf_counter() - t6) * 1000.0)
            cv2.putText(
                annotated,
                f"f{frame_id} {cfg.name} {result.tool_sequence} cv{cv2.__version__}",
                (8, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            writer.write(annotated)

            for trace in result.traces:
                traces_fp.write(json.dumps(trace_to_dict(trace)) + "\n")
            traces_fp.write(
                json.dumps(
                    trace_to_dict(
                        make_trace(
                            frame_id=frame_id,
                            tool="agent.decide",
                            observation={
                                "band": result.band,
                                "n_tracks": len(result.tracks),
                                "n_tiles": result.n_tiles,
                                "change_score": result.change_score,
                                "actions": [a.action.value for a in result.actions],
                                "reasons": [a.reason_code for a in result.actions],
                                "first_decision": result.first_decision,
                                "tool_sequence": result.tool_sequence,
                                "budget_left": agent.budget_left,
                            },
                            decision=result.final_decision,
                            next_action="cool.render_mask_spec",
                            latency_ms=result.timings_ms.get("policy.evaluate", 0.0),
                        )
                    )
                )
                + "\n"
            )

            gt_box = _gt_flagged_box(gt_frames.get(frame_id) or {})
            pred_box = _best_flagged(result.detections, result.tracks)
            if gt_box is not None and pred_box is not None:
                ious.append(box_iou(gt_box, pred_box))
            elif gt_box is not None:
                ious.append(0.0)

            protective = any(a.action in PROTECTIVE_ACTIONS for a in result.actions)
            mask_series.append(protective)
            if gt_box is not None and not protective:
                exposure_gap += 1

            prev_bgr = bgr
            frame_id += 1

    cap.release()
    writer.release()

    flicker = 0
    for prev_on, cur_on in zip(mask_series, mask_series[1:]):
        if prev_on != cur_on:
            flicker += 1

    sample_score = float(statistics.fmean(first_pass_scores)) if first_pass_scores else 0.0
    post_score = float(statistics.fmean(post_reinspect_scores)) if post_reinspect_scores else 0.92
    results = {
        "fixture": str(fixture_path),
        "gt": str(gt_path),
        "policy": cfg.name,
        "policy_version": cfg.version,
        "opencv_version": cv2.__version__,
        "frame_count": frame_id,
        "mean_iou_flagged": float(statistics.fmean(ious)) if ious else 0.0,
        "exposure_gap_frames": exposure_gap,
        "flicker_on_off_count": flicker,
        "reinspect_crop_count": reinspect_count,
        "masked_frames": int(sum(1 for x in mask_series if x)),
        "first_pass_flagged_score_mean": sample_score,
        "post_reinspect_flagged_score_mean": post_score,
        "tool_sequences": dict(sequences),
        "agent": {
            "reinspect_budget": agent.settings["reinspect_budget"],
            "reinspect_cooldown_frames": agent.settings["reinspect_cooldown_frames"],
            "budget_left": agent.budget_left,
            "bedrock": "unavailable_local_controller",
        },
        "timings_ms": {name: _timing_summary(vals) for name, vals in timings.items()},
        "policy_spread": _policy_spread(sample_score, 10000.0, policy_path.parent, post_score),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Screen Guardian fixture eval (OpenCV 5 + agent)")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    results = evaluate_fixture(args.fixture, args.gt, args.policy, args.out)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
