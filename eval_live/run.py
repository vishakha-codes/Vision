"""Live-path eval CLI over the synthetic benign screen corpus.

    python -m eval_live.run --mode aggressive --level engine --out experiments/eval_live/engine_agg/
    python -m eval_live.run --mode aggressive --level engine --sequences desktop   # window-aware desktop

Levels:
  analyzer  frame → ``LiveAnalyzer.analyze`` → masks (proposal recall/precision too)
  engine    ``edge.live_engine.run_headless`` with analyzer_latency_frames ∈ {0, 2, 5}
  legacy    v1 path: ``edge.reflex.Reflex(detect_mode="nsfw")`` + oracle ``box_fn``
  all       every level that imports

The detector is always the GT oracle (see ``eval_live.oracle``): this
measures proposals + masking + engine, **not** NSFW model accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from eval_live.metrics import SequenceEval, evaluate_sequence, layouts_of, summarize
from eval_live.oracle import ORACLE_VARIANTS, LegacyOracle, OracleTruth, make_live_analyzer, set_oracle_variant
from eval_live.sequences import DESKTOP_SEQUENCE_NAMES, SEQUENCE_NAMES, Sequence, make_sequence
from vision_core.live.contracts import ProtectionMode

LEVELS = ("analyzer", "engine", "legacy")
LEGACY_POLICY = {"aggressive": "strict", "medium": "balanced"}


def _grid_for(width: int) -> int:
    return max(1, int(math.ceil(width / 1728.0)))


def _tstats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    vals = sorted(vals)
    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 2),
        "p50": round(vals[len(vals) // 2], 2),
        "p95": round(vals[min(len(vals) - 1, int(math.ceil(0.95 * len(vals))) - 1)], 2),
        "max": round(vals[-1], 2),
    }


class TimedAnalyzer:
    """Forwards to an analyzer and records ``analyze`` wall time + its timings."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.wall_ms: list[float] = []
        self.parts: dict[str, list[float]] = defaultdict(list)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def set_mode(self, mode) -> None:
        self._inner.set_mode(mode)

    def analyze(self, frame_bgr, *, changed=None, t_ms: float = 0.0, windows=None):
        t0 = time.perf_counter()
        if windows is not None and _takes_windows(self._inner):
            res = self._inner.analyze(frame_bgr, changed=changed, t_ms=t_ms, windows=windows)
        else:
            res = self._inner.analyze(frame_bgr, changed=changed, t_ms=t_ms)
        self.wall_ms.append((time.perf_counter() - t0) * 1000.0)
        for k, v in (getattr(res, "timings_ms", None) or {}).items():
            try:
                self.parts[k].append(float(v))
            except (TypeError, ValueError):
                pass
        return res


def _takes_windows(obj, fn: str | None = "analyze") -> bool:
    import inspect

    try:
        params = inspect.signature(getattr(obj, fn) if fn else obj).parameters
    except (AttributeError, TypeError, ValueError):
        return False
    return "windows" in params


def _owners(frame_masks) -> list:
    return [getattr(m, "window_id", None) for m in frame_masks]


# --------------------------------------------------------------------------- level runners


def run_analyzer_level(seq: Sequence, mode: ProtectionMode, impl: str, grid: int) -> dict:
    truth = OracleTruth(seq.gt, seq.width, seq.height)
    analyzer, label, note = make_live_analyzer(mode, truth, impl)
    timed = TimedAnalyzer(analyzer)
    masks: list[list] = []
    owners: list[list] = []
    props: list[list] = []
    for i, frame in enumerate(truth.iter_frames(seq.frames())):
        res = timed.analyze(frame, changed=None, t_ms=seq.gt[i].t_ms, windows=seq.gt[i].windows)
        masks.append([m.xyxy for m in res.masks])
        owners.append(_owners(res.masks))
        props.append([r.xyxy for r in res.regions])
    ev = evaluate_sequence(seq.name, seq.gt, masks, seq.width, seq.height, proposals=props, grid=grid, owners=owners)
    return {
        "eval": ev,
        "masks": masks,
        "impl": label,
        "note": note,
        "timings": {"analyze_wall_ms": list(timed.wall_ms), **{f"analyzer.{k}": v for k, v in timed.parts.items()}},
        "oracle_locate": dict(truth.stats),
    }


def run_engine_level(seq: Sequence, mode: ProtectionMode, impl: str, grid: int, latency: int, analyze_every: int) -> dict:
    from edge.live_engine import run_headless

    truth = OracleTruth(seq.gt, seq.width, seq.height)
    analyzer, label, note = make_live_analyzer(mode, truth, impl)
    timed = TimedAnalyzer(analyzer)
    kw = {}
    if seq.layout == "desktop" and _takes_windows(run_headless, None):
        kw["windows"] = [g.windows for g in seq.gt]  # the window list rides along, as in the live capture
    t0 = time.perf_counter()
    out = run_headless(
        truth.iter_frames(seq.frames()),
        timed,
        t_ms=[g.t_ms for g in seq.gt],
        analyze_every=analyze_every,
        analyzer_latency_frames=latency,
        **kw,
    )
    wall = (time.perf_counter() - t0) * 1000.0
    masks = [[m.xyxy for m in frame_masks] for frame_masks in out]
    ev = evaluate_sequence(seq.name, seq.gt, masks, seq.width, seq.height, grid=grid, owners=[_owners(f) for f in out])
    return {
        "eval": ev,
        "masks": masks,
        "impl": label,
        "note": note,
        "timings": {
            "engine_frame_ms_incl_render": [wall / max(1, len(seq))],
            "analyze_wall_ms": list(timed.wall_ms),
            "analyze_calls": [float(len(timed.wall_ms))],
            **{f"analyzer.{k}": v for k, v in timed.parts.items()},
        },
        "oracle_locate": dict(truth.stats),
    }


LEGACY_BRAINS = {
    # v1 measured ~113 ms per NudeNet global+tiles tick > the 80 ms budget, so
    # production runs with ``_brain_slow`` set: ticks only every 3 frames /
    # 250 ms and the change gate is ignored. ``brain_budget_ms=0`` forces that.
    "slow_brain": {"brain_budget_ms": 0.0},
    # Zero-cost detector: ticks on every change (best case for v1).
    "instant_brain": {"brain_budget_ms": 1e9},
}


def run_legacy_level(seq: Sequence, mode: ProtectionMode, grid: int, brain: str = "slow_brain") -> dict:
    from edge.reflex import Reflex
    from vision_core.contracts import PROTECTIVE_ACTIONS, Frame
    from vision_core.detect_nsfw import make_nsfw_detect
    from vision_core.reinspect_nsfw import make_nsfw_reinspect

    truth = OracleTruth(seq.gt, seq.width, seq.height)
    oracle = LegacyOracle(truth)
    policy = LEGACY_POLICY.get(mode.value, "balanced")
    detect = make_nsfw_detect(policy, box_fn=oracle.box_fn, tile=False)
    reflex = Reflex(
        policy,
        detect_mode="nsfw",
        detect_fn=detect,
        reinspect_fn=make_nsfw_reinspect(mode="local", detect_fn=detect),
        **LEGACY_BRAINS[brain],
    )
    masks: list[list] = []
    step_ms: list[float] = []
    for i, frame in enumerate(seq.frames()):
        truth.register(frame, i)
        t0 = time.perf_counter()
        state = reflex.step(Frame(i, seq.gt[i].t_ms, frame, seq.width, seq.height))
        step_ms.append((time.perf_counter() - t0) * 1000.0)
        masks.append([
            tuple(map(float, a.region.xyxy))
            for a in state.actions
            if a.action in PROTECTIVE_ACTIONS and a.region is not None
        ])
    ev = evaluate_sequence(seq.name, seq.gt, masks, seq.width, seq.height, grid=grid)
    return {
        "eval": ev,
        "masks": masks,
        "impl": f"edge.reflex.Reflex(detect_mode=nsfw, policy={policy}, {brain}) + oracle box_fn",
        "note": "v1 baseline: part-box clusters, no scroll follow, frost dropped on live_scene_changed",
        "timings": {"reflex_step_ms": step_ms},
        "oracle_locate": dict(truth.stats),
    }


# --------------------------------------------------------------------------- video


def _annotate(frame: np.ndarray, gt, masks, width: int = 864) -> np.ndarray:
    """Masks drawn as frosted (pixelated, lightened) boxes; GT red = flagged, green = not."""
    img = frame.copy()
    for m in masks:
        x1, y1, x2, y2 = (int(round(v)) for v in m)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if x2 > x1 and y2 > y1:
            roi = img[y1:y2, x1:x2]
            small = cv2.resize(roi, (max(1, (x2 - x1) // 24), max(1, (y2 - y1) // 24)), interpolation=cv2.INTER_AREA)
            img[y1:y2, x1:x2] = (cv2.resize(small, (x2 - x1, y2 - y1)) * 0.5 + 110).astype(np.uint8)
    th = max(2, img.shape[1] // 800)
    for p in gt.pictures:
        col = (0, 0, 255) if p.flagged else (0, 190, 0)
        cv2.rectangle(img, p.xyxy[:2], (p.xyxy[2] - 1, p.xyxy[3] - 1), col, th)
    for m in masks:
        cv2.rectangle(img, (int(m[0]), int(m[1])), (int(m[2]), int(m[3])), (255, 120, 0), th)
    out = cv2.resize(img, (width, int(round(img.shape[0] * width / img.shape[1]))), interpolation=cv2.INTER_AREA)
    return out


def write_video(path: Path, seqs_masks: list[tuple[Sequence, SequenceEval, list]], semantics: str, title: str) -> None:
    writer = None
    for seq, ev, masks in seqs_masks:
        for i, frame in enumerate(seq.frames()):
            img = _annotate(frame, seq.gt[i], masks[i])
            rec = ev.records[i]
            bad = bool(rec["exposed"][semantics])
            label = f"{title} | {seq.name} f{i:03d} dy={seq.gt[i].scroll_dy}{' NAV' if seq.gt[i].after_nav else ''}"
            cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
            cv2.putText(img, label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            if bad:
                cv2.putText(img, "EXPOSED", (img.shape[1] - 90, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            if writer is None:
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (img.shape[1], img.shape[0]))
            writer.write(img)
    if writer is not None:
        writer.release()


# --------------------------------------------------------------------------- driver


def _variants(levels: list[str], analyzers: list[str], latencies: list[int]) -> list[dict]:
    out = []
    for level in levels:
        if level == "analyzer":
            out += [{"level": "analyzer", "analyzer": a, "variant": f"analyzer[{a}]"} for a in analyzers]
        elif level == "engine":
            out += [
                {"level": "engine", "analyzer": a, "latency": lat, "variant": f"engine[{a}]/latency={lat}"}
                for a in analyzers
                for lat in latencies
            ]
        elif level == "legacy":
            out += [{"level": "legacy", "brain": b, "variant": f"legacy[reflex/{b}]"} for b in LEGACY_BRAINS]
    return out


def _check_levels(levels: list[str]) -> tuple[list[str], list[dict]]:
    ok, skipped = [], []
    for level in levels:
        try:
            if level == "engine":
                from edge.live_engine import run_headless  # noqa: F401
            elif level == "legacy":
                from edge.reflex import Reflex  # noqa: F401
                from vision_core.detect_nsfw import make_nsfw_detect  # noqa: F401
            ok.append(level)
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            print(f"[eval_live] SKIP level={level}: module not importable yet ({reason})", flush=True)
            skipped.append({"level": level, "reason": reason})
    return ok, skipped


def _merge_timings(dst: dict[str, list[float]], src: dict[str, list[float]]) -> None:
    for k, v in src.items():
        dst[k].extend(float(x) for x in v)


def evaluate(
    *,
    mode: str,
    levels: list[str],
    out_dir: Path,
    res_list: list[str],
    sequences: list[str],
    frames: int,
    seed: int = 0,
    analyzers: list[str] | None = None,
    latencies: list[int] | None = None,
    analyze_every: int = 1,
    video: bool = False,
    size: tuple[int, int] | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pmode = ProtectionMode(mode)
    sem = "medium" if pmode == ProtectionMode.MEDIUM else "aggressive"
    analyzers = analyzers or ["auto"]
    latencies = latencies if latencies is not None else [0, 2, 5]
    levels, skipped = _check_levels(levels)
    variants = _variants(levels, analyzers, latencies)
    runs: dict[tuple[str, str], dict] = {}
    videos: dict[tuple[str, str], list] = defaultdict(list)
    errors: list[dict] = []
    t_start = time.perf_counter()
    for res in res_list:
        gt_pages: dict[str, dict] = {}
        for name in sequences:
            seq = make_sequence(name, res=res, seed=seed, n=frames, size=size)
            grid = _grid_for(seq.width)
            for pid, page in seq.pages.items():
                gt_pages.setdefault(pid, page.gt())
            for var in variants:
                key = (var["variant"], res)
                t0 = time.perf_counter()
                try:
                    if var["level"] == "analyzer":
                        r = run_analyzer_level(seq, pmode, var["analyzer"], grid)
                    elif var["level"] == "engine":
                        r = run_engine_level(seq, pmode, var["analyzer"], grid, var["latency"], analyze_every)
                    else:
                        r = run_legacy_level(seq, pmode, grid, var["brain"])
                except Exception as exc:  # noqa: BLE001 - other workstreams may be mid-edit
                    msg = f"{type(exc).__name__}: {exc}"
                    print(f"[eval_live] ERROR {res} {name} {var['variant']}: {msg}", flush=True)
                    errors.append({"variant": var["variant"], "res": res, "sequence": name, "error": msg})
                    continue
                run = runs.setdefault(key, {
                    "latency": int(var.get("latency", 0) or 0),
                    "level": var["level"],
                    "variant": var["variant"],
                    "res": res,
                    "size": [seq.width, seq.height],
                    "impl": r["impl"],
                    "note": r["note"],
                    "evals": [],
                    "timings": defaultdict(list),
                    "oracle_locate": defaultdict(int),
                    "wall_s": 0.0,
                })
                run["evals"].append(r["eval"])
                _merge_timings(run["timings"], r["timings"])
                for k, v in r["oracle_locate"].items():
                    run["oracle_locate"][k] += v
                run["wall_s"] += time.perf_counter() - t0
                if video and res == res_list[0]:
                    videos[key].append((seq, r["eval"], r["masks"]))
                ex = summarize([r["eval"]], sem)["exposure"]
                print(f"[eval_live] {res:6s} {name:20s} {var['variant']:32s} exposed={ex['frames']}/{len(seq)} "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)
            del seq
        (out_dir / f"corpus_gt_{res}.json").write_text(json.dumps({"pages": list(gt_pages.values())}) + "\n")

    results_runs = []
    csv_rows = []
    for (variant, res), run in runs.items():
        evals: list[SequenceEval] = run["evals"]
        settle = run["latency"] + 3  # desktop over-mask / area numbers once results have landed
        overall = summarize(evals, sem, settle=settle)
        per_layout = {lay: summarize(evals, sem, layout=lay, settle=settle) for lay in layouts_of(evals)}
        per_seq = {e.name: summarize([e], sem, settle=settle) for e in evals}
        results_runs.append({
            "level": run["level"],
            "variant": variant,
            "res": res,
            "size": run["size"],
            "impl": run["impl"],
            "note": run["note"],
            "overall": overall,
            "per_layout": per_layout,
            "per_sequence": per_seq,
            "timings_ms": {k: _tstats(v) for k, v in run["timings"].items()},
            "oracle_locate": dict(run["oracle_locate"]),
            "wall_s": round(run["wall_s"], 2),
        })
        for lay, s in [("ALL", overall)] + sorted(per_layout.items()):
            csv_rows.append(_csv_row(variant, res, lay, s))
        if video and (variant, res) in videos:
            fname = "annotated_" + variant.replace("/", "_").replace("[", "_").replace("]", "").replace("=", "") + f"_{res}.mp4"
            write_video(out_dir / fname, videos[(variant, res)], sem, variant)

    results = {
        "meta": {
            "mode": mode,
            "semantics": sem,
            "levels": levels,
            "resolutions": res_list,
            "sequences": sequences,
            "frames_per_sequence": frames,
            "seed": seed,
            "analyzers": analyzers,
            "latencies": latencies,
            "analyze_every": analyze_every,
            "opencv": cv2.__version__,
            "wall_s": round(time.perf_counter() - t_start, 1),
            "measures": "proposals + masking + engine with a GT oracle detector on benign synthetic pages",
            "does_not_measure": "NSFW model accuracy (no objectionable data exists in this corpus)",
        },
        "skipped": skipped,
        "errors": errors,
        "runs": results_runs,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if csv_rows:
        with (out_dir / "per_layout.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    return results


def _csv_row(variant: str, res: str, layout: str, s: dict) -> dict:
    ex = s["exposure"]
    prop = s.get("proposals", {}).get("iou0.5", {})
    lag = s["scroll_follow_lag_frames"]
    return {
        "variant": variant,
        "res": res,
        "layout": layout,
        "frames": s["frames"],
        "exposure_rate": ex["rate"],
        "exposure_frames": ex["frames"],
        "scroll_exposure_rate": ex["during_scroll"]["rate"],
        "nav_exposure_rate": ex["after_navigation"]["rate"],
        "coverage_picture": s["coverage"]["picture_mean"],
        "coverage_parts": s["coverage"]["parts_mean"],
        "over_mask_aggr_frac": s["over_mask"]["aggressive"]["frac_mean"],
        "over_mask_med_frac": s["over_mask"]["medium"]["frac_mean"],
        "lag_mean": lag.get("mean"),
        "lag_events": lag["events"],
        "lag_unresolved": lag["unresolved"],
        "ttc_mean": s["time_to_cover_frames"].get("mean"),
        "recall_iou0.5": prop.get("recall"),
        "precision_iou0.5": prop.get("precision"),
    }


def print_table(results: dict) -> None:
    hdr = f"{'variant':34s} {'res':6s} {'exp':>6s} {'scroll':>7s} {'nav':>6s} {'cov':>6s} {'overA':>7s} {'overM':>7s} {'lag':>5s} {'unres':>5s} {'R@.5':>5s} {'P@.5':>5s} {'ms/f':>7s}"
    print(hdr)
    fmt = lambda v, n=3: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
    for r in results["runs"]:
        s = r["overall"]
        ex = s["exposure"]
        lag = s["scroll_follow_lag_frames"]
        prop = s.get("proposals", {}).get("iou0.5", {})
        t = r["timings_ms"]
        ms = (t.get("analyze_wall_ms") or t.get("reflex_step_ms") or {}).get("mean")
        if r["level"] == "engine":
            ms = t.get("engine_frame_ms_incl_render", {}).get("mean")
        print(f"{r['variant']:34s} {r['res']:6s} {fmt(ex['rate']):>6s} {fmt(ex['during_scroll']['rate']):>7s} {fmt(ex['after_navigation']['rate']):>6s} "
              f"{fmt(s['coverage']['picture_mean']):>6s} {fmt(s['over_mask']['aggressive']['frac_mean'], 4):>7s} "
              f"{fmt(s['over_mask']['medium']['frac_mean'], 4):>7s} {fmt(lag.get('mean'), 2):>5s} {lag['unresolved']:>5d} "
              f"{fmt(prop.get('recall'), 2):>5s} {fmt(prop.get('precision'), 2):>5s} {fmt(ms, 1):>7s}")
    desk = [r for r in results["runs"] if "desktop" in r["overall"]]
    if desk:
        print()
        print(f"{'desktop: variant':34s} {'res':6s} {'sysUI':>6s} {'maxM':>6s} {'outOwn':>7s} {'leak':>6s} {'1flag':>6s} {'clean':>6s}")
        for r in desk:
            d = r["overall"]["desktop"]
            print(f"{r['variant']:34s} {r['res']:6s} {d['sys_cover_px_max']:>6d} {fmt(d['max_mask_frac']):>6s} {d['outside_own_px_max']:>7d} "
                  f"{d['front_leak_px_max']:>6d} {fmt(d['one_flag_area_ratio_max'], 2):>6s} {fmt(d['clean_over_mask_max'], 4):>6s}")
    for s in results["skipped"]:
        print(f"SKIPPED {s['level']}: {s['reason']}")
    for e in results.get("errors", []):
        print(f"ERROR {e['variant']} {e['res']} {e['sequence']}: {e['error']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Screen Guardian live-path eval (synthetic benign screens, GT oracle)")
    parser.add_argument("--mode", choices=["aggressive", "medium"], default="aggressive")
    parser.add_argument("--level", default="all", help="analyzer|engine|legacy|all, or a comma list")
    parser.add_argument("--out", default=None,
                        help="output dir (default experiments/eval_live/latest_<mode>/)")
    parser.add_argument("--res", choices=["half", "retina", "both"], default="both")
    parser.add_argument("--frames", type=int, default=36, help="frames per sequence (12 fps)")
    parser.add_argument("--sequences", default="web",
                        help="comma list of names, or web (the page sequences), desktop (desk_*), all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analyzer", default="auto",
                        help="comma list of auto|real|gt|stub (analyzer/engine levels); gt = perfect proposals")
    parser.add_argument("--latencies", default="0,2,5", help="engine analyzer_latency_frames values")
    parser.add_argument("--analyze-every", type=int, default=1)
    parser.add_argument("--video", action="store_true", help="write annotated mp4s (synthetic frames only)")
    parser.add_argument("--oracle", default="classic", choices=sorted(ORACLE_VARIANTS),
                        help="oracle score variant (eval_live.oracle.ORACLE_VARIANTS); classic = RawDet EXPOSED 0.9")
    args = parser.parse_args(argv)
    set_oracle_variant(args.oracle)
    levels = list(LEVELS) if args.level == "all" else [x.strip() for x in args.level.split(",") if x.strip()]
    bad = [x for x in levels if x not in LEVELS]
    if bad:
        parser.error(f"unknown level(s) {bad}; expected {list(LEVELS)} or all")
    res_list = ["half", "retina"] if args.res == "both" else [args.res]
    seqs: list[str] = []
    for name in (x.strip() for x in args.sequences.split(",") if x.strip()):
        seqs += {"web": list(SEQUENCE_NAMES), "desktop": list(DESKTOP_SEQUENCE_NAMES),
                 "all": list(SEQUENCE_NAMES) + list(DESKTOP_SEQUENCE_NAMES)}.get(name, [name])
    out_dir = Path(args.out) if args.out else Path("experiments") / "eval_live" / f"latest_{args.mode}"
    results = evaluate(
        mode=args.mode,
        levels=levels,
        out_dir=out_dir,
        res_list=res_list,
        sequences=seqs,
        frames=args.frames,
        seed=args.seed,
        analyzers=[a.strip() for a in args.analyzer.split(",") if a.strip()],
        latencies=[int(x) for x in args.latencies.split(",") if x.strip()],
        analyze_every=args.analyze_every,
        video=args.video,
    )
    print_table(results)
    print(f"wrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
