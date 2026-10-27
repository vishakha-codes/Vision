"""eval_live oracle adapters, legacy baseline end to end, CLI smoke (small frames)."""

from __future__ import annotations

import json

import pytest

from eval_live.metrics import summarize
from eval_live.oracle import GTProposalAnalyzer, LegacyOracle, OracleDetector, OracleTruth
from eval_live.run import evaluate, run_analyzer_level, run_legacy_level
from eval_live.sequences import make_sequence
from vision_core.live.contracts import ProtectionMode

SIZE = (864, 559)


def _flagged_seq(name="search_smooth", n=6):
    seq = make_sequence(name, size=SIZE, n=n)
    assert any(g.flagged() for g in seq.gt)
    return seq


def test_truth_locates_views_copies_and_resizes() -> None:
    seq = _flagged_seq()
    truth = OracleTruth(seq.gt, *SIZE)
    frame = next(iter(truth.iter_frames(seq.frames())))
    assert truth.locate(frame)[:3] == (0, 0.0, 0.0)
    assert truth.locate(frame[100:300, 200:500])[:3] == (0, 200.0, 100.0)
    copy = frame[160:360, 400:640].copy()
    idx, x0, y0, _, _ = truth.locate(copy)
    assert abs(x0 - 400) <= 4 and abs(y0 - 160) <= 4
    import cv2

    small = cv2.resize(frame, (432, 280), interpolation=cv2.INTER_AREA)
    loc = truth.locate(small)
    assert loc[1:3] == (0.0, 0.0) and loc[3] == pytest.approx(2.0, abs=0.01)


def test_oracle_detector_returns_parts_in_crop_pixels() -> None:
    seq = _flagged_seq()
    truth = OracleTruth(seq.gt, *SIZE)
    frame = next(iter(truth.iter_frames(seq.frames())))
    pic = seq.gt[0].flagged()[0]
    x1, y1, x2, y2 = pic.xyxy
    dets = OracleDetector(truth).detect(frame[y1:y2, x1:x2])
    assert dets
    for d in dets:
        box = d.xyxy if hasattr(d, "xyxy") else d["xyxy"]
        assert 0 <= box[0] < box[2] <= x2 - x1 and 0 <= box[1] < box[3] <= y2 - y1


def test_gt_proposal_analyzer_is_perfect() -> None:
    seq = _flagged_seq(n=5)
    r = run_analyzer_level(seq, ProtectionMode.AGGRESSIVE, "gt", 1)
    s = summarize([r["eval"]], "aggressive")
    assert s["exposure"]["frames"] == 0
    assert s["proposals"]["iou0.7"]["recall"] == 1.0
    med = run_analyzer_level(seq, ProtectionMode.MEDIUM, "gt", 1)
    sm = summarize([med["eval"]], "medium")
    assert sm["exposure"]["frames"] == 0
    assert sm["over_mask"]["medium"]["on_flagged_non_part_px_mean"] == 0


def test_stub_analyzer_covers_nothing() -> None:
    seq = _flagged_seq(n=3)
    r = run_analyzer_level(seq, ProtectionMode.AGGRESSIVE, "stub", 1)
    assert summarize([r["eval"]], "aggressive")["exposure"]["rate"] == 1.0


def test_legacy_box_fn_returns_orig_space_parts() -> None:
    seq = _flagged_seq()
    truth = OracleTruth(seq.gt, *SIZE)
    frame = next(iter(truth.iter_frames(seq.frames())))
    raw = LegacyOracle(truth).box_fn(frame)
    n_parts = sum(len(p.parts) for p in seq.gt[0].flagged())
    assert len(raw) == n_parts and all(item["space"] == "orig" for item in raw)


def test_legacy_baseline_end_to_end() -> None:
    seq = make_sequence("nav_search_to_news", size=SIZE, n=10)
    r = run_legacy_level(seq, ProtectionMode.MEDIUM, 1, "instant_brain")
    assert len(r["masks"]) == 10
    agg = summarize([r["eval"]], "aggressive")
    med = summarize([r["eval"]], "medium")
    # v1 masks part clusters, not pictures: parts mostly covered, pictures not
    assert med["coverage"]["parts_mean"] > 2 * agg["coverage"]["picture_mean"]
    assert agg["coverage"]["picture_mean"] < 0.5
    assert agg["exposure"]["rate"] == 1.0
    assert r["timings"]["reflex_step_ms"]


def test_cli_evaluate_writes_results(tmp_path) -> None:
    res = evaluate(
        mode="aggressive",
        levels=["analyzer", "legacy"],
        out_dir=tmp_path,
        res_list=["half"],
        sequences=["shop_momentum"],
        frames=4,
        analyzers=["gt", "stub"],
        size=SIZE,
    )
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["meta"]["does_not_measure"].startswith("NSFW model accuracy")
    variants = {r["variant"] for r in data["runs"]}
    assert {"analyzer[gt]", "analyzer[stub]", "legacy[reflex/slow_brain]"} <= variants
    assert (tmp_path / "per_layout.csv").exists()
    gt = [r for r in res["runs"] if r["variant"] == "analyzer[gt]"][0]
    assert gt["overall"]["exposure"]["frames"] == 0
    assert "shopping" in gt["per_layout"]


def test_engine_level_runs_or_skips(tmp_path) -> None:
    pytest.importorskip("edge.live_engine")
    data = evaluate(
        mode="aggressive",
        levels=["engine"],
        out_dir=tmp_path,
        res_list=["half"],
        sequences=["news_smooth"],
        frames=4,
        analyzers=["gt"],
        latencies=[0, 2],
        size=SIZE,
    )
    assert {r["variant"] for r in data["runs"]} == {"engine[gt]/latency=0", "engine[gt]/latency=2"}
    assert all(r["overall"]["frames"] == 4 for r in data["runs"])


def test_oracle_injects_into_real_analyzer() -> None:
    pytest.importorskip("vision_core.live.analyzer")
    seq = _flagged_seq(n=2)
    r = run_analyzer_level(seq, ProtectionMode.AGGRESSIVE, "real", 1)
    assert r["impl"].startswith("real+oracle_detector") or r["impl"] == "real_proposals+oracle_wrap"
    assert r["oracle_locate"]["unlocated"] == 0
