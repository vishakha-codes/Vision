"""eval_live metrics on hand-built cases."""

from __future__ import annotations

import pytest

from eval_live.metrics import evaluate_sequence, follow_iou, match_proposals, summarize
from eval_live.sequences import FrameGT, GTPicture

W, H = 200, 150
SCREEN = (0, 0, W, H)


def _pic(gid: str, box, *, flagged: bool, parts=()) -> GTPicture:
    return GTPicture(gid=gid, xyxy=box, full_xyxy=box, clip=SCREEN, flagged=flagged, kind="picture",
                     parts=tuple(parts), full_parts=tuple(parts))


def _gts(boxes_per_frame, *, dy=None) -> list[FrameGT]:
    out = []
    for i, pics in enumerate(boxes_per_frame):
        out.append(FrameGT(index=i, t_ms=i * 83.3, layout="hand", pictures=pics,
                           scroll_dy=(dy[i] if dy else 0)))
    return out


def _static(n=4):
    flag = (20, 20, 80, 80)
    other = (120, 20, 180, 80)
    return _gts([[_pic("f", flag, flagged=True, parts=[(30, 30, 50, 50)]), _pic("o", other, flagged=False)]
                 for _ in range(n)]), flag, other


def test_perfect_masks_zero_exposure() -> None:
    gts, flag, _ = _static()
    ev = evaluate_sequence("t", gts, [[flag]] * 4, W, H)
    s = summarize([ev], "aggressive")
    assert s["exposure"]["frames"] == 0
    assert s["coverage"]["picture_mean"] == pytest.approx(1.0)
    assert s["over_mask"]["aggressive"]["px_mean"] == 0
    # same whole-picture mask is over-mask under MEDIUM semantics (only parts wanted)
    assert s["over_mask"]["medium"]["on_flagged_non_part_px_mean"] > 0
    assert summarize([ev], "medium")["exposure"]["frames"] == 0
    assert s["time_to_cover_frames"]["covered_on_arrival_rate"] == 1.0


def test_no_masks_every_frame_exposed() -> None:
    gts, _, _ = _static()
    s = summarize([evaluate_sequence("t", gts, [[]] * 4, W, H)], "aggressive")
    assert s["exposure"]["frames"] == 4 and s["exposure"]["rate"] == 1.0
    assert s["time_to_cover_frames"]["never_covered"] == 1


def test_exposure_threshold_is_five_percent() -> None:
    gts, flag, _ = _static(1)
    x1, y1, x2, y2 = flag  # 60×60 = 3600 px
    ok = [(x1, y1, x2, y2 - 2)]  # 120 px uncovered = 3.3%
    bad = [(x1, y1, x2, y2 - 6)]  # 360 px = 10%
    assert summarize([evaluate_sequence("t", gts, [ok], W, H)], "aggressive")["exposure"]["frames"] == 0
    assert summarize([evaluate_sequence("t", gts, [bad], W, H)], "aggressive")["exposure"]["frames"] == 1


def test_parts_only_mask_exposes_aggressive_not_medium() -> None:
    gts, _, _ = _static(2)
    ev = evaluate_sequence("t", gts, [[(30, 30, 50, 50)]] * 2, W, H, part_pad=0)
    assert summarize([ev], "aggressive")["exposure"]["frames"] == 2
    assert summarize([ev], "medium")["exposure"]["frames"] == 0
    assert summarize([ev], "medium")["over_mask"]["medium"]["px_mean"] == 0


def test_masks_one_frame_late_give_lag_one() -> None:
    a = (20, 20, 80, 80)
    b = (20, 60, 80, 120)  # one 40 px scroll at frame 3; IoU(a, b) = 0.2
    boxes = [a, a, a, b, b, b]
    gts = _gts([[_pic("f", x, flagged=True, parts=[(x[0] + 10, x[1] + 10, x[0] + 30, x[1] + 30)])] for x in boxes],
               dy=[0, 0, 0, 40, 0, 0])
    late = [[boxes[max(0, i - 1)]] for i in range(len(boxes))]
    ev = evaluate_sequence("t", gts, late, W, H)
    s = summarize([ev], "aggressive")
    lag = s["scroll_follow_lag_frames"]
    assert lag["events"] == 1 and lag["unresolved"] == 0 and lag["mean"] == 1.0
    assert s["exposure"]["frames"] == 1 and s["exposure"]["during_scroll"]["frames"] == 1
    on_time = summarize([evaluate_sequence("t", gts, [[x] for x in boxes], W, H)], "aggressive")
    assert on_time["scroll_follow_lag_frames"]["mean"] == 0.0
    stuck = summarize([evaluate_sequence("t", gts, [[a]] * 6, W, H)], "aggressive")
    assert stuck["scroll_follow_lag_frames"]["unresolved"] == 1


def test_follow_iou_ignores_neighbour_and_merges_split_masks() -> None:
    t = (0.0, 0.0, 100.0, 100.0)
    neighbour = (95.0, 0.0, 200.0, 100.0)
    assert follow_iou(t, [t, neighbour], (0, 0, 300, 300)) == pytest.approx(1.0)
    halves = [(0.0, 0.0, 50.0, 100.0), (50.0, 0.0, 100.0, 100.0)]
    assert follow_iou(t, halves, (0, 0, 300, 300)) == pytest.approx(1.0)


def test_proposal_recall_precision_thresholds() -> None:
    gts, flag, other = _static(1)
    junk = (0, 100, 30, 140)
    m = match_proposals(gts[0], [flag, other, junk], 0.5)
    assert (m["tp_gt"], m["n_gt"], m["tp_prop"], m["n_prop"]) == (2, 2, 2, 3)
    shifted = (flag[0] + 15, flag[1], flag[2] + 15, flag[3])  # IoU = 45/75 = 0.6
    assert match_proposals(gts[0], [shifted], 0.5)["tp_gt"] == 1
    assert match_proposals(gts[0], [shifted], 0.7)["tp_gt"] == 0
    ev = evaluate_sequence("t", gts, [[]], W, H, proposals=[[flag, other, junk]])
    p = summarize([ev], "aggressive")["proposals"]
    assert p["iou0.5"]["recall"] == 1.0 and p["iou0.5"]["precision"] == pytest.approx(2 / 3, abs=1e-3)
