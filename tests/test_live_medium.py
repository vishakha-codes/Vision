"""Medium (teens) mode: covered classes, per-class thresholds, pre-frost
policy, second-look token bucket across a cold start.

Benign synthetic pages only; the detector is ``eval_live.fakenet.FakeRawNet``
(real letterbox / decode / NMS, synthetic model output on magenta markers).
"""

from __future__ import annotations

import json

import pytest

from eval_live.fakenet import FakeRawNet
from vision_core.live.analyzer import make_analyzer
from vision_core.live.budget import TokenBucket
from vision_core.live.contracts import PictureRegion, ProtectionMode, VerdictState
from vision_core.live.modes import (
    REASON_PENDING,
    Evidence,
    NormDet,
    decide,
    load_mode_policy,
    masks_for,
)
from vision_core.live.synth import make_page

REGION = PictureRegion("r0", (100.0, 100.0, 500.0, 400.0))


def _ev(*dets: tuple[str, float], second: tuple[tuple[str, float], ...] | None = None) -> Evidence:
    box = (0.2, 0.2, 0.5, 0.6)
    first = tuple(NormDet(lab, sc, box) for lab, sc in dets)
    two = None if second is None else tuple(NormDet(lab, sc, box) for lab, sc in second)
    return Evidence(dets=first, dets2=two)


# ---------------------------------------------------------------- classes / thresholds


def test_medium_counts_covered_intimate_classes_at_060() -> None:
    med = load_mode_policy(ProtectionMode.MEDIUM)
    for lab in ("FEMALE_BREAST_COVERED", "FEMALE_GENITALIA_COVERED", "BUTTOCKS_COVERED", "ANUS_COVERED"):
        assert lab in med.classes
        assert med.on_threshold(lab) == pytest.approx(0.60)
        assert decide(med, REGION, _ev((lab, 0.72))).state is VerdictState.UNSAFE
        assert decide(med, REGION, _ev((lab, 0.59))).state is VerdictState.UNSURE
        assert decide(med, REGION, _ev((lab, 0.40))).state is VerdictState.SAFE  # swimwear-ish stays clear
    assert med.on_threshold("FEMALE_BREAST_EXPOSED") == pytest.approx(0.45)
    assert decide(med, REGION, _ev(("FEMALE_BREAST_EXPOSED", 0.46))).state is VerdictState.UNSAFE
    assert decide(med, REGION, _ev(("FEMALE_BREAST_EXPOSED", 0.35))).state is VerdictState.UNSURE


def test_medium_still_excludes_faces_feet_armpits_belly() -> None:
    med = load_mode_policy(ProtectionMode.MEDIUM)
    for lab in ("FACE_FEMALE", "FACE_MALE", "FEET_EXPOSED", "ARMPITS_EXPOSED", "BELLY_EXPOSED", "BELLY_COVERED", "MALE_BREAST_EXPOSED"):
        assert lab not in med.classes
        assert decide(med, REGION, _ev((lab, 0.99))).state is VerdictState.SAFE


def test_user_live_scores_now_mask_in_medium() -> None:
    """The scores the user's real screen produced (26 Oct) used to give nothing."""
    med = load_mode_policy(ProtectionMode.MEDIUM)
    ev = _ev(("FEMALE_BREAST_COVERED", 0.72), ("FEMALE_GENITALIA_COVERED", 0.64), ("FACE_FEMALE", 0.9))
    v = decide(med, REGION, ev)
    assert v.state is VerdictState.UNSAFE
    assert v.labels == ("FEMALE_BREAST_COVERED", "FEMALE_GENITALIA_COVERED")
    masks = masks_for(med, REGION, v, (1000, 800), evidence=ev)
    assert masks and all(m.reason == "UNSAFE_PARTS" for m in masks)


def test_unknown_or_never_labels_in_a_policy_file_are_dropped(tmp_path) -> None:
    raw = json.loads((load_mode_policy.__globals__["POLICY_DIR"] / "live_medium.json").read_text())
    raw["classes"] = raw["classes"] + ["MALE_GENITALIA_COVERED", "female_breast_exposed", "FACE_FEMALE"]
    (tmp_path / "live_medium.json").write_text(json.dumps(raw))
    pol = load_mode_policy(ProtectionMode.MEDIUM, tmp_path)
    assert "MALE_GENITALIA_COVERED" not in pol.classes  # not a NudeNet label
    assert "female_breast_exposed" not in pol.classes  # labels are case-exact
    assert "FACE_FEMALE" not in pol.classes


def test_missing_policy_file_has_the_same_medium_defaults(tmp_path) -> None:
    built_in = load_mode_policy(ProtectionMode.MEDIUM, tmp_path)  # no JSON there
    shipped = load_mode_policy(ProtectionMode.MEDIUM)
    assert built_in.classes == shipped.classes
    assert built_in.class_score_on == shipped.class_score_on
    assert built_in.class_score_unsure == shipped.class_score_unsure
    assert built_in.prefrost_mode == shipped.prefrost_mode == "adaptive"


# ---------------------------------------------------------------- pre-frost policy


def test_prefrost_block_per_mode() -> None:
    agg = load_mode_policy(ProtectionMode.AGGRESSIVE)
    med = load_mode_policy(ProtectionMode.MEDIUM)
    off = load_mode_policy(ProtectionMode.OFF)
    assert (agg.prefrost_mode, med.prefrost_mode, off.prefrost_mode) == ("always", "adaptive", "never")
    for pol in (agg, med, off):
        assert pol.prefrost_heavy_masks == 3
        assert pol.prefrost_heavy_window_s == pytest.approx(10.0)
        assert not hasattr(pol, "prefrost_pending")  # superseded
    assert agg.prefrost_active() and agg.prefrost_active(heavy=True)
    assert not med.prefrost_active() and med.prefrost_active(heavy=True)
    assert not off.prefrost_active(heavy=True)


def test_adaptive_prefrost_only_when_heavy() -> None:
    med = load_mode_policy(ProtectionMode.MEDIUM)
    pending = decide(med, REGION, None)
    assert pending.state is VerdictState.PENDING
    assert masks_for(med, REGION, pending, (1000, 800)) == []
    heavy = masks_for(med, REGION, pending, (1000, 800), heavy=True)
    assert [m.reason for m in heavy] == [REASON_PENDING]
    agg = load_mode_policy(ProtectionMode.AGGRESSIVE)
    assert [m.reason for m in masks_for(agg, REGION, pending, (1000, 800))] == [REASON_PENDING]


@pytest.mark.parametrize(("legacy", "want"), [(True, "always"), (False, "never")])
def test_legacy_prefrost_pending_key_still_loads(tmp_path, legacy: bool, want: str) -> None:
    raw = json.loads((load_mode_policy.__globals__["POLICY_DIR"] / "live_medium.json").read_text())
    raw.pop("prefrost")
    raw["prefrost_pending"] = legacy
    (tmp_path / "live_medium.json").write_text(json.dumps(raw))
    pol = load_mode_policy(ProtectionMode.MEDIUM, tmp_path)
    assert pol.prefrost_mode == want and pol.prefrost_heavy_masks == 3


def test_aggressive_clipped_release_matches_its_note() -> None:
    agg = load_mode_policy(ProtectionMode.AGGRESSIVE)
    raw = json.loads((load_mode_policy.__globals__["POLICY_DIR"] / "live_aggressive.json").read_text())
    assert agg.clipped_release_static == 3
    assert "3 calls" in raw["notes"]["clipped_release"]


# ---------------------------------------------------------------- token bucket


def test_bucket_rebases_when_the_clock_jumps_back() -> None:
    b = TokenBucket(rate_per_s=4.0, burst=4.0)
    # First call on a monotonic clock (~321,000 s), then the capture clock (0 s).
    assert all(b.take(321_000.0) for _ in range(4)) and not b.take(321_000.0)
    b.available(0.083)
    assert b.resyncs == 1 and b.last_s == pytest.approx(0.083)
    assert b.take(0.6)  # refills on the new base instead of waiting 89 hours
    assert b.available(5.0) == pytest.approx(4.0)
    assert b.tokens <= b.burst


def test_bucket_ignores_non_finite_time() -> None:
    b = TokenBucket(rate_per_s=4.0, burst=2.0)
    assert b.take(1.0) and b.take(1.0) and not b.take(1.0)
    assert not b.take(float("nan"))
    assert b.take(1.5)


def test_second_looks_continue_after_cold_start_at_t0() -> None:
    """Frame 0 stamped t_ms = 0 with > 4 UNSURE pictures, then a new page:
    second looks (and therefore medium masks) must keep coming."""
    pages = [make_page(1728, 1117, seed=s, flagged_every=2) for s in (1, 2)]
    assert all(len(p.flagged) > 4 for p in pages)
    an = make_analyzer(mode=ProtectionMode.MEDIUM, detector=FakeRawNet({"FEMALE_BREAST_EXPOSED": 0.38}))
    looks, masks = [], []
    for i in range(12):
        res = an.analyze(pages[i // 6].image, t_ms=i * 500.0)  # frame 0 at t_ms = 0
        looks.append(res.counts.get("n_second_look", 0))
        masks.append(len(res.masks))
    assert sum(looks[:6]) >= len(pages[0].flagged)
    assert sum(looks[6:]) >= len(pages[1].flagged), looks  # broke before: bucket never refilled
    assert masks[5] >= len(pages[0].flagged) and masks[11] >= len(pages[1].flagged), masks
