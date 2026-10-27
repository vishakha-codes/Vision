"""eval_live oracle score variants + the raw-tensor fake detector."""

from __future__ import annotations

import pytest

from eval_live import oracle
from eval_live.fakenet import FakeRawNet
from eval_live.metrics import summarize
from eval_live.oracle import (
    ORACLE_VARIANTS,
    LegacyOracle,
    OracleDetector,
    OracleTruth,
    TimeShifted,
    make_live_analyzer,
    make_oracle_detector,
    set_oracle_variant,
)
from eval_live.run import run_analyzer_level
from eval_live.sequences import make_sequence
from vision_core.live.contracts import ProtectionMode

SIZE = (864, 559)


@pytest.fixture(autouse=True)
def _reset_variant():
    yield
    set_oracle_variant("classic")


def _seq(n: int = 6):
    seq = make_sequence("search_smooth", size=SIZE, n=n)
    assert any(g.flagged() for g in seq.gt)
    return seq


def test_variant_selection() -> None:
    truth = OracleTruth(_seq().gt, *SIZE)
    assert isinstance(make_oracle_detector(truth, "classic"), OracleDetector)
    for name in ("realistic", "confusion", "covered", "unsure", "unsure_shift"):
        assert isinstance(make_oracle_detector(truth, name), FakeRawNet)
    with pytest.raises(ValueError):
        set_oracle_variant("nope")
    set_oracle_variant("unsure_shift")
    an, label, note = make_live_analyzer(ProtectionMode.MEDIUM, truth, "real")
    assert isinstance(an, TimeShifted) and "unsure_shift" in label and "t_ms + 1000" in note


def test_confusion_variant_decodes_covered_top_and_exposed_runner_up() -> None:
    seq = _seq()
    truth = OracleTruth(seq.gt, *SIZE)
    frame = next(iter(truth.iter_frames(seq.frames())))
    pic = seq.gt[0].flagged()[0]
    x1, y1, x2, y2 = pic.xyxy
    det = make_oracle_detector(truth, "confusion")
    labels = {d.label: d.score for d in det.detect(frame[y1:y2, x1:x2])}
    assert labels["FEMALE_BREAST_COVERED"] == pytest.approx(0.55)
    assert labels["FEMALE_BREAST_EXPOSED"] == pytest.approx(0.50)


@pytest.mark.parametrize("variant", ["classic", "realistic", "confusion", "covered", "unsure"])
def test_medium_masks_flagged_pictures_for_every_variant(variant: str) -> None:
    """Medium used to show nothing on covered / confused scores."""
    set_oracle_variant(variant)
    seq = _seq(n=8)
    r = run_analyzer_level(seq, ProtectionMode.MEDIUM, "real", 1)
    s = summarize([r["eval"]], "medium")
    assert any(r["masks"]), variant
    assert s["coverage"]["parts_mean"] > 0.5, (variant, s["coverage"])


def test_legacy_oracle_uses_the_variant_top_class() -> None:
    truth = OracleTruth(_seq().gt, *SIZE)
    set_oracle_variant("confusion")
    assert LegacyOracle(truth).profile == (("FEMALE_BREAST_COVERED", 0.55),)
    set_oracle_variant("classic")
    assert LegacyOracle(truth).profile == ((oracle.ORACLE_LABEL, oracle.ORACLE_SCORE),)
    assert set(ORACLE_VARIANTS) >= {"classic", "realistic", "confusion", "covered", "unsure"}
