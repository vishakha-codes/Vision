"""LiveAnalyzer mode logic with an injected stub detector (no model, no media).

The stub "finds" the pure-magenta marker that ``synth`` paints inside
flagged pictures and reports it as a body-part box.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import pytest

from vision_core.live.analyzer import make_analyzer
from vision_core.live.budget import TokenBucket
from vision_core.live.cache import VerdictCache, hash_crop
from vision_core.live.contracts import MaskStyle, ProtectionMode, VerdictState
from vision_core.live.detector import RawDet
from vision_core.live.modes import REASON_PENDING, load_mode_policy
from vision_core.live.regions import _iou
from vision_core.live.synth import MARKER_BGR, SynthPage, make_page, photo_texture, scroll_view


class MarkerDetector:
    """Returns a part box over magenta pixels; counts crops it was shown."""

    def __init__(self, label: str = "FEMALE_BREAST_EXPOSED", scores: list[float] | None = None) -> None:
        self.label = label
        self.scores = list(scores or [0.9])
        self.calls = 0
        self.crops = 0

    def detect_batch(self, images):
        score = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        out = []
        for img in images:
            self.crops += 1
            mk = np.array(MARKER_BGR, np.int16)
            m = cv2.inRange(img, np.clip(mk - 20, 0, 255).astype(np.uint8), np.clip(mk + 20, 0, 255).astype(np.uint8))
            ys, xs = np.nonzero(m)
            if xs.size < 20:
                out.append([])
                continue
            out.append([RawDet(self.label, score, (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)))])
        return out


def _screen(markers: int = 2) -> tuple[np.ndarray, list[tuple], list[tuple], list[tuple]]:
    """1728×1117 page: 4 pictures with gutters, the first ``markers`` flagged."""
    rng = np.random.default_rng(3)
    img = np.full((1117, 1728, 3), 250, np.uint8)
    pics = [(100, 100, 500, 420), (520, 100, 920, 420), (100, 500, 700, 1000), (900, 600, 1500, 1000)]
    marks = []
    for i, (x1, y1, x2, y2) in enumerate(pics):
        img[y1:y2, x1:x2] = photo_texture(y2 - y1, x2 - x1, rng)
        if i < markers:
            mx, my = x1 + (x2 - x1) // 3, y1 + (y2 - y1) // 3
            cv2.rectangle(img, (mx, my), (mx + 60, my + 50), MARKER_BGR, -1)
            marks.append((mx, my, mx + 61, my + 51))
    cv2.putText(img, "caption text under the pictures", (100, 470), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    return img, pics, pics[:markers], marks


def _mask_boxes(res, reason_prefix: str = ""):
    return [m.xyxy for m in res.masks if m.reason.startswith(reason_prefix)]


def test_aggressive_covers_whole_picture_with_pad() -> None:
    img, pics, flagged, _ = _screen()
    a = make_analyzer(ProtectionMode.AGGRESSIVE, detector=MarkerDetector(), budget_ms=10_000)
    res = a.analyze(img, t_ms=1000)
    boxes = _mask_boxes(res, "UNSAFE_PICTURE")
    assert len(boxes) == 2
    for f in flagged:
        best = max(boxes, key=lambda b: _iou(b, f))
        assert _iou(best, f) > 0.9
        assert best[0] <= f[0] and best[1] <= f[1] and best[2] >= f[2] and best[3] >= f[3]
    states = {v.state for v in res.verdicts.values()}
    assert VerdictState.SAFE in states and VerdictState.UNSAFE in states
    assert res.counts["n_pending"] == 0


def test_aggressive_explicit_hit_uses_opaque_style() -> None:
    img, *_ = _screen(1)
    a = make_analyzer(ProtectionMode.AGGRESSIVE, detector=MarkerDetector("FEMALE_GENITALIA_EXPOSED"), budget_ms=10_000)
    res = a.analyze(img, t_ms=1000)
    assert [m.style for m in res.masks] == [MaskStyle.OPAQUE]


def test_medium_masks_padded_parts_not_whole_picture() -> None:
    img, pics, flagged, marks = _screen()
    a = make_analyzer(ProtectionMode.MEDIUM, detector=MarkerDetector(), budget_ms=10_000)
    res = a.analyze(img, t_ms=1000)
    boxes = _mask_boxes(res, "UNSAFE_PARTS")
    assert len(boxes) == 2
    for m in marks:
        best = max(boxes, key=lambda b: _iou(b, m))
        w, h = m[2] - m[0], m[3] - m[1]
        # ~12 % pad each side, much smaller than the picture.
        assert best[0] == pytest.approx(m[0] - 0.12 * w, abs=3)
        assert best[3] == pytest.approx(m[3] + 0.12 * h, abs=3)
        assert (best[2] - best[0]) * (best[3] - best[1]) < 0.2 * (pics[0][2] - pics[0][0]) * (pics[0][3] - pics[0][1])


def test_medium_ignores_suggestive_classes() -> None:
    img, *_ = _screen()
    a = make_analyzer(ProtectionMode.MEDIUM, detector=MarkerDetector("BELLY_EXPOSED"), budget_ms=10_000)
    assert a.analyze(img, t_ms=1000).masks == []
    a.set_mode(ProtectionMode.AGGRESSIVE)
    assert len(a.analyze(img, t_ms=1100).masks) == 2


def test_never_mask_classes_never_mask() -> None:
    img, *_ = _screen()
    for label in ("FACE_FEMALE", "FEET_EXPOSED", "ARMPITS_EXPOSED", "BELLY_COVERED"):
        a = make_analyzer(ProtectionMode.AGGRESSIVE, detector=MarkerDetector(label), budget_ms=10_000)
        assert a.analyze(img, t_ms=1000).masks == [], label


def test_off_returns_nothing_quickly() -> None:
    img, *_ = _screen()
    det = MarkerDetector()
    a = make_analyzer(ProtectionMode.OFF, detector=det)
    res = a.analyze(img, t_ms=1000)
    assert res.masks == [] and res.regions == [] and det.crops == 0
    assert res.timings_ms["total"] < 5.0


def test_pending_regions_are_prefrosted_then_classified() -> None:
    img, pics, flagged, _ = _screen()
    det = MarkerDetector()
    a = make_analyzer(ProtectionMode.AGGRESSIVE, detector=det, budget_ms=0.0)
    res = a.analyze(img, t_ms=1000)
    pending = [m for m in res.masks if m.reason == REASON_PENDING]
    assert len(pending) == 4 and det.crops == 0
    assert all(m.verdict is VerdictState.PENDING for m in pending)
    assert all(v.state is VerdictState.PENDING for v in res.verdicts.values())
    # Medium does not pre-frost.
    a.set_mode(ProtectionMode.MEDIUM)
    assert a.analyze(img, t_ms=1050).masks == []
    a.set_mode(ProtectionMode.AGGRESSIVE)
    a._budget_override = 10_000.0
    res2 = a.analyze(img, t_ms=1100)
    assert {m.reason for m in res2.masks} == {"UNSAFE_PICTURE"}
    assert res2.counts["n_classified"] == 4


def test_budget_leaves_rest_pending_for_next_call() -> None:
    page = make_page(1728, 1117, seed=6)
    det = MarkerDetector()
    a = make_analyzer(ProtectionMode.AGGRESSIVE, detector=det, budget_ms=1e-3, max_batch=2)
    res = a.analyze(page.image, t_ms=1000)
    # At least one batch always runs (progress), the rest stay pending.
    assert 0 < res.counts["n_classified"] < res.counts["n_regions"]
    assert res.counts["n_pending"] == res.counts["n_regions"] - res.counts["n_classified"]
    seen = res.counts["n_classified"]
    res2 = a.analyze(page.image, t_ms=1100)
    assert res2.counts["n_cache_hits"] >= seen


def test_cache_hit_after_scroll_costs_no_inference() -> None:
    page = make_page(3456, 2434, seed=8, flagged_every=3)
    det = MarkerDetector()
    a = make_analyzer(ProtectionMode.AGGRESSIVE, detector=det, budget_ms=10_000)
    r0 = a.analyze(scroll_view(page.image, 0, 2234), t_ms=1000)
    crops0 = det.crops
    r1 = a.analyze(scroll_view(page.image, 40, 2234), t_ms=1100)
    # Only pictures clipped at the frame edge show new pixels.
    edge = [r for r in r1.regions if r.xyxy[1] <= 1 or r.xyxy[3] >= 2233]
    assert r1.counts["n_classified"] <= len(edge)
    assert r1.counts["n_cache_hits"] >= r1.counts["n_regions"] - len(edge)
    assert det.crops - crops0 <= 3 * max(1, len(edge))
    # Masks followed the content up by 40 px; ids are stable.
    m0 = {m.region_id: m.xyxy for m in r0.masks}
    m1 = {m.region_id: m.xyxy for m in r1.masks}
    common = set(m0) & set(m1)
    assert common
    for rid in common:
        if m0[rid][1] > 60:
            assert m1[rid][1] == pytest.approx(m0[rid][1] - 40, abs=1.5)
    assert any(v.source == "cache" for v in r1.verdicts.values())


def test_unsure_second_look_token_bucket() -> None:
    img, *_ = _screen(4)
    # First pass weak (0.2 = UNSURE for aggressive), second look strong.
    det = MarkerDetector(scores=[0.2, 0.9])
    a = make_analyzer(
        ProtectionMode.AGGRESSIVE, detector=det, budget_ms=10_000, second_look_rate_per_s=1.0, second_look_burst=2.0
    )
    res = a.analyze(img, t_ms=1000)
    assert res.counts["n_second_look"] == 2  # burst
    states = sorted(v.state.value for v in res.verdicts.values())
    assert states.count("UNSAFE") == 2 and states.count("UNSURE") == 2
    # UNSURE is still covered in aggressive (fail closed).
    assert {m.reason for m in res.masks} == {"UNSAFE_PICTURE", "UNSURE_PICTURE"}
    res = a.analyze(img, t_ms=1500)  # +0.5 token
    assert res.counts["n_second_look"] == 0
    res = a.analyze(img, t_ms=2600)  # refilled
    assert res.counts["n_second_look"] >= 1


def test_medium_unsure_waits_for_second_look() -> None:
    img, *_ = _screen(1)
    det = MarkerDetector(scores=[0.35, 0.35])  # UNSURE band for medium (0.30..0.45)
    a = make_analyzer(ProtectionMode.MEDIUM, detector=det, budget_ms=10_000, second_look_rate_per_s=1.0, second_look_burst=1.0)
    a.bucket.tokens = 0.0  # bucket empty
    res = a.analyze(img, t_ms=1000)
    assert res.masks == []  # no token yet → nothing masked
    res = a.analyze(img, t_ms=2500)
    assert res.counts["n_second_look"] == 1
    assert [m.reason for m in res.masks] == ["UNSURE_PARTS"]


def test_second_look_clears_false_unsure() -> None:
    img, *_ = _screen(1)
    det = MarkerDetector(scores=[0.2, 0.05])
    a = make_analyzer(ProtectionMode.AGGRESSIVE, detector=det, budget_ms=10_000)
    res = a.analyze(img, t_ms=1000)
    assert res.masks == []
    assert all(v.state is VerdictState.SAFE for v in res.verdicts.values())


def test_changed_rect_keeps_masks_for_whole_frame() -> None:
    img, pics, flagged, _ = _screen()
    det = MarkerDetector()
    a = make_analyzer(ProtectionMode.AGGRESSIVE, detector=det, budget_ms=10_000)
    a.analyze(img, t_ms=1000)
    crops = det.crops
    img2 = img.copy()
    rng = np.random.default_rng(99)
    img2[600:1000, 900:1500] = photo_texture(400, 600, rng)  # picture 4 changed in place
    res = a.analyze(img2, changed=(900.0, 600.0, 1500.0, 1000.0), t_ms=1100)
    assert len(_mask_boxes(res, "UNSAFE_PICTURE")) == 2  # still from memory
    assert det.crops - crops == 1  # only the changed picture
    assert len(res.regions) == 4


def test_video_region_keeps_id_and_cover_while_content_changes() -> None:
    rng = np.random.default_rng(4)
    det = MarkerDetector()
    a = make_analyzer(ProtectionMode.AGGRESSIVE, detector=det, budget_ms=10_000)
    ids = set()
    kinds = []
    for t in range(4):
        img = np.full((1117, 1728, 3), 250, np.uint8)
        img[200:800, 300:1300] = photo_texture(600, 1000, rng)
        cv2.rectangle(img, (700, 400), (800, 480), MARKER_BGR, -1)
        res = a.analyze(img, t_ms=1000 + 100 * t)
        ids |= {r.region_id for r in res.regions}
        kinds.append(res.regions[0].kind)
        assert len(_mask_boxes(res, "UNSAFE")) == 1
    assert len(ids) == 1 and kinds[-1] == "video"


def test_policies_document_modes() -> None:
    agg = load_mode_policy(ProtectionMode.AGGRESSIVE)
    med = load_mode_policy(ProtectionMode.MEDIUM)
    off = load_mode_policy(ProtectionMode.OFF)
    assert agg.mask == "picture" and agg.prefrost_mode == "always" and agg.score_on <= 0.3
    assert med.mask == "parts" and med.prefrost_mode == "adaptive" and med.score_on >= 0.4
    assert med.classes < agg.classes
    assert off.mask == "none"
    for pol in (agg, med):
        assert not any(c.startswith(("FACE_", "FEET_", "ARMPITS_")) or c == "BELLY_COVERED" for c in pol.classes)


def test_token_bucket_refills_and_caps() -> None:
    b = TokenBucket(rate_per_s=4.0, burst=4.0)
    assert all(b.take(0.0) for _ in range(4)) and not b.take(0.0)
    assert b.take(0.25) and not b.take(0.25)
    assert b.available(100.0) == pytest.approx(4.0)


def test_verdict_cache_hamming_ttl_lru_and_size_buckets() -> None:
    rng = np.random.default_rng(0)
    img = photo_texture(300, 400, rng)
    d, a = hash_crop(img)
    d2, a2 = hash_crop(img[1:299, 1:399])
    cache = VerdictCache(max_entries=2, ttl_ms=1000, reuse_any_size=lambda v: v == "UNSAFE")
    cache.put(d, a, (400, 300), "SAFE", 0)
    assert cache.get(d2, a2, (398, 298), 10) == "SAFE"  # near-duplicate hash
    assert cache.get(d, a, (1600, 1200), 10) is None  # SAFE not reused 4× bigger
    cache.put(d, a, (400, 300), "UNSAFE", 20)
    assert cache.get(d, a, (1600, 1200), 30) == "UNSAFE"  # UNSAFE reused at any size
    assert cache.get(d, a, (400, 300), 5000) is None  # TTL
    for i in range(3):
        other = photo_texture(300, 400, np.random.default_rng(10 + i))
        cache.put(*hash_crop(other), (400, 300), i, 6000 + i)
    assert len(cache) == 2  # LRU cap


def test_cold_and_cached_timing_with_stub_detector() -> None:
    page = make_page(3456, 2434, seed=2)
    a = make_analyzer(ProtectionMode.AGGRESSIVE, detector=MarkerDetector(), budget_ms=120)
    r0 = a.analyze(scroll_view(page.image, 0, 2234), t_ms=1000)
    r1 = a.analyze(scroll_view(page.image, 0, 2234), t_ms=1100)
    assert r0.counts["n_regions"] >= 18
    assert r1.counts["n_classified"] == 0 and r1.counts["n_crops"] == 0  # all hits (or thin edge strips)
    assert r1.counts["n_cache_hits"] >= r1.counts["n_regions"] - 2
    assert r1.timings_ms["total"] < 150.0  # ~25 ms on M4 Pro; loose for CI
