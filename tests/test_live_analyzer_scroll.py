"""Analyzer through scrolling sequences: pictures scrolling in / out under a
static toolbar keep their id and cover (stub marker detector, cv2 drawing only)."""

from __future__ import annotations

import cv2
import numpy as np

from test_live_analyzer import MarkerDetector
from vision_core.live.analyzer import make_analyzer
from vision_core.live.contracts import ProtectionMode
from vision_core.live.modes import REASON_CLIPPED
from vision_core.live.synth import MARKER_BGR, photo_texture

W, H, BAR = 1728, 1117, 108
PIC = (500, 0, 1100, 420)  # page coords of the flagged picture (marker at its centre)


def _page() -> np.ndarray:
    rng = np.random.default_rng(5)
    page = np.full((4000, W, 3), 250, np.uint8)
    for y in range(200, 3800, 520):
        x1, y1, x2, y2 = PIC[0], y, PIC[2], y + 420
        page[y1:y2, x1:x2] = photo_texture(420, 600, rng)
    # Only the picture at page y=1240 is flagged; its marker sits low in it.
    cv2.rectangle(page, (760, 1240 + 300), (840, 1240 + 360), MARKER_BGR, -1)
    return page


def _frame(page: np.ndarray, y0: int) -> np.ndarray:
    view = page[y0 : y0 + H - BAR]
    frame = np.full((H, W, 3), 255, np.uint8)
    frame[:BAR - 2] = (236, 236, 238)
    frame[BAR - 2 : BAR] = (200, 200, 200)  # toolbar border
    frame[BAR:] = view
    return frame


def _flagged_rect(y0: int) -> tuple[float, float, float, float]:
    top = 1240 - y0 + BAR
    return (500.0, max(float(BAR), top), 1100.0, min(float(H), top + 420.0))


def _covered(res, rect) -> float:
    cov = np.zeros((H, W), bool)
    for m in res.masks:
        x1, y1, x2, y2 = (int(round(v)) for v in m.xyxy)
        cov[max(0, y1) : y2, max(0, x1) : x2] = True
    x1, y1, x2, y2 = (int(v) for v in rect)
    return float(cov[y1:y2, x1:x2].mean()) if y2 > y1 else 1.0


def test_aggressive_covers_picture_while_it_scrolls_in_and_out() -> None:
    page = _page()
    a = make_analyzer(ProtectionMode.AGGRESSIVE, detector=MarkerDetector(), budget_ms=10_000)
    ids = set()
    for step, y0 in enumerate(range(200, 1800, 40)):
        res = a.analyze(_frame(page, y0), t_ms=1000 + 83 * step)
        rect = _flagged_rect(y0)
        if rect[3] - rect[1] < 12:
            continue  # not on screen (or a few px)
        assert _covered(res, rect) >= 0.95, (y0, rect, [(m.reason, m.xyxy) for m in res.masks])
        for reg in res.regions:
            if reg.xyxy[0] == 500 and abs(reg.xyxy[1] - rect[1]) < 8:
                ids.add(reg.region_id)
    assert len(ids) <= 2  # one id while partly visible, kept after it is seen whole


def test_clipped_cover_reason_then_released_when_seen_whole() -> None:
    page = _page()
    a = make_analyzer(ProtectionMode.AGGRESSIVE, detector=MarkerDetector(), budget_ms=10_000)
    # A benign picture (page y=1760) entering at the bottom: covered while clipped ...
    y0 = 1760 + 120 - (H - BAR)  # it shows 120 px
    res = a.analyze(_frame(page, y0), t_ms=1000)
    entering = (500.0, float(H - 120), 1100.0, float(H))
    assert _covered(res, entering) >= 0.95
    assert any(m.reason == REASON_CLIPPED for m in res.masks)
    # ... and not covered once fully visible and checked SAFE.
    res = a.analyze(_frame(page, 1760 - 300), t_ms=1100)
    benign = (500.0, 300.0 + BAR, 1100.0, 720.0 + BAR)
    assert _covered(res, benign) < 0.05


def test_medium_keeps_part_boxes_while_scrolling_out() -> None:
    page = _page()
    a = make_analyzer(ProtectionMode.MEDIUM, detector=MarkerDetector(), budget_ms=10_000)
    a.analyze(_frame(page, 1100), t_ms=1000)  # flagged picture whole on screen
    for step, y0 in enumerate(range(1140, 1500, 40)):
        res = a.analyze(_frame(page, y0), t_ms=1100 + 83 * step)
        marker = (760.0, 1240 + 300 - y0 + BAR, 840.0, 1240 + 360 - y0 + BAR)
        if marker[3] <= BAR + 4:
            break
        vis = (marker[0], max(float(BAR), marker[1]), marker[2], marker[3])
        assert _covered(res, vis) >= 0.95, (y0, [(m.reason, m.xyxy) for m in res.masks])
        on_pic = [m for m in res.masks if m.xyxy[1] < _flagged_rect(y0)[3] and m.xyxy[3] > _flagged_rect(y0)[1]]
        assert all((m.xyxy[2] - m.xyxy[0]) < 300 for m in on_pic)  # parts, not the picture


def test_toolbar_band_is_not_proposed() -> None:
    page = _page()
    a = make_analyzer(ProtectionMode.AGGRESSIVE, detector=MarkerDetector(), budget_ms=10_000)
    for step, y0 in enumerate((300, 340, 380)):
        res = a.analyze(_frame(page, y0), t_ms=1000 + 83 * step)
    assert a._clip_top is not None and abs(a._clip_top - BAR) <= 4
    assert all(r.xyxy[1] >= BAR - 4 for r in res.regions)
