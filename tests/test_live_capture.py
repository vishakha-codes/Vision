"""Live capture + screen-space overlay. No real display required."""

from __future__ import annotations

import numpy as np
import pytest

from edge.capture import LiveCapture, MssGrabber, QuartzGrabber, open_source, preferred_live_backend
from edge.coords import DisplayGeom, capture_xyxy_to_cocoa, scale_from_sizes
from edge.reflex import Reflex
from edge.screen_overlay import MemoryOverlayBackend, ScreenOverlay, _mask_boxes
from vision_core.contracts import ActionType, ProtectionAction, TrackedRegion, TrackMode


def _geom() -> DisplayGeom:
    return DisplayGeom(
        left_pt=0.0,
        top_pt=0.0,
        width_pt=1440.0,
        height_pt=900.0,
        width_px=2880,
        height_px=1800,
        scale_x=2.0,
        scale_y=2.0,
        cocoa_origin_x=0.0,
        cocoa_origin_y=0.0,
        cocoa_height_pt=900.0,
    )


def test_scale_from_sizes_retina() -> None:
    sx, sy = scale_from_sizes(2880, 1800, 1440.0, 900.0)
    assert sx == 2.0
    assert sy == 2.0


def test_capture_box_maps_to_cocoa_bottom_left() -> None:
    geom = _geom()
    # 200×100 px box at top-left of the grab → 100×50 pt, y from top = 0
    x, y, w, h = capture_xyxy_to_cocoa((0.0, 0.0, 200.0, 100.0), geom)
    assert x == 0.0
    assert w == 100.0
    assert h == 50.0
    assert y == 850.0  # 900 - 0 - 50


def test_live_capture_uses_injected_grabber() -> None:
    frame = np.full((180, 320, 3), 30, dtype=np.uint8)
    geom = _geom()

    def grabber(**_kwargs):
        return frame.copy(), geom

    cap = LiveCapture(grabber=grabber, target_fps=15.0)
    assert cap.available is True
    got = cap.read()
    assert got is not None
    assert got.width == 2880 or got.image_bgr.shape[1] == 320
    assert got.image_bgr.shape == (180, 320, 3)
    assert cap.geom is geom
    cap.release()


def test_live_capture_permission_denied_message() -> None:
    cap = LiveCapture()
    assert cap.available is False
    with pytest.raises(RuntimeError, match="pytest|Screen Recording|capture unavailable"):
        cap.require()


def test_open_source_live_is_live_capture() -> None:
    cap = open_source("live")
    assert isinstance(cap, LiveCapture)
    assert cap.read() is None


def test_quartz_and_mss_grabbers_exist() -> None:
    assert QuartzGrabber().grab.__name__ == "grab"
    assert MssGrabber().grab.__name__ == "grab"
    assert preferred_live_backend() in {"quartz", "mss"}


def test_overlay_places_rects_in_screen_space() -> None:
    from vision_core.contracts import Frame

    backend = MemoryOverlayBackend()
    overlay = ScreenOverlay(backend=backend)
    region = TrackedRegion(
        1,
        (200.0, 100.0, 600.0, 400.0),
        "flagged",
        0.9,
        0.1,
        0.0,
        TrackMode.TRACKED,
        mask_on=True,
    )
    action = ProtectionAction(ActionType.BLUR, region, "FLAGGED_ABOVE_OVERLAY", "balanced-v2", "t")
    image = np.full((1800, 2880, 3), 16, dtype=np.uint8)
    state = Reflex("balanced").step(None)
    state.actions = [action]
    state.frame = Frame(0, 0.0, image, 2880, 1800)
    overlay.sync(state, _geom())
    assert backend.rects
    _x, _y, w, h = backend.rects[0]
    # 400×300 px at (200,100) / scale 2 = 200×150 pt, padded by 8
    assert w == pytest.approx(200.0 + 16.0)
    assert h == pytest.approx(150.0 + 16.0)
    assert overlay.mask_count() >= 1
    overlay.close()
    assert backend.closed is True
    assert overlay.mask_count() == 0


def test_live_capture_passes_exclude_window_ids() -> None:
    frame = np.full((180, 320, 3), 30, dtype=np.uint8)
    geom = _geom()
    seen: dict = {}

    def grabber(*, exclude_window_ids=None, exclude_titles=None, **_kwargs):
        seen["ids"] = list(exclude_window_ids or [])
        seen["titles"] = exclude_titles
        return frame.copy(), geom

    cap = LiveCapture(grabber=grabber, target_fps=15.0)
    cap.exclude_window_ids = [25731, 99]
    got = cap.read()
    assert got is not None
    assert 25731 in seen["ids"]
    assert 99 in seen["ids"]
    cap.release()


def test_mask_boxes_skip_non_protective() -> None:
    region = TrackedRegion(1, (0.0, 0.0, 10.0, 10.0), "flagged", 0.2, 0.8, 0.0, TrackMode.TRACKED)
    action = ProtectionAction(ActionType.NO_ACTION, region, "BELOW_THRESHOLD", "balanced-v2", "t")
    reflex = Reflex("balanced")
    state = reflex.step(None)
    state.actions = [action]
    assert _mask_boxes(state) == []
