"""On-device NSFW detect. Mocks for CI; one real-model smoke on a blank frame."""

from __future__ import annotations

import numpy as np
import pytest

from agent.orchestrator import Orchestrator
from edge.reflex import Reflex
from vision_core.contracts import ActionType, Frame, PROTECTIVE_ACTIONS
from vision_core.cluster_nsfw import MAX_CLUSTERS, ClusterHold, live_scene_changed
from vision_core.detect_nsfw import (
    EXPLICIT_CLASSES,
    INTIMATE_CLASSES,
    NEVER_MASK_CLASSES,
    SUGGESTIVE_CLASSES,
    DETECTOR_ID,
    detect_nsfw,
    emit_classes_for_policy,
    label_bucket,
    last_detect_stats,
    make_nsfw_detect,
    nudenet_available,
    nudenet_color_input,
    propose_content_tiles,
)
from vision_core.reinspect_nsfw import make_nsfw_reinspect, reinspect_nsfw, reinspect_rekognition


def _frame(h: int = 240, w: int = 320) -> np.ndarray:
    return np.full((h, w, 3), 36, dtype=np.uint8)


def test_explicit_box_becomes_flagged() -> None:
    def box_fn(_img):
        return [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.91, "box": [20, 20, 80, 80]},
            {"class": "FACE_FEMALE", "score": 0.99, "box": [120, 20, 40, 40]},
        ]

    dets = detect_nsfw(_frame(), box_fn=box_fn, min_score=0.50)
    assert len(dets) == 1
    assert dets[0].label == "flagged"
    assert dets[0].detector_id == DETECTOR_ID
    assert dets[0].score == pytest.approx(0.91)
    assert dets[0].xyxy[2] > dets[0].xyxy[0]


def test_news_like_faces_are_not_flagged() -> None:
    def box_fn(_img):
        return [
            {"class": "FACE_MALE", "score": 0.98, "box": [10, 10, 60, 60]},
            {"class": "FACE_FEMALE", "score": 0.97, "box": [80, 10, 60, 60]},
            {"class": "BELLY_COVERED", "score": 0.90, "box": [10, 80, 60, 60]},
            {"class": "FEET_EXPOSED", "score": 0.88, "box": [80, 80, 50, 40]},
            {"class": "ARMPITS_EXPOSED", "score": 0.86, "box": [20, 140, 50, 40]},
        ]

    assert detect_nsfw(_frame(), box_fn=box_fn, policy_name="balanced") == []
    stats = last_detect_stats()
    assert stats.n_after_policy == 0
    assert "FACE_" in stats.reason and "ignored" in stats.reason


def test_policy_floors_change_emission() -> None:
    def box_fn(_img):
        return [{"class": "BUTTOCKS_EXPOSED", "score": 0.50, "box": [8, 8, 90, 90]}]

    assert detect_nsfw(_frame(), box_fn=box_fn, policy_name="strict")
    assert detect_nsfw(_frame(), box_fn=box_fn, policy_name="balanced")
    assert detect_nsfw(_frame(), box_fn=box_fn, policy_name="relaxed") == []


def test_image_level_score_is_tiled_not_fullscreen() -> None:
    image = _frame(300, 400)

    def score_fn(crop):
        # Only the top-left tile is "hot".
        return 0.92 if crop.shape[0] <= 256 and crop.mean() < 40 else 0.01

    dets = detect_nsfw(image, score_fn=score_fn, min_score=0.80)
    assert dets
    for det in dets:
        x1, y1, x2, y2 = det.xyxy
        assert (x2 - x1) * (y2 - y1) < image.shape[0] * image.shape[1]


def test_reinspect_nsfw_remaps_crop_box() -> None:
    def detect_fn(crop, **_kwargs):
        return detect_nsfw(
            crop,
            box_fn=lambda _c: [{"class": "ANUS_EXPOSED", "score": 0.87, "box": [2, 2, 20, 20]}],
            min_score=0.50,
        )

    result = reinspect_nsfw(
        _frame(),
        (10.0, 10.0, 50.0, 50.0),
        margin=0,
        detect_fn=detect_fn,
    )
    assert result.label == "flagged"
    assert result.score == pytest.approx(0.87)
    assert result.xyxy[0] >= 10.0


def test_rekognition_uses_in_memory_bytes_only() -> None:
    calls: list[dict] = []

    class Client:
        def detect_moderation_labels(self, **kwargs):
            calls.append(kwargs)
            return {"ModerationLabels": [{"Name": "Explicit Nudity", "Confidence": 91.0}]}

    result = reinspect_rekognition(_frame(), (4.0, 4.0, 80.0, 80.0), client=Client())
    assert result.label == "flagged"
    assert result.score == pytest.approx(0.91)
    assert calls and "Bytes" in calls[0]["Image"]
    assert isinstance(calls[0]["Image"]["Bytes"], (bytes, bytearray))


def test_reflex_nsfw_mode_masks_on_high_score() -> None:
    def box_fn(_img):
        return [{"class": "MALE_GENITALIA_EXPOSED", "score": 0.93, "box": [30, 30, 100, 100]}]

    detect = make_nsfw_detect("strict", box_fn=box_fn)
    reinspect = make_nsfw_reinspect(mode="local", detect_fn=detect)
    reflex = Reflex(
        "strict",
        brain_every=1,
        detect_mode="nsfw",
        detect_fn=detect,
        reinspect_fn=reinspect,
    )
    image = _frame(240, 320)
    state = reflex.step(Frame(0, 0.0, image, 320, 240))
    assert state.detect_mode == "nsfw"
    assert any(a.action is ActionType.OPAQUE_MASK for a in state.actions)


def test_nsfw_http_backend_posts_live_tiles() -> None:
    from agent.orchestrator import FrameResult

    calls: list[int] = []

    class Spy:
        def perceive(self, **_kwargs):
            raise AssertionError("fixture /v1/perceive must not be used for live nsfw")

        def live(self, jpeg_bytes, **_kwargs):
            calls.append(len(jpeg_bytes))
            assert jpeg_bytes[:2] == b"\xff\xd8"
            return FrameResult(
                actions=[],
                traces=[],
                detections=[],
                tracks=[],
                change_score=0.0,
                cut=False,
                reinspect_count=0,
                timed_out=False,
                timings_ms={},
                tool_sequence="LIVE_TILE",
                companion_line="ok",
                first_decision="NO_ACTION",
                final_decision="NO_ACTION",
                band="low",
            )

    reflex = Reflex(
        "balanced",
        backend="http://192.0.2.10:8080",
        detect_mode="nsfw",
        detect_fn=make_nsfw_detect("balanced", box_fn=lambda _i: []),
        http_client=Spy(),
        brain_every=1,
    )
    assert reflex.backend_kind == "cloud"
    assert reflex.backend_url == "http://192.0.2.10:8080"
    assert reflex.orch is not None
    state = reflex.step(Frame(0, 0.0, _frame(), 320, 240))
    assert calls and calls[0] > 0
    assert reflex.last_payload_bytes == calls[0]
    assert state.backend == "cloud"


def test_orchestrator_mid_band_still_reinspects() -> None:
    def box_fn(_img):
        return [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.60, "box": [20, 20, 90, 90]}]

    detect = make_nsfw_detect("balanced", box_fn=box_fn)
    orch = Orchestrator("policies/balanced.json", detect_fn=detect, reinspect_fn=make_nsfw_reinspect(detect_fn=detect))
    image = _frame()
    result = orch.process_frame(0, image, None, 0.0)
    assert result.reinspect_count == 1


@pytest.mark.skipif(not nudenet_available(), reason="nudenet weights missing")
def test_real_nudenet_blank_frame_is_empty() -> None:
    dets = detect_nsfw(_frame(96, 96), min_score=0.50)
    assert dets == []


def test_explicit_class_set_is_conservative() -> None:
    assert "FACE_FEMALE" not in EXPLICIT_CLASSES
    assert "FACE_FEMALE" not in INTIMATE_CLASSES
    assert "FEMALE_BREAST_COVERED" not in EXPLICIT_CLASSES
    assert "FEMALE_BREAST_COVERED" in SUGGESTIVE_CLASSES
    assert "FEMALE_GENITALIA_EXPOSED" in EXPLICIT_CLASSES
    assert "FACE_FEMALE" in NEVER_MASK_CLASSES
    assert label_bucket("FACE_FEMALE") == "never"
    assert label_bucket("BELLY_EXPOSED") == "suggestive"
    assert label_bucket("FEMALE_GENITALIA_EXPOSED") == "explicit"
    assert "FEMALE_GENITALIA_COVERED" in emit_classes_for_policy("balanced")
    assert "FEMALE_GENITALIA_COVERED" not in emit_classes_for_policy("relaxed")
    assert "FACE_FEMALE" not in emit_classes_for_policy("strict")


def test_nudenet_input_is_rgba_not_bgr() -> None:
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[:] = (10, 80, 200)
    out = nudenet_color_input(bgr)
    assert out.shape == (4, 4, 4)
    assert out[0, 0].tolist() == [200, 80, 10, 255]


def test_content_tiles_find_small_hot_region() -> None:
    image = np.full((1200, 1800, 3), 18, dtype=np.uint8)
    image[80:360, 1480:1760] = 200

    def box_fn(crop):
        height, width = crop.shape[:2]
        if max(height, width) <= 640 and float(crop.max()) > 150:
            return [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.88, "box": [10, 10, 80, 80]}]
        return []

    dets = detect_nsfw(image, box_fn=box_fn, min_score=0.50, tile=True)
    assert dets
    assert dets[0].xyxy[0] > 900
    tiles = propose_content_tiles(image)
    assert tiles


def test_last_stats_log_below_threshold() -> None:
    def box_fn(_img):
        return [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.21, "box": [20, 20, 80, 80]}]

    assert detect_nsfw(_frame(), box_fn=box_fn, policy_name="balanced") == []
    stats = last_detect_stats()
    assert stats.n_raw == 1
    assert stats.n_after_policy == 0
    assert "0.21" in stats.reason
    assert "balanced" in stats.reason
    assert "no intimate/explicit" in stats.reason


def test_covered_intimate_emits_on_balanced_not_relaxed() -> None:
    def box_fn(_img):
        return [
            {"class": "FEMALE_GENITALIA_COVERED", "score": 0.61, "box": [20, 20, 90, 90]},
            {"class": "FEMALE_BREAST_COVERED", "score": 0.58, "box": [120, 20, 80, 80]},
        ]

    balanced = detect_nsfw(_frame(), box_fn=box_fn, policy_name="balanced")
    strict = detect_nsfw(_frame(), box_fn=box_fn, policy_name="strict")
    relaxed = detect_nsfw(_frame(), box_fn=box_fn, policy_name="relaxed")
    assert len(balanced) >= 1
    assert len(strict) >= 1
    assert relaxed == []


def test_belly_exposed_emits_on_balanced() -> None:
    def box_fn(_img):
        return [{"class": "BELLY_EXPOSED", "score": 0.64, "box": [16, 16, 100, 90]}]

    dets = detect_nsfw(_frame(), box_fn=box_fn, policy_name="balanced")
    assert dets
    assert dets[0].label == "flagged"
    assert last_detect_stats().n_after_policy >= 1


def test_user_log_labels_emit_on_balanced() -> None:
    """Exact NudeNet top-3 from the live paste must emit boxes on balanced."""

    def box_fn(_img):
        return [
            {"class": "FACE_FEMALE", "score": 0.75, "box": [120, 20, 40, 40]},
            {"class": "BELLY_EXPOSED", "score": 0.64, "box": [40, 80, 90, 70]},
            {"class": "FEMALE_GENITALIA_COVERED", "score": 0.61, "box": [50, 150, 80, 60]},
        ]

    dets = detect_nsfw(_frame(), box_fn=box_fn, policy_name="balanced")
    stats = last_detect_stats()
    assert stats.n_after_policy >= 1
    assert len(dets) >= 1
    assert "mask=" in stats.reason
    assert "BELLY_EXPOSED:0.64" in stats.reason
    assert "FEMALE_GENITALIA_COVERED:0.61" in stats.reason
    assert "FACE_FEMALE:0.75" not in stats.reason


def test_face_only_reason_is_honest() -> None:
    def box_fn(_img):
        return [{"class": "FACE_FEMALE", "score": 0.75, "box": [20, 20, 80, 80]}]

    assert detect_nsfw(_frame(), box_fn=box_fn, policy_name="balanced") == []
    reason = last_detect_stats().reason
    assert "FACE_FEMALE ignored" in reason
    assert "no intimate/explicit above floor" in reason


def test_balanced_nsfw_boxes_spawn_overlay() -> None:
    from edge.coords import DisplayGeom
    from edge.screen_overlay import MemoryOverlayBackend, ScreenOverlay

    def box_fn(_img):
        return [
            {"class": "FACE_FEMALE", "score": 0.75, "box": [400, 40, 120, 120]},
            {"class": "BELLY_EXPOSED", "score": 0.64, "box": [200, 300, 220, 180]},
            {"class": "FEMALE_GENITALIA_COVERED", "score": 0.61, "box": [240, 500, 160, 140]},
        ]

    detect = make_nsfw_detect("balanced", box_fn=box_fn)
    reflex = Reflex(
        "balanced",
        brain_every=1,
        detect_mode="nsfw",
        detect_fn=detect,
        reinspect_fn=make_nsfw_reinspect(detect_fn=detect),
    )
    image = _frame(900, 1440)
    state = reflex.step(Frame(0, 0.0, image, 1440, 900))
    assert last_detect_stats().n_after_policy >= 1
    assert any(a.action in PROTECTIVE_ACTIONS for a in state.actions)
    backend = MemoryOverlayBackend()
    overlay = ScreenOverlay(backend=backend)
    overlay.sync(
        state,
        DisplayGeom(
            left_pt=0.0,
            top_pt=0.0,
            width_pt=1440.0,
            height_pt=900.0,
            width_px=1440,
            height_px=900,
            scale_x=1.0,
            scale_y=1.0,
            cocoa_origin_x=0.0,
            cocoa_origin_y=0.0,
            cocoa_height_pt=900.0,
        ),
    )
    assert backend.rects
    assert overlay.mask_count() >= 1
    overlay.close()


def _spread_intimate_boxes(n: int) -> list[dict]:
    """Far-apart thumbs so NMS keeps one cluster per box.

    Packed to stay inside a ≤960 work image so ``_downscale`` does not
    clip later boxes onto each other.
    """
    boxes = []
    cols = 6
    stride = 120.0
    for i in range(n):
        x = 20.0 + float(i % cols) * stride
        y = 20.0 + float(i // cols) * stride
        boxes.append({"class": "FEMALE_BREAST_EXPOSED", "score": 0.70, "box": [x, y, 50, 50]})
    return boxes


def _wide_frame(n: int) -> np.ndarray:
    cols = min(n, 6)
    rows = max(1, (n + 5) // 6)
    return _frame(40 + rows * 120, 40 + cols * 120)


@pytest.mark.parametrize("n", [1, 2, 8, 15])
def test_cluster_count_follows_input_box_groups(n: int) -> None:
    """Cluster count is unique figures after NMS — not a fake 5."""

    def box_fn(_img):
        return _spread_intimate_boxes(n)

    image = _wide_frame(n)
    dets = detect_nsfw(image, box_fn=box_fn, policy_name="balanced")
    stats = last_detect_stats()
    assert stats.n_clusters == n
    assert len(dets) == n
    assert stats.clusters_capped is False


def test_cluster_safety_cap_is_serp_grid_not_five() -> None:
    n = MAX_CLUSTERS + 6

    def box_fn(_img):
        return _spread_intimate_boxes(n)

    dets = detect_nsfw(_wide_frame(n), box_fn=box_fn, policy_name="balanced")
    stats = last_detect_stats()
    assert stats.n_clusters == MAX_CLUSTERS
    assert len(dets) == MAX_CLUSTERS
    assert stats.clusters_capped is True
    assert stats.cluster_cap == MAX_CLUSTERS


def test_far_apart_boxes_are_two_clusters() -> None:
    def box_fn(_img):
        return [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.70, "box": [20, 20, 60, 60]},
            {"class": "BUTTOCKS_EXPOSED", "score": 0.66, "box": [520, 380, 60, 60]},
        ]

    dets = detect_nsfw(_frame(480, 640), box_fn=box_fn, policy_name="balanced")
    stats = last_detect_stats()
    assert stats.n_after_policy == 2
    assert stats.n_clusters == 2
    assert len(dets) == 2


def test_overlapping_breast_belly_is_one_cluster() -> None:
    def box_fn(_img):
        return [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.72, "box": [40, 40, 80, 70]},
            {"class": "BELLY_EXPOSED", "score": 0.64, "box": [50, 90, 70, 70]},
        ]

    dets = detect_nsfw(_frame(300, 320), box_fn=box_fn, policy_name="balanced")
    stats = last_detect_stats()
    assert stats.n_after_policy == 2
    assert stats.n_clusters == 1
    assert len(dets) == 1
    x1, y1, x2, y2 = dets[0].xyxy
    assert x1 <= 40 and y1 <= 40 and x2 >= 120 and y2 >= 160


def test_hysteresis_keeps_039_after_065() -> None:
    hold = ClusterHold(on_score=0.30, hold_score=0.20)

    def high(_img):
        return [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.65, "box": [30, 30, 90, 90]}]

    def low(_img):
        return [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.39, "box": [32, 32, 88, 88]}]

    first = detect_nsfw(_frame(), box_fn=high, policy_name="balanced", hold=hold)
    assert first and first[0].score == pytest.approx(0.65)
    second = detect_nsfw(_frame(), box_fn=low, policy_name="balanced", hold=hold)
    assert second
    assert second[0].score == pytest.approx(0.39)


def test_buttocks_044_and_exposed_039_emit_on_balanced() -> None:
    def box_fn(_img):
        return [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.39, "box": [20, 20, 80, 80]},
            {"class": "BUTTOCKS_EXPOSED", "score": 0.44, "box": [200, 20, 80, 80]},
        ]

    dets = detect_nsfw(_frame(240, 360), box_fn=box_fn, policy_name="balanced")
    assert last_detect_stats().n_clusters == 2
    assert len(dets) == 2


def test_scene_change_releases_nsfw_hold() -> None:
    dark = np.full((240, 320, 3), 72, dtype=np.uint8)
    bright = np.full((240, 320, 3), 92, dtype=np.uint8)
    assert live_scene_changed(dark, bright) is True

    def box_fn(img):
        if float(img.mean()) < 80:
            return [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.80, "box": [30, 30, 80, 80]}]
        return [{"class": "FACE_FEMALE", "score": 0.90, "box": [30, 30, 40, 40]}]

    detect = make_nsfw_detect("balanced", box_fn=box_fn)
    reflex = Reflex(
        "balanced",
        brain_every=99,
        brain_every_ms=10_000.0,
        brain_budget_ms=10_000.0,
        change_gate=9.0,
        detect_mode="nsfw",
        detect_fn=detect,
        reinspect_fn=make_nsfw_reinspect(detect_fn=detect),
    )
    first = reflex.step(Frame(0, 0.0, dark, 320, 240))
    assert any(a.action in PROTECTIVE_ACTIONS for a in first.actions)
    second = reflex.step(Frame(1, 40.0, bright, 320, 240))
    assert not any(a.action in PROTECTIVE_ACTIONS for a in second.actions)


def test_face_never_joins_a_cluster() -> None:
    def box_fn(_img):
        return [
            {"class": "FACE_FEMALE", "score": 0.99, "box": [40, 40, 80, 80]},
            {"class": "FACE_MALE", "score": 0.97, "box": [200, 40, 80, 80]},
        ]

    assert detect_nsfw(_frame(), box_fn=box_fn, policy_name="balanced") == []
    assert last_detect_stats().n_clusters == 0


def test_two_thumbs_keep_two_overlay_windows() -> None:
    from edge.coords import DisplayGeom
    from edge.screen_overlay import MemoryOverlayBackend, ScreenOverlay

    def box_fn(_img):
        return [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.70, "box": [40, 40, 80, 80]},
            {"class": "BUTTOCKS_EXPOSED", "score": 0.66, "box": [900, 500, 80, 80]},
        ]

    detect = make_nsfw_detect("balanced", box_fn=box_fn)
    reflex = Reflex(
        "balanced",
        brain_every=1,
        detect_mode="nsfw",
        detect_fn=detect,
        reinspect_fn=make_nsfw_reinspect(detect_fn=detect),
    )
    image = _frame(720, 1200)
    state = reflex.step(Frame(0, 0.0, image, 1200, 720))
    assert last_detect_stats().n_clusters == 2
    backend = MemoryOverlayBackend()
    overlay = ScreenOverlay(backend=backend)
    geom = DisplayGeom(
        left_pt=0.0,
        top_pt=0.0,
        width_pt=1200.0,
        height_pt=720.0,
        width_px=1200,
        height_px=720,
        scale_x=1.0,
        scale_y=1.0,
        cocoa_origin_x=0.0,
        cocoa_origin_y=0.0,
        cocoa_height_pt=720.0,
    )
    overlay.sync(state, geom)
    assert overlay.mask_count() == 2
    overlay.sync(state, geom)
    assert overlay.mask_count() == 2
    overlay.close()


@pytest.mark.parametrize("n", [1, 2, 8, 15])
def test_overlay_window_count_follows_clusters(n: int) -> None:
    from edge.coords import DisplayGeom
    from edge.screen_overlay import MemoryOverlayBackend, ScreenOverlay

    def box_fn(_img):
        return _spread_intimate_boxes(n)

    image = _wide_frame(n)
    detect = make_nsfw_detect("balanced", box_fn=box_fn)
    reflex = Reflex(
        "balanced",
        brain_every=1,
        detect_mode="nsfw",
        detect_fn=detect,
        reinspect_fn=make_nsfw_reinspect(detect_fn=detect),
    )
    height, width = image.shape[:2]
    state = reflex.step(Frame(0, 0.0, image, width, height))
    assert last_detect_stats().n_clusters == n
    backend = MemoryOverlayBackend()
    overlay = ScreenOverlay(backend=backend)
    geom = DisplayGeom(
        left_pt=0.0,
        top_pt=0.0,
        width_pt=float(width),
        height_pt=float(height),
        width_px=width,
        height_px=height,
        scale_x=1.0,
        scale_y=1.0,
        cocoa_origin_x=0.0,
        cocoa_origin_y=0.0,
        cocoa_height_pt=float(height),
    )
    overlay.sync(state, geom)
    assert overlay.mask_count() == n
    overlay.close()


def test_overlay_shrinks_when_cluster_count_drops() -> None:
    from edge.coords import DisplayGeom
    from edge.screen_overlay import MemoryOverlayBackend, ScreenOverlay

    calls = {"n": 0}

    def box_fn(_img):
        calls["n"] += 1
        return _spread_intimate_boxes(8 if calls["n"] == 1 else 2)

    detect = make_nsfw_detect("balanced", box_fn=box_fn)
    reflex = Reflex(
        "balanced",
        brain_every=1,
        detect_mode="nsfw",
        detect_fn=detect,
        reinspect_fn=make_nsfw_reinspect(detect_fn=detect),
    )
    image = _wide_frame(8)
    height, width = image.shape[:2]
    backend = MemoryOverlayBackend()
    overlay = ScreenOverlay(backend=backend)
    geom = DisplayGeom(
        left_pt=0.0,
        top_pt=0.0,
        width_pt=float(width),
        height_pt=float(height),
        width_px=width,
        height_px=height,
        scale_x=1.0,
        scale_y=1.0,
        cocoa_origin_x=0.0,
        cocoa_origin_y=0.0,
        cocoa_height_pt=float(height),
    )
    first = reflex.step(Frame(0, 0.0, image, width, height))
    overlay.sync(first, geom)
    assert overlay.mask_count() == 8
    second = reflex.step(Frame(1, 40.0, image, width, height))
    overlay.sync(second, geom)
    assert last_detect_stats().n_clusters == 2
    assert overlay.mask_count() == 2
    overlay.close()


def _empty_cloud_result():
    from agent.orchestrator import FrameResult

    return FrameResult(
        actions=[],
        traces=[],
        detections=[],
        tracks=[],
        change_score=0.0,
        cut=False,
        reinspect_count=0,
        timed_out=False,
        timings_ms={},
        tool_sequence="LIVE_TILE",
        companion_line="ok",
        first_decision="NO_ACTION",
        final_decision="NO_ACTION",
        band="low",
    )


def test_clusters_stay_frosted_across_30_frames_when_detect_goes_zero() -> None:
    """after=3 then after=0 (covered by frost) must keep multiple frost windows."""
    from edge.coords import DisplayGeom
    from edge.screen_overlay import MemoryOverlayBackend, ScreenOverlay

    calls = {"n": 0}

    def box_fn(_img):
        calls["n"] += 1
        if calls["n"] <= 3:
            return [
                {"class": "FEMALE_BREAST_EXPOSED", "score": 0.80, "box": [40, 40, 80, 80]},
                {"class": "BUTTOCKS_EXPOSED", "score": 0.72, "box": [900, 40, 80, 80]},
                {"class": "BELLY_EXPOSED", "score": 0.66, "box": [40, 520, 80, 80]},
            ]
        return []

    detect = make_nsfw_detect("balanced", box_fn=box_fn)
    reflex = Reflex(
        "balanced",
        brain_every=1,
        detect_mode="nsfw",
        detect_fn=detect,
        reinspect_fn=make_nsfw_reinspect(detect_fn=detect),
    )
    image = _frame(720, 1200)
    backend = MemoryOverlayBackend()
    overlay = ScreenOverlay(backend=backend)
    geom = DisplayGeom(
        left_pt=0.0,
        top_pt=0.0,
        width_pt=1200.0,
        height_pt=720.0,
        width_px=1200,
        height_px=720,
        scale_x=1.0,
        scale_y=1.0,
        cocoa_origin_x=0.0,
        cocoa_origin_y=0.0,
        cocoa_height_pt=720.0,
    )
    counts: list[int] = []
    for i in range(35):
        state = reflex.step(Frame(i, float(i) * 40.0, image, 1200, 720))
        overlay.sync(state, geom)
        counts.append(overlay.mask_count())
        assert any(a.action in PROTECTIVE_ACTIONS for a in state.actions)
        assert overlay.mask_count() >= 2
    assert min(counts) >= 2
    assert counts[-1] >= 2
    assert last_detect_stats().n_after_policy == 0
    assert last_detect_stats().n_clusters >= 2
    overlay.close()


def test_cloud_no_action_keeps_local_clusters() -> None:
    class CloudNo:
        def live(self, jpeg_bytes, **_kwargs):
            return _empty_cloud_result()

    def box_fn(_img):
        return [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.80, "box": [40, 40, 80, 80]},
            {"class": "BUTTOCKS_EXPOSED", "score": 0.70, "box": [400, 40, 80, 80]},
        ]

    detect = make_nsfw_detect("balanced", box_fn=box_fn)
    reflex = Reflex(
        "balanced",
        backend="http://192.0.2.10:8080",
        detect_mode="nsfw",
        detect_fn=detect,
        http_client=CloudNo(),
        brain_every=1,
    )
    image = _frame(240, 640)
    for i in range(30):
        state = reflex.step(Frame(i, float(i) * 40.0, image, 640, 240))
        assert state.backend == "cloud"
        assert any(a.action in PROTECTIVE_ACTIONS for a in state.actions)
        assert reflex.last_result is not None
        assert reflex.last_result.final_decision in {"BLUR", "HOLD", "OPAQUE_MASK", "PIXELATE"}


def test_cloud_fail_keeps_last_boxes() -> None:
    from edge.http_brain import PerceiveError

    class Flaky:
        n = 0

        def live(self, jpeg_bytes, **_kwargs):
            self.n += 1
            if self.n == 1:
                return _empty_cloud_result()
            raise PerceiveError("timeout")

    def box_fn(_img):
        return [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.80, "box": [40, 40, 90, 90]}]

    detect = make_nsfw_detect("balanced", box_fn=box_fn)
    reflex = Reflex(
        "balanced",
        backend="http://192.0.2.10:8080",
        detect_mode="nsfw",
        detect_fn=detect,
        http_client=Flaky(),
        brain_every=1,
    )
    image = _frame()
    first = reflex.step(Frame(0, 0.0, image, 320, 240))
    assert any(a.action in PROTECTIVE_ACTIONS for a in first.actions)
    second = reflex.step(Frame(1, 40.0, image, 320, 240))
    assert any(a.action in PROTECTIVE_ACTIONS for a in second.actions)
    assert second.backend == "local"
    assert second.backend_error


def test_overlay_paint_is_not_a_scene_change() -> None:
    prev = np.full((240, 320, 3), 72, dtype=np.uint8)
    cur = prev.copy()
    cur[30:120, 40:160] = 220
    frost = [(40.0, 30.0, 160.0, 120.0)]
    assert live_scene_changed(prev, cur, ignore_boxes=frost) is False
    assert live_scene_changed(
        np.full((240, 320, 3), 72, dtype=np.uint8),
        np.full((240, 320, 3), 92, dtype=np.uint8),
    ) is True
