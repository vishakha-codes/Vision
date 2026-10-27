from vision_core.contracts import Detection
from vision_core.geometry import hysteresis_mask_on
from vision_core.policy import evaluate, load_policy


def test_mask_stays_on_inside_deadband() -> None:
    on_th, off_th = 0.55, 0.50
    assert hysteresis_mask_on(False, 0.60, on_th, off_th) is True
    assert hysteresis_mask_on(True, 0.52, on_th, off_th) is True
    assert hysteresis_mask_on(True, 0.49, on_th, off_th) is False
    assert hysteresis_mask_on(False, 0.52, on_th, off_th) is False


def test_balanced_hysteresis_does_not_flicker_in_deadband() -> None:
    policy = load_policy("policies/balanced.json")
    scores = [0.70, 0.52, 0.58, 0.51, 0.60]
    mask_on = False
    transitions = 0
    for score in scores:
        det = Detection(
            label="flagged",
            score=score,
            xyxy=(40.0, 40.0, 140.0, 140.0),
            detector_id="test",
            frame_id=0,
        )
        # Carry mask_on across evaluate by feeding a tracked-like detection
        # through evaluate, then using the returned region's flag.
        from vision_core.contracts import TrackedRegion, TrackMode

        region = TrackedRegion(
            track_id=1,
            xyxy=det.xyxy,
            label="flagged",
            stable_score=score,
            uncertainty=1.0 - score,
            last_seen_ms=0.0,
            mode=TrackMode.TRACKED,
            mask_on=mask_on,
        )
        action = evaluate([region], policy)[0]
        nxt = bool(action.region.mask_on) if action.region is not None else False
        if nxt != mask_on:
            transitions += 1
        mask_on = nxt
    assert mask_on is True
    assert transitions == 1


def test_drop_below_off_threshold_releases_mask() -> None:
    from vision_core.contracts import TrackedRegion, TrackMode

    policy = load_policy("policies/balanced.json")
    high = TrackedRegion(
        track_id=1,
        xyxy=(40.0, 40.0, 140.0, 140.0),
        label="flagged",
        stable_score=0.80,
        uncertainty=0.2,
        last_seen_ms=0.0,
        mode=TrackMode.TRACKED,
        mask_on=False,
    )
    on_action = evaluate([high], policy)[0]
    assert on_action.region is not None and on_action.region.mask_on is True
    low = TrackedRegion(
        track_id=1,
        xyxy=high.xyxy,
        label="flagged",
        stable_score=0.20,
        uncertainty=0.8,
        last_seen_ms=0.0,
        mode=TrackMode.TRACKED,
        mask_on=True,
    )
    off_action = evaluate([low], policy)[0]
    assert off_action.region is not None and off_action.region.mask_on is False
