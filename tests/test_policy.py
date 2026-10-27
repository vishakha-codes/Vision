from vision_core.contracts import ActionType, Detection
from vision_core.policy import evaluate, load_policy


def _mid_score_flagged() -> Detection:
    return Detection(
        label="flagged",
        score=0.55,
        xyxy=(40.0, 40.0, 140.0, 140.0),
        detector_id="test",
        frame_id=0,
    )


def test_strict_and_relaxed_differ_on_same_detection() -> None:
    det = _mid_score_flagged()
    strict = evaluate([det], load_policy("policies/strict.json"))[0]
    relaxed = evaluate([det], load_policy("policies/relaxed.json"))[0]
    balanced = evaluate([det], load_policy("policies/balanced.json"))[0]
    assert strict.action == ActionType.OPAQUE_MASK
    assert relaxed.action == ActionType.NO_ACTION
    assert balanced.action != relaxed.action
    assert strict.action != relaxed.action


def test_safe_class_never_masks() -> None:
    det = Detection(
        label="safe",
        score=0.99,
        xyxy=(40.0, 40.0, 140.0, 140.0),
        detector_id="test",
        frame_id=0,
    )
    for name in ("strict", "balanced", "relaxed"):
        action = evaluate([det], f"policies/{name}.json")[0]
        assert action.action == ActionType.NO_ACTION
        assert action.reason_code == "SAFE_CLASS"
