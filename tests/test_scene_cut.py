import numpy as np

from vision_core.contracts import TrackedRegion, TrackMode
from vision_core.scene_cut import scene_cut
from vision_core.track_and_predict import expire_tracks


def test_similar_frames_are_not_a_cut() -> None:
    a = np.full((80, 120, 3), 40, dtype=np.uint8)
    b = a.copy()
    b[10:20, 10:20] = 50
    assert scene_cut(a, b, change_score=0.02) is False


def test_opposite_frames_are_a_cut() -> None:
    black = np.zeros((80, 120, 3), dtype=np.uint8)
    white = np.full((80, 120, 3), 255, dtype=np.uint8)
    assert scene_cut(black, white, change_score=0.9) is True


def test_scene_cut_expires_tracks() -> None:
    track = TrackedRegion(
        track_id=3,
        xyxy=(4.0, 4.0, 20.0, 20.0),
        label="flagged",
        stable_score=0.8,
        uncertainty=0.2,
        last_seen_ms=100.0,
        mode=TrackMode.TRACKED,
    )
    black = np.zeros((80, 120, 3), dtype=np.uint8)
    white = np.full((80, 120, 3), 255, dtype=np.uint8)
    assert scene_cut(black, white) is True
    expired = expire_tracks([track], last_seen_ms=200.0)
    assert len(expired) == 1
    assert expired[0].mode == TrackMode.EXPIRED
    assert expired[0].track_id == 3
