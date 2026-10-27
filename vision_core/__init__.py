"""Shared vision primitives used by both laptop edge and AWS tools.

Public tool names are stable and become Bedrock tools in Phase 2:
scene_change, extract_tiles, detect_regions, reinspect_crop,
track_and_predict, scene_cut, render_mask_spec, evaluate_fixture.
"""

from vision_core.detect_regions import detect_regions
from vision_core.extract_tiles import extract_tiles
from vision_core.policy import evaluate
from vision_core.reinspect_crop import reinspect_crop
from vision_core.render_mask_spec import render_mask_spec
from vision_core.scene_change import scene_change
from vision_core.scene_cut import scene_cut
from vision_core.track_and_predict import track_and_predict

__all__ = [
    "scene_change",
    "extract_tiles",
    "detect_regions",
    "reinspect_crop",
    "track_and_predict",
    "scene_cut",
    "render_mask_spec",
    "evaluate_fixture",
    "evaluate",
]


def __getattr__(name: str):
    if name == "evaluate_fixture":
        from vision_core.eval import evaluate_fixture as _evaluate_fixture

        return _evaluate_fixture
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
