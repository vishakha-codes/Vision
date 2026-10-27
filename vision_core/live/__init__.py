"""Rebuilt live path: picture regions → verdicts → masks. See ``contracts``."""

from vision_core.live.contracts import (
    AnalyzeResult,
    Box,
    LiveAnalyzer,
    MaskSpec,
    MaskStyle,
    PictureRegion,
    ProtectionMode,
    Verdict,
    VerdictState,
)

__all__ = [
    "make_analyzer",
    "AnalyzeResult",
    "Box",
    "LiveAnalyzer",
    "MaskSpec",
    "MaskStyle",
    "PictureRegion",
    "ProtectionMode",
    "Verdict",
    "VerdictState",
]


def __getattr__(name: str):
    # Lazy so importing the contracts never loads cv2.dnn / the model.
    if name == "make_analyzer":
        from vision_core.live.analyzer import make_analyzer

        return make_analyzer
    raise AttributeError(name)
