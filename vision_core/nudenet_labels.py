"""NudeNet 320n label strings and class groups — the one source of truth.

Used by the live path (``vision_core.live.detector`` / ``modes``) and the
legacy / cloud path (``vision_core.detect_nsfw``, which re-exports these
names). ``NUDENET_LABELS`` is the model's output column order and must match
``nudenet/nudenet.py`` ``__labels`` in the installed wheel exactly
(``tests/test_live_detector.py`` checks it).
"""

from __future__ import annotations

# Output column order of NudeNet 3.4 ``nudenet.py`` (``__labels``).
NUDENET_LABELS: tuple[str, ...] = (
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
)

# Never a mask, even at 0.99 (news faces, feet, armpits, clothed bellies).
NEVER_MASK_CLASSES = frozenset(
    {
        "FACE_FEMALE",
        "FACE_MALE",
        "FEET_EXPOSED",
        "FEET_COVERED",
        "ARMPITS_COVERED",
        "ARMPITS_EXPOSED",
        "BELLY_COVERED",
    }
)

EXPLICIT_CLASSES = frozenset(
    {
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "BUTTOCKS_EXPOSED",
    }
)

# Clothed intimate areas (lingerie, swimwear, see-through). Live medium counts
# these at a stricter threshold; legacy balanced / strict mask them too.
COVERED_INTIMATE_CLASSES = frozenset(
    {
        "FEMALE_GENITALIA_COVERED",
        "FEMALE_BREAST_COVERED",
        "BUTTOCKS_COVERED",
        "ANUS_COVERED",
    }
)

# Masked on legacy balanced + strict (unioned with explicit). Relaxed drops
# these unless they are also in EXPLICIT_CLASSES.
SUGGESTIVE_CLASSES = COVERED_INTIMATE_CLASSES | frozenset(
    {
        "FEMALE_BREAST_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "BELLY_EXPOSED",
        "BUTTOCKS_EXPOSED",
        "ANUS_EXPOSED",
    }
)

INTIMATE_CLASSES = SUGGESTIVE_CLASSES | EXPLICIT_CLASSES


def is_never_mask(label: str) -> bool:
    name = str(label or "")
    if name in NEVER_MASK_CLASSES:
        return True
    return name.startswith("FACE_") or name.startswith("FEET_") or name.startswith("ARMPITS_")


def label_bucket(label: str) -> str:
    """``never`` / ``explicit`` / ``suggestive`` / ``other``."""
    name = str(label or "")
    if is_never_mask(name):
        return "never"
    if name in EXPLICIT_CLASSES:
        return "explicit"
    if name in SUGGESTIVE_CLASSES:
        return "suggestive"
    return "other"


# Columns worth scoring beyond each anchor's top class (see
# ``vision_core.live.detector.decode_yolov8``): every class any mode can count.
MASKABLE_LABELS: tuple[str, ...] = tuple(lab for lab in NUDENET_LABELS if lab in INTIMATE_CLASSES)
