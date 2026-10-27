"""render_mask_spec — apply opaque / blur / pixelate for eval annotation only."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from vision_core.contracts import ActionType, Frame, MaskSpec, ProtectionAction
from vision_core.geometry import as_int_box


def _bgr(image: Frame | np.ndarray) -> np.ndarray:
    if isinstance(image, Frame):
        return image.image_bgr
    return image


def _odd_kernel(size: int, minimum: int = 5) -> int:
    k = max(minimum, size // 3)
    if k % 2 == 0:
        k += 1
    return min(k, 51)


def _apply_region(out: np.ndarray, xyxy: tuple[int, int, int, int], action: ActionType) -> None:
    x1, y1, x2, y2 = xyxy
    roi = out[y1:y2, x1:x2]
    if roi.size == 0:
        return
    if action in (ActionType.OPAQUE_MASK, ActionType.FULL_WINDOW_MASK, ActionType.HOLD):
        roi[:] = (12, 12, 12)
        return
    if action == ActionType.BLUR:
        k = _odd_kernel(max(x2 - x1, y2 - y1))
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        return
    if action == ActionType.PIXELATE:
        rw, rh = max(1, x2 - x1), max(1, y2 - y1)
        small = cv2.resize(roi, (max(1, rw // 12), max(1, rh // 12)), interpolation=cv2.INTER_LINEAR)
        out[y1:y2, x1:x2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)


def render_mask_spec(
    actions: Sequence[ProtectionAction],
    frame: Frame | np.ndarray,
) -> tuple[np.ndarray, list[MaskSpec]]:
    """Return an annotated BGR copy plus the list of applied mask specs."""
    image = _bgr(frame)
    out = image.copy()
    height, width = out.shape[:2]
    specs: list[MaskSpec] = []

    for item in actions:
        region = item.region
        if item.action == ActionType.FULL_WINDOW_MASK:
            box = (0, 0, width, height)
        elif region is None:
            continue
        else:
            box = as_int_box(region.xyxy, width, height)

        if item.action not in (
            ActionType.NO_ACTION,
            ActionType.RELEASE,
            ActionType.REVIEW,
        ):
            _apply_region(out, box, item.action)

        color = (40, 40, 220) if item.action not in (ActionType.NO_ACTION, ActionType.RELEASE) else (80, 180, 80)
        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), color, 2)
        label = f"{item.action.value} {item.reason_code}"
        if region is not None:
            label = f"t{region.track_id} {region.label} {region.stable_score:.2f} {item.action.value}"
        cv2.putText(
            out,
            label[:64],
            (box[0], max(16, box[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        specs.append(
            MaskSpec(
                action=item.action.value,
                xyxy=box,
                reason_code=item.reason_code,
                track_id=None if region is None else region.track_id,
            )
        )
    return out, specs
