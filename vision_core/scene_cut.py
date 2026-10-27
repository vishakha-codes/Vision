"""scene_cut — histogram mismatch or change-score spike; kill stale tracks."""

from __future__ import annotations

import cv2
import numpy as np

from vision_core.contracts import Frame

_HS_BINS = (24, 24)
_HS_RANGES = [0, 180, 0, 256]
_V_BINS = [32]
_V_RANGES = [0, 256]


def _bgr(image: Frame | np.ndarray) -> np.ndarray:
    if isinstance(image, Frame):
        return image.image_bgr
    return image


def _norm_hist(image_bgr: np.ndarray, channels: list[int], bins: list[int] | tuple[int, ...], ranges: list[int]) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], channels, None, bins, ranges)
    cv2.normalize(hist, hist)
    return hist


def scene_cut(
    prev: Frame | np.ndarray,
    cur: Frame | np.ndarray,
    *,
    change_score: float | None = None,
    hist_corr_min: float = 0.52,
    change_spike: float = 0.38,
    luma_delta_min: float = 0.28,
) -> bool:
    """True when the scene is no longer the same shot.

    ``True`` means the caller should expire / drop stale tracks.
    Hue-only histograms miss black↔white cuts, so value/luma is included.
    """
    prev_bgr = _bgr(prev)
    cur_bgr = _bgr(cur)
    hs_corr = float(
        cv2.compareHist(
            _norm_hist(prev_bgr, [0, 1], _HS_BINS, _HS_RANGES),
            _norm_hist(cur_bgr, [0, 1], _HS_BINS, _HS_RANGES),
            cv2.HISTCMP_CORREL,
        )
    )
    v_corr = float(
        cv2.compareHist(
            _norm_hist(prev_bgr, [2], _V_BINS, _V_RANGES),
            _norm_hist(cur_bgr, [2], _V_BINS, _V_RANGES),
            cv2.HISTCMP_CORREL,
        )
    )
    prev_luma = float(np.mean(cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)))
    cur_luma = float(np.mean(cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)))
    luma_delta = abs(prev_luma - cur_luma) / 255.0
    hist_cut = hs_corr < hist_corr_min or v_corr < hist_corr_min or luma_delta >= luma_delta_min
    spike = change_score is not None and change_score >= change_spike
    return bool(hist_cut or spike)
