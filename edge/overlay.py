"""OpenCV HUD: ``render_mask_spec`` plus a policy/mood bar.

This window is an optional debug preview (``--hud``). Live protection is
the macOS overlay in ``edge.screen_overlay`` — not this imshow copy.
Replay still uses this HUD (or a preview mp4 when headless). Window
titles stay local — they are never copied into traces.
"""

from __future__ import annotations

import cv2
import numpy as np

from edge.companion import draw_companion
from edge.reflex import ReflexState, STATUS_NOT_PROTECTING
from vision_core.render_mask_spec import render_mask_spec

WINDOW_NAME = "Screen Guardian"


def window_title(state: ReflexState) -> str:
    """Local window title only. Do not log or send this string."""
    extra = f" | backend={state.backend} | detect={state.detect_mode}"
    err = f" | {state.backend_error}" if state.backend_error else ""
    return f"Screen Guardian | {state.policy_name} | {state.mood} | {state.status}{extra}{err}"


def _hud_color(state: ReflexState) -> tuple[int, int, int]:
    if state.status == STATUS_NOT_PROTECTING:
        return (40, 40, 200)
    if state.mood == "covering":
        return (70, 150, 40)
    if state.mood == "unsure":
        return (20, 170, 210)
    return (160, 160, 160)


def render_overlay(state: ReflexState) -> np.ndarray:
    """Compose masks, HUD, and the companion island on a BGR copy."""
    if state.frame is None:
        canvas = np.full((360, 640, 3), 16, dtype=np.uint8)
    else:
        canvas, _specs = render_mask_spec(state.actions, state.frame)

    height, width = canvas.shape[:2]
    bar_h = 28
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_h), (16, 16, 20), thickness=-1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

    err = f"  |  {state.backend_error}" if state.backend_error else ""
    hud = (
        f"{state.policy_name}  |  {state.mood}  |  {state.status}"
        f"  |  detect={state.detect_mode}  |  backend={state.backend}{err}"
    )
    cv2.putText(
        canvas,
        hud[:120],
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        _hud_color(state),
        1,
        cv2.LINE_AA,
    )
    draw_companion(canvas, state.mood, state.companion_line)
    return canvas


def try_open_window(name: str = WINDOW_NAME) -> bool:
    """Return True if OpenCV highgui can show a window on this machine."""
    try:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        probe = np.zeros((16, 16, 3), dtype=np.uint8)
        cv2.imshow(name, probe)
        cv2.waitKey(1)
        return True
    except cv2.error:
        return False


def show(state: ReflexState, canvas: np.ndarray, *, delay_ms: int = 1) -> int:
    """imshow + waitKey. Returns the key code, or -1 if the window failed."""
    try:
        cv2.imshow(WINDOW_NAME, canvas)
        try:
            cv2.setWindowTitle(WINDOW_NAME, window_title(state))
        except cv2.error:
            pass
        return int(cv2.waitKey(delay_ms) & 0xFF)
    except cv2.error:
        return -1


def close_window() -> None:
    try:
        cv2.destroyWindow(WINDOW_NAME)
        cv2.waitKey(1)
    except cv2.error:
        pass
