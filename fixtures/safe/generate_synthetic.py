"""Generate a safe synthetic moving-regions clip and per-frame ground truth.

Colored shapes only. No people, no downloaded media, no NSFW content.
Uses the isolated vision/.venv OpenCV 5 install.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

WIDTH = 640
HEIGHT = 360
FPS = 30
DURATION_SEC = 25
FRAME_COUNT = FPS * DURATION_SEC
OUT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = OUT_DIR / "synthetic_moving_regions.mp4"
GT_PATH = OUT_DIR / "synthetic_moving_regions.gt.json"


def _clamp_box(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(WIDTH - 1, x))
    y1 = max(0, min(HEIGHT - 1, y))
    x2 = max(x1 + 1, min(WIDTH, x + w))
    y2 = max(y1 + 1, min(HEIGHT, y + h))
    return x1, y1, x2, y2


def _region_state(frame_idx: int) -> list[dict]:
    t = frame_idx / FPS

    # r1: red rectangle, left-to-right bounce (safe)
    r1_w, r1_h = 80, 50
    r1_x = int(40 + (WIDTH - 160) * (0.5 + 0.5 * math.sin(2 * math.pi * t / 8.0)))
    r1_y = 40 + int(20 * math.sin(2 * math.pi * t / 3.0))
    r1 = {
        "id": "r1",
        "class": "safe",
        "color_bgr": (40, 40, 220),
        "shape": "rect",
        "xyxy": _clamp_box(r1_x, r1_y, r1_w, r1_h),
    }

    # r2: magenta circle, diagonal orbit — flagged class for later policy tests
    r2_w, r2_h = 70, 70
    r2_x = int(WIDTH / 2 - 35 + 180 * math.cos(2 * math.pi * t / 10.0))
    r2_y = int(HEIGHT / 2 - 35 + 90 * math.sin(2 * math.pi * t / 10.0))
    r2 = {
        "id": "r2",
        "class": "flagged",
        "color_bgr": (200, 40, 200),
        "shape": "circle",
        "xyxy": _clamp_box(r2_x, r2_y, r2_w, r2_h),
    }

    # r3: cyan triangle-ish rect, vertical bounce (safe)
    r3_w, r3_h = 60, 90
    r3_x = WIDTH - 140 + int(25 * math.sin(2 * math.pi * t / 5.0))
    r3_y = int(30 + (HEIGHT - 150) * (0.5 + 0.5 * math.sin(2 * math.pi * t / 6.0)))
    r3 = {
        "id": "r3",
        "class": "safe",
        "color_bgr": (220, 200, 40),
        "shape": "rect",
        "xyxy": _clamp_box(r3_x, r3_y, r3_w, r3_h),
    }
    return [r1, r2, r3]


def _draw(frame: np.ndarray, region: dict) -> None:
    x1, y1, x2, y2 = region["xyxy"]
    color = region["color_bgr"]
    if region["shape"] == "circle":
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        radius = max(8, min(x2 - x1, y2 - y1) // 2)
        cv2.circle(frame, (cx, cy), radius, color, thickness=-1)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)
    cv2.putText(
        frame,
        region["id"],
        (x1, max(16, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    if not str(cv2.__version__).startswith("5."):
        raise SystemExit(f"Expected OpenCV 5.x in the venv, got {cv2.__version__}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(VIDEO_PATH), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open VideoWriter for {VIDEO_PATH}")

    frames_gt: list[dict] = []
    for i in range(FRAME_COUNT):
        frame = np.full((HEIGHT, WIDTH, 3), 28, dtype=np.uint8)
        cv2.rectangle(frame, (16, 16), (WIDTH - 16, HEIGHT - 16), (48, 48, 48), 1)
        regions = _region_state(i)
        for region in regions:
            _draw(frame, region)
        writer.write(frame)
        frames_gt.append(
            {
                "frame": i,
                "boxes": [
                    {
                        "id": r["id"],
                        "class": r["class"],
                        "xyxy": list(r["xyxy"]),
                    }
                    for r in regions
                ],
            }
        )
    writer.release()

    gt = {
        "video": VIDEO_PATH.name,
        "generator": "fixtures/safe/generate_synthetic.py",
        "notes": "Synthetic colored shapes only. r2 is the flagged class for policy tests.",
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "duration_sec": DURATION_SEC,
        "frame_count": FRAME_COUNT,
        "regions": {
            "r1": {"class": "safe", "label": "red rectangle"},
            "r2": {"class": "flagged", "label": "magenta circle"},
            "r3": {"class": "safe", "label": "cyan rectangle"},
        },
        "frames": frames_gt,
    }
    GT_PATH.write_text(json.dumps(gt, indent=2) + "\n")
    print(f"wrote {VIDEO_PATH}")
    print(f"wrote {GT_PATH}")


if __name__ == "__main__":
    main()
