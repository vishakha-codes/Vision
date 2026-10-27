"""Show Gomi alone and cycle island states.

    PYTHONPATH=. .venv/bin/python -m edge.gomi
"""

from __future__ import annotations

import signal
import sys
import time

from edge.gomi.island import open_island
from edge.gomi.states import GuardianEvent


def _cycle_event(step: int) -> GuardianEvent:
    slot = step % 5
    if slot == 0:
        return GuardianEvent(status="LOCAL_HOLD", mask_count=0, companion_line="Watching. All clear.")
    if slot == 1:
        return GuardianEvent(
            status="CLOUD_LOOK",
            mask_count=2,
            last_action="OPAQUE_MASK",
            mood="covering",
            companion_line="Covering that area so the screen stays appropriate.",
        )
    if slot == 2:
        return GuardianEvent(
            status="CLOUD_LOOK",
            mask_count=1,
            last_action="REVIEW",
            mood="unsure",
            band="mid",
            working=True,
            companion_line="Taking a closer look.",
        )
    if slot == 3:
        return GuardianEvent(status="LOCAL_HOLD", mask_count=0, companion_line="Watching. All clear.")
    return GuardianEvent(status="NOT_PROTECTING", paused=True, protecting=False, companion_line="Not covering this frame.")


def main() -> int:
    runtime = open_island(headless=False)
    print(
        "Gomi island is up at the top of the screen. "
        "Click the pill to expand: character on the left, skins and settings on the right. "
        "Click empty space or press Esc to collapse. Drag to move. Ctrl-C quits.",
        flush=True,
    )
    stop = {"yes": False}

    def _handle(*_args: object) -> None:
        stop["yes"] = True
        try:
            from AppKit import NSApp

            NSApp.terminate_(None)
        except Exception:
            pass

    signal.signal(signal.SIGINT, _handle)
    step = 0
    last = time.monotonic()
    runtime.on_state(_cycle_event(step))
    try:
        while not stop["yes"]:
            runtime.pump()
            now = time.monotonic()
            if now - last >= 2.4:
                step += 1
                runtime.on_state(_cycle_event(step))
                last = now
            time.sleep(0.016)
    finally:
        runtime.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
