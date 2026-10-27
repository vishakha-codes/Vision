"""Conflict-safe Gomi island boot. Call from ``edge.app``.

Keeps the merge tiny: CLI flags + start / pump / stop + listener chain.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from edge.gomi.island import GomiRuntime, open_island

MaskFn = Callable[[Any], None]


def add_cli_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--companion",
        dest="companion",
        action="store_true",
        default=True,
        help="show the Gomi island (default on)",
    )
    parser.add_argument(
        "--no-companion",
        dest="companion",
        action="store_false",
        help="do not show the Gomi island",
    )


def chain_mask_listeners(*fns: MaskFn | None) -> MaskFn | None:
    """Call every non-None listener. Lets sibling code keep its own callback."""
    active = [fn for fn in fns if fn is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def _all(state: Any) -> None:
        for fn in active:
            fn(state)

    return _all


def start_gomi(
    *,
    enabled: bool = True,
    headless: bool = False,
    character: str | None = None,
    policy: str = "balanced",
    detect: str = "synthetic",
) -> GomiRuntime | None:
    if not enabled:
        return None
    return open_island(headless=headless, character=character, policy=policy, detect=detect)


def pump_gomi(runtime: GomiRuntime | None) -> None:
    if runtime is not None:
        runtime.pump()


def stop_gomi(runtime: GomiRuntime | None) -> None:
    if runtime is not None:
        runtime.stop()


def window_ids(runtime: GomiRuntime | None) -> list[int]:
    if runtime is None:
        return []
    return runtime.window_ids()


def front_gomi(runtime: GomiRuntime | None) -> None:
    if runtime is not None:
        runtime.front()


def island_backend_name(runtime: GomiRuntime | None) -> str:
    if runtime is None:
        return "off"
    return type(runtime.backend).__name__
