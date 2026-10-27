"""Laptop companion: Gomi island + HUD portrait.

Artwork is original clay skins (pebble, fox, ghost, bear, cloud). Not Mochi, not Coucou.
Lines come from ``agent.companion.say`` — Bedrock is not imported.
The floating island lives in ``edge.gomi``; this module keeps the old
``mood_from`` / ``say`` / ``draw_companion`` sockets the overlay already calls.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from agent.companion import say as agent_say
from edge.gomi.character import draw_companion_hud
from vision_core.contracts import (
    PROTECTIVE_ACTIONS,
    ActionType,
    AgentTrace,
    ProtectionAction,
)

MOODS = ("watching", "unsure", "covering", "lost")


def say(reason_code: str) -> str:
    """One short status line. Delegates to the existing kids-AI socket."""
    return agent_say(reason_code)


def mood_from(
    *,
    status: str,
    actions: Sequence[ProtectionAction] | None = None,
    traces: Iterable[AgentTrace] | None = None,
    band: str = "low",
    reinspect_count: int = 0,
) -> str:
    """Map traces / actions onto watching / unsure / covering / lost."""
    if status == "NOT_PROTECTING":
        return "lost"
    items = list(actions or [])
    if any(a.action in PROTECTIVE_ACTIONS for a in items):
        return "covering"
    reasons = {a.reason_code for a in items}
    tools = {t.tool for t in (traces or [])}
    if (
        "UNCERTAIN_REINSPECT" in reasons
        or any(a.action is ActionType.REVIEW for a in items)
        or reinspect_count > 0
        or band == "mid"
        or "cool.reinspect_crop" in tools
    ):
        return "unsure"
    return "watching"


def draw_companion(
    canvas,
    mood: str,
    line: str,
    *,
    origin: tuple[int, int] | None = None,
) -> None:
    """Draw one Gomi portrait and the current ``companion.say`` line."""
    draw_companion_hud(canvas, mood, line, origin=origin)
