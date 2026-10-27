"""Gomi island state machine. No window, no AppKit — safe for pytest."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from vision_core.contracts import PROTECTIVE_ACTIONS, ActionType

ALERT_HOLD_S = 1.25

# User-visible copy. Keep this list free of Coucou / Mochi trademarks.
HEADLINE = {
    "idle": "Gomi · watching",
    "alert": "Gomi · covering",
    "working": "Gomi · looking closer",
    "clear": "Gomi · all clear",
    "sleep": "Gomi · paused",
}

SCENE_WATCHING = "watching"
SCENE_ALL_CLEAR = "all clear"
SCENE_PAUSED = "paused"
SCENE_CLOSER = "taking a closer look"


class GomiState(str, Enum):
    IDLE = "idle"
    ALERT = "alert"
    WORKING = "working"
    CLEAR = "clear"
    SLEEP = "sleep"


@dataclass(frozen=True)
class GuardianEvent:
    """Thin, window-free view of a reflex tick."""

    status: str = "LOCAL_HOLD"
    policy_name: str = "balanced"
    backend: str = "local"
    companion_line: str = "Watching. All clear."
    mood: str = "watching"
    paused: bool = False
    protecting: bool = True
    mask_count: int = 0
    band: str = "low"
    working: bool = False
    last_action: str = "NO_ACTION"

    @staticmethod
    def idle() -> GuardianEvent:
        return GuardianEvent()


@dataclass(frozen=True)
class IslandSnapshot:
    state: GomiState = GomiState.IDLE
    headline: str = HEADLINE["idle"]
    policy_name: str = "balanced"
    backend: str = "local"
    last_action: str = "NO_ACTION"
    scene_line: str = SCENE_WATCHING
    companion_line: str = "Watching. All clear."
    mask_count: int = 0
    t: float = 0.0

    @staticmethod
    def empty() -> IslandSnapshot:
        return IslandSnapshot()


def _action_name(action: Any) -> str:
    raw = getattr(action, "action", action)
    value = getattr(raw, "value", raw)
    return str(value)


def _is_protective(action: Any) -> bool:
    raw = getattr(action, "action", action)
    if raw in PROTECTIVE_ACTIONS:
        return True
    if raw is ActionType.FULL_WINDOW_MASK:
        return True
    name = _action_name(action)
    return name in {"OPAQUE_MASK", "BLUR", "PIXELATE", "FULL_WINDOW_MASK"}


def _tools(traces: Iterable[Any] | None) -> set[str]:
    return {str(getattr(trace, "tool", "")) for trace in (traces or [])}


def event_from_reflex(state: Any) -> GuardianEvent:
    """Map a ReflexState (or a duck-typed stand-in) onto GuardianEvent."""
    actions = list(getattr(state, "actions", []) or [])
    traces = list(getattr(state, "traces", []) or [])
    mask_count = sum(1 for item in actions if _is_protective(item))
    last_action = "NO_ACTION"
    for item in actions:
        last_action = _action_name(item)
    tools = _tools(traces)
    reasons = {str(getattr(item, "reason_code", "")) for item in actions}
    working = bool(
        getattr(state, "band", "low") == "mid"
        or getattr(state, "mood", "") == "unsure"
        or "cool.reinspect_crop" in tools
        or "cool.reinspect_nsfw" in tools
        or "UNCERTAIN_REINSPECT" in reasons
        or any(_action_name(item) == "REVIEW" for item in actions)
    )
    return GuardianEvent(
        status=str(getattr(state, "status", "") or ""),
        policy_name=str(getattr(state, "policy_name", "balanced") or "balanced"),
        backend=str(getattr(state, "backend", "local") or "local"),
        companion_line=str(getattr(state, "companion_line", "") or ""),
        mood=str(getattr(state, "mood", "watching") or "watching"),
        paused=bool(getattr(state, "paused", False)),
        protecting=bool(getattr(state, "protecting", False)),
        mask_count=mask_count,
        band=str(getattr(state, "band", "low") or "low"),
        working=working,
        last_action=last_action,
    )


def scene_line_for(state: GomiState, mask_count: int) -> str:
    if state is GomiState.SLEEP:
        return SCENE_PAUSED
    if mask_count > 0:
        suffix = "s" if mask_count != 1 else ""
        return f"masked {mask_count} region{suffix}"
    if state is GomiState.WORKING:
        return SCENE_CLOSER
    if state is GomiState.CLEAR:
        return SCENE_ALL_CLEAR
    return SCENE_WATCHING


def visible_strings(snap: IslandSnapshot) -> list[str]:
    """Every string the island may paint. Used to keep trademarks out of UI."""
    return [
        snap.headline,
        snap.policy_name,
        snap.backend,
        snap.last_action,
        snap.scene_line,
        snap.companion_line,
        "Gomi",
        "policy",
        "backend",
    ]


class GomiBrain:
    """Guardian events → Gomi state. Clock is injected so tests stay headless."""

    def __init__(self) -> None:
        self.prev_masks = 0
        self.state = GomiState.IDLE
        self.alert_until = 0.0
        self.last_event = GuardianEvent.idle()
        self.last_snap = IslandSnapshot.empty()

    def apply(self, event: GuardianEvent, now: float) -> IslandSnapshot:
        self.last_event = event
        self.state = self._choose(event, now)
        self.prev_masks = event.mask_count
        self.last_snap = self._snapshot(event, self.state, now)
        return self.last_snap

    def tick(self, now: float) -> IslandSnapshot:
        return self.apply(self.last_event, now)

    def _choose(self, event: GuardianEvent, now: float) -> GomiState:
        if event.paused or event.status == "NOT_PROTECTING":
            self.alert_until = 0.0
            return GomiState.SLEEP
        if event.mask_count > self.prev_masks:
            self.alert_until = now + ALERT_HOLD_S
        if event.mask_count > 0:
            # Stay alert for the whole cover. Working only when there is
            # no frost yet (a distinct reinspect look).
            return GomiState.ALERT
        if now < self.alert_until:
            return GomiState.ALERT
        if event.working:
            return GomiState.WORKING
        return GomiState.CLEAR

    def _snapshot(self, event: GuardianEvent, state: GomiState, now: float) -> IslandSnapshot:
        line = event.companion_line or "Watching the screen."
        return IslandSnapshot(
            state=state,
            headline=HEADLINE[state.value],
            policy_name=event.policy_name,
            backend=event.backend,
            last_action=event.last_action,
            scene_line=scene_line_for(state, event.mask_count),
            companion_line=line,
            mask_count=event.mask_count,
            t=now,
        )
