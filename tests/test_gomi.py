"""Gomi island: state machine, event mapping, original art. Headless."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from edge.companion import draw_companion, mood_from, say
from edge.gomi.character import blink_envelope, draw_gomi, pose_at
from edge.gomi.island import GomiRuntime, MemoryIsland, step_spring
from edge.gomi.states import (
    ALERT_HOLD_S,
    GomiBrain,
    GomiState,
    GuardianEvent,
    event_from_reflex,
    visible_strings,
)
from edge.gomi_boot import chain_mask_listeners, start_gomi, stop_gomi
from edge.reflex import STATUS_CLOUD_LOOK, STATUS_NOT_PROTECTING
from vision_core.contracts import ActionType, ProtectionAction, TrackedRegion, TrackMode, new_trace_id


def _mask(action: ActionType = ActionType.OPAQUE_MASK) -> ProtectionAction:
    return ProtectionAction(
        action,
        TrackedRegion(1, (0, 0, 8, 8), "flagged", 0.9, 0.1, 0.0, TrackMode.TRACKED),
        "FLAGGED_STRICT_BLOCK",
        "strict-v2",
        new_trace_id(),
    )


def test_paused_and_not_protecting_are_sleep() -> None:
    brain = GomiBrain()
    paused = brain.apply(GuardianEvent(paused=True, status="LOCAL_HOLD", mask_count=1), 0.0)
    assert paused.state is GomiState.SLEEP
    assert paused.scene_line == "paused"
    lost = brain.apply(GuardianEvent(status=STATUS_NOT_PROTECTING, protecting=False), 1.0)
    assert lost.state is GomiState.SLEEP


def test_new_mask_is_alert_then_settles() -> None:
    brain = GomiBrain()
    first = brain.apply(GuardianEvent(status=STATUS_CLOUD_LOOK, mask_count=1, last_action="OPAQUE_MASK"), 0.0)
    assert first.state is GomiState.ALERT
    assert first.scene_line == "masked 1 region"
    holding = brain.apply(
        GuardianEvent(status=STATUS_CLOUD_LOOK, mask_count=1, last_action="OPAQUE_MASK"),
        ALERT_HOLD_S + 0.05,
    )
    assert holding.state is GomiState.ALERT
    assert "covering" in holding.headline


def test_stays_alert_while_overlay() -> None:
    brain = GomiBrain()
    first = brain.apply(
        GuardianEvent(status=STATUS_CLOUD_LOOK, mask_count=2, last_action="BLUR", working=True),
        0.0,
    )
    assert first.state is GomiState.ALERT
    later = brain.apply(
        GuardianEvent(status=STATUS_CLOUD_LOOK, mask_count=2, last_action="BLUR", working=True, band="mid"),
        ALERT_HOLD_S + 2.0,
    )
    assert later.state is GomiState.ALERT


def test_no_masks_is_clear() -> None:
    brain = GomiBrain()
    snap = brain.apply(GuardianEvent(status="LOCAL_HOLD", mask_count=0), 0.0)
    assert snap.state is GomiState.CLEAR
    assert snap.scene_line == "all clear"


def test_reinspect_is_working() -> None:
    brain = GomiBrain()
    snap = brain.apply(
        GuardianEvent(status=STATUS_CLOUD_LOOK, mask_count=0, working=True, mood="unsure", band="mid"),
        0.0,
    )
    assert snap.state is GomiState.WORKING
    assert snap.scene_line == "taking a closer look"


def test_event_from_reflex_counts_masks() -> None:
    fake = SimpleNamespace(
        status=STATUS_CLOUD_LOOK,
        policy_name="strict",
        backend="local",
        companion_line="Cover is on.",
        mood="covering",
        paused=False,
        protecting=True,
        actions=[_mask(), _mask(ActionType.BLUR)],
        traces=[SimpleNamespace(tool="cool.reinspect_crop")],
        band="mid",
    )
    event = event_from_reflex(fake)
    assert event.mask_count == 2
    assert event.working is True
    assert event.last_action == "BLUR"
    assert event.policy_name == "strict"


def test_visible_copy_is_gomi_not_trademark() -> None:
    brain = GomiBrain()
    snap = brain.apply(GuardianEvent(mask_count=2, last_action="OPAQUE_MASK"), 0.0)
    blob = " ".join(visible_strings(snap)).lower()
    assert "gomi" in blob
    for banned in ("coucou", "mochi", "notchbuddy", "gumo"):
        assert banned not in blob


def test_draw_gomi_changes_pixels() -> None:
    canvas = np.full((120, 120, 3), 16, dtype=np.uint8)
    draw_gomi(canvas, origin=(8, 8), size=96, state=GomiState.IDLE, t=0.4)
    assert float(canvas.mean()) > 16.0
    hud = np.full((200, 320, 3), 16, dtype=np.uint8)
    draw_companion(hud, "watching", say("NO_ACTION"))
    assert float(hud.mean()) > 16.0
    assert mood_from(status=STATUS_NOT_PROTECTING) == "lost"


def test_memory_island_records_events() -> None:
    backend = MemoryIsland()
    runtime = GomiRuntime(backend)
    runtime.on_state(
        SimpleNamespace(
            status=STATUS_CLOUD_LOOK,
            policy_name="balanced",
            backend="local",
            companion_line="Softening a flagged area.",
            mood="covering",
            paused=False,
            protecting=True,
            actions=[_mask(ActionType.BLUR)],
            traces=[],
            band="low",
        )
    )
    assert backend.snapshots
    assert backend.snapshots[-1].state is GomiState.ALERT
    runtime.set_hover(True)
    runtime.pump()
    assert runtime.expand_target == 0.0
    runtime.toggle_pin()
    runtime.pump()
    assert runtime.expand_target == 1.0
    runtime.stop()
    assert backend.closed is True


def test_expand_does_not_auto_toggle_on_state_flicker() -> None:
    """Idle/alert/clear may animate Gomi but must not open or close the pill."""
    runtime = GomiRuntime(MemoryIsland())
    assert runtime.expand_target == 0.0
    watching = SimpleNamespace(
        status=STATUS_CLOUD_LOOK,
        policy_name="balanced",
        backend="local",
        companion_line="Watching. All clear.",
        mood="watching",
        paused=False,
        protecting=False,
        actions=[],
        traces=[],
        band="low",
    )
    covering = SimpleNamespace(
        status=STATUS_CLOUD_LOOK,
        policy_name="balanced",
        backend="local",
        companion_line="Softening a flagged area.",
        mood="covering",
        paused=False,
        protecting=True,
        actions=[_mask(ActionType.BLUR)],
        traces=[],
        band="low",
    )
    runtime.on_state(watching)
    runtime.set_hover(True)
    runtime.pump()
    assert runtime.pinned is False
    assert runtime.expand_target == 0.0
    runtime.on_state(covering)
    runtime.pump()
    runtime.on_state(watching)
    runtime.pump()
    assert runtime.expand_target == 0.0
    runtime.toggle_pin()
    runtime.on_state(covering)
    runtime.pump()
    runtime.on_state(watching)
    runtime.set_hover(False)
    runtime.pump()
    assert runtime.pinned is True
    assert runtime.expand_target == 1.0
    runtime.toggle_pin()
    runtime.on_state(covering)
    runtime.pump()
    assert runtime.expand_target == 0.0
    runtime.stop()


def test_click_spring_reaches_expanded() -> None:
    value, vel = 0.0, 0.0
    for _ in range(45):
        value, vel = step_spring(value, vel, 1.0, 1.0 / 60.0)
    assert value > 0.85


def test_idle_gomi_animates_across_time() -> None:
    a = np.full((80, 80, 3), 16, dtype=np.uint8)
    b = np.full((80, 80, 3), 16, dtype=np.uint8)
    draw_gomi(a, origin=(0, 0), size=80, state=GomiState.IDLE, t=0.0)
    draw_gomi(b, origin=(0, 0), size=80, state=GomiState.IDLE, t=0.55)
    assert not np.array_equal(a, b)
    assert blink_envelope(0.03) > 0.2
    assert blink_envelope(1.4) == 0.0
    idle = pose_at(GomiState.IDLE, 0.4)
    assert idle.sx != 1.0 or idle.sy != 1.0


def test_headless_boot_has_no_window() -> None:
    runtime = start_gomi(enabled=True, headless=True)
    assert runtime is not None
    assert runtime.window_ids() == []
    runtime.on_state(SimpleNamespace(status=STATUS_NOT_PROTECTING, paused=True, protecting=False, actions=[], traces=[]))
    assert runtime.brain.state is GomiState.SLEEP
    stop_gomi(runtime)
    assert start_gomi(enabled=False) is None


def test_chain_mask_listeners_calls_both() -> None:
    seen: list[int] = []
    both = chain_mask_listeners(lambda s: seen.append(1), lambda s: seen.append(2))
    assert both is not None
    both(SimpleNamespace())
    assert seen == [1, 2]
    assert chain_mask_listeners(None, None) is None
