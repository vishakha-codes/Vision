"""Clay skins, morph interrupt, sprung island layout. Headless."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from edge.gomi.catalog import DEFAULT_SKIN, REQUIRED_IDS, SKIN_IDS, STATE_IDS, load_catalog
from edge.gomi.character import draw_gomi
from edge.gomi.geometry import Axis, IslandGeometry, layout_for
from edge.gomi.island import GomiRuntime, MemoryIsland
from edge.gomi.morph import Morph
from edge.gomi.prefs import load_prefs, save_prefs
from edge.gomi.states import GomiState


def test_catalog_has_five_skins_and_shared_ids() -> None:
    catalog = load_catalog()
    assert tuple(catalog) == SKIN_IDS
    assert DEFAULT_SKIN == "pebble"
    for skin, states in catalog.items():
        assert tuple(states) == STATE_IDS
        ids = set(states["idle"].shapes)
        for state, fig in states.items():
            assert set(fig.shapes) == ids
            for required in REQUIRED_IDS:
                assert required in fig.shapes
            assert fig.skin == skin
            assert fig.state == state


def test_default_face_is_clay_not_teal_seed() -> None:
    canvas = np.zeros((96, 96, 3), dtype=np.uint8)
    draw_gomi(canvas, origin=(0, 0), size=96, state=GomiState.IDLE, t=0.2, skin="pebble")
    mask = canvas.sum(axis=2) > 40
    assert bool(mask.any())
    bgr = canvas[mask].mean(axis=0)
    assert bgr[2] > bgr[0] + 8  # cream clay, not the old teal seed


def test_morph_interrupt_does_not_queue() -> None:
    morph = Morph(load_catalog(), "pebble", "idle")
    morph.retarget("alert", 0.0)
    mid = morph.base_at(0.08)
    assert morph.source is not None
    morph.retarget("sleep", 0.08)
    assert morph.state == "sleep"
    assert morph._u == 0.0
    again = morph.base_at(0.08)
    assert again.state in {"alert", "sleep", "idle"}
    assert mid.shapes["body"].ry != 0


def test_open_springs_and_close_does_not_overshoot() -> None:
    axis = Axis(220.0)
    axis.spring_to(520.0)
    peak = axis.value
    for _ in range(60):
        axis.step(1.0 / 60.0)
        peak = max(peak, axis.value)
    assert peak > 450.0

    closing = Axis(520.0)
    closing.curve_towards(220.0)
    prev = closing.value
    for _ in range(30):
        closing.step(1.0 / 60.0)
        assert closing.value <= prev + 0.05
        prev = closing.value
    assert closing.value < 300.0


def test_expanded_layout_puts_character_left_of_options() -> None:
    lay = layout_for(520.0, 176.0, 20.0, 72.0)
    assert lay.char_x + lay.char_d < lay.panel_x + 1.0
    kinds = {hit.kind for hit in lay.hits}
    assert kinds == {"character", "policy", "detect"}
    skins = [hit.value for hit in lay.hits if hit.kind == "character"]
    assert skins == ["pebble", "fox", "ghost", "bear", "cloud"]
    assert all(hit.x >= lay.panel_x - 1.0 for hit in lay.hits)
    compact = layout_for(220.0, 38.0, 6.0, 28.0)
    assert compact.hits == []


def test_geometry_open_moves_character_slot() -> None:
    geom = IslandGeometry()
    geom.set_expanded(True)
    for _ in range(50):
        lay = geom.step(1.0 / 60.0)
    assert lay.width > 450.0
    assert lay.char_d > 60.0
    assert lay.char_x > 12.0
    geom.set_expanded(False)
    heights = []
    for _ in range(25):
        lay = geom.step(1.0 / 60.0)
        heights.append(lay.height)
    assert heights[-1] < heights[0]
    assert all(heights[i] <= heights[i - 1] + 0.2 for i in range(1, len(heights)))


def test_prefs_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "island.json"
    save_prefs({"character_id": "fox", "policy": "strict", "detect": "nsfw"}, path)
    data = load_prefs(path)
    assert data["character_id"] == "fox"
    assert data["policy"] == "strict"
    assert data["detect"] == "nsfw"
    save_prefs({"character_id": "nope"}, path)
    assert load_prefs(path)["character_id"] == "pebble"


def test_activate_switches_skin_without_opening_from_hover(tmp_path: Path) -> None:
    runtime = GomiRuntime(MemoryIsland(), character="pebble", prefs_path=tmp_path / "island.json")
    runtime.set_hover(True)
    runtime.pump()
    assert runtime.expand_target == 0.0
    runtime.pinned = True
    runtime.expand_target = 1.0
    geom = runtime.geom
    geom.expanded = True
    geom.w.value = 520.0
    geom.h.value = 176.0
    geom.r.value = 22.0
    geom.cx.value = 20.0
    geom.cd.value = 72.0
    hit = next(h for h in runtime.current_layout().hits if h.value == "bear")
    runtime.activate(hit)
    assert runtime.character_id == "bear"
    assert runtime.morph.skin == "bear"
    runtime.stop()
