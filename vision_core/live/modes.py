"""Protection mode → classes, thresholds and mask rules.

Numbers live in ``vision/policies/live_{aggressive,medium,off}.json`` (each
documents its choices). Detector evidence is stored **mode-independent**
(boxes normalised to the picture rect), so a mode switch or a scroll only
re-runs :func:`decide` / :func:`masks_for`, never the model.

- AGGRESSIVE: any hit → whole picture (+pad). UNSURE → whole picture too
  (fail closed). PENDING → pre-frost (``prefrost.mode = "always"``).
- MEDIUM: explicit hits (≥ 0.45) and covered intimate hits (≥ 0.60) →
  padded part boxes unioned per figure. UNSURE → nothing until the second
  look; still UNSURE after it → parts. PENDING → pre-frost only on a "heavy"
  page (``prefrost.mode = "adaptive"``; the engine decides what is heavy from
  ``prefrost_heavy_masks`` / ``prefrost_heavy_window_s``).
- OFF: nothing.

FACE / FEET / ARMPITS / BELLY_COVERED never count, whatever the JSON says.
Label strings must be NudeNet's exactly (``vision_core.nudenet_labels``);
unknown names in a policy file are dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from vision_core.cluster_nsfw import cluster_xyxys
from vision_core.nudenet_labels import (
    COVERED_INTIMATE_CLASSES,
    EXPLICIT_CLASSES,
    INTIMATE_CLASSES,
    NUDENET_LABELS,
    is_never_mask,
)
from vision_core.live.contracts import (
    Box,
    MaskSpec,
    MaskStyle,
    PictureRegion,
    ProtectionMode,
    Verdict,
    VerdictState,
)

POLICY_DIR = Path(__file__).resolve().parents[2] / "policies"

REASON_UNSAFE_PICTURE = "UNSAFE_PICTURE"
REASON_UNSURE_PICTURE = "UNSURE_PICTURE"
REASON_UNSAFE_PARTS = "UNSAFE_PARTS"
REASON_UNSURE_PARTS = "UNSURE_PARTS"
REASON_PENDING = "PENDING_PREFROST"
REASON_CLIPPED = "CLIPPED_UNVERIFIED"  # scrolling in, whole picture not checked yet

PREFROST_MODES = ("always", "adaptive", "never")


@dataclass(frozen=True)
class NormDet:
    """A detector box normalised to its picture rect (0..1), so it follows
    the picture through scrolls and survives in the cache."""

    label: str
    score: float
    box: Box


@dataclass
class Evidence:
    """Mode-independent detector output for one picture's content."""

    dets: tuple[NormDet, ...] = ()
    dets2: tuple[NormDet, ...] | None = None  # second look (None = not run)
    latency_ms: float = 0.0
    n_crops: int = 0

    @property
    def second_look(self) -> bool:
        return self.dets2 is not None

    def max_score(self, classes: frozenset[str]) -> float:
        pool = list(self.dets) + list(self.dets2 or ())
        return max((d.score for d in pool if d.label in classes), default=0.0)


@dataclass
class ModePolicy:
    name: str
    mode: ProtectionMode
    classes: frozenset[str]
    score_on: float
    score_unsure: float
    class_score_on: dict[str, float] = field(default_factory=dict)
    class_score_unsure: dict[str, float] = field(default_factory=dict)
    mask: str = "picture"  # "picture" | "parts" | "none"
    picture_pad_px: float = 6.0
    part_pad_frac: float = 0.12
    unsure_action: str = "mask"  # "mask" | "second_look" | "allow"
    unsure_after_second_look: str = "mask"  # "mask" | "allow"
    # Pre-frost of PENDING (not yet classified) pictures: "always" | "adaptive"
    # (only while the page is heavy: ≥ heavy_masks active masks within the
    # last heavy_window_s seconds) | "never".
    prefrost_mode: str = "never"
    prefrost_heavy_masks: int = 3
    prefrost_heavy_window_s: float = 10.0
    style: MaskStyle = MaskStyle.FROST
    explicit_style: MaskStyle = MaskStyle.FROST
    second_look_rate_per_s: float = 4.0
    second_look_burst: float = 4.0
    second_look_margin_frac: float = 0.06
    budget_ms: float = 120.0
    clipped_fail_closed: bool = False  # cover pictures cut by an edge until seen whole
    clipped_release_static: int = 3  # ...unless still for this many calls,
    clipped_release_aspect: float = 0.5  # SAFE, and visible h ≥ aspect × w
    clipped_release_view: float = 0.3  # or ≥ this share of the content height
    clipped_edge_band: float = 0.0  # parts mode: band (× width) along a cut edge of a hit picture

    def on_threshold(self, label: str) -> float:
        return float(self.class_score_on.get(label, self.score_on))

    def unsure_threshold(self, label: str) -> float:
        return float(self.class_score_unsure.get(label, self.score_unsure))

    def prefrost_active(self, heavy: bool = False) -> bool:
        """Whether PENDING pictures are pre-frosted now (``heavy``: the engine's
        "many masks were active recently" signal, used by ``adaptive``)."""
        return self.prefrost_mode == "always" or (self.prefrost_mode == "adaptive" and bool(heavy))


def _default_classes(mode: ProtectionMode) -> frozenset[str]:
    if mode is ProtectionMode.AGGRESSIVE:
        return frozenset(INTIMATE_CLASSES)
    if mode is ProtectionMode.MEDIUM:
        return frozenset(EXPLICIT_CLASSES | COVERED_INTIMATE_CLASSES)
    return frozenset()


# Built-in fallbacks when ``live_<mode>.json`` is missing or omits a key.
_MEDIUM_COVERED_ON = 0.60
_MEDIUM_COVERED_UNSURE = 0.45
_DEFAULT_CLASS_ON = {
    ProtectionMode.MEDIUM: {c: _MEDIUM_COVERED_ON for c in COVERED_INTIMATE_CLASSES},
}
_DEFAULT_CLASS_UNSURE = {
    ProtectionMode.MEDIUM: {c: _MEDIUM_COVERED_UNSURE for c in COVERED_INTIMATE_CLASSES},
}


def _prefrost_block(raw: dict, mode: ProtectionMode) -> tuple[str, int, float]:
    """``"prefrost": {"mode", "heavy_masks", "heavy_window_s"}``. Older files
    with only ``"prefrost_pending": bool`` map to ``always`` / ``never``."""
    default = {
        ProtectionMode.AGGRESSIVE: "always",
        ProtectionMode.MEDIUM: "adaptive",
        ProtectionMode.OFF: "never",
    }[mode]
    block = raw.get("prefrost")
    if not isinstance(block, dict):
        block = {}
        if "prefrost_pending" in raw:
            block["mode"] = "always" if raw.get("prefrost_pending") else "never"
    pmode = str(block.get("mode", default)).strip().lower()
    if pmode not in PREFROST_MODES:
        pmode = default
    if mode is ProtectionMode.OFF:
        pmode = "never"
    return pmode, max(1, int(block.get("heavy_masks", 3))), max(0.0, float(block.get("heavy_window_s", 10.0)))


def policy_path(mode: ProtectionMode, policy_dir: Path | str | None = None) -> Path:
    return Path(policy_dir or POLICY_DIR) / f"live_{mode.value}.json"


def load_mode_policy(mode: ProtectionMode | str, policy_dir: Path | str | None = None) -> ModePolicy:
    """Read ``live_<mode>.json``; falls back to built-in defaults if missing."""
    mode = ProtectionMode(mode)
    path = policy_path(mode, policy_dir)
    raw: dict = json.loads(path.read_text()) if path.is_file() else {}
    classes = frozenset(raw.get("classes") or _default_classes(mode))
    known = frozenset(NUDENET_LABELS)
    classes = frozenset(c for c in classes if c in known and not is_never_mask(c))
    sl = dict(raw.get("second_look") or {})
    defaults = {
        ProtectionMode.AGGRESSIVE: (0.25, 0.15, "picture", "mask"),
        ProtectionMode.MEDIUM: (0.45, 0.30, "parts", "second_look"),
        ProtectionMode.OFF: (1.01, 1.01, "none", "allow"),
    }[mode]
    class_on = dict(_DEFAULT_CLASS_ON.get(mode, {}) if "class_score_on" not in raw else {})
    class_on.update({str(k): float(v) for k, v in (raw.get("class_score_on") or {}).items()})
    class_unsure = dict(_DEFAULT_CLASS_UNSURE.get(mode, {}) if "class_score_unsure" not in raw else {})
    class_unsure.update({str(k): float(v) for k, v in (raw.get("class_score_unsure") or {}).items()})
    pf_mode, pf_masks, pf_window = _prefrost_block(raw, mode)
    return ModePolicy(
        name=str(raw.get("name") or f"live_{mode.value}"),
        mode=mode,
        classes=classes,
        score_on=float(raw.get("score_on", defaults[0])),
        score_unsure=float(raw.get("score_unsure", defaults[1])),
        class_score_on=class_on,
        class_score_unsure=class_unsure,
        mask=str(raw.get("mask", defaults[2])),
        picture_pad_px=float(raw.get("picture_pad_px", 6.0)),
        part_pad_frac=float(raw.get("part_pad_frac", 0.12)),
        unsure_action=str(raw.get("unsure_action", defaults[3])),
        unsure_after_second_look=str(raw.get("unsure_after_second_look", "mask")),
        prefrost_mode=pf_mode,
        prefrost_heavy_masks=pf_masks,
        prefrost_heavy_window_s=pf_window,
        style=MaskStyle(raw.get("style", "frost")),
        explicit_style=MaskStyle(raw.get("explicit_style", raw.get("style", "frost"))),
        second_look_rate_per_s=float(sl.get("rate_per_s", 4.0)),
        second_look_burst=float(sl.get("burst", 4.0)),
        second_look_margin_frac=float(sl.get("margin_frac", 0.06)),
        budget_ms=float(raw.get("budget_ms", 120.0)),
        clipped_fail_closed=bool(raw.get("clipped_fail_closed", mode is ProtectionMode.AGGRESSIVE)),
        clipped_release_static=int((raw.get("clipped_release") or {}).get("static_calls", 3)),
        clipped_release_aspect=float((raw.get("clipped_release") or {}).get("min_aspect", 0.5)),
        clipped_release_view=float((raw.get("clipped_release") or {}).get("min_view_frac", 0.3)),
        clipped_edge_band=float(raw.get("clipped_edge_band", 0.0)),
    )


# ---------------------------------------------------------------- decisions


def _to_capture(region: PictureRegion, box: Box) -> Box:
    x1, y1, x2, y2 = region.xyxy
    w, h = x2 - x1, y2 - y1
    return (x1 + box[0] * w, y1 + box[1] * h, x1 + box[2] * w, y1 + box[3] * h)


def decide(policy: ModePolicy, region: PictureRegion, evidence: Evidence | None, *, source: str = "local") -> Verdict:
    """Evidence → SAFE / UNSAFE / UNSURE (PENDING when there is none yet)."""
    if evidence is None:
        return Verdict(VerdictState.PENDING, source=source)
    rel = [d for d in evidence.dets if d.label in policy.classes]
    rel2 = [d for d in (evidence.dets2 or ()) if d.label in policy.classes]
    hits = [d for d in rel + rel2 if d.score >= policy.on_threshold(d.label)]
    # After a second look, only its own weak hits keep a picture UNSURE.
    weak_pool = rel2 if evidence.second_look else rel
    weak = [d for d in weak_pool if d.score >= policy.unsure_threshold(d.label)]
    top = max((d.score for d in rel + rel2), default=0.0)
    if hits:
        state, used = VerdictState.UNSAFE, hits
    elif weak:
        state, used = VerdictState.UNSURE, weak
    else:
        state, used = VerdictState.SAFE, []
    return Verdict(
        state=state,
        score=float(top),
        labels=tuple(sorted({d.label for d in used})),
        parts=tuple(_to_capture(region, d.box) for d in used),
        source=source,
        latency_ms=float(evidence.latency_ms),
    )


def needs_second_look(policy: ModePolicy, verdict: Verdict, evidence: Evidence | None) -> bool:
    return (
        evidence is not None
        and not evidence.second_look
        and verdict.state is VerdictState.UNSURE
        and policy.unsure_action in {"mask", "second_look"}
        and policy.second_look_rate_per_s > 0.0
    )


def _clip(box: Box, bound: Box) -> Box | None:
    x1, y1 = max(box[0], bound[0]), max(box[1], bound[1])
    x2, y2 = min(box[2], bound[2]), min(box[3], bound[3])
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return (x1, y1, x2, y2)


def _pad(box: Box, px: float) -> Box:
    return (box[0] - px, box[1] - px, box[2] + px, box[3] + px)


def masks_for(
    policy: ModePolicy,
    region: PictureRegion,
    verdict: Verdict,
    frame_wh: tuple[int, int],
    *,
    evidence: Evidence | None = None,
    reason: str | None = None,
    heavy: bool = False,
) -> list[MaskSpec]:
    """MaskSpecs this mode wants for one picture. ``reason`` forces a
    whole-picture cover with that reason code (fail-closed paths).
    ``heavy``: the page is heavy, so an ``adaptive`` policy pre-frosts
    PENDING pictures (see :meth:`ModePolicy.prefrost_active`)."""
    if policy.mode is ProtectionMode.OFF or policy.mask == "none":
        return []
    frame = (0.0, 0.0, float(frame_wh[0]), float(frame_wh[1]))
    state = verdict.state
    is_tile = region.kind == "tile"
    if reason is not None:
        rect = _clip(_pad(region.xyxy, policy.picture_pad_px), frame)
        return [MaskSpec(region.region_id, rect, policy.style, reason, state)] if rect else []
    if state is VerdictState.PENDING:
        if policy.prefrost_active(heavy) and not is_tile:
            rect = _clip(_pad(region.xyxy, policy.picture_pad_px), frame)
            return [MaskSpec(region.region_id, rect, policy.style, REASON_PENDING, VerdictState.PENDING)] if rect else []
        return []
    if state is VerdictState.SAFE:
        return []
    if state is VerdictState.UNSURE:
        if policy.unsure_action == "allow":
            return []
        second = evidence is not None and evidence.second_look
        if policy.unsure_action == "second_look" and not second:
            return []
        if second and policy.unsure_after_second_look == "allow":
            return []
    explicit = any(lab in EXPLICIT_CLASSES for lab in verdict.labels)
    style = policy.explicit_style if (state is VerdictState.UNSAFE and explicit) else policy.style
    if policy.mask == "picture" or not verdict.parts:
        reason = REASON_UNSAFE_PICTURE if state is VerdictState.UNSAFE else REASON_UNSURE_PICTURE
        # A tile hit covers the whole picture-like area the tile came from.
        base = region.group_xyxy if (is_tile and policy.mask == "picture" and region.group_xyxy) else region.xyxy
        rect = _clip(_pad(base, policy.picture_pad_px), frame)
        return [MaskSpec(region.region_id, rect, style, reason, state)] if rect else []
    # Parts: pad each box by a fraction of its size, union per figure, clip to the picture.
    padded = []
    for x1, y1, x2, y2 in verdict.parts:
        px, py = (x2 - x1) * policy.part_pad_frac, (y2 - y1) * policy.part_pad_frac
        padded.append((x1 - px, y1 - py, x2 + px, y2 + py))
    reason = REASON_UNSAFE_PARTS if state is VerdictState.UNSAFE else REASON_UNSURE_PARTS
    out: list[MaskSpec] = []
    bound = _clip(region.xyxy, frame) or region.xyxy
    for i, box in enumerate(sorted(cluster_xyxys(padded), key=lambda b: (b[1], b[0]))):
        rect = _clip(box, bound)
        if rect:
            out.append(MaskSpec(f"{region.region_id}#p{i}", rect, style, reason, state))
    return out
