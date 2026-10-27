"""Deterministic policy.evaluate over Strict / Balanced / Relaxed JSON.

Intended mapping on ``fixtures/safe/synthetic_moving_regions``
(first-pass HSV ~0.64, reinspect ~0.92):

- Strict first-pass: OPAQUE_MASK (class=block, overlay 0.40). High band — no reinspect.
- Balanced first-pass: BLUR (overlay 0.55). Mid band — reinspect only if budget/cooldown allow.
- Relaxed first-pass: NO_ACTION / REVIEW (overlay 0.94). Mid band — budgeted reinspect, still no mask.

Post-reinspect (~0.92) must not collapse onto one mask:

- Strict: OPAQUE_MASK (class=block / block 0.70)
- Balanced: BLUR (overlay 0.55, block 0.97 — above typical reinspect)
- Relaxed: NO_ACTION (overlay 0.94 — above typical reinspect)

The old uncertain band ran from ``min_conf-0.12`` to ``max(overlay+0.08, block-0.05)``,
which swallowed every first-pass score on this fixture and forced 750/750 reinspections.
Bands are now a narrow mid window; high and low skip ``cool.reinspect_crop``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from vision_core.contracts import (
    FLAGGED_LABELS,
    ActionType,
    Detection,
    ProtectionAction,
    TrackedRegion,
    TrackMode,
    new_trace_id,
)
from vision_core.geometry import box_area, hysteresis_mask_on

_POLICIES_DIR = Path(__file__).resolve().parents[1] / "policies"

# Default mid window when a policy JSON omits explicit bounds.
# High = score >= overlay (already would mask). Low = score < min_conf-ish.
_DEFAULT_BAND_LO_SLACK = 0.02


@dataclass
class PolicyConfig:
    name: str
    version: str
    description: str
    thresholds: dict
    classes: dict
    agent: dict
    raw: dict


def load_policy(src: PolicyConfig | str | Path | dict) -> PolicyConfig:
    if isinstance(src, PolicyConfig):
        return src
    if isinstance(src, dict):
        data = src
    else:
        path = Path(src)
        if not path.exists():
            alt = _POLICIES_DIR / path.name
            if alt.exists():
                path = alt
        data = json.loads(Path(path).read_text())
    name = str(data.get("name", "unknown"))
    version = f"{name}-v{data.get('version', 1)}"
    return PolicyConfig(
        name=name,
        version=version,
        description=str(data.get("description", "")),
        thresholds=dict(data.get("thresholds") or {}),
        classes=dict(data.get("classes") or {}),
        agent=dict(data.get("agent") or {}),
        raw=data,
    )


def uncertain_bounds(policy: PolicyConfig) -> tuple[float, float]:
    """Return [lo, hi) for the mid / reinspect band.

    High-confidence scores are ``>= hi`` and skip reinspect.
    Low-confidence scores are ``< lo`` and skip reinspect.
    """
    th = policy.thresholds
    if "uncertain_band_low" in th and "uncertain_band_high" in th:
        lo = float(th["uncertain_band_low"])
        hi = float(th["uncertain_band_high"])
        if hi <= lo:
            hi = lo + 0.08
        return lo, hi
    min_conf = float(th.get("flagged_class_min_confidence", 0.5))
    overlay = float(th.get("overlay_trigger_score", 0.55))
    lo = min_conf - _DEFAULT_BAND_LO_SLACK
    hi = overlay
    if hi <= lo:
        hi = lo + 0.08
    return lo, hi


def in_uncertain_band(score: float, policy: PolicyConfig, label: str) -> bool:
    if label not in FLAGGED_LABELS:
        return False
    lo, hi = uncertain_bounds(policy)
    return lo <= float(score) < hi


def score_band(score: float, policy: PolicyConfig, label: str) -> str:
    """Classify a detection score as ``high``, ``mid``, or ``low``."""
    if label not in FLAGGED_LABELS:
        return "low"
    lo, hi = uncertain_bounds(policy)
    value = float(score)
    if value >= hi:
        return "high"
    if value >= lo:
        return "mid"
    return "low"


def class_enabled(policy: PolicyConfig, label: str) -> bool:
    spec = _class_spec(policy, label)
    return str(spec.get("action", "allow")) != "allow"


def agent_settings(policy: PolicyConfig) -> dict:
    raw = policy.agent or {}
    return {
        "reinspect_budget": int(raw.get("reinspect_budget", 40)),
        "reinspect_cooldown_frames": int(raw.get("reinspect_cooldown_frames", 12)),
        "tool_timeout_ms": float(raw.get("tool_timeout_ms", 250)),
    }


def _as_region(item: TrackedRegion | Detection) -> TrackedRegion:
    if isinstance(item, TrackedRegion):
        return item
    return TrackedRegion(
        track_id=-1,
        xyxy=item.xyxy,
        label=item.label,
        stable_score=item.score,
        uncertainty=max(0.0, min(1.0, 1.0 - item.score)),
        last_seen_ms=0.0,
        mode=TrackMode.NEW,
    )


def _class_spec(policy: PolicyConfig, label: str) -> dict:
    if label in policy.classes:
        return policy.classes[label]
    if label in FLAGGED_LABELS:
        return policy.classes.get("flagged") or policy.classes.get("explicit_region") or {}
    return policy.classes.get("safe") or {}


def evaluate(
    regions: Sequence[TrackedRegion | Detection],
    policy: PolicyConfig | str | Path | dict,
    *,
    frame_id: int = 0,
    trace_id: str | None = None,
) -> list[ProtectionAction]:
    """Map regions to ProtectionAction. Same detections differ across presets.

    Strict masks flagged at mid scores. Relaxed may return NO_ACTION there.
    """
    del frame_id  # reserved for Phase 2 tool traces
    cfg = load_policy(policy)
    tid = trace_id or new_trace_id()
    th = cfg.thresholds
    min_area = float(th.get("flagged_class_min_area_px", 0))
    min_conf = float(th.get("flagged_class_min_confidence", 0.5))
    overlay_th = float(th.get("overlay_trigger_score", 0.55))
    block_th = float(th.get("block_trigger_score", 0.8))
    off_th = min(min_conf, overlay_th)

    actions: list[ProtectionAction] = []
    for item in regions:
        region = _as_region(item)
        spec = _class_spec(cfg, region.label)
        class_action = str(spec.get("action", "allow"))
        score = float(region.stable_score)
        area = box_area(region.xyxy)

        if region.mode == TrackMode.EXPIRED:
            updated = replace(region, mask_on=False)
            actions.append(
                ProtectionAction(ActionType.RELEASE, updated, "TRACK_EXPIRED", cfg.version, tid)
            )
            continue

        if region.label not in FLAGGED_LABELS:
            updated = replace(region, mask_on=False)
            actions.append(
                ProtectionAction(ActionType.NO_ACTION, updated, "SAFE_CLASS", cfg.version, tid)
            )
            continue

        if area < min_area:
            updated = replace(region, mask_on=False)
            actions.append(
                ProtectionAction(ActionType.NO_ACTION, updated, "AREA_TOO_SMALL", cfg.version, tid)
            )
            continue

        if region.mode == TrackMode.HELD and region.mask_on:
            updated = replace(region, mask_on=True)
            actions.append(
                ProtectionAction(ActionType.HOLD, updated, "TRACK_HELD", cfg.version, tid)
            )
            continue

        mask_on = hysteresis_mask_on(region.mask_on, score, overlay_th, off_th)

        if score < min_conf and not mask_on:
            reason = "UNCERTAIN_REINSPECT" if in_uncertain_band(score, cfg, region.label) else "BELOW_THRESHOLD"
            action = ActionType.REVIEW if reason == "UNCERTAIN_REINSPECT" else ActionType.NO_ACTION
            updated = replace(region, mask_on=False, mode=TrackMode.REINSPECT if action is ActionType.REVIEW else region.mode)
            actions.append(ProtectionAction(action, updated, reason, cfg.version, tid))
            continue

        if not mask_on:
            reason = "UNCERTAIN_REINSPECT" if in_uncertain_band(score, cfg, region.label) else "BELOW_OVERLAY"
            action = ActionType.REVIEW if reason == "UNCERTAIN_REINSPECT" else ActionType.NO_ACTION
            updated = replace(region, mask_on=False)
            actions.append(ProtectionAction(action, updated, reason, cfg.version, tid))
            continue

        if score >= block_th or class_action == "block":
            action = ActionType.OPAQUE_MASK
            reason = "FLAGGED_ABOVE_BLOCK" if score >= block_th else "FLAGGED_STRICT_BLOCK"
        elif class_action == "overlay":
            action = ActionType.BLUR
            reason = "FLAGGED_ABOVE_OVERLAY"
        else:
            action = ActionType.NO_ACTION
            reason = "CLASS_ALLOW"

        updated = replace(region, mask_on=action != ActionType.NO_ACTION)
        actions.append(ProtectionAction(action, updated, reason, cfg.version, tid))
    return actions
