"""Shared dataclasses for laptop reflex and Graviton/COOL tool calls.

vision_core stays free of PySide, screen capture, and Bedrock.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4

import numpy as np


class TrackMode(str, Enum):
    NEW = "NEW"
    TRACKED = "TRACKED"
    REINSPECT = "REINSPECT"
    HELD = "HELD"
    EXPIRED = "EXPIRED"


class ActionType(str, Enum):
    OPAQUE_MASK = "OPAQUE_MASK"
    BLUR = "BLUR"
    PIXELATE = "PIXELATE"
    NO_ACTION = "NO_ACTION"
    HOLD = "HOLD"
    RELEASE = "RELEASE"
    FULL_WINDOW_MASK = "FULL_WINDOW_MASK"
    REVIEW = "REVIEW"


FLAGGED_LABELS = frozenset({"flagged", "explicit_region"})
PROTECTIVE_ACTIONS = frozenset(
    {
        ActionType.OPAQUE_MASK,
        ActionType.BLUR,
        ActionType.PIXELATE,
        ActionType.FULL_WINDOW_MASK,
        ActionType.HOLD,
    }
)


@dataclass
class Frame:
    frame_id: int
    t_ms: float
    image_bgr: np.ndarray
    width: int
    height: int


@dataclass
class Detection:
    label: str
    score: float
    xyxy: tuple[float, float, float, float]
    detector_id: str
    frame_id: int


@dataclass
class TrackedRegion:
    track_id: int
    xyxy: tuple[float, float, float, float]
    label: str
    stable_score: float
    uncertainty: float
    last_seen_ms: float
    mode: TrackMode
    velocity: tuple[float, float] = (0.0, 0.0)
    hits: int = 1
    misses: int = 0
    mask_on: bool = False


@dataclass
class ProtectionAction:
    action: ActionType
    region: TrackedRegion | None
    reason_code: str
    policy_version: str
    trace_id: str


@dataclass
class AgentTrace:
    trace_id: str
    frame_id: int
    tool: str
    observation: dict[str, Any]
    decision: str
    next_action: str
    latency_ms: float
    image_persisted: bool = False


@dataclass
class SceneChangeResult:
    change_score: float
    binary: np.ndarray
    contours: list
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class Tile:
    image_bgr: np.ndarray
    xyxy: tuple[int, int, int, int]
    area: int
    change_score: float


@dataclass
class MaskSpec:
    action: str
    xyxy: tuple[int, int, int, int]
    reason_code: str
    track_id: int | None = None


def new_trace_id() -> str:
    return uuid4().hex[:16]


def _jsonable(value: Any) -> Any:
    """Drop raw pixels from anything that might be logged."""
    if isinstance(value, np.ndarray):
        return {"omitted": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Frame):
        return {
            "frame_id": value.frame_id,
            "t_ms": value.t_ms,
            "width": value.width,
            "height": value.height,
        }
    if isinstance(value, Tile):
        return {"xyxy": list(value.xyxy), "area": value.area, "change_score": value.change_score}
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items() if k != "image_bgr"}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items() if k != "image_bgr"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def make_trace(
    *,
    frame_id: int,
    tool: str,
    observation: dict[str, Any],
    decision: str,
    next_action: str,
    latency_ms: float,
    trace_id: str | None = None,
) -> AgentTrace:
    """Build a privacy-safe tool trace. image_persisted is always False."""
    return AgentTrace(
        trace_id=trace_id or new_trace_id(),
        frame_id=frame_id,
        tool=tool,
        observation=_jsonable(observation),
        decision=decision,
        next_action=next_action,
        latency_ms=float(latency_ms),
        image_persisted=False,
    )


def trace_to_dict(trace: AgentTrace) -> dict[str, Any]:
    payload = asdict(trace)
    payload["observation"] = _jsonable(payload.get("observation") or {})
    payload["image_persisted"] = False
    payload.pop("image_bgr", None)
    return payload
