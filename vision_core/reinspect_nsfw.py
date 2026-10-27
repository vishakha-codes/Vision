"""Second look for ``--detect nsfw``.

Default: re-run the on-device NudeNet detector on an in-memory crop.
Optional: AWS Rekognition ``DetectModerationLabels`` on that crop only
(``--reinspect rekognition``). Crops are JPEG-encoded in RAM and never
written to disk or S3. The full desktop is never uploaded.

Off the synthetic eval path. Do not enable Rekognition for the 750-frame
fixture (would break the ~40 reinspect / IoU ~0.94 numbers and would
send tiles to AWS).
"""

from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

from vision_core.contracts import Detection, Frame
from vision_core.detect_nsfw import detect_nsfw
from vision_core.encode_tile import encode_tile
from vision_core.geometry import as_int_box, box_iou, expand_box
from vision_core.reinspect_crop import ReinspectResult

# Rekognition labels that count as unwanted adult imagery. Not a claim
# that every adult site will hit these, or that news photos never will.
_REKOGNITION_HINTS = (
    "explicit nudity",
    "graphic female nudity",
    "graphic male nudity",
    "sexual activity",
    "illustrated explicit nudity",
    "nudity",
)


def _bgr(image: Frame | np.ndarray) -> np.ndarray:
    if isinstance(image, Frame):
        return image.image_bgr
    return image


def reinspect_nsfw(
    image: Frame | np.ndarray,
    box: tuple[float, float, float, float],
    margin: int = 20,
    frame_id: int = 0,
    *,
    scale: float = 1.0,
    detect_fn: Callable[..., list[Detection]] | None = None,
) -> ReinspectResult:
    """Crop ``box+margin`` in memory and re-run the nsfw detector."""
    bgr = _bgr(image)
    height, width = bgr.shape[:2]
    crop_xyxy = as_int_box(expand_box(box, margin, width, height), width, height)
    x1, y1, x2, y2 = crop_xyxy
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return ReinspectResult(label="unknown", score=0.0, xyxy=box, detections=[], crop_xyxy=crop_xyxy)

    used = crop
    scale_used = 1.0
    if scale > 1.0 and min(crop.shape[0], crop.shape[1]) < 160:
        used = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scale_used = scale

    if detect_fn is None:
        raw = detect_nsfw(used, frame_id=frame_id, stage="reinspect", record_stats=False)
    else:
        # A ``make_nsfw_detect`` closure owns the live ClusterHold; use its
        # hold-free twin so a crop never rewrites held full-frame clusters or
        # ``LAST_STATS`` (v1 bug #12). Plain callables are used as given.
        runner = getattr(detect_fn, "hold_free", detect_fn)
        raw = runner(used, frame_id=frame_id, stage="reinspect")
    remapped: list[Detection] = []
    for det in raw:
        dx1, dy1, dx2, dy2 = det.xyxy
        remapped.append(
            Detection(
                label=det.label,
                score=det.score,
                xyxy=(
                    x1 + dx1 / scale_used,
                    y1 + dy1 / scale_used,
                    x1 + dx2 / scale_used,
                    y1 + dy2 / scale_used,
                ),
                detector_id=f"{det.detector_id}:reinspect",
                frame_id=frame_id,
            )
        )
    if not remapped:
        return ReinspectResult(label="unknown", score=0.0, xyxy=box, detections=[], crop_xyxy=crop_xyxy)
    remapped.sort(key=lambda d: (box_iou(d.xyxy, box), d.score), reverse=True)
    best = remapped[0]
    return ReinspectResult(
        label=best.label,
        score=best.score,
        xyxy=best.xyxy,
        detections=remapped,
        crop_xyxy=crop_xyxy,
    )


def _moderation_score(labels: list[dict[str, Any]]) -> tuple[str, float]:
    best_name = "unknown"
    best = 0.0
    for item in labels:
        name = str(item.get("Name") or item.get("name") or "")
        conf = float(item.get("Confidence") or item.get("confidence") or 0.0) / 100.0
        lowered = name.lower()
        if any(hint in lowered for hint in _REKOGNITION_HINTS):
            if conf > best:
                best = conf
                best_name = "flagged"
    return best_name, max(0.0, min(1.0, best))


def reinspect_rekognition(
    image: Frame | np.ndarray,
    box: tuple[float, float, float, float],
    margin: int = 20,
    frame_id: int = 0,
    *,
    client: Any | None = None,
    min_confidence: float = 60.0,
) -> ReinspectResult:
    """Optional AWS second look. Crop bytes only. Never persist.

    Uses the default boto3 / AWS CLI profile if ``client`` is omitted.
    """
    bgr = _bgr(image)
    height, width = bgr.shape[:2]
    crop_xyxy = as_int_box(expand_box(box, margin, width, height), width, height)
    x1, y1, x2, y2 = crop_xyxy
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return ReinspectResult(label="unknown", score=0.0, xyxy=box, detections=[], crop_xyxy=crop_xyxy)

    payload = encode_tile(crop)
    if not payload:
        return ReinspectResult(label="unknown", score=0.0, xyxy=box, detections=[], crop_xyxy=crop_xyxy)
    if client is None:
        import boto3

        client = boto3.client("rekognition")
    response = client.detect_moderation_labels(
        Image={"Bytes": payload},
        MinConfidence=float(min_confidence),
    )
    labels = list(response.get("ModerationLabels") or [])
    label, score = _moderation_score(labels)
    detections: list[Detection] = []
    if score > 0.0:
        detections.append(
            Detection(
                label=label,
                score=score,
                xyxy=tuple(float(v) for v in crop_xyxy),
                detector_id="rekognition_moderation_v1:reinspect",
                frame_id=frame_id,
            )
        )
    return ReinspectResult(
        label=label,
        score=score,
        xyxy=tuple(float(v) for v in crop_xyxy),
        detections=detections,
        crop_xyxy=crop_xyxy,
    )


def make_nsfw_reinspect(
    *,
    mode: str = "local",
    detect_fn: Callable[..., list[Detection]] | None = None,
    rekognition_client: Any | None = None,
):
    """Orchestrator-compatible ``reinspect_fn``."""

    key = (mode or "local").strip().lower()

    def _local(
        image: Frame | np.ndarray,
        box: tuple[float, float, float, float],
        margin: int = 20,
        frame_id: int = 0,
        **_kwargs: Any,
    ) -> ReinspectResult:
        return reinspect_nsfw(image, box, margin=margin, frame_id=frame_id, detect_fn=detect_fn)

    def _rekog(
        image: Frame | np.ndarray,
        box: tuple[float, float, float, float],
        margin: int = 20,
        frame_id: int = 0,
        **_kwargs: Any,
    ) -> ReinspectResult:
        return reinspect_rekognition(
            image,
            box,
            margin=margin,
            frame_id=frame_id,
            client=rekognition_client,
        )

    if key == "rekognition":
        return _rekog
    return _local
