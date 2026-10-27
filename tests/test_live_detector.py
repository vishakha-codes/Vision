"""cv2.dnn NudeNet detector: letterbox, YOLOv8 decode, batching, ORT parity.

Only benign synthetic images (gradients, noise, drawn shapes) go through the
model; no weights or media are downloaded.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_core.detect_nsfw import nudenet_available
from vision_core.live.detector import (
    NUDENET_LABELS,
    Letterbox,
    NudeNetDetector,
    _infer_input_size,
    decode_yolov8,
    letterbox,
    make_blob,
)
from vision_core.live.synth import photo_texture

needs_model = pytest.mark.skipif(not nudenet_available(), reason="nudenet weights missing")


def _benign(n: int = 4) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    imgs = [photo_texture(int(rng.integers(120, 700)), int(rng.integers(120, 700)), rng) for _ in range(n - 1)]
    imgs.append(np.full((200, 300, 3), 128, np.uint8))
    return imgs


def test_letterbox_pads_right_bottom_and_scales() -> None:
    img = np.full((100, 400, 3), 200, np.uint8)
    out, meta = letterbox(img, 320)
    assert out.shape == (320, 320, 3)
    assert meta.scale == pytest.approx(0.8)
    assert out[:80, :].min() == 200 and out[80:, :].max() == 0
    small, meta2 = letterbox(np.full((50, 40, 3), 9, np.uint8), 320)
    assert meta2.scale == pytest.approx(6.4)  # small crops are upscaled
    assert small[:320, :256].min() == 9


def test_decode_maps_back_to_input_pixels() -> None:
    n = 2100
    out = np.zeros((1, 4 + len(NUDENET_LABELS), n), np.float32)
    # One box centred at (160, 80) in model px, 64×32, class 3 at 0.9.
    out[0, :4, 0] = (160, 80, 64, 32)
    out[0, 4 + 3, 0] = 0.9
    out[0, :4, 1] = (162, 81, 64, 32)  # duplicate → NMS
    out[0, 4 + 3, 1] = 0.6
    out[0, :4, 2] = (20, 20, 10, 10)  # below floor
    out[0, 4 + 5, 2] = 0.05
    dets = decode_yolov8(out, [Letterbox(0.8, 400, 100)], NUDENET_LABELS, score_floor=0.12)[0]
    assert len(dets) == 1
    d = dets[0]
    assert d.label == NUDENET_LABELS[3] and d.score == pytest.approx(0.9)
    assert d.xyxy[0] == pytest.approx((160 - 32) / 0.8)
    assert d.xyxy[3] == pytest.approx(min(100.0, (80 + 16) / 0.8))


def test_input_size_from_model_name() -> None:
    from pathlib import Path

    assert _infer_input_size(Path("320n.onnx")) == 320
    assert _infer_input_size(Path("640m.onnx")) == 640
    assert _infer_input_size(Path("custom.onnx")) == 320


@needs_model
def test_cv2_dnn_is_primary_backend() -> None:
    det = NudeNetDetector(backend="auto", warmup=False)
    assert det.backend == "cv2"


@needs_model
def test_parity_cv2_dnn_vs_onnxruntime_raw_and_decoded() -> None:
    cv = NudeNetDetector(backend="cv2", warmup=False)
    ort = NudeNetDetector(backend="ort", warmup=False)
    imgs = _benign(4)
    blob, metas = make_blob(imgs, cv.input_size)
    raw_cv = cv.forward(blob)
    raw_ort = ort.forward(blob)
    assert raw_cv.shape == raw_ort.shape == (4, 22, 2100)
    # Box rows are in model px (0..320); scores 0..1.
    assert np.abs(raw_cv[:, :4] - raw_ort[:, :4]).max() < 0.05
    assert np.abs(raw_cv[:, 4:] - raw_ort[:, 4:]).max() < 1e-3
    dec_cv = decode_yolov8(raw_cv, metas, cv.labels, score_floor=0.05)
    dec_ort = decode_yolov8(raw_ort, metas, ort.labels, score_floor=0.05)
    for a, b in zip(dec_cv, dec_ort):
        assert [d.label for d in a] == [d.label for d in b]
        for da, db in zip(a, b):
            assert da.score == pytest.approx(db.score, abs=1e-3)
            assert np.allclose(da.xyxy, db.xyxy, atol=0.5)


@needs_model
def test_batch_equals_singles() -> None:
    det = NudeNetDetector(backend="cv2", warmup=False, score_floor=0.05)
    imgs = _benign(3)
    batched = det.detect_batch(imgs)
    singles = [det.detect(i) for i in imgs]
    for a, b in zip(batched, singles):
        assert [(d.label, round(d.score, 3)) for d in a] == [(d.label, round(d.score, 3)) for d in b]


@needs_model
def test_blank_crop_has_no_confident_hits() -> None:
    det = NudeNetDetector(backend="cv2", warmup=False)
    dets = det.detect(np.full((320, 320, 3), 250, np.uint8))
    assert all(d.score < 0.3 for d in dets)


# ---------------------------------------------------------------- multi-class decode


def _one_anchor(scores: dict[str, float], n: int = 2100) -> np.ndarray:
    out = np.zeros((1, 4 + len(NUDENET_LABELS), n), np.float32)
    for a in range(3):  # a few overlapping anchors, as a real YOLO head fires
        out[0, :4, a] = (160 + a, 120 + a, 80, 60)
        for lab, sc in scores.items():
            out[0, 4 + NUDENET_LABELS.index(lab), a] = sc - 0.01 * a
    return out


def test_decode_keeps_exposed_runner_up_under_covered_top() -> None:
    """Covered is the anchor's top class, exposed the runner-up: both survive
    (per-class NMS), so medium's 0.45 exposed threshold can fire."""
    out = _one_anchor({"FEMALE_BREAST_COVERED": 0.55, "FEMALE_BREAST_EXPOSED": 0.50, "FACE_FEMALE": 0.2})
    dets = decode_yolov8(out, [Letterbox(1.0, 320, 320)], NUDENET_LABELS, score_floor=0.12)[0]
    by = {d.label: d for d in dets}
    assert set(by) == {"FEMALE_BREAST_COVERED", "FEMALE_BREAST_EXPOSED"}  # NMS: one box per class
    assert by["FEMALE_BREAST_EXPOSED"].score == pytest.approx(0.50)
    assert by["FEMALE_BREAST_COVERED"].score == pytest.approx(0.55)
    assert by["FEMALE_BREAST_EXPOSED"].xyxy == pytest.approx(by["FEMALE_BREAST_COVERED"].xyxy)
    # The old top-class-only decode loses the exposed score.
    old = decode_yolov8(out, [Letterbox(1.0, 320, 320)], NUDENET_LABELS, score_floor=0.12, extra_labels=None)[0]
    assert [d.label for d in old] == ["FEMALE_BREAST_COVERED"]


def test_decode_extra_columns_skip_never_mask_runner_ups() -> None:
    """Only maskable columns are added; a face runner-up is not."""
    out = _one_anchor({"BELLY_EXPOSED": 0.6, "FACE_FEMALE": 0.5})
    dets = decode_yolov8(out, [Letterbox(1.0, 320, 320)], NUDENET_LABELS, score_floor=0.12)[0]
    assert [d.label for d in dets] == ["BELLY_EXPOSED"]
    # ...but a face that is the top class is still reported (probes / logs).
    out = _one_anchor({"FACE_FEMALE": 0.7, "BUTTOCKS_COVERED": 0.3})
    labels = sorted(d.label for d in decode_yolov8(out, [Letterbox(1.0, 320, 320)], NUDENET_LABELS)[0])
    assert labels == ["BUTTOCKS_COVERED", "FACE_FEMALE"]


def test_decode_caps_candidates_on_noisy_output() -> None:
    rng = np.random.default_rng(1)
    out = np.zeros((1, 4 + len(NUDENET_LABELS), 2100), np.float32)
    out[0, :2] = rng.uniform(0, 320, (2, 2100))
    out[0, 2:4] = 20.0
    out[0, 4:] = rng.uniform(0, 0.3, (len(NUDENET_LABELS), 2100))
    out[0, 4 + NUDENET_LABELS.index("ANUS_EXPOSED"), 7] = 0.99
    dets = decode_yolov8(out, [Letterbox(1.0, 320, 320)], NUDENET_LABELS, max_candidates=64)[0]
    assert 0 < len(dets) <= 64
    assert dets[0].label == "ANUS_EXPOSED" and dets[0].score == pytest.approx(0.99)


def test_labels_match_the_installed_nudenet_wheel() -> None:
    nn = pytest.importorskip("nudenet.nudenet")
    assert tuple(nn.__dict__["__labels"]) == NUDENET_LABELS


def test_policy_and_class_groups_use_exact_nudenet_labels() -> None:
    import json

    from vision_core.live.modes import POLICY_DIR
    from vision_core.nudenet_labels import (
        COVERED_INTIMATE_CLASSES,
        EXPLICIT_CLASSES,
        INTIMATE_CLASSES,
        NEVER_MASK_CLASSES,
        SUGGESTIVE_CLASSES,
    )

    known = set(NUDENET_LABELS)
    for group in (COVERED_INTIMATE_CLASSES, EXPLICIT_CLASSES, INTIMATE_CLASSES, NEVER_MASK_CLASSES, SUGGESTIVE_CLASSES):
        assert set(group) <= known
    for path in sorted(POLICY_DIR.glob("live_*.json")):
        raw = json.loads(path.read_text())
        names = set(raw.get("classes") or ()) | set(raw.get("class_score_on") or {}) | set(raw.get("class_score_unsure") or {})
        assert names <= known, (path.name, names - known)


def test_fake_raw_net_goes_through_the_real_decode() -> None:
    from eval_live.fakenet import FakeRawNet
    from vision_core.live.synth import MARKER_BGR

    img = np.full((240, 400, 3), 90, np.uint8)
    img[60:120, 100:180] = MARKER_BGR
    net = FakeRawNet({"FEMALE_BREAST_COVERED": 0.62, "FEMALE_BREAST_EXPOSED": 0.48})
    assert net.backend == "fake"
    dets = net.detect_batch([img, np.full((100, 100, 3), 90, np.uint8)])
    assert dets[1] == []
    by = {d.label: d for d in dets[0]}
    assert set(by) == {"FEMALE_BREAST_COVERED", "FEMALE_BREAST_EXPOSED"}
    assert by["FEMALE_BREAST_EXPOSED"].score == pytest.approx(0.48)
    x1, y1, x2, y2 = by["FEMALE_BREAST_COVERED"].xyxy
    assert abs(x1 - 100) < 3 and abs(y1 - 60) < 3 and abs(x2 - 180) < 3 and abs(y2 - 120) < 3
    assert net.calls == 1 and net.crops == 2
