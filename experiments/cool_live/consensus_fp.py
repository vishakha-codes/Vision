"""False-flag rate of L2 view / aggregation schemes on benign synthetic pictures.

150 procedural textures (``vision_core.live.synth.photo_texture``, 480..1000
px, seed 5000+), NudeNet 320n, verdict per mode via ``modes.decide``.
No real or explicit imagery exists here, so this measures only the
false-flag side; recall on true positives is NOT measured.

    .venv/bin/python experiments/cool_live/consensus_fp.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

VISION = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VISION))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cloud.second_look import SecondLookService, _nms_norm, consensus, run_views  # noqa: E402
from vision_core.live.contracts import ProtectionMode  # noqa: E402
from vision_core.live.regions import tile_grid  # noqa: E402
from vision_core.live.synth import photo_texture  # noqa: E402


def per_view(det, img, views):
    h, w = img.shape[:2]
    crops = []
    for b, f in views:
        c = img[int(b[1]):int(np.ceil(b[3])), int(b[0]):int(np.ceil(b[2]))]
        crops.append(cv2.flip(c, 1) if f else c)
    out = []
    for (b, f), ds in zip(views, det.detect_batch(crops)):
        vw = b[2] - b[0]
        lst = []
        for d in ds:
            x1, y1, x2, y2 = d.xyxy
            if f:
                x1, x2 = vw - x2, vw - x1
            lst.append((d.label, d.score, ((x1 + b[0]) / w, (y1 + b[1]) / h, (x2 + b[0]) / w, (y2 + b[1]) / h)))
        out.append(lst)
    return out


def main(n: int = 150) -> None:
    svc = SecondLookService()
    det = svc.detector
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for seed in range(n):
        rng = np.random.default_rng(5000 + seed)
        L = int(rng.choice([480, 640, 800, 1000]))
        img = photo_texture(int(L * rng.uniform(0.6, 1.0)), L, rng)
        h, w = img.shape[:2]
        whole = (0.0, 0.0, float(w), float(h))
        g2 = [(b, False) for b in tile_grid(whole, max(320, 0.6 * L), 0.25)]
        g3 = [(b, False) for b in tile_grid(whole, max(128, 0.4 * L), 0.2)]
        schemes = {
            "max(whole) [~local first pass, small pictures]": ([(whole, False)], 1),
            "max(whole+2x2) [~local first pass, large pictures]": ([(whole, False)] + g2, 1),
            "max(3x3 at 0.4) [rejected L2 draft]": (g3, 1),
        }
        for name, (views, _k) in schemes.items():
            pv = per_view(det, img, views)
            dets = tuple(_nms_norm([d for lst in pv for d in lst]))
            for mode in (ProtectionMode.AGGRESSIVE, ProtectionMode.MEDIUM):
                counts[f"{name} / {mode.value}"][svc._verdict(mode, dets)["state"]] += 1
        for prof in ("l2", "deep"):
            res = run_views(det, img, prof)
            for mode in (ProtectionMode.AGGRESSIVE, ProtectionMode.MEDIUM):
                counts[f"SHIPPED {prof} consensus / {mode.value}"][svc._verdict(mode, res.dets)["state"]] += 1
    out = {
        "pictures": n,
        "source": "vision_core.live.synth.photo_texture, benign only; recall not measured",
        "opencv": cv2.__version__,
        "schemes": {k: {"SAFE": c["SAFE"], "UNSURE": c["UNSURE"], "UNSAFE": c["UNSAFE"],
                        "flagged_rate": round((c["UNSURE"] + c["UNSAFE"]) / n, 3)} for k, c in counts.items()},
    }
    (VISION / "experiments" / "cool_live" / "consensus_fp.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
