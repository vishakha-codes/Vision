"""Parity of COOL (4.14, static graph) vs stock OpenCV 5 (dynamic graph) on the box.

Reads the bench JSONs + raw-output npz files written by ``bench.py`` and
writes ``parity.json``: max |delta| of the raw (B, 22, 2100) output on 4 fixed
blobs, and per-picture top-detection agreement on the 150-picture workload.
Also folds the three bench runs into ``summary.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VARIANTS = ("cool_static_b4", "stock_dynamic", "stock_static_b4")


def main() -> None:
    ref = np.load(HERE / "raw_stock_dynamic.npz")["raw"]
    ref_json = json.loads((HERE / "stock_dynamic.json").read_text())
    out = {"reference": "stock_dynamic (OpenCV 5.0.0 wheel, original nudenet 320n.onnx)", "raw": {}, "top_dets": {}}
    for name in ("cool_static_b4", "stock_static_b4"):
        raw = np.load(HERE / f"raw_{name}.npz")["raw"]
        diff = np.abs(raw.astype(np.float64) - ref.astype(np.float64))
        cls_ref, cls = ref[:, 4:, :], raw[:, 4:, :]
        out["raw"][name] = {
            "shape": list(raw.shape),
            "max_abs_diff_class_scores": float(np.abs(cls - cls_ref).max()),
            "max_abs_diff_box_px": float(diff[:, :4, :].max()),
            "max_class_score_ref": float(cls_ref.max()),
            "argmax_class_agreement": float(np.mean(cls.argmax(1) == cls_ref.argmax(1))),
        }
        other = json.loads((HERE / f"{name}.json").read_text())
        pairs = list(zip(ref_json["top_dets"], other["top_dets"]))
        same = sum(1 for a, b in pairs if [x[0] for x in a] == [x[0] for x in b])
        score_d = [abs(x[1] - y[1]) for a, b in pairs for x, y in zip(a, b)]
        out["top_dets"][name] = {
            "pictures": len(pairs),
            "same_label_list": same,
            "pictures_with_any_det_ref": sum(1 for a, _ in pairs if a),
            "max_abs_score_diff": max(score_d) if score_d else 0.0,
        }
    (HERE / "parity.json").write_text(json.dumps(out, indent=2))

    runs = {v: json.loads((HERE / f"{v}.json").read_text()) for v in VARIANTS}
    base = runs["stock_dynamic"]["crops_per_s"]
    summary = {
        "instance_id": runs["cool_static_b4"]["instance_id"],
        "instance_type": runs["cool_static_b4"]["instance_type"],
        "workload": runs["cool_static_b4"]["workload"],
        "rows": {
            v: {
                "opencv": r["opencv"]["version"],
                "cool": r["opencv"]["cool"],
                "custom_hal": r["opencv"]["custom_hal"],
                "graph": r["detector"]["model"],
                "crops_per_s": r["crops_per_s"],
                "speedup_vs_stock_dynamic": round(r["crops_per_s"] / base, 3),
                "picture_p50_ms": r["picture_latency_ms"]["p50"],
                "picture_p95_ms": r["picture_latency_ms"]["p95"],
                "forward_mean_ms": r["stages_ms"]["forward_ms"]["mean"],
                "non_dnn_mean_ms": round(sum(r["stages_ms"][k]["mean"] for k in ("decode_ms", "hash_ms", "prep_ms", "post_ms")), 3),
                "cpu_util_of_2_vcpu": r["cpu"]["cpu_util_of_2_vcpu"],
                "usd_per_1k_crops": r["cost"]["usd_per_1k_crops"],
            }
            for v, r in runs.items()
        },
        "parity": out,
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
