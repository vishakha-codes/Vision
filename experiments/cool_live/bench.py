"""COOL vs stock OpenCV on the L2 live workload (run ON the Graviton box).

Workload per picture = exactly what ``POST /v1/second_look`` does, minus HTTP:
JPEG decode → dHash/aHash → views (whole, mirrored whole, 2×2 at 0.6 × the
long side) → letterbox + blobFromImages → NudeNet 320n forward →
YOLOv8 decode + NMSBoxesBatched → cross-view consensus (k=2) + NMS.

Pictures are benign procedural textures from ``vision_core.live.synth``
(no real or explicit imagery), sizes 160..1400 px long side, fixed seed, so
every variant sees identical bytes.

    # COOL (static-shape graph; classic importer)
    LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib ~/venv-cool-api/bin/python \
        experiments/cool_live/bench.py --variant cool_static_b4 --out experiments/cool_live/cool.json
    # stock OpenCV 5 (original dynamic graph, new engine) and the same static graph
    ~/venv-stock-opencv/bin/python experiments/cool_live/bench.py --variant stock_dynamic --out ...
    ~/venv-stock-opencv/bin/python experiments/cool_live/bench.py --variant stock_static_b4 --out ...
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import urllib.request
from pathlib import Path

VISION = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VISION))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cloud.cool_detector import StaticBatchDetector, opencv_build  # noqa: E402
from cloud.second_look import CONSENSUS_K, consensus, views_for  # noqa: E402
from vision_core.live.cache import hash_crop  # noqa: E402
from vision_core.live.detector import NudeNetDetector, decode_yolov8, make_blob  # noqa: E402
from vision_core.live.synth import photo_texture  # noqa: E402

C8G_LARGE_USD_H = 0.07976  # us-east-1 on-demand Linux, AWS price list 26 Oct 2026


def pictures(n: int, seed: int = 7) -> list[bytes]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        long_side = int(rng.choice([160, 240, 320, 480, 640, 800, 1000, 1400], p=[.1, .15, .15, .2, .15, .1, .1, .05]))
        aspect = float(rng.uniform(0.5, 1.8))
        w, h = (long_side, max(32, int(long_side / aspect))) if aspect >= 1 else (max(32, int(long_side * aspect)), long_side)
        img = photo_texture(h, w, rng, text_overlay=bool(i % 5 == 0))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append(buf.tobytes())
    return out


def make_detector(variant: str) -> NudeNetDetector:
    if variant.endswith("static_b4"):
        return StaticBatchDetector(VISION / ".models", batch=4, warmup=True)
    if variant.endswith("static_b1"):
        return StaticBatchDetector(VISION / ".models", batch=1, warmup=True)
    if variant == "stock_dynamic":
        import nudenet

        return NudeNetDetector(Path(nudenet.__file__).parent / "320n.onnx", backend="cv2", warmup=True)
    raise SystemExit(f"unknown variant {variant}")


def one(det: NudeNetDetector, jpeg: bytes) -> tuple[dict, list]:
    t = {}
    t0 = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    t1 = time.perf_counter()
    hash_crop(img)
    t2 = time.perf_counter()
    h, w = img.shape[:2]
    views = views_for(w, h, "l2")
    crops = [img[int(b[1]):int(np.ceil(b[3])), int(b[0]):int(np.ceil(b[2]))] for b, _ in views]
    crops = [cv2.flip(c, 1) if f else c for c, (_, f) in zip(crops, views)]
    blob, metas = make_blob(crops, det.input_size)
    t3 = time.perf_counter()
    raw = det.forward(blob)
    t4 = time.perf_counter()
    per = decode_yolov8(raw, metas, det.labels, score_floor=det.score_floor, nms_iou=det.nms_iou)
    per_view = []
    for (b, flip), dets in zip(views, per):
        vw = b[2] - b[0]
        lst = []
        for d in dets:
            x1, x2 = (vw - d.xyxy[2], vw - d.xyxy[0]) if flip else (d.xyxy[0], d.xyxy[2])
            lst.append((d.label, d.score, ((x1 + b[0]) / w, (d.xyxy[1] + b[1]) / h, (x2 + b[0]) / w, (d.xyxy[3] + b[1]) / h)))
        per_view.append(lst)
    final = consensus(per_view, CONSENSUS_K["l2"])  # same aggregation as the service
    t5 = time.perf_counter()
    t.update(decode_ms=(t1 - t0) * 1e3, hash_ms=(t2 - t1) * 1e3, prep_ms=(t3 - t2) * 1e3,
             forward_ms=(t4 - t3) * 1e3, post_ms=(t5 - t4) * 1e3, total_ms=(t5 - t0) * 1e3, views=len(views))
    return t, [(d.label, round(d.score, 4)) for d in final[:5]]


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q)) if xs else 0.0


def instance_id() -> str:
    try:
        tok = urllib.request.urlopen(urllib.request.Request(
            "http://169.254.169.254/latest/api/token", method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}), timeout=1).read().decode()
        req = urllib.request.Request("http://169.254.169.254/latest/meta-data/instance-id",
                                     headers={"X-aws-ec2-metadata-token": tok})
        return urllib.request.urlopen(req, timeout=1).read().decode()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw-out", type=Path, default=None, help="npz of raw outputs on fixed blobs (parity)")
    args = ap.parse_args()

    det = make_detector(args.variant)
    pics = pictures(args.n)
    for j in pics[:5]:
        one(det, j)  # warm caches / allocator

    rows, tops = [], []
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    w0 = time.perf_counter()
    for j in pics:
        t, top = one(det, j)
        rows.append(t)
        tops.append(top)
    wall = time.perf_counter() - w0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu_s = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    n_views = int(sum(r["views"] for r in rows))
    crops_per_s = n_views / wall
    stages = {k: {"mean": float(np.mean([r[k] for r in rows])), "p50": pct([r[k] for r in rows], 50),
                  "p95": pct([r[k] for r in rows], 95)}
              for k in ("decode_ms", "hash_ms", "prep_ms", "forward_ms", "post_ms", "total_ms")}

    raw_info = None
    if args.raw_out:
        rng = np.random.default_rng(123)
        crops = [photo_texture(int(s), int(s * 1.3), rng) for s in (200, 320, 500, 900)]
        blob, _ = make_blob(crops, det.input_size)
        raw = det.forward(blob)
        np.savez_compressed(args.raw_out, raw=raw, blob_sum=float(blob.sum()))
        raw_info = str(args.raw_out)

    out = {
        "variant": args.variant,
        "instance_id": instance_id(),
        "instance_type": "c8g.large (2 vCPU Graviton4, 4 GiB)",
        "opencv": opencv_build(),
        "detector": {"class": type(det).__name__, "model": str(det.model_path.name), "backend": det.backend,
                     "input": det.input_size, "static_batch": getattr(det, "static_batch", None)},
        "workload": {"pictures": len(pics), "views_total": n_views, "views_per_picture": n_views / len(pics),
                     "profile": "l2", "seed": 7, "source": "vision_core.live.synth.photo_texture (benign)"},
        "wall_s": round(wall, 3),
        "crops_per_s": round(crops_per_s, 2),
        "pictures_per_s": round(len(pics) / wall, 2),
        "picture_latency_ms": {"p50": round(stages["total_ms"]["p50"], 2), "p95": round(stages["total_ms"]["p95"], 2),
                               "mean": round(stages["total_ms"]["mean"], 2)},
        "stages_ms": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in stages.items()},
        "cpu": {"process_cpu_s": round(cpu_s, 2), "cpu_util_of_2_vcpu": round(cpu_s / wall / 2.0, 3)},
        "cost": {"usd_per_hour": C8G_LARGE_USD_H,
                 "usd_per_1k_crops": round(C8G_LARGE_USD_H / 3600.0 / crops_per_s * 1000.0, 6),
                 "usd_per_1k_pictures": round(C8G_LARGE_USD_H / 3600.0 / (len(pics) / wall) * 1000.0, 6),
                 "note": "instance-hour cost at 100% busy; excludes EBS/data transfer"},
        "top_dets": tops,
        "raw_outputs": raw_info,
        "threads": int(cv2.getNumThreads()),
        "env": {"LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")},
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("variant", "crops_per_s", "picture_latency_ms", "cpu", "cost")}, indent=1))


if __name__ == "__main__":
    main()
