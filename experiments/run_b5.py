#!/usr/bin/env python3
"""B5 wrapper: evaluate_fixture (Balanced / B3) + instance metadata."""
from __future__ import annotations

import json
import os
import platform
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402

from vision_core.eval import evaluate_fixture  # noqa: E402

FIXTURE = _ROOT / "fixtures" / "safe" / "synthetic_moving_regions.mp4"
GT_PATH = _ROOT / "fixtures" / "safe" / "synthetic_moving_regions.gt.json"
POLICY_PATH = _ROOT / "policies" / "balanced.json"

INSTANCE_ID = ""  # read from IMDS on the box
INSTANCE_TYPE = "c8g.large"
AMI_ID = "ami-01db31139bc5615d8"
AZ = "us-east-1d"
PUBLIC_IP = ""  # read from IMDS on the box


def _imds(path: str) -> str | None:
    try:
        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            token = resp.read().decode()
        req = urllib.request.Request(
            f"http://169.254.169.254/latest/meta-data/{path}",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.read().decode().strip()
    except Exception:
        return None


def main() -> int:
    variant = sys.argv[1] if len(sys.argv) > 1 else "cool"
    if variant not in {"cool", "stock"}:
        raise SystemExit(f"usage: run_b5.py cool|stock, got {variant}")
    out_dir = _ROOT / "experiments" / f"b5_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    instance_id = _imds("instance-id") or INSTANCE_ID
    instance_type = _imds("instance-type") or INSTANCE_TYPE
    ami_id = _imds("ami-id") or AMI_ID
    az = _imds("placement/availability-zone") or AZ
    public_ip = _imds("public-ipv4") or PUBLIC_IP

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    results = evaluate_fixture(FIXTURE, GT_PATH, POLICY_PATH, out_dir)
    wall_s = time.perf_counter() - t0
    finished = datetime.now(timezone.utc).isoformat()

    timings = results.get("timings_ms") or {}
    stage_means = {k: v.get("mean") for k, v in timings.items() if isinstance(v, dict)}
    per_frame_mean = sum(stage_means.values()) if stage_means else None

    enriched = dict(results)
    enriched.update(
        {
            "b5_variant": variant,
            "opencv_version": cv2.__version__,
            "opencv_package": os.environ.get("B5_OPENCV_PACKAGE") or None,
            "instance_id": instance_id,
            "instance_type": instance_type,
            "ami": ami_id,
            "availability_zone": az,
            "public_ip": public_ip,
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "cool_path": os.environ.get("B5_COOL_PATH"),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
            "wall_clock_s": wall_s,
            "per_frame_stage_mean_ms": per_frame_mean,
            "started_utc": started,
            "finished_utc": finished,
            "policy": "balanced",
            "source": "vision_core.eval.evaluate_fixture + agent.orchestrator (Balanced / B3)",
            "notes": (
                "Same fixture, same Balanced policy, same c8g.large. "
                + (
                    "COOL AMI cv2 at /opt/cool (4.14.0-pre, NOT OpenCV 5)."
                    if variant == "cool"
                    else "Stock pip opencv-python in a separate venv (not /opt/cool)."
                )
            ),
        }
    )
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(enriched, indent=2) + "\n")
    print(json.dumps(enriched, indent=2))
    print(f"wrote {out_path} wall_clock_s={wall_s:.3f} opencv={cv2.__version__}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
