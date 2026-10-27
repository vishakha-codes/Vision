"""Explain-my-screen smoke against real Bedrock (+ the deployed runtime with --agentcore).

    cd vision && .venv/bin/python -m kids_chat.scripts.smoke_explain [--agentcore] [--save /tmp/x.png]

Uses the synthetic lever diagram from FakeSnapshotProvider (drawn in code, one painted
"covered" box), so no real screen is sent. Prints the answer, the draw ops mapped to
Quartz points, and whether any mark touched the covered box. --save renders the marks
onto the picture for a visual check.
"""

from __future__ import annotations

import argparse
import io
import sys
import uuid
from datetime import datetime

from kids_chat import draw
from kids_chat.agent import screen_tools
from kids_chat.backends import AgentCoreBackend, LocalBackend, load_outputs
from kids_chat.snapshot import FakeSnapshotProvider


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agentcore", action="store_true")
    ap.add_argument("--question", default="How does this work?")
    ap.add_argument("--save", default="")
    ap.add_argument("--model", default="", help="override the explain model (local path only)")
    args = ap.parse_args()
    o = load_outputs()
    b = AgentCoreBackend(o["runtime_arn"]) if args.agentcore else LocalBackend(o)
    if args.model and not args.agentcore:
        b.deps().explain_model = args.model
    snap = FakeSnapshotProvider().get()
    actor = "smoke-" + uuid.uuid4().hex
    payload = {"action": "chat", "prompt": args.question, "actor_id": actor, "age_band": "kids",
               "session_id": f"{actor}-{datetime.now():%Y%m%d}-1", "explain": snap.payload()}
    text, ops, done = "", [], {}
    for ev in b.turn(payload):
        if ev["type"] == "delta":
            text += ev["text"]
        elif ev["type"] == "draw":
            ops += ev["ops"]
        elif ev["type"] in ("done", "error"):
            done = ev
    print("answer:", text.strip())
    print("done:", {k: v for k, v in done.items() if k != "usage"})
    print("ops (grid):", ops)
    print("ops (quartz):", draw.to_screen(ops, snap.display, snap.masks_norm))
    bad = [op for op in ops if screen_tools.touches_mask(op, snap.masks_norm)]
    print("marks on covered box:", len(bad))
    if args.save:
        from PIL import Image, ImageDraw
        im = Image.open(io.BytesIO(snap.frames[0])).convert("RGB")
        d = ImageDraw.Draw(im)
        sx, sy = snap.w / 1000, snap.h / 1000
        for op in ops:
            k = op["kind"]
            if k == "pointer":
                x, y = op["at"]; d.ellipse([x*sx-8, y*sy-8, x*sx+8, y*sy+8], outline=(220, 0, 0), width=4)
            elif k == "box":
                x, y, w, h = op["rect"]; d.rectangle([x*sx, y*sy, (x+w)*sx, (y+h)*sy], outline=(220, 0, 0), width=4)
            elif k == "circle":
                (x, y), r = op["center"], op["r"]; d.ellipse([x*sx-r*sx, y*sy-r*sx, x*sx+r*sx, y*sy+r*sx], outline=(220, 0, 0), width=4)
            elif k == "arrow":
                (a, c), (e, f) = op["from"], op["to"]; d.line([a*sx, c*sy, e*sx, f*sy], fill=(220, 0, 0), width=4)
            if op.get("label") or op.get("text"):
                p = op.get("at") or op.get("center") or op.get("to") or op.get("rect", [0, 0])[:2]
                d.text((p[0]*sx + 10, p[1]*sy - 20), op.get("label") or op.get("text"), fill=(0, 0, 200))
        im.save(args.save)
        print("saved", args.save)
    return 0 if ops and not bad and not done.get("blocked") else 1


if __name__ == "__main__":
    sys.exit(main())
