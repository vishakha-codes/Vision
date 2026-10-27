import json
from dataclasses import asdict

import numpy as np

from vision_core.contracts import make_trace, trace_to_dict


def test_traces_never_persist_images() -> None:
    pixels = np.full((16, 16, 3), 90, dtype=np.uint8)
    trace = make_trace(
        frame_id=4,
        tool="reinspect_crop",
        observation={"score": 0.61, "oops": pixels, "image_bgr": pixels},
        decision="uncertain",
        next_action="policy.evaluate",
        latency_ms=1.2,
    )
    assert trace.image_persisted is False
    payload = trace_to_dict(trace)
    assert payload["image_persisted"] is False
    dumped = json.dumps(payload)
    assert "image_bgr" not in dumped
    assert not isinstance(trace.observation.get("oops"), np.ndarray)
    raw = json.dumps(asdict(trace), default=str)
    assert "image_persisted" in raw
    # Raw pixel dumps would be thousands of ints; we only store a tiny meta object.
    assert dumped.count("90") < 8
