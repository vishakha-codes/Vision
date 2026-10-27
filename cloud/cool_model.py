"""Make NudeNet 320n loadable by COOL's classic OpenCV 4.x ONNX importer.

Why: COOL on the Graviton4 AMI is OpenCV **4.14.0-pre**. Its classic ONNX
importer rejects the stock ``320n.onnx`` (dynamic ``batch × 3 × H × W``):

    Concat (/model.11/Concat): Inconsistent shape for ConcatLayer

The graph is fine once its shapes are fixed. This script takes the
``320n.onnx`` that ships inside the installed ``nudenet`` wheel (nothing is
downloaded), pins the input to ``B × 3 × 320 × 320`` and runs onnx-simplifier
(constant-folds the shape arithmetic that the classic importer cannot do).
Same weights, same ops; parity with the original graph is checked by
``experiments/cool_live/parity.py`` (stock cv2 5 / onnxruntime vs COOL).

Run on any machine with ``onnx`` + ``onnxsim`` (a scratch venv is enough):

    python cloud/cool_model.py --src <nudenet>/320n.onnx --out-dir .models --batch 1 8

Outputs ``320n_static_b{B}.onnx`` (never ``320n.onnx``: that name is the
stock model ``resolve_model_path`` prefers).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def make_static(src: Path, out: Path, batch: int, size: int = 320) -> Path:
    import onnx
    from onnxsim import simplify

    model = onnx.load(str(src))
    name = model.graph.input[0].name
    simplified, ok = simplify(model, overwrite_input_shapes={name: [batch, 3, size, size]})
    if not ok:
        raise RuntimeError("onnxsim could not validate the simplified graph")
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(simplified, str(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="nudenet wheel 320n.onnx")
    ap.add_argument("--out-dir", type=Path, default=Path(".models"))
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--size", type=int, default=320)
    args = ap.parse_args()
    for b in args.batch:
        path = make_static(args.src, args.out_dir / f"320n_static_b{b}.onnx", b, args.size)
        print(path)


if __name__ == "__main__":
    main()
