"""Verify / copy the bundled NudeNet 320n weights into vision/.models/.

The 320n ONNX file ships inside the nudenet wheel. This command does not
download porn, does not fetch a huge extra blob, and never writes captures.
"""

from __future__ import annotations

import shutil
import sys

from vision_core.detect_nsfw import MODELS_DIR, resolve_model_path


def main() -> int:
    src = resolve_model_path()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / src.name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    print(f"nsfw weights ready: {dest} ({dest.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
