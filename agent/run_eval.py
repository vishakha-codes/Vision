"""CLI alias: ``python -m agent.run_eval`` — same flags as ``python -m vision_core.eval``."""

from __future__ import annotations

from vision_core.eval import main

if __name__ == "__main__":
    raise SystemExit(main())
