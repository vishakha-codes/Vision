from __future__ import annotations

import sys
from pathlib import Path

import pytest

VISION_ROOT = Path(__file__).resolve().parents[1]
if str(VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(VISION_ROOT))


@pytest.fixture(autouse=True)
def _cwd_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(VISION_ROOT)
    monkeypatch.setenv("SG_ENGINE_LOG", "off")  # never write test events into the real engine.log
