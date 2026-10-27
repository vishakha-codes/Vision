"""Launch / stop the native SwiftUI island (vision/island, product GuardianIsland).

The engine imports these. The island is a separate process: if the engine dies,
the island stays up and shows "Not protecting", which is what a guardian wants.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ISLAND_DIR = Path(__file__).resolve().parents[1] / "island"
DEFAULT_BINARY = ISLAND_DIR / ".build" / "release" / "GuardianIsland"
BUILD_HINT = f"cd {ISLAND_DIR} && swift build -c release"


def island_binary() -> Path:
    """Path to the island executable (`SG_ISLAND_BIN` overrides, for tests)."""
    override = os.environ.get("SG_ISLAND_BIN")
    return Path(override) if override else DEFAULT_BINARY


def start_island(sock_path: str | None = None) -> subprocess.Popen | None:
    """Start the island. Returns None (and prints a build hint) if it isn't built."""
    binary = island_binary()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        print(f"[island] GuardianIsland is not built ({binary}). Build it with:\n  {BUILD_HINT}",
              file=sys.stderr)
        return None
    env = os.environ.copy()
    if sock_path:
        env["SG_ISLAND_SOCK"] = str(sock_path)
    return subprocess.Popen(
        [str(binary)],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        start_new_session=True,
    )


def stop_island(proc: subprocess.Popen | None, timeout: float = 3.0) -> None:
    """Terminate the island process (SIGTERM, then SIGKILL after `timeout`)."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
