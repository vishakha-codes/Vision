"""Start / stop the kids chat service as its own process (for the engine's launcher).

    from kids_chat.launcher import start_chat_service, stop_chat_service
    proc = start_chat_service()      # None if disabled (SG_CHAT_SERVICE=0) or already running
    ...
    stop_chat_service(proc)

The service is a separate process with its own socket, so the vision engine and the
chat never share a crash, a lock or a socket. Its stderr goes to
``<support dir>/chat-service.log`` (event names and reason codes only, never text).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

from kids_chat.config import home_dir, socket_path

VISION = Path(__file__).resolve().parents[1]


def _listening(path: Path) -> bool:
    if not path.exists():
        return False
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(0.5)
        s.connect(str(path))
        return True
    except OSError:
        return False
    finally:
        s.close()


def start_chat_service(python: str | None = None) -> subprocess.Popen | None:
    if os.environ.get("SG_CHAT_SERVICE", "1") == "0" or _listening(socket_path()):
        return None
    home = home_dir()
    home.mkdir(parents=True, exist_ok=True)
    log = open(home / "chat-service.log", "ab", buffering=0)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(VISION) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.Popen([python or sys.executable, "-m", "kids_chat.service", "--verbose"], cwd=str(VISION), env=env,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log, start_new_session=True)


def stop_chat_service(proc: subprocess.Popen | None, timeout: float = 3.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
