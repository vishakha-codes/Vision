"""L2 cloud lane: escalate hard pictures to the COOL second look on Graviton.

The vision thread calls :meth:`CloudLane.submit` with a picture crop it could
not settle; it returns at once (never blocks, never raises). One worker
thread JPEG-encodes the crop, POSTs it to ``/v1/second_look`` with a bearer
token and a hard deadline (500 ms), and queues the answer. On its next tick
the vision thread calls :meth:`CloudLane.apply` with the analyzer's
``VerdictCache``: each answer is merged into the cached evidence for that
picture's dHash / aHash / size, so the next ``decide`` sees it (the cache is
thread-confined; only the vision thread touches it).

Failure behaviour: the L1 verdict stays (AGGRESSIVE keeps covering UNSURE /
PENDING pictures). A circuit breaker opens after ``breaker_failures``
consecutive failures or timeouts and sheds every submit for
``breaker_open_s``, then lets one probe through. When the failures are
*connect* failures (the box is down, or this Mac's IP is not in the security
group: ``cloud/deploy/box.sh allow-my-ip``), :attr:`CloudLane.problem` says
"COOL unreachable (IP not allowed?)".

Warm connection: :meth:`CloudLane.warm` (called when the engine starts the
lane) opens the TLS connection with a ``/health`` request, and the worker
pings ``/health`` every ``keepalive_s`` while idle (the server keeps idle
connections ``--timeout-keep-alive 75`` s), so an escalation never pays the
TLS handshake.

Privacy: only the picture crop is sent (no titles, URLs, window or app
names); the server keeps it in RAM and never writes it to disk.

Config (no hard-coded endpoint or secret), env first, then
``~/.screen-guardian/cloud.json``:

- ``SG_L2_URL`` / ``l2_url``: e.g. ``https://<box-ip>:8443``
- ``SG_L2_TOKEN``, else the macOS Keychain item ``screen-guardian-l2-token``
- ``SG_L2_CERT`` / ``l2_cert``: pinned self-signed cert (default
  ``~/.screen-guardian/l2-cert.pem``); hostname is not checked, the chain is
- ``SG_L2_TIMEOUT_MS`` (500), ``SG_L2_RATE`` (3/s), ``SG_L2_BURST`` (3)
"""

from __future__ import annotations

import base64
import json
import os
import queue
import ssl
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from vision_core.live.modes import Evidence, NormDet

CONFIG_PATH = Path.home() / ".screen-guardian" / "cloud.json"
DEFAULT_CERT = Path.home() / ".screen-guardian" / "l2-cert.pem"
KEYCHAIN_SERVICE = "screen-guardian-l2-token"

# Escalation reasons, highest priority first (see the hook spec in the report).
REASON_LOCAL_UNAVAILABLE = "local_unavailable"
REASON_UNSURE = "unsure_after_local"
REASON_LARGE_WEAK = "large_weak"
_PRIORITY = {REASON_LOCAL_UNAVAILABLE: 0, REASON_UNSURE: 1, REASON_LARGE_WEAK: 2}

# Agent verdicts (L3) folded into evidence.
AGENT_CONFIRM = "CONFIRM"
AGENT_CLEAR = "CLEAR"
AGENT_LABEL = "AGENT_CONFIRMED"


def _keychain_token() -> str:
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=2.0,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


@dataclass
class CloudLaneConfig:
    url: str = ""
    token: str = ""
    cert_path: str = ""
    timeout_s: float = 0.5
    queue_max: int = 8
    rate_per_s: float = 3.0
    burst: float = 3.0
    breaker_failures: int = 3
    breaker_open_s: float = 15.0
    jpeg_quality: int = 85
    max_edge: int = 1024
    min_edge: int = 32
    resend_after_s: float = 600.0
    keepalive_s: float = 30.0  # idle /health ping (server keep-alive is 75 s)

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.token)

    @classmethod
    def from_env(cls, path: Path | None = None) -> "CloudLaneConfig":
        raw: dict[str, Any] = {}
        p = path or CONFIG_PATH
        try:
            if p.is_file():
                raw = json.loads(p.read_text())
        except Exception:
            raw = {}
        env = os.environ
        cert = env.get("SG_L2_CERT") or raw.get("l2_cert") or (str(DEFAULT_CERT) if DEFAULT_CERT.is_file() else "")
        return cls(
            url=(env.get("SG_L2_URL") or raw.get("l2_url") or "").rstrip("/"),
            token=env.get("SG_L2_TOKEN") or raw.get("l2_token") or _keychain_token(),
            cert_path=cert,
            timeout_s=float(env.get("SG_L2_TIMEOUT_MS") or raw.get("l2_timeout_ms") or 500) / 1000.0,
            rate_per_s=float(env.get("SG_L2_RATE") or raw.get("l2_rate_per_s") or 3.0),
            burst=float(env.get("SG_L2_BURST") or raw.get("l2_burst") or 3.0),
        )


@dataclass
class CloudResult:
    """One answer for one picture, ready to merge into the evidence cache."""

    dhash: int
    ahash: int
    wh: tuple[float, float]
    region_id: str
    reason: str
    source: str  # "cloud" | "cloud_cache" | "agent"
    state: str  # server verdict for its mode: SAFE | UNSURE | UNSAFE
    score: float
    labels: tuple[str, ...]
    dets: tuple[NormDet, ...]
    latency_ms: float  # submit → answer (queue wait + network + server)
    rtt_ms: float  # HTTP round trip only
    server_ms: float
    n_views: int = 0
    opencv: str = ""
    late: bool = False
    agent_verdict: str = ""


@dataclass(order=True)
class _Job:
    priority: int
    seq: int
    region_id: str = field(compare=False)
    dhash: int = field(compare=False)
    ahash: int = field(compare=False)
    wh: tuple[float, float] = field(compare=False)
    crop: np.ndarray = field(compare=False, repr=False)
    mode: str = field(compare=False)
    reason: str = field(compare=False)
    t_submit: float = field(compare=False)


class _Bucket:
    def __init__(self, rate: float, burst: float) -> None:
        self.rate, self.burst = float(rate), float(burst)
        self.tokens, self.t = float(burst), time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


def merge_evidence(base: Evidence | None, res: CloudResult) -> Evidence:
    """Cloud / agent answer → evidence for ``modes.decide``.

    The cloud look replaces the second-look slot (``dets2``): a clean cloud
    look can clear a local UNSURE, but local first-pass hits at or above a
    mode's ON threshold still make the picture UNSAFE (fail closed).
    CONFIRM adds a whole-picture box at score 1.0 under the top label of the
    UNSURE it confirms (a class of the mode, so ``decide`` → UNSAFE; with no
    label it is ``AGENT_CONFIRMED``, which no mode counts); CLEAR is a clean
    look (local hits above ON still win).
    """
    dets = base.dets if base is not None else ()
    lat = (base.latency_ms if base is not None else 0.0) + res.latency_ms
    n = (base.n_crops if base is not None else 0) + res.n_views
    if res.agent_verdict == AGENT_CONFIRM:
        top = res.labels[0] if res.labels else AGENT_LABEL
        return Evidence(dets, (NormDet(top, 1.0, (0.0, 0.0, 1.0, 1.0)),), lat, n)
    if res.agent_verdict == AGENT_CLEAR:
        return Evidence(dets, (), lat, n)
    return Evidence(dets, tuple(res.dets), lat, n)


class CloudLane:
    """See module docstring. Thread-safe ``submit`` / ``apply`` / ``stats``."""

    def __init__(
        self,
        config: CloudLaneConfig | None = None,
        *,
        on_result: Callable[[CloudResult], None] | None = None,
        transport: Callable[[dict, float], dict] | None = None,
    ) -> None:
        self.cfg = config or CloudLaneConfig.from_env()
        self.on_result = on_result  # worker thread; e.g. send stubborn_unsure to the agent
        self._transport = transport or self._http_post
        self._q: queue.PriorityQueue[_Job] = queue.PriorityQueue()
        self._results: list[CloudResult] = []
        self._lock = threading.Lock()
        self._inflight: set[tuple[int, int]] = set()
        self._recent: OrderedDict[tuple[int, int], float] = OrderedDict()
        self._bucket = _Bucket(self.cfg.rate_per_s, self.cfg.burst)
        self._fails = 0
        self._open_until = 0.0
        self._probing = False
        self._seq = 0
        self._client = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.counts: dict[str, int] = {
            "submitted": 0, "shed_disabled": 0, "shed_breaker": 0, "shed_rate": 0, "shed_dup": 0,
            "shed_queue": 0, "ok": 0, "failed": 0, "timeouts": 0, "late": 0, "flips": 0, "applied": 0,
            "breaker_opens": 0,
        }
        self.latencies_ms: list[float] = []
        self._last_io = 0.0
        self._connect_fails = 0  # consecutive connect-type failures (requests or pings)
        self.last_error = ""
        self.counts["pings"] = 0
        self.counts["ping_failed"] = 0

    # ------------------------------------------------------------ state

    @property
    def state(self) -> str:
        if not self.cfg.enabled:
            return "disabled"
        now = time.monotonic()
        if now < self._open_until:
            return "open"
        return "half_open" if self._open_until and self._fails >= self.cfg.breaker_failures else "closed"

    @property
    def problem(self) -> str:
        """Human text when the cloud is not usable ("" when it is)."""
        if not self.cfg.enabled:
            return ""
        if self._connect_fails >= max(1, self.cfg.breaker_failures) or (self.state == "open" and self._connect_fails):
            return "COOL unreachable (IP not allowed?)"
        if self.state == "open":
            return "COOL offline"
        return ""

    def warm(self) -> None:
        """Open the TLS connection now (worker thread; returns at once)."""
        if not self.cfg.enabled:
            return
        self._last_io = 0.0  # the worker pings on its first idle tick
        self._ensure_worker()

    def set_budget(self, rate_per_s: float, burst: float | None = None) -> None:
        """Escalation budget (the L3 agent may tune it per app/site)."""
        with self._lock:
            self._bucket = _Bucket(max(0.0, rate_per_s), burst if burst is not None else max(1.0, rate_per_s))

    def stats(self) -> dict[str, Any]:
        lat = sorted(self.latencies_ms[-500:])
        pick = lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 1) if lat else 0.0  # noqa: E731
        return {**self.counts, "state": self.state, "queued": self._q.qsize(), "p50_ms": pick(0.5), "p95_ms": pick(0.95)}

    # ------------------------------------------------------------ vision thread

    def submit(
        self,
        *,
        region_id: str,
        dhash: int,
        ahash: int,
        wh: tuple[float, float],
        crop_bgr: np.ndarray,
        mode: str,
        reason: str = REASON_UNSURE,
    ) -> bool:
        """Queue one picture. Returns False when shed (disabled, breaker open,
        over budget, duplicate, or full queue). Copies the crop."""
        try:
            if not self.cfg.enabled:
                self.counts["shed_disabled"] += 1
                return False
            if crop_bgr is None or crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < self.cfg.min_edge:
                return False
            key = (int(dhash), int(ahash))
            now = time.monotonic()
            with self._lock:
                if now < self._open_until or (self._probing and self._fails >= self.cfg.breaker_failures):
                    self.counts["shed_breaker"] += 1
                    return False
                t_done = self._recent.get(key)
                if key in self._inflight or (t_done is not None and now - t_done < self.cfg.resend_after_s):
                    self.counts["shed_dup"] += 1
                    return False
                if self._q.qsize() >= self.cfg.queue_max:
                    self.counts["shed_queue"] += 1
                    return False
                if not self._bucket.take():
                    self.counts["shed_rate"] += 1
                    return False
                if self._fails >= self.cfg.breaker_failures:
                    self._probing = True  # half-open: this one is the probe
                self._inflight.add(key)
                self._seq += 1
                job = _Job(_PRIORITY.get(reason, 3), self._seq, region_id, key[0], key[1],
                           (float(wh[0]), float(wh[1])), np.ascontiguousarray(crop_bgr).copy(), mode, reason, now)
            self._q.put(job)
            self.counts["submitted"] += 1
            self._ensure_worker()
            return True
        except Exception:
            return False

    def drain(self) -> list[CloudResult]:
        with self._lock:
            out, self._results = self._results, []
        return out

    def apply(self, cache, now_ms: float, *, decide_fn: Callable[[Evidence], str] | None = None) -> list[CloudResult]:
        """Merge finished answers into the analyzer's ``VerdictCache`` (call on
        the vision thread). ``decide_fn(evidence) -> state`` (optional) counts
        verdict flips for logs/metrics."""
        done = self.drain()
        for res in done:
            base = cache.get(res.dhash, res.ahash, res.wh, now_ms)
            base_ev = base if isinstance(base, Evidence) else None
            merged = merge_evidence(base_ev, res)
            if decide_fn is not None and base_ev is not None:
                try:
                    if decide_fn(base_ev) != decide_fn(merged):
                        self.counts["flips"] += 1
                except Exception:
                    pass
            cache.put(res.dhash, res.ahash, res.wh, merged, now_ms)
            self.counts["applied"] += 1
        return done

    def inject_verdict(self, *, dhash: int, ahash: int, wh: tuple[float, float], verdict: str,
                       labels: tuple[str, ...] = (), region_id: str = "") -> None:
        """Queue an L3 agent CONFIRM / CLEAR for the next :meth:`apply`."""
        if verdict not in (AGENT_CONFIRM, AGENT_CLEAR):
            return
        res = CloudResult(int(dhash), int(ahash), (float(wh[0]), float(wh[1])), region_id, "agent", "agent",
                          "UNSAFE" if verdict == AGENT_CONFIRM else "SAFE", 1.0 if verdict == AGENT_CONFIRM else 0.0,
                          tuple(labels), (), 0.0, 0.0, 0.0, agent_verdict=verdict)
        with self._lock:
            self._results.append(res)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass

    # ------------------------------------------------------------ worker

    def _ensure_worker(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="sg-cloud-lane", daemon=True)
            self._thread.start()

    def _encode(self, crop: np.ndarray) -> str:
        h, w = crop.shape[:2]
        scale = min(1.0, self.cfg.max_edge / float(max(h, w)))
        if scale < 1.0:
            crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        if crop.ndim == 3 and crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg.jpeg_quality)])
        if not ok:
            raise ValueError("jpeg encode failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _ssl_context(self) -> ssl.SSLContext | bool:
        if not self.cfg.url.startswith("https"):
            return True
        if self.cfg.cert_path:
            ctx = ssl.create_default_context(cafile=self.cfg.cert_path)
            ctx.check_hostname = False  # pinned self-signed cert; the box IP changes on restart
            return ctx
        return True

    def _http(self):
        import httpx

        if self._client is None:
            self._client = httpx.Client(
                verify=self._ssl_context(),
                headers={"Authorization": f"Bearer {self.cfg.token}"},
                http2=False,
                limits=httpx.Limits(max_keepalive_connections=2, keepalive_expiry=self.cfg.keepalive_s * 3),
            )
        return self._client

    def _http_post(self, body: dict, timeout_s: float) -> dict:
        import httpx

        resp = self._http().post(f"{self.cfg.url}/v1/second_look", json=body, timeout=httpx.Timeout(timeout_s))
        resp.raise_for_status()
        return resp.json()

    def _ping(self) -> None:
        """Keep the TLS connection warm (only with the real HTTP transport)."""
        if self._transport != self._http_post:
            return
        import httpx

        self._last_io = time.monotonic()
        try:
            self._http().get(f"{self.cfg.url}/health", timeout=httpx.Timeout(3.0, connect=2.0))
            self.counts["pings"] += 1
            self._connect_fails = 0
            self.last_error = ""
        except Exception as exc:  # noqa: BLE001
            self.counts["ping_failed"] += 1
            if self._is_connect_error(exc):
                self._connect_fails += 1
            self.last_error = f"{type(exc).__name__}: {str(exc)[:120]}"

    @staticmethod
    def _is_connect_error(exc: BaseException) -> bool:
        name = type(exc).__name__.lower()
        return "connect" in name or isinstance(exc, (ConnectionError, OSError))

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.5)
            except queue.Empty:
                if time.monotonic() - self._last_io >= self.cfg.keepalive_s:
                    self._ping()
                continue
            key = (job.dhash, job.ahash)
            ok = False
            try:
                body = {
                    "jpeg_b64": self._encode(job.crop),
                    "dhash": f"{job.dhash:016x}",
                    "ahash": f"{job.ahash:016x}",
                    "mode": job.mode,
                    "profile": "l2",
                    "picture_id": job.region_id[:64],
                    "reason": job.reason[:32],
                }
                job.crop = None  # drop the pixels as soon as they are encoded
                t0 = time.monotonic()
                self._last_io = t0
                out = self._transport(body, self.cfg.timeout_s)
                t1 = time.monotonic()
                self._connect_fails = 0
                rtt = (t1 - t0) * 1000.0
                late = rtt > self.cfg.timeout_s * 1000.0
                v = out.get("verdict") or {}
                res = CloudResult(
                    job.dhash, job.ahash, job.wh, job.region_id, job.reason, str(out.get("source", "cloud")),
                    str(v.get("state", "")), float(v.get("score", 0.0)), tuple(v.get("labels") or ()),
                    tuple(NormDet(str(d["label"]), float(d["score"]), tuple(float(x) for x in d["box"]))
                          for d in out.get("dets") or ()),
                    (t1 - job.t_submit) * 1000.0, rtt, float((out.get("timings_ms") or {}).get("total", 0.0)),
                    int(out.get("n_views", 0)), str((out.get("opencv") or {}).get("version", "")), late,
                )
                with self._lock:
                    self._results.append(res)
                self.latencies_ms.append(rtt)
                self.counts["ok"] += 1
                if late:
                    self.counts["late"] += 1
                ok = not late
                if self.on_result is not None:
                    try:
                        self.on_result(res)
                    except Exception:
                        pass
            except Exception as exc:
                name = type(exc).__name__.lower()
                self.counts["timeouts" if "timeout" in name else "failed"] += 1
                if self._is_connect_error(exc):
                    self._connect_fails += 1
                self.last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
            finally:
                with self._lock:
                    self._inflight.discard(key)
                    if ok:
                        self._recent[key] = time.monotonic()
                        while len(self._recent) > 4096:
                            self._recent.popitem(last=False)
                    self._record(ok)

    def _record(self, ok: bool) -> None:
        """Breaker bookkeeping (lock held)."""
        self._probing = False
        if ok:
            self._fails = 0
            self._open_until = 0.0
            return
        self._fails += 1
        if self._fails >= self.cfg.breaker_failures:
            self._open_until = time.monotonic() + self.cfg.breaker_open_s
            self.counts["breaker_opens"] += 1
