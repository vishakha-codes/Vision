"""Engine side of the L2 cloud lane (COOL second look) and the L3 agent.

``CloudGlue.from_config(engine)`` builds nothing unless the Mac is set up
for the cloud (``~/.screen-guardian/cloud.json`` / ``SG_L2_*`` with a
token, or an AgentCore runtime ARN). Without it the engine runs exactly as
before: no lane, no agent, ``backend`` = "local".

With it:

- the analyzer gets the ``CloudLane`` (it submits hard pictures and merges
  answers on the vision thread; see ``vision_core.live.analyzer``);
- a cloud answer that is still UNSURE becomes a ``stubborn_unsure`` agent
  event (app = bundle id or process name from the window list; never a
  title or URL; labels are reduced to tiers by the agent client);
- an agent CONFIRM / CLEAR is injected back into the lane (applied on the
  next analyzer call); a returned policy sets the lane's escalation budget;
  the engine's mode becomes ``agent.effective_mode(pin_mode, app)`` for the
  frontmost app, which can only *raise* protection above the parent's PIN
  mode;
- ``heavy_page`` per app every 60 s, ``session_summary`` (reason-code
  counts) every 30 min and on quit, ``flush_pending`` at start;
- every cloud / agent outcome goes to the engine log under ``event``.

``backend`` for the island: "local" (no cloud), "local+COOL" (lane closed /
half-open), "COOL offline" (breaker open).
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from typing import Any, Callable

from vision_core.live.contracts import ProtectionMode, VerdictState

HEAVY_PERIOD_S = 60.0
SUMMARY_PERIOD_S = 30 * 60.0
MODE_CHECK_S = 1.0


def cloud_wanted() -> bool:
    """Off with ``SG_CLOUD=0`` and under pytest (tests inject fakes)."""
    if os.environ.get("SG_CLOUD", "1") == "0":
        return False
    return "PYTEST_CURRENT_TEST" not in os.environ or os.environ.get("SG_CLOUD") == "1"


def _agent_configured() -> bool:
    if os.environ.get("SG_AGENT_RUNTIME_ARN"):
        return True
    try:
        import json
        from pathlib import Path

        cfg = json.loads((Path.home() / ".screen-guardian" / "cloud.json").read_text())
        return bool(cfg.get("agent_runtime_arn"))
    except Exception:
        return False


class CloudGlue:
    def __init__(self, engine: Any, *, lane: Any = None, agent: Any = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.engine = engine
        self.lane = lane
        self.agent = agent
        self.clock = clock
        self._lock = threading.Lock()
        self._bundle: dict[int, str] = {}
        self._apps_of_region: OrderedDict[str, str] = OrderedDict()
        self._seen_regions: OrderedDict[str, None] = OrderedDict()
        self._per_app: dict[str, Counter] = defaultdict(Counter)
        self.reason_counts: Counter = Counter()
        self._t_start = clock()
        self._t_heavy = clock()
        self._t_summary = clock()
        self._t_mode = 0.0
        self._front_app = ""
        self.events: list[dict] = []  # last few, for status / tests
        self._lane_base: dict[str, int] = {}

    # ------------------------------------------------------------ build

    @classmethod
    def from_config(cls, engine: Any) -> "CloudGlue | None":
        if not cloud_wanted():
            return None
        try:
            from vision_core.live.cloud_lane import CloudLane, CloudLaneConfig
        except Exception:
            return None
        glue = cls(engine)
        try:
            cfg = CloudLaneConfig.from_env()
            if cfg.enabled:
                glue.lane = CloudLane(cfg, on_result=glue.on_cloud)
        except Exception as exc:  # noqa: BLE001 - stay local
            glue._event("cloud_config_error", error=f"{type(exc).__name__}: {exc}"[:200])
        if glue.lane is not None or _agent_configured():
            try:
                from agent.agentcore_client import AgentClient

                glue.agent = AgentClient(on_result=glue.on_agent)
            except Exception as exc:  # noqa: BLE001
                glue._event("agent_unavailable", error=f"{type(exc).__name__}: {exc}"[:200])
        if glue.lane is None and glue.agent is None:
            return None
        glue.start()
        return glue

    def start(self) -> None:
        if self.lane is not None and hasattr(self.lane, "warm"):
            try:
                self.lane.warm()  # TLS handshake now, not on the first escalation
            except Exception:  # noqa: BLE001
                pass
        if self.agent is not None:
            try:
                n = self.agent.flush_pending()
                if n:
                    self._event("agent_flush", n=int(n))
            except Exception:  # noqa: BLE001
                pass

    def attach(self, analyzer: Any) -> None:
        if self.lane is not None and analyzer is not None:
            try:
                analyzer.cloud_lane = self.lane
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------ status

    def backend(self, local: str = "local") -> str:
        if self.lane is None:
            return local
        st = getattr(self.lane, "state", "disabled")
        if st == "disabled":
            return local
        problem = str(getattr(self.lane, "problem", "") or "")
        if problem:
            return problem  # "COOL unreachable (IP not allowed?)" / "COOL offline"
        if st == "open":
            return "COOL offline"
        return f"{local}+COOL"

    # ------------------------------------------------------------ apps

    def app_for_window(self, window_id: int | None) -> str:
        """Bundle id (from the owner pid) or the process name; never a title."""
        layout = getattr(getattr(self.engine, "core", None), "layout", None)
        win = layout.window(window_id) if (layout is not None and window_id is not None) else None
        if win is None:
            return "unknown.app"
        pid = int(getattr(win, "owner_pid", 0) or 0)
        if pid and pid not in self._bundle:
            bid = ""
            try:
                from AppKit import NSRunningApplication

                app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
                bid = str(app.bundleIdentifier() or "") if app is not None else ""
            except Exception:  # noqa: BLE001
                bid = ""
            self._bundle[pid] = bid
        bid = self._bundle.get(pid, "")
        return bid or (str(getattr(win, "owner", "") or "").strip().lower().replace(" ", "-") or "unknown.app")

    def front_app(self) -> str:
        layout = getattr(getattr(self.engine, "core", None), "layout", None)
        if layout is None:
            return ""
        for w in layout.windows:
            if w.is_app:
                return self.app_for_window(w.window_id)
        return ""

    # ------------------------------------------------------------ engine hooks

    def on_result(self, result: Any) -> None:
        """After each analyzer result (vision thread): per-app picture /
        reason counts for heavy_page and session_summary."""
        with self._lock:
            for reg in getattr(result, "regions", ()) or ():
                if reg.kind == "tile" or reg.region_id in self._seen_regions:
                    continue
                self._seen_regions[reg.region_id] = None
                while len(self._seen_regions) > 4096:
                    self._seen_regions.popitem(last=False)
                app = self.app_for_window(getattr(reg, "window_id", None))
                self._apps_of_region[reg.region_id] = app
                while len(self._apps_of_region) > 1024:
                    self._apps_of_region.popitem(last=False)
                c = self._per_app[app]
                c["pictures"] += 1
                v = (getattr(result, "verdicts", {}) or {}).get(reg.region_id)
                if v is not None and v.state is VerdictState.UNSURE:
                    c["unsure"] += 1
                if v is not None and v.state is VerdictState.UNSAFE:
                    c["unsafe"] += 1
            new = {m.region_id.split("#")[0]: m.reason for m in getattr(result, "masks", ()) or () if m.reason}
            for rid, reason in new.items():
                key = f"mask:{rid}"
                if key not in self._seen_regions:
                    self._seen_regions[key] = None
                    self.reason_counts[reason] += 1
            sub = (getattr(result, "counts", {}) or {}).get("n_cloud_submitted", 0)
            if sub:
                self.reason_counts["CLOUD_ESCALATED"] += int(sub)

    def tick(self) -> None:
        """Main / capture thread, ~1 Hz is enough."""
        now = self.clock()
        if self.lane is not None:
            problem = str(getattr(self.lane, "problem", "") or "")
            if problem != getattr(self, "_problem", ""):
                self._problem = problem
                if problem:
                    self._event("cloud_unreachable" if "unreachable" in problem else "cloud_offline", text=problem,
                                error=str(getattr(self.lane, "last_error", ""))[:160])
                else:
                    self._event("cloud_back")
        if self.agent is None:
            return
        if now - self._t_mode >= MODE_CHECK_S:
            self._t_mode = now
            app = self.front_app()
            if app:
                self._front_app = app
                self.refresh_mode(app)
        if now - self._t_heavy >= HEAVY_PERIOD_S:
            minutes = max(1e-6, (now - self._t_heavy) / 60.0)
            self._t_heavy = now
            with self._lock:
                per_app, self._per_app = self._per_app, defaultdict(Counter)
            shed = self._lane_shed_delta()
            for app, c in per_app.items():
                if not c["pictures"]:
                    continue
                try:
                    self.agent.heavy_page(app=app, mode_floor=self._pin(), pictures_per_min=round(c["pictures"] / minutes, 2),
                                          escalations_per_min=round(c.get("escalated", 0) / minutes, 2),
                                          unsure_rate=round(c["unsure"] / max(1, c["pictures"]), 3), shed_rate=shed)
                    self._event("agent_heavy_page", app=app, pictures=int(c["pictures"]))
                except Exception:  # noqa: BLE001
                    pass
        if now - self._t_summary >= SUMMARY_PERIOD_S:
            self.send_summary()

    def send_summary(self) -> None:
        if self.agent is None:
            return
        now = self.clock()
        minutes = round((now - self._t_summary) / 60.0, 1)
        self._t_summary = now
        with self._lock:
            counts = dict(self.reason_counts)
            self.reason_counts = Counter()
        try:
            self.agent.session_summary(counts=counts, minutes=minutes, mode_floor=self._pin())
            self._event("agent_session_summary", minutes=minutes, n=sum(counts.values()))
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        self.send_summary()
        if self.lane is not None:
            try:
                self.lane.close()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------ callbacks (worker threads)

    def on_cloud(self, res: Any) -> None:
        """CloudLane worker: one L2 answer (merged into the cache by the analyzer)."""
        self._event("cloud_result", state=res.state, source=res.source, reason=res.reason,
                    rtt_ms=round(float(res.rtt_ms), 1), late=bool(res.late))
        if res.state != "UNSURE" or self.agent is None:
            return
        meta = {}
        an = getattr(self.engine, "analyzer", None)
        try:
            meta = dict((getattr(an, "escalations", {}) or {}).get((int(res.dhash), int(res.ahash)), {}))
        except Exception:  # noqa: BLE001
            meta = {}
        app = self.app_for_window(meta.get("window_id")) if meta else self._apps_of_region.get(res.region_id, "unknown.app")
        labels = tuple(meta.get("labels") or res.labels or ())
        with self._lock:
            self._per_app[app]["escalated"] += 1
        try:
            self.agent.stubborn_unsure(
                app=app, mode=self._mode(), mode_floor=self._pin(), dhash=int(res.dhash), labels=labels,
                l1_score=float(meta.get("l1_score", 0.0)), l2_score=float(res.score), l2_state=str(res.state),
                context={"dhash": int(res.dhash), "ahash": int(res.ahash), "wh": tuple(res.wh), "labels": labels,
                         "app": app, "region_id": res.region_id},
            )
            self._event("agent_stubborn_unsure", app=app)
        except Exception:  # noqa: BLE001
            pass

    def on_agent(self, out: dict, ctx: dict) -> None:
        """AgentClient worker: one decision."""
        verdict = str(out.get("verdict") or "")
        self._event("agent_result", type=str(out.get("type", "")), verdict=verdict, lane=str(out.get("lane", "")))
        if self.lane is not None and verdict in ("CONFIRM", "CLEAR") and ctx.get("dhash") is not None:
            try:
                self.lane.inject_verdict(dhash=int(ctx["dhash"]), ahash=int(ctx.get("ahash", 0)), wh=tuple(ctx.get("wh") or (0, 0)),
                                         verdict=verdict, labels=tuple(ctx.get("labels") or ()),
                                         region_id=str(ctx.get("region_id", "")))
            except Exception:  # noqa: BLE001
                pass
        pol = out.get("policy")
        if isinstance(pol, dict) and self.lane is not None and pol.get("escalation_rate_per_s") is not None:
            try:
                self.lane.set_budget(float(pol["escalation_rate_per_s"]))
            except Exception:  # noqa: BLE001
                pass
        app = str(ctx.get("app") or out.get("app") or "")
        if app and app == (self._front_app or self.front_app()):
            self.refresh_mode(app)

    # ------------------------------------------------------------ mode

    def refresh_mode(self, app: str) -> None:
        if self.agent is None or not hasattr(self.engine, "apply_effective_mode"):
            return
        try:
            want = ProtectionMode(self.agent.effective_mode(self._pin(), app))
        except Exception:  # noqa: BLE001
            return
        if want is not self.engine.mode:
            self.engine.apply_effective_mode(want)
            self._event("mode_raised" if want is not self._pin_mode() else "mode_restored", app=app, mode=want.value)

    # ------------------------------------------------------------ helpers

    def _pin_mode(self) -> ProtectionMode:
        return getattr(self.engine, "pin_mode", None) or self.engine.mode

    def _pin(self) -> str:
        return self._pin_mode().value

    def _mode(self) -> str:
        return self.engine.mode.value

    def _lane_shed_delta(self) -> float:
        if self.lane is None:
            return 0.0
        c = dict(getattr(self.lane, "counts", {}) or {})
        shed = sum(v for k, v in c.items() if k.startswith("shed_") and k != "shed_disabled")
        sub = c.get("submitted", 0)
        prev_shed, prev_sub = self._lane_base.get("shed", 0), self._lane_base.get("sub", 0)
        self._lane_base = {"shed": shed, "sub": sub}
        tried = (shed - prev_shed) + (sub - prev_sub)
        return round((shed - prev_shed) / tried, 3) if tried > 0 else 0.0

    def _event(self, name: str, **fields: Any) -> None:
        ev = {"event": name, **fields}
        self.events.append(ev)
        del self.events[:-32]
        log = getattr(self.engine, "log", None)
        if log is not None:
            try:
                log.record({"mode": self.engine.mode.value, **ev})
            except Exception:  # noqa: BLE001
                pass
