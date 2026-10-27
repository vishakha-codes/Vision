"""Token bucket for rate-limited work (second looks, reinspects).

Replaces the old "40 for the whole process" counter: capacity ``burst``,
refilled at ``rate_per_s``. Time is passed in explicitly (seconds) so tests
and replays are deterministic; nothing here reads a clock.

If time goes backwards (a replay restarts, or the caller switched time base,
e.g. a first frame stamped with ``time.monotonic()`` and later ones with a
capture clock starting at 0), the bucket re-bases on the new clock instead of
waiting for the old one to be reached again: ``last_s`` jumps to ``now_s``
and the bucket gets one ``RESYNC_REFILL_S`` worth of tokens, since the real
elapsed time is unknown. Refill then continues normally on the new base.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Tokens credited on a backwards clock jump: as if this many seconds passed.
RESYNC_REFILL_S = 0.25


@dataclass
class TokenBucket:
    rate_per_s: float = 4.0
    burst: float = 4.0
    tokens: float = field(default=-1.0)
    last_s: float | None = None
    resyncs: int = 0  # how many backwards clock jumps were absorbed

    def __post_init__(self) -> None:
        self.rate_per_s = max(0.0, float(self.rate_per_s))
        self.burst = max(0.0, float(self.burst))
        if self.tokens < 0.0:
            self.tokens = self.burst

    def _refill(self, now_s: float) -> None:
        now = float(now_s)
        if not math.isfinite(now):
            return
        if self.last_s is None:
            self.last_s = now
            return
        dt = now - self.last_s
        if dt < 0.0:
            # Clock jumped back / time base changed: re-base, credit a little.
            self.resyncs += 1
            dt = RESYNC_REFILL_S
        if dt > 0.0:
            self.tokens = min(self.burst, self.tokens + dt * self.rate_per_s)
        self.last_s = now

    def available(self, now_s: float) -> float:
        self._refill(now_s)
        return self.tokens

    def can_take(self, now_s: float, n: float = 1.0) -> bool:
        return self.available(now_s) >= n

    def take(self, now_s: float, n: float = 1.0) -> bool:
        """Spend ``n`` tokens if available. Returns whether it did."""
        if not self.can_take(now_s, n):
            return False
        self.tokens -= n
        return True
