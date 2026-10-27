"""Structured logs + CloudWatch metrics, batched off the request path.

Every request logs one JSON line to stdout (journald on the box). Metrics
are aggregated in RAM and flushed with ``PutMetricData`` every
``FLUSH_S`` as statistic sets (namespace ``ScreenGuardian``), so a request
never waits on CloudWatch. Disabled when ``SG_METRICS=0`` or boto3 has no
credentials (tests, laptop).

Metrics (dimension ``Service``): ``SecondLookLatencyMs``, ``SecondLookInferMs``,
``SecondLookRequests``, ``SecondLookCacheHits``, ``VerdictUnsure``,
``VerdictUnsafe``, ``ReviewRequests``, ``AuthFailures``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import defaultdict

NAMESPACE = "ScreenGuardian"
FLUSH_S = 30.0


def log(event: str, **fields) -> None:
    rec = {"ts": round(time.time(), 3), "event": event, **fields}
    sys.stdout.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")
    sys.stdout.flush()


class Metrics:
    def __init__(self, service: str, *, enabled: bool | None = None) -> None:
        self.service = service
        if enabled is None:
            enabled = os.environ.get("SG_METRICS", "1") != "0"
        self.enabled = enabled
        self._lock = threading.Lock()
        self._stats: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._thread: threading.Thread | None = None
        self._client = None

    def put(self, name: str, value: float, unit: str = "Count") -> None:
        if not self.enabled:
            return
        with self._lock:
            self._stats[(name, unit)].append(float(value))
        if self._thread is None:
            self._start()

    def _start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._loop, name="sg-metrics", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while True:
            time.sleep(FLUSH_S)
            try:
                self.flush()
            except Exception as exc:  # never break serving over metrics
                log("metrics_flush_failed", error=str(exc)[:200])

    def flush(self) -> int:
        with self._lock:
            stats, self._stats = self._stats, defaultdict(list)
        if not stats:
            return 0
        if self._client is None:
            import boto3

            self._client = boto3.client("cloudwatch", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        data = []
        for (name, unit), values in stats.items():
            data.append(
                {
                    "MetricName": name,
                    "Dimensions": [{"Name": "Service", "Value": self.service}],
                    "Unit": unit,
                    "StatisticValues": {
                        "SampleCount": float(len(values)),
                        "Sum": float(sum(values)),
                        "Minimum": float(min(values)),
                        "Maximum": float(max(values)),
                    },
                }
            )
        for i in range(0, len(data), 20):
            self._client.put_metric_data(Namespace=NAMESPACE, MetricData=data[i : i + 20])
        return len(data)
