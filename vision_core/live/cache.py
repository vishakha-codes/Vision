"""Perceptual-hash verdict cache: a scrolled or revisited picture costs ~0 ms.

Hashes are computed here (``cv2.img_hash`` is not in this wheel): the crop
is reduced once to 32×32 gray, then

- dHash (64 bit): 9×8 area resize, sign of the horizontal gradient;
- aHash (64 bit): 8×8 area resize, above / below the mean.

Lookup is exact first, then a Hamming search over a banded index (8 bands
of 8 bits: two hashes within ``max_hamming`` ≤ 7 bits must share a band, by
pigeonhole). Both hashes must match within tolerance. Entries are LRU with a
TTL, and size-bucket aware: SAFE / UNSURE evidence from a 120 px thumbnail
is not reused for the same picture shown at 900 px (the detector sees more
there), but UNSAFE evidence is reused at any size (fail closed).

Values are opaque to the cache (the analyzer stores mode-independent
detector evidence, so a mode switch needs no re-inference).

Per-entry overrides (``put(..., scope=, ttl_ms=, any_size=, refresh=)``): the
analyzer stores regions over ~20 % of the screen **scoped to their window**,
with a short TTL that is not refreshed by hits and without any-size reuse,
so a layout-dominated hash (wallpaper, window chrome) cannot bring an old
verdict back after the page changes.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

_BANDS = 8
_MASK64 = (1 << 64) - 1


def gray32(bgr: np.ndarray) -> np.ndarray:
    """Area-resize to 32×32 (colour first, so the big crop is touched once)."""
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
    h, w = bgr.shape[:2]
    k = min(h, w) // 64
    if k >= 2:
        # Exact integer area step first (non-integer INTER_AREA on a large
        # crop is ~10× slower); same pixels in → same hash out.
        bgr = cv2.resize(bgr[: (h // k) * k, : (w // k) * k], (w // k, h // k), interpolation=cv2.INTER_AREA)
    small = cv2.resize(bgr, (32, 32), interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small


def _bits_to_int(bits: np.ndarray) -> int:
    out = 0
    for bit in bits.reshape(-1)[:64]:
        out = (out << 1) | int(bool(bit))
    return out


def dhash64(g32: np.ndarray) -> int:
    small = cv2.resize(g32, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return _bits_to_int(small[:, 1:] > small[:, :-1])


def ahash64(g32: np.ndarray) -> int:
    small = cv2.resize(g32, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
    return _bits_to_int(small > float(small.mean()))


def hash_crop(bgr: np.ndarray) -> tuple[int, int]:
    """``(dhash, ahash)`` of a BGR crop. Empty crops hash to ``(0, 0)``."""
    if bgr is None or bgr.size == 0:
        return 0, 0
    g = gray32(bgr)
    return dhash64(g), ahash64(g)


def hamming(a: int, b: int) -> int:
    return ((a ^ b) & _MASK64).bit_count()


def size_bucket(w: float, h: float) -> tuple[int, int]:
    """Half-octave buckets of width / height (log2 × 2)."""
    return (
        int(round(2.0 * math.log2(max(1.0, float(w))))),
        int(round(2.0 * math.log2(max(1.0, float(h))))),
    )


def _bands(h: int) -> list[int]:
    return [(h >> (8 * i)) & 0xFF for i in range(_BANDS)]


@dataclass
class CacheEntry:
    key: int
    dhash: int
    ahash: int
    bucket: tuple[int, int]
    value: Any
    created_ms: float
    seen_ms: float
    hits: int = 0
    aspect: float = 0.0  # w / h of the crop the value was computed on
    scope: object = None  # lookups must name the same scope (None = global)
    ttl_ms: float | None = None  # None = the cache default
    any_size: bool = True  # may ``reuse_any_size`` apply to this entry
    refresh: bool = True  # hits extend the entry's life


class VerdictCache:
    """LRU + TTL cache keyed by perceptual hash and size bucket."""

    def __init__(
        self,
        *,
        max_entries: int = 1024,
        ttl_ms: float = 15 * 60 * 1000.0,
        max_hamming: int = 6,
        bucket_slack: int = 1,
        aspect_tol: float = 0.03,
        reuse_any_size: Callable[[Any], bool] | None = None,
    ) -> None:
        self.max_entries = max(1, int(max_entries))
        self.ttl_ms = float(ttl_ms)
        self.max_hamming = max(0, min(7, int(max_hamming)))
        self.bucket_slack = max(0, int(bucket_slack))
        # Values may hold boxes normalised to the crop, so a crop with another
        # shape (the same picture half scrolled out) must not reuse them.
        self.aspect_tol = float(aspect_tol)
        # Predicate: may this value be reused at any size? (UNSAFE evidence.)
        self.reuse_any_size = reuse_any_size or (lambda _v: False)
        self._entries: OrderedDict[int, CacheEntry] = OrderedDict()
        self._index: list[dict[int, set[int]]] = [dict() for _ in range(_BANDS)]
        self._next_key = 0
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._index = [dict() for _ in range(_BANDS)]

    def _unindex(self, entry: CacheEntry) -> None:
        for i, band in enumerate(_bands(entry.dhash)):
            keys = self._index[i].get(band)
            if keys is not None:
                keys.discard(entry.key)
                if not keys:
                    del self._index[i][band]

    def _drop(self, key: int) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._unindex(entry)

    def _bucket_ok(self, entry: CacheEntry, bucket: tuple[int, int]) -> bool:
        if (
            abs(entry.bucket[0] - bucket[0]) <= self.bucket_slack
            and abs(entry.bucket[1] - bucket[1]) <= self.bucket_slack
        ):
            return True
        return entry.any_size and bool(self.reuse_any_size(entry.value))

    def _expired(self, entry: CacheEntry, now_ms: float) -> bool:
        ttl = self.ttl_ms if entry.ttl_ms is None else entry.ttl_ms
        since = entry.seen_ms if entry.refresh else entry.created_ms
        return now_ms - since > ttl

    def lookup(
        self, dhash: int, ahash: int, wh: tuple[float, float], now_ms: float, *, scope: object = None
    ) -> CacheEntry | None:
        """Best live entry within tolerance (lowest combined Hamming)."""
        bucket = size_bucket(*wh)
        aspect = float(wh[0]) / max(1.0, float(wh[1]))
        candidates: set[int] = set()
        for i, band in enumerate(_bands(dhash)):
            candidates |= self._index[i].get(band, set())
        best: CacheEntry | None = None
        best_d = 10**9
        expired: list[int] = []
        for key in candidates:
            entry = self._entries.get(key)
            if entry is None:
                continue
            if self._expired(entry, now_ms):
                expired.append(key)
                continue
            if entry.scope != scope:
                continue
            dd = hamming(entry.dhash, dhash)
            if dd > self.max_hamming:
                continue
            da = hamming(entry.ahash, ahash)
            if da > self.max_hamming + 2:
                continue
            if not self._bucket_ok(entry, bucket):
                continue
            if entry.aspect and abs(entry.aspect - aspect) > self.aspect_tol * max(aspect, entry.aspect):
                continue
            score = dd + da
            if score < best_d:
                best, best_d = entry, score
                if score == 0:
                    break
        for key in expired:
            self._drop(key)
        if best is None:
            self.misses += 1
            return None
        best.seen_ms = float(now_ms)
        best.hits += 1
        self._entries.move_to_end(best.key)
        self.hits += 1
        return best

    def get(
        self, dhash: int, ahash: int, wh: tuple[float, float], now_ms: float, *, scope: object = None
    ) -> Any | None:
        entry = self.lookup(dhash, ahash, wh, now_ms, scope=scope)
        return None if entry is None else entry.value

    def put(
        self,
        dhash: int,
        ahash: int,
        wh: tuple[float, float],
        value: Any,
        now_ms: float,
        *,
        scope: object = None,
        ttl_ms: float | None = None,
        any_size: bool = True,
        refresh: bool = True,
    ) -> CacheEntry:
        """Insert, or replace the entry with identical hashes, bucket and scope."""
        bucket = size_bucket(*wh)
        for key in list(self._index[0].get(_bands(dhash)[0], set())):
            entry = self._entries.get(key)
            if (
                entry is not None
                and entry.dhash == dhash
                and entry.ahash == ahash
                and entry.bucket == bucket
                and entry.scope == scope
            ):
                entry.value = value
                entry.seen_ms = float(now_ms)
                entry.created_ms = float(now_ms)
                entry.ttl_ms, entry.any_size, entry.refresh = ttl_ms, bool(any_size), bool(refresh)
                self._entries.move_to_end(key)
                return entry
        key = self._next_key
        self._next_key += 1
        entry = CacheEntry(
            key, dhash, ahash, bucket, value, float(now_ms), float(now_ms), aspect=float(wh[0]) / max(1.0, float(wh[1])),
            scope=scope, ttl_ms=ttl_ms, any_size=bool(any_size), refresh=bool(refresh),
        )
        self._entries[key] = entry
        for i, band in enumerate(_bands(dhash)):
            self._index[i].setdefault(band, set()).add(key)
        while len(self._entries) > self.max_entries:
            oldest = next(iter(self._entries))
            self._drop(oldest)
        return entry
