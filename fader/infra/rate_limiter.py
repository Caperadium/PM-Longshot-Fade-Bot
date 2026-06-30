"""infra/rate_limiter.py

Async token-bucket rate limiter with separate write/read queues.
Writes prioritized over reads.

Usage:
    limiter = RateLimiter(write_per_s=10, write_burst=20, read_per_s=5, read_burst=10)
    await limiter.acquire("write")  # blocks until a write token is available
    await limiter.acquire("read")
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Literal

logger = logging.getLogger(__name__)

Kind = Literal["write", "read"]


class _Bucket:
    """Token bucket with configurable rate and burst."""

    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate       # tokens/second
        self._burst = burst     # max tokens
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait)


class RateLimiter:
    """Two-queue (write > read) async token bucket."""

    def __init__(
        self,
        write_per_s: float = 10.0,
        write_burst: int = 20,
        read_per_s: float = 5.0,
        read_burst: int = 10,
    ) -> None:
        self._write = _Bucket(write_per_s, write_burst)
        self._read = _Bucket(read_per_s, read_burst)
        # Write waiters get priority by draining read queue when writes pending
        self._write_waiters = 0

    async def acquire(self, kind: Kind = "read") -> None:
        if kind == "write":
            self._write_waiters += 1
            try:
                await self._write.acquire()
            finally:
                self._write_waiters -= 1
        else:
            # Yield briefly if writes are waiting
            if self._write_waiters > 0:
                await asyncio.sleep(0.01)
            await self._read.acquire()

    async def backoff_429(self, retry: int) -> None:
        """Exponential backoff with jitter for 429 responses."""
        base = 2.0 ** min(retry, 6)
        jitter = random.uniform(0, base * 0.2)
        wait = base + jitter
        logger.warning(f"429 rate-limited; backoff {wait:.1f}s (retry={retry})")
        await asyncio.sleep(wait)
