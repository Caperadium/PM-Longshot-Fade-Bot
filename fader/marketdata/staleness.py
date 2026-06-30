"""marketdata/staleness.py

Per-contract staleness and feed-wide gap-halt logic.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class StalenessTracker:
    """
    Tracks when each token last received a ws update.
    Also tracks a feed-wide last-update time for gap halt.
    """

    def __init__(
        self,
        max_staleness_s: int = 30,
        gap_halt_s: int = 60,
    ) -> None:
        self._max_staleness_s = max_staleness_s
        self._gap_halt_s = gap_halt_s
        self._last_update: Dict[str, float] = {}  # token_id -> monotonic ts
        self._feed_last_update: float = time.monotonic()
        self._gap_halted: bool = False
        self._gap_halt_ts: Optional[float] = None

    def touch(self, token_id: str) -> None:
        """Call whenever a ws message arrives for this token."""
        now = time.monotonic()
        self._last_update[token_id] = now
        self._feed_last_update = now
        if self._gap_halted:
            self._gap_halted = False
            self._gap_halt_ts = None
            logger.info("Gap halt cleared — ws feed resumed")

    def is_stale(self, token_id: str) -> bool:
        """True if no update received within max_staleness_s."""
        last = self._last_update.get(token_id)
        if last is None:
            return True
        return (time.monotonic() - last) > self._max_staleness_s

    def check_gap_halt(self) -> bool:
        """Return True if feed-wide gap halt is active. Also sets the flag."""
        elapsed = time.monotonic() - self._feed_last_update
        if elapsed > self._gap_halt_s:
            if not self._gap_halted:
                self._gap_halted = True
                self._gap_halt_ts = time.monotonic()
                logger.error(f"GAP HALT: no ws updates for {elapsed:.0f}s")
        return self._gap_halted

    @property
    def gap_halted(self) -> bool:
        return self._gap_halted

    def feed_silence_s(self) -> float:
        return time.monotonic() - self._feed_last_update

    def set_params(self, max_staleness_s: int, gap_halt_s: int) -> None:
        self._max_staleness_s = max_staleness_s
        self._gap_halt_s = gap_halt_s
