"""engine/state_publisher.py

Periodically snapshots in-memory engine state to engine_state table
for dashboard consumption. Runs as asyncio task.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

from persistence.repos import engine_state_repo, positions_repo, orders_repo

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set(key: str, value: Any) -> None:
    engine_state_repo.publish(key, value)


class StatePublisher:
    def __init__(
        self,
        ws_client,
        book_store,
        staleness,
        risk,
        reconciler,
        publish_interval_s: float = 2.0,
    ) -> None:
        self._ws = ws_client
        self._books = book_store
        self._staleness = staleness
        self._risk = risk
        self._reconciler = reconciler
        self._interval = publish_interval_s
        self._task = None
        self._start_ts = _utc_now()

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        while True:
            try:
                self._publish()
            except Exception as e:
                logger.warning(f"StatePublisher error: {e}")
            await asyncio.sleep(self._interval)

    def _publish(self) -> None:
        # WS connectivity
        _set("ws_connected", self._ws.connected)
        _set("ws_reconnect_count", self._ws._reconnect_count)

        # Feed staleness / gap halt
        _set("gap_halted", self._staleness.gap_halted)
        _set("feed_silence_s", round(self._staleness.feed_silence_s(), 1))

        # Circuit breaker
        _set("breaker_tripped", self._risk.breaker_tripped)

        # Bankroll
        _set("bankroll", self._reconciler.bankroll)

        # Per-contract last update age
        ages: Dict[str, float] = {}
        now = time.monotonic()
        for token_id in self._books.all_token_ids():
            book = self._books.get(token_id)
            if book:
                age = now - book.last_update_ts if book.last_update_ts else -1
                ages[token_id] = round(age, 1)
        _set("token_last_update_ages", ages)

        # Open position count
        n_open = positions_repo.open_count()
        n_orders = orders_repo.pending_count()
        _set("open_positions", n_open)
        _set("pending_orders", n_orders)
        _set("engine_start_ts", self._start_ts)
        _set("published_at", _utc_now())
