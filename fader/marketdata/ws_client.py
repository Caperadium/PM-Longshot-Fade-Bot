"""marketdata/ws_client.py

Persistent CLOB websocket client.

- Subscribes to all enabled slug token_ids.
- Maintains OrderBook per contract in memory via BookStore.
- Reconnects with exponential backoff.
- Full /book REST resync on reconnect (bounded-concurrent).
- Emits per-contract last_update_ts via StalenessTracker.
- Handles: book, price_change, last_trade_price, best_bid_ask, new_market,
           market_resolved, tick_size_change events.
- Sends application-level "PING" text frame every ws_ping_interval_s.
- Internal feed-silence watchdog force-reconnects a half-open socket
  (no library-level keepalive can detect this; see _watchdog_loop).

Dynamic subscribe/unsubscribe: call subscribe_tokens() / unsubscribe_tokens()
at runtime (dashboard slug add/remove triggers this without reconnect).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:
    websockets = None  # type: ignore

from marketdata.book_state import BookStore
from marketdata.staleness import StalenessTracker
from marketdata.rest_market import fetch_order_book_snapshot

logger = logging.getLogger(__name__)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
PING_INTERVAL_S = 10
PONG_TIMEOUT_S = 25
FORCE_RECONNECT_S = 90
WATCHDOG_INTERVAL_S = 5
RECONNECT_BASE_S = 1.0
RECONNECT_MAX_S = 60.0


class WsClient:
    """
    Async CLOB websocket client with auto-reconnect.

    new_market_cb: optional coroutine called with market dict on new_market event.
    market_resolved_cb: optional coroutine called with resolution dict.
    """

    def __init__(
        self,
        book_store: BookStore,
        staleness: StalenessTracker,
        ws_url: str = WS_URL,
        new_market_cb: Optional[Callable] = None,
        market_resolved_cb: Optional[Callable] = None,
        band_low: float = 0.80,
        band_high: float = 0.95,
        resync_concurrency: int = 8,
    ) -> None:
        self._books = book_store
        self._staleness = staleness
        self._ws_url = ws_url
        self._new_market_cb = new_market_cb
        self._market_resolved_cb = market_resolved_cb
        self._band_low = band_low
        self._band_high = band_high
        # COLD: constructor-only, no hot-reload setter (see plan FIX 2).
        self._resync_concurrency = max(1, resync_concurrency)

        self._subscribed: Set[str] = set()
        self._ws = None
        self._running = False
        self._connected = False
        self._reconnect_count = 0
        self._on_connect_cbs: List[Callable] = []  # called after each (re)connect

        # Fire-and-forget task registry — prevents GC of event-dispatch tasks
        # (new_market_cb / market_resolved_cb) per FIX 7.
        self._bg_tasks: set = set()

        # FIX 1b: feed-silence watchdog (forced reconnect of half-open sockets).
        self._watchdog_task: Optional[asyncio.Task] = None
        self.first_data_received = False  # armed only after first resync baseline
        self._force_close_issued = False  # spam guard; cleared on (re)connect
        self._force_reconnect_s = FORCE_RECONNECT_S

        # FIX 1a: app-level ping/pong (belt-and-suspenders; off by default).
        self._ping_interval_s = PING_INTERVAL_S
        self._pong_timeout_s = PONG_TIMEOUT_S
        self._expect_pong = False
        self._last_pong_ts = time.monotonic()

    @property
    def connected(self) -> bool:
        return self._connected

    def set_band(self, band_low: float, band_high: float) -> None:
        self._band_low = band_low
        self._band_high = band_high

    def set_watchdog(
        self,
        force_reconnect_s: int,
        ping_interval_s: Optional[int] = None,
        pong_timeout_s: Optional[int] = None,
        expect_pong: Optional[bool] = None,
    ) -> None:
        """Hot-reloadable watchdog/ping params (mirrors set_band)."""
        self._force_reconnect_s = force_reconnect_s
        if ping_interval_s is not None:
            self._ping_interval_s = ping_interval_s
        if pong_timeout_s is not None:
            self._pong_timeout_s = pong_timeout_s
        if expect_pong is not None:
            self._expect_pong = expect_pong

    async def start(self, token_ids: List[str]) -> None:
        self._subscribed = set(token_ids)
        self._running = True
        asyncio.create_task(self._run_loop())
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def stop(self) -> None:
        self._running = False
        if self._watchdog_task:
            self._watchdog_task.cancel()
            self._watchdog_task = None
        if self._ws:
            await self._ws.close()

    async def subscribe_tokens(self, token_ids: List[str]) -> None:
        new = [t for t in token_ids if t not in self._subscribed]
        if not new:
            return
        self._subscribed.update(new)
        if self._ws and self._connected:
            msg = json.dumps({"assets_ids": new, "operation": "subscribe"})
            try:
                await self._ws.send(msg)
                await self._resync_books(new)
            except Exception as e:
                logger.warning(f"subscribe_tokens send failed: {e}")

    async def unsubscribe_tokens(self, token_ids: List[str]) -> None:
        for t in token_ids:
            self._subscribed.discard(t)
        if self._ws and self._connected and token_ids:
            try:
                msg = json.dumps({"assets_ids": token_ids, "operation": "unsubscribe"})
                await self._ws.send(msg)
            except Exception as e:
                logger.warning(f"unsubscribe_tokens send failed: {e}")

    async def _run_loop(self) -> None:
        backoff = RECONNECT_BASE_S
        while self._running:
            try:
                await self._connect_and_run()
                backoff = RECONNECT_BASE_S
            except Exception as e:
                self._connected = False
                self._reconnect_count += 1
                logger.warning(
                    f"WS disconnected (#{self._reconnect_count}): {e}; "
                    f"reconnect in {backoff:.0f}s"
                )
                from infra import telegram
                telegram.fire(telegram.alert_ws_disconnect(str(e)))
                if not self._running:
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_MAX_S)

    async def _connect_and_run(self) -> None:
        if websockets is None:
            raise ImportError("websockets package not installed")

        async with websockets.connect(
            self._ws_url,
            ping_interval=None,  # we send our own keepalives
            close_timeout=5,
        ) as ws:
            self._ws = ws
            self._connected = True
            self._force_close_issued = False  # clear spam guard on fresh connect
            # Reset the feed-silence baseline BEFORE resync so the watchdog
            # doesn't count outage time against the healthy new socket and
            # force-close it during a slow reconnect resync (the first_data_received
            # gate only protects the very first connect).
            self._staleness.mark_alive()
            logger.info(f"WS connected to {self._ws_url}")

            # Subscribe to all tracked tokens
            if self._subscribed:
                sub_msg = json.dumps({
                    "assets_ids": list(self._subscribed),
                    "type": "market",
                    "custom_feature_enabled": True,
                })
                await ws.send(sub_msg)

            # Resync all books via REST (bounded-concurrent, see FIX 2)
            await self._resync_books(list(self._subscribed))
            # Arm the watchdog only once a full snapshot baseline exists.
            # Sticky thereafter — never reset across later reconnects.
            self.first_data_received = True

            from infra import telegram
            telegram.fire(telegram.alert_ws_reconnect())

            # Start ping task
            ping_task = asyncio.create_task(self._ping_loop(ws))
            try:
                async for raw in ws:
                    await self._handle_message(raw)
            finally:
                ping_task.cancel()
                self._connected = False

    async def _ping_loop(self, ws) -> None:
        """Application-level keepalive (FIX 1a).

        Sends a "PING" text frame every ws_ping_interval_s. Polymarket CLOB
        market WS expects this (protocol-level ping/pong is unreliable).
        If ws_expect_pong is enabled and no PONG arrives within
        ws_pong_timeout_s, close the socket to break the read loop (the
        feed-silence watchdog in FIX 1b is the primary/reliable layer;
        this is belt-and-suspenders, off by default).
        """
        self._last_pong_ts = time.monotonic()
        while True:
            await asyncio.sleep(self._ping_interval_s)
            try:
                await ws.send("PING")
            except Exception:
                break
            if self._expect_pong:
                silence = time.monotonic() - self._last_pong_ts
                if silence > self._pong_timeout_s:
                    logger.warning(
                        f"No PONG for {silence:.0f}s (> {self._pong_timeout_s}s) — closing socket"
                    )
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    break

    async def _resync_books(self, token_ids: List[str]) -> None:
        """Full /book REST resync for the given token IDs.

        Bounded-concurrent (FIX 2): runs BEFORE the read loop so the
        snapshot-before-delta invariant (BookStore.delta on an empty book
        appends a spurious level) is preserved, but fans out fetches under
        a semaphore so 500+ tokens resync in seconds, not minutes.
        """
        sem = asyncio.Semaphore(self._resync_concurrency)
        loop = asyncio.get_running_loop()

        async def one(token_id: str) -> None:
            async with sem:
                try:
                    snap = await loop.run_in_executor(
                        None, lambda t=token_id: fetch_order_book_snapshot(t)
                    )
                    if snap:
                        self._books.snapshot(token_id, snap["bids"], snap["asks"])
                        self._staleness.touch(token_id)
                        book = self._books.get(token_id)
                        if book:
                            book.update_band_tracker(self._band_low, self._band_high)
                except Exception as e:
                    logger.warning(f"book resync for {token_id[:16]}: {e}")

        await asyncio.gather(*(one(t) for t in token_ids), return_exceptions=True)

    async def _watchdog_loop(self) -> None:
        """FIX 1b: force-reconnect a half-open socket after sustained silence.

        ping_interval=None disables library keepalive, and a manual
        ws.ping() with a discarded pong future can't detect a dead TCP
        socket — async for raw in ws would block forever. This watchdog is
        the primary, mode-independent recovery layer.
        """
        while True:
            try:
                if (
                    self._connected
                    and self._ws is not None
                    and self.first_data_received
                    and not self._force_close_issued
                ):
                    silence = self._staleness.feed_silence_s()
                    if silence > self._force_reconnect_s:
                        logger.warning(
                            f"WS feed silent for {silence:.0f}s "
                            f"(> {self._force_reconnect_s}s) — forcing reconnect"
                        )
                        self._force_close_issued = True
                        try:
                            await self._ws.close()
                        except Exception as e:
                            logger.warning(f"watchdog force-close failed: {e}")
            except Exception as e:
                logger.warning(f"watchdog loop error: {e}")
            await asyncio.sleep(WATCHDOG_INTERVAL_S)

    def _spawn_bg(self, coro) -> None:
        """Fire-and-forget an event-dispatch coroutine, retaining a ref
        (GC-safe; FIX 7). Not a telegram coro, so doesn't use telegram.fire."""
        t = asyncio.create_task(coro)
        self._bg_tasks.add(t)
        t.add_done_callback(self._bg_tasks.discard)

    async def _handle_message(self, raw: str) -> None:
        # Application-level PONG (FIX 1a) — Polymarket may reply as plain
        # text rather than a JSON event.
        if isinstance(raw, str) and raw.strip() == "PONG":
            self._last_pong_ts = time.monotonic()
            return

        try:
            events = json.loads(raw)
        except json.JSONDecodeError:
            return

        if not isinstance(events, list):
            events = [events]

        for event in events:
            etype = event.get("event_type", "")

            if etype == "book":
                token_id = event.get("asset_id", "")
                if token_id:
                    self._books.snapshot(
                        token_id,
                        event.get("bids", []),
                        event.get("asks", []),
                    )
                    self._staleness.touch(token_id)
                    book = self._books.get(token_id)
                    if book:
                        book.update_band_tracker(self._band_low, self._band_high)

            elif etype == "price_change":
                token_id = event.get("asset_id", "")
                changes = event.get("changes", [])
                if token_id:
                    for change in changes:
                        side = change.get("side", "")
                        price = change.get("price", "")
                        size = change.get("size", "0")
                        self._books.delta(token_id, side, price, size)
                    self._staleness.touch(token_id)
                    book = self._books.get(token_id)
                    if book:
                        book.update_band_tracker(self._band_low, self._band_high)

            elif etype == "last_trade_price":
                # Feeds backtest limit-fill model; note last trade per token
                token_id = event.get("asset_id", "")
                if token_id:
                    self._staleness.touch(token_id)

            elif etype == "best_bid_ask":
                token_id = event.get("asset_id", "")
                if token_id:
                    self._staleness.touch(token_id)

            elif etype == "new_market":
                if self._new_market_cb:
                    self._spawn_bg(self._new_market_cb(event))

            elif etype == "market_resolved":
                if self._market_resolved_cb:
                    self._spawn_bg(self._market_resolved_cb(event))

            elif etype == "tick_size_change":
                logger.debug(f"tick_size_change: {event}")
