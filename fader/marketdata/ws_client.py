"""marketdata/ws_client.py

Persistent CLOB websocket client.

- Subscribes to all enabled slug token_ids.
- Maintains OrderBook per contract in memory via BookStore.
- Reconnects with exponential backoff.
- Full /book REST resync on reconnect.
- Emits per-contract last_update_ts via StalenessTracker.
- Handles: book, price_change, last_trade_price, best_bid_ask, new_market,
           market_resolved, tick_size_change events.
- Sends client PING every 10s.

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
    ) -> None:
        self._books = book_store
        self._staleness = staleness
        self._ws_url = ws_url
        self._new_market_cb = new_market_cb
        self._market_resolved_cb = market_resolved_cb
        self._band_low = band_low
        self._band_high = band_high

        self._subscribed: Set[str] = set()
        self._ws = None
        self._running = False
        self._connected = False
        self._reconnect_count = 0
        self._on_connect_cbs: List[Callable] = []  # called after each (re)connect

    @property
    def connected(self) -> bool:
        return self._connected

    def set_band(self, band_low: float, band_high: float) -> None:
        self._band_low = band_low
        self._band_high = band_high

    async def start(self, token_ids: List[str]) -> None:
        self._subscribed = set(token_ids)
        self._running = True
        asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
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
                asyncio.create_task(telegram.alert_ws_disconnect(str(e)))
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
            logger.info(f"WS connected to {self._ws_url}")

            # Subscribe to all tracked tokens
            if self._subscribed:
                sub_msg = json.dumps({
                    "assets_ids": list(self._subscribed),
                    "type": "market",
                    "custom_feature_enabled": True,
                })
                await ws.send(sub_msg)

            # Resync all books via REST
            await self._resync_books(list(self._subscribed))

            from infra import telegram
            asyncio.create_task(telegram.alert_ws_reconnect())

            # Start ping task
            ping_task = asyncio.create_task(self._ping_loop(ws))
            try:
                async for raw in ws:
                    await self._handle_message(raw)
            finally:
                ping_task.cancel()
                self._connected = False

    async def _ping_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(PING_INTERVAL_S)
            try:
                await ws.ping()
            except Exception:
                break

    async def _resync_books(self, token_ids: List[str]) -> None:
        """Full /book REST resync for the given token IDs."""
        loop = asyncio.get_event_loop()
        for token_id in token_ids:
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

    async def _handle_message(self, raw: str) -> None:
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
                    asyncio.create_task(self._new_market_cb(event))

            elif etype == "market_resolved":
                if self._market_resolved_cb:
                    asyncio.create_task(self._market_resolved_cb(event))

            elif etype == "tick_size_change":
                logger.debug(f"tick_size_change: {event}")
