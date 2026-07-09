"""tests/test_stability_fixes.py

Targeted tests for the VPS stability fix plan (FIX 1-8 + L1/L2):
  - FIX 1b: WS feed-silence watchdog forces a reconnect close.
  - FIX 3:  rehydrate_resting recovers live resting limit orders from DB+API.
  - FIX 4:  log rotation configured with the expected cap.
  - FIX 5:  cancel_all_resting (graceful shutdown).
  - FIX 8:  execute_write retries only on locked/busy, else raises.
  - Config: new FeedConfig fields + hot-reload wiring.

Run: python -m pytest fader/tests/test_stability_fixes.py -v
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add fader root to path
_FADER_ROOT = Path(__file__).parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))


# =========================================================================
# FIX 1b: WS feed-silence watchdog
# =========================================================================

class TestWsForceReconnectWatchdog(unittest.IsolatedAsyncioTestCase):
    """Watchdog must force-close a half-open socket after sustained silence,
    but only once armed (first_data_received) and not spam repeat closes."""

    async def _run_watchdog_briefly(self, ws, interval=0.01, settle=0.05):
        import marketdata.ws_client as ws_mod
        with patch.object(ws_mod, "WATCHDOG_INTERVAL_S", interval):
            task = asyncio.create_task(ws._watchdog_loop())
            await asyncio.sleep(settle)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_force_close_on_sustained_silence(self):
        from marketdata.ws_client import WsClient

        staleness = MagicMock()
        staleness.feed_silence_s.return_value = 999.0
        ws = WsClient(book_store=MagicMock(), staleness=staleness)
        ws._connected = True
        ws._ws = MagicMock()
        ws._ws.close = AsyncMock()
        ws.first_data_received = True
        ws._force_reconnect_s = 10

        await self._run_watchdog_briefly(ws)

        ws._ws.close.assert_awaited()
        self.assertTrue(ws._force_close_issued)

    async def test_no_trip_before_first_data_received(self):
        """Gate: watchdog must stay inert until the first resync baseline
        completes, even with arbitrarily long silence."""
        from marketdata.ws_client import WsClient

        staleness = MagicMock()
        staleness.feed_silence_s.return_value = 999.0
        ws = WsClient(book_store=MagicMock(), staleness=staleness)
        ws._connected = True
        ws._ws = MagicMock()
        ws._ws.close = AsyncMock()
        ws.first_data_received = False  # not armed
        ws._force_reconnect_s = 10

        await self._run_watchdog_briefly(ws)

        ws._ws.close.assert_not_awaited()

    async def test_no_trip_when_disconnected(self):
        from marketdata.ws_client import WsClient

        staleness = MagicMock()
        staleness.feed_silence_s.return_value = 999.0
        ws = WsClient(book_store=MagicMock(), staleness=staleness)
        ws._connected = False  # mid-backoff
        ws._ws = MagicMock()
        ws._ws.close = AsyncMock()
        ws.first_data_received = True
        ws._force_reconnect_s = 10

        await self._run_watchdog_briefly(ws)

        ws._ws.close.assert_not_awaited()

    async def test_spam_guard_prevents_repeat_close(self):
        from marketdata.ws_client import WsClient

        staleness = MagicMock()
        staleness.feed_silence_s.return_value = 999.0
        ws = WsClient(book_store=MagicMock(), staleness=staleness)
        ws._connected = True
        ws._ws = MagicMock()
        ws._ws.close = AsyncMock()
        ws.first_data_received = True
        ws._force_reconnect_s = 10
        ws._force_close_issued = True  # already issued this cycle

        await self._run_watchdog_briefly(ws)

        ws._ws.close.assert_not_awaited()

    async def test_no_trip_within_threshold(self):
        from marketdata.ws_client import WsClient

        staleness = MagicMock()
        staleness.feed_silence_s.return_value = 2.0  # well under threshold
        ws = WsClient(book_store=MagicMock(), staleness=staleness)
        ws._connected = True
        ws._ws = MagicMock()
        ws._ws.close = AsyncMock()
        ws.first_data_received = True
        ws._force_reconnect_s = 90

        await self._run_watchdog_briefly(ws)

        ws._ws.close.assert_not_awaited()

    async def test_no_false_trip_during_slow_reconnect_resync(self):
        """Regression: after an outage longer than ws_force_reconnect_s, the
        reconnect must NOT be force-closed by the watchdog before resync's
        first touch.

        Uses a REAL StalenessTracker. The bug: feed_silence carried the whole
        outage duration into the new connection (first_data_received is sticky
        True after the first connect, so it no longer gates), letting a
        watchdog tick force-close the healthy socket during a slow resync.
        Fix: _connect_and_run calls staleness.mark_alive() on (re)connect
        before resync, resetting the silence baseline.
        """
        from marketdata.ws_client import WsClient
        from marketdata.staleness import StalenessTracker

        staleness = StalenessTracker(max_staleness_s=30, gap_halt_s=60)
        # Simulate a long outage: feed has been silent well past the threshold.
        staleness._feed_last_update = time.monotonic() - 1000.0
        self.assertGreater(staleness.feed_silence_s(), 90)

        ws = WsClient(book_store=MagicMock(), staleness=staleness)
        ws._connected = True
        ws._ws = MagicMock()
        ws._ws.close = AsyncMock()
        ws.first_data_received = True  # sticky from a prior connect
        ws._force_close_issued = False
        ws._force_reconnect_s = 90

        # Emulate what _connect_and_run now does right after (re)connect,
        # BEFORE resync touches any token.
        staleness.mark_alive()
        self.assertLess(staleness.feed_silence_s(), 1.0)

        await self._run_watchdog_briefly(ws)

        # No false force-close: the reconnect is treated as freshly alive.
        ws._ws.close.assert_not_awaited()
        self.assertFalse(ws._force_close_issued)

    async def test_connect_calls_mark_alive_before_resync(self):
        """_connect_and_run must reset the feed-silence baseline (mark_alive)
        before awaiting resync, so the watchdog can't fire during it."""
        from marketdata.ws_client import WsClient

        staleness = MagicMock()
        order = []
        staleness.mark_alive.side_effect = lambda: order.append("mark_alive")

        ws = WsClient(book_store=MagicMock(), staleness=staleness)
        ws._subscribed = {"tok"}

        async def fake_resync(_tokens):
            order.append("resync")

        ws._resync_books = fake_resync  # type: ignore[assignment]

        # Minimal async context-manager stub for websockets.connect(...).
        sent = []

        class _FakeWs:
            async def send(self, msg):
                sent.append(msg)

            async def close(self):
                pass

            def __aiter__(self):
                async def _gen():
                    return
                    yield  # pragma: no cover
                return _gen()

        class _Conn:
            async def __aenter__(self):
                return _FakeWs()

            async def __aexit__(self, *a):
                return False

        def _swallow_fire(coro=None, *a, **k):
            # close the coroutine so it isn't reported as never-awaited
            if hasattr(coro, "close"):
                coro.close()

        import marketdata.ws_client as ws_mod
        with patch.object(ws_mod.websockets, "connect", lambda *a, **k: _Conn()), \
             patch("infra.telegram.fire", _swallow_fire):
            await ws._connect_and_run()

        self.assertEqual(order, ["mark_alive", "resync"],
                         "mark_alive must run before resync")


class TestSetWatchdog(unittest.TestCase):
    """set_watchdog mirrors set_band: hot-reloadable, partial updates keep
    unspecified fields unchanged."""

    def test_set_watchdog_updates_all_fields(self):
        from marketdata.ws_client import WsClient

        ws = WsClient(book_store=MagicMock(), staleness=MagicMock())
        ws.set_watchdog(45, 5, 12, True)
        self.assertEqual(ws._force_reconnect_s, 45)
        self.assertEqual(ws._ping_interval_s, 5)
        self.assertEqual(ws._pong_timeout_s, 12)
        self.assertTrue(ws._expect_pong)

    def test_set_watchdog_partial_update_keeps_defaults(self):
        from marketdata.ws_client import WsClient, PING_INTERVAL_S

        ws = WsClient(book_store=MagicMock(), staleness=MagicMock())
        ws.set_watchdog(60)
        self.assertEqual(ws._force_reconnect_s, 60)
        self.assertEqual(ws._ping_interval_s, PING_INTERVAL_S)  # unchanged


class TestResyncConcurrency(unittest.TestCase):
    """resync_concurrency is a COLD constructor kwarg; floor of 1."""

    def test_resync_concurrency_floor_of_one(self):
        from marketdata.ws_client import WsClient

        ws = WsClient(book_store=MagicMock(), staleness=MagicMock(), resync_concurrency=0)
        self.assertEqual(ws._resync_concurrency, 1)

    def test_resync_concurrency_default(self):
        from marketdata.ws_client import WsClient

        ws = WsClient(book_store=MagicMock(), staleness=MagicMock())
        self.assertEqual(ws._resync_concurrency, 8)


# =========================================================================
# Phase 6, item 4: _handle_message dispatch dict + _sync_book() helper.
# Equivalence tests -- same per-event-type behavior as the old if/elif chain.
# =========================================================================

class TestHandleMessageDispatch(unittest.IsolatedAsyncioTestCase):
    def _make_ws(self):
        from marketdata.ws_client import WsClient

        book_store = MagicMock()
        staleness = MagicMock()
        ws = WsClient(book_store=book_store, staleness=staleness)
        return ws, book_store, staleness

    async def test_dispatch_table_covers_all_known_event_types(self):
        ws, _, _ = self._make_ws()
        for etype in (
            "book", "price_change", "last_trade_price", "best_bid_ask",
            "new_market", "market_resolved", "tick_size_change",
        ):
            self.assertIn(etype, ws._dispatch)

    async def test_unknown_event_type_is_noop(self):
        ws, book_store, staleness = self._make_ws()
        await ws._handle_message('[{"event_type": "unknown_thing"}]')
        book_store.snapshot.assert_not_called()
        staleness.touch.assert_not_called()

    async def test_book_event_snapshots_and_syncs(self):
        ws, book_store, staleness = self._make_ws()
        fake_book = MagicMock()
        book_store.get.return_value = fake_book
        msg = (
            '[{"event_type": "book", "asset_id": "tok1", '
            '"bids": [], "asks": []}]'
        )
        await ws._handle_message(msg)
        book_store.snapshot.assert_called_once_with("tok1", [], [])
        staleness.touch.assert_called_once_with("tok1")
        fake_book.update_band_tracker.assert_called_once_with(
            ws._band_low, ws._band_high
        )

    async def test_book_event_missing_asset_id_is_noop(self):
        ws, book_store, staleness = self._make_ws()
        await ws._handle_message('[{"event_type": "book"}]')
        book_store.snapshot.assert_not_called()
        staleness.touch.assert_not_called()

    async def test_price_change_event_deltas_and_syncs(self):
        ws, book_store, staleness = self._make_ws()
        fake_book = MagicMock()
        book_store.get.return_value = fake_book
        msg = (
            '[{"event_type": "price_change", "asset_id": "tok2", '
            '"changes": [{"side": "SELL", "price": "0.9", "size": "10"}]}]'
        )
        await ws._handle_message(msg)
        book_store.delta.assert_called_once_with("tok2", "SELL", "0.9", "10")
        staleness.touch.assert_called_once_with("tok2")
        fake_book.update_band_tracker.assert_called_once_with(
            ws._band_low, ws._band_high
        )

    async def test_last_trade_price_touches_staleness_only(self):
        ws, book_store, staleness = self._make_ws()
        await ws._handle_message(
            '[{"event_type": "last_trade_price", "asset_id": "tok3"}]'
        )
        staleness.touch.assert_called_once_with("tok3")
        book_store.snapshot.assert_not_called()
        book_store.delta.assert_not_called()

    async def test_best_bid_ask_touches_staleness_only(self):
        ws, book_store, staleness = self._make_ws()
        await ws._handle_message(
            '[{"event_type": "best_bid_ask", "asset_id": "tok4"}]'
        )
        staleness.touch.assert_called_once_with("tok4")

    async def test_new_market_dispatches_to_callback(self):
        from marketdata.ws_client import WsClient

        cb = AsyncMock()
        ws = WsClient(
            book_store=MagicMock(), staleness=MagicMock(), new_market_cb=cb,
        )
        await ws._handle_message('[{"event_type": "new_market", "slug": "x"}]')
        await asyncio.sleep(0)  # let the fire-and-forget task run
        cb.assert_called_once()

    async def test_market_resolved_dispatches_to_callback(self):
        from marketdata.ws_client import WsClient

        cb = AsyncMock()
        ws = WsClient(
            book_store=MagicMock(), staleness=MagicMock(), market_resolved_cb=cb,
        )
        await ws._handle_message(
            '[{"event_type": "market_resolved", "asset_id": "tokX"}]'
        )
        await asyncio.sleep(0)
        cb.assert_called_once()

    async def test_pong_text_frame_updates_last_pong_and_skips_dispatch(self):
        ws, book_store, staleness = self._make_ws()
        before = ws._last_pong_ts
        await asyncio.sleep(0.01)
        await ws._handle_message("PONG")
        self.assertGreater(ws._last_pong_ts, before)
        book_store.snapshot.assert_not_called()
        staleness.touch.assert_not_called()


# =========================================================================
# FIX 3: rehydrate_resting
# =========================================================================

class TestRehydrateResting(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        os.environ["POLYMARKET_USER_ADDRESS"] = "0xTEST_USER"
        self.db_path = _FADER_ROOT / "tests" / "test_fader_rehydrate.db"
        from infra.db import set_db_path, init_db
        set_db_path(self.db_path)
        if self.db_path.exists():
            self.db_path.unlink()
        init_db()

    async def asyncTearDown(self):
        if self.db_path.exists():
            self.db_path.unlink()

    def _insert_pending_limit(self, order_id, token_id, price=0.85, size=10.0,
                               created_at=None):
        from infra.db import get_connection
        conn = get_connection()
        try:
            conn.execute(
                """INSERT INTO orders
                   (order_id, idempotency_key, slug, token_id, side, type, price, size, status, created_at)
                VALUES (?, ?, ?, ?, 'BUY', 'LIMIT', ?, ?, 'PENDING', ?)""",
                (order_id, f"ik-{order_id}", "test-slug", token_id, price, size,
                 created_at or datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    async def test_rehydrate_populates_resting_for_live_order(self):
        from execution.order_manager import OrderManager
        from config.config_loader import load_config

        self._insert_pending_limit("live-order-1", "0xTOKEN")

        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = False
        provider.async_fetch_open_orders = AsyncMock(return_value=[
            {"order_id": "live-order-1", "token_id": "0xTOKEN", "status": "LIVE"}
        ])

        om = OrderManager(cfg=cfg, provider=provider)
        count = await om.rehydrate_resting()

        self.assertEqual(count, 1)
        self.assertIn("0xTOKEN", om._resting)
        ro = om._resting["0xTOKEN"]
        self.assertEqual(ro.order_id, "live-order-1")
        self.assertEqual(ro.price, 0.85)
        self.assertEqual(ro.size, 10.0)

    async def test_rehydrate_skips_orders_not_live(self):
        """DB PENDING LIMIT not present in the live API response must not
        be rehydrated — left to the reconciler's mark_vanished path."""
        from execution.order_manager import OrderManager
        from config.config_loader import load_config

        self._insert_pending_limit("vanished-order", "0xTOKEN2")

        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = False
        provider.async_fetch_open_orders = AsyncMock(return_value=[])

        om = OrderManager(cfg=cfg, provider=provider)
        count = await om.rehydrate_resting()

        self.assertEqual(count, 0)
        self.assertNotIn("0xTOKEN2", om._resting)

    async def test_rehydrate_skips_sim_and_fake_ids(self):
        from execution.order_manager import OrderManager
        from config.config_loader import load_config

        self._insert_pending_limit("SIM-12345", "0xTOKEN3")
        self._insert_pending_limit("FAKE-999", "0xTOKEN4")

        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = False
        provider.async_fetch_open_orders = AsyncMock(return_value=[
            {"order_id": "SIM-12345", "token_id": "0xTOKEN3", "status": "LIVE"},
            {"order_id": "FAKE-999", "token_id": "0xTOKEN4", "status": "LIVE"},
        ])

        om = OrderManager(cfg=cfg, provider=provider)
        count = await om.rehydrate_resting()

        self.assertEqual(count, 0)
        self.assertEqual(om._resting, {})

    async def test_rehydrate_newest_per_token_wins(self):
        """Two PENDING LIMIT rows for the same token: the newest wins."""
        from execution.order_manager import OrderManager
        from config.config_loader import load_config

        self._insert_pending_limit(
            "old-order", "0xTOKEN5", price=0.80,
            created_at="2020-01-01T00:00:00+00:00",
        )
        self._insert_pending_limit(
            "new-order", "0xTOKEN5", price=0.90,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = False
        provider.async_fetch_open_orders = AsyncMock(return_value=[
            {"order_id": "old-order", "token_id": "0xTOKEN5", "status": "LIVE"},
            {"order_id": "new-order", "token_id": "0xTOKEN5", "status": "LIVE"},
        ])

        om = OrderManager(cfg=cfg, provider=provider)
        count = await om.rehydrate_resting()

        self.assertEqual(count, 1)
        self.assertEqual(om._resting["0xTOKEN5"].order_id, "new-order")

    async def test_rehydrate_paper_mode_is_noop(self):
        from execution.order_manager import OrderManager
        from config.config_loader import load_config

        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = True
        provider.async_fetch_open_orders = AsyncMock()

        om = OrderManager(cfg=cfg, provider=provider)
        count = await om.rehydrate_resting()

        self.assertEqual(count, 0)
        provider.async_fetch_open_orders.assert_not_called()

    async def test_rehydrate_paper_mode_still_reads_db(self):
        """Paper mode: DB read only, no API verify. PaperProvider never
        creates a PENDING LIMIT row (place_order always returns FILLED),
        so this is a no-op in practice, but the DB read itself must still
        run -- a stray PENDING LIMIT row left over from a mode switch is
        not silently ignored, and a DB read failure is still fatal even
        in paper mode (see test_rehydrate_db_read_failure_alerts_and_raises)."""
        from execution.order_manager import OrderManager
        from config.config_loader import load_config
        import persistence.repos as repos_mod

        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = True
        provider.async_fetch_open_orders = AsyncMock()

        om = OrderManager(cfg=cfg, provider=provider)
        with patch.object(
            repos_mod.orders_repo, "pending_limit_orders", wraps=repos_mod.orders_repo.pending_limit_orders,
        ) as read_mock:
            count = await om.rehydrate_resting()

        read_mock.assert_called_once()
        self.assertEqual(count, 0)
        provider.async_fetch_open_orders.assert_not_called()

    async def test_rehydrate_api_none_keeps_rows_unverified(self):
        """fetch_open_orders() -> None means API unavailable, not "no
        orders". Every DB PENDING LIMIT row must be rehydrated (not
        dropped) and flagged unverified=True."""
        from execution.order_manager import OrderManager
        from config.config_loader import load_config

        self._insert_pending_limit("live-order-2", "0xTOKEN6")

        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = False
        provider.async_fetch_open_orders = AsyncMock(return_value=None)

        om = OrderManager(cfg=cfg, provider=provider)
        count = await om.rehydrate_resting()

        self.assertEqual(count, 1)
        self.assertIn("0xTOKEN6", om._resting)
        self.assertTrue(om._resting["0xTOKEN6"].unverified)

    async def test_rehydrate_db_read_failure_alerts_and_raises(self):
        """A DB read failure (not an API failure) is fatal at startup:
        fire a telegram alert, then re-raise so main.py aborts."""
        from execution.order_manager import OrderManager
        from config.config_loader import load_config
        import persistence.repos as repos_mod

        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = False
        provider.async_fetch_open_orders = AsyncMock(return_value=[])

        om = OrderManager(cfg=cfg, provider=provider)

        with patch.object(
            repos_mod.orders_repo, "pending_limit_orders",
            side_effect=sqlite3.OperationalError("database is locked"),
        ), patch("infra.telegram.alert_error", new=AsyncMock()) as alert_mock:
            with self.assertRaises(sqlite3.OperationalError):
                await om.rehydrate_resting()

        alert_mock.assert_awaited_once()
        provider.async_fetch_open_orders.assert_not_called()

    async def test_rehydrate_db_read_failure_alerts_and_raises_in_paper_mode(self):
        """The DB-read-failure-aborts-startup rule applies in paper mode
        too -- the DB read now runs before the is_paper check."""
        from execution.order_manager import OrderManager
        from config.config_loader import load_config
        import persistence.repos as repos_mod

        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = True
        provider.async_fetch_open_orders = AsyncMock()

        om = OrderManager(cfg=cfg, provider=provider)

        with patch.object(
            repos_mod.orders_repo, "pending_limit_orders",
            side_effect=sqlite3.OperationalError("database is locked"),
        ), patch("infra.telegram.alert_error", new=AsyncMock()) as alert_mock:
            with self.assertRaises(sqlite3.OperationalError):
                await om.rehydrate_resting()

        alert_mock.assert_awaited_once()
        provider.async_fetch_open_orders.assert_not_called()


class TestReverifyUnverifiedResting(unittest.IsolatedAsyncioTestCase):
    """requote_check re-verifies unverified resting orders against the
    live API each tick instead of trusting or dropping them blindly."""

    async def asyncSetUp(self):
        os.environ["POLYMARKET_USER_ADDRESS"] = "0xTEST_USER"
        self.db_path = _FADER_ROOT / "tests" / "test_fader_reverify.db"
        from infra.db import set_db_path, init_db
        set_db_path(self.db_path)
        if self.db_path.exists():
            self.db_path.unlink()
        init_db()

    async def asyncTearDown(self):
        if self.db_path.exists():
            self.db_path.unlink()

    def _make_resting(self, token_id, order_id, unverified=True):
        from execution.order_manager import RestingOrder
        return RestingOrder(
            order_id=order_id, idempotency_key=f"ik-{order_id}", slug="s",
            token_id=token_id, price=0.85, size=10.0, notional=8.5,
            placed_at=time.monotonic(), ttl_s=300, mid=0.85,
            unverified=unverified,
        )

    async def test_reverify_clears_flag_when_confirmed_live(self):
        from execution.order_manager import OrderManager
        from config.config_loader import load_config

        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = False
        provider.async_fetch_open_orders = AsyncMock(return_value=[
            {"order_id": "o-1", "token_id": "0xA", "status": "LIVE"},
        ])

        om = OrderManager(cfg=cfg, provider=provider)
        om._resting["0xA"] = self._make_resting("0xA", "o-1")

        book_store = MagicMock()
        book_store.get.return_value = None  # skip the rest of requote_check
        await om.requote_check(book_store)

        self.assertIn("0xA", om._resting)
        self.assertFalse(om._resting["0xA"].unverified)

    async def test_reverify_drops_order_confirmed_gone(self):
        from execution.order_manager import OrderManager
        from config.config_loader import load_config

        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = False
        provider.async_fetch_open_orders = AsyncMock(return_value=[])

        om = OrderManager(cfg=cfg, provider=provider)
        om._resting["0xB"] = self._make_resting("0xB", "o-2")

        book_store = MagicMock()
        book_store.get.return_value = None
        await om.requote_check(book_store)

        self.assertNotIn("0xB", om._resting)

    async def test_reverify_stays_unverified_on_repeat_api_failure(self):
        from execution.order_manager import OrderManager
        from config.config_loader import load_config

        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = False
        provider.async_fetch_open_orders = AsyncMock(return_value=None)

        om = OrderManager(cfg=cfg, provider=provider)
        om._resting["0xC"] = self._make_resting("0xC", "o-3")

        book_store = MagicMock()
        book_store.get.return_value = None
        await om.requote_check(book_store)

        self.assertIn("0xC", om._resting)
        self.assertTrue(om._resting["0xC"].unverified)

    async def test_reverify_skipped_when_no_unverified_orders(self):
        """No wasted API call when every resting order is already verified."""
        from execution.order_manager import OrderManager
        from config.config_loader import load_config

        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = False
        provider.async_fetch_open_orders = AsyncMock(return_value=[])

        om = OrderManager(cfg=cfg, provider=provider)
        om._resting["0xD"] = self._make_resting("0xD", "o-4", unverified=False)

        book_store = MagicMock()
        book_store.get.return_value = None
        await om.requote_check(book_store)

        provider.async_fetch_open_orders.assert_not_called()


# =========================================================================
# FIX 5: cancel_all_resting (graceful shutdown)
# =========================================================================

class TestCancelAllResting(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        os.environ["POLYMARKET_USER_ADDRESS"] = "0xTEST_USER"
        self.db_path = _FADER_ROOT / "tests" / "test_fader_cancelall.db"
        from infra.db import set_db_path, init_db
        set_db_path(self.db_path)
        if self.db_path.exists():
            self.db_path.unlink()
        init_db()

    async def asyncTearDown(self):
        if self.db_path.exists():
            self.db_path.unlink()

    def _make_resting(self, token_id, order_id):
        from execution.order_manager import RestingOrder
        return RestingOrder(
            order_id=order_id, idempotency_key=f"ik-{order_id}", slug="s",
            token_id=token_id, price=0.85, size=10.0, notional=8.5,
            placed_at=time.monotonic(), ttl_s=300, mid=0.85,
        )

    async def test_cancel_all_resting_clears_paper_mode_without_calling_provider(self):
        from execution.order_manager import OrderManager
        from config.config_loader import load_config

        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = True
        provider.async_cancel_order = AsyncMock()

        om = OrderManager(cfg=cfg, provider=provider)
        om._resting["0xTOKEN"] = self._make_resting("0xTOKEN", "o1")

        await om.cancel_all_resting("shutdown")

        self.assertEqual(om._resting, {})
        provider.async_cancel_order.assert_not_called()

    async def test_cancel_all_resting_cancels_each_live_order(self):
        from execution.order_manager import OrderManager
        from config.config_loader import load_config

        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = False
        provider.async_cancel_order = AsyncMock(return_value={"success": True})

        om = OrderManager(cfg=cfg, provider=provider)
        om._resting["0xA"] = self._make_resting("0xA", "o-a")
        om._resting["0xB"] = self._make_resting("0xB", "o-b")

        await om.cancel_all_resting("shutdown")

        self.assertEqual(om._resting, {})
        self.assertEqual(provider.async_cancel_order.await_count, 2)


# =========================================================================
# FIX 4: log rotation
# =========================================================================

class TestLogRotation(unittest.TestCase):
    def test_setup_logging_uses_rotating_file_handler_with_caps(self):
        import logging
        import tempfile
        from logging.handlers import RotatingFileHandler
        from infra.logging_setup import setup_logging

        root = logging.getLogger()
        before = list(root.handlers)
        log_path = str(Path(tempfile.gettempdir()) / "test_fader_rotate.log")
        try:
            setup_logging(level="INFO", log_file=log_path, max_bytes=12345, backup_count=2)
            new_handlers = [h for h in root.handlers if h not in before]
            rotating = [h for h in new_handlers if isinstance(h, RotatingFileHandler)]
            self.assertTrue(rotating, "setup_logging must add a RotatingFileHandler")
            self.assertEqual(rotating[0].maxBytes, 12345)
            self.assertEqual(rotating[0].backupCount, 2)
        finally:
            for h in list(root.handlers):
                if h not in before:
                    root.removeHandler(h)
                    h.close()
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_setup_logging_default_caps(self):
        import inspect
        from infra.logging_setup import setup_logging

        sig = inspect.signature(setup_logging)
        self.assertEqual(sig.parameters["max_bytes"].default, 10_000_000)
        self.assertEqual(sig.parameters["backup_count"].default, 5)


# =========================================================================
# FIX 8: execute_write retry-on-locked
# =========================================================================

class TestDbWriteRetry(unittest.TestCase):
    def test_execute_write_retries_then_succeeds_on_locked(self):
        from infra import db as db_mod

        failing_conn = MagicMock()
        failing_conn.execute.side_effect = sqlite3.OperationalError("database is locked")

        succeeding_cur = MagicMock()
        succeeding_cur.rowcount = 1
        succeeding_conn = MagicMock()
        succeeding_conn.execute.return_value = succeeding_cur

        with patch.object(db_mod, "get_connection", side_effect=[failing_conn, succeeding_conn]):
            with patch.object(db_mod.time, "sleep", return_value=None):
                result = db_mod.execute_write(
                    "UPDATE x SET y=1", (), retries=3, base_sleep=0.01
                )

        self.assertEqual(result, 1)
        failing_conn.close.assert_called_once()
        succeeding_conn.close.assert_called_once()

    def test_execute_write_raises_immediately_on_non_lock_error(self):
        """Corruption/constraint errors must NOT be retried — fail loud."""
        from infra import db as db_mod

        bad_conn = MagicMock()
        bad_conn.execute.side_effect = sqlite3.OperationalError("no such table: x")

        with patch.object(db_mod, "get_connection", return_value=bad_conn):
            with self.assertRaises(sqlite3.OperationalError):
                db_mod.execute_write("UPDATE x SET y=1", (), retries=3, base_sleep=0.01)

        # Must not retry — get_connection called exactly once.
        bad_conn.execute.assert_called_once()

    def test_execute_write_exhausts_retries_and_returns_zero(self):
        from infra import db as db_mod

        conn = MagicMock()
        conn.execute.side_effect = sqlite3.OperationalError("database is locked")

        with patch.object(db_mod, "get_connection", return_value=conn):
            with patch.object(db_mod.time, "sleep", return_value=None):
                result = db_mod.execute_write(
                    "UPDATE x SET y=1", (), retries=2, base_sleep=0.01
                )

        self.assertEqual(result, 0)
        self.assertEqual(conn.execute.call_count, 2)

    def test_busy_timeout_bumped_to_8000(self):
        src = (_FADER_ROOT / "infra" / "db.py").read_text()
        self.assertIn("PRAGMA busy_timeout=8000", src)

    def test_execute_write_works_against_real_db(self):
        """Integration smoke test against a real (unlocked) DB connection."""
        db_path = _FADER_ROOT / "tests" / "test_fader_execwrite.db"
        from infra.db import set_db_path, init_db, execute_write, get_connection
        set_db_path(db_path)
        if db_path.exists():
            db_path.unlink()
        init_db()
        try:
            rc = execute_write(
                "INSERT OR REPLACE INTO engine_state (key, value_json, updated_at) VALUES (?, ?, ?)",
                ("test_key", '"test_value"', datetime.now(timezone.utc).isoformat()),
            )
            self.assertEqual(rc, 1)
            conn = get_connection()
            try:
                row = conn.execute(
                    "SELECT value_json FROM engine_state WHERE key='test_key'"
                ).fetchone()
            finally:
                conn.close()
            self.assertEqual(row["value_json"], '"test_value"')
        finally:
            if db_path.exists():
                db_path.unlink()


# =========================================================================
# Config: new FeedConfig fields + hot-reload wiring
# =========================================================================

class TestFeedConfigFields(unittest.TestCase):
    def test_new_fields_present_with_defaults(self):
        from config.config_loader import FeedConfig

        fc = FeedConfig()
        self.assertEqual(fc.ws_force_reconnect_s, 90)
        self.assertEqual(fc.ws_ping_interval_s, 10)
        self.assertEqual(fc.ws_pong_timeout_s, 25)
        self.assertFalse(fc.ws_expect_pong)
        self.assertEqual(fc.resync_concurrency, 8)
        self.assertEqual(fc.executor_workers, 16)

    def test_config_yaml_has_new_feed_fields(self):
        import yaml

        raw = yaml.safe_load((_FADER_ROOT / "config" / "config.yaml").read_text())
        feed = raw.get("feed", {})
        for key in (
            "ws_force_reconnect_s", "ws_ping_interval_s", "ws_pong_timeout_s",
            "ws_expect_pong", "resync_concurrency", "executor_workers",
        ):
            self.assertIn(key, feed, f"config.yaml feed: block missing {key!r}")

    def test_apply_hot_includes_four_hot_fields_excludes_cold_ones(self):
        src = (_FADER_ROOT / "config" / "config_loader.py").read_text()
        idx = src.index("def _apply_hot")
        body = src[idx:]
        for hot in (
            "c.feed.ws_force_reconnect_s = n.feed.ws_force_reconnect_s",
            "c.feed.ws_ping_interval_s = n.feed.ws_ping_interval_s",
            "c.feed.ws_pong_timeout_s = n.feed.ws_pong_timeout_s",
            "c.feed.ws_expect_pong = n.feed.ws_expect_pong",
        ):
            self.assertIn(hot, body, f"_apply_hot must hot-reload {hot!r}")
        self.assertNotIn(
            "c.feed.resync_concurrency = n.feed.resync_concurrency", body,
            "resync_concurrency is COLD — must not be hot-reloaded",
        )
        self.assertNotIn(
            "c.feed.executor_workers = n.feed.executor_workers", body,
            "executor_workers is COLD — must not be hot-reloaded",
        )

    def test_load_config_populates_new_feed_fields(self):
        from config.config_loader import load_config

        cfg = load_config()
        self.assertEqual(cfg.feed.ws_force_reconnect_s, 90)
        self.assertEqual(cfg.feed.resync_concurrency, 8)
        self.assertEqual(cfg.feed.executor_workers, 16)


if __name__ == "__main__":
    unittest.main()
