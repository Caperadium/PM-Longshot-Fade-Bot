"""tests/test_vps_review_fixes.py

Targeted tests for the live-trading robustness review fixes:
  - R1: strategy loop reads live bankroll via set_bankroll_source.
  - R2: failed cancel keeps the order tracked and blocks the requote replace.
  - R3: reconciler marks vanished orders FILLED when a position is open.
  - R4: circuit breaker auto-resets at UTC day rollover.
  - R5: resting limit notional counts against deployed caps.
  - R6: DB retention pruning removes old decisions/control_commands.
  - R7: provider place_order treats success=false response body as rejection.
  - R8: StrategyLoop.start() is idempotent (dashboard 'start' command).

Run: python -m pytest fader/tests/test_vps_review_fixes.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add fader root to path
_FADER_ROOT = Path(__file__).parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))


class _DbTestCase(unittest.IsolatedAsyncioTestCase):
    db_name = "test_fader_vps_review.db"

    async def asyncSetUp(self):
        os.environ["POLYMARKET_USER_ADDRESS"] = "0xTEST_USER"
        self.db_path = _FADER_ROOT / "tests" / self.db_name
        from infra.db import set_db_path, init_db
        set_db_path(self.db_path)
        if self.db_path.exists():
            self.db_path.unlink()
        init_db()

    async def asyncTearDown(self):
        if self.db_path.exists():
            self.db_path.unlink()


# =========================================================================
# R1: live bankroll source
# =========================================================================

class TestBankrollSource(unittest.TestCase):
    def _make_loop(self):
        from engine.strategy_loop import StrategyLoop
        return StrategyLoop(
            cfg=MagicMock(), book_store=MagicMock(),
            staleness=MagicMock(), risk=MagicMock(),
        )

    def test_bankroll_tracks_live_source(self):
        sl = self._make_loop()
        sl.set_bankroll(100.0)
        state = {"bal": 100.0}
        sl.set_bankroll_source(lambda: state["bal"])
        self.assertEqual(sl.bankroll, 100.0)
        state["bal"] = 42.5  # poller reconciled a new balance
        self.assertEqual(sl.bankroll, 42.5)

    def test_bankroll_falls_back_to_static_without_source(self):
        sl = self._make_loop()
        sl.set_bankroll(77.0)
        self.assertEqual(sl.bankroll, 77.0)

    def test_bankroll_falls_back_when_source_raises(self):
        sl = self._make_loop()
        sl.set_bankroll(55.0)
        sl.set_bankroll_source(lambda: 1 / 0)
        self.assertEqual(sl.bankroll, 55.0)


# =========================================================================
# Phase 3: BankrollView (reconciler.bankroll_view) -- additive, does not
# touch the float `bankroll`/`bankroll_fn` plumbing tested above.
# =========================================================================

class TestBankrollView(unittest.IsolatedAsyncioTestCase):
    def _make_loop(self):
        from engine.strategy_loop import StrategyLoop
        return StrategyLoop(
            cfg=MagicMock(), book_store=MagicMock(),
            staleness=MagicMock(), risk=MagicMock(),
        )

    def test_no_view_source_wired_returns_none_age(self):
        sl = self._make_loop()
        self.assertIsNone(sl._bankroll_age_s())

    def test_age_reflects_time_since_as_of(self):
        from engine.reconciler import BankrollView
        sl = self._make_loop()
        as_of = time.monotonic() - 45.0
        sl.set_bankroll_view_source(lambda: BankrollView(value=500.0, as_of=as_of))
        age = sl._bankroll_age_s()
        self.assertIsNotNone(age)
        self.assertGreaterEqual(age, 45.0)

    def test_never_reconciled_as_of_zero_returns_none(self):
        from engine.reconciler import BankrollView
        sl = self._make_loop()
        sl.set_bankroll_view_source(lambda: BankrollView(value=0.0, as_of=0.0))
        self.assertIsNone(sl._bankroll_age_s())

    def test_view_source_raising_returns_none(self):
        sl = self._make_loop()
        sl.set_bankroll_view_source(lambda: 1 / 0)
        self.assertIsNone(sl._bankroll_age_s())

    def test_reconciler_bankroll_stays_float_type(self):
        """reconciler.bankroll must remain a bare float -- five call sites
        (main.py set_bankroll/set_bankroll_source, state_publisher's
        json.dumps, strategy_loop's float(...) wrapper) fail silently on
        anything else. bankroll_view is a SEPARATE, additive property."""
        from engine.reconciler import Reconciler
        rc = Reconciler(provider=MagicMock(), risk=MagicMock())
        self.assertIsInstance(rc.bankroll, float)
        view = rc.bankroll_view
        self.assertEqual(view.value, rc.bankroll)
        self.assertIsInstance(view.as_of, float)

    async def test_bankroll_reconcile_updates_view_as_of(self):
        from engine.reconciler import Reconciler
        provider = MagicMock()
        provider.async_fetch_usdc_balance = AsyncMock(return_value=250.0)
        risk = MagicMock()
        risk.check_breaker_against_bankroll.return_value = False
        rc = Reconciler(provider=provider, risk=risk)

        before = rc.bankroll_view.as_of
        with patch("engine.reconciler.engine_state_repo") as mock_repo:
            await rc._reconcile_bankroll()

        self.assertEqual(rc.bankroll, 250.0)
        self.assertEqual(rc.bankroll_view.value, 250.0)
        self.assertGreater(rc.bankroll_view.as_of, before)
        mock_repo.publish.assert_called_once_with("bankroll", 250.0)


# =========================================================================
# R2: failed cancel keeps order tracked, blocks requote replacement
# =========================================================================

class TestCancelFailureSafety(_DbTestCase):
    db_name = "test_fader_cancel_failure.db"

    def _make_om(self, cancel_result):
        from execution.order_manager import OrderManager, RestingOrder
        from config.config_loader import load_config
        from tests.helpers import make_order_result
        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = False
        provider.async_cancel_order = AsyncMock(return_value=cancel_result)
        provider.async_place_order = AsyncMock(
            return_value=make_order_result(status="PENDING", order_id="new-1")
        )
        om = OrderManager(cfg=cfg, provider=provider)
        om._resting["0xT"] = RestingOrder(
            order_id="o-1", idempotency_key="ik-o-1", slug="s",
            token_id="0xT", price=0.85, size=10.0, notional=8.5,
            placed_at=time.monotonic(), ttl_s=300, mid=0.85,
        )
        return om, provider

    async def test_cancel_failure_retracks_order(self):
        om, _ = self._make_om({"success": False, "error": "api down"})
        ok = await om._cancel_resting("0xT", "requote")
        self.assertFalse(ok)
        self.assertIn("0xT", om._resting)  # still managed; cancel retried later

    async def test_cancel_success_pops_order(self):
        om, _ = self._make_om({"success": True})
        ok = await om._cancel_resting("0xT", "requote")
        self.assertTrue(ok)
        self.assertNotIn("0xT", om._resting)

    async def test_requote_skips_replacement_when_cancel_fails(self):
        om, provider = self._make_om({"success": False, "error": "api down"})
        book = MagicMock()
        book.mid = 0.86
        book.best_ask = 0.87
        await om._requote(om._resting["0xT"], book)
        provider.async_place_order.assert_not_called()  # no double exposure


# =========================================================================
# R3: reconciler FILLED-vs-UNKNOWN via open position
# =========================================================================

class TestReconcilerFilledDetection(_DbTestCase):
    db_name = "test_fader_reconciler_filled.db"

    def _insert_order(self, order_id, token_id, status="PENDING"):
        from infra.db import get_connection
        conn = get_connection()
        try:
            conn.execute(
                """INSERT INTO orders
                   (order_id, idempotency_key, slug, token_id, side, type,
                    price, size, status, created_at)
                   VALUES (?, ?, 'test-slug', ?, 'BUY', 'LIMIT', 0.85, 10.0, ?, ?)""",
                (order_id, f"ik-{order_id}", token_id, status,
                 datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def _insert_open_position(self, token_id):
        from infra.db import get_connection
        conn = get_connection()
        try:
            conn.execute(
                """INSERT INTO positions
                   (position_id, slug, condition_id, token_id, outcome,
                    entry_price, size, notional, status, opened_at, source)
                   VALUES (?, 'test-slug', 'cid', ?, 'No', 0.85, 10.0, 8.5,
                           'OPEN', ?, 'ENGINE_FILL')""",
                (f"pos-{token_id}", token_id,
                 datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def _order_status(self, order_id):
        from infra.db import get_connection
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT status FROM orders WHERE order_id=?", (order_id,)
            ).fetchone()
            return row["status"] if row else None
        finally:
            conn.close()

    async def _reconcile(self):
        from engine.reconciler import Reconciler
        provider = MagicMock()
        provider.is_paper = False
        provider.async_fetch_open_orders = AsyncMock(return_value=[])
        om = MagicMock()
        rec = Reconciler(provider=provider, risk=MagicMock(), order_manager=om)
        await rec._reconcile_orders()
        return om

    async def test_vanished_order_with_open_position_marked_filled(self):
        self._insert_order("filled-1", "0xTOK_A")
        self._insert_open_position("0xTOK_A")
        om = await self._reconcile()
        om.mark_filled.assert_called_once_with("filled-1", "0xTOK_A")
        om.mark_vanished.assert_not_called()

    async def test_vanished_order_without_position_marked_unknown(self):
        self._insert_order("gone-1", "0xTOK_B")
        om = await self._reconcile()
        self.assertEqual(self._order_status("gone-1"), "UNKNOWN")
        om.mark_vanished.assert_called_once_with("0xTOK_B")
        om.mark_filled.assert_not_called()


# =========================================================================
# R4: breaker UTC day rollover
#
# Phase 3 (Single-owner state) deleted the in-memory _breaker_tripped/
# _tripped_day flags and their self-resetting logic. breaker_tripped is
# now a read-through property backed by the circuit_breaker DB row keyed
# by UTC day (via BreakerRepo), memoized <=1s. "Rollover" is now implicit:
# a trip recorded for a past day simply has no row for today, so there is
# nothing to explicitly reset. Tests write through BreakerRepo instead of
# poking deleted fields directly.
# =========================================================================

class TestBreakerDayRollover(_DbTestCase):
    db_name = "test_fader_breaker_rollover.db"

    async def test_tripped_yesterday_clears_today(self):
        from engine.risk import RiskManager
        from persistence.repos import breaker_repo
        rm = RiskManager(daily_loss_pct=5.0)
        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1))
        breaker_repo.trip(yesterday.isoformat())
        self.assertFalse(rm.breaker_tripped)  # no row for today -> untripped

    async def test_tripped_today_stays_tripped(self):
        from engine.risk import RiskManager
        rm = RiskManager(daily_loss_pct=5.0)
        rm._trip(rm.today_utc())
        self.assertTrue(rm.breaker_tripped)

    async def test_allow_entry_uses_rollover_aware_check(self):
        from engine.risk import RiskManager
        from persistence.repos import breaker_repo
        rm = RiskManager(daily_loss_pct=5.0, max_deployed_pct=100.0,
                         per_market_cap_pct=100.0)
        breaker_repo.trip("2020-01-01")
        allowed, reason = rm.allow_entry("slug", 10.0, 1000.0, 0.0, 0.0)
        self.assertTrue(allowed, reason)

    async def test_breaker_trip_persists_across_riskmanager_reinstantiation(self):
        """Restart-safety: a trip recorded by one RiskManager instance
        (simulating a process restart) must be visible to a brand new
        instance with no shared in-memory state -- the DB row is the only
        source of truth."""
        from engine.risk import RiskManager
        rm1 = RiskManager(daily_loss_pct=5.0)
        rm1._trip(rm1.today_utc())
        self.assertTrue(rm1.breaker_tripped)

        rm2 = RiskManager(daily_loss_pct=5.0)  # fresh instance, no memo yet
        self.assertTrue(rm2.breaker_tripped)

    async def test_breaker_reset_clears_trip_for_fresh_instance(self):
        from engine.risk import RiskManager
        rm1 = RiskManager(daily_loss_pct=5.0)
        rm1._trip(rm1.today_utc())
        rm1.reset_breaker()

        rm2 = RiskManager(daily_loss_pct=5.0)
        self.assertFalse(rm2.breaker_tripped)

    async def test_breaker_reset_command_round_trip(self):
        """Full dashboard -> engine round trip: issue_command("breaker_reset")
        writes a control_commands row; ControlConsumer polls it and
        dispatches to the same on_command callback main.py wires up
        (engine/control.py's make_on_command), which calls
        engine.risk.reset_breaker() -- writing through BreakerRepo. A
        second RiskManager instance (simulating the dashboard's own
        read, or a restarted engine) must see the reset."""
        from engine.risk import RiskManager
        from engine.control_consumer import ControlConsumer, issue_command
        from engine.control import make_on_command

        rm = RiskManager(daily_loss_pct=5.0)
        rm._trip(rm.today_utc())
        self.assertTrue(rm.breaker_tripped)

        engine = MagicMock()
        engine.risk = rm
        cfg = MagicMock()
        config_watcher = MagicMock()
        stop_event = asyncio.Event()
        restart_requested = {"flag": False}
        on_command = make_on_command(engine, cfg, config_watcher, stop_event, restart_requested)

        consumer = ControlConsumer(breaker_reset_cb=on_command)
        issue_command("breaker_reset")
        await consumer._process_pending()

        self.assertFalse(rm.breaker_tripped)
        # A second, independent RiskManager sees the same reset (DB is
        # the only source of truth -- no in-memory flag to resync).
        rm2 = RiskManager(daily_loss_pct=5.0)
        self.assertFalse(rm2.breaker_tripped)


# =========================================================================
# R5: resting exposure counts against deployed caps
# =========================================================================

class TestRestingExposure(_DbTestCase):
    db_name = "test_fader_resting_exposure.db"

    async def test_resting_exposure_sums_by_slug(self):
        from execution.order_manager import OrderManager, RestingOrder
        from config.config_loader import load_config
        om = OrderManager(cfg=load_config(), provider=MagicMock())
        for i, (slug, notional) in enumerate(
            [("a", 8.5), ("a", 1.5), ("b", 4.0)]
        ):
            om._resting[f"0xT{i}"] = RestingOrder(
                order_id=f"o{i}", idempotency_key=f"ik{i}", slug=slug,
                token_id=f"0xT{i}", price=0.85, size=10.0, notional=notional,
                placed_at=time.monotonic(), ttl_s=300, mid=0.85,
            )
        total, by_slug = om.resting_exposure()
        self.assertAlmostEqual(total, 14.0)
        self.assertAlmostEqual(by_slug["a"], 10.0)
        self.assertAlmostEqual(by_slug["b"], 4.0)


# =========================================================================
# R6: DB retention pruning
# =========================================================================

class TestRetentionPruning(_DbTestCase):
    db_name = "test_fader_retention.db"

    async def test_prune_removes_only_old_rows(self):
        from infra.db import get_connection
        from engine.pollers import Pollers

        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        new_ts = datetime.now(timezone.utc).isoformat()
        conn = get_connection()
        try:
            for ts, slug in [(old_ts, "old"), (new_ts, "new")]:
                conn.execute(
                    "INSERT INTO decisions (ts, slug, decision, reason) "
                    "VALUES (?, ?, 'REJECTED', 'x')",
                    (ts, slug),
                )
            conn.execute(
                "INSERT INTO control_commands (ts, command, status) "
                "VALUES (?, 'stop', 'DONE')",
                (old_ts,),
            )
            conn.execute(
                "INSERT INTO control_commands (ts, command, status) "
                "VALUES (?, 'stop', 'PENDING')",
                (old_ts,),
            )
            conn.commit()
        finally:
            conn.close()

        Pollers._prune_old_rows(retention_days=14)

        conn = get_connection()
        try:
            slugs = [r["slug"] for r in
                     conn.execute("SELECT slug FROM decisions").fetchall()]
            cmd_statuses = [r["status"] for r in
                            conn.execute("SELECT status FROM control_commands").fetchall()]
        finally:
            conn.close()
        self.assertEqual(slugs, ["new"])
        # old PENDING command survives (never silently drop unprocessed cmds)
        self.assertEqual(cmd_statuses, ["PENDING"])


# =========================================================================
# R7: place_order success=false body handled as rejection
# =========================================================================

class TestPlaceOrderRejectionBody(unittest.TestCase):
    """Phase 2: place_order returns a typed OrderResult, not a dict --
    success=false without a duplicate signal maps to status='REJECTED'."""

    def test_success_false_body_is_failure(self):
        from execution.provider import Provider
        provider = Provider(limiter=MagicMock(), mode="live")
        fake_client = MagicMock()
        fake_client.create_and_post_order.return_value = {
            "success": False, "errorMsg": "FOK order killed", "orderID": None,
        }
        provider._clob_client = fake_client
        result = provider.place_order("0xT", "BUY", 0.85, 10.0, "MARKET")
        self.assertFalse(result.success)
        self.assertEqual(result.status, "REJECTED")
        self.assertIn("FOK", result.error)

    def test_success_true_body_passes_through(self):
        from execution.provider import Provider
        provider = Provider(limiter=MagicMock(), mode="live")
        fake_client = MagicMock()
        fake_client.create_and_post_order.return_value = {
            "success": True, "orderID": "0xORDER",
        }
        provider._clob_client = fake_client
        result = provider.place_order("0xT", "BUY", 0.85, 10.0, "MARKET")
        self.assertTrue(result.success)
        self.assertEqual(result.status, "PENDING")
        self.assertEqual(result.order_id, "0xORDER")


# =========================================================================
# R8: StrategyLoop.start() idempotent
# =========================================================================

class TestStrategyLoopStartIdempotent(unittest.IsolatedAsyncioTestCase):
    async def test_double_start_keeps_single_task(self):
        from engine.strategy_loop import StrategyLoop
        cfg = MagicMock()
        cfg.feed.decision_interval_s = 10.0
        cfg.mode = "paper"
        sl = StrategyLoop(cfg=cfg, book_store=MagicMock(),
                          staleness=MagicMock(), risk=MagicMock())
        with patch.object(sl, "_tick", new=AsyncMock()):
            await sl.start()
            first_task = sl._task
            await sl.start()
            self.assertIs(sl._task, first_task)
            await sl.stop()
            try:
                await first_task
            except asyncio.CancelledError:
                pass

    async def test_start_after_stop_restarts(self):
        from engine.strategy_loop import StrategyLoop
        cfg = MagicMock()
        cfg.feed.decision_interval_s = 10.0
        cfg.mode = "paper"
        sl = StrategyLoop(cfg=cfg, book_store=MagicMock(),
                          staleness=MagicMock(), risk=MagicMock())
        with patch.object(sl, "_tick", new=AsyncMock()):
            await sl.start()
            old_task = sl._task
            await sl.stop()
            try:
                await old_task
            except asyncio.CancelledError:
                pass
            await sl.start()
            self.assertIsNot(sl._task, old_task)
            self.assertTrue(sl._running)
            await sl.stop()
            try:
                await sl._task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    unittest.main()
