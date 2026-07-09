"""tests/test_provider_split.py

Phase 2 of the architecture refactor (temp/implementation-plan.md):
BaseProvider / LiveProvider / PaperProvider split, OrderResult, and the
order_manager dispatch that consumes it.

Covers:
  - PaperProvider.place_order returns a FILLED OrderResult with a sim id.
  - LiveProvider.place_order maps both known duplicate shapes to
    status="DUPLICATE" (mandatory dispatch-level test per the plan --
    the existing is_duplicate_error/find_order_by_params unit tests in
    test_live_readiness.py only test the helpers in isolation and would
    not catch a mis-wired dispatch).
  - order_manager._place_market on DUPLICATE: recovery via
    find_order_by_params, then UNKNOWN fallback on failure.
  - Provider(...) compatibility alias + make_provider(...) factory.
  - provider.is_paper property replaces the old provider._mode peek.

Run: python -m pytest fader/tests/test_provider_split.py -v
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add fader root to path
_FADER_ROOT = Path(__file__).parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))


class _DbTestCase(unittest.IsolatedAsyncioTestCase):
    db_name = "test_fader_provider_split.db"

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
# PaperProvider
# =========================================================================

class TestPaperProviderPlaceOrder(unittest.TestCase):
    def test_place_order_returns_filled_with_sim_id(self):
        from execution.provider import PaperProvider
        from infra.rate_limiter import RateLimiter

        provider = PaperProvider(limiter=RateLimiter())
        result = provider.place_order("0xTOKEN", "BUY", 0.85, 10.0, "MARKET")

        self.assertTrue(result.success)
        self.assertEqual(result.status, "FILLED")
        self.assertIsNotNone(result.order_id)
        self.assertTrue(result.order_id.startswith("SIM-"))
        self.assertEqual(result.filled_price, 0.85)

    def test_is_paper_true(self):
        from execution.provider import PaperProvider
        from infra.rate_limiter import RateLimiter
        provider = PaperProvider(limiter=RateLimiter())
        self.assertTrue(provider.is_paper)

    def test_fetch_usdc_allowance_zero(self):
        from execution.provider import PaperProvider
        from infra.rate_limiter import RateLimiter
        provider = PaperProvider(limiter=RateLimiter())
        self.assertEqual(provider.fetch_usdc_allowance(), 0.0)

    def test_fetch_open_orders_empty_list(self):
        """Paper mode has no API to fail against -- always []."""
        from execution.provider import PaperProvider
        from infra.rate_limiter import RateLimiter
        provider = PaperProvider(limiter=RateLimiter())
        self.assertEqual(provider.fetch_open_orders(), [])


class TestPaperProviderBankroll(_DbTestCase):
    async def test_fetch_usdc_balance_subtracts_deployed(self):
        """PaperProvider computes bankroll via PositionsRepo.deployed_total
        directly -- the provider -> engine.risk reverse import dies here."""
        from execution.provider import PaperProvider
        from infra.rate_limiter import RateLimiter
        from infra.db import get_connection
        from datetime import datetime, timezone

        conn = get_connection()
        try:
            conn.execute(
                """INSERT INTO positions
                   (position_id, slug, condition_id, token_id, outcome,
                    entry_price, size, notional, status, opened_at, source)
                VALUES ('p1', 's', 'c', 't1', 'No', 0.85, 10.0, 8.5, 'OPEN', ?, 'ENGINE_FILL')""",
                (datetime.now(timezone.utc).isoformat(),),
            )
            conn.commit()
        finally:
            conn.close()

        provider = PaperProvider(limiter=RateLimiter(), paper_bankroll_usdc=100.0)
        balance = provider.fetch_usdc_balance()
        self.assertAlmostEqual(balance, 100.0 - 8.5)


# =========================================================================
# LiveProvider -- fetch_open_orders None-on-error
# =========================================================================

class TestLiveProviderFetchOpenOrders(unittest.TestCase):
    def test_returns_none_on_error(self):
        """fetch_open_orders() returns None (not []) on API failure --
        Phase 2 intentional behavior change so callers can distinguish
        "API unavailable" from "no orders"."""
        from execution.provider import LiveProvider
        from infra.rate_limiter import RateLimiter

        provider = LiveProvider(limiter=RateLimiter())
        mock_client = MagicMock()
        mock_client.get_orders.side_effect = RuntimeError("API down")
        provider._clob_client = mock_client

        result = provider.fetch_open_orders()
        self.assertIsNone(result)

    def test_returns_list_on_success(self):
        from execution.provider import LiveProvider
        from infra.rate_limiter import RateLimiter

        provider = LiveProvider(limiter=RateLimiter())
        mock_client = MagicMock()
        mock_client.get_orders.return_value = [
            {"id": "o1", "asset_id": "t1", "side": "BUY", "price": "0.85",
             "original_size": "10.0", "size_matched": "0.0", "status": "LIVE"},
        ]
        provider._clob_client = mock_client

        result = provider.fetch_open_orders()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["order_id"], "o1")


# =========================================================================
# LiveProvider.place_order -- DUPLICATE via both known shapes
# =========================================================================

class TestLiveProviderDuplicateShapes(unittest.TestCase):
    """Both duplicate shapes must map to status="DUPLICATE", keyed by the
    is_duplicate_error SIGNAL, not by which code path produced it."""

    def test_success_false_body_duplicate_shape(self):
        """Shape (a): CLOB 200 response with success=false + duplicate
        errorMsg (provider.py's original :377-381 shape)."""
        from execution.provider import LiveProvider
        from infra.rate_limiter import RateLimiter

        provider = LiveProvider(limiter=RateLimiter())
        fake_client = MagicMock()
        fake_client.create_and_post_order.return_value = {
            "success": False,
            "errorMsg": "INVALID_ORDER_DUPLICATED: order already exists",
            "orderID": None,
        }
        provider._clob_client = fake_client

        result = provider.place_order("0xT", "BUY", 0.85, 10.0, "MARKET")
        self.assertFalse(result.success)
        self.assertEqual(result.status, "DUPLICATE")
        self.assertIsNone(result.order_id)

    def test_success_false_body_non_duplicate_is_rejected(self):
        """A success=false body WITHOUT the duplicate signal must map to
        REJECTED, not DUPLICATE (do not conflate the two)."""
        from execution.provider import LiveProvider
        from infra.rate_limiter import RateLimiter

        provider = LiveProvider(limiter=RateLimiter())
        fake_client = MagicMock()
        fake_client.create_and_post_order.return_value = {
            "success": False, "errorMsg": "FOK order killed", "orderID": None,
        }
        provider._clob_client = fake_client

        result = provider.place_order("0xT", "BUY", 0.85, 10.0, "MARKET")
        self.assertFalse(result.success)
        self.assertEqual(result.status, "REJECTED")

    def test_exception_string_duplicate_shape(self):
        """Shape (b): duplicate detected via exception string (provider.py's
        original :387-392 shape). Previously returned success=True/
        order_id=None with NO recovery attempt; Phase 2 unifies this into
        DUPLICATE so order_manager's recovery path runs for it too
        (intentional behavior change, listed in the plan)."""
        from execution.provider import LiveProvider
        from infra.rate_limiter import RateLimiter

        provider = LiveProvider(limiter=RateLimiter())
        fake_client = MagicMock()
        fake_client.create_and_post_order.side_effect = RuntimeError(
            "INVALID_ORDER_DUPLICATED (1337)"
        )
        provider._clob_client = fake_client

        result = provider.place_order("0xT", "BUY", 0.85, 10.0, "MARKET")
        self.assertFalse(result.success)
        self.assertEqual(result.status, "DUPLICATE")
        self.assertIsNone(result.order_id)

    def test_exception_string_non_duplicate_is_rejected(self):
        from execution.provider import LiveProvider
        from infra.rate_limiter import RateLimiter

        provider = LiveProvider(limiter=RateLimiter())
        fake_client = MagicMock()
        fake_client.create_and_post_order.side_effect = RuntimeError("Network timeout")
        provider._clob_client = fake_client

        result = provider.place_order("0xT", "BUY", 0.85, 10.0, "MARKET")
        self.assertFalse(result.success)
        self.assertEqual(result.status, "REJECTED")
        self.assertIn("Network timeout", result.error)


# =========================================================================
# order_manager._place_market dispatch on DUPLICATE (mandatory per plan --
# drives the FULL dispatch path, not just the provider-level shape mapping)
# =========================================================================

class TestPlaceMarketDuplicateDispatch(_DbTestCase):
    def _make_om(self, place_result):
        from execution.order_manager import OrderManager
        from config.config_loader import load_config
        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = False
        provider.async_place_order = AsyncMock(return_value=place_result)
        om = OrderManager(cfg=cfg, provider=provider)
        return om, provider

    async def test_duplicate_success_false_shape_recovers_order(self):
        """DUPLICATE from the success=false body shape: recovery succeeds
        via find_order_by_params -> order saved PENDING, position row
        NOT inserted directly here (order stays PENDING until the
        reconciler sees it filled), decision logged as ENTERED."""
        from execution.provider import OrderResult
        from decimal import Decimal

        result = OrderResult(
            success=False, status="DUPLICATE", order_id=None,
            error="INVALID_ORDER_DUPLICATED", raw={},
        )
        om, provider = self._make_om(result)
        provider.find_order_by_params = MagicMock(
            return_value={"order_id": "real-order-1"}
        )

        await om._place_market(
            "test-slug", "0xTOKEN", Decimal("0.85"), 10.0, filters={},
        )

        from infra.db import get_connection
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT order_id, status FROM orders WHERE order_id='real-order-1'"
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row, "recovered order must be saved")
        self.assertEqual(row["status"], "PENDING")

    async def test_duplicate_exception_shape_recovers_order(self):
        """DUPLICATE from the exception-string shape gets the SAME
        recovery attempt as the success=false shape (Phase 2 intentional
        behavior change -- this shape previously fell straight through to
        UNKNOWN with no recovery)."""
        from execution.provider import OrderResult
        from decimal import Decimal

        result = OrderResult(
            success=False, status="DUPLICATE", order_id=None, error=None, raw={},
        )
        om, provider = self._make_om(result)
        provider.find_order_by_params = MagicMock(
            return_value={"order_id": "real-order-2"}
        )

        await om._place_market(
            "test-slug", "0xTOKEN2", Decimal("0.85"), 10.0, filters={},
        )

        from infra.db import get_connection
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT order_id, status FROM orders WHERE order_id='real-order-2'"
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row, "recovered order must be saved")
        self.assertEqual(row["status"], "PENDING")

    async def test_duplicate_unrecovered_falls_back_to_unknown_no_position(self):
        """When find_order_by_params can't recover a real order_id, the
        order is stored as UNKNOWN and NO position row is inserted --
        this holds for EITHER duplicate shape, since the fallback is
        shared (_handle_duplicate)."""
        from execution.provider import OrderResult
        from decimal import Decimal

        result = OrderResult(
            success=False, status="DUPLICATE", order_id=None,
            error="INVALID_ORDER_DUPLICATED", raw={},
        )
        om, provider = self._make_om(result)
        provider.find_order_by_params = MagicMock(return_value=None)

        await om._place_market(
            "test-slug", "0xTOKEN3", Decimal("0.85"), 10.0, filters={},
        )

        from infra.db import get_connection
        conn = get_connection()
        try:
            order_row = conn.execute(
                "SELECT status FROM orders WHERE token_id='0xTOKEN3'"
            ).fetchone()
            pos_row = conn.execute(
                "SELECT 1 FROM positions WHERE token_id='0xTOKEN3'"
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(order_row)
        self.assertEqual(order_row["status"], "UNKNOWN")
        self.assertIsNone(pos_row, "no position row on unrecovered duplicate")


# =========================================================================
# order_manager._place_market dispatch: FILLED / PENDING / REJECTED /
# UNKNOWN (non-duplicate paths, exercised through the unified OrderResult
# dispatch rather than dict.get("simulated"))
# =========================================================================

class TestPlaceMarketStatusDispatch(_DbTestCase):
    def _make_om(self, place_result):
        from execution.order_manager import OrderManager
        from config.config_loader import load_config
        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = True
        provider.async_place_order = AsyncMock(return_value=place_result)
        om = OrderManager(cfg=cfg, provider=provider)
        return om, provider

    async def test_filled_inserts_position(self):
        from execution.provider import OrderResult
        from decimal import Decimal
        from tests.helpers import make_order_result

        result = make_order_result(status="FILLED", order_id="SIM-1", filled_price=0.85)
        om, _ = self._make_om(result)

        await om._place_market(
            "test-slug", "0xTOKF", Decimal("0.85"), 10.0, filters={},
        )

        from infra.db import get_connection
        conn = get_connection()
        try:
            pos = conn.execute(
                "SELECT status FROM positions WHERE token_id='0xTOKF'"
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(pos)
        self.assertEqual(pos["status"], "OPEN")

    async def test_pending_inserts_position_pending_order(self):
        from tests.helpers import make_order_result
        from decimal import Decimal

        result = make_order_result(status="PENDING", order_id="live-order-9")
        om, _ = self._make_om(result)

        await om._place_market(
            "test-slug", "0xTOKP", Decimal("0.85"), 10.0, filters={},
        )

        from infra.db import get_connection
        conn = get_connection()
        try:
            order_row = conn.execute(
                "SELECT status FROM orders WHERE order_id='live-order-9'"
            ).fetchone()
            pos_row = conn.execute(
                "SELECT status FROM positions WHERE token_id='0xTOKP'"
            ).fetchone()
        finally:
            conn.close()
        self.assertEqual(order_row["status"], "PENDING")
        # Position inserted immediately even for PENDING market orders
        # (closes the 60s reconciler gap) -- matches pre-Phase-2 behavior.
        self.assertIsNotNone(pos_row)

    async def test_rejected_no_order_no_position(self):
        from tests.helpers import make_order_result
        from decimal import Decimal

        result = make_order_result(status="REJECTED", success=False, order_id=None,
                                    error="insufficient balance")
        om, _ = self._make_om(result)

        await om._place_market(
            "test-slug", "0xTOKR", Decimal("0.85"), 10.0, filters={},
        )

        from infra.db import get_connection
        conn = get_connection()
        try:
            order_row = conn.execute(
                "SELECT 1 FROM orders WHERE token_id='0xTOKR'"
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNone(order_row, "rejected order must not be saved")

    async def test_unknown_status_with_no_order_id_stores_unknown(self):
        """success=True but no order_id and not a duplicate -> honest API
        gap, stored as UNKNOWN, no position inserted."""
        from tests.helpers import make_order_result
        from decimal import Decimal

        result = make_order_result(status="UNKNOWN", success=True, order_id=None)
        om, _ = self._make_om(result)

        await om._place_market(
            "test-slug", "0xTOKU", Decimal("0.85"), 10.0, filters={},
        )

        from infra.db import get_connection
        conn = get_connection()
        try:
            order_row = conn.execute(
                "SELECT status FROM orders WHERE token_id='0xTOKU'"
            ).fetchone()
            pos_row = conn.execute(
                "SELECT 1 FROM positions WHERE token_id='0xTOKU'"
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(order_row)
        self.assertEqual(order_row["status"], "UNKNOWN")
        self.assertIsNone(pos_row)


# =========================================================================
# Provider(...) compatibility alias + make_provider(...) factory
# =========================================================================

class TestProviderCompatAlias(unittest.TestCase):
    def test_provider_paper_mode_returns_paper_provider(self):
        from execution.provider import Provider, PaperProvider
        from infra.rate_limiter import RateLimiter

        provider = Provider(limiter=RateLimiter(), mode="paper")
        self.assertIsInstance(provider, PaperProvider)
        self.assertTrue(provider.is_paper)

    def test_provider_live_mode_returns_live_provider(self):
        from execution.provider import Provider, LiveProvider
        from infra.rate_limiter import RateLimiter

        provider = Provider(limiter=RateLimiter(), mode="live")
        self.assertIsInstance(provider, LiveProvider)
        self.assertFalse(provider.is_paper)

    def test_provider_paper_bankroll_threads_through(self):
        from execution.provider import Provider
        from infra.rate_limiter import RateLimiter

        provider = Provider(limiter=RateLimiter(), mode="paper", paper_bankroll_usdc=500.0)
        self.assertEqual(provider._paper_bankroll_usdc, 500.0)


class TestMakeProviderFactory(unittest.TestCase):
    def test_make_provider_paper(self):
        from execution.provider import make_provider, PaperProvider
        from infra.rate_limiter import RateLimiter
        from config.config_loader import AppConfig

        cfg = AppConfig()
        cfg.mode = "paper"
        provider = make_provider(cfg, limiter=RateLimiter())
        self.assertIsInstance(provider, PaperProvider)

    def test_make_provider_live(self):
        from execution.provider import make_provider, LiveProvider
        from infra.rate_limiter import RateLimiter
        from config.config_loader import AppConfig

        cfg = AppConfig()
        cfg.mode = "live"
        provider = make_provider(cfg, limiter=RateLimiter())
        self.assertIsInstance(provider, LiveProvider)


if __name__ == "__main__":
    unittest.main()
