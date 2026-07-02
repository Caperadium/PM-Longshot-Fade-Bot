"""tests/test_paper_resolution.py

Targeted tests for the paper-mode position resolution + paper position
integrity fix plan:
  - D1/D2/M2: engine.reconciler._reconcile_paper_resolutions polls Gamma
    per open paper position's slug, parses outcomes/outcomePrices (JSON
    strings, real Gamma shape), and closes resolved positions with the
    {0,1} payout PnL formula.
  - D3: full_reconcile no longer zero-closes paper positions on startup.
  - D5: paper resolution mirrors the live close's risk/alert path
    (record_pnl_event + telegram alert).
  - D6: OrderManager._insert_position falls back to a per-token position_id
    when market_info is None, instead of colliding every paper limit fill
    onto the same row (P3).
  - P4: cancel_resting_for_disabled_slugs matches mixed-case slugs against
    a lowercase series_filter.
  - M1: paper close_all marks OPEN positions CLOSED directly (no venue to
    sell into) instead of leaving them for a resolver that will never run
    again once close_all has fired.

Run: python -m pytest fader/tests/test_paper_resolution.py -v
"""

from __future__ import annotations

import json
import os
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


def _swallow_fire(coro=None, *a, **k):
    """Stand-in for telegram.fire that doesn't need a running event loop
    and doesn't leave the alert coroutine dangling (avoids a "coroutine
    was never awaited" warning), same pattern as test_stability_fixes.py.
    """
    if hasattr(coro, "close"):
        coro.close()


class _DbTestCase(unittest.IsolatedAsyncioTestCase):
    db_name = "test_fader_paper_resolution.db"

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

    # -- shared DB helpers --------------------------------------------

    def _insert_open_position(
        self, position_id, slug, token_id, outcome="No",
        entry_price=0.85, size=10.0,
    ):
        from infra.db import get_connection
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO positions
                  (position_id, slug, condition_id, token_id, outcome,
                   entry_price, size, notional, status, opened_at, source)
                VALUES (?, ?, 'cid', ?, ?, ?, ?, ?, 'OPEN', ?, 'ENGINE_FILL')
                """,
                (position_id, slug, token_id, outcome, entry_price, size,
                 entry_price * size, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def _get_position(self, position_id):
        from infra.db import get_connection
        conn = get_connection()
        try:
            return conn.execute(
                "SELECT * FROM positions WHERE position_id=?", (position_id,)
            ).fetchone()
        finally:
            conn.close()

    def _open_count(self):
        from infra.db import get_connection
        conn = get_connection()
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM positions WHERE status='OPEN'"
            ).fetchone()[0]
        finally:
            conn.close()

    def _make_reconciler(self, risk=None, order_manager=None):
        from engine.reconciler import Reconciler
        provider = MagicMock()
        provider._mode = "paper"
        return Reconciler(
            provider=provider, risk=risk or MagicMock(), order_manager=order_manager,
        )


def _gamma_meta(outcomes, prices, closed=True, as_json_strings=True):
    """Build a fake Gamma /markets response. Real Gamma shape encodes
    both outcomes and outcomePrices as JSON strings (M2)."""
    if as_json_strings:
        return {
            "outcomes": json.dumps(outcomes),
            "outcomePrices": json.dumps(prices),
            "closed": closed,
        }
    return {"outcomes": outcomes, "outcomePrices": prices, "closed": closed}


# =========================================================================
# D1/D2/M2: _reconcile_paper_resolutions
# =========================================================================

class TestPaperResolution(_DbTestCase):
    db_name = "test_fader_paper_resolution_core.db"

    async def test_no_wins_closes_with_positive_pnl(self):
        """held='No' matches the resolved winner -> payout=1, pnl=(1-entry)*size."""
        self._insert_open_position("pos-1", "slug-a", "0xTOKA", "No", 0.85, 10.0)
        risk = MagicMock()

        meta = _gamma_meta(["Yes", "No"], ["0", "1"], closed=True)  # No wins
        import engine.reconciler as reconciler_mod
        with patch.object(reconciler_mod, "fetch_market_metadata", return_value=meta), \
             patch("infra.telegram.fire", side_effect=_swallow_fire):
            rec = self._make_reconciler(risk=risk)
            await rec._reconcile_paper_resolutions()

        row = self._get_position("pos-1")
        self.assertEqual(row["status"], "CLOSED")
        self.assertAlmostEqual(row["realized_pnl"], (1.0 - 0.85) * 10.0)
        risk.record_pnl_event.assert_called_once()
        self.assertAlmostEqual(risk.record_pnl_event.call_args[0][0], 1.5)

    async def test_yes_wins_closes_with_negative_pnl(self):
        """held='No' but YES resolved -> payout=0, pnl=-entry*size (a loss)."""
        self._insert_open_position("pos-2", "slug-b", "0xTOKB", "No", 0.85, 10.0)
        risk = MagicMock()

        meta = _gamma_meta(["Yes", "No"], ["1", "0"], closed=True)  # Yes wins
        import engine.reconciler as reconciler_mod
        with patch.object(reconciler_mod, "fetch_market_metadata", return_value=meta), \
             patch("infra.telegram.fire", side_effect=_swallow_fire):
            rec = self._make_reconciler(risk=risk)
            await rec._reconcile_paper_resolutions()

        row = self._get_position("pos-2")
        self.assertEqual(row["status"], "CLOSED")
        self.assertAlmostEqual(row["realized_pnl"], -8.5)
        risk.record_pnl_event.assert_called_once()
        self.assertAlmostEqual(risk.record_pnl_event.call_args[0][0], -8.5)

    async def test_market_not_closed_stays_open(self):
        self._insert_open_position("pos-3", "slug-c", "0xTOKC", "No", 0.85, 10.0)
        risk = MagicMock()

        meta = _gamma_meta(["Yes", "No"], ["0", "1"], closed=False)  # not resolved
        import engine.reconciler as reconciler_mod
        with patch.object(reconciler_mod, "fetch_market_metadata", return_value=meta), \
             patch("infra.telegram.fire", side_effect=_swallow_fire):
            rec = self._make_reconciler(risk=risk)
            await rec._reconcile_paper_resolutions()

        row = self._get_position("pos-3")
        self.assertEqual(row["status"], "OPEN")
        risk.record_pnl_event.assert_not_called()

    async def test_metadata_none_stays_open_no_exception(self):
        self._insert_open_position("pos-4", "slug-d", "0xTOKD", "No", 0.85, 10.0)
        risk = MagicMock()

        import engine.reconciler as reconciler_mod
        with patch.object(reconciler_mod, "fetch_market_metadata", return_value=None), \
             patch("infra.telegram.fire", side_effect=_swallow_fire):
            rec = self._make_reconciler(risk=risk)
            await rec._reconcile_paper_resolutions()  # must not raise

        row = self._get_position("pos-4")
        self.assertEqual(row["status"], "OPEN")
        risk.record_pnl_event.assert_not_called()

    async def test_malformed_outcome_prices_stays_open_no_exception(self):
        """Length mismatch / garbage in outcomePrices -> _resolution_from_
        outcome_prices swallows the error and returns "" -> leave OPEN."""
        self._insert_open_position("pos-5", "slug-e", "0xTOKE", "No", 0.85, 10.0)
        risk = MagicMock()

        meta = {
            "outcomes": json.dumps(["Yes", "No"]),
            "outcomePrices": "not-json-and-wrong-length",
            "closed": True,
        }
        import engine.reconciler as reconciler_mod
        with patch.object(reconciler_mod, "fetch_market_metadata", return_value=meta), \
             patch("infra.telegram.fire", side_effect=_swallow_fire):
            rec = self._make_reconciler(risk=risk)
            await rec._reconcile_paper_resolutions()  # must not raise

        row = self._get_position("pos-5")
        self.assertEqual(row["status"], "OPEN")
        risk.record_pnl_event.assert_not_called()

    async def test_outcomes_and_prices_as_json_strings_parsed_correctly(self):
        """Real Gamma shape (M2): both outcomes and outcomePrices are JSON
        *strings*, not lists. Caller must json.loads(outcomes) before
        calling _resolution_from_outcome_prices."""
        self._insert_open_position("pos-6", "slug-f", "0xTOKF", "No", 0.60, 20.0)
        risk = MagicMock()

        meta = {
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0", "1"]',
            "closed": True,
        }
        import engine.reconciler as reconciler_mod
        with patch.object(reconciler_mod, "fetch_market_metadata", return_value=meta), \
             patch("infra.telegram.fire", side_effect=_swallow_fire):
            rec = self._make_reconciler(risk=risk)
            await rec._reconcile_paper_resolutions()

        row = self._get_position("pos-6")
        self.assertEqual(row["status"], "CLOSED")
        self.assertAlmostEqual(row["realized_pnl"], (1.0 - 0.60) * 20.0)

    async def test_multiple_positions_only_resolved_ones_close(self):
        self._insert_open_position("pos-7a", "slug-g", "0xTOKG", "No", 0.85, 10.0)
        self._insert_open_position("pos-7b", "slug-h", "0xTOKH", "No", 0.90, 5.0)
        risk = MagicMock()

        closed_meta = _gamma_meta(["Yes", "No"], ["0", "1"], closed=True)
        open_meta = _gamma_meta(["Yes", "No"], ["0.5", "0.5"], closed=False)

        def _fake_fetch(slug):
            return closed_meta if slug == "slug-g" else open_meta

        import engine.reconciler as reconciler_mod
        with patch.object(reconciler_mod, "fetch_market_metadata", side_effect=_fake_fetch), \
             patch("infra.telegram.fire", side_effect=_swallow_fire):
            rec = self._make_reconciler(risk=risk)
            await rec._reconcile_paper_resolutions()

        self.assertEqual(self._get_position("pos-7a")["status"], "CLOSED")
        self.assertEqual(self._get_position("pos-7b")["status"], "OPEN")
        risk.record_pnl_event.assert_called_once()


# =========================================================================
# D3: full_reconcile no longer zero-closes paper positions on startup
# =========================================================================

class TestFullReconcileNoZeroClose(_DbTestCase):
    db_name = "test_fader_paper_full_reconcile.db"

    async def test_full_reconcile_paper_books_real_pnl_not_zero_close(self):
        self._insert_open_position("pos-full-1", "slug-i", "0xTOKI", "No", 0.85, 10.0)
        risk = MagicMock()
        risk.check_breaker_against_bankroll = MagicMock(return_value=False)

        meta = _gamma_meta(["Yes", "No"], ["0", "1"], closed=True)  # No wins

        import engine.reconciler as reconciler_mod
        from engine.reconciler import Reconciler
        provider = MagicMock()
        provider._mode = "paper"
        provider.async_fetch_usdc_balance = AsyncMock(return_value=100.0)

        with patch.object(reconciler_mod, "fetch_market_metadata", return_value=meta), \
             patch("infra.telegram.fire", side_effect=_swallow_fire):
            rec = Reconciler(provider=provider, risk=risk, order_manager=None)
            await rec.full_reconcile()

        row = self._get_position("pos-full-1")
        # If the old startup zero-close block still ran, this would be
        # CLOSED with realized_pnl == 0.0 regardless of the real outcome.
        self.assertEqual(row["status"], "CLOSED")
        self.assertAlmostEqual(row["realized_pnl"], 1.5)
        self.assertNotEqual(row["realized_pnl"], 0.0)

    async def test_full_reconcile_paper_leaves_unresolved_position_open(self):
        self._insert_open_position("pos-full-2", "slug-j", "0xTOKJ", "No", 0.85, 10.0)
        risk = MagicMock()
        risk.check_breaker_against_bankroll = MagicMock(return_value=False)

        meta = _gamma_meta(["Yes", "No"], ["0.5", "0.5"], closed=False)

        import engine.reconciler as reconciler_mod
        from engine.reconciler import Reconciler
        provider = MagicMock()
        provider._mode = "paper"
        provider.async_fetch_usdc_balance = AsyncMock(return_value=100.0)

        with patch.object(reconciler_mod, "fetch_market_metadata", return_value=meta), \
             patch("infra.telegram.fire", side_effect=_swallow_fire):
            rec = Reconciler(provider=provider, risk=risk, order_manager=None)
            await rec.full_reconcile()  # was previously a blanket zero-close

        row = self._get_position("pos-full-2")
        self.assertEqual(row["status"], "OPEN")


# =========================================================================
# D6: _insert_position fallback position_id when market_info is None
# =========================================================================

class TestInsertPositionFallbackId(_DbTestCase):
    db_name = "test_fader_paper_insert_position.db"

    async def test_two_paper_limit_fills_different_tokens_produce_two_rows(self):
        from execution.order_manager import OrderManager
        from config.config_loader import load_config

        cfg = load_config()
        provider = MagicMock()
        provider._mode = "paper"
        om = OrderManager(cfg=cfg, provider=provider)

        om._insert_position(
            "slug-k", "0xTOKK", 0.85, 10.0, 8.5, "order-a", "idem-a", None,
        )
        om._insert_position(
            "slug-l", "0xTOKL", 0.90, 5.0, 4.5, "order-b", "idem-b", None,
        )

        from infra.db import get_connection
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT position_id, token_id FROM positions ORDER BY token_id"
            ).fetchall()
        finally:
            conn.close()

        self.assertEqual(len(rows), 2)
        ids = {r["position_id"] for r in rows}
        self.assertEqual(
            ids, {"0xTEST_USER:0xTOKK:0", "0xTEST_USER:0xTOKL:0"},
        )

    async def test_place_limit_paper_fill_threads_market_info_through(self):
        """M3: enter() threads market_info into _place_limit's paper branch,
        so a paper limit fill with market_info gets the canonical
        {user}:{condition_id}:{outcome_index} id, not the token fallback."""
        from execution.order_manager import OrderManager
        from execution.provider import MarketInfo
        from config.config_loader import load_config

        cfg = load_config()
        provider = MagicMock()
        provider._mode = "paper"
        om = OrderManager(cfg=cfg, provider=provider)

        book = MagicMock()
        book.best_ask = 0.90
        book.best_bid = 0.80
        book.spread_cents = 10.0  # forces the limit path
        book.mid = 0.85

        market_info = MarketInfo(
            slug="slug-m", condition_id="cond-m", token_id="0xTOKM",
            outcome="No", outcome_index=0, question="", end_date_iso="",
            active=True, closed=False,
        )

        await om.enter(
            "slug-m", "0xTOKM", book, 10.0, filters={}, market_info=market_info,
        )

        from infra.db import get_connection
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT position_id FROM positions WHERE token_id='0xTOKM'"
            ).fetchone()
        finally:
            conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row["position_id"], "0xTEST_USER:cond-m:0")


# =========================================================================
# P4: cancel_resting_for_disabled_slugs case-insensitive filter match
# =========================================================================

class TestCancelRestingCaseInsensitive(_DbTestCase):
    db_name = "test_fader_paper_cancel_resting.db"

    async def test_mixed_case_slug_matches_lowercase_filter(self):
        from execution.order_manager import OrderManager, RestingOrder
        from config.config_loader import AppConfig, SlugRow

        cfg = AppConfig()
        cfg.slugs = [
            SlugRow(
                slug="bitcoin-above-on", enabled=True, market_kind="series",
                series_filter="bitcoin-above",
            ),
        ]
        provider = MagicMock()
        provider._mode = "live"
        provider.async_cancel_order = AsyncMock(return_value={"success": True})

        om = OrderManager(cfg=cfg, provider=provider)
        # Slug as it would be stored from discovery: mixed case.
        om._resting["0xTOKN"] = RestingOrder(
            order_id="o-n", idempotency_key="ik-o-n",
            slug="Bitcoin-Above-100k-On-July-1", token_id="0xTOKN",
            price=0.85, size=10.0, notional=8.5,
            placed_at=time.monotonic(), ttl_s=300, mid=0.85,
        )

        n = await om.cancel_resting_for_disabled_slugs()

        self.assertEqual(n, 0)
        self.assertIn("0xTOKN", om._resting)
        provider.async_cancel_order.assert_not_called()

    async def test_slug_not_matching_any_enabled_filter_is_cancelled(self):
        from execution.order_manager import OrderManager, RestingOrder
        from config.config_loader import AppConfig, SlugRow

        cfg = AppConfig()
        cfg.slugs = [
            SlugRow(
                slug="bitcoin-above-on", enabled=True, market_kind="series",
                series_filter="bitcoin-above",
            ),
        ]
        provider = MagicMock()
        provider._mode = "live"
        provider.async_cancel_order = AsyncMock(return_value={"success": True})

        om = OrderManager(cfg=cfg, provider=provider)
        om._resting["0xTOKO"] = RestingOrder(
            order_id="o-o", idempotency_key="ik-o-o",
            slug="Ethereum-Above-5k-On-July-1", token_id="0xTOKO",
            price=0.85, size=10.0, notional=8.5,
            placed_at=time.monotonic(), ttl_s=300, mid=0.85,
        )

        n = await om.cancel_resting_for_disabled_slugs()

        self.assertEqual(n, 1)
        self.assertNotIn("0xTOKO", om._resting)


# =========================================================================
# M1: paper close_all closes OPEN positions directly; no resurrection
# =========================================================================

class TestPaperCloseAll(_DbTestCase):
    db_name = "test_fader_paper_close_all.db"

    async def test_paper_close_all_closes_all_open_positions_zero_pnl(self):
        from execution.order_manager import OrderManager, RestingOrder
        from config.config_loader import load_config

        self._insert_open_position("pos-ca-1", "slug-p", "0xTOKP", "No", 0.85, 10.0)
        self._insert_open_position("pos-ca-2", "slug-q", "0xTOKQ", "No", 0.90, 5.0)

        cfg = load_config()
        provider = MagicMock()
        provider._mode = "paper"
        provider.async_cancel_all = AsyncMock(return_value={"success": True})

        om = OrderManager(cfg=cfg, provider=provider)
        om._resting["0xTOKP"] = RestingOrder(
            order_id="o-p", idempotency_key="ik-o-p", slug="slug-p",
            token_id="0xTOKP", price=0.85, size=10.0, notional=8.5,
            placed_at=time.monotonic(), ttl_s=300, mid=0.85,
        )

        await om.close_all()

        self.assertEqual(om._resting, {})
        self.assertEqual(self._open_count(), 0)
        row1 = self._get_position("pos-ca-1")
        row2 = self._get_position("pos-ca-2")
        self.assertEqual(row1["status"], "CLOSED")
        self.assertEqual(row1["realized_pnl"], 0.0)
        self.assertEqual(row2["status"], "CLOSED")
        self.assertEqual(row2["realized_pnl"], 0.0)
        provider.async_cancel_all.assert_awaited_once()

    async def test_full_reconcile_after_paper_close_all_no_resurrection(self):
        from execution.order_manager import OrderManager
        from config.config_loader import load_config
        from engine.reconciler import Reconciler

        self._insert_open_position("pos-ca-3", "slug-r", "0xTOKR", "No", 0.85, 10.0)

        cfg = load_config()
        provider = MagicMock()
        provider._mode = "paper"
        provider.async_cancel_all = AsyncMock(return_value={"success": True})
        provider.async_fetch_usdc_balance = AsyncMock(return_value=100.0)

        om = OrderManager(cfg=cfg, provider=provider)
        await om.close_all()
        self.assertEqual(self._open_count(), 0)

        risk = MagicMock()
        risk.check_breaker_against_bankroll = MagicMock(return_value=False)

        import engine.reconciler as reconciler_mod
        with patch.object(reconciler_mod, "fetch_market_metadata") as fetch_mock, \
             patch("infra.telegram.fire", side_effect=_swallow_fire):
            rec = Reconciler(provider=provider, risk=risk, order_manager=om)
            await rec.full_reconcile()  # must not raise, must not resurrect

        self.assertEqual(self._open_count(), 0)
        # No OPEN rows left to poll -> Gamma is never even queried.
        fetch_mock.assert_not_called()
        row = self._get_position("pos-ca-3")
        self.assertEqual(row["status"], "CLOSED")
        self.assertEqual(row["realized_pnl"], 0.0)


if __name__ == "__main__":
    unittest.main()
