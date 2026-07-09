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
        provider.is_paper = True
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
        provider.is_paper = True
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
        provider.is_paper = True
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
        provider.is_paper = True
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
        """M3: enter() threads market_info into _place_limit's fill path,
        so a paper limit fill with market_info gets the canonical
        {user}:{condition_id}:{outcome_index} id, not the token fallback.

        Phase 2: paper/live limit dispatch is unified via OrderResult.status
        (no more provider._mode special-casing inside _place_limit), so the
        mock provider must return a FILLED OrderResult from async_place_order
        the way PaperProvider.place_order actually does."""
        from execution.order_manager import OrderManager
        from execution.provider import MarketInfo
        from config.config_loader import load_config
        from tests.helpers import make_order_result

        cfg = load_config()
        provider = MagicMock()
        provider.is_paper = True
        provider.async_place_order = AsyncMock(
            return_value=make_order_result(status="FILLED", order_id="SIM-m")
        )
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
        provider.is_paper = False
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
        provider.is_paper = False
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
        provider.is_paper = True
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
        provider.is_paper = True
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


# =========================================================================
# Phase 6, item 1: reconcile-failure escalation counter
# =========================================================================

class TestReconcileFailureEscalation(_DbTestCase):
    db_name = "test_fader_reconcile_escalation.db"

    def _get_state(self, key):
        from infra.db import get_connection
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT value_json FROM engine_state WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                return None
            import json as _json
            return _json.loads(row["value_json"])
        finally:
            conn.close()

    async def test_counter_increments_on_consecutive_paper_failures(self):
        rec = self._make_reconciler()

        import engine.reconciler as reconciler_mod
        with patch.object(
            reconciler_mod.positions_repo, "open_for_paper_poll",
            side_effect=RuntimeError("boom"),
        ), patch("infra.telegram.fire", side_effect=_swallow_fire):
            for expected in (1, 2, 3):
                await rec._reconcile_paper_resolutions()
                self.assertEqual(rec.reconcile_failures, expected)
                self.assertEqual(self._get_state("reconcile_failures"), expected)

    async def test_telegram_alert_fires_after_five_consecutive_misses(self):
        rec = self._make_reconciler()

        import engine.reconciler as reconciler_mod
        with patch.object(
            reconciler_mod.positions_repo, "open_for_paper_poll",
            side_effect=RuntimeError("boom"),
        ), patch("infra.telegram.fire", side_effect=_swallow_fire) as fire_mock:
            for _ in range(4):
                await rec._reconcile_paper_resolutions()
            fire_mock.assert_not_called()

            await rec._reconcile_paper_resolutions()  # 5th consecutive miss
            self.assertEqual(rec.reconcile_failures, 5)
            fire_mock.assert_called_once()

    async def test_counter_resets_on_success(self):
        rec = self._make_reconciler()

        import engine.reconciler as reconciler_mod
        with patch.object(
            reconciler_mod.positions_repo, "open_for_paper_poll",
            side_effect=RuntimeError("boom"),
        ), patch("infra.telegram.fire", side_effect=_swallow_fire):
            await rec._reconcile_paper_resolutions()
            await rec._reconcile_paper_resolutions()
            self.assertEqual(rec.reconcile_failures, 2)

        # Next cycle succeeds (no open positions -> early success return).
        with patch.object(
            reconciler_mod.positions_repo, "open_for_paper_poll", return_value=[],
        ), patch("infra.telegram.fire", side_effect=_swallow_fire):
            await rec._reconcile_paper_resolutions()

        self.assertEqual(rec.reconcile_failures, 0)
        self.assertEqual(self._get_state("reconcile_failures"), 0)


# =========================================================================
# Bugfix plan Bug 2: order-reconcile escalation blind spot.
#
# _reconcile_orders' fetch_open_orders()->None path previously logged a
# warning and returned with no counter, no engine_state signal, and no
# telegram alert -- a persistent order-API outage produced warnings
# forever with no escalation. Fixed via a SEPARATE counter
# (_order_reconcile_failures), mirroring the existing positions/paper
# escalation mechanism (>= threshold fires every cycle, not once).
# =========================================================================

class TestOrderReconcileFailureEscalation(_DbTestCase):
    db_name = "test_fader_order_reconcile_escalation.db"

    def _get_state(self, key):
        from infra.db import get_connection
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT value_json FROM engine_state WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                return None
            import json as _json
            return _json.loads(row["value_json"])
        finally:
            conn.close()

    def _make_live_reconciler(self, fetch_open_orders_return):
        from engine.reconciler import Reconciler
        provider = MagicMock()
        provider.is_paper = False
        provider.async_fetch_open_orders = AsyncMock(
            return_value=fetch_open_orders_return
        )
        return Reconciler(provider=provider, risk=MagicMock(), order_manager=None)

    async def test_none_x6_counter_climbs_and_alert_fires_twice(self):
        """fetch_open_orders() -> None six times in a row: counter climbs
        1..6; alert fires at 5 AND at 6 (pins the >= fire-repeatedly
        semantics -- a x5-only test can't distinguish fire-once from
        fire-every-cycle-at-or-above-threshold); engine_state key
        published each cycle."""
        rec = self._make_live_reconciler(None)

        with patch("infra.telegram.fire", side_effect=_swallow_fire) as fire_mock:
            for expected in range(1, 7):
                await rec._reconcile_orders()
                self.assertEqual(rec.order_reconcile_failures, expected)
                self.assertEqual(
                    self._get_state("order_reconcile_failures"), expected
                )

        self.assertEqual(fire_mock.call_count, 2)

    async def test_none_x3_then_success_resets_counter(self):
        """Three consecutive None failures bump the counter; a subsequent
        successful cycle (empty live order list, no DB rows) resets it to
        0 and re-publishes."""
        rec = self._make_live_reconciler(None)

        with patch("infra.telegram.fire", side_effect=_swallow_fire):
            for _ in range(3):
                await rec._reconcile_orders()
            self.assertEqual(rec.order_reconcile_failures, 3)

        rec._provider.async_fetch_open_orders = AsyncMock(return_value=[])
        with patch("infra.telegram.fire", side_effect=_swallow_fire) as fire_mock:
            await rec._reconcile_orders()

        self.assertEqual(rec.order_reconcile_failures, 0)
        self.assertEqual(self._get_state("order_reconcile_failures"), 0)
        fire_mock.assert_not_called()

    async def test_paper_mode_counter_stays_zero(self):
        """Paper mode returns before the fetch -- no order API in paper --
        so the order-reconcile counter must never move."""
        rec = self._make_reconciler()  # is_paper=True via _DbTestCase helper

        await rec._reconcile_orders()

        self.assertEqual(rec.order_reconcile_failures, 0)
        self.assertIsNone(self._get_state("order_reconcile_failures"))


# =========================================================================
# Phase 6, item 8: order-reaper CANCELLED persists (was a silent no-commit)
# =========================================================================

class TestReaperPersistsCancelled(_DbTestCase):
    db_name = "test_fader_reaper_persist.db"

    def _insert_unknown_order(self, order_id, token_id, created_at):
        from infra.db import get_connection
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO orders
                  (order_id, idempotency_key, slug, token_id, side, type,
                   price, size, status, created_at)
                VALUES (?, ?, 'slug-reap', ?, 'BUY', 'LIMIT', 0.85, 10.0,
                        'UNKNOWN', ?)
                """,
                (order_id, f"ik-{order_id}", token_id, created_at),
            )
            conn.commit()
        finally:
            conn.close()

    def _get_order(self, order_id):
        from infra.db import get_connection
        conn = get_connection()
        try:
            return conn.execute(
                "SELECT * FROM orders WHERE order_id=?", (order_id,)
            ).fetchone()
        finally:
            conn.close()

    async def test_stale_unknown_order_actually_persists_as_cancelled(self):
        from datetime import timedelta

        # NOTE: the timestamp-format bug this comment used to describe
        # (ISO 'T' strings vs SQLite datetime() output making sub-day gaps
        # compare wrong) is now FIXED -- reap_stale_unknown computes a
        # Python-side ISO cutoff and TestReaperTtlPinned covers the 1h TTL
        # semantics with pinned timestamps. The 2-day-old timestamp here is
        # retained so this test stays about the commit/persistence fix
        # (Phase 6 item 8), independent of TTL boundary behavior.
        stale_ts = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        self._insert_unknown_order("o-stale-1", "0xTOKSTALE", stale_ts)

        provider = MagicMock()
        provider.is_paper = False
        provider.async_fetch_open_orders = AsyncMock(return_value=[])

        risk = MagicMock()
        from engine.reconciler import Reconciler
        rec = Reconciler(provider=provider, risk=risk, order_manager=None)

        await rec._reconcile_orders()

        # Re-fetch on a FRESH connection -- if the reaper's UPDATE never
        # committed (the pre-Phase-6 bug), this would still read UNKNOWN.
        row = self._get_order("o-stale-1")
        self.assertEqual(row["status"], "CANCELLED")
        self.assertEqual(row["cancel_reason"], "unknown_ttl")

    async def test_fresh_unknown_order_not_reaped(self):
        fresh_ts = datetime.now(timezone.utc).isoformat()
        self._insert_unknown_order("o-fresh-1", "0xTOKFRESH", fresh_ts)

        provider = MagicMock()
        provider.is_paper = False
        provider.async_fetch_open_orders = AsyncMock(return_value=[])

        risk = MagicMock()
        from engine.reconciler import Reconciler
        rec = Reconciler(provider=provider, risk=risk, order_manager=None)

        await rec._reconcile_orders()

        row = self._get_order("o-fresh-1")
        self.assertEqual(row["status"], "UNKNOWN")


# =========================================================================
# Bugfix plan Bug 1: reap_stale_unknown timestamp-format mismatch.
#
# created_at is written as datetime.isoformat() ("...T07:05:11.231837+00:00",
# 'T' separator, microseconds, offset). The pre-fix SQL compared that
# directly against SQLite's datetime(?, '-3600 seconds'), which returns
# "YYYY-MM-DD HH:MM:SS" (space separator, no offset, no microseconds).
# Raw string collation puts 'T' (0x54) > ' ' (0x20), so any created_at on
# the SAME UTC date as the cutoff compares GREATER than the cutoff and is
# never reaped -- the reap only fires once the date prefix itself differs,
# making the effective TTL ~1 day instead of 1 hour.
#
# ALL tests in this class use a PINNED synthetic now_iso, never wall-clock:
# a wall-clock "2 hours old, same day" test would spuriously PASS pre-fix
# whenever it happens to run in the first ~2 hours of a UTC day (date
# prefix already differs), silently hiding the bug depending on run time.
# =========================================================================

class TestReaperTtlPinned(_DbTestCase):
    db_name = "test_fader_reaper_ttl_pinned.db"

    # Same UTC date throughout -- this is the whole point of the bug.
    PINNED_NOW_ISO = "2026-01-15T12:00:00.000000+00:00"

    def _insert_unknown_order(self, order_id, token_id, created_at):
        from infra.db import get_connection
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO orders
                  (order_id, idempotency_key, slug, token_id, side, type,
                   price, size, status, created_at)
                VALUES (?, ?, 'slug-reap-pinned', ?, 'BUY', 'LIMIT', 0.85, 10.0,
                        'UNKNOWN', ?)
                """,
                (order_id, f"ik-{order_id}", token_id, created_at),
            )
            conn.commit()
        finally:
            conn.close()

    def _insert_order_with_status(self, order_id, token_id, status, created_at):
        from infra.db import get_connection
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO orders
                  (order_id, idempotency_key, slug, token_id, side, type,
                   price, size, status, created_at)
                VALUES (?, ?, 'slug-reap-pinned', ?, 'BUY', 'LIMIT', 0.85, 10.0,
                        ?, ?)
                """,
                (order_id, f"ik-{order_id}", token_id, status, created_at),
            )
            conn.commit()
        finally:
            conn.close()

    def _get_order(self, order_id):
        from infra.db import get_connection
        conn = get_connection()
        try:
            return conn.execute(
                "SELECT * FROM orders WHERE order_id=?", (order_id,)
            ).fetchone()
        finally:
            conn.close()

    def test_1_two_hours_before_pinned_now_same_date_is_reaped(self):
        """Test #1 (red-green): UNKNOWN order 2h before a PINNED now_iso, on
        the SAME UTC date, must be reaped. This is the exact case the
        buggy string comparison missed -- demonstrated failing against
        pre-fix code, then passing post-fix (see task report)."""
        from persistence.repos import orders_repo

        created_at = "2026-01-15T10:00:00.000000+00:00"  # 2h before pinned now
        self._insert_unknown_order("o-pinned-2h", "0xTOKP2H", created_at)

        orders_repo.reap_stale_unknown(self.PINNED_NOW_ISO)

        row = self._get_order("o-pinned-2h")
        self.assertEqual(row["status"], "CANCELLED")
        self.assertEqual(row["cancel_reason"], "unknown_ttl")

    def test_2_thirty_minutes_before_pinned_now_not_reaped(self):
        """TTL is 1h, not 0 -- a 30-minute-old UNKNOWN order must survive."""
        from persistence.repos import orders_repo

        created_at = "2026-01-15T11:30:00.000000+00:00"  # 30 min before now
        self._insert_unknown_order("o-pinned-30m", "0xTOKP30M", created_at)

        orders_repo.reap_stale_unknown(self.PINNED_NOW_ISO)

        row = self._get_order("o-pinned-30m")
        self.assertEqual(row["status"], "UNKNOWN")
        self.assertIsNone(row["cancel_reason"])

    def test_3_sixty_one_minutes_before_pinned_now_boundary_is_reaped(self):
        """Boundary case: 61 minutes before pinned now, same UTC date --
        exactly the gap the string collation bug broke (any same-day
        created_at compared GREATER than the cutoff regardless of time)."""
        from persistence.repos import orders_repo

        created_at = "2026-01-15T10:59:00.000000+00:00"  # 61 min before now
        self._insert_unknown_order("o-pinned-61m", "0xTOKP61M", created_at)

        orders_repo.reap_stale_unknown(self.PINNED_NOW_ISO)

        row = self._get_order("o-pinned-61m")
        self.assertEqual(row["status"], "CANCELLED")
        self.assertEqual(row["cancel_reason"], "unknown_ttl")

    def test_4_non_unknown_statuses_untouched(self):
        """PENDING/FILLED/CANCELLED/FAILED rows old enough to qualify by
        timestamp alone must NOT be touched -- only status='UNKNOWN' is
        eligible for the reaper."""
        from persistence.repos import orders_repo

        old_ts = "2026-01-15T09:00:00.000000+00:00"  # 3h before pinned now
        for status in ("PENDING", "FILLED", "CANCELLED", "FAILED"):
            self._insert_order_with_status(
                f"o-pinned-{status}", f"0xTOK{status}", status, old_ts
            )

        orders_repo.reap_stale_unknown(self.PINNED_NOW_ISO)

        for status in ("PENDING", "FILLED", "CANCELLED", "FAILED"):
            row = self._get_order(f"o-pinned-{status}")
            self.assertEqual(row["status"], status)


if __name__ == "__main__":
    unittest.main()
