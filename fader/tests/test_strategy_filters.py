"""tests/test_strategy_filters.py

Two layers of coverage for the live 11-filter stack, post-Phase-4
(fader/strategy/filters.py -- the pure, shared filter core used by both
the live engine and the backtest engine):

1. Integration-level tests driving StrategyLoop._tick end-to-end (the
   original Phase 0 characterization tests, KEPT UNCHANGED -- they prove
   the live wiring: StrategyLoop builds FilterParams/EntrySnapshot
   correctly, calls evaluate_pregate/evaluate_entry in the right order,
   and maps FilterResult back onto log_rejected with the exact reason
   strings and detail dicts the dashboard/decisions table depend on).
2. Table-driven unit tests directly against evaluate_pregate/evaluate_entry
   (new, Phase 4) -- one case per filter, the None-policy matrix, and both
   paper-mode carve-outs, without going through StrategyLoop/DB/asyncio at
   all. These are the fast, precise tests for the pure core itself.

Construction pattern for the integration tests follows
test_vps_review_fixes.py:57 (StrategyLoop built directly with a real
book_store/staleness/risk and a MagicMock-free AppConfig) and
test_vps_review_fixes.py:351,369 (StrategyLoop driven via
IsolatedAsyncioTestCase + asyncio, with a MagicMock OrderManager).

DB setup mirrors test_live_readiness.py's _DbTestCase-style pattern
(set_db_path + init_db against a throwaway per-test-file DB, cleaned up
in tearDown).

Every expected (decision, reason, filters_json-derived field) pair below
was independently reproduced against the current code in a scratch
probe script before being written here -- not guessed.

Run: python -m pytest fader/tests/test_strategy_filters.py -v
"""

from __future__ import annotations

import datetime
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


class _StrategyLoopTestCase(unittest.IsolatedAsyncioTestCase):
    """Shared fixture: real AppConfig, real BookStore/StalenessTracker/
    RiskManager, real (throwaway) SQLite DB -- only the OrderManager and
    the external REST calls (fetch_volumes / compute_dte) are mocked."""

    db_name = "test_fader_strategy_filters.db"

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

    # -- helpers --------------------------------------------------------

    def _make_cfg(self, mode="live"):
        from config.config_loader import AppConfig, SlugRow
        cfg = AppConfig()
        cfg.mode = mode
        cfg.strategy.band_low = 0.80
        cfg.strategy.band_high = 0.95
        cfg.strategy.min_dte = 0
        cfg.strategy.max_dte = 365
        cfg.strategy.min_time_in_band_s = 0
        cfg.strategy.order_notional_usd = 10.0
        cfg.strategy.alpha = 0.0
        cfg.filters.min_24h_volume = 1000.0
        cfg.filters.min_total_volume = 10000.0
        cfg.filters.min_book_depth = 0.0
        cfg.slugs = [SlugRow(slug="test-slug", enabled=True, market_kind="binary")]
        return cfg

    def _make_market_info(self, end_date_iso=None):
        from execution.provider import MarketInfo
        if end_date_iso is None:
            end_date_iso = (
                datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(days=30)
            ).isoformat()
        return MarketInfo(
            slug="test-slug", condition_id="0xCOND", token_id="tok-1",
            outcome="No", outcome_index=1, question="Q?",
            end_date_iso=end_date_iso, active=True, closed=False,
        )

    def _make_loop(self, cfg, risk=None):
        from engine.strategy_loop import StrategyLoop
        from engine.risk import RiskManager
        from marketdata.book_state import BookStore
        from marketdata.staleness import StalenessTracker
        books = BookStore()
        staleness = StalenessTracker()
        risk = risk or RiskManager()
        om = MagicMock()
        om.resting_exposure.return_value = (0.0, {})
        om.enter = AsyncMock()
        sl = StrategyLoop(
            cfg=cfg, book_store=books, staleness=staleness, risk=risk,
            order_manager=om,
        )
        sl.load_markets({"test-slug": self._make_market_info()})
        sl.set_bankroll(1000.0)
        return sl, books, staleness, om

    def _seed_book(self, books, ask="0.85", ask_size="100", bid="0.80",
                    band_entry_offset_s=700.0):
        book = books.get_or_create("tok-1")
        book.apply_snapshot(
            bids_raw=[{"price": bid, "size": "50"}],
            asks_raw=[{"price": ask, "size": ask_size}],
        )
        if band_entry_offset_s is not None:
            book.band_entry_ts = time.monotonic() - band_entry_offset_s
        return book

    def _decisions(self):
        from infra.db import get_connection
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT decision, reason, filters_json FROM decisions ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    async def _tick_with_patches(self, sl, volumes=None, dte_side_effect=None):
        patches = [patch("engine.strategy_loop.positions_repo.deployed_total", return_value=(0.0, {}))]
        if volumes is not None:
            patches.append(patch("engine.strategy_loop.fetch_volumes", return_value=volumes))
        if dte_side_effect is not None:
            patches.append(patch("engine.strategy_loop.compute_dte", side_effect=dte_side_effect))
        for p in patches:
            p.start()
        try:
            await sl._tick()
        finally:
            for p in reversed(patches):
                p.stop()


# =========================================================================
# Filter 1/2: no_book, ask_out_of_band, dte_out_of_range (live, dte=None)
# =========================================================================

class TestPreGateFilters(_StrategyLoopTestCase):
    async def test_missing_book_rejects_no_book(self):
        """book_store has no book at all for the token -> 'no_book'."""
        cfg = self._make_cfg("live")
        sl, books, staleness, om = self._make_loop(cfg)
        # No book created for tok-1 at all.
        await self._tick_with_patches(sl)
        rows = self._decisions()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["decision"], "REJECTED")
        self.assertEqual(rows[0]["reason"], "no_book")
        om.enter.assert_not_called()

    async def test_book_with_no_asks_rejects_no_book(self):
        """book exists but best_ask is None (empty ask side) -> 'no_book'
        (strategy_loop.py:157-158: 'book is None or book.best_ask is None')."""
        cfg = self._make_cfg("live")
        sl, books, staleness, om = self._make_loop(cfg)
        books.get_or_create("tok-1")  # empty book, no snapshot applied
        await self._tick_with_patches(sl)
        rows = self._decisions()
        self.assertEqual(rows[0]["reason"], "no_book")
        om.enter.assert_not_called()

    async def test_ask_out_of_band_rejects(self):
        cfg = self._make_cfg("live")
        sl, books, staleness, om = self._make_loop(cfg)
        self._seed_book(books, ask="0.50", bid="0.49")  # below band_low=0.80
        await self._tick_with_patches(sl)
        rows = self._decisions()
        self.assertEqual(rows[0]["reason"], "ask_out_of_band")
        om.enter.assert_not_called()

    async def test_live_dte_none_rejects_dte_out_of_range(self):
        """Live mode: MarketInfo has no end_date_iso, and the Gamma
        fallback (compute_dte) raises -> dte_val resolves to None ->
        rejected with 'dte_out_of_range' (fail-closed). Pins
        strategy_loop.py:167-173."""
        cfg = self._make_cfg("live")
        sl, books, staleness, om = self._make_loop(cfg)
        sl.load_markets({"test-slug": self._make_market_info(end_date_iso="")})
        self._seed_book(books)
        await self._tick_with_patches(sl, dte_side_effect=Exception("gamma down"))
        rows = self._decisions()
        self.assertEqual(rows[0]["reason"], "dte_out_of_range")
        om.enter.assert_not_called()


# =========================================================================
# Filters 3-11 reject reasons (live mode)
# =========================================================================

class TestFullStackRejectReasons(_StrategyLoopTestCase):
    async def test_not_in_band_long_enough(self):
        cfg = self._make_cfg("live")
        cfg.strategy.min_time_in_band_s = 600
        sl, books, staleness, om = self._make_loop(cfg)
        self._seed_book(books, band_entry_offset_s=0.0)  # just entered band
        staleness.touch("tok-1")
        await self._tick_with_patches(
            sl, volumes={"volume_24h": 2000.0, "volume_total": 20000.0}
        )
        rows = self._decisions()
        self.assertEqual(rows[0]["reason"], "not_in_band_long_enough")
        om.enter.assert_not_called()

    async def test_low_24h_volume(self):
        cfg = self._make_cfg("live")
        sl, books, staleness, om = self._make_loop(cfg)
        self._seed_book(books)
        staleness.touch("tok-1")
        await self._tick_with_patches(
            sl, volumes={"volume_24h": 5.0, "volume_total": 20000.0}
        )
        rows = self._decisions()
        self.assertEqual(rows[0]["reason"], "low_24h_volume")
        om.enter.assert_not_called()

    async def test_low_total_volume(self):
        cfg = self._make_cfg("live")
        sl, books, staleness, om = self._make_loop(cfg)
        self._seed_book(books)
        staleness.touch("tok-1")
        await self._tick_with_patches(
            sl, volumes={"volume_24h": 2000.0, "volume_total": 5.0}
        )
        rows = self._decisions()
        self.assertEqual(rows[0]["reason"], "low_total_volume")
        om.enter.assert_not_called()

    async def test_insufficient_depth(self):
        cfg = self._make_cfg("live")
        cfg.filters.min_book_depth = 1000.0
        sl, books, staleness, om = self._make_loop(cfg)
        self._seed_book(books, ask_size="1")  # tiny depth: 0.85 * 1 = 0.85 USD
        staleness.touch("tok-1")
        await self._tick_with_patches(
            sl, volumes={"volume_24h": 2000.0, "volume_total": 20000.0}
        )
        rows = self._decisions()
        self.assertEqual(rows[0]["reason"], "insufficient_depth")
        om.enter.assert_not_called()

    async def test_stale_data_live_mode(self):
        """Live mode: token never 'touch'-ed by the staleness tracker ->
        is_stale() is True -> 'stale_data'. This is the live counterpart
        of the paper carve-out pinned below."""
        cfg = self._make_cfg("live")
        sl, books, staleness, om = self._make_loop(cfg)
        self._seed_book(books)
        # staleness.touch("tok-1") intentionally NOT called
        await self._tick_with_patches(
            sl, volumes={"volume_24h": 2000.0, "volume_total": 20000.0}
        )
        rows = self._decisions()
        self.assertEqual(rows[0]["reason"], "stale_data")
        om.enter.assert_not_called()

    async def test_position_already_open(self):
        cfg = self._make_cfg("live")
        sl, books, staleness, om = self._make_loop(cfg)
        self._seed_book(books)
        staleness.touch("tok-1")
        from infra.db import get_connection
        conn = get_connection()
        conn.execute(
            "INSERT INTO positions (position_id, slug, token_id, outcome, "
            "entry_price, size, notional, status, opened_at, source) VALUES "
            "('p1','test-slug','tok-1','No',0.85,10,10,'OPEN',"
            "'2024-01-01T00:00:00Z','ENGINE_FILL')"
        )
        conn.commit()
        conn.close()
        await self._tick_with_patches(
            sl, volumes={"volume_24h": 2000.0, "volume_total": 20000.0}
        )
        rows = self._decisions()
        self.assertEqual(rows[0]["reason"], "position_already_open")
        om.enter.assert_not_called()

    async def test_risk_cap(self):
        """max_deployed_pct=1% of a $100 bankroll = $1 cap; the $10 order
        breaches it -> RiskManager.allow_entry returns (False, ...) ->
        logged as 'risk_cap'."""
        from engine.risk import RiskManager
        cfg = self._make_cfg("live")
        risk = RiskManager(max_deployed_pct=1.0)
        sl, books, staleness, om = self._make_loop(cfg, risk=risk)
        sl.set_bankroll(100.0)
        self._seed_book(books)
        staleness.touch("tok-1")
        await self._tick_with_patches(
            sl, volumes={"volume_24h": 2000.0, "volume_total": 20000.0}
        )
        rows = self._decisions()
        self.assertEqual(rows[0]["reason"], "risk_cap")
        om.enter.assert_not_called()

    async def test_risk_cap_logs_bankroll_age_when_stale(self):
        """Phase 3: when a risk_cap rejection fires and the wired
        bankroll_view source reports a value older than 2x the bankroll
        poll interval, bankroll_age_s is added to the filters dict."""
        import time as time_mod
        from engine.risk import RiskManager
        from engine.reconciler import BankrollView
        cfg = self._make_cfg("live")
        risk = RiskManager(max_deployed_pct=1.0)
        sl, books, staleness, om = self._make_loop(cfg, risk=risk)
        sl.set_bankroll(100.0)
        stale_as_of = time_mod.monotonic() - 100.0  # well past 2x30s
        sl.set_bankroll_view_source(
            lambda: BankrollView(value=100.0, as_of=stale_as_of), poll_interval_s=30.0,
        )
        self._seed_book(books)
        staleness.touch("tok-1")
        await self._tick_with_patches(
            sl, volumes={"volume_24h": 2000.0, "volume_total": 20000.0}
        )
        rows = self._decisions()
        self.assertEqual(rows[0]["reason"], "risk_cap")
        self.assertIn("bankroll_age_s", rows[0]["filters_json"])
        om.enter.assert_not_called()

    async def test_risk_cap_omits_bankroll_age_when_fresh(self):
        """No bankroll_age_s key when the reconcile is recent (< 2x poll
        interval) -- keeps the filters dict unchanged in the common case."""
        import time as time_mod
        from engine.risk import RiskManager
        from engine.reconciler import BankrollView
        cfg = self._make_cfg("live")
        risk = RiskManager(max_deployed_pct=1.0)
        sl, books, staleness, om = self._make_loop(cfg, risk=risk)
        sl.set_bankroll(100.0)
        fresh_as_of = time_mod.monotonic() - 1.0
        sl.set_bankroll_view_source(
            lambda: BankrollView(value=100.0, as_of=fresh_as_of), poll_interval_s=30.0,
        )
        self._seed_book(books)
        staleness.touch("tok-1")
        await self._tick_with_patches(
            sl, volumes={"volume_24h": 2000.0, "volume_total": 20000.0}
        )
        rows = self._decisions()
        self.assertEqual(rows[0]["reason"], "risk_cap")
        self.assertNotIn("bankroll_age_s", rows[0]["filters_json"])

    async def test_risk_cap_when_breaker_already_tripped(self):
        """allow_entry() checks breaker_tripped FIRST; when the breaker is
        already tripped going into the tick, the rejection surfaces as
        'risk_cap' with risk_reason='circuit_breaker_tripped' -- NOT the
        standalone 'circuit_breaker' reason (that one only fires when the
        breaker trips freshly inside the SAME tick, after allow_entry has
        already passed; see test_circuit_breaker_trips_within_tick)."""
        from engine.risk import RiskManager
        cfg = self._make_cfg("live")
        risk = RiskManager(daily_loss_pct=5.0)
        sl, books, staleness, om = self._make_loop(cfg, risk=risk)
        sl.set_bankroll(100.0)
        self._seed_book(books)
        staleness.touch("tok-1")
        # Trip the breaker before the tick runs.
        from infra.db import get_connection
        conn = get_connection()
        conn.execute(
            "INSERT INTO circuit_breaker (day, realized_pnl) VALUES (?, ?)",
            (risk.today_utc(), -50.0),
        )
        conn.commit()
        conn.close()
        risk.check_breaker_against_bankroll(100.0)  # trips + persists
        self.assertTrue(risk.breaker_tripped)

        await self._tick_with_patches(
            sl, volumes={"volume_24h": 2000.0, "volume_total": 20000.0}
        )
        rows = self._decisions()
        self.assertEqual(rows[0]["reason"], "risk_cap")
        self.assertIn("circuit_breaker_tripped", rows[0]["filters_json"])
        om.enter.assert_not_called()

    async def test_circuit_breaker_trips_within_tick(self):
        """Breaker not yet tripped when allow_entry() runs (so allow_entry
        passes), but check_breaker_against_bankroll() -- called right
        after -- newly trips it from the day's realized_pnl row. This is
        the ONLY path that logs the standalone 'circuit_breaker' reason
        (strategy_loop.py:309-312)."""
        from engine.risk import RiskManager
        from infra.db import get_connection
        cfg = self._make_cfg("live")
        risk = RiskManager(daily_loss_pct=5.0, max_deployed_pct=100.0,
                            per_market_cap_pct=100.0)
        sl, books, staleness, om = self._make_loop(cfg, risk=risk)
        sl.set_bankroll(100.0)
        self._seed_book(books)
        staleness.touch("tok-1")

        conn = get_connection()
        conn.execute(
            "INSERT INTO circuit_breaker (day, realized_pnl) VALUES (?, ?)",
            (risk.today_utc(), -10.0),
        )
        conn.commit()
        conn.close()
        self.assertFalse(risk.breaker_tripped, "breaker must not be pre-tripped for this test")

        await self._tick_with_patches(
            sl, volumes={"volume_24h": 2000.0, "volume_total": 20000.0}
        )
        rows = self._decisions()
        self.assertEqual(rows[0]["reason"], "circuit_breaker")
        self.assertEqual(rows[0]["filters_json"], "{}")
        om.enter.assert_not_called()


# =========================================================================
# Paper-mode carve-outs (two SEPARATE bypasses, strategy_loop.py:249,283)
# =========================================================================

class TestPaperModeCarveOuts(_StrategyLoopTestCase):
    async def test_paper_mode_skips_staleness_check(self):
        """Paper mode: staleness.touch() is never called for tok-1 (would
        be 'stale_data' in live mode, per
        test_stale_data_live_mode above) -- but the paper carve-out at
        strategy_loop.py:283 ('self._cfg.mode != "paper" and ...') means
        the stale check is skipped entirely and the entry proceeds."""
        cfg = self._make_cfg("paper")
        sl, books, staleness, om = self._make_loop(cfg)
        self._seed_book(books)
        # staleness.touch("tok-1") intentionally NOT called.
        await self._tick_with_patches(
            sl, volumes={"volume_24h": 2000.0, "volume_total": 20000.0}
        )
        rows = self._decisions()
        self.assertEqual(len(rows), 0, "no rejection logged -- entry proceeded")
        om.enter.assert_called_once()

    async def test_paper_mode_skips_min_time_in_band_check(self):
        """Paper mode: band_entry_ts is set to 'just now' (0s in band),
        which would be 'not_in_band_long_enough' in live mode with
        min_time_in_band_s=600 (per test_not_in_band_long_enough above) --
        but the paper carve-out at strategy_loop.py:249
        ('cfg.mode != "paper" and not book.is_in_band_long_enough(...)')
        means the check is skipped and the entry proceeds."""
        cfg = self._make_cfg("paper")
        cfg.strategy.min_time_in_band_s = 600
        sl, books, staleness, om = self._make_loop(cfg)
        self._seed_book(books, band_entry_offset_s=0.0)  # just entered band
        staleness.touch("tok-1")
        await self._tick_with_patches(
            sl, volumes={"volume_24h": 2000.0, "volume_total": 20000.0}
        )
        rows = self._decisions()
        self.assertEqual(len(rows), 0, "no rejection logged -- entry proceeded")
        om.enter.assert_called_once()

    async def test_paper_mode_both_carveouts_together_still_enters(self):
        """Both carve-outs active simultaneously: staleness never touched
        AND band_entry_ts just set -- paper mode still enters because
        BOTH gates are bypassed independently."""
        cfg = self._make_cfg("paper")
        cfg.strategy.min_time_in_band_s = 600
        sl, books, staleness, om = self._make_loop(cfg)
        self._seed_book(books, band_entry_offset_s=0.0)
        # staleness.touch("tok-1") intentionally NOT called.
        await self._tick_with_patches(
            sl, volumes={"volume_24h": 2000.0, "volume_total": 20000.0}
        )
        rows = self._decisions()
        self.assertEqual(len(rows), 0)
        om.enter.assert_called_once()


# =========================================================================
# Volume-fetch failure -> fail-closed rejection
# =========================================================================

class TestVolumeFetchFailureFailsClosed(_StrategyLoopTestCase):
    async def test_fetch_volumes_exception_yields_zeros_and_rejection(self):
        """_get_volumes() catches any exception from fetch_volumes() and
        substitutes {'volume_24h': 0.0, 'volume_total': 0.0}
        (strategy_loop.py:369-381) -- zeros always fail the
        min_24h_volume gate (default 1000.0), so an API outage fails
        CLOSED (rejects) rather than silently skipping the filter."""
        cfg = self._make_cfg("live")
        sl, books, staleness, om = self._make_loop(cfg)
        self._seed_book(books)
        staleness.touch("tok-1")

        with patch("engine.strategy_loop.positions_repo.deployed_total", return_value=(0.0, {})), \
             patch("engine.strategy_loop.fetch_volumes", side_effect=Exception("api down")):
            await sl._tick()

        rows = self._decisions()
        self.assertEqual(rows[0]["reason"], "low_24h_volume")
        self.assertIn('"volume_24h": 0.0', rows[0]["filters_json"])
        self.assertIn('"volume_total": 0.0', rows[0]["filters_json"])
        om.enter.assert_not_called()


# =========================================================================
# Phase 4: table-driven tests against the pure core directly
# (strategy/filters.py -- no StrategyLoop, no DB, no asyncio)
# =========================================================================

from strategy.filters import EntrySnapshot, FilterParams, evaluate_entry, evaluate_pregate  # noqa: E402


def _params(**overrides) -> FilterParams:
    base = dict(
        band_low=0.80, band_high=0.95,
        min_dte=0, max_dte=365,
        min_time_in_band_s=600,
        min_24h_volume=1000.0, min_total_volume=10000.0, min_book_depth=0.0,
        check_staleness=True, check_time_in_band=True,
        missing_dte="reject",
    )
    base.update(overrides)
    return FilterParams(**base)


def _snapshot(**overrides) -> EntrySnapshot:
    base = dict(
        best_ask=0.85, dte=30.0, seconds_in_band=700.0,
        volume_24h=2000.0, volume_total=20000.0, ask_depth_usd=500.0,
        is_stale=False, has_open_position=False,
    )
    base.update(overrides)
    return EntrySnapshot(**base)


class TestEvaluatePregateTable(unittest.TestCase):
    """evaluate_pregate: filters 1-2 only (band, DTE) -- one case per
    reject reason plus the None-policy matrix for best_ask/dte."""

    def test_best_ask_none_rejects_no_book(self):
        r = evaluate_pregate(None, 30.0, _params())
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "no_book")
        self.assertEqual(r.detail, {})

    def test_ask_below_band_rejects_ask_out_of_band(self):
        r = evaluate_pregate(0.50, 30.0, _params())
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "ask_out_of_band")
        self.assertEqual(r.detail, {"no_ask": 0.50, "band_low": 0.80, "band_high": 0.95})

    def test_ask_above_band_rejects_ask_out_of_band(self):
        r = evaluate_pregate(0.99, 30.0, _params())
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "ask_out_of_band")

    def test_ask_at_band_edges_passes(self):
        self.assertTrue(evaluate_pregate(0.80, 30.0, _params()).passed)
        self.assertTrue(evaluate_pregate(0.95, 30.0, _params()).passed)

    def test_dte_out_of_range_rejects(self):
        r = evaluate_pregate(0.85, 500.0, _params())
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "dte_out_of_range")
        self.assertEqual(r.detail, {"dte": 500.0, "min_dte": 0, "max_dte": 365})

    def test_dte_none_rejects_under_reject_policy_live(self):
        """missing_dte='reject' (live): dte=None fails closed."""
        r = evaluate_pregate(0.85, None, _params(missing_dte="reject"))
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "dte_out_of_range")
        self.assertEqual(r.detail, {"dte": None, "min_dte": 0, "max_dte": 365})

    def test_dte_none_skips_under_skip_policy_backtest(self):
        """missing_dte='skip' (backtest): dte=None passes through
        (fail-open) and is recorded in `skipped`, matching the pinned
        backtest divergence (test_backtest_engine.py
        TestDteNoneRowsFailOpenPin)."""
        r = evaluate_pregate(0.85, None, _params(missing_dte="skip"))
        self.assertTrue(r.passed)
        self.assertEqual(r.skipped, ("dte",))

    def test_all_pass_no_skips_when_dte_present(self):
        r = evaluate_pregate(0.85, 30.0, _params(missing_dte="skip"))
        self.assertTrue(r.passed)
        self.assertEqual(r.skipped, ())

    def test_generic_none_pass_rule_is_forbidden(self):
        """A blanket 'any None input passes' rule would silently flip
        live's fail-closed DTE handling to fail-open -- explicitly
        forbidden by the plan. Prove best_ask=None still rejects even
        under the backtest 'skip' policy (skip only applies to dte)."""
        r = evaluate_pregate(None, None, _params(missing_dte="skip"))
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "no_book")


class TestEvaluateEntryTable(unittest.TestCase):
    """evaluate_entry: full filters 1-8 stack -- one case per filter,
    both paper carve-outs, and the None-skip policy for volumes/depth/
    staleness."""

    def test_all_filters_pass(self):
        r = evaluate_entry(_snapshot(), _params())
        self.assertTrue(r.passed)
        self.assertEqual(r.reason, "all_filters_passed")
        self.assertEqual(r.skipped, ())

    def test_pregate_reject_propagates_no_book(self):
        r = evaluate_entry(_snapshot(best_ask=None), _params())
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "no_book")

    def test_pregate_reject_propagates_ask_out_of_band(self):
        r = evaluate_entry(_snapshot(best_ask=0.50), _params())
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "ask_out_of_band")

    def test_pregate_reject_propagates_dte_out_of_range(self):
        r = evaluate_entry(_snapshot(dte=999.0), _params())
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "dte_out_of_range")

    def test_not_in_band_long_enough_rejects(self):
        r = evaluate_entry(_snapshot(seconds_in_band=10.0), _params())
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "not_in_band_long_enough")
        self.assertEqual(r.detail, {"time_in_band_s": 10.0, "min_time_in_band_s": 600})

    def test_seconds_in_band_none_rejects_not_in_band_long_enough(self):
        r = evaluate_entry(_snapshot(seconds_in_band=None), _params())
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "not_in_band_long_enough")

    def test_check_time_in_band_false_paper_carveout_skips_gate(self):
        """Paper carve-out #1 (strategy_loop.py check_staleness=not
        is_paper): check_time_in_band=False means even 0 seconds in band
        passes -- matches test_paper_mode_skips_min_time_in_band_check
        above (integration-level proof of the same behavior)."""
        r = evaluate_entry(
            _snapshot(seconds_in_band=0.0),
            _params(check_time_in_band=False),
        )
        self.assertTrue(r.passed)

    def test_low_24h_volume_rejects(self):
        r = evaluate_entry(_snapshot(volume_24h=5.0), _params())
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "low_24h_volume")
        self.assertEqual(
            r.detail,
            {"volume_24h": 5.0, "volume_total": 20000.0, "min_24h_volume": 1000.0},
        )

    def test_volume_24h_none_is_skipped_not_rejected(self):
        """volume_24h=None -> filter SKIPPED (recorded in `skipped`), never
        a rejection -- this is the backtest snapshot shape (volumes not
        reconstructable from historical Polymarket data)."""
        r = evaluate_entry(_snapshot(volume_24h=None), _params())
        self.assertTrue(r.passed)
        self.assertIn("min_24h_volume", r.skipped)

    def test_low_total_volume_rejects(self):
        r = evaluate_entry(_snapshot(volume_total=5.0), _params())
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "low_total_volume")
        self.assertEqual(
            r.detail,
            {"volume_24h": 2000.0, "volume_total": 5.0, "min_total_volume": 10000.0},
        )

    def test_volume_total_none_is_skipped_not_rejected(self):
        r = evaluate_entry(_snapshot(volume_total=None), _params())
        self.assertTrue(r.passed)
        self.assertIn("min_total_volume", r.skipped)

    def test_insufficient_depth_rejects(self):
        r = evaluate_entry(
            _snapshot(ask_depth_usd=0.85),
            _params(min_book_depth=1000.0),
        )
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "insufficient_depth")
        self.assertEqual(r.detail, {"ask_depth_usd": 0.85, "min_book_depth": 1000.0})

    def test_zero_min_book_depth_disables_the_gate(self):
        """min_book_depth=0 (the config default) means the depth gate
        never fires, matching strategy_loop.py's
        `cfg.filters.min_book_depth > 0 and ...` guard."""
        r = evaluate_entry(
            _snapshot(ask_depth_usd=0.0001),
            _params(min_book_depth=0.0),
        )
        self.assertTrue(r.passed)

    def test_ask_depth_usd_none_is_skipped_not_rejected(self):
        r = evaluate_entry(
            _snapshot(ask_depth_usd=None),
            _params(min_book_depth=1000.0),
        )
        self.assertTrue(r.passed)
        self.assertIn("min_book_depth", r.skipped)

    def test_stale_data_rejects(self):
        r = evaluate_entry(_snapshot(is_stale=True), _params())
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "stale_data")
        self.assertEqual(r.detail, {"stale": True})

    def test_is_stale_none_is_skipped_not_rejected(self):
        r = evaluate_entry(_snapshot(is_stale=None), _params())
        self.assertTrue(r.passed)
        self.assertIn("stale_data", r.skipped)

    def test_check_staleness_false_paper_carveout_skips_gate(self):
        """Paper carve-out #2: check_staleness=False means even
        is_stale=True never rejects -- matches
        test_paper_mode_skips_staleness_check above."""
        r = evaluate_entry(
            _snapshot(is_stale=True),
            _params(check_staleness=False),
        )
        self.assertTrue(r.passed)

    def test_both_paper_carveouts_together_still_passes(self):
        r = evaluate_entry(
            _snapshot(seconds_in_band=0.0, is_stale=True),
            _params(check_time_in_band=False, check_staleness=False),
        )
        self.assertTrue(r.passed)

    def test_position_already_open_rejects(self):
        r = evaluate_entry(_snapshot(has_open_position=True), _params())
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "position_already_open")
        self.assertEqual(r.detail, {})

    def test_skipped_union_accumulates_across_multiple_none_fields(self):
        """All three backtest-shape None fields (volumes + depth) skip
        independently and all show up in the union -- this is exactly the
        set backtest/engine.py surfaces as BACKTEST_ALWAYS_SKIPPED and
        metrics.py's universe_discrepancies reads."""
        r = evaluate_entry(
            _snapshot(volume_24h=None, volume_total=None, ask_depth_usd=None),
            _params(),
        )
        self.assertTrue(r.passed)
        self.assertEqual(
            set(r.skipped),
            {"min_24h_volume", "min_total_volume", "min_book_depth"},
        )

    def test_filter_order_ask_checked_before_dte(self):
        """Both ask and dte out of range simultaneously -> ask_out_of_band
        wins (filter 1 before filter 2), matching current code order."""
        r = evaluate_entry(_snapshot(best_ask=0.50, dte=999.0), _params())
        self.assertEqual(r.reason, "ask_out_of_band")

    def test_filter_order_volume_before_depth_before_staleness(self):
        """When multiple later filters would fail, the first one in stack
        order (24h volume) wins."""
        r = evaluate_entry(
            _snapshot(volume_24h=5.0, ask_depth_usd=0.0, is_stale=True),
            _params(min_book_depth=1000.0),
        )
        self.assertEqual(r.reason, "low_24h_volume")


if __name__ == "__main__":
    unittest.main()
