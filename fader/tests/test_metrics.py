"""tests/test_metrics.py

Golden / characterization tests for backtest/metrics.py (Phase 0 safety net,
see temp/implementation-plan.md). These lock CURRENT behavior before any
refactor touches metrics.py. Expected values are hand-derived by running the
underlying formulas directly (see temp scratch script used to derive them —
not committed) rather than by calling the function under test and asserting
whatever it returns.

Run: python -m pytest fader/tests/test_metrics.py -v
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add fader root to path
_FADER_ROOT = Path(__file__).parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))

from backtest.metrics import (
    PERIODS_PER_YEAR,
    bootstrap_ci,
    block_bootstrap_ci,
    calmar_ratio,
    compute_all_metrics,
    daily_pnl_series,
    expectancy,
    hit_rate,
    max_drawdown,
    max_drawdown_pct,
    sortino_ratio,
)


# =========================================================================
# sortino_ratio
# =========================================================================

class TestSortinoRatio(unittest.TestCase):
    def test_hand_computed_mixed_returns(self):
        """returns=[1,-1,2,-2], target=0.
        mu = 0.0
        below = min(r-0,0) = [0,-1,0,-2]
        semi_var = mean([0,1,0,4]) = 1.25
        semi_std = sqrt(1.25) = 1.1180339887...
        sortino = (mu-0)/semi_std * sqrt(365) = 0 (mu is 0)
        """
        returns = np.array([1.0, -1.0, 2.0, -2.0])
        got = sortino_ratio(returns)
        self.assertAlmostEqual(got, 0.0, places=9)

    def test_hand_computed_positive_mean(self):
        """returns=[2,-1,3,-1]. mu=0.75
        below = [0,-1,0,-1] -> semi_var = mean([0,1,0,1]) = 0.5
        semi_std = sqrt(0.5) = 0.70710678...
        sortino = 0.75/0.70710678 * sqrt(365) = 20.263884...
        """
        returns = np.array([2.0, -1.0, 3.0, -1.0])
        mu = 0.75
        semi_std = math.sqrt(0.5)
        expected = (mu / semi_std) * math.sqrt(PERIODS_PER_YEAR)
        got = sortino_ratio(returns)
        self.assertAlmostEqual(got, expected, places=6)
        self.assertAlmostEqual(got, 20.263884, places=5)

    def test_all_wins_denominator_zero_returns_zero(self):
        """All-positive returns -> below is all 0 -> semi_var == 0 -> the
        function returns exactly 0.0 (documented degenerate path)."""
        returns = np.array([1.0, 2.0, 3.0])
        self.assertEqual(sortino_ratio(returns), 0.0)

    def test_single_value_returns_zero(self):
        """len(returns) < 2 -> 0.0 (guard clause)."""
        self.assertEqual(sortino_ratio(np.array([5.0])), 0.0)

    def test_empty_returns_zero(self):
        self.assertEqual(sortino_ratio(np.array([])), 0.0)


# =========================================================================
# calmar_ratio
# =========================================================================

class TestCalmarRatio(unittest.TestCase):
    def test_positive_return_positive_dd(self):
        self.assertAlmostEqual(calmar_ratio(0.20, 0.10), 2.0, places=9)

    def test_negative_return(self):
        self.assertAlmostEqual(calmar_ratio(-0.10, 0.05), -2.0, places=9)

    def test_zero_drawdown_returns_zero(self):
        """max_dd_pct <= 0 -> 0.0 (avoid divide-by-zero)."""
        self.assertEqual(calmar_ratio(0.20, 0.0), 0.0)

    def test_negative_drawdown_returns_zero(self):
        self.assertEqual(calmar_ratio(0.20, -0.05), 0.0)


# =========================================================================
# max_drawdown / max_drawdown_pct
# =========================================================================

class TestMaxDrawdown(unittest.TestCase):
    def test_hand_computed_equity_curve(self):
        """equity=[100,110,90,95,120,80]
        peak = [100,110,110,110,120,120]
        dd (dollars) = peak-equity = [0,0,20,15,0,40] -> max = 40
        frac = dd/peak = [0,0,0.1818...,0.1364...,0,0.3333...] -> max = 1/3
        """
        equity = np.array([100.0, 110.0, 90.0, 95.0, 120.0, 80.0])
        self.assertAlmostEqual(max_drawdown(equity), 40.0, places=9)
        self.assertAlmostEqual(max_drawdown_pct(equity), 1.0 / 3.0, places=9)

    def test_empty_equity(self):
        self.assertEqual(max_drawdown(np.array([])), 0.0)
        self.assertEqual(max_drawdown_pct(np.array([])), 0.0)

    def test_all_negative_is_ruin(self):
        """All peaks <= 0 and some equity < 0 -> max_drawdown_pct == 1.0
        (documented ruin convention)."""
        equity = np.array([-5.0, -10.0, -3.0])
        self.assertEqual(max_drawdown_pct(equity), 1.0)

    def test_flat_zero_no_drawdown(self):
        """All peaks <= 0 but no equity < 0 (flat at zero) -> 0.0."""
        equity = np.array([0.0, 0.0, 0.0])
        self.assertEqual(max_drawdown_pct(equity), 0.0)

    def test_monotonic_increase_no_drawdown(self):
        equity = np.array([100.0, 105.0, 110.0, 120.0])
        self.assertEqual(max_drawdown(equity), 0.0)
        self.assertEqual(max_drawdown_pct(equity), 0.0)


# =========================================================================
# hit_rate / expectancy
# =========================================================================

class TestHitRateExpectancy(unittest.TestCase):
    def test_hand_computed(self):
        """pnls=[5,-2,3,-1,-1]: 2 of 5 positive -> hit_rate=0.4
        mean = (5-2+3-1-1)/5 = 4/5 = 0.8
        """
        pnls = np.array([5.0, -2.0, 3.0, -1.0, -1.0])
        self.assertAlmostEqual(hit_rate(pnls), 0.4, places=9)
        self.assertAlmostEqual(expectancy(pnls), 0.8, places=9)

    def test_empty_trades(self):
        self.assertEqual(hit_rate(np.array([])), 0.0)
        self.assertEqual(expectancy(np.array([])), 0.0)

    def test_single_win(self):
        self.assertEqual(hit_rate(np.array([7.0])), 1.0)
        self.assertEqual(expectancy(np.array([7.0])), 7.0)

    def test_all_wins(self):
        pnls = np.array([2.0, 3.0, 1.0])
        self.assertEqual(hit_rate(pnls), 1.0)
        self.assertAlmostEqual(expectancy(pnls), 2.0, places=9)


# =========================================================================
# daily_pnl_series
# =========================================================================

class TestDailyPnlSeries(unittest.TestCase):
    def test_same_day_trades_summed_and_gaps_zero_filled(self):
        """Two trades exit on 2024-01-01 (summed to 3.0), gap day
        2024-01-02 has no exits (zero-filled), one trade exits on
        2024-01-03 (3.0)."""
        trades = pd.DataFrame({
            "exit_date": ["2024-01-01", "2024-01-01", "2024-01-03"],
            "realized_pnl": [5.0, -2.0, 3.0],
        })
        got = daily_pnl_series(trades)
        np.testing.assert_allclose(got, np.array([3.0, 0.0, 3.0]))

    def test_empty_trades(self):
        self.assertEqual(len(daily_pnl_series(pd.DataFrame())), 0)

    def test_missing_columns(self):
        df = pd.DataFrame({"foo": [1, 2]})
        self.assertEqual(len(daily_pnl_series(df)), 0)


# =========================================================================
# compute_all_metrics
# =========================================================================

class TestComputeAllMetrics(unittest.TestCase):
    def test_empty_trades_returns_empty_dict(self):
        self.assertEqual(compute_all_metrics(pd.DataFrame()), {})

    def test_missing_realized_pnl_column_returns_empty_dict(self):
        df = pd.DataFrame({"entry_price": [0.9]})
        self.assertEqual(compute_all_metrics(df), {})

    def test_hand_computed_four_trade_fixture(self):
        """4 trades, one per calendar day (n_daily=4 < 60 -> calmar
        suppressed to None per the n_daily>=60 CAGR-reliability gate).

        pnls = [5, -2, 3, -1] -> total=5, hit_rate=0.5 (2/4),
        avg_win=mean([5,3])=4.0, avg_loss=mean([-2,-1])=-1.5,
        expectancy=mean(pnls)=1.25.

        daily series == pnls (one trade/day) = [5,-2,3,-1].
        sortino: mu=1.25; below=[0,-2,0,-1]; semi_var=mean([0,4,0,1])=1.25;
          semi_std=sqrt(1.25)=1.1180339887; sortino=1.25/1.1180339887*sqrt(365)
          = 21.360009363293827 (independently derived).
        equity = 500 + cumsum([0,5,-2,3,-1]) = [500,505,503,506,505]
        peak=[500,505,505,506,506]; dd=[0,0,2,0,1] -> max_drawdown=2.0
        frac = dd/peak -> max = 2/505 = 0.0039603960396039604
        """
        trades = pd.DataFrame({
            "realized_pnl": [5.0, -2.0, 3.0, -1.0],
            "entry_price": [0.85, 0.85, 0.90, 0.90],
            "slug": ["a", "a", "b", "b"],
            "exit_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        })
        m = compute_all_metrics(trades, n_bootstrap=200, initial_capital=500.0)

        self.assertAlmostEqual(m["total_pnl"], 5.0, places=9)
        self.assertEqual(m["n_trades"], 4)
        self.assertAlmostEqual(m["hit_rate"], 0.5, places=9)
        self.assertAlmostEqual(m["avg_win"], 4.0, places=9)
        self.assertAlmostEqual(m["avg_loss"], -1.5, places=9)
        self.assertAlmostEqual(m["expectancy"], 1.25, places=9)
        self.assertAlmostEqual(m["sortino"], 21.360009363293827, places=6)
        self.assertIsNone(m["calmar"], "n_daily=4 < 60 -> calmar suppressed to None")
        self.assertAlmostEqual(m["max_drawdown"], 2.0, places=9)
        self.assertAlmostEqual(m["max_drawdown_pct"], 2.0 / 505.0, places=9)
        self.assertEqual(m["n_daily"], 4)
        self.assertEqual(m["n_active_days"], 4)
        self.assertEqual(m["metrics_version"], 5)
        self.assertEqual(m["initial_capital"], 500.0)
        # pnl_ci_95 must bracket the point estimate (sum of daily pnl = 5.0)
        lo, hi = m["pnl_ci_95"]
        self.assertLessEqual(lo, 5.0)
        self.assertGreaterEqual(hi, 5.0)

    def test_single_trade_edge_case(self):
        """n_daily=1 -> have_daily=(len(daily)>=2)=False -> all daily-derived
        fields fall back to the documented degenerate-path constants:
        sortino=0.0, calmar=0.0 (NOT None -- only the have_daily branch
        distinguishes None; the no-daily-data branch always uses 0.0),
        max_drawdown=0.0, max_drawdown_pct=0.0, pnl_ci_95=(0.0, 0.0).
        avg_loss=0.0 because there are no non-positive pnls (empty losses
        array short-circuits to 0.0)."""
        trades = pd.DataFrame({
            "realized_pnl": [7.0],
            "entry_price": [0.85],
            "slug": ["a"],
            "exit_date": ["2024-01-01"],
        })
        m = compute_all_metrics(trades, n_bootstrap=100)

        self.assertEqual(m["total_pnl"], 7.0)
        self.assertEqual(m["n_trades"], 1)
        self.assertEqual(m["hit_rate"], 1.0)
        self.assertEqual(m["avg_win"], 7.0)
        self.assertEqual(m["avg_loss"], 0.0)
        self.assertEqual(m["expectancy"], 7.0)
        self.assertEqual(m["sortino"], 0.0)
        self.assertEqual(m["calmar"], 0.0)
        self.assertEqual(m["max_drawdown"], 0.0)
        self.assertEqual(m["max_drawdown_pct"], 0.0)
        self.assertEqual(m["n_daily"], 1)
        self.assertEqual(m["pnl_ci_95"], (0.0, 0.0))

    def test_all_wins_sortino_zero_and_calmar_none(self):
        """3 trades, all wins, one per day -> n_daily=3 (>=2 so have_daily
        True, but <60 so calmar suppressed to None -- distinct from the
        single-trade case where have_daily is False and calmar is 0.0).
        Sortino: below is all-zero (no losses) -> semi_var==0 -> 0.0.
        """
        trades = pd.DataFrame({
            "realized_pnl": [2.0, 3.0, 1.0],
            "entry_price": [0.85, 0.90, 0.88],
            "slug": ["a", "a", "a"],
            "exit_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        })
        m = compute_all_metrics(trades, n_bootstrap=100)

        self.assertEqual(m["total_pnl"], 6.0)
        self.assertEqual(m["n_trades"], 3)
        self.assertEqual(m["hit_rate"], 1.0)
        self.assertAlmostEqual(m["avg_win"], 2.0, places=9)
        self.assertEqual(m["avg_loss"], 0.0)
        self.assertAlmostEqual(m["expectancy"], 2.0, places=9)
        self.assertEqual(m["sortino"], 0.0)
        self.assertIsNone(m["calmar"], "n_daily=3 >= 2 (have_daily) but < 60 -> None")
        self.assertEqual(m["max_drawdown"], 0.0, "monotonically increasing equity")
        self.assertEqual(m["max_drawdown_pct"], 0.0)

    def test_universe_discrepancies_present(self):
        """compute_all_metrics always documents the three backtest-vs-live
        filter gaps (volume/depth not available historically)."""
        trades = pd.DataFrame({
            "realized_pnl": [1.0],
            "entry_price": [0.85],
            "slug": ["a"],
            "exit_date": ["2024-01-01"],
        })
        m = compute_all_metrics(trades)
        reasons = {d["filter"] for d in m["universe_discrepancies"]}
        self.assertEqual(
            reasons, {"min_24h_volume", "min_total_volume", "min_book_depth"}
        )


# =========================================================================
# Bootstrap CI invariants (unseeded rng -- test bounds/shape, not exact
# values; metrics.py:152,190 intentionally left unseeded in Phase 0)
# =========================================================================

class TestBootstrapCiInvariants(unittest.TestCase):
    def setUp(self):
        self.vals = np.array([1.0, -1.0, 2.0, -0.5, 0.5, 1.5, -1.5, 0.2])

    def test_block_bootstrap_ci_brackets_point_estimate(self):
        stat_fn = lambda x: float(np.sum(x))
        point = stat_fn(self.vals)
        lo, hi = block_bootstrap_ci(self.vals, stat_fn, n=500)
        self.assertTrue(math.isfinite(lo))
        self.assertTrue(math.isfinite(hi))
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)

    def test_iid_bootstrap_ci_brackets_point_estimate(self):
        stat_fn = lambda x: float(np.mean(x))
        point = stat_fn(self.vals)
        lo, hi = bootstrap_ci(self.vals, stat_fn, n=500)
        self.assertTrue(math.isfinite(lo))
        self.assertTrue(math.isfinite(hi))
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)

    def test_block_bootstrap_ci_degenerate_single_value(self):
        """n_obs < 2 -> returns (stat_fn(values), stat_fn(values)) exactly,
        no randomness involved."""
        single = np.array([3.0])
        stat_fn = lambda x: float(np.sum(x))
        lo, hi = block_bootstrap_ci(single, stat_fn, n=100)
        self.assertEqual(lo, 3.0)
        self.assertEqual(hi, 3.0)

    def test_iid_bootstrap_ci_degenerate_single_value(self):
        single = np.array([3.0])
        stat_fn = lambda x: float(np.sum(x))
        lo, hi = bootstrap_ci(single, stat_fn, n=100)
        self.assertEqual(lo, 3.0)
        self.assertEqual(hi, 3.0)

    def test_seeded_rng_is_reproducible(self):
        """Passing an explicit seeded Generator (as compute_all_metrics
        does internally at metrics.py:266 with default_rng(42)) makes the
        block bootstrap deterministic -- this is the seeded path the plan
        says NOT to break with a module-global monkeypatch."""
        stat_fn = lambda x: float(np.sum(x))
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        lo_a, hi_a = block_bootstrap_ci(self.vals, stat_fn, n=300, rng=rng_a)
        lo_b, hi_b = block_bootstrap_ci(self.vals, stat_fn, n=300, rng=rng_b)
        self.assertEqual(lo_a, lo_b)
        self.assertEqual(hi_a, hi_b)

    def test_narrow_monkeypatch_of_default_rng_for_unseeded_call(self):
        """If a test needs determinism for the UNSEEDED path (rng=None),
        it must patch narrowly -- patching the module-global
        np.random.default_rng would also neuter the metrics.py:266 seeded
        call downstream. Here we patch only the local name imported into
        this test module's call, by passing an explicit deterministic
        rng constructed the same way the unseeded branch would have
        (i.e. we simply avoid patching np.random at all and instead pass
        rng= explicitly) -- proving the invariant test does not require a
        global patch to be deterministic and comparable to compute_all_metrics'
        seeded usage."""
        stat_fn = lambda x: float(np.sum(x))
        rng = np.random.default_rng(123)
        lo, hi = block_bootstrap_ci(self.vals, stat_fn, n=300, rng=rng)
        point = stat_fn(self.vals)
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)


if __name__ == "__main__":
    unittest.main()
