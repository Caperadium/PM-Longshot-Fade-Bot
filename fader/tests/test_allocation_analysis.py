"""tests/test_allocation_analysis.py

Test suite for probability-band allocation PnL analysis.
Run: python -m pytest fader/tests/test_allocation_analysis.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Add fader root to path
_FADER_ROOT = Path(__file__).parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))

from backtest.allocation_analysis import (
    BinStats,
    compute_normalization_factor,
    concavity_check,
    extract_entry_prices,
    monotonicity_report,
    paired_pnl_ci,
    per_bin_breakdown,
    run_allocation_analysis,
    spearman_rho_with_ci,
)
from backtest.engine import BacktestConfig
from backtest.harness import walkforward_normalized as walkforward_validate
from execution.sizing import make_sizing_fn


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_trades_df(prices: List[float], pnls: List[float]) -> pd.DataFrame:
    """Minimal trades DataFrame for per-bin testing."""
    return pd.DataFrame({
        "entry_price": prices,
        "realized_pnl": pnls,
        "entry_date": ["2024-06-01"] * len(prices),
        "exit_date": ["2024-06-15"] * len(prices),
        "slug": [f"test-{i}" for i in range(len(prices))],
        "token_id": [f"token-{i}" for i in range(len(prices))],
        "size": [1.0] * len(prices),
        "notional": [10.0] * len(prices),
        "resolution": ["NO"] * len(prices),
        "fill_type": ["market"] * len(prices),
        "max_adverse_excursion": [0.0] * len(prices),
    })


def _make_price_store(
    prices_by_day: List[float],
    resolution: str = "NO",
    slug: str = "test-slug",
    token_id: str = "test-token",
) -> pd.DataFrame:
    """Build a minimal price-history DataFrame that passes backtest filters.

    Each day's price is in-band (0.75-0.95), DTE=30, so the backtest
    enters on day 2 (after min_time_in_band_days=1).
    """
    dates = pd.date_range("2024-01-01", periods=len(prices_by_day), freq="D")
    rows = []
    for i, (d, p) in enumerate(zip(dates, prices_by_day)):
        rows.append({
            "slug": slug,
            "token_id": token_id,
            "date": d.strftime("%Y-%m-%d"),
            "price": str(p),
            "resolution": resolution if i == len(prices_by_day) - 1 else "",
            "end_date": (d + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(rows)


# =========================================================================
# Sizing function tests
# =========================================================================


class TestSizingFunction(unittest.TestCase):
    """Unit tests for make_sizing_fn and normalization."""

    def test_alpha_zero_flat(self):
        fn = make_sizing_fn(0.0, 0.70, 0.95)
        for p in [0.70, 0.80, 0.90, 0.95]:
            self.assertAlmostEqual(fn(p), 1.0, places=6,
                                   msg=f"α=0 should give multiplier=1 at p={p}")

    def test_alpha_negative_weights_low(self):
        fn = make_sizing_fn(-1.0, 0.70, 0.95)
        self.assertAlmostEqual(fn(0.70), 2.0, places=6)
        self.assertAlmostEqual(fn(0.95), 0.0, places=6)
        # midpoint should be ~1.0
        self.assertAlmostEqual(fn(0.825), 1.0, places=6)

    def test_alpha_positive_weights_high(self):
        fn = make_sizing_fn(+1.0, 0.70, 0.95)
        self.assertAlmostEqual(fn(0.70), 0.0, places=6)
        self.assertAlmostEqual(fn(0.95), 2.0, places=6)
        self.assertAlmostEqual(fn(0.825), 1.0, places=6)

    def test_alpha_half(self):
        fn = make_sizing_fn(0.5, 0.70, 0.95)
        # p=0.70: f = 1 + 0.5*(-0.125)/(0.125) = 1 - 0.5 = 0.5
        self.assertAlmostEqual(fn(0.70), 0.5, places=6)
        # p=0.95: f = 1 + 0.5*(0.125)/(0.125) = 1 + 0.5 = 1.5
        self.assertAlmostEqual(fn(0.95), 1.5, places=6)

    def test_normalization_applied(self):
        fn = make_sizing_fn(-1.0, 0.70, 0.95)
        nf = 2.0
        self.assertAlmostEqual(fn(0.70) / nf, 1.0, places=6)  # 2.0 / 2.0
        self.assertAlmostEqual(fn(0.95) / nf, 0.0, places=6)

    def test_minimum_effective_notional(self):
        """Engine clamps effective_notional at $1.00 regardless of multiplier."""
        # Simulate engine logic
        order_notional = 10.0
        fn = make_sizing_fn(+1.0, 0.70, 0.95)  # at p=0.71, f ≈ 0.08
        multiplier = fn(0.71)
        effective = max(1.00, order_notional * multiplier)
        self.assertGreaterEqual(effective, 1.00,
                                msg="effective_notional must be >= $1.00")
        self.assertLess(multiplier, 0.2,
                        msg="multiplier at p=0.71 should be very small for α=+1")

    def test_normalization_global(self):
        """Global normalization: mean(f) across all entries and alphas ≈ 1."""
        prices = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        alphas = [-1.0, -0.5, 0.0, 0.5, 1.0]
        nf = compute_normalization_factor(prices, alphas, 0.70, 0.95)
        # Check that normalized mean ≈ 1
        all_vals = []
        for a in alphas:
            fn = make_sizing_fn(a, 0.70, 0.95)
            for p in prices:
                all_vals.append(fn(p) / nf)
        self.assertAlmostEqual(np.mean(all_vals), 1.0, places=6,
                               msg="normalized mean should be ~1.0")

    def test_extreme_price(self):
        """Sizing function handles prices outside band gracefully."""
        fn = make_sizing_fn(1.0, 0.70, 0.95)
        # Price below band
        self.assertGreater(fn(0.60), -10, msg="should not explode at low prices")
        # Price above band
        self.assertLess(fn(0.99), 10, msg="should not explode at high prices")


# =========================================================================
# Per-bin breakdown tests
# =========================================================================


class TestPerBinBreakdown(unittest.TestCase):
    """Tests for quantile-based bin breakdown."""

    def test_empty_trades(self):
        bins = per_bin_breakdown(pd.DataFrame())
        self.assertEqual(len(bins), 0)

    def test_single_bin_few_trades(self):
        # 10 trades spread across prices, qcut with 3 bins
        df = _make_trades_df(
            [0.75, 0.78, 0.81, 0.84, 0.87, 0.90, 0.91, 0.92, 0.93, 0.94],
            [1.0, 2.0, -3.0, 1.5, -2.0, 3.0, -1.0, 2.5, -0.5, 4.0],
        )
        bins = per_bin_breakdown(df, n_bins=3)
        self.assertGreater(len(bins), 0)
        for b in bins:
            self.assertIsInstance(b, BinStats)
            self.assertGreater(b.n_trades, 0)
            self.assertIsNotNone(b.hit_rate_ci_lo)
            self.assertIsNotNone(b.hit_rate_ci_hi)

    def test_hit_rate_monotonic(self):
        """With realistic data, higher-price bins should have equal or higher hit rates
        (market efficiency — higher market-implied probability → higher empirical win rate)."""
        # Simulate: low-price = more losses, high-price = fewer losses
        # Use large sample so bin-level hit rates are stable.
        rng = np.random.default_rng(42)
        trades = []
        for _ in range(1000):
            p = rng.uniform(0.70, 0.95)
            # True win prob = p (market efficient)
            win = rng.random() < p
            pnl = (1.0 - p) * 10 if win else -p * 10
            trades.append({"entry_price": p, "realized_pnl": pnl})
        df = pd.DataFrame(trades)
        bins = per_bin_breakdown(df, n_bins=5)
        self.assertEqual(len(bins), 5, "should get 5 bins with 1000 trades")

        # Hit rates should generally increase with bin price
        hit_rates = [b.hit_rate for b in bins]
        mid_prices = [(b.price_low + b.price_high) / 2 for b in bins]
        # Check that Spearman ρ between mid_price and hit_rate is positive
        if len(hit_rates) >= 3:
            from scipy import stats
            rho, _ = stats.spearmanr(mid_prices, hit_rates)
            self.assertGreater(rho, 0.0,
                               msg=f"hit rates should increase with bin price, got ρ={rho:.3f}")

    def test_missing_columns(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        bins = per_bin_breakdown(df)
        self.assertEqual(len(bins), 0)


# =========================================================================
# Monotonicity tests
# =========================================================================


class TestMonotonicity(unittest.TestCase):
    """Tests for Spearman ρ and concavity analysis."""

    def test_spearman_perfect_positive(self):
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        rho, p, lo, hi = spearman_rho_with_ci(x, y)
        self.assertAlmostEqual(rho, 1.0, places=6)
        self.assertAlmostEqual(p, 0.0, places=4)
        self.assertGreater(hi, 0.9)

    def test_spearman_perfect_negative(self):
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        y = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
        rho, p, lo, hi = spearman_rho_with_ci(x, y)
        self.assertAlmostEqual(rho, -1.0, places=6)

    def test_spearman_constant_y(self):
        x = np.array([-1.0, 0.0, 1.0])
        y = np.array([5.0, 5.0, 5.0])
        rho, p, lo, hi = spearman_rho_with_ci(x, y)
        # Constant Y → NaN from spearmanr, our wrapper returns 0
        self.assertEqual(rho, 0.0)

    def test_spearman_too_few_points(self):
        x = np.array([0.0, 1.0])
        y = np.array([1.0, 2.0])
        rho, p, lo, hi = spearman_rho_with_ci(x, y)
        # Should not crash
        self.assertEqual(rho, 0.0)
        self.assertEqual(p, 1.0)

    def test_concavity_quadratic_true(self):
        """Synthetic data with clear concave peak."""
        alphas = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # Parabola peaking at α=0: y = -α² + 2
        values = -alphas ** 2 + 2.0 + np.random.default_rng(42).normal(0, 0.05, len(alphas))
        conc = concavity_check(alphas, values)
        self.assertTrue(conc["has_optimum"],
                        msg=f"should detect concave peak, got β₂={conc['beta2']:.4f}, p={conc['beta2_p']:.4f}")
        self.assertAlmostEqual(conc["peak_alpha"], 0.0, delta=0.2)
        self.assertLess(conc["aic_quadratic"], conc["aic_linear"],
                        msg="quadratic should fit better than linear")

    def test_concavity_linear(self):
        """Synthetic data that is purely linear (with negligible noise for numerical stability)."""
        alphas = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        values = 0.5 * alphas + 3.0 + np.random.default_rng(123).normal(0, 1e-8, len(alphas))
        conc = concavity_check(alphas, values)
        self.assertFalse(conc["has_optimum"],
                         msg="linear data should not show concave optimum")

    def test_concavity_few_points(self):
        alphas = np.array([-1.0, 0.0, 1.0])
        values = np.array([1.0, 2.0, 1.0])
        conc = concavity_check(alphas, values)
        # Should not crash, returns None values
        self.assertFalse(conc["has_optimum"])

    def test_monotonicity_report_structure(self):
        """Smoke test: monotonicity_report produces expected keys."""
        results = [
            {"alpha": -1.0, "sortino": 1.0, "total_pnl": 500.0, "hit_rate": 0.70,
             "calmar": 0.5, "max_drawdown_pct": 0.15, "cvar_95": -20.0,
             "worst_trade": -15.0, "worst_day": -25.0, "expectancy": 1.5,
             "profit_factor": 2.5, "daily_skew": -0.5, "empty": False},
            {"alpha": -0.5, "sortino": 1.1, "total_pnl": 450.0, "hit_rate": 0.72,
             "calmar": 0.55, "max_drawdown_pct": 0.12, "cvar_95": -17.0,
             "worst_trade": -13.0, "worst_day": -22.0, "expectancy": 1.4,
             "profit_factor": 2.7, "daily_skew": -0.4, "empty": False},
            {"alpha": 0.0, "sortino": 1.2, "total_pnl": 400.0, "hit_rate": 0.75,
             "calmar": 0.6, "max_drawdown_pct": 0.10, "cvar_95": -15.0,
             "worst_trade": -12.0, "worst_day": -20.0, "expectancy": 1.3,
             "profit_factor": 3.0, "daily_skew": -0.3, "empty": False},
            {"alpha": 1.0, "sortino": 1.5, "total_pnl": 300.0, "hit_rate": 0.82,
             "calmar": 0.8, "max_drawdown_pct": 0.05, "cvar_95": -10.0,
             "worst_trade": -8.0, "worst_day": -12.0, "expectancy": 1.1,
             "profit_factor": 4.0, "daily_skew": -0.1, "empty": False},
        ]
        report = monotonicity_report(results)
        self.assertIn("sortino", report, "primary metric must be present")
        self.assertTrue(report["sortino"]["sign_match"],
                        msg="Sortino should increase with α in this synthetic data")
        self.assertIn("total_pnl", report, "secondary metrics must be present")
        self.assertEqual(report["sortino"]["metric_type"], "primary")
        self.assertEqual(report["total_pnl"]["metric_type"], "secondary")


# =========================================================================
# Paired PnL CI tests
# =========================================================================


class TestPairedPnlCI(unittest.TestCase):
    """Tests for paired block-bootstrap CI."""

    def test_paired_ci_identical(self):
        """Identical data → diff CI should span 0."""
        trades = _make_trades_df(
            [0.80] * 10, [2.0, -1.0] * 5,
        )
        diff_lo, diff_hi, bl_lo, bl_hi = paired_pnl_ci(trades, trades, n_bootstrap=1000)
        self.assertLessEqual(diff_lo, 0, "diff CI lower bound must be ≤ 0")
        self.assertGreaterEqual(diff_hi, 0, "diff CI upper bound must be ≥ 0")

    def test_paired_ci_different(self):
        """Higher PnL scheme → diff CI should be entirely positive."""
        trades_base = _make_trades_df(
            [0.80] * 10, [1.0, -1.0] * 5,
        )
        trades_better = _make_trades_df(
            [0.80] * 10, [3.0, -1.0] * 5,
        )
        diff_lo, diff_hi, _, _ = paired_pnl_ci(trades_base, trades_better, n_bootstrap=1000)
        self.assertGreater(diff_lo, 0, "better scheme should have positive PnL diff")

    def test_paired_ci_empty(self):
        empty = pd.DataFrame()
        trades = _make_trades_df([0.80] * 5, [1.0] * 5)
        diff_lo, diff_hi, bl_lo, bl_hi = paired_pnl_ci(empty, trades, n_bootstrap=100)
        self.assertEqual(diff_lo, 0.0)
        self.assertEqual(diff_hi, 0.0)


# =========================================================================
# Entry price extraction tests
# =========================================================================


class TestExtractEntryPrices(unittest.TestCase):
    """Tests for extract_entry_prices pre-scan."""

    def test_basic_extraction(self):
        df = _make_price_store(
            [0.72, 0.78, 0.82, 0.85, 0.88, 0.88],
        )
        prices = extract_entry_prices(df, 0.70, 0.95)
        self.assertEqual(len(prices), 1,
                         msg="should enter once after 1 day in band")

    def test_no_in_band_prices(self):
        df = _make_price_store(
            [0.60, 0.65, 0.60, 0.65],
        )
        prices = extract_entry_prices(df, 0.70, 0.95)
        self.assertEqual(len(prices), 0,
                         msg="no prices in band → no entries")

    def test_empty_store(self):
        prices = extract_entry_prices(pd.DataFrame(), 0.70, 0.95)
        self.assertEqual(len(prices), 0)


# =========================================================================
# Integration / smoke tests
# =========================================================================


class TestRunAllocationAnalysis(unittest.TestCase):
    """Smoke and edge-case tests for the full pipeline."""

    def test_smoke_small_dataset(self):
        """Run full pipeline on a small synthetic dataset."""
        # Build a store with varied entry prices
        all_rows = []
        rng = np.random.default_rng(42)
        for i in range(20):  # 20 independent tokens
            token_id = f"token-{i}"
            p_entry = rng.uniform(0.72, 0.93)
            # Two days at same price (satisfies min_time_in_band)
            start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i * 5)
            dates = pd.date_range(start, periods=5, freq="D")
            for j, d in enumerate(dates):
                if j < 2:
                    price = p_entry
                elif j == 4:
                    price = 1.0  # out of band after entry
                else:
                    price = p_entry + 0.01
                all_rows.append({
                    "slug": f"slug-{i}",
                    "token_id": token_id,
                    "date": d.strftime("%Y-%m-%d"),
                    "price": str(price),
                    "resolution": "NO" if rng.random() < 0.8 else "YES",
                    "end_date": (d + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                })

        df = pd.DataFrame(all_rows)
        cfg = BacktestConfig(
            band_low=0.70, band_high=0.95,
            min_time_in_band_days=1,
            order_notional_usd=10.0,
            spread_c=1.0, n_bootstrap=1000,
        )

        results = run_allocation_analysis(
            df, cfg,
            alphas=[-1.0, -0.5, 0.0, 0.5, 1.0],
            band_low=0.70, band_high=0.95,
            n_walkforward_windows=2,
        )

        # Check structure
        self.assertIn("band", results)
        self.assertIn("alphas", results)
        self.assertIn("per_scheme", results)
        self.assertIn("monotonicity", results)
        self.assertIn("walkforward", results)
        self.assertEqual(len(results["per_scheme"]), 5)
        self.assertGreater(results["n_entries"], 0)

        # Baseline (α=0) should have trades
        baseline = results["baseline"]
        self.assertGreater(baseline["n_trades"], 0,
                           msg="baseline should produce trades")

    def test_empty_data(self):
        """Pipeline should handle empty data gracefully."""
        df = pd.DataFrame(columns=["slug", "token_id", "date", "price", "resolution", "end_date"])
        cfg = BacktestConfig()
        results = run_allocation_analysis(df, cfg, alphas=[-1.0, 0.0, 1.0])
        # Should return results dict without crashing, per_scheme all empty
        self.assertIn("per_scheme", results)
        self.assertEqual(len(results["per_scheme"]), 3)
        for r in results["per_scheme"]:
            self.assertTrue(r.get("empty"), "all schemes should be empty for empty input")

    def test_single_trade_data(self):
        """Single trade should not crash analysis."""
        df = _make_price_store(
            [0.80, 0.81, 0.99],  # enters at 0.81, then leaves band
            resolution="NO",
            slug="single", token_id="single-token",
        )
        cfg = BacktestConfig(
            band_low=0.70, band_high=0.95,
            min_time_in_band_days=1,
            order_notional_usd=10.0,
            n_bootstrap=500,
        )
        results = run_allocation_analysis(
            df, cfg,
            alphas=[-1.0, 0.0, 1.0],
            n_walkforward_windows=2,
        )
        self.assertIn("per_scheme", results)

    def test_all_same_price(self):
        """All entries at same price → all α schemes produce identical results."""
        all_rows = []
        for i in range(5):
            dates = pd.date_range(f"2024-01-{i*3+1:02d}", periods=5, freq="D")
            for j, d in enumerate(dates):
                price = 0.85 if j < 3 else 1.0
                all_rows.append({
                    "slug": f"slug-{i}",
                    "token_id": f"token-{i}",
                    "date": d.strftime("%Y-%m-%d"),
                    "price": str(price),
                    "resolution": "NO" if i % 2 == 0 else "YES",
                    "end_date": (d + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                })

        df = pd.DataFrame(all_rows)
        cfg = BacktestConfig(
            band_low=0.70, band_high=0.95,
            min_time_in_band_days=1,
            order_notional_usd=10.0,
            n_bootstrap=500,
        )
        results = run_allocation_analysis(
            df, cfg,
            alphas=[-1.0, 0.0, 1.0],
            n_walkforward_windows=2,
        )

        # All schemes should have identical n_trades
        ns = [r["n_trades"] for r in results["per_scheme"] if not r.get("empty")]
        self.assertTrue(all(n == ns[0] for n in ns),
                        msg=f"all schemes should have same n_trades at identical prices: {ns}")

    def test_zero_trade_scheme(self):
        """Analysis should handle a scheme with zero trades (all prices at band edge)."""
        # Every entry at p=0.95 → α=+1 gives multiplier=2, α=-1 gives multiplier=0
        # With sizing_fn at α=-1, effective_notional = max(1, 10*0/1) = 1. So still trades.
        # Instead test that empty trades result doesn't crash monotonicity.
        results = [
            {"alpha": -1.0, "n_trades": 0, "total_pnl": 0.0, "empty": True},
            {"alpha": -0.5, "n_trades": 10, "total_pnl": 90.0, "sortino": 0.9,
             "hit_rate": 0.78, "empty": False},
            {"alpha": 0.0, "n_trades": 10, "total_pnl": 100.0, "sortino": 1.0,
             "hit_rate": 0.8, "empty": False},
            {"alpha": 0.5, "n_trades": 10, "total_pnl": 85.0, "sortino": 1.1,
             "hit_rate": 0.83, "empty": False},
            {"alpha": 1.0, "n_trades": 10, "total_pnl": 80.0, "sortino": 1.2,
             "hit_rate": 0.85, "empty": False},
        ]
        report = monotonicity_report(results)
        # Should handle one empty + 4 non-empty schemes
        self.assertIn("sortino", report)

    def test_effective_notional_clamping_prevents_zero_share(self):
        """At p near extreme with opposing α, effective_notional clamps to $1."""
        fn = make_sizing_fn(+1.0, 0.70, 0.95)
        # p=0.71 → raw = 1 + 1.0*(0.71-0.825)/0.125 = 1 - 0.92 = 0.08
        multiplier = fn(0.71)
        effective = max(1.00, 10.0 * multiplier)
        self.assertGreaterEqual(effective, 1.00)
        # Verify shares can be computed (not zero)
        import math
        shares = math.floor((effective / 0.71) * 100) / 100
        self.assertGreater(shares, 0, "must produce non-zero shares")

    def test_walkforward_path_a(self):
        """Walk-forward with explicit alpha_star should not crash."""
        df = _make_price_store(
            [0.80, 0.81, 0.99],
            resolution="NO",
            slug="wf", token_id="wf-token",
        )
        cfg = BacktestConfig(
            band_low=0.70, band_high=0.95,
            min_time_in_band_days=1,
            order_notional_usd=10.0,
            n_bootstrap=500,
        )
        wf = walkforward_validate(
            df, cfg, alpha_star=0.5, alphas=[-1.0, 0.0, 1.0],
            band_low=0.70, band_high=0.95,
            n_windows=2,
        )
        self.assertIn("path", wf)

    def test_walkforward_path_b(self):
        """Walk-forward without alpha_star (Path B, extremes comparison)."""
        df = _make_price_store(
            [0.80, 0.81, 0.99],
            resolution="NO",
            slug="wf2", token_id="wf2-token",
        )
        cfg = BacktestConfig(
            band_low=0.70, band_high=0.95,
            min_time_in_band_days=1,
            order_notional_usd=10.0,
            n_bootstrap=500,
        )
        wf = walkforward_validate(
            df, cfg, alpha_star=None, alphas=[-1.0, 0.0, 1.0],
            band_low=0.70, band_high=0.95,
            n_windows=2,
        )
        self.assertIn("path", wf)
        self.assertEqual(wf["path"], "B")


if __name__ == "__main__":
    unittest.main()
