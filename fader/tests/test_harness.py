"""tests/test_harness.py

Tests for backtest/harness.py (Phase 5, see temp/implementation-plan.md).

Covers:
  - run_config matches direct run_backtest + compute_all_metrics on a
    synthetic store (no drift introduced by the extraction wrapper).
  - run_grid over a small grid of configs (sequential + multi-worker paths).
  - Each of the three walk-forward variants (walkforward_normalized,
    walkforward_lean, window_stability) smoke-tested on synthetic windows
    -- they are mathematically distinct and must stay that way (see
    harness.py's module docstring), so each gets its own dedicated test
    rather than a shared parametrized one.

Run: python -m pytest fader/tests/test_harness.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

# Add fader root to path
_FADER_ROOT = Path(__file__).parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))

from backtest.engine import BacktestConfig, run_backtest
from backtest.harness import (
    BandCache,
    MetricsRow,
    build_band_cache,
    run_config,
    run_grid,
    walkforward_lean,
    walkforward_normalized,
    window_stability,
)
from backtest.metrics import compute_all_metrics
from execution.sizing import make_sizing_fn


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_contract_rows(slug, token_id, prices, resolution, start="2024-01-01", end_date="2024-06-01"):
    dates = pd.date_range(start, periods=len(prices), freq="D")
    rows = []
    n = len(prices)
    for i, (d, p) in enumerate(zip(dates, prices)):
        res = resolution if i == n - 1 else ""
        rows.append({
            "slug": slug, "token_id": token_id,
            "date": d.strftime("%Y-%m-%d"), "price": str(p),
            "resolution": res, "end_date": end_date,
        })
    return rows


def _small_store() -> pd.DataFrame:
    """3 independent contracts, entries a few days apart, all resolve NO."""
    rows = []
    rows += _make_contract_rows("t1", "tok-1", [0.85, 0.86, 0.87, 1.0], "NO", start="2024-01-01")
    rows += _make_contract_rows("t2", "tok-2", [0.82, 0.83, 0.84, 1.0], "NO", start="2024-02-01")
    rows += _make_contract_rows("t3", "tok-3", [0.90, 0.91, 0.92, 1.0], "YES", start="2024-03-01")
    return pd.DataFrame(rows)


def _spanning_store(n_months: int = 4) -> pd.DataFrame:
    """One contract per calendar month so walk-forward windows each get data."""
    rows = []
    for m in range(1, n_months + 1):
        start = f"2024-{m:02d}-01"
        rows += _make_contract_rows(
            f"wf-{m}", f"tok-wf-{m}",
            [0.82, 0.83, 0.84, 1.0], "NO" if m % 2 == 0 else "YES",
            start=start,
        )
    return pd.DataFrame(rows)


_CFG = BacktestConfig(
    band_low=0.80, band_high=0.95,
    min_dte=0, max_dte=365,
    min_time_in_band_days=1,
    order_notional_usd=10.0,
    spread_c=1.0, slippage_c=0.0, adverse_selection_c=0.0,
    n_bootstrap=200,
)


# ---------------------------------------------------------------------------
# run_config
# ---------------------------------------------------------------------------


class TestRunConfig(unittest.TestCase):
    def test_matches_direct_run_backtest_and_metrics(self):
        df = _small_store()
        row = run_config(df, _CFG)

        trades_df, _ = run_backtest(df, _CFG)
        expected = compute_all_metrics(
            trades_df, n_bootstrap=_CFG.n_bootstrap, initial_capital=500.0,
            skipped_filters=trades_df.attrs.get("skipped_filters"),
        )

        self.assertFalse(row.empty)
        self.assertEqual(row.n_trades, expected["n_trades"])
        self.assertAlmostEqual(row.total_pnl, expected["total_pnl"], places=9)
        self.assertAlmostEqual(row.sortino, expected["sortino"] or 0.0, places=9)
        self.assertEqual(row.calmar, expected["calmar"])
        self.assertAlmostEqual(row.hit_rate, expected["hit_rate"], places=9)
        self.assertEqual(row.pnl_ci_95, expected["pnl_ci_95"])
        self.assertEqual(len(row.trades_df), len(trades_df))

    def test_empty_result_on_no_trades(self):
        df = pd.DataFrame(_make_contract_rows("out", "tok-out", [0.50, 0.50, 0.50], "NO"))
        row = run_config(df, _CFG)
        self.assertTrue(row.empty)
        self.assertEqual(row.n_trades, 0)
        self.assertEqual(row.total_pnl, 0.0)

    def test_empty_store_returns_empty_row(self):
        row = run_config(pd.DataFrame(columns=["slug", "token_id", "date", "price", "resolution", "end_date"]), _CFG)
        self.assertTrue(row.empty)

    def test_returns_metrics_row_type(self):
        df = _small_store()
        row = run_config(df, _CFG)
        self.assertIsInstance(row, MetricsRow)


# ---------------------------------------------------------------------------
# run_grid
# ---------------------------------------------------------------------------


class TestRunGrid(unittest.TestCase):
    def test_grid_over_two_configs_sequential(self):
        df = _small_store()
        cfg_a = BacktestConfig(
            band_low=0.80, band_high=0.95, min_time_in_band_days=1,
            order_notional_usd=10.0, spread_c=1.0, n_bootstrap=200,
        )
        cfg_b = BacktestConfig(
            band_low=0.80, band_high=0.95, min_time_in_band_days=1,
            order_notional_usd=10.0, spread_c=1.0, n_bootstrap=200,
            sizing_fn=make_sizing_fn(0.5, 0.80, 0.95),
        )
        rows = run_grid(df, [cfg_a, cfg_b], workers=1)
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertIsInstance(r, MetricsRow)
        # Order preserved: index 0 corresponds to cfg_a
        row_a = run_config(df, cfg_a)
        self.assertAlmostEqual(rows[0].total_pnl, row_a.total_pnl, places=9)

    def test_grid_order_preserved_with_multiple_workers(self):
        """workers>1 dispatches via ProcessPoolExecutor -- results must
        still come back in the same order as the input grid (chunked
        positionally, not by completion order)."""
        df = _small_store()
        configs = [
            BacktestConfig(
                band_low=bl, band_high=0.95, min_time_in_band_days=1,
                order_notional_usd=10.0, spread_c=1.0, n_bootstrap=100,
            )
            for bl in [0.70, 0.75, 0.80, 0.85]
        ]
        rows = run_grid(df, configs, workers=2)
        self.assertEqual(len(rows), 4)
        # Sequential reference run for order-equivalence check
        seq_rows = [run_config(df, cfg) for cfg in configs]
        for r, s in zip(rows, seq_rows):
            self.assertAlmostEqual(r.total_pnl, s.total_pnl, places=6)
            self.assertEqual(r.n_trades, s.n_trades)


# ---------------------------------------------------------------------------
# Walk-forward variant 1: walkforward_normalized
# (= allocation_analysis.walkforward_validate, normalized sizing)
# ---------------------------------------------------------------------------


class TestWalkforwardNormalized(unittest.TestCase):
    def test_path_a_smoke(self):
        df = _spanning_store(4)
        wf = walkforward_normalized(
            df, _CFG, alpha_star=0.3, alphas=[-1.0, 0.0, 0.3, 1.0],
            band_low=0.80, band_high=0.95, n_windows=2,
        )
        self.assertEqual(wf.get("path"), "A")
        self.assertIn("star_win_frac", wf)

    def test_path_b_smoke(self):
        df = _spanning_store(4)
        wf = walkforward_normalized(
            df, _CFG, alpha_star=None, alphas=[-1.0, 0.0, 1.0],
            band_low=0.80, band_high=0.95, n_windows=2,
        )
        self.assertEqual(wf.get("path"), "B")
        self.assertIn("high_prob_win_frac", wf)


# ---------------------------------------------------------------------------
# Walk-forward variant 2: walkforward_lean
# (= grid_sweep._oos_validate, UNNORMALIZED sizing, per-band baseline cache)
# ---------------------------------------------------------------------------


class TestWalkforwardLean(unittest.TestCase):
    def test_smoke_with_band_cache(self):
        df = _spanning_store(4)
        cache = build_band_cache(df, 0.80, 0.95, n_windows=2, n_bootstrap=200)
        self.assertIsInstance(cache, BandCache)

        sizing_fn = make_sizing_fn(0.5, 0.80, 0.95)
        result = walkforward_lean(
            cache, df, 0.80, 0.95, alpha=0.5, sizing_fn=sizing_fn, n_bootstrap=200,
        )
        self.assertIn("oos_win_frac", result)
        self.assertIn("oos_valid", result)
        self.assertEqual(result["oos_n_windows"], len(cache.windows))


# ---------------------------------------------------------------------------
# Walk-forward variant 3: window_stability
# (= crypto_sweep._run_walkforward_for_top_configs, same fixed config per
# window, no baseline comparison -- reports cross-window consistency grade)
# ---------------------------------------------------------------------------


class TestWindowStability(unittest.TestCase):
    def test_smoke_top_configs(self):
        df = _spanning_store(4)
        df_out = pd.DataFrame([
            {"band_low": 0.80, "alpha": 0.0, "min_dte": 0, "max_dte": 365, "score": 1.0},
        ])
        out = window_stability(
            df_out, df, top_n=1, n_windows=2, n_bootstrap=200,
            order_notional_usd=10.0, initial_capital=500.0,
        )
        self.assertEqual(len(out), 1)
        self.assertIn("stability", out[0])
        self.assertIn("stability_grade", out[0]["stability"])
        self.assertIn("per_window", out[0])


if __name__ == "__main__":
    unittest.main()
