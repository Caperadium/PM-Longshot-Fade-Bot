"""tests/test_backtest_engine.py

Characterization tests for backtest/engine.py (Phase 0 safety net, see
temp/implementation-plan.md). Locks CURRENT run_backtest() behavior --
entry timing, PnL signs, and filter exclusions -- before Phase 4 unifies
the filter stack with the live engine.

Synthetic price-store rows are built directly as a DataFrame (run_backtest
accepts a DataFrame or a ContractPriceStore; a plain DataFrame with the
documented schema -- slug, token_id, date, price, resolution, end_date --
is equivalent and simpler to construct here).

All expected values below (entry price, size, pnl) were derived by running
the engine's own arithmetic by hand (spread/slippage cost, floor-to-cent
sizing, payout formula) in a scratch script, not by calling run_backtest
and asserting whatever it returned.

Run: python -m pytest fader/tests/test_backtest_engine.py -v
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

# Add fader root to path
_FADER_ROOT = Path(__file__).parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))

from backtest.engine import BacktestConfig, run_backtest


def _rows_for_contract(slug, token_id, dates, prices, resolution_at_last,
                        end_date="2024-02-01"):
    rows = []
    n = len(dates)
    for i, (d, p) in enumerate(zip(dates, prices)):
        res = resolution_at_last if i == n - 1 else ""
        rows.append({
            "slug": slug, "token_id": token_id,
            "date": d.strftime("%Y-%m-%d"), "price": str(p),
            "resolution": res, "end_date": end_date,
        })
    return rows


class TestThreeContractCharacterization(unittest.TestCase):
    """3 contracts: one resolves NO (win), one resolves YES (loss), one
    stays out of band the whole time (excluded)."""

    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        rows = []
        rows += _rows_for_contract(
            "resolves-no", "tok-a", dates,
            [0.85, 0.86, 0.87, 0.88, 0.89], "NO",
        )
        rows += _rows_for_contract(
            "resolves-yes", "tok-b", dates,
            [0.90, 0.91, 0.92, 0.93, 0.94], "YES",
        )
        rows += _rows_for_contract(
            "out-of-band", "tok-c", dates,
            [0.50] * 5, "NO",
        )
        self.df = pd.DataFrame(rows)
        self.cfg = BacktestConfig(
            band_low=0.80, band_high=0.95,
            min_dte=0, max_dte=365,
            min_time_in_band_days=1,
            order_notional_usd=10.0,
            spread_c=1.0, slippage_c=0.0, adverse_selection_c=0.0,
        )

    def test_out_of_band_contract_produces_no_trade(self):
        trades_df, _ = run_backtest(self.df, self.cfg)
        self.assertNotIn("out-of-band", set(trades_df["slug"]))

    def test_exactly_two_trades_from_in_band_contracts(self):
        trades_df, _ = run_backtest(self.df, self.cfg)
        self.assertEqual(len(trades_df), 2)
        self.assertEqual(
            sorted(trades_df["slug"].tolist()),
            ["resolves-no", "resolves-yes"],
        )

    def test_entry_enters_on_first_in_band_day(self):
        """min_time_in_band_days=1 -> entry on the very first in-band day
        (2024-01-01), not later."""
        trades_df, _ = run_backtest(self.df, self.cfg)
        for _, row in trades_df.iterrows():
            self.assertEqual(row["entry_date"], "2024-01-01")

    def test_entry_price_and_size_hand_computed_no_contract(self):
        """entry_price = price(0.85) + (spread_c+slippage_c+adverse_c)/100
        = 0.85 + 0.01 = 0.86.
        size = floor((notional/entry_price)*100)/100
             = floor((10/0.86)*100)/100 = floor(1162.79...)/100 = 11.62.
        payout=1 (NO resolution) -> pnl = (1 - 0.86) * 11.62 = 1.6268.
        """
        trades_df, _ = run_backtest(self.df, self.cfg)
        row = trades_df[trades_df["slug"] == "resolves-no"].iloc[0]
        self.assertAlmostEqual(row["entry_price"], 0.86, places=9)
        self.assertAlmostEqual(row["size"], 11.62, places=9)
        self.assertAlmostEqual(row["realized_pnl"], 1.6268, places=6)
        self.assertGreater(row["realized_pnl"], 0.0, "NO resolution -> winning trade")

    def test_entry_price_and_size_hand_computed_yes_contract(self):
        """entry_price = 0.90 + 0.01 = 0.91.
        size = floor((10/0.91)*100)/100 = floor(1098.90...)/100 = 10.98.
        payout=0 (YES resolution) -> pnl = (0 - 0.91) * 10.98 = -9.9918.
        """
        trades_df, _ = run_backtest(self.df, self.cfg)
        row = trades_df[trades_df["slug"] == "resolves-yes"].iloc[0]
        self.assertAlmostEqual(row["entry_price"], 0.91, places=9)
        self.assertAlmostEqual(row["size"], 10.98, places=9)
        self.assertAlmostEqual(row["realized_pnl"], -9.9918, places=6)
        self.assertLess(row["realized_pnl"], 0.0, "YES resolution -> losing trade")

    def test_pnl_signs_match_resolution(self):
        trades_df, _ = run_backtest(self.df, self.cfg)
        no_row = trades_df[trades_df["slug"] == "resolves-no"].iloc[0]
        yes_row = trades_df[trades_df["slug"] == "resolves-yes"].iloc[0]
        self.assertGreater(no_row["realized_pnl"], 0.0)
        self.assertLess(yes_row["realized_pnl"], 0.0)

    def test_equity_curve_final_cumulative_matches_trade_sum(self):
        trades_df, equity_df = run_backtest(self.df, self.cfg)
        expected_total = trades_df["realized_pnl"].sum()
        self.assertAlmostEqual(
            equity_df["cumulative_pnl"].iloc[-1], expected_total, places=6
        )


class TestDteFilterExclusion(unittest.TestCase):
    def test_dte_out_of_range_excludes_contract(self):
        """end_date fixed 2 days out from the first row; DTE shrinks to
        ~0-2 across the 5-day window, always outside a tight [10, 20]
        filter -- no entry should ever occur."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        rows = _rows_for_contract(
            "dte-out-of-range", "tok-e", dates, [0.85] * 5, "NO",
            end_date="2024-01-03",
        )
        df = pd.DataFrame(rows)
        cfg = BacktestConfig(
            band_low=0.80, band_high=0.95,
            min_dte=10, max_dte=20,
            min_time_in_band_days=1,
            order_notional_usd=10.0,
        )
        trades_df, _ = run_backtest(df, cfg)
        self.assertEqual(len(trades_df), 0)


class TestMinTimeInBandFilterExclusion(unittest.TestCase):
    def test_entry_delayed_until_min_time_in_band_satisfied(self):
        """min_time_in_band_days=3: the days-in-band counter increments
        each in-band day; entry should land on the 3rd in-band day
        (2024-01-03), not the 1st."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        rows = _rows_for_contract(
            "tib-test", "tok-f", dates, [0.85] * 5, "NO",
            end_date="2024-06-01",
        )
        df = pd.DataFrame(rows)
        cfg = BacktestConfig(
            band_low=0.80, band_high=0.95,
            min_dte=0, max_dte=365,
            min_time_in_band_days=3,
            order_notional_usd=10.0,
        )
        trades_df, _ = run_backtest(df, cfg)
        self.assertEqual(len(trades_df), 1)
        self.assertEqual(trades_df.iloc[0]["entry_date"], "2024-01-03")

    def test_never_satisfies_min_time_in_band_produces_no_trade(self):
        """Only 2 in-band days total, but min_time_in_band_days=3 --
        the position never opens (and the market resolves anyway, but
        there is no entry to resolve)."""
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        rows = _rows_for_contract(
            "tib-never", "tok-g", dates, [0.85, 0.85], "NO",
            end_date="2024-06-01",
        )
        df = pd.DataFrame(rows)
        cfg = BacktestConfig(
            band_low=0.80, band_high=0.95,
            min_dte=0, max_dte=365,
            min_time_in_band_days=3,
            order_notional_usd=10.0,
        )
        trades_df, _ = run_backtest(df, cfg)
        self.assertEqual(len(trades_df), 0)


class TestDteNoneRowsFailOpenPin(unittest.TestCase):
    """Pins the documented divergence between live (dte=None -> reject,
    fail-closed) and backtest (dte=None -> DTE filter skipped entirely,
    fail-open) that Phase 4's evaluate_entry(missing_dte=...) policy must
    preserve for the backtest engine.

    Mechanism (backtest/engine.py): ``end_date`` empty -> ``dte = None``;
    the range check is only applied ``if dte is not None`` -- when it is
    None, the check is skipped and the row proceeds through the rest of
    the filter stack as if DTE were always in range.
    """

    def test_missing_end_date_yields_dte_none_and_entry_still_allowed(self):
        dates = pd.date_range("2024-03-01", periods=3, freq="D")
        rows = _rows_for_contract(
            "no-end-date", "tok-d", dates, [0.85, 0.86, 0.87], "NO",
            end_date="",  # empty -> compute_dte_from_dates never called; dte=None
        )
        df = pd.DataFrame(rows)
        cfg = BacktestConfig(
            band_low=0.80, band_high=0.95,
            min_dte=0, max_dte=365,
            min_time_in_band_days=1,
            order_notional_usd=10.0,
        )
        trades_df, _ = run_backtest(df, cfg)
        self.assertEqual(
            len(trades_df), 1,
            "dte=None rows currently enter (fail-open) -- this pins the "
            "known divergence from live's fail-closed dte=None handling",
        )
        row = trades_df.iloc[0]
        self.assertEqual(row["entry_date"], "2024-03-01")
        self.assertAlmostEqual(row["entry_price"], 0.86, places=9)
        self.assertAlmostEqual(row["realized_pnl"], 1.6268, places=6)

    def test_even_a_narrow_dte_filter_does_not_exclude_none_dte_rows(self):
        """A narrow [10, 20] DTE window would exclude any contract with a
        real end_date close by -- but with no end_date at all (dte=None)
        the filter never fires, so the trade still occurs. This is the
        precise fail-open behavior Phase 4 must not silently flip."""
        dates = pd.date_range("2024-03-01", periods=3, freq="D")
        rows = _rows_for_contract(
            "no-end-date-narrow", "tok-h", dates, [0.85, 0.86, 0.87], "NO",
            end_date="",
        )
        df = pd.DataFrame(rows)
        cfg = BacktestConfig(
            band_low=0.80, band_high=0.95,
            min_dte=10, max_dte=20,
            min_time_in_band_days=1,
            order_notional_usd=10.0,
        )
        trades_df, _ = run_backtest(df, cfg)
        self.assertEqual(len(trades_df), 1)


if __name__ == "__main__":
    unittest.main()
