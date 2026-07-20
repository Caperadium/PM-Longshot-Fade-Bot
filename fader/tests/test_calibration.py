"""tests/test_calibration.py

Tests for backtest/calibration.py (Phase 3, calibration dashboard plan).

Pure/I-O-free module: no network, no Streamlit. Synthetic snapshot
DataFrames are built with STRING columns throughout, matching the real
CSV-backed ContractPriceStore.snapshot() reality (see historical.py).

Run: python -m pytest fader/tests/test_calibration.py -v
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path
from typing import List

import pandas as pd

# Add fader root to path
_FADER_ROOT = Path(__file__).parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))

from backtest.calibration import (
    CalibrationParams,
    band_summary,
    bot_trade_calibration,
    bucket_calibration,
    build_observations,
    filter_by_series,
    monthly_edge,
    wilson_interval,
)


def _row(
    slug: str,
    token_id: str,
    obs_date: str,
    price,
    resolution: str = "",
    end_date: str = "",
    fetched_at: str = "2026-01-01T00:00:00+00:00",
) -> dict:
    """Build one snapshot row with the CSV-string reality: every value is
    stored/read back as a string."""
    return {
        "slug": slug,
        "token_id": token_id,
        "date": obs_date,
        "price": str(price),
        "resolution": resolution,
        "end_date": end_date,
        "fetched_at": fetched_at,
    }


def _snapshot(rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).astype(str)


class TestWilsonInterval(unittest.TestCase):
    def test_n_zero_guard(self):
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))
        self.assertEqual(wilson_interval(5, 0), (0.0, 0.0))

    def test_known_value_k8_n10(self):
        lo, hi = wilson_interval(8, 10)
        self.assertAlmostEqual(lo, 0.49, delta=0.01)
        self.assertAlmostEqual(hi, 0.94, delta=0.01)

    def test_interval_within_bounds(self):
        for k, n in [(0, 1), (1, 1), (3, 7), (50, 100), (99, 100), (1, 1000)]:
            lo, hi = wilson_interval(k, n)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)
            self.assertLessEqual(lo, hi)

    def test_k_zero(self):
        lo, hi = wilson_interval(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_k_equals_n(self):
        lo, hi = wilson_interval(10, 10)
        self.assertLess(lo, 1.0)
        self.assertLessEqual(hi, 1.0)
        self.assertGreater(lo, 0.0)


class TestBuildObservations(unittest.TestCase):
    def setUp(self):
        self.params = CalibrationParams(dte_days=4, tolerance_days=1)

    def test_exact_hit_at_target(self):
        # end_date - 4 days = 2026-01-06 exactly.
        rows = [
            _row("slug-a", "tok1", "2026-01-06", 0.85, end_date="2026-01-10"),
            _row("slug-a", "tok1", "2026-01-10", 1, resolution="YES", end_date="2026-01-10"),
        ]
        obs = build_observations(_snapshot(rows), self.params)
        self.assertEqual(len(obs), 1)
        r = obs.iloc[0]
        self.assertEqual(r["token_id"], "tok1")
        self.assertEqual(r["slug"], "slug-a")
        self.assertEqual(r["end_date"], date(2026, 1, 10))
        self.assertEqual(r["obs_date"], date(2026, 1, 6))
        self.assertEqual(r["obs_dte"], 4)
        self.assertAlmostEqual(r["no_price"], 0.85)
        self.assertAlmostEqual(r["yes_implied"], 0.15)
        self.assertEqual(r["resolution"], "YES")
        self.assertTrue(bool(r["yes_won"]))

    def test_nearest_within_tolerance_tie_earlier_date(self):
        # target = 2026-01-06. Rows at 01-05 (dist 1) and 01-07 (dist 1)
        # tie; earlier date (01-05) must win.
        rows = [
            _row("slug-b", "tok2", "2026-01-05", 0.80, end_date="2026-01-10"),
            _row("slug-b", "tok2", "2026-01-07", 0.90, end_date="2026-01-10"),
            _row("slug-b", "tok2", "2026-01-10", 1, resolution="NO", end_date="2026-01-10"),
        ]
        obs = build_observations(_snapshot(rows), self.params)
        self.assertEqual(len(obs), 1)
        r = obs.iloc[0]
        self.assertEqual(r["obs_date"], date(2026, 1, 5))
        self.assertAlmostEqual(r["no_price"], 0.80)
        self.assertFalse(bool(r["yes_won"]))

    def test_dropped_when_beyond_tolerance(self):
        # target = 2026-01-06, only candidate at 01-03 -> distance 3 > tolerance 1.
        rows = [
            _row("slug-c", "tok3", "2026-01-03", 0.85, end_date="2026-01-10"),
            _row("slug-c", "tok3", "2026-01-10", 1, resolution="YES", end_date="2026-01-10"),
        ]
        obs = build_observations(_snapshot(rows), self.params)
        self.assertTrue(obs.empty)

    def test_unresolved_token_excluded(self):
        rows = [
            _row("slug-d", "tok4", "2026-01-06", 0.85, end_date="2026-01-10"),
            _row("slug-d", "tok4", "2026-01-10", 0.95, end_date="2026-01-10"),
        ]
        obs = build_observations(_snapshot(rows), self.params)
        self.assertTrue(obs.empty)

    def test_resolution_joined_from_final_row(self):
        # Resolution lives only on the final row; must be joined back onto
        # the earlier (DTE-N) observation row.
        rows = [
            _row("slug-e", "tok5", "2026-01-06", 0.88, resolution="", end_date="2026-01-10"),
            _row("slug-e", "tok5", "2026-01-07", 0.90, resolution="", end_date="2026-01-10"),
            _row("slug-e", "tok5", "2026-01-10", 0.99, resolution="NO", end_date="2026-01-10"),
        ]
        obs = build_observations(_snapshot(rows), self.params)
        self.assertEqual(len(obs), 1)
        r = obs.iloc[0]
        # obs row picked should be the 01-06 exact hit, not the resolution row.
        self.assertAlmostEqual(r["no_price"], 0.88)
        self.assertEqual(r["resolution"], "NO")
        self.assertFalse(bool(r["yes_won"]))

    def test_garbage_end_date_dropped(self):
        rows = [
            _row("slug-f", "tok6", "2026-01-06", 0.85, end_date="not-a-date"),
            _row("slug-f", "tok6", "2026-01-10", 1, resolution="YES", end_date=""),
        ]
        obs = build_observations(_snapshot(rows), self.params)
        self.assertTrue(obs.empty)

    def test_garbage_end_date_row_ignored_but_token_kept(self):
        # One row has a bad end_date; another has a good one. Token
        # should still resolve using the good end_date.
        rows = [
            _row("slug-g", "tok7", "2026-01-06", 0.85, end_date="garbage"),
            _row("slug-g", "tok7", "2026-01-06", 0.85, end_date="2026-01-10"),
            _row("slug-g", "tok7", "2026-01-10", 1, resolution="YES", end_date="2026-01-10"),
        ]
        obs = build_observations(_snapshot(rows), self.params)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs.iloc[0]["end_date"], date(2026, 1, 10))

    def test_empty_input_returns_empty_frame_with_columns(self):
        obs = build_observations(pd.DataFrame(), self.params)
        self.assertTrue(obs.empty)
        expected_cols = {
            "slug", "token_id", "end_date", "obs_date", "obs_dte",
            "no_price", "yes_implied", "resolution", "yes_won",
        }
        self.assertEqual(set(obs.columns), expected_cols)

    def test_window_days_filters_old_end_dates(self):
        rows = [
            _row("slug-h", "tok8", "2026-01-06", 0.85, resolution="", end_date="2026-01-10"),
            _row("slug-h", "tok8", "2026-01-10", 1, resolution="YES", end_date="2026-01-10"),
            _row("slug-i", "tok9", "2025-01-06", 0.85, resolution="", end_date="2025-01-10"),
            _row("slug-i", "tok9", "2025-01-10", 1, resolution="NO", end_date="2025-01-10"),
        ]
        params = CalibrationParams(dte_days=4, tolerance_days=1, window_days=30)
        obs = build_observations(_snapshot(rows), params, as_of=date(2026, 1, 15))
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs.iloc[0]["token_id"], "tok8")

    def test_degenerate_prices_ignored_when_picking_obs(self):
        # 0, 1, 0.0 at/near the target date must be skipped in favor of a
        # valid in-range price, even though the degenerate row is closer
        # to the target date.
        rows = [
            _row("slug-j", "tok10", "2026-01-06", "1", end_date="2026-01-10"),  # degenerate, dist 0
            _row("slug-j", "tok10", "2026-01-07", "0.88", end_date="2026-01-10"),  # valid, dist 1
            _row("slug-j", "tok10", "2026-01-10", "0", resolution="NO", end_date="2026-01-10"),
        ]
        obs = build_observations(_snapshot(rows), self.params)
        self.assertEqual(len(obs), 1)
        r = obs.iloc[0]
        self.assertAlmostEqual(r["no_price"], 0.88)
        self.assertEqual(r["obs_date"], date(2026, 1, 7))

    def test_zero_and_one_and_zero_point_zero_all_ignored(self):
        rows = [
            _row("slug-k", "tok11", "2026-01-05", "0", end_date="2026-01-10"),
            _row("slug-k", "tok11", "2026-01-06", "1", end_date="2026-01-10"),
            _row("slug-k", "tok11", "2026-01-07", "0.0", end_date="2026-01-10"),
            _row("slug-k", "tok11", "2026-01-10", "0.99", resolution="YES", end_date="2026-01-10"),
        ]
        obs = build_observations(_snapshot(rows), self.params)
        # No valid (0,1)-open-interval candidate within tolerance of the
        # target (2026-01-06) -> token dropped.
        self.assertTrue(obs.empty)


class TestBucketCalibration(unittest.TestCase):
    def _obs(self, rows: List[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_hand_built_buckets(self):
        rows = [
            {"slug": "s", "token_id": "a", "end_date": date(2026, 1, 1),
             "obs_date": date(2025, 12, 28), "obs_dte": 4,
             "no_price": 0.88, "yes_implied": 0.12, "resolution": "NO", "yes_won": False},
            {"slug": "s", "token_id": "b", "end_date": date(2026, 1, 1),
             "obs_date": date(2025, 12, 28), "obs_dte": 4,
             "no_price": 0.87, "yes_implied": 0.13, "resolution": "YES", "yes_won": True},
            {"slug": "s", "token_id": "c", "end_date": date(2026, 1, 1),
             "obs_date": date(2025, 12, 28), "obs_dte": 4,
             "no_price": 0.78, "yes_implied": 0.22, "resolution": "YES", "yes_won": True},
        ]
        result = bucket_calibration(self._obs(rows), bucket_width=0.05)
        # Bucket [0.10, 0.15) mid 0.125: rows a, b.
        b1 = result[result["bucket_mid"].round(4) == 0.125].iloc[0]
        self.assertEqual(b1["n"], 2)
        self.assertAlmostEqual(b1["mean_implied"], 0.125)
        self.assertAlmostEqual(b1["yes_rate"], 0.5)
        self.assertAlmostEqual(b1["edge"], 0.125 - 0.5)
        # Bucket [0.20, 0.25) mid 0.225: row c only.
        b2 = result[result["bucket_mid"].round(4) == 0.225].iloc[0]
        self.assertEqual(b2["n"], 1)
        self.assertAlmostEqual(b2["mean_implied"], 0.22)
        self.assertAlmostEqual(b2["yes_rate"], 1.0)
        self.assertAlmostEqual(b2["edge"], 0.22 - 1.0)
        self.assertEqual(len(result), 2)

    def test_empty_buckets_absent(self):
        rows = [
            {"slug": "s", "token_id": "a", "end_date": date(2026, 1, 1),
             "obs_date": date(2025, 12, 28), "obs_dte": 4,
             "no_price": 0.90, "yes_implied": 0.10, "resolution": "NO", "yes_won": False},
        ]
        result = bucket_calibration(self._obs(rows), bucket_width=0.05)
        self.assertEqual(len(result), 1)

    def test_empty_input(self):
        result = bucket_calibration(pd.DataFrame(), bucket_width=0.05)
        self.assertTrue(result.empty)
        self.assertIn("bucket_mid", result.columns)


class TestBandSummary(unittest.TestCase):
    def _obs(self, rows):
        return pd.DataFrame(rows)

    def test_known_values_and_boundary_inclusivity(self):
        rows = [
            {"no_price": 0.80, "yes_won": False},  # boundary low, included, NO win
            {"no_price": 0.95, "yes_won": False},  # boundary high, included, NO win
            {"no_price": 0.79, "yes_won": False},  # excluded (below band)
            {"no_price": 0.96, "yes_won": True},   # excluded (above band)
            {"no_price": 0.85, "yes_won": True},   # inside, YES win (NO loses)
        ]
        result = band_summary(self._obs(rows), 0.80, 0.95)
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["avg_no_price"], (0.80 + 0.95 + 0.85) / 3)
        # no_win_rate = mean(not yes_won) over the 3 in-band rows: 2 NO wins, 1 loss
        self.assertAlmostEqual(result["no_win_rate"], 2 / 3)
        self.assertAlmostEqual(result["edge_pp"], (2 / 3) - ((0.80 + 0.95 + 0.85) / 3))
        self.assertGreaterEqual(result["wilson_high"], result["wilson_low"])

    def test_n_zero_returns_zeros(self):
        result = band_summary(pd.DataFrame(), 0.80, 0.95)
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["avg_no_price"], 0.0)
        self.assertEqual(result["no_win_rate"], 0.0)
        self.assertEqual(result["edge_pp"], 0.0)
        self.assertEqual(result["wilson_low"], 0.0)
        self.assertEqual(result["wilson_high"], 0.0)

    def test_n_zero_when_no_rows_in_band(self):
        rows = [{"no_price": 0.50, "yes_won": True}]
        result = band_summary(self._obs(rows), 0.80, 0.95)
        self.assertEqual(result["n"], 0)


class TestMonthlyEdge(unittest.TestCase):
    def test_two_months_known_edges_sorted(self):
        rows = [
            {"end_date": date(2026, 1, 5), "no_price": 0.80, "yes_won": False},
            {"end_date": date(2026, 1, 15), "no_price": 0.90, "yes_won": True},
            {"end_date": date(2026, 2, 3), "no_price": 0.85, "yes_won": False},
            {"end_date": date(2026, 2, 10), "no_price": 0.85, "yes_won": False},
        ]
        result = monthly_edge(pd.DataFrame(rows), 0.80, 0.95)
        self.assertEqual(list(result["month"]), ["2026-01", "2026-02"])

        jan = result[result["month"] == "2026-01"].iloc[0]
        self.assertEqual(jan["n"], 2)
        self.assertAlmostEqual(jan["avg_no_price"], (0.80 + 0.90) / 2)
        self.assertAlmostEqual(jan["no_win_rate"], 0.5)  # one NO win, one loss
        self.assertAlmostEqual(jan["edge_pp"], 0.5 - (0.80 + 0.90) / 2)

        feb = result[result["month"] == "2026-02"].iloc[0]
        self.assertEqual(feb["n"], 2)
        self.assertAlmostEqual(feb["avg_no_price"], 0.85)
        self.assertAlmostEqual(feb["no_win_rate"], 1.0)  # both NO wins
        self.assertAlmostEqual(feb["edge_pp"], 1.0 - 0.85)

    def test_empty_input(self):
        result = monthly_edge(pd.DataFrame(), 0.80, 0.95)
        self.assertTrue(result.empty)


class TestBotTradeCalibration(unittest.TestCase):
    def test_pnl_sign_determines_wins(self):
        rows = [
            {"entry_price": "0.88", "realized_pnl": "1.5"},   # win
            {"entry_price": "0.87", "realized_pnl": "-8.5"},  # loss
            {"entry_price": "0.86", "realized_pnl": 2.0},     # win (numeric already)
        ]
        result = bot_trade_calibration(pd.DataFrame(rows), bucket_width=0.05)
        self.assertEqual(len(result), 1)  # all three fall in [0.85, 0.90) bucket
        r = result.iloc[0]
        self.assertEqual(r["n"], 3)
        self.assertAlmostEqual(r["avg_entry"], (0.88 + 0.87 + 0.86) / 3)
        self.assertAlmostEqual(r["win_rate"], 2 / 3)
        self.assertAlmostEqual(r["edge_pp"], (2 / 3) - (0.88 + 0.87 + 0.86) / 3)

    def test_none_and_nan_pnl_dropped(self):
        rows = [
            {"entry_price": "0.88", "realized_pnl": None},
            {"entry_price": "0.87", "realized_pnl": float("nan")},
            {"entry_price": "0.90", "realized_pnl": "3.0"},
        ]
        result = bot_trade_calibration(pd.DataFrame(rows), bucket_width=0.05)
        total_n = int(result["n"].sum())
        self.assertEqual(total_n, 1)

    def test_string_inputs_coerced(self):
        rows = [{"entry_price": "0.91", "realized_pnl": "-1.0"}]
        result = bot_trade_calibration(pd.DataFrame(rows), bucket_width=0.05)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["avg_entry"], 0.91)
        self.assertAlmostEqual(result.iloc[0]["win_rate"], 0.0)

    def test_empty_input(self):
        result = bot_trade_calibration(pd.DataFrame(), bucket_width=0.05)
        self.assertTrue(result.empty)


class TestFilterBySeries(unittest.TestCase):
    def test_substring_match(self):
        rows = [
            {"slug": "bitcoin-above-100k-on-2026-01-01", "token_id": "a"},
            {"slug": "will-the-highest-temperature-in-seoul-be-3c", "token_id": "b"},
            {"slug": "ethereum-above-5k-on-2026-01-01", "token_id": "c"},
        ]
        obs = pd.DataFrame(rows)
        result = filter_by_series(obs, "bitcoin-above")
        self.assertEqual(list(result["token_id"]), ["a"])

        result2 = filter_by_series(obs, "highest-temperature-in-seoul")
        self.assertEqual(list(result2["token_id"]), ["b"])

    def test_empty_input(self):
        result = filter_by_series(pd.DataFrame(), "bitcoin-above")
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
