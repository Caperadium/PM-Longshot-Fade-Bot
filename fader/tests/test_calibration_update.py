"""tests/test_calibration_update.py

Unit tests for the calibration-data updater added to backtest/historical.py
(engine background poller that keeps DATA/historical_prices.csv fresh for
the dashboard's Calibration tab):

  - ContractPriceStore.save() atomic write (temp file + os.replace, no
    residue on success)
  - select_stale_unresolved(): finds tokens fetched while open whose
    resolution was never stamped, so they can be re-checked against Gamma
  - _series_scan_start(): recent-window discovery start date, self-healing
    against the newest end_date already stored per series
  - update_calibration_data(): end-to-end orchestration (network calls
    mocked; no real HTTP)

Run: python -m pytest fader/tests/test_calibration_update.py -v
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

_FADER_ROOT = Path(__file__).parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))

from backtest.historical import (
    ContractPriceStore,
    _series_scan_start,
    select_stale_unresolved,
    update_calibration_data,
)
from config.config_loader import SlugRow


class TestSelectStaleUnresolved(unittest.TestCase):
    """select_stale_unresolved: pure grouping/filtering, no network/store."""

    def setUp(self):
        # Fixed "now" so grace-window math is deterministic.
        self.now = datetime(2026, 7, 19, 1, 30, 0, tzinfo=timezone.utc)

    @staticmethod
    def _row(token_id, slug, resolution, end_date):
        return {
            "slug": slug, "token_id": token_id, "date": "2026-01-01",
            "price": "0.9", "resolution": resolution, "end_date": end_date,
            "fetched_at": "",
        }

    def test_qualifying_token_selected(self):
        rows = [
            self._row("A", "m-a", "", "2026-07-01"),
            self._row("A", "m-a", "", "2026-07-02"),  # 2nd row, still all-empty
        ]
        result = select_stale_unresolved(rows, self.now)
        self.assertEqual(result, [("m-a", "A", "2026-07-02")])

    def test_token_with_resolution_row_skipped(self):
        rows = [
            self._row("B", "m-b", "", "2026-07-01"),
            self._row("B", "m-b", "NO", "2026-07-01"),  # resolution stamped
        ]
        result = select_stale_unresolved(rows, self.now)
        self.assertEqual(result, [])

    def test_empty_end_date_skipped(self):
        rows = [self._row("C", "m-c", "", "")]
        result = select_stale_unresolved(rows, self.now)
        self.assertEqual(result, [])

    def test_unparseable_end_date_skipped(self):
        rows = [self._row("C2", "m-c2", "", "not-a-date")]
        result = select_stale_unresolved(rows, self.now)
        self.assertEqual(result, [])

    def test_future_end_date_skipped(self):
        rows = [self._row("D", "m-d", "", "2026-12-31")]
        result = select_stale_unresolved(rows, self.now)
        self.assertEqual(result, [])

    def test_grace_window_excludes_recent_close(self):
        # end_date is "today" (relative to self.now); +2h grace hasn't
        # elapsed yet at 01:30 UTC.
        rows = [self._row("E", "m-e", "", "2026-07-19")]
        result = select_stale_unresolved(rows, self.now, grace_hours=2)
        self.assertEqual(result, [])

    def test_grace_window_includes_after_grace_elapsed(self):
        later = self.now + timedelta(hours=1)  # now 02:30, past the 02:00 grace cutoff
        rows = [self._row("E", "m-e", "", "2026-07-19")]
        result = select_stale_unresolved(rows, later, grace_hours=2)
        self.assertEqual(result, [("m-e", "E", "2026-07-19")])

    def test_cap_and_oldest_first_ordering(self):
        rows = []
        for i in range(5):
            end_date = f"2026-01-0{i + 1}"
            rows.append(self._row(f"T{i}", f"m-{i}", "", end_date))
        result = select_stale_unresolved(rows, self.now, cap=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(
            [r[2] for r in result],
            ["2026-01-01", "2026-01-02", "2026-01-03"],
        )


class TestSeriesScanStart(unittest.TestCase):
    def setUp(self):
        self.today = date(2026, 7, 19)

    def test_latest_end_date_none_falls_back_to_lookback_window(self):
        result = _series_scan_start(
            today=self.today, lookback_days=14,
            latest_end_date=None, series_from=date(2024, 1, 1),
        )
        self.assertEqual(result, date(2026, 7, 5))  # today - 14d

    def test_latest_end_date_older_than_lookback_extends_window_back(self):
        # Self-healing: an outage left the newest stored end_date well
        # before the lookback window -- scan should reach back to it.
        result = _series_scan_start(
            today=self.today, lookback_days=14,
            latest_end_date=date(2026, 6, 1), series_from=date(2024, 1, 1),
        )
        self.assertEqual(result, date(2026, 6, 1))

    def test_latest_end_date_newer_than_lookback_uses_lookback_window(self):
        result = _series_scan_start(
            today=self.today, lookback_days=14,
            latest_end_date=date(2026, 7, 15), series_from=date(2024, 1, 1),
        )
        self.assertEqual(result, date(2026, 7, 5))  # lookback window wins

    def test_series_from_floor_applies(self):
        # series_from is newer than the lookback window -> clamp forward.
        result = _series_scan_start(
            today=self.today, lookback_days=30,
            latest_end_date=None, series_from=date(2026, 7, 10),
        )
        self.assertEqual(result, date(2026, 7, 10))


class TestAtomicSave(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="fader_calib_atomic_")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_leaves_valid_csv_no_tmp_residue(self):
        store_path = Path(self._tmpdir) / "historical_prices.csv"
        store = ContractPriceStore(path=store_path)
        store.upsert("m-a", "tok-a", "2026-01-01", 0.9, end_date="2026-01-05")
        store.save()

        self.assertTrue(store_path.exists())
        self.assertFalse(store_path.with_suffix(".tmp").exists())

        reloaded = ContractPriceStore(path=store_path)
        rows = reloaded.get_rows("tok-a")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["price"], "0.9")


class TestUpdateCalibrationData(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="fader_calib_update_")
        self._store_path = Path(self._tmpdir) / "historical_prices.csv"

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_store(self) -> ContractPriceStore:
        return ContractPriceStore(path=self._store_path)

    def test_discovery_uses_recent_window_not_series_from_date(self):
        store = self._make_store()
        slug_row = SlugRow(
            slug="bitcoin-above-on", enabled=True, market_kind="series",
            series_from_date="2024-01-01", series_filter="bitcoin-above",
        )
        configs = {slug_row.slug: slug_row}

        with patch("backtest.historical.discover_series_markets",
                   return_value=[]) as mock_discover, \
             patch("backtest.historical._fetch_one_market") as mock_fetch, \
             patch("backtest.historical.fetch_market_metadata") as mock_meta, \
             patch("backtest.historical.fetch_price_history") as mock_hist:
            update_calibration_data(store, [slug_row.slug], configs, lookback_days=14)

        mock_discover.assert_called_once()
        kwargs = mock_discover.call_args.kwargs
        self.assertEqual(kwargs["base_slug"], slug_row.slug)
        self.assertEqual(kwargs["series_filter"], "bitcoin-above")
        # Recent window, NOT the full series_from_date backfill scan.
        self.assertNotEqual(kwargs["from_date"], date(2024, 1, 1))
        self.assertGreater(kwargs["from_date"], date(2024, 1, 1))
        # No stored rows -> nothing stale, nothing new discovered -> no
        # fetch/metadata calls at all.
        mock_fetch.assert_not_called()
        mock_meta.assert_not_called()
        mock_hist.assert_not_called()

    def test_only_tokens_absent_from_store_are_fetched(self):
        store = self._make_store()
        store.upsert(
            "bitcoin-above-jan-1", "tok-existing", "2026-01-01", 0.9,
            end_date="2026-01-05",
        )
        store.save()

        slug_row = SlugRow(
            slug="bitcoin-above-on", enabled=True, market_kind="series",
            series_from_date="2024-01-01", series_filter="bitcoin-above",
        )
        configs = {slug_row.slug: slug_row}

        discovered = [
            {"slug": "bitcoin-above-jan-5", "token_id": "tok-existing",
             "end_date": "2026-01-05", "resolution": ""},
            {"slug": "bitcoin-above-jan-6", "token_id": "tok-new",
             "end_date": "2026-01-06", "resolution": ""},
        ]

        def fake_fetch_one(item):
            slug, token_id, end_date, resolution = item
            return {
                "slug": slug, "token_id": token_id,
                "history": [{"t": 1750000000, "p": 0.5}],
                "end_date": end_date, "resolution": resolution,
            }

        with patch("backtest.historical.discover_series_markets",
                   return_value=discovered), \
             patch("backtest.historical._fetch_one_market",
                   side_effect=fake_fetch_one) as mock_fetch, \
             patch("backtest.historical.fetch_market_metadata",
                   return_value=None), \
             patch("backtest.historical.fetch_price_history", return_value=[]):
            stats = update_calibration_data(store, [slug_row.slug], configs)

        # Only the not-yet-stored token gets submitted to the worker.
        self.assertEqual(mock_fetch.call_count, 1)
        called_item = mock_fetch.call_args[0][0]
        self.assertEqual(called_item[1], "tok-new")
        self.assertEqual(stats["new_fetched"], 1)
        self.assertEqual(stats["discovered"], 2)

        # No .tmp residue after the trailing store.save().
        self.assertTrue(self._store_path.exists())
        self.assertFalse(self._store_path.with_suffix(".tmp").exists())

    def test_unresolved_past_end_token_gets_resolution_stamped(self):
        store = self._make_store()
        # Fetched while open long ago: resolution never stamped, end_date
        # is well past the grace window -- a select_stale_unresolved hit.
        store.upsert(
            "some-binary-market", "tok-stale", "2020-01-01", 0.9,
            end_date="2020-01-05",
        )
        store.save()

        meta = {
            "clobTokenIds": json.dumps(["tok-yes", "tok-stale"]),
            "outcomes": json.dumps(["Yes", "No"]),
            "outcomePrices": json.dumps(["1", "0"]),
            "closed": True,
        }

        # slugs=[] -- isolate step 3 (stale re-check, driven purely by
        # store contents) from steps 1-2 (discovery/fetch, driven by the
        # slugs argument).
        with patch("backtest.historical.discover_series_markets") as mock_discover, \
             patch("backtest.historical._fetch_one_market") as mock_fetch, \
             patch("backtest.historical.fetch_market_metadata",
                   return_value=meta) as mock_meta, \
             patch("backtest.historical.fetch_price_history",
                   return_value=[{"t": 1577836800, "p": 0.91}]) as mock_hist:
            stats = update_calibration_data(store, [], {})

        mock_discover.assert_not_called()
        mock_fetch.assert_not_called()
        mock_meta.assert_called_once_with("some-binary-market")
        mock_hist.assert_called_once_with("tok-stale")

        self.assertEqual(stats["resolutions_stamped"], 1)
        self.assertEqual(stats["failed"], 0)

        rows = store.get_rows("tok-stale")
        self.assertTrue(any(r["resolution"] for r in rows))

        self.assertTrue(self._store_path.exists())
        self.assertFalse(self._store_path.with_suffix(".tmp").exists())

    def test_still_open_token_is_skipped_not_stamped(self):
        store = self._make_store()
        store.upsert(
            "some-binary-market", "tok-open", "2020-01-01", 0.9,
            end_date="2020-01-05",
        )
        store.save()

        meta = {
            "clobTokenIds": json.dumps(["tok-yes", "tok-open"]),
            "outcomes": json.dumps(["Yes", "No"]),
            "outcomePrices": json.dumps(["1", "0"]),
            "closed": False,  # not resolved yet on Gamma
        }

        with patch("backtest.historical.discover_series_markets") as mock_discover, \
             patch("backtest.historical._fetch_one_market") as mock_fetch, \
             patch("backtest.historical.fetch_market_metadata", return_value=meta), \
             patch("backtest.historical.fetch_price_history") as mock_hist:
            stats = update_calibration_data(store, [], {})

        mock_discover.assert_not_called()
        mock_fetch.assert_not_called()
        mock_hist.assert_not_called()  # never re-pulled -- still open
        self.assertEqual(stats["resolutions_stamped"], 0)

        rows = store.get_rows("tok-open")
        self.assertTrue(all(not r["resolution"] for r in rows))


if __name__ == "__main__":
    unittest.main()
