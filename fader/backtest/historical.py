"""backtest/historical.py

Historical price data fetcher and storage.
Adapted from V2-Polymarket-BTC-Pricing-Program/core/backtesting/polymarket_fetcher.py.

Fetches CLOB /prices-history for each slug's NO token and stores in
DATA/historical_prices.csv with deduplication on (token_id, date).

Generic daily series: set ``market_kind=series`` in slugs.csv and provide
``series_filter`` (substring that identifies individual market slugs) and
``series_from_date`` (YYYY-MM-DD when the series started). The fetcher
discovers all child markets by querying Gamma /events for every date from
series_from_date to yesterday.

Backward compat: ``market_kind=btc_daily`` is treated as an alias for
series with filter="bitcoin-above" and from_date="2024-01-01".
"""

from __future__ import annotations

import csv
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Progress callback: receives a human-readable status line. Default prints to
# stdout (flushed) so a CLI run shows live progress and any hang is obvious.
ProgressFn = Callable[[str], None]


def _default_progress(msg: str) -> None:
    print(f"[fetch] {msg}", flush=True)
    logger.info(msg)

CLOB_PRICES_URL = "https://clob.polymarket.com/prices-history"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
DATA_DIR = Path(__file__).parent.parent / "DATA"
PRICES_CSV = DATA_DIR / "historical_prices.csv"
MAX_RETRIES = 3
CHECKPOINT_EVERY = 100  # save store every N fetched markets (interrupt safety)

BTC_SERIES_SLUG = "bitcoin-above-on"
# Backward-compat alias — no longer referenced in generic code paths.
# Listed here so dashboard/backtest_page.py imports still resolve.

# Shared series-discovery utilities (moved to rest_market.py)
from marketdata.rest_market import (
    GAMMA_EVENTS_URL,
    CALL_DELAY_S,
    _derive_series_filter,
    _series_date_slug_groups,
    _resolution_from_outcome_prices,
    discover_series_markets,
    parse_series_date,
)

def _get_json(url: str, params: Optional[Dict] = None, timeout: int = 15) -> Any:
    for retry in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = 2 ** (retry + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if retry == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** retry)


def _clob_history_raw(token_id: str, **params: Any) -> List[Dict[str, Any]]:
    """Raw CLOB /prices-history call returning the ``history`` list (or [])."""
    p: Dict[str, Any] = {"market": token_id}
    p.update(params)
    try:
        data = _get_json(CLOB_PRICES_URL, params=p)
        history = data.get("history", []) if isinstance(data, dict) else []
        return history if isinstance(history, list) else []
    except Exception as e:
        logger.warning(f"fetch_price_history({token_id[:16]}, {params}): {e}")
        return []


def fetch_price_history(token_id: str) -> List[Dict[str, Any]]:
    """
    Fetch daily price candles for a token from CLOB /prices-history.

    Mirrors the V2 BTC pricing program's proven strategy (which the fidelity=720
    fallback is what actually recovers legacy daily markets):

      1. interval=1d  -- native daily candles; full history for markets that
         have them (returns 0 for older legacy tokens).
      2. fallback interval=max & fidelity=720 (12h candles); prefer the
         midnight-UTC candle per day, else keep all points.

    CLOB's ``fidelity`` is in MINUTES (720 = 12h); the daily grid is requested
    via ``interval`` not fidelity. CLOB ``t`` is UNIX seconds.

    Returns list of {"t": seconds, "p": price} dicts.
    """
    # --- attempt 1: native daily candles ---
    history = _clob_history_raw(token_id, interval="1d")
    if history:
        return history

    # --- attempt 2: 12h candles, prefer midnight-UTC points ---
    time.sleep(CALL_DELAY_S)
    all_720 = _clob_history_raw(token_id, interval="max", fidelity=720)
    midnight = [
        pt for pt in all_720
        if datetime.fromtimestamp(int(pt.get("t", 0)), tz=timezone.utc).hour == 0
        and int(pt.get("t", 0)) > 0
    ]
    return midnight or all_720


def resolve_no_token_id(slug: str) -> Optional[str]:
    """Resolve NO token_id for a slug via Gamma API."""
    try:
        data = _get_json(GAMMA_MARKETS_URL, params={"slug": slug})
        markets = data if isinstance(data, list) else data.get("markets", [])
        if not markets:
            # Resolved/closed markets are excluded from the default query;
            # retry including closed so historical slugs still resolve.
            data = _get_json(GAMMA_MARKETS_URL, params={"slug": slug, "closed": "true"})
            markets = data if isinstance(data, list) else data.get("markets", [])
        if not markets:
            return None
        mkt = markets[0]
        raw_tokens = mkt.get("clobTokenIds", "[]")
        raw_outcomes = mkt.get("outcomes", "[]")
        token_ids = json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
        outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
        for i, o in enumerate(outcomes):
            if o.strip().lower() == "no" and i < len(token_ids):
                return token_ids[i]
    except Exception as e:
        logger.warning(f"resolve_no_token_id({slug}): {e}")
    return None


class ContractPriceStore:
    """
    CSV store: one row per (token_id, date).
    Schema: slug, token_id, date, price, resolution, end_date
    """

    SCHEMA = ["slug", "token_id", "date", "price", "resolution", "end_date", "fetched_at"]

    def __init__(self, path: Path = PRICES_CSV) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: List[Dict[str, str]] = []
        self._index: Dict[str, int] = {}  # (token_id, date) -> row index
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        with open(self._path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._data = list(reader)
        # Backfill SCHEMA columns missing from older-version CSVs
        for row in self._data:
            for col in self.SCHEMA:
                if col not in row:
                    row[col] = ""
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index = {
            (row["token_id"], row["date"]): i
            for i, row in enumerate(self._data)
        }

    def upsert(
        self,
        slug: str,
        token_id: str,
        date: str,
        price: float,
        resolution: Optional[str] = None,
        end_date: str = "",
    ) -> None:
        key = (token_id, date)
        now_iso = datetime.now(timezone.utc).isoformat()
        row = {
            "slug": slug,
            "token_id": token_id,
            "date": date,
            "price": str(price),
            "resolution": resolution or "",
            "end_date": end_date,
            "fetched_at": now_iso,
        }
        if key in self._index:
            self._data[self._index[key]] = row
        else:
            self._index[key] = len(self._data)
            self._data.append(row)

    def save(self) -> None:
        with open(self._path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SCHEMA)
            writer.writeheader()
            writer.writerows(self._data)

    def get_rows(self, token_id: str) -> List[Dict[str, str]]:
        return [r for r in self._data if r["token_id"] == token_id]

    def all_slugs(self) -> List[str]:
        return list({r["slug"] for r in self._data})

    def snapshot(self) -> "pd.DataFrame":
        """Return an immutable copy of the store as a DataFrame.

        Rows are deduplicated on (token_id, date), keeping the latest
        ``fetched_at`` timestamp when available.  The returned DataFrame
        is detached from the mutable store — it will not change on
        subsequent upserts or re-fetches.
        """
        import pandas as pd  # deferred — light import, avoid top-level

        if not self._data:
            return pd.DataFrame(columns=self.SCHEMA)
        df = pd.DataFrame(self._data)
        # Keep the latest fetch per (token_id, date). When fetched_at is
        # available use it to resolve duplicates; otherwise keep the last
        # row in store order (the most recent upsert comes last).
        if "fetched_at" in df.columns:
            df = df.sort_values("fetched_at").drop_duplicates(
                subset=["token_id", "date"], keep="last"
            )
        else:
            df = df.drop_duplicates(
                subset=["token_id", "date"], keep="last"
            )
        return df.reset_index(drop=True)

    def resolution_path(self) -> Path:
        """Path to the companion resolutions CSV (separate from price data)."""
        return self._path.parent / "resolutions.csv"

    def save_resolution(
        self,
        slug: str,
        token_id: str,
        resolution: str,
        resolved_at: Optional[str] = None,
    ) -> None:
        """Append a resolution record to the resolutions CSV.

        Resolutions are stored separately from price rows so that
        re-fetches cannot silently overwrite settlement data.  The
        backtest engine joins resolutions onto the price snapshot at
        settlement time.
        """
        import csv
        rpath = self.resolution_path()
        existed = rpath.exists()
        with open(rpath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["slug", "token_id", "resolution", "resolved_at"]
            )
            if not existed:
                writer.writeheader()
            writer.writerow({
                "slug": slug,
                "token_id": token_id,
                "resolution": resolution,
                "resolved_at": resolved_at or datetime.now(timezone.utc).isoformat(),
            })

    def load_resolutions(self) -> Dict[str, str]:
        """Load resolutions as {token_id: resolution} dict (latest wins)."""
        rpath = self.resolution_path()
        if not rpath.exists():
            return {}
        import csv
        out: Dict[str, str] = {}
        with open(rpath, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[row["token_id"]] = row["resolution"]
        return out


def _fetch_one_market(
    item: Tuple[str, Optional[str], str, str]
) -> Dict[str, Any]:
    """Worker: resolve token (if needed) and pull price history for one market.

    Pure network/IO, no shared state — safe to run in a thread pool. Returns a
    dict the main thread consumes to mutate the store serially.
    """
    slug, known_token_id, end_date, resolution = item
    token_id = known_token_id or resolve_no_token_id(slug)
    if token_id is None:
        return {"slug": slug, "token_id": None, "history": [],
                "end_date": end_date, "resolution": resolution}
    history = fetch_price_history(token_id)
    return {"slug": slug, "token_id": token_id, "history": history,
            "end_date": end_date, "resolution": resolution}


def fetch_and_store(
    slugs: List[str],
    store: Optional[ContractPriceStore] = None,
    refresh_existing: bool = False,
    slug_configs: Optional[Dict[str, Any]] = None,
    progress: Optional[ProgressFn] = None,
    max_workers: int = 12,
) -> ContractPriceStore:
    """
    Fetch historical prices for all given slugs and store them.

    Price pulls run concurrently across a thread pool (network-bound); the
    store is mutated only on the main thread as results arrive, so no locking
    is needed and checkpointing stays consistent.

    Args:
        slugs: Market slugs to fetch. A slug whose config has
               ``market_kind`` in (``"series"``, ``"btc_daily"``) is expanded
               via discover_series_markets before fetching.
        store: Existing store to update (creates new if None)
        refresh_existing: Re-fetch even if data exists
        slug_configs: Optional ``{slug: config}`` dict where each config has
                      ``.market_kind`` (str), ``.series_from_date`` (str),
                      ``.series_filter`` (str). Accepts SlugRow or any
                      duck-typed object with those attributes.
        progress: Optional callback receiving status lines. Defaults to
                  printing to stdout so progress/hangs are visible in the CLI.
        max_workers: Thread-pool size for concurrent CLOB price pulls.

    Returns:
        Updated ContractPriceStore
    """
    report = progress or _default_progress

    if store is None:
        store = ContractPriceStore()

    configs = slug_configs or {}

    report(f"Starting fetch for {len(slugs)} slug(s): {', '.join(slugs)}")

    # Expand series slugs into individual markets. Each entry carries
    # (slug, token_id, end_date, resolution) so the store can be settled.
    expanded: List[Tuple[str, Optional[str], str, str]] = []
    for slug in slugs:
        cfg = configs.get(slug)
        kind = cfg.market_kind if cfg else "binary"

        # backward compat: btc_daily → series
        if kind == "btc_daily":
            series_filter = "bitcoin-above"
            series_from = "2024-01-01"
        elif kind == "series":
            series_filter = (cfg.series_filter if cfg and cfg.series_filter
                             else _derive_series_filter(slug))
            series_from = (cfg.series_from_date if cfg and cfg.series_from_date
                           else "2024-01-01")
        else:
            expanded.append((slug, None, "", ""))
            continue

        report(
            f"Expanding series slug '{slug}' "
            f"(filter='{series_filter}', from={series_from})..."
        )
        try:
            from_date = datetime.strptime(series_from, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            from_date = date(2024, 1, 1)
        for m in discover_series_markets(
            base_slug=slug,
            series_filter=series_filter,
            from_date=from_date,
            progress=report,
        ):
            expanded.append((
                m["slug"], m["token_id"],
                m.get("end_date", ""), m.get("resolution", ""),
            ))

    # Pre-skip markets whose token we already have (resume support). Items with
    # an unknown token are always submitted (resolved inside the worker).
    existing_tokens = {r["token_id"] for r in store._data}
    work: List[Tuple[str, Optional[str], str, str]] = []
    n_skipped = 0
    for item in expanded:
        known = item[1]
        if known and known in existing_tokens and not refresh_existing:
            n_skipped += 1
        else:
            work.append(item)

    n_total = len(work)
    report(
        f"Fetching price history for {n_total} market(s) "
        f"({n_skipped} already cached) using {max_workers} workers..."
    )
    n_fetched = n_failed = n_points = 0

    def _apply(res: Dict[str, Any]) -> int:
        """Apply one worker result to the store (main thread). Returns points added."""
        slug, token_id = res["slug"], res["token_id"]
        end_date, resolution = res["end_date"], res["resolution"]
        if token_id is None:
            return -1  # signal failure
        added = 0
        last_date: Optional[str] = None
        for point in res["history"]:
            ts = point.get("t", 0)  # CLOB returns UNIX seconds
            price = float(point.get("p", 0))
            if ts <= 0 or price <= 0:
                continue
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")
            store.upsert(slug, token_id, date_str, price, end_date=end_date)
            added += 1
            if last_date is None or date_str > last_date:
                last_date = date_str
        # Stamp resolution onto the final observed day so the backtest can settle
        # positions held to expiry (payout = 1 if NO won, else 0).
        if resolution and last_date is not None:
            last_price = next(
                (float(r["price"]) for r in store.get_rows(token_id)
                 if r["date"] == last_date), 0.0,
            )
            store.upsert(slug, token_id, last_date, last_price,
                         resolution=resolution, end_date=end_date)
        return added

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one_market, item): item for item in work}
        done = 0
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            added = _apply(res)
            if added < 0:
                n_failed += 1
                report(f"  [{done}/{n_total}] {res['slug']}: could not resolve token_id")
                continue
            n_fetched += 1
            n_points += added
            report(
                f"  [{done}/{n_total}] {res['slug']}: {added} points "
                f"(res={res['resolution'] or 'open'})"
            )
            # Periodic checkpoint so a long run survives interruption and can be
            # resumed (rerun with refresh_existing=False skips already-fetched).
            if n_fetched % CHECKPOINT_EVERY == 0:
                store.save()
                report(f"  ...checkpoint saved ({len(store._data)} rows)")

    store.save()
    report(
        f"Fetch complete: {n_fetched} fetched, {n_skipped} skipped, "
        f"{n_failed} failed; {n_points} points; "
        f"{len(store._data)} total rows saved to {store._path.name}"
    )
    return store
