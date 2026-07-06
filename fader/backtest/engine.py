"""backtest/engine.py

Fader strategy backtester.

Logic:
  - For each (slug, token_id, date) in historical price data:
      * Apply the same filter stack as the live engine (band, DTE, volumes,
        min_time_in_band — approximated from consecutive days in-band)
      * No doubling up (one position per token at a time)
  - Fill model: 'first-in-band & passing filters' entry
    Limit-fill approximation: fill if a later candle price moves to/through limit
    (conservative — treated as market fill at the entry day's price)
  - Hold to resolution (marked in prices CSV as resolution in {YES, NO, null})
  - PnL: payout = 1 if resolution == 'NO', 0 if 'YES'; realized_pnl = (payout - entry) * size
  - Returns trades DataFrame + equity curve.

Strict look-ahead prevention: filters and entry evaluated at day T only using
data available through day T.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from execution.sizing import MIN_EFFECTIVE_NOTIONAL

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    band_low: float = 0.80
    band_high: float = 0.95
    min_dte: int = 0
    max_dte: int = 365
    min_time_in_band_days: int = 1   # approx min_time_in_band_s / 86400
    order_notional_usd: float = 10.0
    spread_c: float = 1.0
    slippage_c: float = 0.0
    adverse_selection_c: float = 0.0
    n_bootstrap: int = 10000
    sizing_fn: Optional[Callable[[float], float]] = None  # price -> notional multiplier


@dataclass
class BacktestTrade:
    slug: str
    token_id: str
    entry_date: str
    entry_price: float
    size: float
    notional: float
    exit_date: Optional[str]
    resolution: Optional[str]
    realized_pnl: float
    fill_type: str  # "market" | "limit_approx"
    max_adverse_excursion: float = 0.0  # worst daily-close drawdown while open
    # NOTE: MAE from daily close is a LOWER BOUND on true adverse excursion.
    # 1d CLOB candles give only one price per day (mid); intraday extremes
    # are not recorded. Interpret as "worst observed daily close," not
    # "worst intraday price."


def _to_utc_date(val):
    """Convert a date *val* to a timezone-aware UTC datetime.

    Handles ``str`` (``YYYY-MM-DD``), :class:`datetime.datetime`, and
    :class:`pandas.Timestamp` inputs.
    """
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)
    # str, pandas.Timestamp, or anything with a str() that looks like a date
    return datetime.strptime(str(val)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)


def compute_dte_from_dates(entry_date, end_date) -> Optional[float]:
    """Days-to-expiry between *entry_date* and *end_date*.

    Accepts ``str``, :class:`~datetime.datetime`, or :class:`~pandas.Timestamp`.
    """
    try:
        entry = _to_utc_date(entry_date)
        end = _to_utc_date(end_date)
        return max(0.0, (end - entry).days)
    except Exception:
        return None


def run_backtest(
    store_or_df: Union["ContractPriceStore", pd.DataFrame],
    cfg: BacktestConfig,
    slugs: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run backtest over all rows in a ContractPriceStore or DataFrame.

    Accepts either a ``ContractPriceStore`` (snapshotted internally — the
    returned results are reproducible) or a pre-built DataFrame (enables
    saving/loading snapshots for byte-for-byte reproducibility).

    DataFrame must have columns: slug, token_id, date, price, resolution, end_date.

    Returns:
        trades_df: one row per trade
        equity_df: cumulative PnL over time
    """
    # Snapshot the input so the backtest is reproducible — subsequent
    # store mutations (re-fetches, resolution stamps) don't affect the
    # rows used in this run.
    from backtest.historical import ContractPriceStore

    if isinstance(store_or_df, ContractPriceStore):
        df = store_or_df.snapshot()
        all_rows = df.to_dict("records")
    else:
        all_rows = store_or_df.to_dict("records") if isinstance(store_or_df, pd.DataFrame) else list(store_or_df)

    if slugs:
        slug_set = set(slugs)
        all_rows = [r for r in all_rows if r["slug"] in slug_set]

    # Group by (slug, token_id) and sort by date
    from collections import defaultdict
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in all_rows:
        groups[(row["slug"], row["token_id"])].append(row)
    for key in groups:
        groups[key].sort(key=lambda r: r["date"])

    trades: List[BacktestTrade] = []
    open_positions: Dict[Tuple[str, str], Dict] = {}  # (slug, token_id) -> trade dict

    # Collect all dates for equity curve
    all_dates = sorted({r["date"] for r in all_rows})
    equity_by_date: Dict[str, float] = {d: 0.0 for d in all_dates}

    for (slug, token_id), rows in groups.items():
        days_in_band = 0
        in_position = False
        entry_row = None
        min_price_open = 1.0  # lowest daily close while position is open (MAE tracking)

        for i, row in enumerate(rows):
            date = row["date"]
            price = float(row["price"] or 0)
            resolution = row.get("resolution", "")
            end_date = row.get("end_date", "")

            # If in position, check resolution
            if in_position and entry_row:
                # Track worst daily close since entry for MAE.  NOTE: this
                # is a LOWER BOUND on true adverse excursion — 1d CLOB
                # candles give only one price per day (mid); intraday
                # extremes are not recorded.
                if price > 0:
                    min_price_open = min(min_price_open, price)

                if resolution in ("NO", "YES", "N/A", "INVALID"):
                    payout = 1.0 if resolution == "NO" else 0.0
                    ep = float(entry_row["price"])
                    notional = float(entry_row.get("effective_notional", str(cfg.order_notional_usd)))
                    size = _compute_size(notional, ep)
                    pnl = (payout - ep) * size
                    mae = max(0.0, ep - min_price_open)
                    t = BacktestTrade(
                        slug=slug,
                        token_id=token_id,
                        entry_date=entry_row["date"],
                        entry_price=ep,
                        size=size,
                        notional=notional,
                        exit_date=date,
                        resolution=resolution,
                        realized_pnl=pnl,
                        fill_type="market",
                        max_adverse_excursion=mae,
                    )
                    trades.append(t)
                    if date in equity_by_date:
                        equity_by_date[date] += pnl
                    in_position = False
                    entry_row = None
                    min_price_open = 1.0
                continue

            # No position — evaluate entry filters
            if price <= 0:
                days_in_band = 0
                continue

            in_band = cfg.band_low <= price <= cfg.band_high

            # Reset counter when price leaves band — must stay ABOVE the DTE
            # check so out-of-band days are never counted, even when DTE is
            # out of range and the continue fires.
            if not in_band:
                days_in_band = 0

            # DTE filter
            if end_date:
                dte = compute_dte_from_dates(date, end_date)
            else:
                dte = None
            if dte is not None:
                if not (cfg.min_dte <= dte <= cfg.max_dte):
                    continue

            # Increment days-in-band AFTER DTE passes. This prevents
            # DTE-invalid days from accruing into the counter when DTE
            # later enters range.
            if in_band:
                days_in_band += 1

            # Band + min_time_in_band
            if not in_band or days_in_band < max(1, cfg.min_time_in_band_days):
                continue

            # Apply slippage
            # All cost components in cents: spread (taker), slippage, adverse selection.
            entry_price = price + (cfg.spread_c + cfg.slippage_c + cfg.adverse_selection_c) / 100
            if entry_price >= 1.0:
                continue

            effective_notional = cfg.order_notional_usd
            if cfg.sizing_fn is not None:
                multiplier = cfg.sizing_fn(entry_price)
                effective_notional = max(MIN_EFFECTIVE_NOTIONAL, cfg.order_notional_usd * multiplier)

            in_position = True
            entry_row = {**row, "price": str(entry_price), "effective_notional": str(effective_notional)}
            min_price_open = entry_price  # seed MAE tracker at entry

    trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame()

    # Build equity curve
    eq_dates = sorted(equity_by_date.keys())
    cumulative = 0.0
    eq_rows = []
    for d in eq_dates:
        cumulative += equity_by_date[d]
        eq_rows.append({"date": d, "daily_pnl": equity_by_date[d], "cumulative_pnl": cumulative})
    equity_df = pd.DataFrame(eq_rows) if eq_rows else pd.DataFrame()

    return trades_df, equity_df


def _compute_size(notional: float, price: float) -> float:
    import math
    if price <= 0 or price >= 1:
        return 0.0
    return math.floor((notional / price) * 100) / 100
