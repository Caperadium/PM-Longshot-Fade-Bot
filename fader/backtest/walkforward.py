"""backtest/walkforward.py

Descriptive (model-free) walk-forward stability report.

Splits historical data into N contiguous calendar windows and runs the
exact same strategy in every window.  No parameter is optimised or carried
across windows — the report answers "was performance consistent or
concentrated in one lucky stretch?"

Option A only (per review decision).  An expanding-window cross-validation
variant may be added later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, run_backtest
from backtest.metrics import compute_all_metrics

_DAYS_PER_MONTH = 30.44  # average Gregorian month


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class WalkforwardWindow:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_trades: int
    metrics: Dict


def partition_calendar_windows(
    store_or_df,
    n_windows: int = 4,
    window_months: Optional[float] = None,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    """Split store data into N contiguous calendar windows by observation date.

    Each window covers an equal-width span of calendar time.  The last
    window is inclusive of ``t_max``.

    Returns a list of ``(start, end_inclusive, df_slice)``, chronological.
    """
    from backtest.historical import ContractPriceStore

    if isinstance(store_or_df, ContractPriceStore):
        df = store_or_df.snapshot()
    elif isinstance(store_or_df, pd.DataFrame):
        df = store_or_df.copy()
    else:
        raise TypeError(f"Expected ContractPriceStore or DataFrame, got {type(store_or_df)}")

    if df.empty:
        return []

    dates = pd.to_datetime(df["date"], utc=True, errors="coerce")
    d_min, d_max = dates.min(), dates.max()
    if pd.isna(d_min) or pd.isna(d_max) or d_min >= d_max:
        return []

    span = d_max - d_min

    if window_months is not None:
        width = pd.Timedelta(days=window_months * _DAYS_PER_MONTH)
        n = max(1, int(math.ceil(span / width))) if span > pd.Timedelta(0) else 1
    else:
        n = max(1, int(n_windows))
        width = span / n if span > pd.Timedelta(0) else pd.Timedelta(days=1)

    windows = []
    for i in range(n):
        start = d_min + i * width
        if i == n - 1:
            end = d_max
            mask = (dates >= start) & (dates <= end)
        else:
            end = d_min + (i + 1) * width
            mask = (dates >= start) & (dates < end)
        windows.append((start, end, df[mask.values].copy()))
    return windows


def window_summary(
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]],
    cfg: BacktestConfig,
    slugs: Optional[List[str]] = None,
    n_bootstrap: int = 4000,
    initial_capital: float = 500.0,
) -> Tuple[List[WalkforwardWindow], Dict]:
    """Run backtest in each window and return per-window metrics + stability.

    Returns ``(per_window, stability)``.
    """
    per_window: List[WalkforwardWindow] = []
    for start, end, df_slice in windows:
        label = f"{start:%Y-%m-%d}  ..  {end:%Y-%m-%d}"
        if df_slice.empty:
            per_window.append(WalkforwardWindow(
                label=label, start=start, end=end,
                n_trades=0, metrics={},
            ))
            continue
        try:
            trades_df, _ = run_backtest(df_slice, cfg, slugs=slugs)
        except Exception:
            per_window.append(WalkforwardWindow(
                label=label, start=start, end=end,
                n_trades=0, metrics={},
            ))
            continue
        m = compute_all_metrics(
            trades_df,
            n_bootstrap=n_bootstrap,
            initial_capital=initial_capital,
        ) if not trades_df.empty else {}
        per_window.append(WalkforwardWindow(
            label=label, start=start, end=end,
            n_trades=m.get("n_trades", 0), metrics=m,
        ))

    stability = _stability_report(per_window)
    return per_window, stability


# ---------------------------------------------------------------------------
# Stability report (internal)
# ---------------------------------------------------------------------------


def _stability_report(per_window: List[WalkforwardWindow]) -> Dict:
    """Cross-window consistency statistics.

    Uses Sortino (not Sharpe) for cross-window risk-adjusted comparison.
    Sortino only penalises downside deviation, making it more honest for
    negatively-skewed strategies like the NO-fader.

    Returns a dict suitable for display in the dashboard.
    """
    sortinos = [
        w.metrics["sortino"]
        for w in per_window
        if w.n_trades > 0 and "sortino" in w.metrics and w.metrics.get("sortino", 0) != 0.0
    ]
    pnls = [
        w.metrics["total_pnl"]
        for w in per_window
        if w.n_trades > 0 and "total_pnl" in w.metrics
    ]
    hit_rates = [
        w.metrics["hit_rate"]
        for w in per_window
        if w.n_trades > 0 and "hit_rate" in w.metrics
    ]

    n_windows = len(per_window)
    n_nonempty = len(sortinos)

    if n_nonempty == 0:
        return {
            "n_windows": n_windows,
            "n_nonempty": 0,
            "sortino_cv": None,
            "sortino_min": None,
            "sortino_max": None,
            "sortino_mean": None,
            "pnl_concentration": None,
            "pnl_total": 0.0,
            "hit_rate_min": None,
            "hit_rate_max": None,
            "stability_grade": "no_data",
        }

    sortino_cv = float(np.std(sortinos, ddof=1) / abs(np.mean(sortinos))) if n_nonempty > 1 and np.mean(sortinos) != 0 else 0.0
    total_pnl = sum(pnls) if pnls else 0.0
    pnl_concentration = float(max(pnls) / total_pnl) if total_pnl > 0 else (1.0 if pnls else 0.0)

    # Stability grade heuristic — Sortino thresholds are wider than Sharpe
    # because the denominator (semi-deviation) already filters upside noise
    # and is inherently more volatile across windows.
    if sortino_cv < 0.5 and pnl_concentration < 0.5:
        grade = "stable"
    elif sortino_cv < 1.0 and pnl_concentration < 0.7:
        grade = "moderate"
    elif n_nonempty < 2:
        grade = "too_few_windows"
    else:
        grade = "unstable"

    return {
        "n_windows": n_windows,
        "n_nonempty": n_nonempty,
        "sortino_cv": sortino_cv,
        "sortino_min": float(min(sortinos)) if sortinos else None,
        "sortino_max": float(max(sortinos)) if sortinos else None,
        "sortino_mean": float(np.mean(sortinos)) if sortinos else None,
        "pnl_concentration": pnl_concentration,
        "pnl_total": total_pnl,
        "hit_rate_min": float(min(hit_rates)) if hit_rates else None,
        "hit_rate_max": float(max(hit_rates)) if hit_rates else None,
        "stability_grade": grade,
    }
