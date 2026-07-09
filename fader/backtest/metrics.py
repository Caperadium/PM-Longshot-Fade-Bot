"""backtest/metrics.py

Backtest performance metrics with bootstrapped CIs.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# BTC markets trade 24/7, so a daily return series annualizes by sqrt(365),
# not the 252 trading-day convention used for equities.
PERIODS_PER_YEAR = 365
METRICS_VERSION = 5  # v5: removed Sharpe (misleading for neg-skew strategies), promoted Sortino + Calmar; v4: +block bootstrap, tail metrics

# Known backtest-vs-live filter gaps: historical Polymarket data cannot
# reconstruct these three live-engine filters (Phase 4, strategy/filters.py
# skipped-filter union). Keyed by the same filter-name strings the shared
# filter core (strategy/filters.py) records in FilterResult.skipped /
# EntrySnapshot None-fields, so callers can pass exactly what the backtest
# engine actually skipped instead of relying on this fixed assumption.
_UNIVERSE_DISCREPANCY_DEFS: Dict[str, Dict] = {
    "min_24h_volume": {
        "filter": "min_24h_volume",
        "applied_in_live": True,
        "applied_in_backtest": False,
        "reason": (
            "Polymarket Gamma /markets returns current volume24hr only "
            "— no historical volume time series. Using cumulative "
            "volume would introduce look-ahead bias (a market that "
            "traded heavily post-entry would appear liquid at entry)."
        ),
        "bias_direction": "optimistic (wider universe, same fill model)",
    },
    "min_total_volume": {
        "filter": "min_total_volume",
        "applied_in_live": True,
        "applied_in_backtest": False,
        "reason": "Same as min_24h_volume — cumulative volumeNum is current-value only.",
        "bias_direction": "optimistic (wider universe, same fill model)",
    },
    "min_book_depth": {
        "filter": "min_book_depth",
        "applied_in_live": True,
        "applied_in_backtest": False,
        "reason": "Historical order-book snapshots are not available from CLOB.",
        "bias_direction": "optimistic (may include thin-book markets)",
    },
}
# Historical default (pre-Phase-4): always exactly these three, regardless
# of what any given run actually skipped. Preserved as the fallback when
# compute_all_metrics() is called without an explicit skipped_filters
# argument, so existing callers/goldens are byte-identical.
_DEFAULT_UNIVERSE_DISCREPANCY_KEYS: Tuple[str, ...] = (
    "min_24h_volume", "min_total_volume", "min_book_depth",
)


def _build_universe_discrepancies(skipped_filters: Optional[Tuple[str, ...]]) -> List[Dict]:
    """Build the universe_discrepancies list. `skipped_filters=None` (the
    default) reproduces the historical fixed 3-item list byte-for-byte.
    When provided (e.g. by backtest/engine.py's run metadata), only
    filters BOTH known to this table AND actually skipped in that run are
    listed — unknown skipped-filter names (e.g. "dte", "stale_data", which
    are not documented universe discrepancies) are silently ignored."""
    keys = _DEFAULT_UNIVERSE_DISCREPANCY_KEYS if skipped_filters is None else skipped_filters
    return [_UNIVERSE_DISCREPANCY_DEFS[k] for k in keys if k in _UNIVERSE_DISCREPANCY_DEFS]




def sortino_ratio(returns: np.ndarray, periods_per_year: float = PERIODS_PER_YEAR, target: float = 0.0) -> float:
    """Annualized Sortino ratio (downside deviation only).

    Uses the standard definition: sqrt(mean(min(r - target, 0)^2)) over ALL
    periods — not just the subset below target. This accounts for downside
    frequency as well as magnitude, unlike the subset-std variant.
    For negatively-skewed strategies this gives a more honest risk-read.
    """
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    # semi-deviation over full series (upside periods contribute 0)
    below = np.minimum(returns - target, 0.0)
    semi_var = np.mean(below ** 2)
    if semi_var == 0:
        return 0.0
    semi_std = math.sqrt(semi_var)
    return float((mu - target) / semi_std * math.sqrt(periods_per_year))


def calmar_ratio(annualized_return_pct: float, max_dd_pct: float) -> float:
    """Calmar ratio = annualized fractional return / max drawdown (fraction).

    Both arguments must be unitless (e.g. 0.15 for 15 %).
    Higher → better risk-adjusted return. Negative return → negative Calmar.
    """
    if max_dd_pct <= 0:
        return 0.0
    return float(annualized_return_pct / max_dd_pct)


def max_drawdown_pct(equity: np.ndarray) -> float:
    """Max drawdown as a fraction of peak equity (0.0 = no drawdown, 1.0 = ruin)."""
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    if np.all(peak <= 0):
        # All peaks are ≤ 0: either flat-zero (no drawdown) or all-negative (ruin).
        return 1.0 if np.any(equity < 0) else 0.0
    dd = peak - equity
    # Fractional drawdown at each point.  peak > 0 for at least the first
    # non-negative point, so division is safe where it matters.
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = dd / peak
    frac[peak <= 0] = 0.0
    return float(np.max(frac))


def daily_pnl_series(trades: pd.DataFrame) -> np.ndarray:
    """Collapse trades into one PnL value per calendar day (zero-filled).

    PnL is realized on the resolution day (``exit_date``); same-day bets — which
    are driven by the same BTC move and thus correlated — are summed into a
    single daily portfolio P&L. Idle days between first and last activity are
    included as 0 so the time axis (and hence volatility) is honest.
    """
    if trades.empty or "exit_date" not in trades.columns or "realized_pnl" not in trades.columns:
        return np.array([])
    d = trades.dropna(subset=["exit_date"]).copy()
    d["exit_date"] = pd.to_datetime(d["exit_date"], errors="coerce").dt.tz_localize(None)
    d = d.dropna(subset=["exit_date"])
    if d.empty:
        return np.array([])
    daily = d.groupby(d["exit_date"].dt.normalize())["realized_pnl"].sum()
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full, fill_value=0.0).values


def max_drawdown(equity: np.ndarray) -> float:
    """Max peak-to-trough drawdown of an equity curve, in dollars (positive)."""
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def hit_rate(pnls: np.ndarray) -> float:
    if len(pnls) == 0:
        return 0.0
    return float(np.mean(pnls > 0))


def expectancy(pnls: np.ndarray) -> float:
    return float(np.mean(pnls)) if len(pnls) > 0 else 0.0


def _auto_block_len(values: np.ndarray) -> int:
    """Automatic block-length selection via Patton, Politis & White (2009).

    Finds the first lag where the sample autocorrelation drops below
    ``2 / sqrt(n)`` and uses that as the block length.  Falls back to
    ``max(1, floor(sqrt(n)))`` when the ACF never drops (highly persistent
    returns).
    """
    n = len(values)
    if n < 4:
        return max(1, n // 2)
    # Compute ACF up to n//4 lags via pandas (handles edge cases cleanly).
    s = pd.Series(values)
    acf_vals = [s.autocorr(lag=lag) for lag in range(1, min(n // 4, 50) + 1)]
    threshold = 2.0 / np.sqrt(n)
    for lag, acf in enumerate(acf_vals, start=1):
        if acf <= threshold:
            return max(1, lag)
    # Fallback when the series is highly persistent.
    return max(1, int(np.sqrt(n)))


def block_bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n: int = 10000,
    ci: float = 0.95,
    block_len: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Moving-block bootstrap CI (Politis & Romano, 1994).

    Preserves within-series correlation structure — unlike the IID
    bootstrap, which assumes independent observations and produces
    overconfident (too-narrow) intervals when daily returns exhibit
    volatility clustering or serial dependence.

    ``block_len`` is chosen automatically via the PPW (2009) rule when
    omitted.  Returns ``(lower, upper)`` at the given CI level.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_obs = len(values)
    if n_obs < 2:
        v = stat_fn(values)
        return v, v
    if block_len is None:
        block_len = _auto_block_len(values)
    block_len = min(block_len, n_obs - 1)
    block_len = max(1, block_len)

    n_blocks = int(np.ceil(n_obs / block_len))
    stats = np.empty(n)
    for i in range(n):
        # Sample blocks with replacement.
        idx = rng.integers(0, n_obs - block_len + 1, size=n_blocks)
        blocks = [values[j : j + block_len] for j in idx]
        sample = np.concatenate(blocks)[:n_obs]
        stats[i] = stat_fn(sample)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(stats, alpha * 100))
    hi = float(np.percentile(stats, (1 - alpha) * 100))
    return lo, hi


def bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n: int = 10000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """IID bootstrap CI (legacy). Prefer ``block_bootstrap_ci`` for
    financial time series — IID resampling assumes independent draws and
    underestimates uncertainty when returns are serially correlated.

    Kept as an internal reference; the public API uses block bootstrap.
    """
    if rng is None:
        rng = np.random.default_rng()
    if len(values) < 2:
        v = stat_fn(values)
        return v, v
    stats = np.array([
        stat_fn(rng.choice(values, size=len(values), replace=True))
        for _ in range(n)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(stats, alpha * 100))
    hi = float(np.percentile(stats, (1 - alpha) * 100))
    return lo, hi


def per_market_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group realized PnL by slug.
    df must have columns: slug, realized_pnl
    """
    return (
        df.groupby("slug")["realized_pnl"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "total_pnl", "count": "n_trades", "mean": "avg_pnl"})
        .reset_index()
        .sort_values("total_pnl", ascending=False)
    )


def compute_all_metrics(
    trades: pd.DataFrame,
    n_bootstrap: int = 10000,
    initial_capital: float = 500.0,
    skipped_filters: Optional[Tuple[str, ...]] = None,
) -> Dict:
    """
    Compute full metric suite from a trades DataFrame.

    trades must have columns:
      - realized_pnl: float
      - entry_price: float
      - slug: str

    ``initial_capital`` is added to the equity curve (cumulative PnL) before
    computing drawdown percentages and Calmar.  It does NOT affect Sortino
    or skewness (scale-invariant).  Default: $500 — roughly 20 concurrent
    $10 notional positions at 50 % margin.

    Sharpe has been removed — it is unreliable for negatively-skewed
    strategies like the NO-fader where gains are frequent but small and
    losses are rare but large.  Sortino and Calmar replace it.

    ``skipped_filters``: optional iterable of filter-name strings that were
    NOT evaluated in the run that produced ``trades`` (Phase 4:
    strategy/filters.py's FilterResult.skipped union, surfaced by
    backtest/engine.py's run metadata). When omitted (None, the default),
    the historical fixed 3-item universe_discrepancies list is returned
    unchanged. When provided, only the subset that is both a known
    live-vs-backtest gap and actually skipped is listed.

    Returns a dict with:
      total_pnl, hit_rate, avg_win, avg_loss, expectancy,
      sortino, calmar, max_drawdown, max_drawdown_pct, daily_skew,
      daily_kurtosis, daily_var_95, daily_var_99, daily_cvar_95,
      daily_normality_p, n_daily, n_active_days,
      per_market, pnl_ci_95,
      initial_capital, metrics_version, elapsed_ms, universe_discrepancies
    """
    if trades.empty or "realized_pnl" not in trades.columns:
        return {}

    t0 = time.perf_counter()

    # Use only trades with a valid exit_date for all metrics so that trade-level
    # stats (total_pnl, hit_rate, etc.) and daily-level stats (Sortino, drawdown)
    # report on the same universe.
    if "exit_date" in trades.columns:
        settled = trades.dropna(subset=["exit_date"])
    else:
        settled = trades

    pnls = settled["realized_pnl"].dropna().values
    daily = daily_pnl_series(settled)
    have_daily = len(daily) >= 2
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    rng = np.random.default_rng(42)
    capital = max(initial_capital, 1.0)  # $1 floor avoids degenerate Calmar / DD%

    n_daily = len(daily)
    n_active = int(np.sum(daily != 0))

    if have_daily:
        # Equity = starting bankroll + cumulative daily PnL.
        # DD $ is invariant to the bankroll shift, but DD % and Calmar
        # become realistic because peak equity includes the capital base.
        equity = capital + np.cumsum(np.concatenate([[0], daily]))
        pnl_ci_lo, pnl_ci_hi = block_bootstrap_ci(
            daily, lambda x: float(np.sum(x)), n=n_bootstrap, rng=rng
        )
        sortino_val = sortino_ratio(daily, periods_per_year=PERIODS_PER_YEAR)
        dd_val = max_drawdown(equity)
        dd_pct_val = max_drawdown_pct(equity)
        # Calmar = CAGR / max drawdown fraction, both computed on the
        # same equity curve (capital + cumulative daily PnL). Suppressed
        # when n_daily < 60 — CAGR is unreliable on short samples (e.g.
        # 5 % return over 30 d compounds to ~80 % annualised).
        if n_daily >= 60 and capital > 0 and len(equity) > 1 and equity[0] > 0:
            final = equity[-1]
            n_years = len(daily) / PERIODS_PER_YEAR
            if n_years > 0 and final > 0:
                cagr = (final / equity[0]) ** (1.0 / n_years) - 1.0
            else:
                cagr = 0.0
        else:
            cagr = 0.0
        calmar_val = calmar_ratio(cagr, dd_pct_val) if n_daily >= 60 else None
        pnl_ci_vals = (pnl_ci_lo, pnl_ci_hi)
        # Skewness of daily returns (negative = left tail, insurance-like risk)
        daily_skew = float(pd.Series(daily).skew())
        # Excess kurtosis (> 0 = fatter tails than normal; > 2 = very heavy)
        daily_kurtosis = float(pd.Series(daily).kurtosis())
        # Historical VaR (5th / 1st percentile of daily PnL)
        daily_var_95 = float(np.percentile(daily, 5))
        daily_var_99 = float(np.percentile(daily, 1))
        # CVaR 95 % — expected loss GIVEN the loss exceeds VaR 95
        tail = daily[daily <= daily_var_95]
        daily_cvar_95 = float(tail.mean()) if len(tail) > 0 else daily_var_95
        # D'Agostino-Pearson omnibus normality test (combines skew + kurtosis).
        # p < 0.05 → reject normality → Sortino has more honest risk-read.
        try:
            _, normality_p = sp_stats.normaltest(daily)
            daily_normality_p = float(normality_p)
        except Exception:
            daily_normality_p = float("nan")
    else:
        sortino_val = 0.0
        dd_val = 0.0
        dd_pct_val = 0.0
        calmar_val = 0.0
        pnl_ci_vals = (0.0, 0.0)
        daily_skew = 0.0
        daily_kurtosis = 0.0
        daily_var_95 = 0.0
        daily_var_99 = 0.0
        daily_cvar_95 = 0.0
        daily_normality_p = float("nan")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "total_pnl": float(np.sum(pnls)),
        "n_trades": int(len(pnls)),
        "hit_rate": hit_rate(pnls),
        "avg_win": float(np.mean(wins)) if len(wins) else 0.0,
        "avg_loss": float(np.mean(losses)) if len(losses) else 0.0,
        "expectancy": expectancy(pnls),
        "sortino": sortino_val,
        "calmar": calmar_val,
        "max_drawdown": dd_val,
        "max_drawdown_pct": dd_pct_val,
        "daily_skew": daily_skew,
        "daily_kurtosis": daily_kurtosis,
        "daily_var_95": daily_var_95,
        "daily_var_99": daily_var_99,
        "daily_cvar_95": daily_cvar_95,
        "daily_normality_p": daily_normality_p,
        "n_daily": n_daily,
        "n_active_days": n_active,
        "pnl_ci_95": pnl_ci_vals,
        "per_market": per_market_attribution(settled).to_dict("records"),
        "initial_capital": capital,
        "metrics_version": METRICS_VERSION,
        "elapsed_ms": elapsed_ms,
        "universe_discrepancies": _build_universe_discrepancies(skipped_filters),
    }
