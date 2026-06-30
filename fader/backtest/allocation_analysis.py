"""backtest/allocation_analysis.py

Probability-band allocation PnL analysis.

Sweeps a tilt parameter a across [-1, +1], running the backtest with
different capital-allocation weighting toward high-prob vs low-prob
contracts. Reports per-scheme metrics, monotonicity tests, concavity
analysis, and walk-forward validation.

Run as: python -m fader.backtest.allocation_analysis
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from backtest.engine import BacktestConfig, run_backtest
from backtest.historical import ContractPriceStore
from backtest.metrics import (
    PERIODS_PER_YEAR,
    block_bootstrap_ci,
    calmar_ratio,
    compute_all_metrics,
    daily_pnl_series,
    max_drawdown_pct,
    sortino_ratio,
)
from backtest.walkforward import partition_calendar_windows
from execution.sizing import make_sizing_fn

logger = logging.getLogger(__name__)


def _make_sized_fn(alpha: float, band_low: float, band_high: float,
                   norm_factor: float = 1.0) -> Callable[[float], float]:
    """Wrap make_sizing_fn with optional norm_factor divisor (analysis-only)."""
    base_fn = make_sizing_fn(alpha, band_low, band_high)
    if norm_factor == 1.0:
        return base_fn
    return lambda p: base_fn(p) / norm_factor


# ---------------------------------------------------------------------------
# Normalization factor (analysis-only)
# ---------------------------------------------------------------------------


def compute_normalization_factor(
    entry_prices: List[float],
    alphas: List[float],
    band_low: float,
    band_high: float,
) -> float:
    """Global normalization factor so mean(f(p, a)) ≈ 1 across all entries and a.

    Computes raw multipliers for every (price, a) pair pooled together,
    then returns the grand mean. Applied once so total deployed capital is
    comparable across a schemes regardless of price clustering.
    """
    if not entry_prices or not alphas:
        return 1.0

    all_raw = []
    for alpha in alphas:
        fn = make_sizing_fn(alpha, band_low, band_high)
        for p in entry_prices:
            all_raw.append(fn(p))

    mean_raw = float(np.mean(all_raw))
    if mean_raw <= 0:
        return 1.0
    return mean_raw


def extract_entry_prices(
    store_or_df, band_low: float, band_high: float
) -> List[float]:
    """Extract all entry prices that would pass the band filter.

    Runs a lightweight scan: for each (slug, token_id) time series,
    collect the first in-band price after min_time_in_band satisfied.
    Mirrors the engine's entry logic (without cost adjustments).
    """
    from backtest.historical import ContractPriceStore

    if isinstance(store_or_df, ContractPriceStore):
        df = store_or_df.snapshot()
    elif isinstance(store_or_df, pd.DataFrame):
        df = store_or_df.copy()
    else:
        return []

    if df.empty or "price" not in df.columns:
        return []

    df = df.copy()
    df["price_f"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price_f"])
    df = df[(df["price_f"] > 0) & (df["price_f"] < 1)]

    prices = []
    grouped = df.groupby(["slug", "token_id"]) if "slug" in df.columns else []
    if not callable(getattr(grouped, "__iter__", None)):
        # Fallback if groupby columns missing
        return df["price_f"].tolist()

    for (slug, token_id), group in grouped:
        group = group.sort_values("date")
        days_in_band = 0
        in_position = False
        min_days = 1  # mirror engine default
        for _, row in group.iterrows():
            p = float(row["price_f"])
            if p <= 0:
                days_in_band = 0
                continue
            in_band = band_low <= p <= band_high
            if not in_band:
                days_in_band = 0
            else:
                days_in_band += 1
            if in_position:
                # once in position, don't count further entries
                continue
            if in_band and days_in_band >= min_days:
                prices.append(p)
                in_position = True

    return prices


# ---------------------------------------------------------------------------
# Per-bin breakdown
# ---------------------------------------------------------------------------


@dataclass
class BinStats:
    bin_label: str
    price_low: float
    price_high: float
    n_trades: int
    total_pnl: float
    avg_pnl: float
    hit_rate: float
    hit_rate_ci_lo: float
    hit_rate_ci_hi: float
    max_loss: float


def per_bin_breakdown(
    trades_df: pd.DataFrame,
    n_bins: int = 5,
    ci: float = 0.95,
) -> List[BinStats]:
    """Quantile-based bin breakdown of trades by entry price.

    Uses equal-trade-count bins so sparse price regions don't get
    misleadingly small samples.
    """
    if trades_df.empty or "entry_price" not in trades_df.columns:
        return []

    df = trades_df.dropna(subset=["entry_price", "realized_pnl"]).copy()
    if df.empty:
        return []

    df["price_bin"] = pd.qcut(df["entry_price"], n_bins, duplicates="drop")
    actual_bins = df["price_bin"].cat.categories if hasattr(df["price_bin"].cat, "categories") else []
    if len(actual_bins) == 0:
        return []

    rng = np.random.default_rng(42)
    results: List[BinStats] = []

    for interval in actual_bins:
        mask = df["price_bin"] == interval
        bin_trades = df[mask]
        n = len(bin_trades)
        if n == 0:
            continue

        pnls = bin_trades["realized_pnl"].values
        wins = pnls[pnls > 0]
        hit = float(np.mean(pnls > 0))

        # Bootstrap CI for hit rate
        n_bootstrap = min(10000, max(100, n * 100))
        hit_bool = (pnls > 0).astype(float)
        hit_lo, hit_hi = block_bootstrap_ci(
            hit_bool,
            lambda x: float(np.mean(x)),
            n=n_bootstrap,
            ci=ci,
            rng=rng,
        )

        results.append(BinStats(
            bin_label=str(interval),
            price_low=float(interval.left),
            price_high=float(interval.right),
            n_trades=n,
            total_pnl=float(np.sum(pnls)),
            avg_pnl=float(np.mean(pnls)),
            hit_rate=hit,
            hit_rate_ci_lo=hit_lo,
            hit_rate_ci_hi=hit_hi,
            max_loss=float(np.min(pnls)) if len(pnls) else 0.0,
        ))

    return results


# ---------------------------------------------------------------------------
# Paired bootstrap CI for scheme comparisons
# ---------------------------------------------------------------------------


def paired_pnl_ci(
    trades_baseline: pd.DataFrame,
    trades_scheme: pd.DataFrame,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
) -> Tuple[float, float, float, float]:
    """Paired block-bootstrap CI for total PnL difference.

    Resamples the same calendar days for both schemes, computes total PnL
    in each bootstrap iteration, and returns CIs for the difference.

    Returns (diff_lo, diff_hi, baseline_lo, baseline_hi).
    """
    rng = np.random.default_rng(42)

    daily_base = daily_pnl_series(trades_baseline)
    daily_scheme = daily_pnl_series(trades_scheme)

    # Align to common date range
    if len(daily_base) == 0 or len(daily_scheme) == 0:
        return (0.0, 0.0, 0.0, 0.0)

    n = max(len(daily_base), len(daily_scheme))
    if len(daily_base) < n:
        daily_base = np.pad(daily_base, (0, n - len(daily_base)))
    if len(daily_scheme) < n:
        daily_scheme = np.pad(daily_scheme, (0, n - len(daily_scheme)))

    diffs = np.empty(n_bootstrap)
    base_totals = np.empty(n_bootstrap)

    # Same-block resample for both series simultaneously
    block_len = max(1, int(np.sqrt(n)))
    n_blocks = int(np.ceil(n / block_len))

    for i in range(n_bootstrap):
        idx = rng.integers(0, n - block_len + 1, size=n_blocks)
        blocks_base = [daily_base[j : j + block_len] for j in idx]
        blocks_scheme = [daily_scheme[j : j + block_len] for j in idx]
        sample_base = np.concatenate(blocks_base)[:n]
        sample_scheme = np.concatenate(blocks_scheme)[:n]
        diffs[i] = np.sum(sample_scheme) - np.sum(sample_base)
        base_totals[i] = np.sum(sample_base)

    alpha = (1 - ci) / 2
    diff_lo = float(np.percentile(diffs, alpha * 100))
    diff_hi = float(np.percentile(diffs, (1 - alpha) * 100))
    base_lo = float(np.percentile(base_totals, alpha * 100))
    base_hi = float(np.percentile(base_totals, (1 - alpha) * 100))

    return diff_lo, diff_hi, base_lo, base_hi


# ---------------------------------------------------------------------------
# Single-scheme runner
# ---------------------------------------------------------------------------


def _run_scheme(
    df: pd.DataFrame,
    cfg: BacktestConfig,
    sizing_fn: Optional[Callable[[float], float]],
    alpha: float,
) -> Dict:
    """Run backtest for one a and compute metrics. Handles empty results."""
    scheme_cfg = BacktestConfig(
        band_low=cfg.band_low,
        band_high=cfg.band_high,
        min_dte=cfg.min_dte,
        max_dte=cfg.max_dte,
        min_time_in_band_days=cfg.min_time_in_band_days,
        order_notional_usd=cfg.order_notional_usd,
        spread_c=cfg.spread_c,
        slippage_c=cfg.slippage_c,
        adverse_selection_c=cfg.adverse_selection_c,
        n_bootstrap=cfg.n_bootstrap,
        sizing_fn=sizing_fn,
    )

    try:
        trades_df, _ = run_backtest(df, scheme_cfg)
    except Exception as e:
        logger.warning(f"a={alpha:+.1f}: backtest failed: {e}")
        trades_df = pd.DataFrame()

    if trades_df.empty:
        return {
            "alpha": alpha,
            "n_trades": 0,
            "total_pnl": 0.0,
            "sortino": 0.0,
            "calmar": None,
            "max_drawdown_pct": 0.0,
            "hit_rate": 0.0,
            "expectancy": 0.0,
            "profit_factor": float("inf"),
            "cvar_95": 0.0,
            "worst_trade": 0.0,
            "worst_day": 0.0,
            "mean_concurrent": 0.0,
            "max_concurrent": 0,
            "daily_skew": 0.0,
            "daily_kurtosis": 0.0,
            "pnl_ci_95": (0.0, 0.0),
            "per_bin": [],
            "empty": True,
        }

    metrics = compute_all_metrics(trades_df, n_bootstrap=cfg.n_bootstrap)

    # Additional metrics not in compute_all_metrics
    pnls = trades_df["realized_pnl"].dropna().values
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    profit_factor = abs(float(np.sum(wins)) / float(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else float("inf")

    if "exit_date" in trades_df.columns:
        daily = daily_pnl_series(trades_df)
        worst_day = float(np.min(daily)) if len(daily) > 0 else 0.0
    else:
        worst_day = 0.0

    # Mean/max concurrent positions
    if "entry_date" in trades_df.columns and "exit_date" in trades_df.columns:
        concurrent = _concurrent_positions(trades_df)
        mean_concurrent = float(np.mean(concurrent)) if len(concurrent) > 0 else 0.0
        max_concurrent = int(np.max(concurrent)) if len(concurrent) > 0 else 0
    else:
        mean_concurrent = 0.0
        max_concurrent = 0

    return {
        "alpha": alpha,
        "n_trades": int(len(pnls)),
        "total_pnl": float(np.sum(pnls)),
        "sortino": metrics.get("sortino", 0.0),
        "calmar": metrics.get("calmar"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
        "hit_rate": metrics.get("hit_rate", 0.0),
        "expectancy": metrics.get("expectancy", 0.0),
        "profit_factor": profit_factor,
        "cvar_95": metrics.get("daily_cvar_95", 0.0),
        "worst_trade": float(np.min(pnls)) if len(pnls) > 0 else 0.0,
        "worst_day": worst_day,
        "mean_concurrent": mean_concurrent,
        "max_concurrent": max_concurrent,
        "daily_skew": metrics.get("daily_skew", 0.0),
        "daily_kurtosis": metrics.get("daily_kurtosis", 0.0),
        "pnl_ci_95": metrics.get("pnl_ci_95", (0.0, 0.0)),
        "per_bin": per_bin_breakdown(trades_df),
        "empty": False,
    }


def _concurrent_positions(trades_df: pd.DataFrame) -> np.ndarray:
    """Count concurrent open positions over time from trade entry/exit dates."""
    events = []
    for _, t in trades_df.iterrows():
        entry = t.get("entry_date")
        exit_ = t.get("exit_date")
        if entry is None:
            continue
        events.append((pd.Timestamp(str(entry)), 1))
        if exit_ is not None and pd.notna(exit_):
            try:
                events.append((pd.Timestamp(str(exit_)), -1))
            except Exception:
                pass

    if not events:
        return np.array([0])

    events.sort(key=lambda x: x[0])
    concurrent = []
    count = 0
    for _, delta in events:
        count += delta
        concurrent.append(count)
    return np.array(concurrent)


# ---------------------------------------------------------------------------
# Monotonicity analysis
# ---------------------------------------------------------------------------


def spearman_rho_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    ci: float = 0.95,
) -> Tuple[float, float, float, float]:
    """Spearman ρ with two-tailed p-value and 95% CI via Fisher z-transform."""
    if len(x) < 4:
        return 0.0, 1.0, 0.0, 0.0
    try:
        rho, pval = sp_stats.spearmanr(x, y)
    except Exception:
        return 0.0, 1.0, 0.0, 0.0
    if not np.isfinite(rho):
        return 0.0, 1.0, 0.0, 0.0
    rho = float(rho)
    pval = float(pval)

    # Fisher z-transform for CI
    if abs(rho) >= 1.0:
        return rho, pval, rho, rho

    z = math.atanh(rho)
    se = 1.0 / math.sqrt(len(x) - 3)
    z_crit = sp_stats.norm.ppf(1 - (1 - ci) / 2)
    z_lo = z - z_crit * se
    z_hi = z + z_crit * se
    rho_lo = math.tanh(z_lo)
    rho_hi = math.tanh(z_hi)
    return rho, pval, rho_lo, rho_hi


def concavity_check(
    alphas: np.ndarray,
    values: np.ndarray,
) -> Dict:
    """Quadratic fit M ~ b₀ + b₁a + b₂a² and concavity assessment."""
    n = len(alphas)
    if n < 4:
        return {
            "quadratic_r2": None,
            "beta2": None,
            "beta2_p": None,
            "peak_alpha": None,
            "aic_quadratic": None,
            "aic_linear": None,
            "has_optimum": False,
        }

    # Linear fit
    X_lin = np.column_stack([np.ones(n), alphas])
    beta_lin, resid_lin, _, _ = np.linalg.lstsq(X_lin, values, rcond=None)
    rss_lin = float(np.sum(resid_lin ** 2)) if len(resid_lin) > 0 else float("inf")

    # Quadratic fit
    X_quad = np.column_stack([np.ones(n), alphas, alphas ** 2])
    beta_quad, resid_quad, _, _ = np.linalg.lstsq(X_quad, values, rcond=None)

    # Full covariance via normal equations for p(b₂)
    try:
        XtX_inv = np.linalg.inv(X_quad.T @ X_quad)
        sigma2 = float(np.sum(resid_quad ** 2)) / (n - 3) if n > 3 else float(np.sum(resid_quad ** 2))
        se_beta2 = math.sqrt(sigma2 * XtX_inv[2, 2])
        t_stat = beta_quad[2] / se_beta2 if se_beta2 > 0 else 0.0
        beta2_p = float(2 * sp_stats.t.sf(abs(t_stat), n - 3))
    except Exception:
        se_beta2 = float("inf")
        beta2_p = 1.0

    # R² for quadratic
    ss_tot = float(np.sum((values - np.mean(values)) ** 2))
    rss_quad = float(np.sum(resid_quad ** 2))
    r2 = 1.0 - rss_quad / ss_tot if ss_tot > 0 else 0.0

    # AIC (small-sample corrected for quadratic, corrected for linear)
    k_quad, k_lin = 3, 2
    aic_quad = n * math.log(rss_quad / n) + 2 * k_quad + (2 * k_quad * (k_quad + 1)) / max(1, n - k_quad - 1)
    aic_lin = n * math.log(rss_lin / n) + 2 * k_lin + (2 * k_lin * (k_lin + 1)) / max(1, n - k_lin - 1)

    # Optimum
    beta2 = float(beta_quad[2])
    beta1 = float(beta_quad[1])
    peak_alpha = -beta1 / (2 * beta2) if abs(beta2) > 1e-12 else float("inf")
    has_optimum = beta2 < 0 and beta2_p < 0.05 and -1.0 < peak_alpha < 1.0

    return {
        "quadratic_r2": r2,
        "beta2": beta2,
        "beta2_p": beta2_p,
        "peak_alpha": peak_alpha,
        "aic_quadratic": aic_quad,
        "aic_linear": aic_lin,
        "has_optimum": has_optimum,
    }


def monotonicity_report(
    results: List[Dict],
) -> Dict:
    """Compute monotonicity statistics across a sweep results.

    Only includes non-empty schemes.
    """
    nonempty = [r for r in results if not r.get("empty", False)]
    if len(nonempty) < 4:
        return {"error": "too_few_schemes", "n_nonempty": len(nonempty)}

    alphas = np.array([r["alpha"] for r in nonempty])

    primary_key = "sortino"
    secondary_keys = [
        "total_pnl", "calmar", "max_drawdown_pct", "hit_rate",
        "expectancy", "profit_factor", "cvar_95", "worst_trade",
        "worst_day", "daily_skew",
    ]

    # Filter: only keys present in results
    available_secondary = [k for k in secondary_keys if k in nonempty[0]]

    monotonic: Dict[str, Dict] = {}

    for key in [primary_key] + available_secondary:
        values = np.array([r.get(key, 0.0) or 0.0 for r in nonempty])
        if np.all(values == values[0]):
            rho, pval, rlo, rhi = 0.0, 1.0, 0.0, 0.0
        else:
            rho, pval, rlo, rhi = spearman_rho_with_ci(alphas, values)

        conc = concavity_check(alphas, values) if key == primary_key else None

        # Expected direction
        if key in ("sortino", "calmar", "hit_rate", "profit_factor"):
            expected_sign = +1  # higher a → more conservative → metric improves
        elif key in ("total_pnl", "max_drawdown_pct", "cvar_95",
                     "worst_trade", "worst_day"):
            expected_sign = -1  # higher a → less aggressive → return/risk decreases
        elif key == "daily_skew":
            expected_sign = +1  # higher a → less tail risk → skew improves (less negative)
        else:
            expected_sign = 0

        sign_match = (expected_sign == 0) or (rho * expected_sign >= 0) if rho != 0 else True

        entry: Dict = {
            "spearman_rho": rho,
            "spearman_p": pval,
            "spearman_ci_lo": rlo,
            "spearman_ci_hi": rhi,
            "expected_sign": expected_sign,
            "sign_match": sign_match,
            "metric_type": "primary" if key == primary_key else "secondary",
        }
        if conc is not None:
            entry["concavity"] = conc

        monotonic[key] = entry

    return monotonic


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------


def walkforward_validate(
    store_or_df,
    base_cfg: BacktestConfig,
    alpha_star: Optional[float],
    alphas: List[float],
    band_low: float,
    band_high: float,
    n_windows: int = 4,
) -> Dict:
    """Walk-forward validation of allocation tilt.

    Path A (alpha_star is not None): compare a=a* vs a=0 per window.
    Path B (alpha_star is None): compare a=+1 vs a=-1 per window.
    """
    from backtest.historical import ContractPriceStore

    if isinstance(store_or_df, ContractPriceStore):
        work_df = store_or_df.snapshot()
    elif isinstance(store_or_df, pd.DataFrame):
        work_df = store_or_df.copy()
    else:
        return {"error": "invalid_input"}

    windows = partition_calendar_windows(work_df, n_windows=n_windows)
    if not windows:
        return {"error": "no_windows", "n_windows": 0}

    norm_factor = compute_normalization_factor(
        extract_entry_prices(work_df, band_low, band_high),
        alphas, band_low, band_high,
    )

    per_window: List[Dict] = []

    for start, end, df_slice in windows:
        label = f"{start:%Y-%m-%d}..{end:%Y-%m-%d}"
        if df_slice.empty:
            per_window.append({"label": label, "error": "empty_slice"})
            continue

        if alpha_star is not None:
            # Path A: a* vs baseline
            fn_star = _make_sized_fn(alpha_star, band_low, band_high, norm_factor)
            fn_base = None  # baseline = fixed notional

            r_star = _run_scheme(df_slice, base_cfg, fn_star, alpha_star)
            r_base = _run_scheme(df_slice, base_cfg, fn_base, 0.0)

            sortino_diff = (r_star.get("sortino") or 0.0) - (r_base.get("sortino") or 0.0)
            per_window.append({
                "label": label,
                "sortino_star": r_star.get("sortino"),
                "sortino_baseline": r_base.get("sortino"),
                "sortino_diff": sortino_diff,
                "star_outperforms": sortino_diff > 0,
            })
        else:
            # Path B: extremes comparison
            fn_neg = _make_sized_fn(-1.0, band_low, band_high, norm_factor)
            fn_pos = _make_sized_fn(+1.0, band_low, band_high, norm_factor)

            r_neg = _run_scheme(df_slice, base_cfg, fn_neg, -1.0)
            r_pos = _run_scheme(df_slice, base_cfg, fn_pos, +1.0)

            sortino_diff = (r_pos.get("sortino") or 0.0) - (r_neg.get("sortino") or 0.0)
            per_window.append({
                "label": label,
                "sortino_pos": r_pos.get("sortino"),
                "sortino_neg": r_neg.get("sortino"),
                "sortino_diff": sortino_diff,
                "high_prob_wins": sortino_diff > 0,
            })

    # Summary
    n_valid = sum(1 for w in per_window if "sortino_diff" in w)
    if n_valid == 0:
        return {"error": "no_valid_windows", "n_windows": len(windows), "per_window": per_window}

    if alpha_star is not None:
        n_star_wins = sum(1 for w in per_window if w.get("star_outperforms", False))
        mean_diff = float(np.mean([w["sortino_diff"] for w in per_window if "sortino_diff" in w]))
        oos_valid = n_valid >= 2 and n_star_wins / n_valid >= 0.70
        return {
            "path": "A",
            "alpha_star": alpha_star,
            "n_windows": len(windows),
            "n_valid": n_valid,
            "n_star_wins": n_star_wins,
            "star_win_frac": n_star_wins / n_valid,
            "mean_sortino_diff": mean_diff,
            "oos_valid": oos_valid,
            "per_window": per_window,
        }
    else:
        n_high_wins = sum(1 for w in per_window if w.get("high_prob_wins", False))
        mean_diff = float(np.mean([w["sortino_diff"] for w in per_window if "sortino_diff" in w]))
        oos_valid = n_valid >= 2 and n_high_wins / n_valid >= 0.70
        return {
            "path": "B",
            "n_windows": len(windows),
            "n_valid": n_valid,
            "n_high_prob_wins": n_high_wins,
            "high_prob_win_frac": n_high_wins / n_valid,
            "mean_sortino_diff": mean_diff,
            "oos_valid": oos_valid,
            "per_window": per_window,
        }


# ---------------------------------------------------------------------------
# Main analysis runner
# ---------------------------------------------------------------------------


def run_allocation_analysis(
    store_or_df,
    cfg: Optional[BacktestConfig] = None,
    alphas: Optional[List[float]] = None,
    band_low: Optional[float] = None,
    band_high: Optional[float] = None,
    n_walkforward_windows: int = 4,
    n_bootstrap: int = 10000,
) -> Dict:
    """Run full allocation analysis across a sweep.

    Args:
        store_or_df: ContractPriceStore or DataFrame with historical data.
        cfg: Base backtest config (uses defaults if not provided).
        alphas: Tilt values to sweep. Default: [-1.0, -0.8, ..., +1.0].
        band_low: Override config band_low.
        band_high: Override config band_high.
        n_walkforward_windows: Number of calendar windows for OOS validation.
        n_bootstrap: Bootstrap iterations for CIs.

    Returns:
        Dict with keys: band, alphas, per_scheme, monotonicity,
                        optimal_alpha, walkforward, elapsed_s.
    """
    t0 = time.perf_counter()

    from backtest.historical import ContractPriceStore

    if cfg is None:
        cfg = BacktestConfig(
            band_low=0.70, band_high=0.95,
            order_notional_usd=10.0,
            spread_c=1.0, slippage_c=0.0, adverse_selection_c=0.0,
            n_bootstrap=n_bootstrap,
        )

    # Apply band overrides
    bl = band_low if band_low is not None else cfg.band_low
    bh = band_high if band_high is not None else cfg.band_high

    if alphas is None:
        alphas = [round(x, 1) for x in np.arange(-1.0, 1.05, 0.2).tolist()]

    # Snapshot data
    if isinstance(store_or_df, ContractPriceStore):
        df = store_or_df.snapshot()
    elif isinstance(store_or_df, pd.DataFrame):
        df = store_or_df.copy()
    else:
        return {"error": "invalid_input", "elapsed_s": time.perf_counter() - t0}

    # Compute global normalization
    entry_prices = extract_entry_prices(df, bl, bh)
    norm_factor = compute_normalization_factor(entry_prices, alphas, bl, bh)
    logger.info(
        f"Band [{bl:.2f}, {bh:.2f}]: {len(entry_prices)} entries, "
        f"norm_factor={norm_factor:.4f}"
    )

    # Baseline (a=0, no sizing_fn)
    baseline_result = _run_scheme(df, cfg, None, 0.0)
    baseline_total = baseline_result.get("total_pnl", 0.0)

    # Sweep a
    per_scheme: List[Dict] = []
    for alpha in alphas:
        fn = _make_sized_fn(alpha, bl, bh, norm_factor)
        result = _run_scheme(df, cfg, fn, alpha)

        # Paired PnL CI vs baseline
        if not result.get("empty") and not baseline_result.get("empty"):
            # Re-run baseline with matching config to get trades
            bl_trades, _ = run_backtest(df, BacktestConfig(
                band_low=bl, band_high=bh,
                min_dte=cfg.min_dte, max_dte=cfg.max_dte,
                min_time_in_band_days=cfg.min_time_in_band_days,
                order_notional_usd=cfg.order_notional_usd,
                spread_c=cfg.spread_c, slippage_c=cfg.slippage_c,
                adverse_selection_c=cfg.adverse_selection_c,
                n_bootstrap=cfg.n_bootstrap,
                sizing_fn=None,
            ))
            scheme_cfg2 = BacktestConfig(
                band_low=bl, band_high=bh,
                min_dte=cfg.min_dte, max_dte=cfg.max_dte,
                min_time_in_band_days=cfg.min_time_in_band_days,
                order_notional_usd=cfg.order_notional_usd,
                spread_c=cfg.spread_c, slippage_c=cfg.slippage_c,
                adverse_selection_c=cfg.adverse_selection_c,
                n_bootstrap=cfg.n_bootstrap,
                sizing_fn=fn,
            )
            scheme_trades, _ = run_backtest(df, scheme_cfg2)
            diff_lo, diff_hi, bl_lo, bl_hi = paired_pnl_ci(bl_trades, scheme_trades)
            result["pnl_vs_baseline_ci"] = (diff_lo, diff_hi)
            result["baseline_pnl_ci"] = (bl_lo, bl_hi)
            result["total_pnl"] = float(scheme_trades["realized_pnl"].sum()) if not scheme_trades.empty else 0.0
        else:
            result["pnl_vs_baseline_ci"] = (0.0, 0.0)
            result["baseline_pnl_ci"] = (0.0, 0.0)

        per_scheme.append(result)
        logger.info(
            f"a={alpha:+.1f}: n={result['n_trades']}, "
            f"PnL=${result['total_pnl']:.2f}, "
            f"Sortino={result['sortino']:.3f}"
        )

    # Monotonicity
    monotonic = monotonicity_report(per_scheme)

    # Optimal a (from primary metric concavity, if significant)
    sortino_conc = monotonic.get("sortino", {}).get("concavity")
    if sortino_conc and sortino_conc.get("has_optimum"):
        optimal_alpha = sortino_conc["peak_alpha"]
    else:
        optimal_alpha = None

    # Walk-forward validation
    wf = walkforward_validate(
        df, cfg, optimal_alpha, alphas, bl, bh,
        n_windows=n_walkforward_windows,
    )

    elapsed = time.perf_counter() - t0

    return {
        "band": [bl, bh],
        "alphas": alphas,
        "norm_factor": norm_factor,
        "n_entries": len(entry_prices),
        "baseline": baseline_result,
        "per_scheme": per_scheme,
        "monotonicity": monotonic,
        "optimal_alpha": optimal_alpha,
        "walkforward": wf,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Formatted report
# ---------------------------------------------------------------------------


def format_report(results: Dict) -> str:
    """Produce a formatted text report from analysis results."""
    lines = []
    band = results.get("band", [0.0, 1.0])
    alphas = results.get("alphas", [])

    lines.append("=" * 72)
    lines.append("PROBABILITY-BAND ALLOCATION PnL ANALYSIS")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Band: [{band[0]:.2f}, {band[1]:.2f}]")
    lines.append(f"Alphas: {alphas}")
    lines.append(f"Entries: {results.get('n_entries', 0)}")
    lines.append(f"Norm factor: {results.get('norm_factor', 1.0):.4f}")
    lines.append(f"Elapsed: {results.get('elapsed_s', 0):.1f}s")
    lines.append("")

    # --- Per-scheme table ---
    lines.append("-" * 72)
    lines.append("PER-SCHEME METRICS")
    lines.append("-" * 72)
    header = (
        f"{'a':>5s}  {'Trades':>6s}  {'Total PnL':>10s}  {'Sortino':>8s}  "
        f"{'Calmar':>8s}  {'Hit%':>6s}  {'ProfitF':>8s}  {'CVaR95':>8s}  "
        f"{'MaxDD%':>7s}  {'WorstDay':>9s}  {'WorstTrd':>9s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results.get("per_scheme", []):
        if r.get("empty"):
            lines.append(f"{r['alpha']:+5.1f}  {'(no trades)':>6s}")
            continue
        calmar_str = f"{r['calmar']:.3f}" if r.get("calmar") is not None else "N/A"
        lines.append(
            f"{r['alpha']:+5.1f}  {r['n_trades']:6d}  "
            f"${r['total_pnl']:9.2f}  {r['sortino']:8.3f}  "
            f"{calmar_str:>8s}  {r['hit_rate']:5.1%}  "
            f"{r['profit_factor']:8.2f}  "
            f"${r['cvar_95']:7.2f}  {r['max_drawdown_pct']:6.1%}  "
            f"${r['worst_day']:8.2f}  ${r['worst_trade']:8.2f}"
        )
    lines.append("")

    # --- PnL vs baseline ---
    lines.append("-" * 72)
    lines.append("TOTAL PNL VS BASELINE (a=0, fixed notional)")
    lines.append("-" * 72)
    for r in results.get("per_scheme", []):
        if r.get("alpha") == 0.0 or r.get("empty"):
            continue
        ci = r.get("pnl_vs_baseline_ci", (0.0, 0.0))
        bl_ci = r.get("baseline_pnl_ci", (0.0, 0.0))
        lines.append(
            f"a={r['alpha']:+5.1f}:  dPnL = ${ci[0]:+.2f} .. ${ci[1]:+.2f}   "
            f"(baseline ${bl_ci[0]:.2f} .. ${bl_ci[1]:.2f})"
        )
    lines.append("")

    # --- Monotonicity ---
    lines.append("-" * 72)
    lines.append("MONOTONICITY (Spearman ρ vs a)")
    lines.append("-" * 72)
    lines.append(
        f"{'Metric':<20s} {'Type':>9s}  {'ρ':>7s}  {'p-val':>7s}  "
        f"{'CI':>18s}  {'Expected':>9s}  {'Match':>5s}"
    )
    lines.append("-" * 72)
    for key, m in results.get("monotonicity", {}).items():
        if key == "error":
            continue
        ci_str = f"[{m['spearman_ci_lo']:+.3f}, {m['spearman_ci_hi']:+.3f}]"
        sign_str = f"{m['expected_sign']:+d}" if m['expected_sign'] != 0 else "none"
        lines.append(
            f"{key:<20s} {m['metric_type']:>9s}  "
            f"{m['spearman_rho']:+7.3f}  {m['spearman_p']:7.4f}  "
            f"{ci_str:>18s}  {sign_str:>9s}  "
            f"{'Y' if m['sign_match'] else 'N':>5s}"
        )
    lines.append("")

    # --- Concavity ---
    sortino_m = results.get("monotonicity", {}).get("sortino", {})
    conc = sortino_m.get("concavity")
    if conc:
        lines.append("-" * 72)
        lines.append("PRIMARY METRIC CONCAVITY (Sortino ~ b₀ + b₁a + b₂a²)")
        lines.append("-" * 72)
        lines.append(f"  R²: {conc.get('quadratic_r2', 0):.4f}")
        lines.append(f"  b₂: {conc.get('beta2', 0):+.4f}  (p={conc.get('beta2_p', 1):.4f})")
        lines.append(f"  AIC quadratic: {conc.get('aic_quadratic', 0):.2f}")
        lines.append(f"  AIC linear:    {conc.get('aic_linear', 0):.2f}")
        if conc.get("has_optimum"):
            lines.append(f"  Peak a: {conc['peak_alpha']:+.3f}  (significant interior optimum)")
        else:
            lines.append(f"  Peak a: {conc.get('peak_alpha', float('nan')):+.3f}  (no significant interior optimum)")
        lines.append("")

    # --- Optimal a ---
    opt = results.get("optimal_alpha")
    lines.append(f"Optimal a (full-sample): {opt:+.3f}" if opt is not None else "Optimal a: none (monotonic or flat)")

    # --- Walk-forward ---
    wf = results.get("walkforward", {})
    lines.append("")
    lines.append("-" * 72)
    lines.append(f"WALK-FORWARD VALIDATION (Path {wf.get('path', '?')})")
    lines.append("-" * 72)
    if wf.get("error"):
        lines.append(f"  Error: {wf['error']}")
    else:
        if wf.get("path") == "A":
            lines.append(f"  a* = {wf.get('alpha_star', 0):+.2f} vs baseline (a=0)")
            lines.append(f"  Windows: {wf.get('n_star_wins', 0)}/{wf.get('n_valid', 0)} a* wins")
            lines.append(f"  Win fraction: {wf.get('star_win_frac', 0):.1%}")
            lines.append(f"  Mean Sortino diff: {wf.get('mean_sortino_diff', 0):+.3f}")
            lines.append(f"  OOS valid: {'YES' if wf.get('oos_valid') else 'NO'} (≥70% threshold)")
        else:
            lines.append(f"  a = -1 vs a = +1 (extremes comparison)")
            lines.append(f"  Windows: {wf.get('n_high_prob_wins', 0)}/{wf.get('n_valid', 0)} high-prob wins")
            lines.append(f"  Win fraction: {wf.get('high_prob_win_frac', 0):.1%}")
            lines.append(f"  Mean Sortino diff: {wf.get('mean_sortino_diff', 0):+.3f}")
            lines.append(f"  OOS valid: {'YES' if wf.get('oos_valid') else 'NO'} (≥70% threshold)")

        lines.append("")
        lines.append("  Per-window details:")
        for w in wf.get("per_window", []):
            if "sortino_diff" in w:
                lines.append(
                    f"    {w['label']:>22s}  dSortino={w['sortino_diff']:+.3f}  "
                    f"{'Y' if w.get('star_outperforms', w.get('high_prob_wins', False)) else 'N'}"
                )
            else:
                lines.append(f"    {w['label']:>22s}  {w.get('error', 'no data')}")

    lines.append("")
    lines.append("=" * 72)
    lines.append("CAVEATS")
    lines.append("=" * 72)
    lines.append("1. Exploratory analysis; p-values descriptive, not confirmatory.")
    lines.append("2. Backtest omits min_book_depth, min_24h_volume, min_total_volume filters.")
    lines.append("   Low-price markets (~0.70) may be thinner -> fill model more optimistic at low end.")
    lines.append(f"3. Band [{band[0]:.2f}, {band[1]:.2f}] wider than live [0.80, 0.95] —")
    lines.append("   run sensitivity check on live band before generalizing.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run allocation analysis from CLI and print report."""
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

    store = ContractPriceStore()

    if store._data is None or len(store._data) == 0:
        print("No historical data found. Run historical fetch first.")
        sys.exit(1)

    # Primary: expanded band
    print("Running primary analysis on expanded band [0.70, 0.95]...")
    results_expanded = run_allocation_analysis(
        store,
        band_low=0.70, band_high=0.95,
        n_walkforward_windows=4,
    )
    print(format_report(results_expanded))

    # Sensitivity: current live band
    print("\n\nRunning sensitivity analysis on live band [0.80, 0.95]...")
    results_current = run_allocation_analysis(
        store,
        band_low=0.80, band_high=0.95,
        n_walkforward_windows=4,
    )
    print(format_report(results_current))

    # Side-by-side summary
    print("\n\n" + "=" * 72)
    print("BAND SENSITIVITY SUMMARY")
    print("=" * 72)
    for label, res in [("Expanded [0.70,0.95]", results_expanded),
                        ("Current  [0.80,0.95]", results_current)]:
        opt = res.get("optimal_alpha")
        sortino_m = res.get("monotonicity", {}).get("sortino", {})
        rho = sortino_m.get("spearman_rho", 0)
        print(f"  {label}:  optimal_a={opt}, Sortino ρ={rho:+.3f}")


if __name__ == "__main__":
    main()
