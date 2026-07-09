"""backtest/band_sweep.py

Sweep lower band bound from 0.50 to 0.80 in 0.05 increments,
fixed upper bound 0.95. For each band, run full allocation tilt
analysis and report optimal tilt + metrics.

Also supports 4D grid sweep (band_low × alpha × min_dte × max_dte)
for brute-force parameter optimization.

Run as: python -c "import sys; sys.path.insert(0,'fader'); from backtest.band_sweep import main; main()"
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.allocation_analysis import run_allocation_analysis
from backtest.engine import BacktestConfig
from backtest.harness import HarnessDefaults, load_store, run_config
from backtest.historical import ContractPriceStore
from backtest.metrics import (
    PERIODS_PER_YEAR,
    compute_all_metrics,
    sortino_ratio,
)
from backtest.report import write_report
from execution.sizing import MIN_EFFECTIVE_NOTIONAL, make_sizing_fn

# band_sweep uses the shared defaults unmodified (no divergence to name).
_DEFAULTS = HarnessDefaults()

logger = logging.getLogger(__name__)


@dataclass
class BandResult:
    band_low: float
    band_high: float
    n_entries: int
    optimal_alpha: Optional[float]  # quadratic-fit peak
    best_sweep_alpha: float         # sweep point with highest Sortino
    sortino_at_peak: float          # Sortino at sweep point nearest quadratic peak
    sortino_best: float             # Sortino at best sweep point
    sortino_baseline: float
    sortino_min: float              # worst Sortino across sweep
    sortino_max: float              # best Sortino across sweep
    sortino_rho: float              # Spearman ρ(Sortino, α)
    has_optimum: bool               # concavity significant + peak interior
    total_pnl_baseline: float
    total_pnl_best: float           # PnL at best sweep point
    oos_valid: bool                 # walk-forward validates
    oos_win_frac: float
    elapsed_s: float


def sweep_lower_bound(
    store,
    low_values: List[float],
    high: float = 0.95,
    n_walkforward_windows: int = 4,
    n_bootstrap: int = 5000,
) -> List[BandResult]:
    """Sweep lower band bound, return sorted results.

    Args:
        store: ContractPriceStore or DataFrame with historical data.
        low_values: Lower bound values to test, e.g. [0.50, 0.55, ..., 0.80].
        high: Fixed upper bound.
        n_walkforward_windows: Calendar windows for OOS validation.
        n_bootstrap: Bootstrap iterations.

    Returns:
        List of BandResult sorted by band_low ascending.
    """
    results: List[BandResult] = []

    for low in low_values:
        t0 = time.perf_counter()
        logger.info(f"--- Band [{low:.2f}, {high:.2f}] ---")

        cfg = BacktestConfig(
            band_low=low, band_high=high,
            min_time_in_band_days=1,
            order_notional_usd=10.0,
            spread_c=1.0, slippage_c=0.0, adverse_selection_c=0.0,
            n_bootstrap=n_bootstrap,
        )

        analysis = run_allocation_analysis(
            store, cfg,
            band_low=low, band_high=high,
            n_walkforward_windows=n_walkforward_windows,
        )

        mono = analysis.get("monotonicity", {})
        sortino_m = mono.get("sortino", {})
        conc = sortino_m.get("concavity", {})

        # Find min/max Sortino across sweep points (not interpolated peak)
        sweep_schemes = [r for r in analysis.get("per_scheme", []) if not r.get("empty")]
        sortinos = [r.get("sortino", 0.0) or 0.0 for r in sweep_schemes]
        sortino_min = float(np.min(sortinos)) if sortinos else 0.0
        sortino_max = float(np.max(sortinos)) if sortinos else 0.0

        # Best sweep point (highest observed Sortino, not quadratic peak)
        if sweep_schemes:
            best_sweep = max(sweep_schemes, key=lambda r: r.get("sortino", 0.0) or 0.0)
            best_sweep_alpha = best_sweep["alpha"]
            best_sweep_sortino = best_sweep.get("sortino", 0.0) or 0.0
            best_sweep_pnl = best_sweep.get("total_pnl", 0.0)
        else:
            best_sweep_alpha = 0.0
            best_sweep_sortino = 0.0
            best_sweep_pnl = 0.0

        baseline = analysis.get("baseline", {})
        per_scheme = analysis.get("per_scheme", [])

        # Sortino at quadratic peak (interpolated optimum)
        optimal_alpha = analysis.get("optimal_alpha")
        sortino_at_peak = baseline.get("sortino", 0.0)
        if optimal_alpha is not None and sweep_schemes:
            # Find closest sweep point to quadratic peak
            closest = min(sweep_schemes,
                          key=lambda r: abs(r["alpha"] - optimal_alpha))
            sortino_at_peak = closest.get("sortino", 0.0) or 0.0

        wf = analysis.get("walkforward", {})

        elapsed = time.perf_counter() - t0

        br = BandResult(
            band_low=low,
            band_high=high,
            n_entries=analysis.get("n_entries", 0),
            optimal_alpha=optimal_alpha,
            best_sweep_alpha=best_sweep_alpha,
            sortino_at_peak=sortino_at_peak,
            sortino_best=best_sweep_sortino,
            sortino_baseline=baseline.get("sortino", 0.0) or 0.0,
            sortino_min=sortino_min,
            sortino_max=sortino_max,
            sortino_rho=sortino_m.get("spearman_rho", 0.0),
            has_optimum=conc.get("has_optimum", False),
            total_pnl_baseline=baseline.get("total_pnl", 0.0),
            total_pnl_best=best_sweep_pnl,
            oos_valid=wf.get("oos_valid", False),
            oos_win_frac=wf.get(
                "star_win_frac",
                wf.get("high_prob_win_frac", 0.0),
            ),
            elapsed_s=elapsed,
        )

        results.append(br)
        logger.info(
            f"  entries={br.n_entries}, opt_a={br.optimal_alpha}, "
            f"Sortino_best={br.sortino_best:.3f}, "
            f"Sortino_base={br.sortino_baseline:.3f}, "
            f"OOS_valid={br.oos_valid} ({br.oos_win_frac:.0%})"
        )

    # Sort by band_low ascending (narrowest first = highest low bound)
    results.sort(key=lambda r: r.band_low, reverse=False)
    return results


def format_band_sweep_report(results: List[BandResult]) -> str:
    """Formatted report for band sweep results."""
    lines = []
    lines.append("=" * 100)
    lines.append("BAND LOWER-BOUND SWEEP — ALLOCATION TILT ANALYSIS")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Each row: fixed upper=0.95, lower bound swept from 0.50 to 0.80")
    lines.append("Tilt α: -1 = weight low-prob contracts, +1 = weight high-prob contracts")
    lines.append("")

    # Header
    header = (
        f"{'Band':>10s}  {'Entries':>7s}  "
        f"{'a*':>6s}  {'Best a':>6s}  "
        f"{'Sortino':>8s}  {'Sortino':>8s}  {'Sortino':>8s}  "
        f"{'Sortino':>8s}  {'Sortino':>8s}  "
        f"{'PnL':>9s}  {'PnL':>9s}  "
        f"{'Convex?':>7s}  {'OOS?':>5s}  {'OOS%':>6s}  "
        f"{'Sec':>5s}"
    )
    subheader = (
        f"{'':>10s}  {'':>7s}  "
        f"{'peak':>6s}  {'best':>6s}  "
        f"{'at best':>8s}  {'at a=0':>8s}  {'min':>8s}  "
        f"{'max':>8s}  {'rho':>8s}  "
        f"{'at a=0':>9s}  {'at best':>9s}  "
        f"{'':>7s}  {'':>5s}  {'':>6s}  "
        f"{'':>5s}"
    )
    lines.append(header)
    lines.append(subheader)
    lines.append("-" * 100)

    for r in results:
        opt_str = f"{r.optimal_alpha:+.3f}" if r.optimal_alpha is not None else "none"
        best_str = f"{r.best_sweep_alpha:+.1f}"
        lines.append(
            f"[{r.band_low:.2f},{r.band_high:.2f}]  {r.n_entries:7d}  "
            f"{opt_str:>6s}  {best_str:>6s}  "
            f"{r.sortino_best:8.3f}  {r.sortino_baseline:8.3f}  "
            f"{r.sortino_min:8.3f}  {r.sortino_max:8.3f}  "
            f"{r.sortino_rho:+8.3f}  "
            f"${r.total_pnl_baseline:8.2f}  ${r.total_pnl_best:8.2f}  "
            f"{'Y' if r.has_optimum else 'N':>7s}  "
            f"{'Y' if r.oos_valid else 'N':>5s}  "
            f"{r.oos_win_frac:5.0%}  "
            f"{r.elapsed_s:5.0f}"
        )

    lines.append("")
    lines.append("-" * 100)
    lines.append("INTERPRETATION GUIDE")
    lines.append("-" * 100)
    lines.append("")
    lines.append("1. OPTIMAL TILT DIRECTION")
    lines.append("   a* > 0: tilt capital toward HIGHER-probability contracts (closer to 0.95)")
    lines.append("   a* < 0: tilt capital toward LOWER-probability contracts (closer to band_low)")
    lines.append("   a* ~ 0: equal-weight (current behavior) is already near-optimal")
    lines.append("")
    lines.append("2. DECISION CRITERIA (in priority order)")
    lines.append("   a) OOS VALID = Y: the optimum holds across calendar windows — not overfit")
    lines.append("   b) Sortino at a* > Sortino at a=0: tilt improves risk-adjusted return")
    lines.append("   c) Has convex peak (Convex? = Y): relationship is concave, peak is genuine optimum")
    lines.append("   d) Sortino ρ sign: if +, higher α → better Sortino (conservative wins)")
    lines.append("                      if -, lower α → better Sortino (aggressive wins)")
    lines.append("")
    lines.append("3. ALLOCATION RULE (if tilt is justified)")
    lines.append("   Given optimal a* and band [low, high]:")
    lines.append("     multiplier = 1 + a* * (fill_price - mid) / (range/2)")
    lines.append("     effective_notional = max($1.00, base_notional * multiplier)")
    lines.append("   This means:")
    lines.append("     - Contracts near the favored end get up to 2x base notional")
    lines.append("     - Contracts near the disfavored end get as low as $1.00 notional")
    lines.append("     - The tilt is smooth/linear across the band")
    lines.append("")
    lines.append("4. BAND WIDTH MATTERS")
    lines.append("   Narrow bands (e.g. [0.80, 0.95]) give weak tilt signal — price variation")
    lines.append("   within the band is too small for tilt to matter. Wider bands (e.g. [0.55, 0.95])")
    lines.append("   amplify the difference between high-prob and low-prob contracts.")
    lines.append("   ONLY tilt if the band is wide enough to produce a robust, OOS-valid optimum.")
    lines.append("")
    lines.append("5. RISK CAVEAT")
    lines.append("   Tilting toward low-prob contracts (a < 0) increases:")
    lines.append("     - Max drawdown %")
    lines.append("     - Worst single-trade loss")
    lines.append("     - Worst single-day loss")
    lines.append("   Tilting toward high-prob contracts (a > 0) reduces these but may reduce PnL.")
    lines.append("   Sortino balances these — it's the primary metric for this reason.")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 4D Grid Sweep: band_low × alpha × min_dte × max_dte
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class GridResult:
    band_low: float
    band_high: float
    alpha: float
    min_dte: int
    max_dte: int
    n_trades: int
    total_pnl: float
    sortino: float
    calmar: Optional[float]
    max_drawdown_pct: float
    hit_rate: float
    expectancy: float
    profit_factor: float
    worst_trade: float
    worst_day: float
    cvar_95: float
    daily_skew: float
    pnl_ci_lo: float
    pnl_ci_hi: float
    elapsed_s: float


def _generate_dte_pairs(
    min_dte_values: List[int],
    max_dte_values: List[int],
) -> List[Tuple[int, int]]:
    """Generate valid (min_dte, max_dte) pairs where min_dte < max_dte."""
    pairs = []
    for mn in min_dte_values:
        for mx in max_dte_values:
            if mn < mx:
                pairs.append((mn, mx))
    return pairs


def sweep_grid(
    store_or_df,
    band_low_values: List[float],
    band_high: float,
    alpha_values: List[float],
    min_dte_values: List[int],
    max_dte_values: List[int],
    order_notional_usd: float = 10.0,
    min_time_in_band_days: int = 1,
    spread_c: float = 1.0,
    slug_filter: Optional[str] = None,
) -> List[GridResult]:
    """4D grid sweep across band_low × alpha × min_dte × max_dte.

    For each valid (band_low, alpha, min_dte, max_dte) combination, runs a
    single backtest and records performance metrics. Invalid DTE combos
    (min_dte >= max_dte) are automatically excluded.

    Args:
        store_or_df: ContractPriceStore or DataFrame with historical data.
        band_low_values: Lower bound values, e.g. [0.50, 0.55, ..., 0.90].
        band_high: Fixed upper bound (default 0.95).
        alpha_values: Tilt values, e.g. [0.0, 0.2, ..., 1.0].
        min_dte_values: Min DTE values, e.g. [0, 1, ..., 7].
        max_dte_values: Max DTE values, e.g. [1, 2, ..., 7].
        order_notional_usd: Base notional per trade.
        min_time_in_band_days: Days required in band before entry.
        spread_c: Spread cost in cents.
        slug_filter: If set, only backtest contracts whose slug contains
                     this substring (e.g. "bitcoin-above" for BTC only).

    Returns:
        List of GridResult sorted by Sortino descending.
    """
    # Snapshot data
    if isinstance(store_or_df, ContractPriceStore):
        df = store_or_df.snapshot()
    elif isinstance(store_or_df, pd.DataFrame):
        df = store_or_df.copy()
    else:
        raise TypeError("store_or_df must be ContractPriceStore or DataFrame")

    # Filter by slug
    if slug_filter:
        df = df[df["slug"].str.contains(slug_filter, na=False)]
        if df.empty:
            logger.warning(f"No data matched slug_filter='{slug_filter}'")
            return []

    dte_pairs = _generate_dte_pairs(min_dte_values, max_dte_values)

    # Build full parameter grid
    param_grid = list(itertools.product(
        band_low_values, alpha_values, dte_pairs,
    ))
    total_combos = len(param_grid)
    logger.info(
        f"Grid sweep: {len(band_low_values)} bands × {len(alpha_values)} alphas × "
        f"{len(dte_pairs)} DTE pairs = {total_combos} combinations"
    )

    results: List[GridResult] = []
    t_start = time.perf_counter()

    for idx, (bl, alpha, (min_dte, max_dte)) in enumerate(param_grid):
        t0 = time.perf_counter()

        sizing_fn = make_sizing_fn(alpha, bl, band_high)

        cfg = BacktestConfig(
            band_low=bl,
            band_high=band_high,
            min_dte=min_dte,
            max_dte=max_dte,
            min_time_in_band_days=min_time_in_band_days,
            order_notional_usd=order_notional_usd,
            spread_c=spread_c,
            slippage_c=0.0,
            adverse_selection_c=0.0,
            n_bootstrap=_DEFAULTS.n_bootstrap,
            sizing_fn=sizing_fn,
        )

        try:
            row = run_config(df, cfg)
        except Exception as e:
            logger.warning(
                f"[{idx+1}/{total_combos}] bl={bl:.2f} a={alpha:+.1f} "
                f"dte=[{min_dte},{max_dte}]: backtest failed: {e}"
            )
            continue

        elapsed = time.perf_counter() - t0

        if row.empty:
            results.append(GridResult(
                band_low=bl, band_high=band_high,
                alpha=alpha, min_dte=min_dte, max_dte=max_dte,
                n_trades=0, total_pnl=0.0, sortino=0.0,
                calmar=None, max_drawdown_pct=0.0, hit_rate=0.0,
                expectancy=0.0, profit_factor=float("inf"),
                worst_trade=0.0, worst_day=0.0, cvar_95=0.0,
                daily_skew=0.0, pnl_ci_lo=0.0, pnl_ci_hi=0.0,
                elapsed_s=elapsed,
            ))
            continue

        trades_df = row.trades_df
        metrics = row.metrics

        pnls = trades_df["realized_pnl"].dropna().values
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        profit_factor = (
            abs(float(np.sum(wins)) / float(np.sum(losses)))
            if len(losses) > 0 and np.sum(losses) != 0
            else float("inf")
        )

        # Worst day from daily PnL series
        if "exit_date" in trades_df.columns:
            from backtest.metrics import daily_pnl_series
            daily = daily_pnl_series(trades_df)
            worst_day = float(np.min(daily)) if len(daily) > 0 else 0.0
        else:
            worst_day = 0.0

        pnl_ci = metrics.get("pnl_ci_95", (0.0, 0.0))

        gr = GridResult(
            band_low=bl,
            band_high=band_high,
            alpha=alpha,
            min_dte=min_dte,
            max_dte=max_dte,
            n_trades=int(len(pnls)),
            total_pnl=float(np.sum(pnls)),
            sortino=metrics.get("sortino", 0.0),
            calmar=metrics.get("calmar"),
            max_drawdown_pct=metrics.get("max_drawdown_pct", 0.0),
            hit_rate=metrics.get("hit_rate", 0.0),
            expectancy=metrics.get("expectancy", 0.0),
            profit_factor=profit_factor,
            worst_trade=float(np.min(pnls)) if len(pnls) > 0 else 0.0,
            worst_day=worst_day,
            cvar_95=metrics.get("daily_cvar_95", 0.0),
            daily_skew=metrics.get("daily_skew", 0.0),
            pnl_ci_lo=pnl_ci[0] if isinstance(pnl_ci, tuple) else 0.0,
            pnl_ci_hi=pnl_ci[1] if isinstance(pnl_ci, tuple) else 0.0,
            elapsed_s=elapsed,
        )
        results.append(gr)

        # Progress log every 50 or at end
        if (idx + 1) % 50 == 0 or (idx + 1) == total_combos:
            elapsed_total = time.perf_counter() - t_start
            rate = (idx + 1) / elapsed_total if elapsed_total > 0 else 0
            remaining = (total_combos - idx - 1) / rate if rate > 0 else 0
            logger.info(
                f"[{idx+1}/{total_combos}] {idx+1} done, "
                f"{rate:.1f}/s, ETA {remaining:.0f}s"
            )

    total_elapsed = time.perf_counter() - t_start
    logger.info(
        f"Grid sweep complete: {len(results)} results in {total_elapsed:.0f}s "
        f"({total_elapsed/len(results):.1f}s/run)" if results else "Grid sweep: no results"
    )

    # Sort by Sortino descending
    results.sort(key=lambda r: r.sortino, reverse=True)
    return results


def format_grid_report(
    results: List[GridResult],
    top_n: int = 40,
) -> str:
    """Formatted report for 4D grid sweep results.

    Args:
        results: Grid sweep results (should be pre-sorted).
        top_n: Number of top results to show in the summary table.
    """
    lines = []
    lines.append("=" * 120)
    lines.append("4D GRID SWEEP — band_low × alpha × min_dte × max_dte")
    lines.append("=" * 120)
    lines.append("")

    if not results:
        lines.append("No results.")
        return "\n".join(lines)

    nonempty = [r for r in results if r.n_trades > 0]

    # ── Top-N table ──
    lines.append(f"Top {min(top_n, len(nonempty))} by Sortino (of {len(nonempty)} with trades):")
    lines.append("")

    header = (
        f"{'Rank':>4s}  "
        f"{'Band':>10s}  {'a':>5s}  {'DTE':>8s}  "
        f"{'Trades':>6s}  {'Sortino':>8s}  {'Calmar':>8s}  "
        f"{'PnL':>9s}  {'Hit%':>6s}  {'MaxDD%':>7s}  "
        f"{'CVaR95':>8s}  {'WorstDay':>9s}  {'Skew':>6s}"
    )
    lines.append(header)
    lines.append("-" * 120)

    for i, r in enumerate(nonempty[:top_n]):
        calmar_str = f"{r.calmar:.3f}" if r.calmar is not None else "N/A"
        lines.append(
            f"{i+1:4d}  "
            f"[{r.band_low:.2f},{r.band_high:.2f}]  {r.alpha:+.1f}  "
            f"[{r.min_dte},{r.max_dte}]   "
            f"{r.n_trades:6d}  {r.sortino:8.3f}  {calmar_str:>8s}  "
            f"${r.total_pnl:8.2f}  {r.hit_rate:5.1%}  "
            f"{r.max_drawdown_pct:6.1%}  "
            f"${r.cvar_95:7.2f}  ${r.worst_day:8.2f}  "
            f"{r.daily_skew:+5.2f}"
        )

    lines.append("")
    lines.append("-" * 120)

    # ── Best per band_low ──
    lines.append("")
    lines.append("BEST PARAMETERS PER BAND_LOW (highest Sortino):")
    lines.append("-" * 80)
    lines.append(
        f"{'Band':>10s}  {'a':>5s}  {'DTE':>8s}  "
        f"{'Sortino':>8s}  {'PnL':>9s}  {'Trades':>6s}"
    )
    lines.append("-" * 80)

    for bl in sorted({r.band_low for r in nonempty}):
        band_results = [r for r in nonempty if r.band_low == bl]
        if not band_results:
            continue
        best = max(band_results, key=lambda r: r.sortino)
        lines.append(
            f"[{best.band_low:.2f},{best.band_high:.2f}]  {best.alpha:+.1f}  "
            f"[{best.min_dte},{best.max_dte}]   "
            f"{best.sortino:8.3f}  ${best.total_pnl:8.2f}  {best.n_trades:6d}"
        )

    lines.append("")
    lines.append("-" * 80)

    # ── Best per alpha ──
    lines.append("")
    lines.append("BEST PARAMETERS PER ALPHA (highest Sortino):")
    lines.append("-" * 80)
    lines.append(
        f"{'a':>5s}  {'Band':>10s}  {'DTE':>8s}  "
        f"{'Sortino':>8s}  {'PnL':>9s}  {'Trades':>6s}"
    )
    lines.append("-" * 80)

    for alpha in sorted({r.alpha for r in nonempty}):
        alpha_results = [r for r in nonempty if r.alpha == alpha]
        if not alpha_results:
            continue
        best = max(alpha_results, key=lambda r: r.sortino)
        lines.append(
            f"{best.alpha:+.1f}  "
            f"[{best.band_low:.2f},{best.band_high:.2f}]  "
            f"[{best.min_dte},{best.max_dte}]   "
            f"{best.sortino:8.3f}  ${best.total_pnl:8.2f}  {best.n_trades:6d}"
        )

    lines.append("")
    lines.append("-" * 80)

    # ── DTE sensitivity: average Sortino by DTE range width ──
    lines.append("")
    lines.append("DTE RANGE WIDTH SENSITIVITY (avg Sortino by max_dte - min_dte):")
    lines.append("-" * 60)
    lines.append(f"{'Width':>6s}  {'N':>6s}  {'Avg Sortino':>11s}  {'Best Sortino':>12s}")
    lines.append("-" * 60)

    width_groups: Dict[int, List[GridResult]] = {}
    for r in nonempty:
        width = r.max_dte - r.min_dte
        width_groups.setdefault(width, []).append(r)

    for width in sorted(width_groups):
        group = width_groups[width]
        avg_sortino = float(np.mean([r.sortino for r in group]))
        best_sortino = max(r.sortino for r in group)
        lines.append(
            f"{width:6d}  {len(group):6d}  {avg_sortino:11.4f}  {best_sortino:12.4f}"
        )

    lines.append("")
    lines.append("=" * 120)
    lines.append("CAVEATS")
    lines.append("=" * 120)
    lines.append("1. Grid sweep runs single backtests (no walk-forward, no concavity check).")
    lines.append("2. Results are in-sample — top Sortino may be overfit.")
    lines.append("3. Validate top candidates with walk-forward before deploying.")
    lines.append("4. Backtest omits min_book_depth, min_24h_volume, min_total_volume filters.")
    lines.append("5. DTE values beyond ~30 may be affected by sparse data at long horizons.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run band sweep + 4D grid sweep from CLI."""
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
    )

    store = load_store()
    if store.empty:
        print("No historical data. Run historical fetch first.")
        sys.exit(1)

    mode = sys.argv[1] if len(sys.argv) > 1 else "grid"

    if mode == "band":
        # ── Original band-only sweep ──
        low_values = [round(x, 2) for x in np.arange(0.50, 0.82, 0.05).tolist()]

        print(f"Sweeping lower bounds: {low_values}")
        print(f"Fixed upper bound: 0.95")
        print(f"This will run {len(low_values)} full allocation analyses")
        print(f"Estimated time: ~{len(low_values) * 60}s\n")

        results = sweep_lower_bound(
            store,
            low_values=low_values,
            high=_DEFAULTS.band_high,
            n_walkforward_windows=4,
            n_bootstrap=_DEFAULTS.n_bootstrap,
        )

        report = format_band_sweep_report(results)
        print(report)

        out_path = Path(__file__).parent.parent / "DATA" / "band_sweep_report.txt"
        write_report(out_path, "", [report])
        print(f"\nReport saved to: {out_path}")

    else:
        # ── 4D grid sweep (default) ──
        band_low_values = [round(x, 2) for x in np.arange(0.50, 0.92, 0.05).tolist()]
        alpha_values = [round(x, 1) for x in np.arange(0.0, 1.05, 0.2).tolist()]
        min_dte_values = list(range(0, 8))   # 0..7
        max_dte_values = list(range(1, 8))   # 1..7
        band_high = _DEFAULTS.band_high

        dte_pairs = _generate_dte_pairs(min_dte_values, max_dte_values)

        total = len(band_low_values) * len(alpha_values) * len(dte_pairs)
        print(f"4D Grid Sweep — BTC only")
        print(f"  band_low:  {band_low_values}")
        print(f"  band_high: {band_high}")
        print(f"  alpha:     {alpha_values}")
        print(f"  min_dte:   {min_dte_values}")
        print(f"  max_dte:   {max_dte_values}")
        print(f"  Valid DTE pairs: {len(dte_pairs)}")
        print(f"  Total combinations: {total}")
        print(f"  Estimated time: ~{total * 0.5:.0f}s "
              f"({total * 0.5 / 60:.1f} min)")
        print()

        results = sweep_grid(
            store,
            band_low_values=band_low_values,
            band_high=band_high,
            alpha_values=alpha_values,
            min_dte_values=min_dte_values,
            max_dte_values=max_dte_values,
            order_notional_usd=10.0,
            slug_filter="bitcoin-above",
        )

        report = format_grid_report(results, top_n=40)
        print(report)

        out_path = Path(__file__).parent.parent / "DATA" / "grid_sweep_report.txt"
        write_report(out_path, "", [report])
        print(f"\nReport saved to: {out_path}")

        # Also save full CSV
        csv_path = Path(__file__).parent.parent / "DATA" / "grid_sweep_results.csv"
        df_out = pd.DataFrame([r.__dict__ for r in results])
        df_out.to_csv(csv_path, index=False)
        print(f"Full results CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
