"""
crypto_sweep.py — Per-market 4D parameter sweep for crypto series.

Sweeps band_low x alpha x (min_dte, max_dte) independently for each
crypto market (BTC, ETH, SOL).

Uses multiprocessing (8 workers) for parallel backtest evaluation.
Top-N configs per market are re-validated with walk-forward OOS
(calendar-window stability, no parameter re-optimization) to flag overfitting.

Outputs:
  - DATA/crypto_sweep/{slug_filter}_results.csv   (full grid)
  - DATA/crypto_sweep/{slug_filter}_top30.csv     (top 30 by composite score)
  - DATA/crypto_sweep/summary.md                  (top 10 per market + walk-forward OOS)
"""

from __future__ import annotations

import itertools
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("crypto_sweep")

# Ensure fader/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest.harness import HarnessDefaults, load_store  # noqa: E402

# crypto_sweep uses $25 notional per trade (vs the $10 shared default) --
# this market runs a larger simulated bankroll (INITIAL_CAPITAL=$1000
# below) than the other CLIs' backtests. Divergence kept explicit here
# rather than silently baked into harness.py (Phase 5 plan requirement).
CRYPTO_DEFAULTS = replace(HarnessDefaults(), order_notional_usd=25.0)

# ---------------------------------------------------------------------------
# Parameter ranges
# ---------------------------------------------------------------------------

BAND_LOWS = [round(x, 2) for x in np.arange(0.50, 0.91, 0.05)]
BAND_HIGH = 0.95
ALPHAS = [round(x, 2) for x in np.arange(-0.50, 0.51, 0.05)]
MIN_DTES = list(range(0, 8))   # 0..7
MAX_DTES = list(range(1, 8))   # 1..7

# Crypto markets to sweep (slug_filter -> display name)
CRYPTO_MARKETS = [
    ("bitcoin-above", "BTC"),
    ("ethereum-above", "ETH"),
    ("solana-above", "SOL"),
    ("xrp-above", "XRP"),
]

# Walk-forward OOS validation (top N configs per market get re-tested)
WALKFORWARD_TOP_N = 3
WALKFORWARD_WINDOWS = 4
WALKFORWARD_BOOTSTRAP = 1000

INITIAL_CAPITAL = 1000.0
ORDER_NOTIONAL = CRYPTO_DEFAULTS.order_notional_usd  # $25 (see CRYPTO_DEFAULTS comment above)
MIN_TIME_IN_BAND_DAYS = 1
SPREAD_C = 1.0
N_BOOTSTRAP = CRYPTO_DEFAULTS.n_bootstrap
N_WORKERS = 8

# Progress reporting: how often to print from main thread
PROGRESS_EVERY_N = 200


# ---------------------------------------------------------------------------
# Composite score (risk-adjusted PnL)
# ---------------------------------------------------------------------------

def composite_score(m: dict) -> float:
    """Risk-adjusted score: reward Sortino + return, penalise max DD + tail risk.

    Higher = better.
    """
    sortino = m.get("sortino") or 0.0
    total_pnl = m.get("total_pnl") or 0.0
    max_dd_pct = max(m.get("max_drawdown_pct") or 0.0, 0.001)
    cvar = abs(m.get("daily_cvar_95") or 0.0)
    calmar = m.get("calmar") or 0.0

    return_pct = total_pnl / INITIAL_CAPITAL

    reward = (
        1.0 * sortino
        + 0.5 * return_pct * 100
        + 1.0 * calmar
    )
    penalty = (
        0.5 * (max_dd_pct * 100)
        + 0.2 * (cvar / max(abs(total_pnl), 1.0)) * 100
    )
    return reward - penalty


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dte_pairs() -> List[Tuple[int, int]]:
    """Valid (min_dte, max_dte) pairs: min < max."""
    return [(lo, hi) for lo in MIN_DTES for hi in MAX_DTES if lo < hi]


def _build_all_configs() -> List[Tuple[float, float, int, int]]:
    """Cartesian product of band_low x alpha x dte pairs."""
    configs = []
    for band_low in BAND_LOWS:
        for alpha in ALPHAS:
            for min_dte, max_dte in _dte_pairs():
                configs.append((band_low, alpha, min_dte, max_dte))
    return configs


def _slug_filter_to_prefix(filt: str) -> str:
    return filt.replace("-", "_")


# ---------------------------------------------------------------------------
# Worker function (module-level for pickling)
# ---------------------------------------------------------------------------

def _build_result_row(
    label: str,
    slug_filter: str,
    band_low: float,
    alpha: float,
    min_dte: int,
    max_dte: int,
    m: dict,
) -> dict:
    """Convert metrics dict to a flat output row."""
    row = {
        "market": label,
        "slug_filter": slug_filter,
        "band_low": band_low,
        "band_high": BAND_HIGH,
        "alpha": alpha,
        "min_dte": min_dte,
        "max_dte": max_dte,
        "n_trades": m["n_trades"],
        "total_pnl": round(m["total_pnl"], 4),
        "hit_rate": round(m["hit_rate"], 4),
        "expectancy": round(m["expectancy"], 4),
        "sortino": round(m["sortino"], 4),
        "calmar": round(m["calmar"], 4) if m["calmar"] else None,
        "max_drawdown": round(m["max_drawdown"], 4),
        "max_drawdown_pct": round(m["max_drawdown_pct"], 6),
        "daily_var_95": round(m["daily_var_95"], 4),
        "daily_var_99": round(m["daily_var_99"], 4),
        "daily_cvar_95": round(m["daily_cvar_95"], 4),
        "daily_skew": round(m["daily_skew"], 4),
        "daily_kurtosis": round(m["daily_kurtosis"], 4),
        "n_daily": m["n_daily"],
        "n_active_days": m["n_active_days"],
        "pnl_ci_lo": round(m["pnl_ci_95"][0], 4),
        "pnl_ci_hi": round(m["pnl_ci_95"][1], 4),
    }
    row["score"] = round(composite_score(row), 4)
    return row


def _run_chunk(args):
    """Run backtests for a chunk of configs on a market subset.

    args = (df_json, label, slug_filter, configs_chunk)
    where df_json is the market-filtered DataFrame serialized as JSON.
    """
    # Rebuild DataFrame from JSON + import inside worker for clean state
    # (this is the spawn-safe pattern backtest/harness.py's run_grid copies:
    # module-level worker fn, df_json serialized and passed INSIDE the
    # worker's args tuple -- no pool initializer).
    import pandas as pd
    from backtest.engine import BacktestConfig
    from backtest.harness import run_config
    from execution.sizing import make_sizing_fn

    df_json, label, slug_filter, configs_chunk = args
    from io import StringIO
    market_df = pd.read_json(StringIO(df_json), orient="table")

    results = []
    for band_low, alpha, min_dte, max_dte in configs_chunk:
        sizing_fn = make_sizing_fn(alpha, float(band_low), BAND_HIGH)
        cfg = BacktestConfig(
            band_low=float(band_low),
            band_high=BAND_HIGH,
            min_dte=min_dte,
            max_dte=max_dte,
            min_time_in_band_days=MIN_TIME_IN_BAND_DAYS,
            order_notional_usd=ORDER_NOTIONAL,
            spread_c=SPREAD_C,
            sizing_fn=sizing_fn,
            n_bootstrap=N_BOOTSTRAP,
        )
        row = run_config(market_df, cfg, initial_capital=INITIAL_CAPITAL)
        if row.empty:
            continue

        results.append(
            _build_result_row(label, slug_filter, band_low, alpha, min_dte, max_dte, row.metrics)
        )

    return results


# ---------------------------------------------------------------------------
# Walk-forward OOS validation (top-N configs per market)
# ---------------------------------------------------------------------------

def _run_walkforward_for_top_configs(
    df_out: pd.DataFrame, market_df: pd.DataFrame, top_n: int
) -> List[dict]:
    """Re-test top_n configs (by score) across calendar windows.

    No parameter is re-optimized on any window — same config applied to
    every window. Returns per-config stability summaries for the report.

    Phase 5: moved to backtest.harness.window_stability (one of three
    distinct walk-forward variants -- see harness.py module docstring).
    This wrapper preserves crypto_sweep's original module-level defaults
    (WALKFORWARD_WINDOWS, WALKFORWARD_BOOTSTRAP, ORDER_NOTIONAL,
    INITIAL_CAPITAL, MIN_TIME_IN_BAND_DAYS, SPREAD_C, BAND_HIGH).
    """
    from backtest.harness import window_stability

    return window_stability(
        df_out, market_df, top_n,
        n_windows=WALKFORWARD_WINDOWS,
        n_bootstrap=WALKFORWARD_BOOTSTRAP,
        order_notional_usd=ORDER_NOTIONAL,
        initial_capital=INITIAL_CAPITAL,
        min_time_in_band_days=MIN_TIME_IN_BAND_DAYS,
        spread_c=SPREAD_C,
        band_high=BAND_HIGH,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(results: List[dict], slug_filter: str, out_dir: str) -> pd.DataFrame:
    df = pd.DataFrame(results)
    if df.empty:
        return df

    df = df.sort_values("score", ascending=False)

    prefix = _slug_filter_to_prefix(slug_filter)
    full_path = os.path.join(out_dir, f"{prefix}_results.csv")
    top_path = os.path.join(out_dir, f"{prefix}_top30.csv")

    df.to_csv(full_path, index=False)
    df.head(30).to_csv(top_path, index=False)
    print(f"  -> {full_path}  ({len(df)} rows)")
    return df


def write_summary(
    all_dfs: Dict[str, pd.DataFrame],
    all_walkforward: Dict[str, List[dict]],
    out_dir: str,
) -> None:
    """Write a markdown summary: top 10 per market + walk-forward OOS validation."""
    lines = [
        "# Crypto Market Parameter Sweep - Summary",
        "",
        f"**Bankroll**: \\${INITIAL_CAPITAL:,.0f}  ",
        f"**Notional**: \\${ORDER_NOTIONAL:,.0f} per trade  ",
        f"**Band low sweep**: {BAND_LOWS[0]:.2f} - {BAND_LOWS[-1]:.2f} (step 0.05)  ",
        f"**Band high**: {BAND_HIGH} (fixed)  ",
        f"**Alpha sweep**: {ALPHAS[0]:.2f} - {ALPHAS[-1]:.2f} (step 0.05)  ",
        f"**DTE sweep**: min {MIN_DTES[0]}-{MIN_DTES[-1]}, max {MAX_DTES[0]}-{MAX_DTES[-1]} (min < max only)  ",
        f"**Workers**: {N_WORKERS}  ",
        f"**Bootstrap**: {N_BOOTSTRAP}  ",
        "",
        "## Composite Score",
        "",
        "`score = Sortino + 0.5*Return%*100 + Calmar - 0.5*MaxDD%*100 - 0.2*(CVaR/|PnL|)*100`",
        "",
        "## Walk-Forward OOS Validation",
        "",
        f"Top {WALKFORWARD_TOP_N} configs per market (by score) are re-tested across "
        f"{WALKFORWARD_WINDOWS} contiguous calendar windows with **no parameter "
        "re-optimization** — the same config is applied to every window. This checks "
        "whether performance is consistent over time or concentrated in one lucky "
        "stretch (overfitting signal). `stability_grade` of `stable` or `moderate` "
        "means the config held up; `unstable` means the grid-search result is likely "
        "overfit to a specific period.",
        "",
        "---",
        "",
    ]

    for label, df in all_dfs.items():
        if df.empty:
            lines.append(f"## {label}\n\n*No valid results.*\n")
            continue

        lines.append(f"## {label}")
        lines.append("")
        lines.append("### Best configuration (by composite score)")
        best = df.iloc[0]
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| band_low | {best['band_low']:.2f} |")
        lines.append(f"| alpha | {best['alpha']:.2f} |")
        lines.append(f"| min_dte | {int(best['min_dte'])} |")
        lines.append(f"| max_dte | {int(best['max_dte'])} |")
        lines.append("")
        lines.append("### Performance")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Score | {best['score']:.2f} |")
        lines.append(f"| Total PnL | \\${best['total_pnl']:,.2f} |")
        lines.append(f"| Sortino | {best['sortino']:.2f} |")
        lines.append(
            f"| Calmar | {best['calmar']:.2f}" if best["calmar"]
            else "| Calmar | - |"
        )
        lines.append(f"| Max DD \\$ | \\${best['max_drawdown']:,.2f} |")
        lines.append(f"| Max DD % | {best['max_drawdown_pct']*100:.1f}% |")
        lines.append(f"| VaR 95 | \\${best['daily_var_95']:,.2f} |")
        lines.append(f"| CVaR 95 | \\${best['daily_cvar_95']:,.2f} |")
        lines.append(f"| Hit Rate | {best['hit_rate']*100:.1f}% |")
        lines.append(f"| N Trades | {int(best['n_trades'])} |")
        lines.append(
            f"| PnL CI 95 | \\${best['pnl_ci_lo']:,.2f} - \\${best['pnl_ci_hi']:,.2f} |"
        )
        lines.append("")

        # Top 10 table
        lines.append("### Top 10 configurations (sorted by risk-adjusted score)")
        lines.append("")
        top10 = df.head(10)
        header = (
            "| # | band_low | alpha | min_dte | max_dte | Score | PnL | Sortino | "
            "Calmar | MaxDD% | VaR95 | nTrades |"
        )
        lines.append(header)
        lines.append(
            "|---|----------|-------|---------|---------|-------|-----|---------|"
            "--------|--------|-------|---------|"
        )
        for i, (_, r) in enumerate(top10.iterrows(), 1):
            calmar_str = f"{r['calmar']:.2f}" if r["calmar"] else "-"
            lines.append(
                f"| {i} | {r['band_low']:.2f} | {r['alpha']:.2f} | "
                f"{int(r['min_dte'])} | {int(r['max_dte'])} | "
                f"{r['score']:.1f} | \\${r['total_pnl']:,.0f} | "
                f"{r['sortino']:.2f} | {calmar_str} | "
                f"{r['max_drawdown_pct']*100:.1f}% | "
                f"\\${r['daily_var_95']:,.0f} | {int(r['n_trades'])} |"
            )
        lines.append("")

        # Walk-forward OOS section
        wf_list = all_walkforward.get(label, [])
        if wf_list:
            lines.append("### Walk-Forward OOS Validation (top configs)")
            lines.append("")
            lines.append(
                "| Rank | band_low | alpha | DTE | Grade | Sortino CV | "
                "Sortino min/max | PnL concentration | Windows w/ trades |"
            )
            lines.append(
                "|------|----------|-------|-----|-------|------------|"
                "-----------------|--------------------|--------------------|"
            )
            for i, wf in enumerate(wf_list, 1):
                st = wf["stability"]
                sortino_cv = f"{st['sortino_cv']:.2f}" if st.get("sortino_cv") is not None else "N/A"
                sortino_rng = (
                    f"{st['sortino_min']:.2f} / {st['sortino_max']:.2f}"
                    if st.get("sortino_min") is not None else "N/A"
                )
                pnl_conc = f"{st['pnl_concentration']*100:.0f}%" if st.get("pnl_concentration") is not None else "N/A"
                lines.append(
                    f"| {i} | {wf['band_low']:.2f} | {wf['alpha']:.2f} | "
                    f"[{wf['min_dte']},{wf['max_dte']}] | {st['stability_grade']} | "
                    f"{sortino_cv} | {sortino_rng} | {pnl_conc} | "
                    f"{st['n_nonempty']}/{st['n_windows']} |"
                )
            lines.append("")

            # Per-window detail for rank #1
            top1 = wf_list[0]
            lines.append(
                f"**Per-window detail, rank 1** (band_low={top1['band_low']:.2f}, "
                f"alpha={top1['alpha']:.2f}, DTE=[{top1['min_dte']},{top1['max_dte']}]):"
            )
            lines.append("")
            lines.append("| Window | N Trades | Sortino | Calmar | Total PnL |")
            lines.append("|--------|----------|---------|--------|-----------|")
            for w in top1["per_window"]:
                if w.n_trades == 0:
                    lines.append(f"| {w.label} | 0 | - | - | - |")
                    continue
                m = w.metrics
                calmar_str = f"{m['calmar']:.2f}" if m.get("calmar") else "-"
                lines.append(
                    f"| {w.label} | {w.n_trades} | {m.get('sortino', 0):.2f} | "
                    f"{calmar_str} | \\${m.get('total_pnl', 0):,.2f} |"
                )
            lines.append("")
        else:
            lines.append("*Walk-forward validation unavailable (no results).*")
            lines.append("")

        lines.append("---")
        lines.append("")

    from backtest.report import write_report

    summary_path = os.path.join(out_dir, "summary.md")
    write_report(summary_path, "", ["\n".join(lines)])
    print(f"\nSummary -> {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_configs = _build_all_configs()
    per_market = len(all_configs)
    total = per_market * len(CRYPTO_MARKETS)

    print("=" * 60)
    print("Crypto Market Parameter Sweep")
    print("=" * 60)
    print(f"Markets:       {[m[0] for m in CRYPTO_MARKETS]}")
    print(f"band_low:      {BAND_LOWS}")
    print(f"band_high:     {BAND_HIGH}")
    print(f"alpha:         {ALPHAS}")
    print(f"min_dte:       {MIN_DTES}")
    print(f"max_dte:       {MAX_DTES}")
    print(f"DTE pairs:     {len(_dte_pairs())}  (min < max only)")
    print(f"Bankroll:      ${INITIAL_CAPITAL:,.0f}")
    print(f"Notional:      ${ORDER_NOTIONAL:,.0f}")
    print(f"Workers:       {N_WORKERS}")
    print(f"Configs/mkt:   {per_market:,}")
    print(f"Total configs: {total:,}")
    print("=" * 60)

    # Load data once
    print("\nLoading historical prices...")
    df = load_store()
    print(f"  {len(df):,} rows, {df['slug'].nunique()} unique slugs")

    # Output directory
    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "DATA", "crypto_sweep"
    )
    os.makedirs(out_dir, exist_ok=True)

    all_dfs: Dict[str, pd.DataFrame] = {}
    all_walkforward: Dict[str, List[dict]] = {}
    t_start = time.perf_counter()

    for slug_filter, label in CRYPTO_MARKETS:
        print(f"\n{'=' * 50}")
        print(f"  {label}  ({slug_filter})")
        print(f"{'=' * 50}")
        t_mkt = time.perf_counter()

        # Filter to this market
        market_df = df[df["slug"].str.contains(slug_filter, na=False)].copy()
        if market_df.empty:
            print(f"  No data for {label} - skipping")
            continue
        print(f"  {len(market_df):,} rows, {market_df['slug'].nunique()} slugs")

        # Serialize for workers (avoids pickling the full df into each chunk)
        df_json = market_df.to_json(orient="table")

        # Split configs into N_WORKERS chunks
        chunks = np.array_split(all_configs, N_WORKERS)
        chunks = [c.tolist() for c in chunks if len(c) > 0]

        print(f"  {len(chunks)} chunks, {len(chunks[0])} configs each  (launching workers...)")

        results: List[dict] = []
        completed = 0

        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = [
                ex.submit(_run_chunk, (df_json, label, slug_filter, chunk))
                for chunk in chunks
            ]
            for fut in as_completed(futures):
                batch = fut.result()
                results.extend(batch)
                completed += len(batch)
                # Progress lines
                if (
                    completed % PROGRESS_EVERY_N < len(batch) + 1
                    or completed >= per_market
                ):
                    elapsed = time.perf_counter() - t_mkt
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (per_market - completed) / rate if rate > 0 else 0
                    print(
                        f"  [{label}] {completed}/{per_market}  "
                        f"({completed / per_market * 100:.0f}%)  "
                        f"ETA {eta / 60:.0f}m  "
                        f"rate {rate:.1f}/s"
                    )

        if not results:
            print(f"  No valid results for {label}")
            continue

        df_out = save_results(results, slug_filter, out_dir)
        all_dfs[label] = df_out

        print(f"  Running walk-forward OOS validation on top {WALKFORWARD_TOP_N} configs...")
        wf_results = _run_walkforward_for_top_configs(df_out, market_df, WALKFORWARD_TOP_N)
        all_walkforward[label] = wf_results

        elapsed_mkt = time.perf_counter() - t_mkt
        print(f"  {label} done in {elapsed_mkt / 60:.1f}m  ({len(results)} configs with trades)")

    write_summary(all_dfs, all_walkforward, out_dir)

    elapsed_total = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"Total time: {elapsed_total / 60:.1f}m")
    print(f"Results in: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
