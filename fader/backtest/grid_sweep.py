"""backtest/grid_sweep.py

2D grid sweep: alpha × band_low with walk-forward OOS validation.

Sweeps alpha [-1.0, +1.0] step 0.05 and band_low [0.50, 0.90] step 0.05,
computing Sortino, Calmar, total PnL, max DD%, daily skew, and OOS
validation for every combination. Outputs a markdown report.

Run as:
  python -c "import sys; sys.path.insert(0,'fader'); from backtest.grid_sweep import main; main()"
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig
from backtest.harness import BandCache, build_band_cache, load_store
from backtest.harness import _lean_metrics_row as _metrics_row
from backtest.harness import _run_lean
from backtest.harness import walkforward_lean as _oos_validate_harness
from backtest.historical import ContractPriceStore
from backtest.metrics import (
    PERIODS_PER_YEAR,
    calmar_ratio,
    daily_pnl_series,
    max_drawdown_pct,
    sortino_ratio,
)
from backtest.report import write_report
from backtest.walkforward import partition_calendar_windows
from execution.sizing import make_sizing_fn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid sweep result
# ---------------------------------------------------------------------------

@dataclass
class GridPoint:
    band_low: float
    band_high: float
    alpha: float
    n_trades: int
    sortino: float
    calmar: Optional[float]
    total_pnl: float
    max_dd_pct: float
    daily_skew: float
    hit_rate: float
    expectancy: float
    n_daily: int
    n_active_days: int
    # OOS
    oos_n_windows: int
    oos_n_valid: int
    oos_win_frac: float
    oos_mean_sortino_diff: float
    oos_valid: bool
    elapsed_s: float


# ---------------------------------------------------------------------------
# Lean scheme runner + band cache + OOS validation: moved to
# backtest/harness.py (Phase 5) as _run_lean, _lean_metrics_row (imported
# above as _metrics_row), BandCache, build_band_cache, and walkforward_lean
# (imported above as _oos_validate_harness -- distinct from
# harness.walkforward_normalized / harness.window_stability; see
# harness.py's module docstring for why all three variants are kept
# separate). Thin call-shape wrappers below preserve this module's
# original defaults (order_notional_usd=10.0, min_time_in_band_days=1,
# spread_c=1.0) and positional signatures so downstream call sites in
# this file are unchanged.
# ---------------------------------------------------------------------------


def _build_band_cache(
    df: pd.DataFrame,
    band_low: float,
    band_high: float,
    n_windows: int,
    n_bootstrap: int,
) -> BandCache:
    return build_band_cache(
        df, band_low, band_high, n_windows, n_bootstrap,
        order_notional_usd=10.0, min_time_in_band_days=1, spread_c=1.0,
    )


def _oos_validate(
    cache: BandCache,
    df: pd.DataFrame,
    band_low: float,
    band_high: float,
    alpha: float,
    sizing_fn: Callable[[float], float],
    n_bootstrap: int,
) -> Dict:
    return _oos_validate_harness(
        cache, df, band_low, band_high, alpha, sizing_fn, n_bootstrap,
        order_notional_usd=10.0, min_time_in_band_days=1, spread_c=1.0,
    )


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_grid_sweep(
    store: ContractPriceStore,
    alphas: List[float],
    band_lows: List[float],
    band_high: float = 0.95,
    n_windows: int = 4,
    n_bootstrap_full: int = 2000,
    n_bootstrap_oos: int = 1000,
) -> List[GridPoint]:
    """Run full 2D grid sweep.

    Returns list of GridPoint results.
    """
    df = store.snapshot()
    if df.empty:
        raise ValueError("No historical data")

    total = len(band_lows) * len(alphas)
    results: List[GridPoint] = []
    combo_idx = 0

    for bl in band_lows:
        t_band_start = time.perf_counter()

        # Build baseline cache for this band
        cache = _build_band_cache(df, bl, band_high, n_windows, n_bootstrap_oos)

        for alpha in alphas:
            combo_idx += 1
            t0 = time.perf_counter()
            logger.info(
                f"[{combo_idx}/{total}] band=[{bl:.2f},{band_high:.2f}] a={alpha:+.2f}"
            )

            # Sizing function
            fn = make_sizing_fn(alpha, bl, band_high) if alpha != 0.0 else None

            # Full-sample backtest
            result = _run_lean(df, bl, band_high, alpha, fn, n_bootstrap_full)
            m = _metrics_row(result)

            # OOS validation (skip baseline — already cached)
            if alpha == 0.0:
                # Use cached baseline; OOS is baseline vs itself → win_frac meaningless
                oos = {
                    "oos_n_windows": len(cache.windows),
                    "oos_n_valid": sum(1 for w in cache.window_baselines if not w.get("empty")),
                    "oos_win_frac": 0.5,  # N/A — baseline is reference
                    "oos_mean_sortino_diff": 0.0,
                    "oos_valid": False,
                }
            else:
                oos = _oos_validate(
                    cache, df, bl, band_high, alpha, fn, n_bootstrap_oos,
                )

            elapsed = time.perf_counter() - t0
            gp = GridPoint(
                band_low=bl,
                band_high=band_high,
                alpha=alpha,
                n_trades=m["n_trades"],
                sortino=m["sortino"],
                calmar=m["calmar"],
                total_pnl=m["total_pnl"],
                max_dd_pct=m["max_dd_pct"],
                daily_skew=m["daily_skew"],
                hit_rate=m["hit_rate"],
                expectancy=m["expectancy"],
                n_daily=m["n_daily"],
                n_active_days=m["n_active_days"],
                oos_n_windows=oos["oos_n_windows"],
                oos_n_valid=oos["oos_n_valid"],
                oos_win_frac=oos["oos_win_frac"],
                oos_mean_sortino_diff=oos["oos_mean_sortino_diff"],
                oos_valid=oos["oos_valid"],
                elapsed_s=elapsed,
            )
            results.append(gp)

            if combo_idx % 20 == 0 or combo_idx == total:
                logger.info(
                    f"  progress: {combo_idx}/{total} done, "
                    f"last Sortino={gp.sortino:.3f}, "
                    f"OOS_valid={gp.oos_valid}"
                )

        t_band = time.perf_counter() - t_band_start
        logger.info(f"Band [{bl:.2f},{band_high:.2f}] complete in {t_band:.1f}s")

    return results


# ---------------------------------------------------------------------------
# CSV checkpoint
# ---------------------------------------------------------------------------

GRID_CSV_COLS = [
    "band_low", "band_high", "alpha",
    "n_trades", "sortino", "calmar", "total_pnl", "max_dd_pct",
    "daily_skew", "hit_rate", "expectancy", "n_daily", "n_active_days",
    "oos_n_windows", "oos_n_valid", "oos_win_frac",
    "oos_mean_sortino_diff", "oos_valid", "elapsed_s",
]


def save_checkpoint(results: List[GridPoint], path: Path) -> None:
    """Save intermediate results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GRID_CSV_COLS)
        writer.writeheader()
        for gp in results:
            writer.writerow({col: getattr(gp, col) for col in GRID_CSV_COLS})
    logger.info(f"Checkpoint: {len(results)} rows → {path}")


def load_checkpoint(path: Path) -> List[GridPoint]:
    """Load intermediate results from CSV."""
    if not path.exists():
        return []
    results: List[GridPoint] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gp = GridPoint(
                band_low=float(row["band_low"]),
                band_high=float(row["band_high"]),
                alpha=float(row["alpha"]),
                n_trades=int(row["n_trades"]),
                sortino=float(row["sortino"]),
                calmar=float(row["calmar"]) if row["calmar"] not in ("", "None") else None,
                total_pnl=float(row["total_pnl"]),
                max_dd_pct=float(row["max_dd_pct"]),
                daily_skew=float(row["daily_skew"]),
                hit_rate=float(row["hit_rate"]),
                expectancy=float(row["expectancy"]),
                n_daily=int(row["n_daily"]),
                n_active_days=int(row["n_active_days"]),
                oos_n_windows=int(row["oos_n_windows"]),
                oos_n_valid=int(row["oos_n_valid"]),
                oos_win_frac=float(row["oos_win_frac"]),
                oos_mean_sortino_diff=float(row["oos_mean_sortino_diff"]),
                oos_valid=row["oos_valid"].lower() == "true",
                elapsed_s=float(row["elapsed_s"]),
            )
            results.append(gp)
    return results


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt_calmar(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.3f}"


def _sortino_grade(v: float) -> str:
    if v >= 2.0:
        return "🟢"
    if v >= 1.0:
        return "🟡"
    if v > 0:
        return "🟠"
    return "🔴"


def generate_markdown_report(results: List[GridPoint], output_path: Path) -> str:
    """Generate markdown report from grid sweep results."""
    if not results:
        return "# Grid Sweep: No Results\n\nNo results produced."

    lines: List[str] = []
    band_high = results[0].band_high

    lines.append("# Alpha × Band-Low Grid Sweep Report")
    lines.append("")
    lines.append(f"**Band upper:** {band_high}  ")
    lines.append(f"**Alphas:** {len(set(r.alpha for r in results))} values  ")
    lines.append(f"**Band lows:** {len(set(r.band_low for r in results))} values  ")
    lines.append(f"**Total combos:** {len(results)}  ")
    lines.append(f"**OOS windows:** {results[0].oos_n_windows} per combo  ")
    lines.append(f"**Generated:** {pd.Timestamp.now():%Y-%m-%d %H:%M}  ")
    lines.append("")

    # ── Top 30 by Sortino ──
    lines.append("## Top 30 by Sortino Ratio")
    lines.append("")
    lines.append(
        "| # | Band | α | Trades | Sortino | Calmar | PnL | MaxDD% | Skew | "
        "Hit% | OOS Win% | OOS? |"
    )
    lines.append(
        "|---|------|---|--------|---------|--------|-----|--------|------|"
        "------|----------|------|"
    )

    by_sortino = sorted(results, key=lambda r: r.sortino, reverse=True)
    for i, r in enumerate(by_sortino[:30], 1):
        grade = _sortino_grade(r.sortino)
        oos_str = f"{r.oos_win_frac:.0%}" if r.alpha != 0.0 else "ref"
        oos_val = "✅" if r.oos_valid else ("—" if r.alpha == 0.0 else "❌")
        lines.append(
            f"| {i} | [{r.band_low:.2f},{r.band_high:.2f}] | {r.alpha:+.2f} | "
            f"{r.n_trades} | {grade} {r.sortino:.3f} | {_fmt_calmar(r.calmar)} | "
            f"${r.total_pnl:.2f} | {r.max_dd_pct:.1%} | {r.daily_skew:+.2f} | "
            f"{r.hit_rate:.1%} | {oos_str} | {oos_val} |"
        )
    lines.append("")

    # ── Sortino pivot ──
    lines.append("## Sortino Ratio by Band × Alpha")
    lines.append("")
    alphas_sorted = sorted(set(r.alpha for r in results))
    bands_sorted = sorted(set(r.band_low for r in results))
    alpha_labels = [f"{a:+.2f}" for a in alphas_sorted]

    # Build lookup
    lookup: Dict[Tuple[float, float], GridPoint] = {}
    for r in results:
        lookup[(r.band_low, r.alpha)] = r

    # Header
    lines.append("| Band \\ α | " + " | ".join(alpha_labels) + " |")
    lines.append("|----------|" + "|".join([":------:" for _ in alpha_labels]) + "|")

    for bl in bands_sorted:
        cells = []
        for a in alphas_sorted:
            gp = lookup.get((bl, a))
            if gp and gp.n_trades > 0:
                grade = _sortino_grade(gp.sortino)
                bold = "**" if gp.oos_valid else ""
                end = "**" if gp.oos_valid else ""
                cells.append(f"{grade} {bold}{gp.sortino:.2f}{end}")
            else:
                cells.append("—")
        lines.append(f"| {bl:.2f} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("> 🟢 ≥2.0  🟡 ≥1.0  🟠 >0  🔴 ≤0 — **bold** = OOS valid (≥70% windows beat baseline)")
    lines.append("")

    # ── OOS valid pivot ──
    lines.append("## OOS Validation (win fraction vs a=0 baseline)")
    lines.append("")
    lines.append("| Band \\ α | " + " | ".join(alpha_labels) + " |")
    lines.append("|----------|" + "|".join([":------:" for _ in alpha_labels]) + "|")

    for bl in bands_sorted:
        cells = []
        for a in alphas_sorted:
            gp = lookup.get((bl, a))
            if gp is None:
                cells.append("—")
            elif gp.alpha == 0.0:
                cells.append("ref")
            elif gp.oos_valid:
                cells.append(f"✅ {gp.oos_win_frac:.0%}")
            else:
                cells.append(f"❌ {gp.oos_win_frac:.0%}")
        lines.append(f"| {bl:.2f} | " + " | ".join(cells) + " |")
    lines.append("")

    # ── Best per band ──
    lines.append("## Best Alpha per Band (by Sortino, OOS-priority)")
    lines.append("")
    lines.append(
        "| Band | Best α | Sortino | Calmar | PnL | MaxDD% | Skew | "
        "Trades | OOS Win% | OOS? |"
    )
    lines.append(
        "|------|--------|---------|--------|-----|--------|------|"
        "--------|----------|------|"
    )

    for bl in bands_sorted:
        band_results = [r for r in results if r.band_low == bl and r.n_trades > 0]
        # Sort: OOS-valid first, then Sortino descending
        band_results.sort(key=lambda r: (r.oos_valid, r.sortino), reverse=True)
        if band_results:
            best = band_results[0]
            oos_val = "✅" if best.oos_valid else ("—" if best.alpha == 0.0 else "❌")
            oos_str = f"{best.oos_win_frac:.0%}" if best.alpha != 0.0 else "ref"
            lines.append(
                f"| [{bl:.2f},{band_high:.2f}] | {best.alpha:+.2f} | "
                f"{best.sortino:.3f} | {_fmt_calmar(best.calmar)} | "
                f"${best.total_pnl:.2f} | {best.max_dd_pct:.1%} | "
                f"{best.daily_skew:+.2f} | {best.n_trades} | {oos_str} | {oos_val} |"
            )
    lines.append("")

    # ── Total PnL pivot ──
    lines.append("## Total PnL by Band × Alpha")
    lines.append("")
    lines.append("| Band \\ α | " + " | ".join(alpha_labels) + " |")
    lines.append("|----------|" + "|".join([":------:" for _ in alpha_labels]) + "|")

    for bl in bands_sorted:
        cells = []
        for a in alphas_sorted:
            gp = lookup.get((bl, a))
            if gp and gp.n_trades > 0:
                cells.append(f"${gp.total_pnl:.0f}")
            else:
                cells.append("—")
        lines.append(f"| {bl:.2f} | " + " | ".join(cells) + " |")
    lines.append("")

    # ── Max DD% pivot ──
    lines.append("## Max Drawdown % by Band × Alpha")
    lines.append("")
    lines.append("| Band \\ α | " + " | ".join(alpha_labels) + " |")
    lines.append("|----------|" + "|".join([":------:" for _ in alpha_labels]) + "|")

    for bl in bands_sorted:
        cells = []
        for a in alphas_sorted:
            gp = lookup.get((bl, a))
            if gp and gp.n_trades > 0:
                cells.append(f"{gp.max_dd_pct:.1%}")
            else:
                cells.append("—")
        lines.append(f"| {bl:.2f} | " + " | ".join(cells) + " |")
    lines.append("")

    # ── Summary: top OOS-valid combos ──
    oos_valid_results = [r for r in results if r.oos_valid and r.n_trades > 0]
    if oos_valid_results:
        lines.append("## OOS-Valid Combinations (≥70% windows beat baseline)")
        lines.append("")
        lines.append(
            "| Band | α | Sortino | Calmar | PnL | MaxDD% | Skew | "
            "Trades | OOS Win% |"
        )
        lines.append(
            "|------|---|---------|--------|-----|--------|------|"
            "--------|----------|"
        )
        oos_valid_sorted = sorted(oos_valid_results, key=lambda r: r.sortino, reverse=True)
        for r in oos_valid_sorted:
            lines.append(
                f"| [{r.band_low:.2f},{r.band_high:.2f}] | {r.alpha:+.2f} | "
                f"{r.sortino:.3f} | {_fmt_calmar(r.calmar)} | "
                f"${r.total_pnl:.2f} | {r.max_dd_pct:.1%} | "
                f"{r.daily_skew:+.2f} | {r.n_trades} | {r.oos_win_frac:.0%} |"
            )
        lines.append("")
    else:
        lines.append("## OOS-Valid Combinations")
        lines.append("")
        lines.append("*No combination achieved OOS validation (≥70% windows beating baseline).*")
        lines.append("")

    # ── Metadata ──
    total_elapsed = sum(r.elapsed_s for r in results)
    lines.append("---")
    lines.append("")
    lines.append(f"**Total compute time:** {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    lines.append("")
    lines.append("### Metric Descriptions")
    lines.append("- **Sortino**: Annualized Sortino ratio (downside deviation only). Primary metric.")
    lines.append("- **Calmar**: CAGR / max drawdown. Higher → better risk-adjusted return.")
    lines.append("- **MaxDD%**: Maximum peak-to-trough drawdown as fraction of peak equity.")
    lines.append("- **Skew**: Daily return skewness. Negative = left tail (insurance-like risk).")
    lines.append("- **OOS Win%**: Fraction of walk-forward windows where this α beats baseline (a=0) on Sortino.")
    lines.append("- **OOS Valid**: ≥70% of windows beat baseline AND ≥2 valid windows.")
    lines.append("")
    lines.append("### Caveats")
    lines.append("1. Backtest omits min_book_depth, min_24h_volume, min_total_volume filters.")
    lines.append("2. Entry at first in-band price after time-in-band satisfied; no spread/slippage costs.")
    lines.append("3. Walk-forward uses contiguous calendar windows — regime shifts affect all windows equally.")
    lines.append("4. Sortino is the primary ranking metric; Calmar & PnL are secondary sanity checks.")

    report = "\n".join(lines)

    # Write to file (title="" -- the markdown H1 is already lines[0])
    write_report(output_path, "", [report])
    logger.info(f"Report: {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load data
    print("Loading historical data...")
    df = load_store()
    if df.empty:
        print("No historical data. Run historical fetch first.")
        sys.exit(1)
    print(f"  {len(df)} rows loaded")

    # Sweep parameters
    alphas = [round(x, 2) for x in np.arange(-1.0, 1.01, 0.05).tolist()]
    band_lows = [round(x, 2) for x in np.arange(0.50, 0.92, 0.05).tolist()]
    band_high = 0.95
    n_windows = 4
    n_bootstrap_full = 2000
    n_bootstrap_oos = 1000

    n_combos = len(band_lows) * len(alphas)
    print(f"Alphas: {alphas[0]:+.2f} .. {alphas[-1]:+.2f} ({len(alphas)} values)")
    print(f"Band lows: {band_lows[0]:.2f} .. {band_lows[-1]:.2f} ({len(band_lows)} values)")
    print(f"Band high: {band_high}")
    print(f"Total combos: {n_combos}")
    print(f"OOS windows: {n_windows}")
    est_per_combo = 1.0  # seconds per combo after cache
    est_total = len(band_lows) * (1 + n_windows) * 0.8 + n_combos * est_per_combo
    print(f"Estimated time: ~{est_total:.0f}s ({est_total/60:.0f} min)")
    print()

    # Checkpoint path
    checkpoint_path = Path(__file__).parent.parent / "DATA" / "grid_sweep_checkpoint.csv"

    # Try resume
    results = load_checkpoint(checkpoint_path)
    if results:
        done_keys = {(r.band_low, r.alpha) for r in results}
        remaining = n_combos - len(done_keys)
        print(f"Resuming from checkpoint: {len(results)}/{n_combos} done, {remaining} remaining")
    else:
        done_keys = set()
        print("Starting fresh sweep")

    new_results: List[GridPoint] = []
    combo_idx = 0

    for bl in band_lows:
        # Check if whole band is done
        band_all_done = all((bl, a) in done_keys for a in alphas)
        if band_all_done:
            combo_idx += len(alphas)
            continue

        t_band = time.perf_counter()

        # Build baseline cache for this band (a=0 full + per-window)
        cache = _build_band_cache(df, bl, band_high, n_windows, n_bootstrap_oos)

        for alpha in alphas:
            combo_idx += 1
            if (bl, alpha) in done_keys:
                continue
            t0 = time.perf_counter()

            # Sizing
            fn = make_sizing_fn(alpha, bl, band_high) if alpha != 0.0 else None

            # Full-sample backtest
            result = _run_lean(df, bl, band_high, alpha, fn, n_bootstrap_full)
            m = _metrics_row(result)

            # OOS validation using cached baseline windows
            if alpha == 0.0:
                oos_n_valid = sum(1 for w in cache.window_baselines if not w.get("empty"))
                oos_win_frac = 0.5  # ref — not comparable
                oos_mean_diff = 0.0
                oos_valid = False
            else:
                oos_n_valid = 0
                oos_n_wins = 0
                oos_diffs = []
                for i, (start, end, df_slice) in enumerate(cache.windows):
                    wb = cache.window_baselines[i]
                    if wb.get("empty") or df_slice.empty:
                        continue
                    w_alpha = _run_lean(df_slice, bl, band_high, alpha, fn, n_bootstrap_oos)
                    wa = _metrics_row(w_alpha)
                    diff = wa["sortino"] - wb.get("sortino", 0.0)
                    oos_diffs.append(diff)
                    oos_n_valid += 1
                    if diff > 0:
                        oos_n_wins += 1

                if oos_n_valid == 0:
                    oos_win_frac = 0.0
                    oos_valid = False
                else:
                    oos_win_frac = oos_n_wins / oos_n_valid
                    oos_valid = oos_n_valid >= 2 and oos_win_frac >= 0.70
                oos_mean_diff = float(np.mean(oos_diffs)) if oos_diffs else 0.0

            elapsed = time.perf_counter() - t0
            gp = GridPoint(
                band_low=bl, band_high=band_high, alpha=alpha,
                n_trades=m["n_trades"], sortino=m["sortino"],
                calmar=m["calmar"], total_pnl=m["total_pnl"],
                max_dd_pct=m["max_dd_pct"], daily_skew=m["daily_skew"],
                hit_rate=m["hit_rate"], expectancy=m["expectancy"],
                n_daily=m["n_daily"], n_active_days=m["n_active_days"],
                oos_n_windows=n_windows, oos_n_valid=oos_n_valid,
                oos_win_frac=oos_win_frac,
                oos_mean_sortino_diff=oos_mean_diff,
                oos_valid=oos_valid, elapsed_s=elapsed,
            )
            new_results.append(gp)
            results.append(gp)

            if combo_idx % 10 == 0 or combo_idx == n_combos:
                print(
                    f"  [{combo_idx}/{n_combos}] "
                    f"band=[{bl:.2f},{band_high:.2f}] a={alpha:+.2f} | "
                    f"Sortino={gp.sortino:.3f} | "
                    f"OOS={gp.oos_valid} ({gp.oos_win_frac:.0%}) | "
                    f"{elapsed:.1f}s"
                )

        # Checkpoint after each band completes
        save_checkpoint(results, checkpoint_path)
        new_results.clear()

        t_band_elapsed = time.perf_counter() - t_band
        n_band_done = sum(1 for r in results if r.band_low == bl)
        print(f"  Band [{bl:.2f},{band_high:.2f}] complete: {n_band_done} combos in {t_band_elapsed:.0f}s")

    # Final save
    if new_results:
        save_checkpoint(results, checkpoint_path)

    # Generate report
    report_path = Path(__file__).parent.parent / "DATA" / "grid_sweep_report.md"
    report = generate_markdown_report(results, report_path)
    # Print ASCII-safe (avoid UnicodeEncodeError on cp1252 terminals)
    print(report.replace("α", "alpha"))
    print(f"\nReport saved to: {report_path}")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
