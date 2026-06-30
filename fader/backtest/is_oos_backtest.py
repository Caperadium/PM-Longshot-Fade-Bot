"""backtest/is_oos_backtest.py

IS/OOS validation for top-candidate parameter sets from grid_sweep_findings.md.

Temporal split: first 70% of calendar time = IS, last 30% = OOS.
No parameters are optimised on OOS data — pure hold-out.

Top candidates (from fader/DATA/grid_sweep_findings.md, Pareto frontier):
  1. Max Calmar   [0.50,0.95] DTE[2,5]  a=0.0
  2. Max Sortino  [0.50,0.95] DTE[3,6]  a=0.0
  3. Conservative [0.55,0.95] DTE[0,5]  a=0.0
  4. Balanced     [0.60,0.95] DTE[5,7]  a=0.0
  5. High PF      [0.70,0.95] DTE[5,7]  a=0.0

Run:
  python -c "import sys; sys.path.insert(0,'fader'); from backtest.is_oos_backtest import main; main()"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, run_backtest
from backtest.historical import ContractPriceStore
from backtest.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "DATA"

IS_FRACTION = 0.70   # first 70% of calendar time


# ---------------------------------------------------------------------------
# Candidate definitions
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    label: str
    profile: str
    band_low: float
    band_high: float
    min_dte: int
    max_dte: int
    alpha: float = 0.0


TOP_CANDIDATES: List[Candidate] = [
    Candidate("Max Calmar",   "Highest risk-adjusted (IS Calmar 8.64)",  0.50, 0.95, 2, 5),
    Candidate("Max Sortino",  "Highest total PnL (IS Sortino 8.58)",     0.50, 0.95, 3, 6),
    Candidate("Conservative", "Slight band tighten (IS Calmar 7.76)",    0.55, 0.95, 0, 5),
    Candidate("Balanced",     "Lower drawdown (IS Calmar 6.58)",         0.60, 0.95, 5, 7),
    Candidate("High PF",      "Best PF without degenerate alpha",        0.70, 0.95, 5, 7),
]


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame, is_fraction: float = 0.70) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split df on calendar time. IS = first `is_fraction`, OOS = remainder."""
    dates = pd.to_datetime(df["date"], utc=True, errors="coerce")
    d_min, d_max = dates.min(), dates.max()
    if pd.isna(d_min) or pd.isna(d_max):
        return df, pd.DataFrame(columns=df.columns)

    cutoff = d_min + (d_max - d_min) * is_fraction
    is_mask = dates <= cutoff
    return df[is_mask.values].copy(), df[~is_mask.values].copy()


# ---------------------------------------------------------------------------
# Run one candidate on a dataframe slice
# ---------------------------------------------------------------------------

def _run_candidate(cand: Candidate, df: pd.DataFrame, n_bootstrap: int = 4000) -> Dict:
    if df.empty:
        return {"empty": True}

    cfg = BacktestConfig(
        band_low=cand.band_low,
        band_high=cand.band_high,
        min_dte=cand.min_dte,
        max_dte=cand.max_dte,
        min_time_in_band_days=1,
        order_notional_usd=10.0,
        spread_c=1.0,
        slippage_c=0.0,
        adverse_selection_c=0.0,
        n_bootstrap=n_bootstrap,
    )
    try:
        trades_df, equity_df = run_backtest(df, cfg)
    except Exception as e:
        logger.warning(f"{cand.label}: backtest failed: {e}")
        return {"empty": True, "error": str(e)}

    if trades_df.empty:
        return {"empty": True, "n_trades": 0}

    m = compute_all_metrics(trades_df, n_bootstrap=n_bootstrap)
    return {"empty": False, "metrics": m, "trades_df": trades_df, "equity_df": equity_df}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v, fmt=".2f", none="N/A") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return none
    return format(v, fmt)


def _pct(v, none="N/A") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return none
    return f"{v:.1%}"


def _degrade(is_val: Optional[float], oos_val: Optional[float]) -> str:
    """IS->OOS change. Negative = degradation."""
    if is_val is None or oos_val is None:
        return "N/A"
    if is_val == 0:
        return "--"
    chg = (oos_val - is_val) / abs(is_val)
    sign = "+" if chg >= 0 else ""
    return f"{sign}{chg:.0%}"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _build_report(
    candidates: List[Candidate],
    is_results: List[Dict],
    oos_results: List[Dict],
    is_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    total_df: pd.DataFrame,
    elapsed_s: float,
) -> str:
    lines: List[str] = []

    def _date_range(df: pd.DataFrame) -> str:
        if df.empty:
            return "N/A"
        dates = pd.to_datetime(df["date"], utc=True, errors="coerce").dropna()
        if dates.empty:
            return "N/A"
        return f"{dates.min():%Y-%m-%d} .. {dates.max():%Y-%m-%d}"

    is_range = _date_range(is_df)
    oos_range = _date_range(oos_df)
    total_range = _date_range(total_df)

    lines.append("# IS / OOS Backtest -- Top Candidates")
    lines.append("")
    lines.append(f"**Generated:** {pd.Timestamp.now():%Y-%m-%d %H:%M}  ")
    lines.append(f"**Data:** {total_range} ({len(total_df):,} rows)  ")
    lines.append(f"**IS period:** {is_range} ({len(is_df):,} rows, {IS_FRACTION:.0%} of calendar time)  ")
    lines.append(f"**OOS period:** {oos_range} ({len(oos_df):,} rows, {1-IS_FRACTION:.0%} of calendar time)  ")
    lines.append(f"**Split:** hard temporal cutoff (no lookahead)  ")
    lines.append(f"**Runtime:** {elapsed_s:.0f}s  ")
    lines.append("")
    lines.append("Candidates sourced from `fader/DATA/grid_sweep_findings.md` Pareto frontier.")
    lines.append("All use alpha=0.0 (uniform sizing).")
    lines.append("")

    for i, (cand, is_r, oos_r) in enumerate(zip(candidates, is_results, oos_results)):
        lines.append(f"---")
        lines.append("")
        lines.append(f"## {i+1}. {cand.label}")
        lines.append(f"*{cand.profile}*")
        lines.append("")
        lines.append(
            f"Band [{cand.band_low:.2f}, {cand.band_high:.2f}] | "
            f"DTE [{cand.min_dte}, {cand.max_dte}] | alpha={cand.alpha:.1f}"
        )
        lines.append("")

        is_m = is_r.get("metrics", {})
        oos_m = oos_r.get("metrics", {})
        is_empty = is_r.get("empty", True)
        oos_empty = oos_r.get("empty", True)

        lines.append("| Metric | IS | OOS | IS->OOS |")
        lines.append("|--------|-----|-----|---------|")

        def row(name: str, key: str, fmt: str = ".2f") -> str:
            iv = is_m.get(key) if not is_empty else None
            ov = oos_m.get(key) if not oos_empty else None
            return f"| {name} | {_fmt(iv, fmt)} | {_fmt(ov, fmt)} | {_degrade(iv, ov)} |"

        def row_pct(name: str, key: str) -> str:
            iv = is_m.get(key) if not is_empty else None
            ov = oos_m.get(key) if not oos_empty else None
            return f"| {name} | {_pct(iv)} | {_pct(ov)} | {_degrade(iv, ov)} |"

        def row_int(name: str, key: str) -> str:
            iv = is_m.get(key) if not is_empty else None
            ov = oos_m.get(key) if not oos_empty else None
            is_str = str(int(iv)) if iv is not None else "N/A"
            oos_str = str(int(ov)) if ov is not None else "N/A"
            return f"| {name} | {is_str} | {oos_str} | -- |"

        lines.append(row_int("Trades (N)", "n_trades"))
        lines.append(row("Calmar", "calmar"))
        lines.append(row("Sortino", "sortino"))
        lines.append(row_pct("Hit rate", "hit_rate"))
        # PF = (hit_rate * avg_win) / ((1-hit_rate) * |avg_loss|)
        def row_pf(name: str) -> str:
            def _pf(m: Dict) -> Optional[float]:
                hr = m.get("hit_rate", 0.0)
                aw = m.get("avg_win", 0.0)
                al = abs(m.get("avg_loss", 0.0))
                if al == 0 or hr >= 1.0:
                    return None
                return (hr * aw) / ((1 - hr) * al)
            iv = _pf(is_m) if not is_empty else None
            ov = _pf(oos_m) if not oos_empty else None
            return f"| {name} | {_fmt(iv)} | {_fmt(ov)} | {_degrade(iv, ov)} |"
        lines.append(row_pf("Profit factor"))
        lines.append(row("Expectancy ($/trade)", "expectancy"))
        lines.append(row_pct("Max drawdown", "max_drawdown_pct"))
        lines.append(row("Total PnL ($)", "total_pnl", ".2f"))

        lines.append("")
        if is_empty:
            lines.append("> IS: no trades generated.")
        elif oos_empty:
            lines.append("> OOS: no trades generated -- check DTE/date availability in OOS window.")
        else:
            is_cal = is_m.get("calmar")
            oos_cal = oos_m.get("calmar")
            if is_cal and oos_cal:
                ratio = oos_cal / is_cal if is_cal != 0 else 0
                if ratio >= 0.70:
                    verdict = "PASS -- OOS Calmar retains >=70% of IS Calmar."
                elif ratio >= 0.40:
                    verdict = "MARGINAL -- OOS Calmar retains 40-70% of IS Calmar."
                else:
                    verdict = "FAIL -- OOS Calmar <40% of IS. Likely overfit or regime change."
                lines.append(f"> **{verdict}** (OOS/IS Calmar = {ratio:.0%})")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Summary Comparison")
    lines.append("")
    lines.append(
        "| Candidate | IS Cal | IS Sort | IS N | OOS Cal | OOS Sort | OOS N | "
        "Calmar retain | Verdict |"
    )
    lines.append(
        "|-----------|--------|---------|------|---------|----------|-------|"
        "---------------|---------|"
    )

    for cand, is_r, oos_r in zip(candidates, is_results, oos_results):
        is_m = is_r.get("metrics", {})
        oos_m = oos_r.get("metrics", {})
        is_empty = is_r.get("empty", True)
        oos_empty = oos_r.get("empty", True)

        is_cal = is_m.get("calmar") if not is_empty else None
        oos_cal = oos_m.get("calmar") if not oos_empty else None
        is_sort = is_m.get("sortino") if not is_empty else None
        oos_sort = oos_m.get("sortino") if not oos_empty else None
        is_n = int(is_m.get("n_trades", 0)) if not is_empty else 0
        oos_n = int(oos_m.get("n_trades", 0)) if not oos_empty else 0

        if is_cal and oos_cal:
            ratio = oos_cal / is_cal
            if ratio >= 0.70:
                ret_str = f"{ratio:.0%}"
                verdict = "PASS"
            elif ratio >= 0.40:
                ret_str = f"{ratio:.0%}"
                verdict = "MARGINAL"
            else:
                ret_str = f"{ratio:.0%}"
                verdict = "FAIL"
        else:
            ret_str = "N/A"
            verdict = "no data"

        lines.append(
            f"| {cand.label} | {_fmt(is_cal)} | {_fmt(is_sort)} | {is_n} | "
            f"{_fmt(oos_cal)} | {_fmt(oos_sort)} | {oos_n} | "
            f"{ret_str} | {verdict} |"
        )

    lines.append("")
    lines.append("**Calmar retain**: OOS Calmar / IS Calmar. >=70% = PASS, 40-70% = MARGINAL, <40% = FAIL.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "1. **Hard temporal split (no shuffle).** IS = earlier data, OOS = later data. "
        "Matches real deployment -- bot never sees future data."
    )
    lines.append(
        "2. **OOS is a single hold-out window.** One 30% slice is not enough to distinguish "
        "luck from skill; use walkforward.py for per-window stability."
    )
    lines.append(
        "3. **OOS N may be lower** because DTE-filtered strategies require contracts "
        "with sufficient days remaining."
    )
    lines.append(
        "4. **No bootstrap CIs here** (speed). Run compute_all_metrics(n_bootstrap=10000) for CI bands."
    )
    lines.append(
        "5. **BTC only.** Results may not generalize to Seoul temperature or other slugs."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("Loading historical data...")
    store = ContractPriceStore()
    df = store.snapshot()
    if df.empty:
        print("No historical data. Run historical fetch first.")
        return

    print(f"  {len(df):,} rows loaded")
    dates = pd.to_datetime(df["date"], utc=True, errors="coerce").dropna()
    print(f"  Date range: {dates.min():%Y-%m-%d} .. {dates.max():%Y-%m-%d}")

    is_df, oos_df = temporal_split(df, IS_FRACTION)
    is_dates = pd.to_datetime(is_df["date"], utc=True, errors="coerce").dropna()
    oos_dates = pd.to_datetime(oos_df["date"], utc=True, errors="coerce").dropna()
    print(f"  IS:  {len(is_df):,} rows  {is_dates.min():%Y-%m-%d} .. {is_dates.max():%Y-%m-%d}")
    print(f"  OOS: {len(oos_df):,} rows  {oos_dates.min():%Y-%m-%d} .. {oos_dates.max():%Y-%m-%d}")
    print()

    t0 = time.perf_counter()
    is_results: List[Dict] = []
    oos_results: List[Dict] = []

    for i, cand in enumerate(TOP_CANDIDATES, 1):
        print(
            f"[{i}/{len(TOP_CANDIDATES)}] {cand.label}: "
            f"band=[{cand.band_low:.2f},{cand.band_high:.2f}] "
            f"DTE=[{cand.min_dte},{cand.max_dte}] a={cand.alpha:.1f}"
        )
        t_cand = time.perf_counter()
        is_r = _run_candidate(cand, is_df, n_bootstrap=4000)
        oos_r = _run_candidate(cand, oos_df, n_bootstrap=4000)
        elapsed = time.perf_counter() - t_cand

        is_m = is_r.get("metrics", {})
        oos_m = oos_r.get("metrics", {})
        is_cal = is_m.get("calmar")
        oos_cal = oos_m.get("calmar")
        retain = f"{oos_cal/is_cal:.0%}" if (is_cal and oos_cal) else "N/A"

        print(
            f"  IS:  N={is_m.get('n_trades',0)}  "
            f"Calmar={_fmt(is_cal)}  Sortino={_fmt(is_m.get('sortino'))}  "
            f"Hit={_pct(is_m.get('hit_rate'))}  MaxDD={_pct(is_m.get('max_drawdown_pct'))}"
        )
        print(
            f"  OOS: N={oos_m.get('n_trades',0)}  "
            f"Calmar={_fmt(oos_cal)}  Sortino={_fmt(oos_m.get('sortino'))}  "
            f"Hit={_pct(oos_m.get('hit_rate'))}  MaxDD={_pct(oos_m.get('max_drawdown_pct'))}"
        )
        print(f"  Calmar retain: {retain}  ({elapsed:.1f}s)")
        print()

        is_results.append(is_r)
        oos_results.append(oos_r)

    elapsed_total = time.perf_counter() - t0
    print(f"Done in {elapsed_total:.1f}s")

    report = _build_report(
        TOP_CANDIDATES, is_results, oos_results,
        is_df, oos_df, df, elapsed_total,
    )

    out_path = DATA_DIR / "is_oos_report.md"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport -> {out_path}")

    print()
    for line in report.split("\n"):
        if "## Summary" in line or line.startswith("|"):
            print(line)


if __name__ == "__main__":
    main()
