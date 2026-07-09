"""backtest/harness.py

Shared backtest harness: one place for the run/collect plumbing that all
five backtest CLIs (band_sweep, allocation_analysis, grid_sweep,
is_oos_backtest, crypto_sweep) previously duplicated.

Phase 5 of temp/implementation-plan.md. Behavior-preserving: no formula
changes, no numeric changes. See "Phase 5 -- Backtest harness
consolidation" for the full design rationale, including why THREE
walk-forward variants are kept as separate functions rather than merged
into one (they are mathematically different -- see each function's
docstring below).

CLIs keep their explicit parameter choices (band ranges, alpha grids, DTE
ranges, bootstrap counts) -- this module only centralizes the mechanical
plumbing: loading a store, running one config + extracting metrics, and
running a grid of configs with multiprocessing.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from io import StringIO
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, run_backtest
from backtest.historical import ContractPriceStore
from backtest.metrics import compute_all_metrics
from backtest.walkforward import partition_calendar_windows, window_summary
from execution.sizing import make_sizing_fn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessDefaults:
    """Shared default parameters across CLIs.

    Individual CLIs override fields via dataclasses.replace() when their
    explicit parameter choice diverges from these defaults (e.g.
    crypto_sweep's CRYPTO_DEFAULTS uses order_notional_usd=25.0) -- the
    divergence stays named and greppable at the CLI call site rather than
    being silently baked into shared code.
    """
    order_notional_usd: float = 10.0
    n_bootstrap: int = 5000
    band_high: float = 0.95
    alpha_step: float = 0.05


# ---------------------------------------------------------------------------
# Store loading
# ---------------------------------------------------------------------------


def load_store(
    slug_filter: Optional[str] = None,
    slugs: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load the historical ContractPriceStore and return a snapshot DataFrame.

    Args:
        slug_filter: if given, keep only rows whose ``slug`` contains this
            substring (mirrors the ``slug_filter`` parameter used by
            ``crypto_sweep``/``band_sweep``'s ``sweep_grid``).
        slugs: if given, keep only rows whose ``slug`` is exactly in this
            set (mirrors ``run_backtest``'s own ``slugs=`` filter).

    Returns:
        A snapshot DataFrame (detached from the mutable store, per
        ``ContractPriceStore.snapshot()``). Empty DataFrame if no data.
    """
    store = ContractPriceStore()
    df = store.snapshot()
    if df.empty:
        return df
    if slug_filter:
        df = df[df["slug"].str.contains(slug_filter, na=False)]
    if slugs:
        slug_set = set(slugs)
        df = df[df["slug"].isin(slug_set)]
    return df


# ---------------------------------------------------------------------------
# Single-config run + metrics extraction
# ---------------------------------------------------------------------------


@dataclass
class MetricsRow:
    """Flat extraction of run_backtest + compute_all_metrics for one config.

    Superset of the fields the five CLIs' own per-config dataclasses pull
    out of the metrics dict (GridResult, GridPoint, BandResult, etc. still
    exist separately -- they add sweep-specific fields like band/alpha/DTE
    that this generic row does not know about). ``metrics`` holds the full
    compute_all_metrics() dict for anything not promoted to a named field.
    """
    n_trades: int
    total_pnl: float
    sortino: float
    calmar: Optional[float]
    max_drawdown_pct: float
    hit_rate: float
    expectancy: float
    daily_skew: float
    daily_cvar_95: float
    n_daily: int
    n_active_days: int
    pnl_ci_95: Tuple[float, float]
    elapsed_ms: float
    empty: bool
    metrics: Dict = field(default_factory=dict)
    trades_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def _empty_metrics_row() -> MetricsRow:
    return MetricsRow(
        n_trades=0, total_pnl=0.0, sortino=0.0, calmar=None,
        max_drawdown_pct=0.0, hit_rate=0.0, expectancy=0.0,
        daily_skew=0.0, daily_cvar_95=0.0, n_daily=0, n_active_days=0,
        pnl_ci_95=(0.0, 0.0), elapsed_ms=0.0, empty=True,
        metrics={}, trades_df=pd.DataFrame(),
    )


def run_config(
    store_or_df,
    cfg: BacktestConfig,
    slugs: Optional[List[str]] = None,
    initial_capital: float = 500.0,
) -> MetricsRow:
    """Run one BacktestConfig and extract a MetricsRow.

    Equivalent to ``run_backtest(store_or_df, cfg) followed by
    compute_all_metrics(trades_df, n_bootstrap=cfg.n_bootstrap)`` --
    this function changes no formulas, only collects the two calls plus
    the flat-field extraction that every CLI repeated by hand.
    """
    try:
        trades_df, _equity_df = run_backtest(store_or_df, cfg, slugs=slugs)
    except Exception as e:
        logger.warning(f"run_config: backtest failed: {e}")
        return _empty_metrics_row()

    if trades_df.empty:
        return _empty_metrics_row()

    skipped_filters = trades_df.attrs.get("skipped_filters")
    m = compute_all_metrics(
        trades_df,
        n_bootstrap=cfg.n_bootstrap,
        initial_capital=initial_capital,
        skipped_filters=skipped_filters,
    )
    if not m:
        return _empty_metrics_row()

    pnl_ci = m.get("pnl_ci_95", (0.0, 0.0))
    return MetricsRow(
        n_trades=m.get("n_trades", 0),
        total_pnl=m.get("total_pnl", 0.0),
        sortino=m.get("sortino", 0.0) or 0.0,
        calmar=m.get("calmar"),
        max_drawdown_pct=m.get("max_drawdown_pct", 0.0),
        hit_rate=m.get("hit_rate", 0.0),
        expectancy=m.get("expectancy", 0.0),
        daily_skew=m.get("daily_skew", 0.0),
        daily_cvar_95=m.get("daily_cvar_95", 0.0),
        n_daily=m.get("n_daily", 0),
        n_active_days=m.get("n_active_days", 0),
        pnl_ci_95=pnl_ci if isinstance(pnl_ci, tuple) else (0.0, 0.0),
        elapsed_ms=m.get("elapsed_ms", 0.0),
        empty=False,
        metrics=m,
        trades_df=trades_df,
    )


# ---------------------------------------------------------------------------
# Grid runner (spawn-safe multiprocessing)
# ---------------------------------------------------------------------------

# NOTE on multiprocessing pattern: this copies crypto_sweep.py's existing
# spawn-safe pattern verbatim (module-level worker fn, df_json serialized
# and passed INSIDE each worker's args tuple). There is no pool initializer
# today -- do not add one; ProcessPoolExecutor workers reconstruct the
# DataFrame and re-import backtest.engine/execution.sizing fresh per call,
# exactly as crypto_sweep._run_chunk did.


def _run_grid_worker(args: Tuple[str, List[BacktestConfig], Optional[List[str]], float]) -> List[Dict]:
    """Top-level (picklable) worker: run a chunk of configs on one market df.

    args = (df_json, configs_chunk, slugs, initial_capital)
    where df_json is the DataFrame serialized via ``to_json(orient="table")``
    (mirrors crypto_sweep.py:516,184).
    """
    df_json, configs_chunk, slugs, initial_capital = args
    market_df = pd.read_json(StringIO(df_json), orient="table")

    out: List[Dict] = []
    for cfg in configs_chunk:
        row = run_config(market_df, cfg, slugs=slugs, initial_capital=initial_capital)
        out.append({"cfg": cfg, "row": row})
    return out


def run_grid(
    store_or_df,
    grid: List[BacktestConfig],
    workers: int = 1,
    slugs: Optional[List[str]] = None,
    initial_capital: float = 500.0,
) -> List[MetricsRow]:
    """Run a list of BacktestConfigs, optionally in parallel.

    ``workers=1`` (default) runs sequentially in-process -- no
    multiprocessing overhead, used by tests and small sweeps.
    ``workers>1`` splits ``grid`` into ``workers`` chunks and dispatches
    them to a ProcessPoolExecutor, mirroring crypto_sweep's existing
    pattern (df serialized to JSON, passed inside each chunk's args --
    avoids re-pickling the full DataFrame into every worker submission
    individually).

    Returns MetricsRow list in the SAME ORDER as ``grid`` (parallel path
    preserves order by chunking positionally and concatenating results in
    chunk order, not completion order).
    """
    if isinstance(store_or_df, ContractPriceStore):
        df = store_or_df.snapshot()
    elif isinstance(store_or_df, pd.DataFrame):
        df = store_or_df
    else:
        raise TypeError(f"Expected ContractPriceStore or DataFrame, got {type(store_or_df)}")

    if workers <= 1 or len(grid) <= 1:
        return [run_config(df, cfg, slugs=slugs, initial_capital=initial_capital) for cfg in grid]

    # A cfg carrying a sizing_fn (make_sizing_fn returns a local closure)
    # cannot be pickled into a worker process; fall back to sequential
    # rather than crash at ex.submit.
    if any(getattr(cfg, "sizing_fn", None) is not None for cfg in grid):
        logger.warning(
            "run_grid: grid contains sizing_fn closures (unpicklable); "
            "running sequentially instead of with %d workers", workers
        )
        return [run_config(df, cfg, slugs=slugs, initial_capital=initial_capital) for cfg in grid]

    from concurrent.futures import ProcessPoolExecutor

    df_json = df.to_json(orient="table")
    n_workers = max(1, min(workers, len(grid)))
    chunks = np.array_split(np.arange(len(grid)), n_workers)
    chunks = [c.tolist() for c in chunks if len(c) > 0]

    results: List[Optional[MetricsRow]] = [None] * len(grid)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {}
        for idx_chunk in chunks:
            cfg_chunk = [grid[i] for i in idx_chunk]
            fut = ex.submit(_run_grid_worker, (df_json, cfg_chunk, slugs, initial_capital))
            futures[fut] = idx_chunk
        for fut, idx_chunk in futures.items():
            batch = fut.result()
            for pos, item in zip(idx_chunk, batch):
                results[pos] = item["row"]

    return [r if r is not None else _empty_metrics_row() for r in results]


# ---------------------------------------------------------------------------
# Walk-forward: THREE distinct methods (see plan -- NOT interchangeable)
# ---------------------------------------------------------------------------
#
# walkforward_normalized: allocation_analysis's Path A/B comparison, using
#   NORMALIZED sizing (_make_sized_fn wraps make_sizing_fn with a global
#   norm_factor from compute_normalization_factor so mean multiplier ~= 1
#   across the whole alpha sweep). Compares a=a* (or a=+1) vs a=0 (or a=-1)
#   per calendar window and reports a win-fraction + OOS-valid flag.
#
# walkforward_lean: grid_sweep's per-(band,alpha) OOS check, using
#   UNNORMALIZED make_sizing_fn directly, with per-band baseline windows
#   pre-computed once via _build_band_cache and reused across every alpha
#   in that band (this is the "lean" performance optimization -- baseline
#   windows are NOT re-run for every alpha).
#
# window_stability: crypto_sweep's per-config walk-forward re-test of the
#   top-N configs from a grid sweep, using the SAME FIXED config in every
#   window (no baseline comparison at all -- reports cross-window
#   consistency via walkforward.window_summary's stability grade).
#
# All three share only calendar-window partitioning (partition_calendar_
# windows) and this module's run/collect plumbing. Their formulas and
# comparison semantics differ and MUST NOT be merged -- doing so would
# change the numbers in existing goldens / test_allocation_analysis.py.


def walkforward_normalized(
    store_or_df,
    base_cfg: BacktestConfig,
    alpha_star: Optional[float],
    alphas: List[float],
    band_low: float,
    band_high: float,
    n_windows: int = 4,
) -> Dict:
    """= allocation_analysis.walkforward_validate (normalized sizing).

    Path A (alpha_star is not None): compare a=a* vs a=0 per window.
    Path B (alpha_star is None): compare a=+1 vs a=-1 per window.
    Sizing uses the GLOBAL normalization factor (mean multiplier ~= 1
    across the full alpha sweep) via compute_normalization_factor +
    _make_sized_fn -- moved here verbatim from allocation_analysis.py.
    """
    # Deferred import: avoids a hard circular import at module load time
    # (allocation_analysis imports harness for run_config/load_store; this
    # function is allocation_analysis's own walkforward_validate moved
    # here, but still needs its analysis-only helpers).
    from backtest.allocation_analysis import (
        _make_sized_fn,
        _run_scheme,
        compute_normalization_factor,
        extract_entry_prices,
    )

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


@dataclass
class BandCache:
    """= grid_sweep.BandCache. Pre-computed per-band baseline (a=0), full
    sample + per-window, reused across every alpha tested in that band."""
    band_low: float
    band_high: float
    baseline_metrics: Dict
    window_baselines: List[Dict]
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]


def _run_lean(
    df: pd.DataFrame,
    band_low: float,
    band_high: float,
    alpha: float,
    sizing_fn: Optional[Callable[[float], float]],
    n_bootstrap: int = 1000,
    order_notional_usd: float = 10.0,
    min_time_in_band_days: int = 1,
    spread_c: float = 1.0,
) -> Dict:
    """= grid_sweep._run_lean. Run backtest + compute_all_metrics for one
    scheme with minimal allocation (no paired CIs, no per-bin)."""
    cfg = BacktestConfig(
        band_low=band_low,
        band_high=band_high,
        min_time_in_band_days=min_time_in_band_days,
        order_notional_usd=order_notional_usd,
        spread_c=spread_c,
        slippage_c=0.0,
        adverse_selection_c=0.0,
        n_bootstrap=n_bootstrap,
        sizing_fn=sizing_fn,
    )
    try:
        trades_df, _ = run_backtest(df, cfg)
    except Exception as e:
        logger.warning(f"band=[{band_low:.2f},{band_high:.2f}] a={alpha:+.2f}: backtest failed: {e}")
        return {"empty": True, "n_trades": 0}

    if trades_df.empty:
        return {"empty": True, "n_trades": 0}

    m = compute_all_metrics(trades_df, n_bootstrap=n_bootstrap)
    return {"empty": False, "trades_df": trades_df, "metrics": m}


def _lean_metrics_row(result: Dict) -> Dict:
    """= grid_sweep._metrics_row. Flat metrics dict from _run_lean result."""
    if result.get("empty"):
        return {
            "n_trades": 0, "sortino": 0.0, "calmar": None,
            "total_pnl": 0.0, "max_dd_pct": 0.0, "daily_skew": 0.0,
            "hit_rate": 0.0, "expectancy": 0.0,
            "n_daily": 0, "n_active_days": 0,
        }
    m = result["metrics"]
    return {
        "n_trades": m.get("n_trades", 0),
        "sortino": m.get("sortino", 0.0) or 0.0,
        "calmar": m.get("calmar"),
        "total_pnl": m.get("total_pnl", 0.0),
        "max_dd_pct": m.get("max_drawdown_pct", 0.0),
        "daily_skew": m.get("daily_skew", 0.0),
        "hit_rate": m.get("hit_rate", 0.0),
        "expectancy": m.get("expectancy", 0.0),
        "n_daily": m.get("n_daily", 0),
        "n_active_days": m.get("n_active_days", 0),
    }


def build_band_cache(
    df: pd.DataFrame,
    band_low: float,
    band_high: float,
    n_windows: int,
    n_bootstrap: int,
    order_notional_usd: float = 10.0,
    min_time_in_band_days: int = 1,
    spread_c: float = 1.0,
) -> BandCache:
    """= grid_sweep._build_band_cache. Pre-compute baseline (a=0)
    full-sample and per-window for one band."""
    base_result = _run_lean(
        df, band_low, band_high, 0.0, None, n_bootstrap,
        order_notional_usd=order_notional_usd,
        min_time_in_band_days=min_time_in_band_days, spread_c=spread_c,
    )
    base_metrics = _lean_metrics_row(base_result)

    windows = partition_calendar_windows(df, n_windows=n_windows)
    window_baselines: List[Dict] = []
    for start, end, df_slice in windows:
        label = f"{start:%Y-%m-%d}..{end:%Y-%m-%d}"
        if df_slice.empty:
            window_baselines.append({"label": label, "empty": True})
            continue
        w_result = _run_lean(
            df_slice, band_low, band_high, 0.0, None, n_bootstrap,
            order_notional_usd=order_notional_usd,
            min_time_in_band_days=min_time_in_band_days, spread_c=spread_c,
        )
        w_metrics = _lean_metrics_row(w_result)
        w_metrics["label"] = label
        window_baselines.append(w_metrics)

    return BandCache(
        band_low=band_low,
        band_high=band_high,
        baseline_metrics=base_metrics,
        window_baselines=window_baselines,
        windows=windows,
    )


def walkforward_lean(
    cache: BandCache,
    df: pd.DataFrame,
    band_low: float,
    band_high: float,
    alpha: float,
    sizing_fn: Callable[[float], float],
    n_bootstrap: int,
    order_notional_usd: float = 10.0,
    min_time_in_band_days: int = 1,
    spread_c: float = 1.0,
) -> Dict:
    """= grid_sweep._oos_validate. Walk-forward OOS: compare alpha vs the
    band's cached baseline (a=0) in each window. UNNORMALIZED sizing_fn
    (make_sizing_fn directly, no global norm_factor) -- distinct from
    walkforward_normalized's normalized comparison."""
    n_valid = 0
    n_wins = 0
    sortino_diffs: List[float] = []

    for i, (start, end, df_slice) in enumerate(cache.windows):
        wb = cache.window_baselines[i]
        if wb.get("empty") or df_slice.empty:
            continue

        w_result = _run_lean(
            df_slice, band_low, band_high, alpha, sizing_fn, n_bootstrap,
            order_notional_usd=order_notional_usd,
            min_time_in_band_days=min_time_in_band_days, spread_c=spread_c,
        )
        w_metrics = _lean_metrics_row(w_result)

        base_sortino = wb.get("sortino", 0.0)
        alpha_sortino = w_metrics.get("sortino", 0.0)
        diff = alpha_sortino - base_sortino

        sortino_diffs.append(diff)
        n_valid += 1
        if diff > 0:
            n_wins += 1

    if n_valid == 0:
        return {
            "oos_n_windows": len(cache.windows),
            "oos_n_valid": 0,
            "oos_win_frac": 0.0,
            "oos_mean_sortino_diff": 0.0,
            "oos_valid": False,
        }

    win_frac = n_wins / n_valid
    mean_diff = float(np.mean(sortino_diffs))
    oos_valid = n_valid >= 2 and win_frac >= 0.70

    return {
        "oos_n_windows": len(cache.windows),
        "oos_n_valid": n_valid,
        "oos_win_frac": win_frac,
        "oos_mean_sortino_diff": mean_diff,
        "oos_valid": oos_valid,
    }


def window_stability(
    df_out: pd.DataFrame,
    market_df: pd.DataFrame,
    top_n: int,
    n_windows: int,
    n_bootstrap: int,
    order_notional_usd: float,
    initial_capital: float = 1000.0,
    min_time_in_band_days: int = 1,
    spread_c: float = 1.0,
    band_high: float = 0.95,
) -> List[Dict]:
    """= crypto_sweep._run_walkforward_for_top_configs. Re-test top_n
    configs (by score, df_out already sorted) across calendar windows
    with the SAME FIXED config per window (no baseline comparison --
    stability comes from walkforward.window_summary's cross-window
    consistency grade, not a win-fraction vs a reference scheme)."""
    windows = partition_calendar_windows(market_df, n_windows=n_windows)
    out = []
    for _, row in df_out.head(top_n).iterrows():
        band_low = float(row["band_low"])
        alpha = float(row["alpha"])
        min_dte = int(row["min_dte"])
        max_dte = int(row["max_dte"])

        sizing_fn = make_sizing_fn(alpha, band_low, band_high)
        cfg = BacktestConfig(
            band_low=band_low,
            band_high=band_high,
            min_dte=min_dte,
            max_dte=max_dte,
            min_time_in_band_days=min_time_in_band_days,
            order_notional_usd=order_notional_usd,
            spread_c=spread_c,
            sizing_fn=sizing_fn,
            n_bootstrap=n_bootstrap,
        )
        try:
            per_window, stability = window_summary(
                windows, cfg,
                n_bootstrap=n_bootstrap,
                initial_capital=initial_capital,
            )
        except Exception as e:
            logger.warning(f"walkforward failed for band_low={band_low} alpha={alpha}: {e}")
            continue

        out.append({
            "band_low": band_low,
            "alpha": alpha,
            "min_dte": min_dte,
            "max_dte": max_dte,
            "is_score": row["score"],
            "per_window": per_window,
            "stability": stability,
        })
    return out
