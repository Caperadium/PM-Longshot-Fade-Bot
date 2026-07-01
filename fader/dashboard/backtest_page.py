"""dashboard/backtest_page.py

Streamlit backtest page.

Can be used two ways:

1. Embedded as a tab inside app.py:
       from dashboard import backtest_page
       with tab_backtest:
           backtest_page.render()

2. Run standalone:
       streamlit run fader/dashboard/backtest_page.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_FADER_ROOT = Path(__file__).resolve().parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))

from backtest.historical import (
    ContractPriceStore,
    fetch_and_store,
    PRICES_CSV,
)
from backtest.engine import BacktestConfig, run_backtest
from backtest.metrics import compute_all_metrics
from config.config_loader import load_config
from execution.sizing import make_sizing_fn


def _expand_series_slugs(
    slugs: list[str],
    store: ContractPriceStore,
    slug_configs: dict,
) -> list[str]:
    """Expand series slugs (e.g. bitcoin-above-on) to their individual market slugs.

    Series slugs are virtual identifiers that map to many individual markets.
    The backtest engine needs the real market slugs that exist in the store.
    """
    stored = set(store.all_slugs())
    expanded: list[str] = []
    for s in slugs:
        cfg = slug_configs.get(s)
        kind = getattr(cfg, "market_kind", "binary") if cfg else "binary"
        if kind in ("series", "btc_daily"):
            # Derive filter keyword (same logic as fetch_and_store)
            if kind == "btc_daily":
                series_filter = "bitcoin-above"
            elif cfg and cfg.series_filter:
                series_filter = cfg.series_filter
            else:
                # Fallback: strip trailing '-on'
                series_filter = s[:-3] if s.endswith("-on") else s
            # Find all stored slugs matching this series
            series_markets = [
                x for x in stored
                if x.startswith(series_filter + "-")
            ]
            if series_markets:
                expanded.extend(series_markets)
            else:
                # No data fetched yet — keep the virtual slug. Backtest
                # will find 0 rows and produce 0 trades (honest zero
                # rather than silently falling back to all data).
                expanded.append(s)
        elif s in stored:
            expanded.append(s)
        # Silently drop config slugs that have no stored data
    return expanded


def render(embedded: bool = True) -> None:
    """Render the backtest UI.

    When ``embedded`` is True the settings live in an expander inside the
    current container (so they don't collide with the engine controls that
    already own the sidebar in app.py). When False (standalone), settings go
    in the sidebar.
    """
    cfg = load_config()
    store = ContractPriceStore(PRICES_CSV)

    st.header("Backtest")

    stored_slugs = store.all_slugs()
    config_slugs = [s.slug for s in cfg.enabled_slugs()]

    # Build set of series-child prefixes so individual market slugs
    # (e.g. bitcoin-above-100k-on-january-1) are excluded from the
    # dropdown — only the virtual series slug is shown.
    series_prefixes: set[str] = set()
    for s in cfg.slugs:
        kind = s.market_kind
        if kind in ("series", "btc_daily"):
            sf = (s.series_filter if s.series_filter
                  else (s.slug[:-3] if s.slug.endswith("-on") else s.slug))
            series_prefixes.add(sf + "-")

    # Only show config slugs + non-series stored slugs (no child markets)
    non_series_stored = [
        s for s in stored_slugs
        if not any(s.startswith(prefix) for prefix in series_prefixes)
    ]
    available_slugs = non_series_stored + [
        s for s in config_slugs if s not in non_series_stored
    ]
    # Default to config slugs that have stored data, else fall back to
    # config slugs so a fetch can be triggered immediately.
    enabled_defaults = [s for s in config_slugs if s in stored_slugs] or config_slugs

    # Settings container: expander when embedded, sidebar when standalone.
    settings = st.expander("Backtest Settings", expanded=True) if embedded else st.sidebar
    if not embedded:
        settings.header("Backtest Settings")

    with settings:
        selected_slugs = st.multiselect(
            "Slugs to backtest",
            options=available_slugs,
            default=enabled_defaults,
            key="bt_slugs",
        )

        col1, col2 = st.columns(2)
        with col1:
            band_low = st.slider(
                "Band Low", 0.50, 0.99, float(cfg.strategy.band_low), 0.01,
                key="bt_band_low",
            )
            min_dte = st.number_input(
                "Min DTE", 0, 365, cfg.strategy.min_dte, key="bt_min_dte"
            )
            notional = st.number_input(
                "Order Notional (USD)", 1.0, 1000.0,
                float(cfg.strategy.order_notional_usd), key="bt_notional",
            )
            alpha = st.slider(
                "Alpha (tilt)", -1.0, 1.0,
                cfg.strategy.alpha, 0.1, key="bt_alpha",
                help="-1 = heavy low-price, 0 = uniform, +1 = heavy high-price"
            )
            slippage_c = st.number_input(
                "Slippage (cents)", 0.0, 10.0,
                float(BacktestConfig.slippage_c), key="bt_slippage",
            )
        with col2:
            band_high = st.slider(
                "Band High", 0.50, 0.99, float(cfg.strategy.band_high), 0.01,
                key="bt_band_high",
            )
            max_dte = st.number_input(
                "Max DTE", 0, 3650, cfg.strategy.max_dte, key="bt_max_dte"
            )
            min_time_days = st.number_input(
                "Min Days in Band", 0, 30,
                max(1, cfg.strategy.min_time_in_band_s // 86400),
                key="bt_min_time_days",
            )
            adv_sel_c = st.number_input(
                "Adverse Selection (cents)", 0.0, 10.0,
                float(BacktestConfig.adverse_selection_c), key="bt_adv_sel",
            )

        n_bootstrap = st.number_input(
            "Bootstrap samples", 1000, 50000, BacktestConfig.n_bootstrap,
            key="bt_n_bootstrap",
        )
        initial_capital = st.number_input(
            "Initial Capital ($)", 0.0, 100000.0, 500.0, 100.0,
            key="bt_initial_capital",
            help="Added to the equity curve before computing drawdown % "
                 "and Calmar. Does not affect Sortino (scale-invariant). "
                 "$500 covers ~20 concurrent $10-notional positions.",
        )

        if st.button("Fetch/Refresh Historical Prices", key="bt_fetch"):
            slugs_to_fetch = selected_slugs or [s.slug for s in cfg.enabled_slugs()]
            slug_configs = {s.slug: s for s in cfg.slugs}

            # Live progress log: each callback line is appended and shown in a
            # scrollable status box so long discovery/fetch runs are visible
            # (and hangs are obvious). Also prints to the CLI via _default_progress.
            status = st.status("Fetching historical prices...", expanded=True)
            log_lines: list[str] = []

            def _on_progress(msg: str) -> None:
                print(f"[fetch] {msg}", flush=True)
                log_lines.append(msg)
                status.update(label=msg)
                status.write(msg)

            try:
                store = fetch_and_store(
                    slugs_to_fetch, store=store, refresh_existing=True,
                    slug_configs=slug_configs, progress=_on_progress,
                )
                status.update(label="Fetch complete", state="complete")
            except Exception as e:  # noqa: BLE001 — surface any fetch error to UI
                status.update(label=f"Fetch failed: {e}", state="error")
                raise
            st.success(f"Fetched data for {len(slugs_to_fetch)} slugs")

    # Run backtest
    if st.button("Run Backtest", type="primary", key="bt_run"):
        if band_low >= band_high:
            st.error(f"Band Low ({band_low}) must be less than Band High ({band_high}).")
        else:
            sizing_fn = None
            if alpha != 0.0:
                sizing_fn = make_sizing_fn(alpha, band_low, band_high)
            bt_cfg = BacktestConfig(
                band_low=band_low,
                band_high=band_high,
                min_dte=int(min_dte),
                max_dte=int(max_dte),
                min_time_in_band_days=int(min_time_days),
                order_notional_usd=float(notional),
                slippage_c=float(slippage_c),
                adverse_selection_c=float(adv_sel_c),
                n_bootstrap=int(n_bootstrap),
                sizing_fn=sizing_fn,
            )
            with st.spinner("Running backtest..."):
                trades_df, equity_df = run_backtest(
                    store, bt_cfg, slugs=_expand_series_slugs(selected_slugs, store, {s.slug: s for s in cfg.slugs})
                )

            # Persist results in session_state so the results display and
            # walk-forward section survive across reruns.
            st.session_state.bt_trades_df = trades_df
            st.session_state.bt_equity_df = equity_df
            st.session_state.bt_cfg = bt_cfg
            st.session_state.bt_has_results = not trades_df.empty

            if trades_df.empty:
                st.warning("No trades found. Relax filters or fetch more data.")

    # ---- Results display (runs from session_state on every render) ----
    if st.session_state.get("bt_has_results"):
        trades_df = st.session_state.bt_trades_df
        equity_df = st.session_state.bt_equity_df
        bt_cfg = st.session_state.bt_cfg

        metrics = compute_all_metrics(
            trades_df,
            n_bootstrap=int(n_bootstrap),
            initial_capital=float(initial_capital),
        )

        # Universe discrepancy warning (volume / depth filters not applied)
        discrepancies = metrics.get("universe_discrepancies", [])
        if discrepancies:
            missing = [d["filter"] for d in discrepancies if not d["applied_in_backtest"]]
            if missing:
                st.warning(
                    "⚠️ **Universe mismatch:** The following live-engine filters "
                    "are NOT applied in backtest because historical data is "
                    "unavailable from Polymarket APIs: "
                    + ", ".join(f"`{m}`" for m in missing)
                    + ". Backtest results may be **optimistically biased** "
                    "(wider universe, same fill model). "
                    "See audit: `BACKTEST_AUDIT.md` §5.",
                    icon="⚠️",
                )

        # Metrics
        st.subheader("Performance Metrics")

        dd_pct = metrics.get("max_drawdown_pct", 0)
        # With initial capital, DD% is anchored to a realistic
        # capital base.  >100 % means loss exceeded starting
        # bankroll — visible to user, not hidden.

        calmar_val = metrics.get("calmar")
        calmar_display = f"{calmar_val:.2f}" if calmar_val is not None else "N/A"
        calmar_help = (
            "Annualized return / max drawdown. Higher = better risk-adjusted."
            if calmar_val is not None
            else "Suppressed — < 60 calendar days; CAGR is unreliable on short samples."
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total PnL", f"${metrics.get('total_pnl', 0):.2f}")
        c2.metric("Trades", str(metrics.get("n_trades", 0)))
        c3.metric("Hit Rate", f"{metrics.get('hit_rate', 0)*100:.1f}%")
        c4.metric("Sortino", f"{metrics.get('sortino', 0):.2f}",
                  help="Annualized return / downside deviation. Only penalises "
                       "loss volatility — honest for negatively-skewed strategies "
                       "like the NO-fader where large losses are rare but real.")
        c5.metric("Calmar", calmar_display, help=calmar_help)

        c6, c7, c8, c9, c10 = st.columns(5)
        c6.metric("Max DD $", f"${metrics.get('max_drawdown', 0):.2f}")
        c7.metric("Max DD %", f"{dd_pct*100:.1f}%",
                  help="Peak-to-trough as % of peak equity.")
        c8.metric("Skewness", f"{metrics.get('daily_skew', 0):.3f}",
                  help="Daily return skewness. Negative = left tail "
                       "(insurance-like risk). Sortino accounts for this; "
                       "Sharpe would not.")
        c9.metric("Expectancy", f"${metrics.get('expectancy', 0):.4f}")
        c10.metric("Avg Win", f"${metrics.get('avg_win', 0):.4f}")

        # Tail risk row
        c11, c12, c13, c14, c15 = st.columns(5)
        c11.metric("VaR 95", f"${metrics.get('daily_var_95', 0):.3f}",
                   help="Historical Value-at-Risk 95%: 5th percentile of daily PnL. "
                        "On 5% of days you lose at least this much.")
        c12.metric("VaR 99", f"${metrics.get('daily_var_99', 0):.3f}",
                   help="Historical VaR 99%: 1st percentile of daily PnL. "
                        "Worst 1% of days.")
        c13.metric("CVaR 95", f"${metrics.get('daily_cvar_95', 0):.3f}",
                   help="Conditional VaR 95%: expected loss GIVEN the loss "
                        "exceeds VaR 95. Always ≤ VaR 95.")
        c14.metric("Kurtosis", f"{metrics.get('daily_kurtosis', 0):.2f}",
                   help="Excess kurtosis of daily returns. > 0 = fatter tails "
                        "than normal. > 2 = very heavy tails.")
        c15.metric("Avg Loss", f"${metrics.get('avg_loss', 0):.4f}")

        normality_p = metrics.get("daily_normality_p", float("nan"))
        ci_pnl = metrics.get("pnl_ci_95", (0, 0))
        norm_label = f"{normality_p:.3f}" if not (normality_p is None or (isinstance(normality_p, float) and math.isfinite(normality_p))) else "N/A" if normality_p is not None else "N/A"
        st.caption(
            f"PnL 95% CI [{ci_pnl[0]:.2f}, {ci_pnl[1]:.2f}] (block bootstrap, n={n_bootstrap:,})  |  "
            f"Normality p={norm_label} "
            f"(p < 0.05 → non-normal returns → Sortino is the right risk measure)"
        )

        # Timing + diagnostics footer
        n_daily = metrics.get("n_daily", 0)
        n_active = metrics.get("n_active_days", 0)
        elapsed = metrics.get("elapsed_ms", 0)
        sample_warn = ""
        if n_daily < 60:
            sample_warn = (
                f" ⚠️ Only {n_daily} calendar days ({n_active} active). "
                "Annualized metrics are unreliable with < 60 days."
            )
        st.caption(
            f"metrics v{metrics.get('metrics_version', '?')}  |  "
            f"{n_active} active / {n_daily} calendar days  |  "
            f"{elapsed:.0f} ms{sample_warn}"
        )

        # Equity curve (shifted by initial capital for display)
        if not equity_df.empty:
            st.subheader("Equity Curve")
            eq_display = equity_df.copy()
            eq_display["cumulative_pnl"] = eq_display["cumulative_pnl"] + float(initial_capital)
            st.line_chart(eq_display.set_index("date")["cumulative_pnl"])

        # Per-market
        st.subheader("Per-Market PnL Attribution")
        pm = metrics.get("per_market", [])
        if pm:
            st.dataframe(pd.DataFrame(pm), width='stretch')

        # Trade list
        st.subheader("Trade List")
        st.dataframe(trades_df, width='stretch')

        # -----------------------------------------------------------
        # Walk-Forward Stability (Option A — descriptive, model-free)
        # -----------------------------------------------------------
        st.divider()
        st.subheader("Walk-Forward Stability")
        st.caption(
            "Splits the data into contiguous calendar windows and runs "
            "the **same** strategy in every window.  No parameter is "
            "optimised or carried across windows.  Stability grades: "
            "green = stable, yellow = moderate, red = unstable."
        )

        n_wf_windows = st.selectbox(
            "Windows", [2, 3, 4, 6, 8], index=2,
            key="bt_wf_n",
            help="Number of equal-width calendar windows.",
        )

        if st.button("Run Walk-Forward", key="bt_wf_run"):
            from backtest.walkforward import (
                partition_calendar_windows,
                window_summary,
            )

            # Reconstruct bt_cfg from session_state (belt) or widget
            # values (suspenders) — the backtest button may not have
            # been clicked in this render.
            sizing_fn2 = None
            if alpha != 0.0:
                sizing_fn2 = make_sizing_fn(alpha, band_low, band_high)
            _bt_cfg = st.session_state.get("bt_cfg") or BacktestConfig(
                band_low=band_low,
                band_high=band_high,
                min_dte=int(min_dte),
                max_dte=int(max_dte),
                min_time_in_band_days=int(min_time_days),
                order_notional_usd=float(notional),
                slippage_c=float(slippage_c),
                adverse_selection_c=float(adv_sel_c),
                n_bootstrap=int(n_bootstrap),
                sizing_fn=sizing_fn2,
            )

            with st.spinner(f"Running walk-forward ({n_wf_windows} windows)..."):
                wf_store = ContractPriceStore(PRICES_CSV)
                wf_windows = partition_calendar_windows(
                    wf_store, n_windows=n_wf_windows,
                )
                wf_per, wf_stab = window_summary(
                    wf_windows, _bt_cfg,
                    slugs=_expand_series_slugs(selected_slugs, wf_store, {s.slug: s for s in cfg.slugs}),
                    n_bootstrap=max(1000, int(n_bootstrap) // 4),
                    initial_capital=float(initial_capital),
                )

            # Stability summary
            grade = wf_stab.get("stability_grade", "no_data")
            grade_color = {
                "stable": "green",
                "moderate": "orange",
                "unstable": "red",
                "too_few_windows": "gray",
                "no_data": "gray",
            }.get(grade, "gray")

            st.markdown(f"**Stability grade:** :{grade_color}[{grade}]")
            sc, sc2, sc3 = st.columns(3)
            sc.metric(
                "Sortino CV",
                f"{wf_stab.get('sortino_cv', 0):.2f}" if wf_stab.get("sortino_cv") is not None else "N/A",
                help="Coefficient of variation of per-window Sortino. "
                     "< 0.5 = stable, > 1.0 = erratic.",
            )
            sc2.metric(
                "PnL Concentration",
                f"{wf_stab.get('pnl_concentration', 0)*100:.0f}%" if wf_stab.get("pnl_concentration") is not None else "N/A",
                help="% of total PnL earned in the best window. "
                     "> 50 % = concentrated, > 70 % = one lucky window.",
            )
            sc3.metric(
                "Sortino Range",
                f"[{wf_stab.get('sortino_min', 0):.2f}, {wf_stab.get('sortino_max', 0):.2f}]"
                if wf_stab.get("sortino_min") is not None else "N/A",
                help="Min / max per-window Sortino.",
            )

            # Per-window table
            if wf_per:
                wf_rows = []
                for w in wf_per:
                    m = w.metrics
                    wf_rows.append({
                        "Window": w.label,
                        "Trades": w.n_trades,
                        "Total PnL": f"${m.get('total_pnl', 0):.2f}",
                        "Hit Rate": f"{m.get('hit_rate', 0)*100:.1f}%" if m else "N/A",
                        "Sortino": f"{m.get('sortino', 0):.2f}" if m else "N/A",
                        "Max DD %": f"{m.get('max_drawdown_pct', 0)*100:.1f}%" if m else "N/A",
                    })
                st.dataframe(
                    pd.DataFrame(wf_rows), width='stretch',
                    hide_index=True,
                )

                # Bar chart: per-window Sortino
                chart_data = []
                for w in wf_per:
                    if w.n_trades > 0 and "sortino" in w.metrics:
                        chart_data.append({
                            "Window": w.label,
                            "Sortino": w.metrics["sortino"],
                        })
                if chart_data:
                    st.bar_chart(
                        pd.DataFrame(chart_data).set_index("Window"),
                        width='stretch',
                    )

            if wf_stab.get("pnl_concentration", 0) > 0.5:
                st.warning(
                    "⚠️ PnL is concentrated in one window (> 50 %). "
                    "Performance may be regime-dependent — be cautious "
                    "extrapolating to a different market environment."
                )

    # -----------------------------------------------------------
    # Parameter Sweep (band × alpha)
    # -----------------------------------------------------------
    st.divider()
    st.subheader("Parameter Sweep")
    st.caption(
        "Sweeps band bounds and alpha tilt to find optimal parameter "
        "combinations. Each (band, alpha) point runs a full backtest. "
        "Heavy: keep ranges narrow or increments coarse."
    )

    import numpy as np

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        alpha_min = st.text_input("Alpha min", "-1.0", key="sw_amin")
    with col_a2:
        alpha_max = st.text_input("Alpha max", "1.0", key="sw_amax")
    with col_a3:
        alpha_step = st.text_input("Alpha step", "0.2", key="sw_astep")

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        band_low_min = st.text_input("Band Low min", "0.70", key="sw_blmin")
    with col_b2:
        band_low_max = st.text_input("Band Low max", "0.80", key="sw_blmax")
    with col_b3:
        band_low_step = st.text_input("Band Low step", "0.05", key="sw_blstep")

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        band_high_min = st.text_input("Band High min", "0.90", key="sw_bhmin")
    with col_c2:
        band_high_max = st.text_input("Band High max", "0.95", key="sw_bhmax")
    with col_c3:
        band_high_step = st.text_input("Band High step", "0.05", key="sw_bhstep")

    if st.button("Run Parameter Sweep", key="bt_sweep_run"):
        from backtest.allocation_analysis import run_allocation_analysis

        try:
            alphas = [round(x, 2) for x in np.arange(
                float(alpha_min), float(alpha_max) + 1e-9, float(alpha_step)
            ).tolist()]
            low_vals = [round(x, 2) for x in np.arange(
                float(band_low_min), float(band_low_max) + 1e-9, float(band_low_step)
            ).tolist()]
            high_vals = [round(x, 2) for x in np.arange(
                float(band_high_min), float(band_high_max) + 1e-9, float(band_high_step)
            ).tolist()]
        except (ValueError, TypeError) as e:
            st.error(f"Invalid number: {e}")
            st.stop()

        total_combos = len(low_vals) * len(high_vals)
        st.info(
            f"Sweeping {len(low_vals)}×{len(high_vals)} = {total_combos} band(s) "
            f"× {len(alphas)} alphas = {total_combos * len(alphas)} backtests"
        )

        sweep_store = ContractPriceStore(PRICES_CSV)
        # Build base config from current widget values
        sizing_fn_sweep = None
        if alpha != 0.0:
            sizing_fn_sweep = make_sizing_fn(alpha, band_low, band_high)
        base_cfg = st.session_state.get("bt_cfg") or BacktestConfig(
            band_low=band_low,
            band_high=band_high,
            min_dte=int(min_dte),
            max_dte=int(max_dte),
            min_time_in_band_days=int(min_time_days),
            order_notional_usd=float(notional),
            slippage_c=float(slippage_c),
            adverse_selection_c=float(adv_sel_c),
            n_bootstrap=int(n_bootstrap),
            sizing_fn=sizing_fn_sweep,
        )

        progress_bar = st.progress(0, text="Starting sweep...")
        sweep_results: list[dict] = []
        n_done = 0

        for low in low_vals:
            for high in high_vals:
                if low >= high:
                    continue
                n_done += 1
                progress_bar.progress(
                    n_done / total_combos,
                    text=f"Band [{low:.2f}, {high:.2f}] ({n_done}/{total_combos})..."
                )

                try:
                    analysis = run_allocation_analysis(
                        sweep_store, base_cfg,
                        alphas=alphas,
                        band_low=low, band_high=high,
                        n_walkforward_windows=0,  # skip OOS for speed
                        n_bootstrap=max(1000, int(n_bootstrap) // 4),
                    )

                    baseline = analysis.get("baseline", {})
                    per_scheme = analysis.get("per_scheme", [])
                    sweep_schemes = [r for r in per_scheme if not r.get("empty")]

                    # Best sweep alpha (highest Sortino)
                    best = None
                    if sweep_schemes:
                        best_scheme = max(
                            sweep_schemes,
                            key=lambda r: r.get("sortino", 0.0) or 0.0,
                        )
                        best = {
                            "alpha": best_scheme["alpha"],
                            "sortino": best_scheme.get("sortino", 0.0) or 0.0,
                            "total_pnl": best_scheme.get("total_pnl", 0.0),
                            "hit_rate": best_scheme.get("hit_rate", 0.0),
                            "n_trades": best_scheme.get("n_trades", 0),
                        }

                    sweep_results.append({
                        "band_low": low,
                        "band_high": high,
                        "n_entries": analysis.get("n_entries", 0),
                        "baseline_sortino": baseline.get("sortino", 0.0) or 0.0,
                        "baseline_pnl": baseline.get("total_pnl", 0.0),
                        "best_alpha": best["alpha"] if best else None,
                        "best_sortino": best["sortino"] if best else 0.0,
                        "best_pnl": best["total_pnl"] if best else 0.0,
                        "best_hit_rate": best["hit_rate"] if best else 0.0,
                        "best_trades": best["n_trades"] if best else 0,
                        "optimal_alpha": analysis.get("optimal_alpha"),
                    })
                except Exception as e:
                    st.error(f"Band [{low:.2f}, {high:.2f}] failed: {e}")
                    sweep_results.append({
                        "band_low": low, "band_high": high,
                        "n_entries": 0, "baseline_sortino": 0.0,
                        "baseline_pnl": 0.0, "best_alpha": None,
                        "best_sortino": 0.0, "best_pnl": 0.0,
                        "best_hit_rate": 0.0, "best_trades": 0,
                        "optimal_alpha": None,
                    })

        progress_bar.empty()

        if sweep_results:
            # Build display table
            rows = []
            for r in sweep_results:
                opt_str = f"{r['optimal_alpha']:+.3f}" if r.get("optimal_alpha") is not None else "—"
                best_a_str = f"{r['best_alpha']:+.1f}" if r.get("best_alpha") is not None else "—"
                rows.append({
                    "Band": f"[{r['band_low']:.2f}, {r['band_high']:.2f}]",
                    "Entries": r["n_entries"],
                    "Sortino (α=0)": f"{r['baseline_sortino']:.3f}",
                    "PnL (α=0)": f"${r['baseline_pnl']:.2f}",
                    "Best α": best_a_str,
                    "Sortino (best)": f"{r['best_sortino']:.3f}",
                    "PnL (best)": f"${r['best_pnl']:.2f}",
                    "Opt α*": opt_str,
                    "Trades": r["best_trades"],
                })

            st.subheader("Sweep Results")
            st.dataframe(
                pd.DataFrame(rows), width='stretch', hide_index=True,
            )

            # Highlight best band
            valid = [r for r in sweep_results if r["best_sortino"] > 0]
            if valid:
                best_row = max(valid, key=lambda r: r["best_sortino"])
                best_band = f"[{best_row['band_low']:.2f}, {best_row['band_high']:.2f}]"
                ba = best_row["best_alpha"]
                st.success(
                    f"🏆 Best: band {best_band}, α={ba:+.1f}, "
                    f"Sortino={best_row['best_sortino']:.3f}, "
                    f"PnL=${best_row['best_pnl']:.2f}"
                )

            # Sortino heatmap if we have 2D data
            if len(low_vals) > 1 and len(high_vals) > 1:
                st.subheader("Sortino Heatmap")
                heatmap_data = {}
                for r in sweep_results:
                    key = f"{r['band_low']:.2f}"
                    if key not in heatmap_data:
                        heatmap_data[key] = {}
                    heatmap_data[key][f"{r['band_high']:.2f}"] = r["best_sortino"]
                hm_df = pd.DataFrame(heatmap_data).T
                hm_df.index.name = "Band Low"
                hm_df.columns.name = "Band High"
                st.dataframe(
                    hm_df.style.background_gradient(
                        axis=None, cmap="RdYlGn"
                    ).format("{:.3f}"),
                    width='stretch',
                )


if __name__ == "__main__":
    # Standalone mode
    st.set_page_config(page_title="Fader Backtest", layout="wide")
    st.title("Fader Backtest")
    render(embedded=False)
