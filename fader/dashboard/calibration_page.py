"""dashboard/calibration_page.py

Streamlit calibration page.

Shows whether the bot's core thesis -- longshot YES contracts on
Polymarket are systematically overpriced (equivalently: NO contracts in
the trading band are systematically underpriced relative to their true
win rate) -- still holds in the historical data, kept fresh by the
engine's background calibration poller (``polling.calibration_fetch_s``,
default 6h; see ``engine/pollers.py``).

Can be used two ways:

1. Embedded as a tab inside app.py:
       from dashboard import calibration_page
       with tab_calibration:
           calibration_page.render()

2. Run standalone:
       streamlit run fader/dashboard/calibration_page.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

_FADER_ROOT = Path(__file__).resolve().parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))

from backtest.calibration import (
    CalibrationParams,
    band_summary,
    bot_trade_calibration,
    build_observations,
    bucket_calibration,
    filter_by_series,
    monthly_edge,
    _parse_date,
)
from backtest.historical import ContractPriceStore, PRICES_CSV
from config.config_loader import apply_config_kv_overrides, load_config
from infra.db import get_connection
from marketdata.rest_market import _derive_series_filter

_WINDOW_OPTIONS = ["30d", "60d", "90d", "All"]
_WINDOW_DAYS = {"30d": 30, "60d": 60, "90d": 90, "All": None}

_BUCKET_FORMAT = {
    "bucket_mid": "{:.3f}",
    "mean_implied": "{:.3f}",
    "yes_rate": "{:.3f}",
    "edge": "{:.3f}",
    "wilson_low": "{:.3f}",
    "wilson_high": "{:.3f}",
}

_BOT_BUCKET_FORMAT = {
    "bucket_mid": "{:.3f}",
    "avg_entry": "{:.3f}",
    "win_rate": "{:.3f}",
    "edge_pp": "{:.3f}",
    "wilson_low": "{:.3f}",
    "wilson_high": "{:.3f}",
}


def _df_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Small local copy of app.py's DB helper (avoid importing app.py,
    which would execute the whole dashboard on import)."""
    conn = get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])
    finally:
        conn.close()


@st.cache_data(ttl=300)
def _load_snapshot(path_str: str, mtime: float) -> pd.DataFrame:
    """Load the price store snapshot, cached 5 min.

    ``mtime`` is part of the cache key purely so the engine's atomic
    ``os.replace`` on the CSV (see ``ContractPriceStore.save()``)
    invalidates the cache -- re-parsing a ~15 MB CSV on every widget
    interaction would otherwise be wasteful.
    """
    return ContractPriceStore(Path(path_str)).snapshot()


def _count_pending_resolution(snapshot_df: pd.DataFrame) -> int:
    """Count tokens with a parseable past end_date but no resolution yet.

    Mirrors the resolution/end_date selection rules in
    ``backtest.calibration.build_observations`` (first non-empty
    resolution across a token's rows; first parseable end_date).
    """
    if snapshot_df is None or snapshot_df.empty:
        return 0
    df = snapshot_df.copy()
    df["_res"] = df["resolution"].fillna("").astype(str).str.strip()
    df["_end"] = df["end_date"].apply(_parse_date)
    today = datetime.now(timezone.utc).date()
    pending = 0
    for _token_id, grp in df.groupby("token_id"):
        if (grp["_res"] != "").any():
            continue
        end_dates = grp["_end"].dropna()
        if end_dates.empty:
            continue
        if end_dates.iloc[0] < today:
            pending += 1
    return pending


def _series_filter_for(row) -> str:
    """Resolve the effective series_filter for a slugs.csv series row,
    matching the fallback chain used elsewhere in the repo (historical.py,
    rest_market.py, pollers.py, order_manager.py, startup.py)."""
    if row.market_kind == "btc_daily":
        return "bitcoin-above"
    if row.series_filter:
        return row.series_filter
    return _derive_series_filter(row.slug)


def render(embedded: bool = True) -> None:
    """Render the calibration UI.

    When ``embedded`` is True the settings live in an expander inside the
    current container (so they don't collide with the engine controls that
    already own the sidebar in app.py). When False (standalone), settings
    go in the sidebar.
    """
    cfg = load_config()
    # Overlay dashboard-written config_kv overrides so the band (and DTE
    # defaults) match what the live engine is actually running with, not
    # just the config.yaml values.
    active_overrides = apply_config_kv_overrides(cfg)

    st.header("Calibration")
    st.caption(
        "Implied YES probability (1 - NO price) vs actual YES resolution "
        "rate, for markets the bot's trading band trades. Positive edge "
        "means the market over-priced the longshot -- the bias the bot "
        "fades still exists."
    )

    csv_path = PRICES_CSV
    mtime = os.path.getmtime(csv_path) if csv_path.exists() else 0.0
    snapshot_df = _load_snapshot(str(csv_path), mtime)

    if snapshot_df.empty:
        st.info(
            "No historical price data yet at `DATA/historical_prices.csv`. "
            "The engine's calibration poller (`polling.calibration_fetch_s`, "
            "default 6h) fetches this automatically once running. It can "
            "also be populated from the Backtest tab's fetch button."
        )
        return

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    settings = st.expander("Calibration Settings", expanded=True) if embedded else st.sidebar
    if not embedded:
        settings.header("Calibration Settings")

    with settings:
        dte_default = (cfg.strategy.min_dte + cfg.strategy.max_dte) // 2
        dte_default = max(0, min(10, dte_default))
        dte_days = st.slider(
            "Days before resolution (N)", 0, 10, value=dte_default, key="cal_dte",
        )
        window_label = st.selectbox(
            "Window", _WINDOW_OPTIONS, index=3, key="cal_window",
        )
        window_days = _WINDOW_DAYS[window_label]

        series_rows = [s for s in cfg.slugs if s.market_kind in ("series", "btc_daily")]
        series_filter_map = {s.slug: _series_filter_for(s) for s in series_rows}
        series_options = list(series_filter_map.keys())

        if series_options:
            selected_series = st.multiselect(
                "Series", options=series_options, default=series_options,
                key="cal_series",
            )
        else:
            selected_series = []

        st.caption(
            "Tolerance fixed at 1 day. Selecting a subset of series "
            "restricts observations to those series only -- binary-slug "
            "markets drop out of the sample when subsetting."
        )

    # ------------------------------------------------------------------
    # Build observations
    # ------------------------------------------------------------------
    params = CalibrationParams(
        dte_days=int(dte_days), tolerance_days=1, bucket_width=0.05,
        window_days=window_days,
    )
    obs = build_observations(snapshot_df, params)

    if series_options and set(selected_series) != set(series_options):
        parts = [filter_by_series(obs, series_filter_map[slug]) for slug in selected_series]
        if parts:
            obs = pd.concat(parts, ignore_index=True).drop_duplicates(
                subset="token_id"
            ).reset_index(drop=True)
        else:
            obs = obs.iloc[0:0].reset_index(drop=True)

    band_low = cfg.strategy.band_low
    band_high = cfg.strategy.band_high

    # ------------------------------------------------------------------
    # 2. Headline
    # ------------------------------------------------------------------
    st.subheader("Headline")
    band_src = (
        "live override (config_kv)"
        if any(k.startswith("strategy.band") for k in active_overrides)
        else "config.yaml"
    )
    st.caption(f"Band {band_low:.2f}-{band_high:.2f} ({band_src}) -- the live engine's effective band.")
    summary = band_summary(obs, band_low, band_high)

    c1, c2, c3, c4 = st.columns(4)
    if summary["n"] > 0:
        edge_pp = summary["edge_pp"] * 100.0
        c1.metric("Resolved markets", f"{summary['n']:,}")
        c2.metric("Avg NO price", f"{summary['avg_no_price']:.3f}")
        c3.metric("NO win rate", f"{summary['no_win_rate']*100:.1f}%")
        c4.metric("Edge", f"{edge_pp:+.1f}pp", delta=f"{edge_pp:+.1f}pp")
        st.caption(
            f"Wilson 95% CI on NO win rate: "
            f"[{summary['wilson_low']*100:.1f}%, {summary['wilson_high']*100:.1f}%]. "
            "Positive edge = NO wins more often than price implies = "
            "longshot bias present."
        )
    else:
        c1.metric("Resolved markets", "-")
        c2.metric("Avg NO price", "-")
        c3.metric("NO win rate", "-")
        c4.metric("Edge", "-")
        st.caption("No observations in the trading band for the current filters.")

    # Freshness
    calibration_fetch_s = getattr(cfg.polling, "calibration_fetch_s", 21600)
    fetched_vals = snapshot_df.get("fetched_at", pd.Series([], dtype=str))
    fetched_vals = fetched_vals.fillna("").astype(str)
    fetched_vals = fetched_vals[fetched_vals != ""]

    if not fetched_vals.empty:
        max_fetched_at = fetched_vals.max()
        hours = calibration_fetch_s / 3600.0 if calibration_fetch_s else 0.0
        st.caption(
            f"Data updated {max_fetched_at} -- auto-refreshed every "
            f"{hours:.1f}h by the engine (polling.calibration_fetch_s)."
        )
        try:
            fetched_dt = datetime.fromisoformat(max_fetched_at)
            if fetched_dt.tzinfo is None:
                fetched_dt = fetched_dt.replace(tzinfo=timezone.utc)
            age_s = (datetime.now(timezone.utc) - fetched_dt).total_seconds()
            if calibration_fetch_s > 0 and age_s > 2 * calibration_fetch_s:
                st.warning(
                    "Newest data is older than 2x the calibration fetch "
                    "interval -- the engine may not be running or the "
                    "calibration poller may be failing.",
                    icon="⚠️",
                )
        except ValueError:
            pass
    else:
        st.caption("No fetched_at timestamps available yet.")

    pending = _count_pending_resolution(snapshot_df)
    st.caption(f"{pending:,} token(s) past end_date still pending resolution.")

    # ------------------------------------------------------------------
    # 3. Calibration curve
    # ------------------------------------------------------------------
    st.subheader("Calibration curve")
    bucket_df = bucket_calibration(obs, bucket_width=params.bucket_width)

    if not bucket_df.empty:
        curve_df = bucket_df.set_index("bucket_mid")[["yes_rate"]].rename(
            columns={"yes_rate": "Actual YES rate"}
        )
        curve_df["Perfect calibration"] = curve_df.index
        st.line_chart(curve_df)
        st.caption(
            f"Trading band corresponds to implied YES "
            f"{1 - band_high:.2f}-{1 - band_low:.2f} (leftmost region). "
            "Points below the diagonal = YES overpriced."
        )
    else:
        st.info("No bucketed observations for the current filters.")

    # ------------------------------------------------------------------
    # 4. Per-bucket table
    # ------------------------------------------------------------------
    st.subheader("Per-bucket detail")
    if not bucket_df.empty:
        styled = bucket_df.style.background_gradient(
            subset=["edge"], cmap="RdYlGn"
        ).format(_BUCKET_FORMAT)
        st.dataframe(styled, width='stretch')
    else:
        st.info("No bucketed observations for the current filters.")

    # ------------------------------------------------------------------
    # 5. Monthly edge trend
    # ------------------------------------------------------------------
    st.subheader("Edge over time (band only)")
    monthly_df = monthly_edge(obs, band_low, band_high)

    if not monthly_df.empty:
        st.bar_chart(monthly_df.set_index("month")["edge_pp"])
        st.dataframe(
            monthly_df[["month", "n", "avg_no_price", "no_win_rate", "edge_pp"]],
            width='stretch', hide_index=True,
        )
        st.caption(
            "Thin months (low n) are noisy -- treat single-month swings "
            "in edge_pp with caution."
        )
    else:
        st.info("No monthly band observations for the current filters.")

    # ------------------------------------------------------------------
    # 6. Bot's own trades
    # ------------------------------------------------------------------
    st.subheader("Bot realized calibration")
    st.caption(
        "Biased sample -- conditioned on all 11 entry filters passing; "
        "not a market-wide estimate."
    )

    positions_df = _df_query(
        "SELECT entry_price, realized_pnl FROM positions "
        "WHERE status='CLOSED' AND realized_pnl IS NOT NULL"
    )

    if positions_df.empty:
        st.info("No closed positions yet.")
    else:
        clean = positions_df.copy()
        clean["entry_price"] = pd.to_numeric(clean["entry_price"], errors="coerce")
        clean["realized_pnl"] = pd.to_numeric(clean["realized_pnl"], errors="coerce")
        clean = clean.dropna(subset=["entry_price", "realized_pnl"])

        if clean.empty:
            st.info("No closed positions with valid entry price / realized PnL yet.")
        else:
            n_bot = len(clean)
            win_rate = float((clean["realized_pnl"] > 0).mean())
            avg_entry = float(clean["entry_price"].mean())
            bot_edge_pp = (win_rate - avg_entry) * 100.0

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Closed trades", f"{n_bot:,}")
            b2.metric("Win rate", f"{win_rate*100:.1f}%")
            b3.metric(
                "Avg entry price", f"{avg_entry:.3f}",
                help="Breakeven win rate implied by the average entry price.",
            )
            b4.metric("Edge", f"{bot_edge_pp:+.1f}pp", delta=f"{bot_edge_pp:+.1f}pp")

            bot_bucket_df = bot_trade_calibration(positions_df, bucket_width=0.05)
            if not bot_bucket_df.empty:
                styled_bot = bot_bucket_df.style.background_gradient(
                    subset=["edge_pp"], cmap="RdYlGn"
                ).format(_BOT_BUCKET_FORMAT)
                st.dataframe(styled_bot, width='stretch')


if __name__ == "__main__":
    # Standalone mode
    st.set_page_config(page_title="Fader Calibration", layout="wide")
    st.title("Fader Calibration")
    render(embedded=False)
