"""dashboard/app.py

Streamlit dashboard for the fader bot (Process B).

Reads engine state from SQLite (WAL; safe concurrent read).
Writes control commands and config_kv to the DB for the engine to consume.

Run:
    streamlit run fader/dashboard/app.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Make fader modules importable (resolve() so a relative __file__ under
# `streamlit run` from any cwd still yields the absolute fader/ root).
_FADER_ROOT = Path(__file__).resolve().parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))

from infra.db import get_connection, init_db
from engine.control_consumer import issue_command
from dashboard import backtest_page

st.set_page_config(
    page_title="Fader Bot", layout="wide", initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------------

def _get_state(key: str) -> Any:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT value_json FROM engine_state WHERE key=?", (key,)
        ).fetchone()
        return json.loads(row["value_json"]) if row else None
    finally:
        conn.close()


def _df_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])
    finally:
        conn.close()


def _scalar(sql: str, params: tuple = (), default=0):
    conn = get_connection()
    try:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row and row[0] is not None else default
    finally:
        conn.close()


def _write_config_kv(key: str, value: Any) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO config_kv (key, value, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), now),
        )
        conn.commit()
    finally:
        conn.close()


def _get_config_kv(key: str, default: Any = None) -> Any:
    """Read config_kv override; fall back to default if no entry exists."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT value FROM config_kv WHERE key=?", (key,)
        ).fetchone()
        if row:
            return json.loads(row["value"])
    finally:
        conn.close()
    return default


def _engine_is_running() -> bool:
    """Return True if engine published state within the last 15 seconds."""
    ts_str = _get_state("published_at")
    if ts_str is None:
        return False
    try:
        published = datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return False
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - published).total_seconds()
    return 0 <= age < 15


def _load_slugs_from_csv() -> list[dict]:
    """Read slugs.csv as list of dicts preserving all columns."""
    slugs_path = _FADER_ROOT / "config" / "slugs.csv"
    if not slugs_path.exists():
        return []
    with open(slugs_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("slug", "").strip()]


def _write_slug_enabled(slugs_data: list[dict], enabled_slugs: set[str]) -> None:
    """Write slugs.csv with updated enabled column. Atomic via temp file."""
    import copy
    data = copy.deepcopy(slugs_data)
    for row in data:
        row["enabled"] = "1" if row["slug"] in enabled_slugs else "0"
    fieldnames = list(data[0].keys()) if data else []
    slugs_path = _FADER_ROOT / "config" / "slugs.csv"
    tmp_path = slugs_path.with_suffix(".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    os.replace(tmp_path, slugs_path)


# Ensure DB exists
try:
    init_db()
except Exception:
    pass

# ------------------------------------------------------------------
# Sidebar — controls
# ------------------------------------------------------------------

st.sidebar.title("Fader Bot Controls")

# Engine status
ws_connected = _get_state("ws_connected") or False
breaker = _get_state("breaker_tripped") or False
gap_halt = _get_state("gap_halted") or False
bankroll = _get_state("bankroll") or 0.0
published_at = _get_state("published_at") or "—"

st.sidebar.markdown(
    f"""
    **WS:** {'Connected' if ws_connected else 'DISCONNECTED'}
    **Breaker:** {'TRIPPED' if breaker else 'OK'}
    **Gap halt:** {'YES' if gap_halt else 'no'}
    **Bankroll:** ${bankroll:.2f}
    **Last update:** {published_at}
    """
)

# Session state — track launched engine process
if "engine_process" not in st.session_state:
    st.session_state.engine_process = None


def _launch_engine():
    """Popen the engine as a detached child and return the process handle.

    Shared by the Start button and the restart relaunch path. Raises on
    failure — callers surface the error.
    """
    engine_script = str(_FADER_ROOT / "run_engine.py")
    project_root = str(_FADER_ROOT.parent)
    log_path = str(_FADER_ROOT / "engine_startup.log")
    # Child inherits its own dup of the fd at spawn, so closing the parent's
    # handle after Popen returns (context exit) is safe and avoids leaking a
    # descriptor on every launch.
    with open(log_path, "w") as log_f:
        return subprocess.Popen(
            [sys.executable, engine_script],
            cwd=project_root,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )


proc = st.session_state.engine_process
if proc is not None and proc.poll() is not None:
    # Process exited since last render.
    st.session_state.engine_process = None
    proc = None

# Cold-restart relaunch: engine confirmed down + a restart was requested and
# the dashboard owns the process (local launch — no external supervisor). The
# old process is still alive (poll() is None) during the ~5-15s graceful
# shutdown, so this only fires once it has actually exited. On the VPS the
# engine runs under systemd (flag never set), so this never fires. The flag is
# cleared only on a successful launch, so a failed attempt retries next rerun.
if st.session_state.get("pending_local_relaunch") and st.session_state.engine_process is None:
    try:
        st.session_state.engine_process = _launch_engine()
        st.session_state.pending_local_relaunch = False
        st.session_state.control_flash = "Engine cold-restarted"
        proc = st.session_state.engine_process
    except Exception as e:
        st.sidebar.error(f"Relaunch failed (will retry): {e}")

engine_running = _engine_is_running()
proc_active = st.session_state.engine_process is not None

# Local cold-restart driver: the dashboard has no external autorefresh, so
# while a dashboard-owned engine is still shutting down (poll() is None) after
# a restart, poll + rerun once a second so the exit gets detected and the
# relaunch above fires. Bounded (~30s) to avoid an infinite loop if graceful
# shutdown hangs — after that the user can Start manually.
if st.session_state.get("pending_local_relaunch"):
    _old = st.session_state.engine_process
    _ticks = st.session_state.get("relaunch_wait_ticks", 0)
    if _old is not None and _old.poll() is None and _ticks < 30:
        st.session_state.relaunch_wait_ticks = _ticks + 1
        time.sleep(1.0)
        st.rerun()
    else:
        st.session_state.relaunch_wait_ticks = 0

st.sidebar.divider()

# Start / Stop controls
if not engine_running and not proc_active:
    if st.sidebar.button("Start Engine", type="primary", width='stretch'):
        try:
            st.session_state.engine_process = _launch_engine()
            st.sidebar.success("Engine starting...")
            time.sleep(3)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to start engine: {e}")
elif proc_active and not engine_running:
    st.sidebar.info("Engine starting... (check engine_startup.log for progress)")
    if st.sidebar.button("Check Status", width='stretch'):
        st.rerun()
else:
    # Engine is running — show stop/breaker controls.
    # Widget-keyed session state (the confirm checkboxes) can only be
    # written from an on_click callback, which runs before widgets are
    # instantiated on the rerun; writing it in the button body raises
    # StreamlitAPIException. Success messages go through a flash slot so
    # they render on the rerun the callback triggers.
    def _issue_stop() -> None:
        issue_command("stop")
        st.session_state.confirm_stop = False
        st.session_state.control_flash = "Stop issued"

    def _issue_close_all() -> None:
        issue_command("close_all")
        st.session_state.confirm_close_all = False
        st.session_state.control_flash = "Close-all issued"

    def _issue_restart() -> None:
        # Engine shuts down gracefully (cancels resting orders) then exits 42.
        # A supervisor cold-starts it: systemd on the VPS, or — for a dashboard-
        # launched local engine — the pending_local_relaunch path above.
        issue_command("restart")
        st.session_state.confirm_restart = False
        if st.session_state.get("engine_process") is not None:
            st.session_state.pending_local_relaunch = True
            st.session_state.relaunch_wait_ticks = 0
        st.session_state.control_flash = (
            "Kill + cold restart issued (engine back in ~5-15s)"
        )

    flash = st.session_state.pop("control_flash", None)
    if flash:
        st.sidebar.success(flash)

    stop_confirm = st.sidebar.checkbox("Confirm STOP Engine", key="confirm_stop")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button("STOP Engine", type="primary", disabled=not stop_confirm,
                  width='stretch', on_click=_issue_stop)
    with col2:
        if st.button("Reset Breaker", width='stretch'):
            issue_command("breaker_reset")
            st.sidebar.success("Breaker reset")

    close_confirm = st.sidebar.checkbox("Confirm CLOSE ALL Positions", key="confirm_close_all")
    st.sidebar.button("CLOSE ALL Positions", type="secondary",
                      disabled=not close_confirm, width='stretch',
                      on_click=_issue_close_all)

    restart_confirm = st.sidebar.checkbox(
        "Confirm KILL + COLD RESTART", key="confirm_restart"
    )
    st.sidebar.button(
        "KILL + COLD RESTART Engine", type="secondary",
        disabled=not restart_confirm, width='stretch', on_click=_issue_restart,
        help="Graceful shutdown (cancels resting orders) then full process "
             "cold start — reloads .env, re-runs telegram.configure + full "
             "reconcile. Use after editing .env or for a clean slate.",
    )

    if st.sidebar.button("Reload Config", width='stretch'):
        issue_command("config_reload")
        st.sidebar.success("Config reload issued")

st.sidebar.divider()
st.sidebar.subheader("Live Params")

new_band_low = st.sidebar.number_input(
    "Band Low", min_value=0.01, max_value=0.99, step=0.01,
    value=float(_get_config_kv("strategy.band_low", 0.80))
)
new_band_high = st.sidebar.number_input(
    "Band High", min_value=0.01, max_value=0.99, step=0.01,
    value=float(_get_config_kv("strategy.band_high", 0.95))
)
new_min_dte = st.sidebar.number_input(
    "Min DTE", min_value=0,
    value=int(_get_config_kv("strategy.min_dte", 0))
)
new_max_dte = st.sidebar.number_input(
    "Max DTE", min_value=1,
    value=int(_get_config_kv("strategy.max_dte", 365))
)
new_notional = st.sidebar.number_input(
    "Order Notional (USD)", min_value=1.0,
    value=float(_get_config_kv("strategy.order_notional_usd", 10.0))
)
new_alpha = st.sidebar.slider(
    "Alpha (tilt)", min_value=-1.0, max_value=1.0, step=0.1,
    value=float(_get_config_kv("strategy.alpha", 0.0)),
    help="-1 = heavy low-price, 0 = uniform, +1 = heavy high-price"
)
new_daily_loss = st.sidebar.number_input(
    "Daily Loss Breaker %", min_value=0.1,
    value=float(_get_config_kv("risk.daily_loss_breaker_pct", 5.0))
)
new_min_24h_vol = st.sidebar.number_input(
    "Min 24h Volume", min_value=0.0,
    value=float(_get_config_kv("filters.min_24h_volume", 1000.0))
)

if st.sidebar.button("Apply Params"):
    _write_config_kv("strategy.band_low", new_band_low)
    _write_config_kv("strategy.band_high", new_band_high)
    _write_config_kv("strategy.min_dte", new_min_dte)
    _write_config_kv("strategy.max_dte", new_max_dte)
    _write_config_kv("strategy.order_notional_usd", new_notional)
    _write_config_kv("strategy.alpha", new_alpha)
    _write_config_kv("risk.daily_loss_breaker_pct", new_daily_loss)
    _write_config_kv("filters.min_24h_volume", new_min_24h_vol)
    issue_command("config_reload")
    st.sidebar.success("Params written + config reload issued")

# Slug management
st.sidebar.divider()
st.sidebar.subheader("Slug Management")

slugs_data = _load_slugs_from_csv()
all_slugs = [row["slug"] for row in slugs_data]
enabled_slugs = [
    row["slug"] for row in slugs_data
    if row.get("enabled", "1").strip() not in ("0", "false", "False", "no", "")
]

# Init committed_slugs on first render
if "committed_slugs" not in st.session_state:
    st.session_state.committed_slugs = set(enabled_slugs)

selected = st.sidebar.multiselect(
    "Active Slugs",
    options=all_slugs,
    default=enabled_slugs,
    key="slug_multiselect",
    help="Checked = enabled. Uncheck to disable. Engine reloads within 5s.",
)

# Detect changes and write back to slugs.csv
current_set = set(selected)
if current_set != st.session_state.committed_slugs:
    _write_slug_enabled(slugs_data, current_set)
    issue_command("config_reload")
    st.session_state.committed_slugs = current_set
    st.sidebar.success(f"Slugs updated ({len(current_set)} enabled). Config reload issued.")

# Add new slug
new_slug_input = st.sidebar.text_input("Add New Slug", key="new_slug_input",
                                       placeholder="e.g. bitcoin-above-on")
if st.sidebar.button("Add Slug") and new_slug_input.strip():
    new_slug = new_slug_input.strip()
    slugs_path = _FADER_ROOT / "config" / "slugs.csv"
    slugs_data = _load_slugs_from_csv()
    existing = {row["slug"] for row in slugs_data}
    if new_slug in existing:
        st.sidebar.error(f"Slug '{new_slug}' already exists")
    else:
        fieldnames = list(slugs_data[0].keys()) if slugs_data else [
            "slug", "enabled", "market_kind", "series_from_date", "series_filter",
            "band_low", "band_high", "size_override", "added_at", "notes"
        ]
        new_row = {fn: "" for fn in fieldnames}
        new_row["slug"] = new_slug
        new_row["enabled"] = "1"
        new_row["market_kind"] = "binary"
        new_row["added_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with open(slugs_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(new_row)

        issue_command("config_reload")
        st.session_state.new_slug_input = ""
        st.session_state.committed_slugs.add(new_slug)
        st.sidebar.success(f"Slug '{new_slug}' added + config reload issued")
        st.rerun()

# ------------------------------------------------------------------
# Main — tabs
# ------------------------------------------------------------------

(
    tab_overview,
    tab_positions,
    tab_orders,
    tab_decisions,
    tab_pnl,
    tab_risk,
    tab_feed,
    tab_backtest,
) = st.tabs([
    "Overview", "Positions", "Orders", "Decisions", "PnL & Metrics", "Risk",
    "Feed", "Backtest",
])

# ---- OVERVIEW ----
with tab_overview:
    st.header("Account Overview")
    open_pos = _scalar("SELECT COUNT(*) FROM positions WHERE status='OPEN'")
    total_notional = _scalar("SELECT COALESCE(SUM(notional),0) FROM positions WHERE status='OPEN'")
    realized_pnl = _scalar("SELECT COALESCE(SUM(realized_pnl),0) FROM positions WHERE status='CLOSED'")
    pending_orders = _scalar("SELECT COUNT(*) FROM orders WHERE status='PENDING'")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Bankroll (USDC)", f"${bankroll:.2f}")
    c2.metric("Deployed", f"${total_notional:.2f}")
    c3.metric("Idle", f"${max(0.0, bankroll - total_notional):.2f}")
    c4.metric("Realized PnL", f"${realized_pnl:.2f}")
    c5.metric("Open Positions", str(open_pos))

    st.divider()
    st.subheader("Engine Status")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("WS Connected", "Yes" if ws_connected else "NO")
    col_b.metric("Circuit Breaker", "TRIPPED" if breaker else "OK")
    col_c.metric("Gap Halt", "YES" if gap_halt else "No")

    feed_silence = _get_state("feed_silence_s") or 0.0
    st.metric("Feed Silence (s)", f"{feed_silence:.1f}")

# ---- POSITIONS ----
with tab_positions:
    st.header("Open Positions")
    df_pos = _df_query(
        """
        SELECT slug, outcome, entry_price, size, notional, opened_at, source
        FROM positions WHERE status='OPEN' ORDER BY opened_at DESC
        """
    )
    if df_pos.empty:
        st.info("No open positions")
    else:
        st.dataframe(df_pos, width='stretch')

    st.divider()
    st.header("Closed Positions")
    df_closed = _df_query(
        """
        SELECT slug, outcome, entry_price, size, notional, realized_pnl, resolved_at
        FROM positions WHERE status='CLOSED' ORDER BY resolved_at DESC LIMIT 200
        """
    )
    if df_closed.empty:
        st.info("No closed positions")
    else:
        st.dataframe(df_closed, width='stretch')

# ---- ORDERS ----
with tab_orders:
    st.header("Pending Orders")
    df_ord = _df_query(
        """
        SELECT order_id, slug, side, type, price, size, status, created_at, ttl_expires_at
        FROM orders WHERE status NOT IN ('FILLED','CANCELLED','FAILED')
        ORDER BY created_at DESC LIMIT 100
        """
    )
    if df_ord.empty:
        st.info("No pending orders")
    else:
        st.dataframe(df_ord, width='stretch')

    st.divider()
    st.header("Order History")
    df_ord_hist = _df_query(
        """
        SELECT order_id, slug, side, type, price, size, status, cancel_reason, created_at
        FROM orders ORDER BY created_at DESC LIMIT 500
        """
    )
    if not df_ord_hist.empty:
        st.dataframe(df_ord_hist, width='stretch')

# ---- DECISIONS ----
with tab_decisions:
    st.header("Decision Log")
    filter_decision = st.selectbox("Filter", ["ALL", "ENTERED", "REJECTED"])
    where_clause = (
        f"WHERE decision='{filter_decision}'"
        if filter_decision != "ALL"
        else ""
    )
    df_dec = _df_query(
        f"""
        SELECT ts, slug, decision, reason, filters_json, order_id
        FROM decisions {where_clause}
        ORDER BY ts DESC LIMIT 500
        """
    )
    if df_dec.empty:
        st.info("No decisions recorded yet")
    else:
        st.dataframe(df_dec, width='stretch')

# ---- PNL & METRICS ----
with tab_pnl:
    st.header("PnL & Performance Metrics")

    df_all = _df_query(
        """
        SELECT realized_pnl, entry_price, size, notional, slug
        FROM positions WHERE status='CLOSED' AND realized_pnl IS NOT NULL
        """
    )
    if df_all.empty:
        st.info("No closed positions yet")
    else:
        wins = df_all[df_all["realized_pnl"] > 0]
        losses = df_all[df_all["realized_pnl"] <= 0]
        total_pnl = df_all["realized_pnl"].sum()
        hit_rate = len(wins) / len(df_all) if len(df_all) > 0 else 0
        avg_win = wins["realized_pnl"].mean() if len(wins) > 0 else 0
        avg_loss = losses["realized_pnl"].mean() if len(losses) > 0 else 0
        expectancy = (
            hit_rate * avg_win + (1 - hit_rate) * avg_loss
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total PnL", f"${total_pnl:.2f}")
        c2.metric("Hit Rate", f"{hit_rate*100:.1f}%")
        c3.metric("Avg Win", f"${avg_win:.3f}")
        c4.metric("Avg Loss", f"${avg_loss:.3f}")
        st.metric("Expectancy (per trade)", f"${expectancy:.4f}")

        st.subheader("Per-Market PnL")
        by_slug = df_all.groupby("slug")["realized_pnl"].sum().reset_index()
        by_slug.columns = ["slug", "total_pnl"]
        by_slug = by_slug.sort_values("total_pnl", ascending=False)
        st.dataframe(by_slug, width='stretch')

        st.subheader("Trade History")
        st.dataframe(
            df_all[["slug", "entry_price", "size", "realized_pnl"]].sort_values(
                "realized_pnl"
            ),
            width='stretch',
        )

# ---- RISK ----
with tab_risk:
    st.header("Risk Dashboard")

    open_notional = _scalar(
        "SELECT COALESCE(SUM(notional),0) FROM positions WHERE status='OPEN'"
    )
    deploy_pct = (open_notional / bankroll * 100) if bankroll > 0 else 0

    c1, c2 = st.columns(2)
    c1.metric("Deployed %", f"{deploy_pct:.1f}%")
    c2.metric("Circuit Breaker", "TRIPPED" if breaker else "OK")

    st.subheader("Daily Circuit Breaker")
    df_cb = _df_query(
        "SELECT day, realized_pnl, tripped, tripped_at, reset_at FROM circuit_breaker ORDER BY day DESC LIMIT 30"
    )
    if not df_cb.empty:
        st.dataframe(df_cb, width='stretch')

    st.subheader("Live Config Values")
    df_kv = _df_query(
        "SELECT key, value, updated_at, updated_by FROM config_kv ORDER BY updated_at DESC"
    )
    if not df_kv.empty:
        st.dataframe(df_kv, width='stretch')

# ---- FEED ----
with tab_feed:
    st.header("Feed Status")

    ages = _get_state("token_last_update_ages") or {}
    if ages:
        df_ages = pd.DataFrame(
            [{"token_id": t[:24], "age_s": a} for t, a in ages.items()]
        )
        st.dataframe(df_ages, width='stretch')
    else:
        st.info("No feed data")

    st.metric("WS Reconnects", str(_get_state("ws_reconnect_count") or 0))
    st.metric("Feed Silence (s)", str(round(_get_state("feed_silence_s") or 0, 1)))
    st.metric("Engine Start", _get_state("engine_start_ts") or "—")

# ---- BACKTEST ----
with tab_backtest:
    backtest_page.render(embedded=True)
