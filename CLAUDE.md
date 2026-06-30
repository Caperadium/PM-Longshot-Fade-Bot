# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Anti-Longshot Polymarket Bot ("Fader Bot") — autonomous trading bot that buys NO contracts on Polymarket CLOB when prices are in a configurable band (default 0.80–0.95), under the thesis that long-shot contracts are systematically overpriced. Holds to resolution; no active exit.

## Commands

```bash
# Live engine
python fader/run_engine.py

# Dashboard (separate process, shares SQLite DB with engine)
streamlit run fader/run_dashboard.py

# Backtest dashboard (standalone)
streamlit run fader/dashboard/backtest_page.py

# Allocation analysis CLI
python -c "import sys; sys.path.insert(0,'fader'); from backtest.allocation_analysis import main; main()"

# Band sweep CLI
python -c "import sys; sys.path.insert(0,'fader'); from backtest.band_sweep import main; main()"

# Tests
python -m pytest fader/tests/test_live_readiness.py -v
python -m pytest fader/tests/test_allocation_analysis.py -v

# Fetch historical data (CLI — discovers series markets + pulls prices)
python -c "import sys; sys.path.insert(0,'fader'); from backtest.historical import fetch_and_store; from config.config_loader import load_config; cfg = load_config(); store = fetch_and_store([s.slug for s in cfg.enabled_slugs()], slug_configs={s.slug: s for s in cfg.slugs}, max_workers=4)"
```

Requires `.env` with `POLYMARKET_PRIVATE_KEY`, `POLYMARKET_USER_ADDRESS`, and optional `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`. Copy from `fader/.env.example`.

## Architecture

**Two-process design** — engine (asyncio event loop) and dashboard (Streamlit), communicating through a shared SQLite WAL database. Both read/write the same `fader/fader.db`.

### Engine startup sequence (engine/main.py)
1. Load config (config.yaml + slugs.csv)
2. Init DB tables
3. Resolve NO token IDs via Gamma API
4. Startup safety checks (MATIC balance, USDC allowance)
5. Full API reconciliation (bankroll, orders, positions)
6. Open CLOB websocket with exponential backoff
7. Wait for initial order books (REST resync on connect)
8. Launch concurrent tasks: strategy loop, order manager, pollers, state publisher, control consumer

### Runtime tasks (all concurrent asyncio)
| Task | Interval | Role |
|---|---|---|
| Strategy loop | 1s | 11-filter evaluation, calls OrderManager.enter() |
| Bankroll poller | 30s | Reconcile USDC/MATIC balances, USDC allowance |
| Resolution poller | 60s | Detect closed/resolved positions, book PnL |
| Discovery poller | 300s | Discover new rung markets (ladder + daily series) |
| State publisher | 2s | Write engine state to DB for dashboard |
| Control consumer | 1s | Process dashboard commands (stop, close_all, etc.) |
| Config watcher | 5s | Hot-reload config.yaml + slugs.csv on file change |

### 11-filter entry stack (engine/strategy_loop.py)
1. NO best ask ∈ [band_low, band_high]
2. DTE ∈ [min_dte, max_dte]
3. Continuously in band ≥ min_time_in_band_s (websocket timestamps)
4. 24h volume ≥ min_24h_volume
5. Cumulative volume ≥ min_total_volume
6. Book depth at NO touch ≥ min_book_depth
7. Data freshness: WS update ≤ max_staleness_s AND no gap-halt
8. No existing OPEN position on this token (no doubling up)
9. Per-market exposure cap not breached
10. Total deployed cap not breached
11. Daily-loss circuit breaker not tripped

### Order placement (execution/order_manager.py)
- Spread ≤ 1c → market order (FOK)
- Spread > 1c → limit at mid, requote on 0.5c mid move, TTL 5min, cancel on band exit
- All orders carry deterministic idempotency keys (execution/idempotency.py)
- Position row inserted immediately on market fill (closes 60s reconciler gap)

### Alpha tilt (execution/sizing.py)
Parameter `alpha` ∈ [-1, +1] redistributes notional across price band. +1 heavier on high-prob (near 0.95), -1 heavier on low-prob (near band_low), 0 = uniform. Floor at $1.00.

### Generic daily series (backtest-data feature)

Market slugs with `market_kind=series` in slugs.csv auto-expand to all individual daily binary markets for that series. Each series requires:
- `series_filter`: substring that identifies child market slugs (e.g. `"bitcoin-above"` matches `bitcoin-above-100k-on-january-1`)
- `series_from_date`: start date for Gamma `/events` discovery scan (YYYY-MM-DD)

`market_kind=btc_daily` is a backward-compat alias that auto-sets `series_filter=bitcoin-above`, `series_from_date=2024-01-01`.

Current series:

| Slug | Filter | From | Markets |
|---|---|---|---|
| `bitcoin-above-on` | `bitcoin-above` | 2024-01-01 | ~3,200 BTC daily price-threshold binaries |
| `highest-temperature-in-seoul-on` | `highest-temperature-in-seoul` | 2025-12-01 | ~200 Seoul daily temp ladder rungs |

**Adding a new series:**
1. Add row to `slugs.csv`: `slug,1,series,<date>,<filter>,,,,<added>,<notes>`
2. That's it — `fetch_and_store()` and `_expand_series_slugs()` auto-discover and expand.

**How it works:**
- `backtest/historical.py` `discover_series_markets()`: iterates every date, queries Gamma `/events` with `{base_slug}-{month}-{day}` and `{base_slug}-{month}-{day}-{year}` formats (legacy + current), collects all child NO tokens whose slug contains `series_filter`.
- `_series_date_slug_groups()`: generates both legacy (no-year) and current (year-suffixed) event slugs per date. Polymarket transitioned formats mid-2026.
- `_derive_series_filter()`: fallback that strips trailing `-on` from base slug when `series_filter` column is empty.
- `dashboard/backtest_page.py` `_expand_series_slugs()`: in the UI, series slugs expand to stored child slugs matching `{filter}-*` prefix.
- Backtest engine (`backtest/engine.py`) is market-agnostic — any slug with CSV data works identically.

## Key modules

| Module | Role |
|---|---|
| `execution/provider.py` | Polymarket REST wrapper (paper + live), MarketInfo, CLOB client |
| `marketdata/ws_client.py` | Persistent CLOB websocket, reconnect, delta/snapshot handling |
| `marketdata/book_state.py` | In-memory OrderBook per contract, band-entry timing tracker |
| `marketdata/staleness.py` | Per-contract staleness + feed-wide gap-halt |
| `engine/risk.py` | Circuit breaker, max deployed, per-market caps, MATIC gate |
| `engine/reconciler.py` | Startup + periodic API reconciliation (bankroll, orders, positions) |
| `engine/control_consumer.py` | Dashboard-to-engine IPC via control_commands table |
| `infra/db.py` | SQLite WAL schema, all 8 tables + indexes |
| `infra/telegram.py` | Telegram alerts (heartbeat, breaker trips, errors) |
| `dashboard/app.py` | 8-tab Streamlit dashboard |
| `dashboard/backtest_page.py` | Backtest UI (embedded + standalone) with parameter sweep |
| `backtest/engine.py` | Backtest engine with filter reapplication |
| `backtest/historical.py` | Historical price fetching, ContractPriceStore, generic daily series discovery |
| `backtest/metrics.py` | Sortino, Calmar, max DD, VaR/CVaR, block bootstrap CIs |
| `backtest/walkforward.py` | Walk-forward stability (calendar window partitioning) |
| `backtest/band_sweep.py` | Sweep lower band bound across allocation tilts |
| `backtest/allocation_analysis.py` | Full α-sweep: monotonicity, concavity, paired CIs |

## Config

Single YAML file at `fader/config/config.yaml`. Hot-reloaded by ConfigWatcher (5s poll of file mtime). Dashboard can also write live overrides to `config_kv` table. Slugs registry in `fader/config/slugs.csv`.

### slugs.csv columns

| Column | Description |
|---|---|
| `slug` | Market slug (virtual base slug for series) |
| `enabled` | 1 = active, 0 = skip |
| `market_kind` | `binary`, `ladder`, `series`, or `btc_daily` (alias) |
| `series_from_date` | First date to scan for series markets (YYYY-MM-DD) |
| `series_filter` | Substring matching individual child market slugs |
| `band_low` / `band_high` | Per-slug band override (empty = use global) |
| `size_override` | Per-slug notional override (empty = use global) |
| `added_at` | Date slug was added |
| `notes` | Human description |

## Database notes

SQLite in WAL mode. Both engine and dashboard read/write concurrently. Idempotency keys prevent duplicate orders on restart. Engine state published to `engine_state` KV table; dashboard commands go through `control_commands` table.

## Known issues

- **`error.md`**: Resolved — `use_container_width` deprecation fixed (all instances → `width='stretch'`).

## Original spec

`Prompt.txt` contains the full original specification from which this bot was built. Reference it for design intent questions.
