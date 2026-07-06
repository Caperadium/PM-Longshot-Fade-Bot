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

# Grid sweep backtest (parameter search)
python -c "import sys; sys.path.insert(0,'fader'); from backtest.grid_sweep import main; main()"

# IS/OOS backtest (in-sample / out-of-sample validation)
python -c "import sys; sys.path.insert(0,'fader'); from backtest.is_oos_backtest import main; main()"

# Crypto parameter sweep (BTC/ETH/SOL/XRP, band_low x alpha x DTE grid + walk-forward OOS)
python -c "import sys; sys.path.insert(0,'fader'); from backtest.crypto_sweep import main; main()"

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
4. Startup safety checks (USDC allowance; MATIC gate removed — CLOB orders are off-chain signed messages, no gas needed)
5. Full API reconciliation (bankroll, orders, positions)
6. Open CLOB websocket with exponential backoff
7. Wait for initial order books (REST resync on connect)
8. Launch concurrent tasks: strategy loop, order manager, pollers, state publisher, control consumer

### Runtime tasks (all concurrent asyncio)
| Task | Interval | Role |
|---|---|---|
| Strategy loop | 1s | 11-filter evaluation, calls OrderManager.enter() |
| Bankroll poller | 30s | Reconcile USDC/MATIC balances, USDC allowance |
| Resolution poller | 60s | Detect closed/resolved positions, book PnL (live: Data API; paper: polls Gamma `/markets?slug=` per open position, see below) |
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
| `ethereum-above-on` | `ethereum-above` | 2025-01-01 | ~1,800 ETH daily price-threshold binaries |
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
| `config/config_loader.py` | AppConfig, load_config, config hot-reload (5s mtime poll) |
| `execution/provider.py` | Polymarket REST wrapper (paper + live), MarketInfo, CLOB client |
| `execution/order_manager.py` | Order placement, market/limit dispatch, requote loop, TTL |
| `execution/sizing.py` | Alpha tilt notional sizing, band redistribution, $1.00 floor |
| `execution/idempotency.py` | Deterministic idempotency keys for all orders |
| `marketdata/ws_client.py` | Persistent CLOB websocket, reconnect, delta/snapshot handling, REST resync |
| `marketdata/book_state.py` | In-memory OrderBook per contract, band-entry timing tracker |
| `marketdata/staleness.py` | Per-contract staleness + feed-wide gap-halt |
| `marketdata/rest_market.py` | REST endpoints: DTE, volumes, series market discovery |
| `engine/risk.py` | Circuit breaker, max deployed, per-market caps |
| `engine/reconciler.py` | Startup + periodic API reconciliation (bankroll, orders, positions) |
| `engine/control_consumer.py` | Dashboard-to-engine IPC via control_commands table |
| `engine/pollers.py` | Bankroll (30s), resolution (60s), discovery (300s) background tasks |
| `engine/state_publisher.py` | Engine → DB state publishing (2s interval) |
| `infra/db.py` | SQLite WAL schema, all 10 tables + 11 indexes |
| `infra/telegram.py` | Telegram alerts (heartbeat, breaker trips, errors) |
| `infra/rate_limiter.py` | Token-bucket rate limiter for API calls |
| `infra/logging_setup.py` | Structured logging configuration |
| `persistence/decision_log.py` | Per-decision log persistence to decisions table |
| `dashboard/app.py` | 8-tab Streamlit dashboard |
| `dashboard/backtest_page.py` | Backtest UI (embedded + standalone) with parameter sweep |
| `backtest/engine.py` | Backtest engine with filter reapplication |
| `backtest/historical.py` | Historical price fetching, ContractPriceStore, generic daily series discovery |
| `backtest/metrics.py` | Sortino, Calmar, max DD, VaR/CVaR, block bootstrap CIs |
| `backtest/walkforward.py` | Walk-forward stability (calendar window partitioning) |
| `backtest/band_sweep.py` | Sweep lower band bound across allocation tilts |
| `backtest/allocation_analysis.py` | Full α-sweep: monotonicity, concavity, paired CIs |
| `backtest/grid_sweep.py` | Grid sweep backtest for parameter search |
| `backtest/is_oos_backtest.py` | IS/OOS (in-sample / out-of-sample) backtest validation |
| `backtest/crypto_sweep.py` | Per-market grid sweep (band_low x alpha x DTE) for BTC/ETH/SOL/XRP + walk-forward OOS validation of top configs |

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

**Paper positions persist across restarts.** Unlike earlier behavior (startup used to bulk-close every OPEN paper position with `realized_pnl=0.0`), the engine no longer wipes paper state on boot. `Reconciler._reconcile_paper_resolutions()` (delegated from `_reconcile_positions()` in paper mode, so it runs both on startup and every 60s resolution poll) polls Gamma per open paper position's slug and closes resolved ones with real signed PnL — unresolved positions stay OPEN, same as the bot holding them live. To start a fresh paper session, stop the engine and delete `fader/fader.db`.

## Known issues

- **`error.md`**: Resolved — `use_container_width` deprecation fixed (all instances → `width='stretch'`); STOP/CLOSE-ALL session-state crash fixed (widget-keyed state must be written via `on_click` callbacks, see dashboard/app.py).

## Original spec

`Prompt.txt` contains the full original specification from which this bot was built. Reference it for design intent questions.

## Change Logging & Documentation
After completing a task, do the following:

1. Append a single entry to `CHANGES.md` per logical task (not per file). Describe the intent and scope of the change, including which files were affected if relevant. Use present tense.

2. Check whether any of the following need updating and update them if so:
   - `CLAUDE.md` — architecture notes, file structure, conventions, anything describing how the project works
   - Anything in the DOCS folder — comprehensive MKdocs files with ALL information about the project

   Only update sections the change actually affects. Do not rewrite accurate sections.

## Pushing to GitHub
When I say "push" or "commit":
1. Read `CHANGES.md` and draft a commit message from it
2. Show me the message for approval
3. Commit and push
4. Clear `CHANGES.md`

## Never Do Without Asking
Before taking any of the following actions, stop and explicitly ask for confirmation:

- Refactoring code that wasn't part of the requested task
- Installing new dependencies
- Deleting or renaming files
- Changing function signatures, interfaces, or APIs
- Modifying configuration files (e.g. package.json, .env, docker, CI/CD)
- Making changes outside the files/scope I specified
- Resolving ambiguity by assumption — if the task is unclear, ask first

When in doubt about whether something falls outside the requested scope, ask.

## Temporary Files

All temporary artifacts — test scripts, summaries, test results, plans, reviews, debug dumps, scratch notes — go into `temp/`. Never leave them at repo root. The directory is gitignored; no need to clean up manually.

## Your Responsibilities
1. Ask, don't assume. If something is unclear, ask before writing a single line. Never make silent assumptions about intent, architecture, or requirements. When running unattended, pick the most reasonable interpretation, proceed, and record the assumption rather than blocking.

2. Implement the simplest solution for simple problems, better solutions for harder problems. Do not over-engineer or add flexibility that isn't needed yet. 

3. Don't touch unrelated code but please do surface bad code or design smells you discover with me so we can address them as a separate issue.

4. Flag uncertainty explicitly. If you're unsure about something, see point 1 above. If it makes sense to do so, conduct a small, localised and low-risk experiment and bring the hypothesis and results to me to discuss. Confidence without certainty causes more damage than admitting a gap.

5. I'm always open to ideas on better ways to do things. Please don't hesitate to suggest a better way, or one that has long lasting impact over a tactical change. (as a few examples)

Do not use non ASCII characters