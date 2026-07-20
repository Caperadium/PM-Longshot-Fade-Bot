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

# Calibration page (standalone; also embedded as 9th dashboard tab)
streamlit run fader/dashboard/calibration_page.py

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
Object construction is a composition root (`engine/build.py`'s `build_engine(cfg) -> Engine`): Db/repos -> limiter/executor -> provider -> risk -> order_manager -> strategy_loop -> reconciler -> pollers/state_publisher/ws, all wired via constructor injection (no late setters). `engine/main.py` itself only does:
1. Load config (config.yaml + slugs.csv) + init DB
2. `build_engine(cfg)` — construct every long-lived object
3. Resolve NO token IDs via Gamma API (`engine/startup.py`'s `resolve_markets`), populate the shared `MarketRegistry` (`engine/registry.py`) the composition root already wired into `ws_client`/`pollers`/`strategy_loop`
4. Startup safety checks (USDC allowance; MATIC gate removed — CLOB orders are off-chain signed messages, no gas needed)
5. Full API reconciliation (bankroll, orders, positions)
6. Open CLOB websocket with exponential backoff
7. Wait for initial order books (REST resync on connect)
8. Rehydrate resting orders (unverified against a `None` API response are kept and re-verified every requote tick, not dropped; a DB read failure here fires a telegram alert and aborts startup), wire bankroll source (+ `bankroll_view` for staleness logging), pre-warm volume cache (`engine/startup.py`)
9. Launch concurrent tasks: strategy loop, order manager, pollers, state publisher, control consumer (control-command dispatch + background config-watch/requote loops live in `engine/control.py`)

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
| Calibration fetch | 6h (`polling.calibration_fetch_s`, 0 disables) | Incremental update of `DATA/historical_prices.csv` for the Calibration tab: recent-window series discovery + resolution stamping of stored-but-unresolved markets (`update_calibration_data`); runs once at startup, then per interval; covers ALL slugs.csv rows incl. disabled |

### 11-filter entry stack (fader/strategy/filters.py + engine/strategy_loop.py)
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

Filters 1-8 are pure functions in `fader/strategy/filters.py`, shared by
the live engine and the backtest engine (Phase 4 of the architecture
refactor) -- one implementation instead of two independently-drifting
copies. Filters 9-11 (stateful: bankroll, cumulative deployed notional,
daily PnL) stay in `engine/risk.py`/`engine/strategy_loop.py`.

**Pregate/entry split** (`strategy/filters.py`): `evaluate_pregate(best_ask,
dte, params)` checks filters 1-2 only (cheap, no volumes/depth/staleness
inputs needed); `evaluate_entry(snapshot, params)` runs the full filters
1-8 stack against a fully-populated `EntrySnapshot`, calling
`evaluate_pregate` internally first. `engine/strategy_loop.py` calls
`evaluate_pregate` first and only fetches volumes (REST, 300s cache) to
build the full `EntrySnapshot` after it passes -- building a full snapshot
for every active market on every tick would be an API storm on every
cache expiry. The series-expanded slug loop keeps its pre-existing
asymmetry: it silently `continue`s on a pregate reject with no
`log_rejected` call (thousands of series markets/tick would flood the
decisions table), while the main slugs.csv loop logs every reject via
`log_rejected(slug, token_id, result.reason, result.detail)`.

**Per-field None semantics** (`FilterParams`/`EntrySnapshot`, normative --
no generic "None means pass" rule exists): `best_ask=None` always rejects
`no_book` in both engines. `dte=None` rejects `dte_out_of_range`
(fail-closed) when `missing_dte="reject"` (live); is skipped
(fail-open, matches today's backtest divergence) when `missing_dte="skip"`
(backtest). `volume_24h`/`volume_total`/`ask_depth_usd`/`is_stale = None`
each independently mark that filter as skipped (recorded in
`FilterResult.skipped`) rather than reject or silently pass.

**Paper-mode carve-outs** (both expressed as `FilterParams` flags, set
False only in paper mode): `check_staleness` (thinly-traded longshot-tail
markets get no WS deltas and go stale fast) and `check_time_in_band`
(those same markets rarely accrue enough continuous in-band WS time) --
two separate bypasses, both must be True for live's full gate.

**Backtest mapping** (`backtest/engine.py`): builds an `EntrySnapshot` per
row with `volume_24h`/`volume_total`/`ask_depth_usd`/`is_stale` always
`None` (not reconstructable from historical Polymarket data) and
`missing_dte="skip"`. `seconds_in_band = days_in_band * 86400` and
`min_time_in_band_s = max(1, min_time_in_band_days) * 86400` are
algebraically identical to the old day-count comparison; the stateful
`days_in_band` counter reset/increment ordering and the `price <= 0`
counter-reset-and-skip stay in the day loop as procedural code -- only the
actual band/DTE/time-in-band comparisons were centralized into the shared
core. `run_backtest()` attaches `trades_df.attrs["skipped_filters"]`
(always the 3 volume/depth filter names) to the returned DataFrame without
changing its `(trades_df, equity_df)` 2-tuple return shape.
`backtest/metrics.py`'s `compute_all_metrics(..., skipped_filters=...)`
reads this to build `universe_discrepancies` dynamically; omitting the
argument (every pre-Phase-4 caller) reproduces the historical fixed
3-item list unchanged.

### Order placement (execution/order_manager.py)
- Spread ≤ 1c → market order (FOK)
- Spread > 1c → limit at mid, requote on 0.5c mid move, TTL 5min, cancel on band exit
- All orders carry deterministic idempotency keys (execution/idempotency.py)
- Position row inserted immediately on market fill (closes 60s reconciler gap)
- `provider.place_order()` returns a typed `OrderResult` (`FILLED|PENDING|REJECTED|DUPLICATE|UNKNOWN`); `_place_market`/`_place_limit` dispatch on `result.status` — the same code path handles paper (instant `FILLED`) and live (`PENDING`/`REJECTED`) placement, no more `provider.is_paper` branching inside the placement methods. `DUPLICATE` (keyed by the `is_duplicate_error` signal, from either a CLOB success=false body or an exception string) runs `find_order_by_params` recovery via `_handle_duplicate`, falling back to `UNKNOWN` (no position insert) if recovery fails.

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
| `config/config_loader.py` | AppConfig, load_config, config hot-reload (5s mtime poll). `DEFAULT_WS_URL` is the single authoritative default for the CLOB market-data websocket URL (`FeedConfig.ws_url` and `marketdata/ws_client.py`'s constructor default both reference it). `apply_config_kv_overrides(cfg)` returns the list of YAML keys currently shadowed by a `config_kv` row; `ConfigWatcher.check_and_reload` logs that list and publishes it to `engine_state` as `active_overrides` (rendered in the dashboard sidebar) |
| `strategy/filters.py` | Pure, I/O-free shared filter core (Phase 4): `FilterParams`, `EntrySnapshot`, `FilterResult`, `evaluate_pregate` (filters 1-2), `evaluate_entry` (filters 1-8). Used by both `engine/strategy_loop.py` (live) and `backtest/engine.py` (backtest) — one implementation instead of two |
| `execution/provider.py` | `BaseProvider`/`LiveProvider`/`PaperProvider` split, `OrderResult` dataclass, `MarketInfo`, CLOB client, `make_provider(cfg,...)` factory, `Provider(...)` compatibility alias, `is_paper` property |
| `execution/order_manager.py` | Order placement, market/limit dispatch on `OrderResult.status`, `_handle_duplicate` recovery, requote loop, TTL |
| `execution/sizing.py` | Alpha tilt notional sizing, band redistribution, $1.00 floor |
| `execution/idempotency.py` | Deterministic idempotency keys for all orders |
| `marketdata/ws_client.py` | Persistent CLOB websocket, reconnect, delta/snapshot handling, REST resync. `_handle_message` dispatches per event-type via a `self._dispatch` dict (built once in `__init__` from `_on_book`/`_on_price_change`/etc.) instead of an if/elif chain; `_sync_book(token_id)` is the shared staleness-touch + band-tracker-update tail for the book/price_change handlers. Constructor's `ws_url` default is `config.config_loader.DEFAULT_WS_URL` (no more local `WS_URL` constant) |
| `marketdata/book_state.py` | In-memory OrderBook per contract, band-entry timing tracker |
| `marketdata/staleness.py` | Per-contract staleness + feed-wide gap-halt |
| `marketdata/rest_market.py` | REST endpoints: DTE, volumes, series market discovery. `parse_market_outcomes(raw) -> dict` (+ `OUTCOME_NO` constant) centralizes Gamma `clobTokenIds`/`outcomes` parsing and NO-outcome lookup, used by `discover_new_rungs`, `discover_series_markets`, and `execution.provider.LiveProvider.resolve_no_token` |
| `engine/build.py` | Composition root: `build_engine(cfg) -> Engine` dataclass, constructor-injects provider/risk/order_manager/strategy_loop/reconciler/pollers/state_publisher/ws_client |
| `engine/startup.py` | Startup-sequence helpers: token resolution (binary + series expansion), volume cache pre-warm |
| `engine/control.py` | Control-command dispatch (`make_on_command`) + background config-watch/requote loops; needs process-lifecycle state (`ConfigWatcher`, `stop_event`) so it stays outside `build.py` |
| `engine/registry.py` | `MarketRegistry`: single owner of `{slug: MarketInfo}` (replaces the former shared `token_map` dict). `get`/`add`/`mark_resolved`/`active_items` (snapshot copy)/`slugs`; no `await` inside any method. Shared by `strategy_loop`, `pollers`, the ws `market_resolved` callback, and `main.py`/`build.py` |
| `engine/risk.py` | Circuit breaker, max deployed, per-market caps. `breaker_tripped` is a property that reads through `BreakerRepo.day_state(today_utc())`, memoized <=1s — no in-memory trip flag, so a trip survives a `RiskManager` restart and a new UTC day simply has no DB row (no explicit rollover step) |
| `engine/reconciler.py` | Startup + periodic API reconciliation (bankroll, orders, positions); skips the order-reconcile cycle when `fetch_open_orders()` returns `None` (API error) instead of mass-marking orders UNKNOWN. `bankroll` stays a plain `float` (five consumers depend on the exact type); `bankroll_view` is a separate `BankrollView(value, as_of)` property used only for staleness logging. Tracks consecutive reconcile failures on two separate counters: `reconcile_failures` (positions/paper-resolutions) and `order_reconcile_failures` (order-reconcile `None`-skip path); both published to `engine_state` and fire `infra.telegram.alert_reconcile_failures` (`>=` threshold of 5, refires every cycle at/above it) after 5 in a row, reset on the next success. The stale-UNKNOWN-to-CANCELLED order reaper (in `_reconcile_orders`) commits via `OrdersRepo.reap_stale_unknown()` with no shared conn — a pre-existing no-commit bug fixed in Phase 6 — and now compares a Python-computed ISO cutoff against `created_at` directly (was a raw-string comparison against SQLite's differently-formatted `datetime()` output, which made the effective TTL ~1 day instead of the intended 1 hour; fixed alongside the escalation-counter split above) |
| `engine/control_consumer.py` | Dashboard-to-engine IPC via control_commands table |
| `engine/pollers.py` | Bankroll (30s), resolution (60s), discovery (300s), calibration fetch (6h) background tasks; discovery mutates the shared `MarketRegistry` (`engine/registry.py`) instead of a bare dict; `_calibration_loop` runs `update_calibration_data` via `asyncio.to_thread`, telegram escalation after 5 consecutive failures |
| `engine/state_publisher.py` | Engine → DB state publishing (2s interval) |
| `infra/db.py` | SQLite WAL schema, all 10 tables + 11 indexes; low-level `get_connection`/`execute_write` primitives that `persistence/repos.py` builds on |
| `infra/telegram.py` | Telegram alerts (heartbeat every `telegram.heartbeat_minutes` (default 3h), breaker trips, errors) + inbound commands: `CommandListenerTask` long-polls getUpdates (sole consumer; incompatible with a webhook on the bot) and answers `/bankroll` from the configured chat with bankroll / open positions / deployed / PnL today+total via a stats fn injected from `engine/main.py` |
| `infra/ipv4.py` | `force_ipv4()`: patches urllib3 address-family selection so all HTTP egress (requests, py-clob-client, Telegram) is IPv4-only; ws_client separately passes `family=AF_INET`. Needed on hosts with unroutable IPv6 to Cloudflare (OVH VPS). Called at engine startup + dashboard load |
| `infra/rate_limiter.py` | Token-bucket rate limiter for API calls |
| `infra/logging_setup.py` | Structured logging configuration |
| `persistence/repos.py` | Typed repository layer (`Db`, `PositionsRepo`, `OrdersRepo`, `BreakerRepo`, `DecisionsRepo`, `ControlRepo`, `EngineStateRepo`, `ConfigKVRepo`) — all engine-side SQL lives here. Module-level default instances (`positions_repo`, `orders_repo`, etc.) are used directly by engine code this phase; `Db.transaction()` gives atomic multi-statement writes (e.g. order-fill bookkeeping). Every method takes an optional `conn`: passed-in conn -> caller owns commit/close; `None` -> repo opens/commits/closes per call. |
| `persistence/decision_log.py` | Per-decision log persistence to decisions table; delegates to `DecisionsRepo`. `log_decision`/`log_entered`/`log_rejected` return `bool` (success/failure of the write) |
| `dashboard/app.py` | 9-tab Streamlit dashboard |
| `dashboard/backtest_page.py` | Backtest UI (embedded + standalone) with parameter sweep |
| `dashboard/calibration_page.py` | Calibration tab (embedded + standalone): implied YES prob vs actual YES resolution rate from the historical price store; DTE-slider observation point, window/series filters, per-bucket table, monthly edge trend, bot-realized-calibration section (biased sample, labelled). Band/DTE defaults come from the engine's EFFECTIVE config (`load_config()` + `apply_config_kv_overrides`), so dashboard-written `config_kv` overrides are honored. Loads store via mtime-keyed `st.cache_data(ttl=300)` — invalidated by the engine's atomic CSV replace |
| `backtest/calibration.py` | Pure, I/O-free calibration core: `CalibrationParams`, `wilson_interval`, `build_observations` (one obs per market at `end_date - dte_days`, nearest row within tolerance), `bucket_calibration`, `band_summary` (NO-side headline), `monthly_edge`, `bot_trade_calibration`, `filter_by_series`. Store price is the NO mid, so implied YES = 1 - price; positive edge = longshots overpriced = thesis holds |
| `backtest/engine.py` | Backtest engine; filters 1-8 delegate to `strategy/filters.py` (Phase 4) with `missing_dte="skip"` (fail-open DTE) and volumes/depth/staleness always `None` (not reconstructable historically) — `run_backtest()` attaches the always-skipped filter names to `trades_df.attrs["skipped_filters"]`. Sizing delegates to `execution.sizing.compute_shares_and_notional` (Phase 5; local `_compute_size` deleted) — only the share-count element is used, `BacktestTrade.notional` stays the stake |
| `backtest/historical.py` | Historical price fetching, ContractPriceStore (atomic `save()`: tmp + `os.replace`, retried on Windows reader locks), generic daily series discovery. `update_calibration_data` is the engine poller's incremental updater: recent-window series discovery (`_series_scan_start`, self-healing back to newest stored end_date after outages), fetch of new tokens, and `select_stale_unresolved` re-fetch that stamps resolutions onto markets first fetched while still open (skip-existing logic would otherwise never resolve them). Engine poller + dashboard fetch button are two writers, last-writer-wins; lost resolution stamps self-heal next cycle |
| `backtest/metrics.py` | Sortino, Calmar, max DD, VaR/CVaR, block bootstrap CIs; `compute_all_metrics(..., skipped_filters=...)` builds `universe_discrepancies` from the actual skipped-filter set when provided, else the historical fixed 3-item list |
| `backtest/walkforward.py` | Walk-forward stability (calendar window partitioning) |
| `backtest/harness.py` | Shared backtest harness (Phase 5): `HarnessDefaults` dataclass (defaults CLIs override via `dataclasses.replace()` when they diverge, e.g. `crypto_sweep.CRYPTO_DEFAULTS`); `load_store()`; `run_config()` (`run_backtest` + `compute_all_metrics` + flat `MetricsRow` extraction); `run_grid()` (spawn-safe multiprocessing — module-level worker fn, `df_json` passed inside each worker's args, no pool initializer). Also holds all THREE walk-forward variants as separate functions — `walkforward_normalized` (= former `allocation_analysis.walkforward_validate`, globally-normalized sizing), `walkforward_lean` (= former `grid_sweep._oos_validate`, unnormalized sizing + per-band baseline cache), `window_stability` (= former `crypto_sweep._run_walkforward_for_top_configs`, same fixed config re-tested per window, no baseline comparison) — kept distinct because their sizing/comparison math differs; do not merge them |
| `backtest/report.py` | Shared report-formatting pieces (Phase 5): `metrics_table` (column-aligned text table from row dicts), `caveats_section` (renders the known backtest-vs-live filter gaps, reading a run's `skipped_filters` when available), `write_report` (compose sections, mkdir + write to a path). The five CLIs' own report bodies are otherwise unchanged — content-identical, not byte-identical, is the bar |
| `backtest/band_sweep.py` | Sweep lower band bound across allocation tilts; loads data via `harness.load_store()`, runs configs via `harness.run_config()`, writes reports via `report.write_report()` — no per-CLI divergence from `HarnessDefaults` |
| `backtest/allocation_analysis.py` | Full α-sweep: monotonicity, concavity, paired CIs. `walkforward_validate` is now a thin delegate to `harness.walkforward_normalized`; its analysis-only helpers (`_run_scheme`, `_make_sized_fn`, `compute_normalization_factor`, `extract_entry_prices`) stay here and are imported back into `harness.py` (deferred, to avoid a circular import) |
| `backtest/grid_sweep.py` | Grid sweep backtest for parameter search. `_run_lean`/`_metrics_row`/`BandCache`/`_build_band_cache`/`_oos_validate` are now thin wrappers around `harness.build_band_cache`/`harness.walkforward_lean` with this CLI's original defaults ($10 notional, 1-day min-time-in-band, 1c spread) preserved |
| `backtest/is_oos_backtest.py` | IS/OOS (in-sample / out-of-sample) backtest validation; `_run_candidate` calls `harness.run_config()` instead of duplicating `run_backtest` + `compute_all_metrics` |
| `backtest/crypto_sweep.py` | Per-market grid sweep (band_low x alpha x DTE) for BTC/ETH/SOL/XRP + walk-forward OOS validation of top configs. Declares `CRYPTO_DEFAULTS = replace(HarnessDefaults(), order_notional_usd=25.0)` at the top (the $25-vs-$10 divergence from the shared default is named and greppable). `_run_chunk`'s multiprocessing pattern (module-level worker fn, `df_json` serialized inside each worker's args tuple) is the reference `harness.run_grid()` copies. `_run_walkforward_for_top_configs` is a thin wrapper around `harness.window_stability` |

## Config

Single YAML file at `fader/config/config.yaml`. Hot-reloaded by ConfigWatcher (5s poll of file mtime). `polling.calibration_fetch_s` (default 21600) controls the calibration data poller; `0` disables it (rechecked every 300s, so hot-reload can re-enable without restart). Dashboard can also write live overrides to `config_kv` table. Slugs registry in `fader/config/slugs.csv`. Every hot-reload logs which YAML keys are currently shadowed by a `config_kv` row and publishes them to `engine_state` as `active_overrides`; the dashboard sidebar shows a caption listing any active overrides so a config.yaml edit that appears to have no effect is easy to diagnose.

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