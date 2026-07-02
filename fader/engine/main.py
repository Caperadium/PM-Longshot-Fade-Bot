"""engine/main.py

asyncio entrypoint for the fader trading engine.

Startup sequence (per plan §Infrastructure):
  1. Load config -> init DB
  2. Resolve token_ids for all enabled slugs
  3. Reconcile open orders/positions/bankroll vs API (ground truth)
  4. Resync books via REST /book
  5. Connect websocket
  6. Only then start strategy loop

Run:
    python -m engine.main
    (or: python fader/engine/main.py)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add fader root to path when run directly
_FADER_ROOT = Path(__file__).resolve().parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))

from dotenv import load_dotenv
from config.config_loader import load_config, ConfigWatcher, apply_config_kv_overrides
from infra.logging_setup import setup_logging
from infra.db import init_db
from infra import telegram
from infra.rate_limiter import RateLimiter
from marketdata.book_state import BookStore
from marketdata.staleness import StalenessTracker
from marketdata.ws_client import WsClient
from execution.provider import Provider, MarketInfo
from execution.order_manager import OrderManager
from marketdata.rest_market import (
    discover_series_markets,
    _derive_series_filter,
    parse_series_date,
    fetch_volumes,
)
from engine.risk import RiskManager
from engine.strategy_loop import StrategyLoop
from engine.reconciler import Reconciler
from engine.pollers import Pollers
from engine.state_publisher import StatePublisher
from engine.control_consumer import ControlConsumer

logger = logging.getLogger(__name__)


async def run() -> None:
    # ------------------------------------------------------------------
    # 1. Config + DB
    # ------------------------------------------------------------------
    load_dotenv(_FADER_ROOT / ".env")  # load .env from fader/
    cfg = load_config()
    setup_logging(
        level="INFO",
        log_file=str(_FADER_ROOT / "engine.log"),
        json_console=False,
    )
    init_db()
    apply_config_kv_overrides(cfg)  # overlay dashboard-written params
    telegram.configure(enabled=cfg.telegram.enabled)

    logger.info(f"Fader engine starting [mode={cfg.mode}]")

    # ------------------------------------------------------------------
    # 2. Core objects
    # ------------------------------------------------------------------
    loop = asyncio.get_running_loop()

    # L2: sized ThreadPoolExecutor for blocking REST calls — set as the
    # default loop executor (so bare run_in_executor(None, ...) calls in
    # strategy_loop/ws_client/etc. also use it) and passed explicitly to
    # Provider.
    executor = ThreadPoolExecutor(max_workers=cfg.feed.executor_workers)
    loop.set_default_executor(executor)

    rl = RateLimiter(
        write_per_s=cfg.ratelimit.write_per_s,
        write_burst=cfg.ratelimit.write_burst,
        read_per_s=cfg.ratelimit.read_per_s,
        read_burst=cfg.ratelimit.read_burst,
    )
    book_store = BookStore()
    staleness = StalenessTracker(
        max_staleness_s=cfg.feed.max_staleness_seconds,
        gap_halt_s=cfg.feed.gap_halt_seconds,
    )
    risk = RiskManager(
        daily_loss_pct=cfg.risk.daily_loss_breaker_pct,
        max_deployed_pct=cfg.risk.max_deployed_pct,
        per_market_cap_pct=cfg.risk.per_market_cap_pct,
    )
    provider = Provider(
        limiter=rl, mode=cfg.mode, executor=executor,
        paper_bankroll_usdc=cfg.bankroll.paper_bankroll_usdc,
    )
    reconciler = Reconciler(provider=provider, risk=risk, order_manager=None)  # set below

    # ------------------------------------------------------------------
    # 3. Resolve token IDs for all enabled slugs
    # ------------------------------------------------------------------
    token_map: Dict[str, Any] = {}
    series_slugs: List[str] = []     # series children tracked separately
    today = datetime.now(timezone.utc).date()

    for slug_row in cfg.enabled_slugs():
        if slug_row.market_kind in ("series", "btc_daily"):
            # --- Series expansion path ---
            series_filter = slug_row.series_filter or _derive_series_filter(slug_row.slug)
            from_date = parse_series_date(slug_row.series_from_date)
            start = max(from_date, today - timedelta(days=7))
            forward = min(cfg.strategy.max_dte + 3, 30)
            end = today + timedelta(days=forward)

            try:
                children = await loop.run_in_executor(
                    None,
                    lambda: discover_series_markets(
                        base_slug=slug_row.slug,
                        series_filter=series_filter,
                        from_date=start,
                        to_date=end,
                        progress=lambda msg: logger.info(msg),
                    ),
                )
            except Exception as e:
                logger.error(f"Could not discover series {slug_row.slug}: {e} — skipping")
                continue

            for child in children:
                # Dedup by token_id
                if any(mi.token_id == child["token_id"] for mi in token_map.values()):
                    continue
                cid = child.get("condition_id", "") or child["token_id"]
                token_map[child["slug"]] = MarketInfo(
                    slug=child["slug"],
                    condition_id=cid,
                    token_id=child["token_id"],
                    outcome="No",
                    outcome_index=0,
                    question=child.get("question", ""),
                    end_date_iso=child.get("end_date", ""),
                    active=child.get("active", True),
                    closed=bool(child.get("resolution", "")),
                )
                series_slugs.append(child["slug"])
            logger.info(
                f"Series {slug_row.slug}: expanded to {len(children)} markets "
                f"(filter='{series_filter}', {start} -> {end})"
            )
        else:
            # --- Existing binary/ladder path (unchanged) ---
            try:
                market_info = await loop.run_in_executor(
                    None, lambda s=slug_row.slug: provider.resolve_no_token(s)
                )
                token_map[slug_row.slug] = market_info
                logger.info(
                    f"Resolved {slug_row.slug} -> NO token {market_info.token_id[:16]}..."
                )
            except Exception as e:
                logger.error(f"Could not resolve {slug_row.slug}: {e} — skipping")

    # Warn about resolved/closed slugs in config
    for slug, mi in token_map.items():
        if mi.closed:
            logger.warning(f"Slug {slug!r} is CLOSED — consider removing from slugs.csv")
        elif not mi.active:
            logger.warning(f"Slug {slug!r} is INACTIVE — will be skipped at runtime")

    if not token_map:
        logger.warning("No slugs resolved. Engine will run but strategy loop idle.")

    # ------------------------------------------------------------------
    # 3.5 Startup safety checks
    # ------------------------------------------------------------------
    # MATIC gate removed — CLOB orders are off-chain signed messages, no gas needed.

    if cfg.mode != "paper":
        try:
            allowance = await provider.async_fetch_usdc_allowance()
            logger.info(f"Startup USDC.e allowance: {allowance:.2f}")
            if allowance < 1.0:
                logger.warning(f"USDC.e allowance {allowance:.2f} — SELL orders may fail")
        except Exception as e:
            logger.warning(f"Startup USDC allowance check failed: {e}")

    # ------------------------------------------------------------------
    # 4. Startup reconcile (API as ground truth)
    # ------------------------------------------------------------------
    await reconciler.full_reconcile()

    # ------------------------------------------------------------------
    # 5. Websocket
    # ------------------------------------------------------------------
    token_ids = [m.token_id for m in token_map.values() if not m.closed and m.active]

    # Callback: mark resolved markets as inactive/closed
    async def on_market_resolved(event: Dict) -> None:
        resolved_token = event.get("asset_id", "")
        for slug, mi in token_map.items():
            if mi.token_id == resolved_token:
                mi.active = False
                mi.closed = True
                logger.info(
                    f"Market resolved via WS: {slug} ({resolved_token[:16]}...)"
                )
                from infra import telegram
                telegram.fire(telegram.send(f"Market resolved: <b>{slug}</b>"))
                break

    # FIX 2: clamp resync concurrency to the read-rate burst so reconnect
    # resync can't trigger a 429 storm.
    resync_concurrency = min(cfg.feed.resync_concurrency, cfg.ratelimit.read_burst)
    ws = WsClient(
        book_store=book_store,
        staleness=staleness,
        ws_url=cfg.feed.ws_url,
        new_market_cb=None,
        market_resolved_cb=on_market_resolved,
        band_low=cfg.strategy.band_low,
        band_high=cfg.strategy.band_high,
        resync_concurrency=resync_concurrency,
    )
    ws.set_watchdog(
        cfg.feed.ws_force_reconnect_s,
        cfg.feed.ws_ping_interval_s,
        cfg.feed.ws_pong_timeout_s,
        cfg.feed.ws_expect_pong,
    )
    await ws.start(token_ids)

    # Wait for initial books (up to 5s)
    for _ in range(50):
        await asyncio.sleep(0.1)
        if ws.connected:
            break
    logger.info(f"WS connected: {ws.connected}")

    # ------------------------------------------------------------------
    # 6. Strategy loop + order manager
    # ------------------------------------------------------------------
    order_manager = OrderManager(cfg=cfg, provider=provider, risk=risk)
    reconciler._order_manager = order_manager  # wire after creation

    # FIX 3: rehydrate any resting LIMIT orders that survived a restart,
    # before any background task (requote_loop) starts reading _resting.
    if cfg.mode != "paper":
        try:
            n_rehydrated = await order_manager.rehydrate_resting()
            if n_rehydrated:
                logger.info(f"Startup rehydrate: {n_rehydrated} resting order(s) recovered")
        except Exception as e:
            logger.error(f"Startup rehydrate_resting failed: {e}")

    strategy_loop = StrategyLoop(
        cfg=cfg,
        book_store=book_store,
        staleness=staleness,
        risk=risk,
    )
    strategy_loop.set_token_map(token_map)
    strategy_loop.set_order_manager(order_manager)
    strategy_loop.set_bankroll(reconciler.bankroll)
    # Live source: risk caps + breaker track the 30s bankroll poller,
    # not the value captured at startup.
    strategy_loop.set_bankroll_source(lambda: reconciler.bankroll)
    strategy_loop.set_series_slugs(series_slugs)

    # Pre-warm volume cache for series slugs (avoids 500+ cold Gamma calls on first tick)
    if series_slugs:
        logger.info(
            f"Pre-warming volume cache for {len(series_slugs)} series markets..."
        )
        batch_size = 10
        for i in range(0, len(series_slugs), batch_size):
            batch = series_slugs[i:i + batch_size]
            tasks = [
                loop.run_in_executor(
                    None, lambda s=slug: fetch_volumes(s)
                )
                for slug in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for slug, result in zip(batch, results):
                if isinstance(result, dict):
                    strategy_loop._volume_cache[slug] = result
                    strategy_loop._volume_cache_ts[slug] = time.monotonic()
                else:
                    logger.warning(
                        f"Volume pre-warm failed for {slug}: {result!r}"
                    )
            if i + batch_size < len(series_slugs):
                await asyncio.sleep(0.1)
        logger.info(
            f"Volume cache pre-warm complete ({len(series_slugs)} slugs)"
        )

    # Callback: update strategy loop when poller discovers new series children
    async def on_slugs_changed(new_slugs: List[Dict]) -> None:
        for s in new_slugs:
            slug = s["slug"]
            if slug not in strategy_loop._series_slugs:
                strategy_loop._series_slugs.append(slug)

    # Pollers
    pollers = Pollers(
        cfg=cfg,
        reconciler=reconciler,
        ws_client=ws,
        token_map=token_map,
        slugs_changed_cb=on_slugs_changed,
        order_manager=order_manager,
    )

    # State publisher
    state_pub = StatePublisher(
        ws_client=ws,
        book_store=book_store,
        staleness=staleness,
        risk=risk,
        reconciler=reconciler,
    )

    # Config watcher
    config_watcher = ConfigWatcher(cfg)

    # Control consumer
    async def on_command(cmd: str, args: Dict) -> None:
        if cmd == "stop":
            logger.info("Stop command received")
            await strategy_loop.stop()
        elif cmd == "start":
            logger.info("Start command received")
            await strategy_loop.start()
        elif cmd == "close_all":
            logger.info("Close-all command received")
            await order_manager.close_all()
        elif cmd == "breaker_reset":
            risk.reset_breaker()
        elif cmd == "config_reload":
            config_watcher.check_and_reload()
            apply_config_kv_overrides(cfg)  # re-apply dashboard overrides
            risk.update_params(
                cfg.risk.daily_loss_breaker_pct,
                cfg.risk.max_deployed_pct,
                cfg.risk.per_market_cap_pct,
            )
            staleness.set_params(
                cfg.feed.max_staleness_seconds,
                cfg.feed.gap_halt_seconds,
            )
            ws.set_band(cfg.strategy.band_low, cfg.strategy.band_high)
            ws.set_watchdog(
                cfg.feed.ws_force_reconnect_s,
                cfg.feed.ws_ping_interval_s,
                cfg.feed.ws_pong_timeout_s,
                cfg.feed.ws_expect_pong,
            )
            await order_manager.cancel_resting_for_disabled_slugs()

    control = ControlConsumer(
        poll_s=cfg.polling.control_poll_s,
        stop_engine_cb=on_command,
        close_all_cb=on_command,
        breaker_reset_cb=on_command,
        config_reload_cb=on_command,
    )

    # Config watch task
    async def config_watch_loop() -> None:
        while True:
            try:
                if config_watcher.check_and_reload():
                    risk.update_params(
                        cfg.risk.daily_loss_breaker_pct,
                        cfg.risk.max_deployed_pct,
                        cfg.risk.per_market_cap_pct,
                    )
                    staleness.set_params(
                        cfg.feed.max_staleness_seconds,
                        cfg.feed.gap_halt_seconds,
                    )
                    ws.set_band(cfg.strategy.band_low, cfg.strategy.band_high)
                    ws.set_watchdog(
                        cfg.feed.ws_force_reconnect_s,
                        cfg.feed.ws_ping_interval_s,
                        cfg.feed.ws_pong_timeout_s,
                        cfg.feed.ws_expect_pong,
                    )
                    strategy_loop.set_bankroll(reconciler.bankroll)
                    await order_manager.cancel_resting_for_disabled_slugs()
            except Exception as e:
                logger.error(f"Config watch error: {e}")
            await asyncio.sleep(5.0)

    # Requote watch task
    async def requote_loop() -> None:
        while True:
            try:
                await order_manager.requote_check(book_store)
            except Exception as e:
                logger.error(f"Requote loop error: {e}")
            await asyncio.sleep(1.0)

    # ------------------------------------------------------------------
    # Start all tasks
    # ------------------------------------------------------------------
    await strategy_loop.start()
    pollers.start()
    state_pub.start()
    control.start()
    heartbeat = telegram.HeartbeatTask(interval_minutes=cfg.telegram.heartbeat_minutes)
    heartbeat.start()

    asyncio.create_task(config_watch_loop())
    asyncio.create_task(requote_loop())

    await telegram.alert_start(cfg.mode)

    logger.info("All engine tasks started. Running.")

    # Keep running until cancelled
    try:
        stop_event = asyncio.Event()

        def _handle_signal(*_):
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _handle_signal)
            except (NotImplementedError, RuntimeError):
                pass  # Windows

        await stop_event.wait()
    except asyncio.CancelledError:
        pass

    logger.info("Engine shutting down")
    await strategy_loop.stop()

    # FIX 5: cancel resting limit orders before the process exits so they
    # don't orphan on the exchange. Timeout path leaves them PENDING in
    # DB; rehydrate_resting recovers them on next boot.
    if cfg.mode != "paper":
        try:
            await asyncio.wait_for(order_manager.cancel_all_resting("shutdown"), timeout=10)
        except asyncio.TimeoutError:
            logger.warning("cancel_all_resting timed out; orphans handled by rehydrate on next boot")
        except Exception as e:
            logger.warning(f"cancel_all_resting error: {e}")

    await ws.stop()
    pollers.stop()
    state_pub.stop()
    control.stop()
    heartbeat.stop()
    await telegram.alert_stop("graceful shutdown")
    executor.shutdown(wait=False)


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
