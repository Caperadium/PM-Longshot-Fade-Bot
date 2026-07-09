"""engine/main.py

asyncio entrypoint for the fader trading engine.

Startup sequence (per plan §Infrastructure):
  1. Load config -> init DB
  2. build_engine(cfg) -- composition root, all constructor wiring
  3. Resolve token_ids for all enabled slugs
  4. Reconcile open orders/positions/bankroll vs API (ground truth)
  5. Resync books via REST /book
  6. Connect websocket
  7. Only then start strategy loop

Object construction lives in engine/build.py; startup-sequence helpers
(token resolution, volume pre-warm) live in engine/startup.py;
control-command dispatch + background maintenance loops live in
engine/control.py (Phase 2 of the architecture refactor,
temp/implementation-plan.md). This module keeps only config/logging
setup, the build_engine() call, startup sequencing, task gather, and
signal handling.

Run:
    python -m engine.main
    (or: python fader/engine/main.py)
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add fader root to path when run directly
_FADER_ROOT = Path(__file__).resolve().parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))

from dotenv import load_dotenv
from config.config_loader import load_config, ConfigWatcher, apply_config_kv_overrides
from infra.logging_setup import setup_logging
from infra.ipv4 import force_ipv4
from infra.db import init_db
from infra import telegram
from engine.build import build_engine
from engine.control_consumer import ControlConsumer
from engine.startup import resolve_markets, prewarm_volume_cache
from engine.control import make_on_command, config_watch_loop, requote_loop

logger = logging.getLogger(__name__)


async def run() -> int:
    # ------------------------------------------------------------------
    # 1. Config + DB
    # ------------------------------------------------------------------
    load_dotenv(_FADER_ROOT / ".env")  # load .env from fader/
    cfg = load_config()
    setup_logging(
        level="INFO", log_file=str(_FADER_ROOT / "engine.log"), json_console=False,
    )
    force_ipv4()  # Cloudflare IPv6 egress is unroutable on this host; prefer IPv4
    init_db()
    apply_config_kv_overrides(cfg)  # overlay dashboard-written params
    telegram.configure(enabled=cfg.telegram.enabled)
    logger.info(f"Fader engine starting [mode={cfg.mode}]")

    # ------------------------------------------------------------------
    # 2. Composition root — all constructor wiring happens here
    # ------------------------------------------------------------------
    loop = asyncio.get_running_loop()
    engine = build_engine(cfg)
    # L2: sized ThreadPoolExecutor for blocking REST calls — set as the
    # default loop executor (bare run_in_executor(None, ...) calls in
    # strategy_loop/ws_client/etc. also use it).
    loop.set_default_executor(engine.executor)

    # ------------------------------------------------------------------
    # 3. Resolve token IDs for all enabled slugs
    # ------------------------------------------------------------------
    resolved, series_slugs = await resolve_markets(cfg, engine.provider, loop)
    for slug, mi in resolved.active_items():
        engine.registry.add(slug, mi)  # ws/pollers/strategy_loop share this registry
    engine.strategy_loop.set_series_slugs(series_slugs)

    # ------------------------------------------------------------------
    # 3.5 Startup safety checks
    # ------------------------------------------------------------------
    # MATIC gate removed — CLOB orders are off-chain signed messages, no gas needed.
    if cfg.mode != "paper":
        try:
            allowance = await engine.provider.async_fetch_usdc_allowance()
            logger.info(f"Startup USDC.e allowance: {allowance:.2f}")
            if allowance < 1.0:
                logger.warning(f"USDC.e allowance {allowance:.2f} — SELL orders may fail")
        except Exception as e:
            logger.warning(f"Startup USDC allowance check failed: {e}")

    # ------------------------------------------------------------------
    # 4. Startup reconcile (API as ground truth)
    # ------------------------------------------------------------------
    await engine.reconciler.full_reconcile()

    # ------------------------------------------------------------------
    # 5. Websocket
    # ------------------------------------------------------------------
    token_ids = [
        mi.token_id for _, mi in engine.registry.active_items()
        if not mi.closed and mi.active
    ]
    await engine.ws_client.start(token_ids)
    for _ in range(50):  # wait for initial books, up to 5s
        await asyncio.sleep(0.1)
        if engine.ws_client.connected:
            break
    logger.info(f"WS connected: {engine.ws_client.connected}")

    # ------------------------------------------------------------------
    # 6. Rehydrate + bankroll wiring
    # ------------------------------------------------------------------
    # FIX 3: rehydrate any resting LIMIT orders that survived a restart,
    # before any background task (requote_loop) starts reading _resting.
    #
    # rehydrate_resting() only raises when the DB read itself fails (API
    # errors are handled internally -- rows are kept unverified instead).
    # A DB read failure means we can't rebuild the resting-order set at
    # all, so let it propagate: main.py aborts startup (Phase 3, Single-
    # owner state). The supervisor (systemd Restart=always /
    # run_engine_supervised.py) bounds the resulting crash-loop.
    if cfg.mode != "paper":
        n_rehydrated = await engine.order_manager.rehydrate_resting()
        if n_rehydrated:
            logger.info(f"Startup rehydrate: {n_rehydrated} resting order(s) recovered")

    engine.strategy_loop.set_bankroll(engine.reconciler.bankroll)
    # Live source: risk caps + breaker track the 30s bankroll poller,
    # not the value captured at startup.
    engine.strategy_loop.set_bankroll_source(lambda: engine.reconciler.bankroll)
    # Second, additive source: (value, as_of) pair used only to log
    # bankroll_age_s in the risk-cap decision filters dict when acting on
    # a stale reconcile (Phase 3) -- does not touch the float plumbing above.
    engine.strategy_loop.set_bankroll_view_source(
        lambda: engine.reconciler.bankroll_view, cfg.polling.bankroll_s,
    )
    await prewarm_volume_cache(engine.strategy_loop, series_slugs, loop)

    # ------------------------------------------------------------------
    # 7. Config watcher + control consumer + signal handling
    # ------------------------------------------------------------------
    config_watcher = ConfigWatcher(cfg)

    # Process-lifecycle signalling. stop_event is set by SIGINT/SIGTERM or by
    # the stop/restart control commands; restart_requested distinguishes a
    # cold-restart (exit 42 -> supervisor relaunches) from a plain shutdown.
    stop_event = asyncio.Event()
    restart_requested = {"flag": False}
    on_command = make_on_command(engine, cfg, config_watcher, stop_event, restart_requested)

    control = ControlConsumer(
        poll_s=cfg.polling.control_poll_s,
        stop_engine_cb=on_command,
        close_all_cb=on_command,
        breaker_reset_cb=on_command,
        config_reload_cb=on_command,
    )

    # ------------------------------------------------------------------
    # Start all tasks
    # ------------------------------------------------------------------
    await engine.strategy_loop.start()
    engine.pollers.start()
    engine.state_publisher.start()
    control.start()
    heartbeat = telegram.HeartbeatTask(interval_minutes=cfg.telegram.heartbeat_minutes)
    heartbeat.start()

    asyncio.create_task(config_watch_loop(engine, cfg, config_watcher))
    asyncio.create_task(requote_loop(engine))

    await telegram.alert_start(cfg.mode)
    logger.info("All engine tasks started. Running.")

    # Keep running until cancelled
    try:
        def _handle_signal(*_):
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _handle_signal)
            except (NotImplementedError, RuntimeError):
                pass  # Windows

        await stop_event.wait()
    except asyncio.CancelledError:
        pass

    logger.info("Engine shutting down")
    await engine.strategy_loop.stop()

    # FIX 5: cancel resting limit orders before the process exits so they
    # don't orphan on the exchange. Timeout path leaves them PENDING in
    # DB; rehydrate_resting recovers them on next boot.
    if cfg.mode != "paper":
        try:
            await asyncio.wait_for(
                engine.order_manager.cancel_all_resting("shutdown"), timeout=10,
            )
        except asyncio.TimeoutError:
            logger.warning("cancel_all_resting timed out; orphans handled by rehydrate on next boot")
        except Exception as e:
            logger.warning(f"cancel_all_resting error: {e}")

    await engine.ws_client.stop()
    engine.pollers.stop()
    engine.state_publisher.stop()
    control.stop()
    heartbeat.stop()
    await telegram.alert_stop(
        "cold restart" if restart_requested["flag"] else "graceful shutdown"
    )
    engine.executor.shutdown(wait=False)

    # Sentinel 42 tells the supervisor (systemd Restart=always / the Windows
    # run_engine_supervised.py wrapper) to relaunch — a full cold start that
    # re-runs load_dotenv, telegram.configure and full_reconcile.
    return 42 if restart_requested["flag"] else 0


def main() -> None:
    try:
        code = asyncio.run(run())
    except KeyboardInterrupt:
        code = 0
    sys.exit(code or 0)


if __name__ == "__main__":
    main()
