"""engine/control.py

Control-command dispatch + long-running background maintenance loops
(config hot-reload watcher, requote loop), extracted from engine/main.py
(Phase 2 of the architecture refactor, temp/implementation-plan.md) to
keep main.py under the target line budget.

These are NOT part of the composition root (engine/build.py) because the
command dispatcher needs process-lifecycle state main.py owns
(ConfigWatcher instance, the shutdown stop_event, restart_requested) --
see engine/build.py's module docstring for why ControlConsumer itself is
still constructed in main.py.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from config.config_loader import apply_config_kv_overrides

logger = logging.getLogger(__name__)


def make_on_command(engine, cfg, config_watcher, stop_event, restart_requested):
    """Control-command dispatcher. Closes over process-lifecycle state
    (config_watcher, stop_event, restart_requested) that belongs to
    main.py, not the composition root."""

    async def on_command(cmd: str, args: Dict[str, Any]) -> None:
        if cmd == "stop":
            logger.info("Stop command received")
            await engine.strategy_loop.stop()
        elif cmd == "restart":
            logger.info("Restart command received — graceful shutdown + cold start")
            restart_requested["flag"] = True
            stop_event.set()
        elif cmd == "start":
            logger.info("Start command received")
            await engine.strategy_loop.start()
        elif cmd == "close_all":
            logger.info("Close-all command received")
            await engine.order_manager.close_all()
        elif cmd == "breaker_reset":
            engine.risk.reset_breaker()
        elif cmd == "config_reload":
            apply_hot_reload(engine, cfg, config_watcher)
            await engine.order_manager.cancel_resting_for_disabled_slugs()

    return on_command


def apply_hot_reload(engine, cfg, config_watcher) -> None:
    config_watcher.check_and_reload()
    apply_config_kv_overrides(cfg)  # re-apply dashboard overrides
    engine.risk.update_params(
        cfg.risk.daily_loss_breaker_pct,
        cfg.risk.max_deployed_pct,
        cfg.risk.per_market_cap_pct,
    )
    engine.staleness.set_params(cfg.feed.max_staleness_seconds, cfg.feed.gap_halt_seconds)
    engine.ws_client.set_band(cfg.strategy.band_low, cfg.strategy.band_high)
    engine.ws_client.set_watchdog(
        cfg.feed.ws_force_reconnect_s, cfg.feed.ws_ping_interval_s,
        cfg.feed.ws_pong_timeout_s, cfg.feed.ws_expect_pong,
    )


async def config_watch_loop(engine, cfg, config_watcher) -> None:
    while True:
        try:
            if config_watcher.check_and_reload():
                apply_hot_reload(engine, cfg, config_watcher)
                engine.strategy_loop.set_bankroll(engine.reconciler.bankroll)
                await engine.order_manager.cancel_resting_for_disabled_slugs()
        except Exception as e:
            logger.error(f"Config watch error: {e}")
        await asyncio.sleep(5.0)


async def requote_loop(engine) -> None:
    while True:
        try:
            await engine.order_manager.requote_check(engine.book_store)
        except Exception as e:
            logger.error(f"Requote loop error: {e}")
        await asyncio.sleep(1.0)
