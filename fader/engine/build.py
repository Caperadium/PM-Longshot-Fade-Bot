"""engine/build.py

Composition root (Phase 2 of the architecture refactor,
temp/implementation-plan.md). Builds every long-lived engine object with
constructor injection -- no late setters, no "OrderManager not set" guard.

`build_engine(cfg)` wires objects in this order:
  Db/repos -> limiter/executor -> provider -> risk -> order_manager
  -> strategy_loop (OrderManager passed to the constructor)
  -> reconciler (same) -> pollers/publisher/ws.

`ControlConsumer` is NOT built here: its callbacks are constructor-only
(no late setter) and the real dispatch closure needs `ConfigWatcher`, the
shutdown `stop_event`, and `restart_requested` -- all process-lifecycle
state that engine/main.py owns per the plan ("task gather, signal
handling"). main.py constructs it directly after build_engine() returns.

`registry` (MarketRegistry, engine/registry.py) is shared mutable state
(Phase 3, Single-owner state -- replaces the former bare `token_map`
dict): build_engine creates the (initially empty) MarketRegistry and
closes the ws/pollers callbacks over it, then engine/main.py's startup
sequence (token resolution) populates it before calling
ws.start(token_ids). The registry object identity never changes, so the
callbacks constructed here see the populated registry once startup has
run.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from config.config_loader import AppConfig
from infra.rate_limiter import RateLimiter
from marketdata.book_state import BookStore
from marketdata.staleness import StalenessTracker
from marketdata.ws_client import WsClient
from execution.provider import make_provider, BaseProvider
from execution.order_manager import OrderManager
from engine.registry import MarketRegistry
from engine.risk import RiskManager
from engine.strategy_loop import StrategyLoop
from engine.reconciler import Reconciler
from engine.pollers import Pollers
from engine.state_publisher import StatePublisher
from persistence import repos as repos_module
from strategy.model_pricer import ModelPricer

logger = logging.getLogger(__name__)


@dataclass
class Engine:
    cfg: AppConfig
    repos: Any
    provider: BaseProvider
    risk: RiskManager
    order_manager: OrderManager
    strategy_loop: StrategyLoop
    reconciler: Reconciler
    pollers: Pollers
    state_publisher: StatePublisher
    ws_client: WsClient
    book_store: BookStore
    staleness: StalenessTracker
    executor: ThreadPoolExecutor
    registry: MarketRegistry = field(default_factory=MarketRegistry)
    series_slugs: List[str] = field(default_factory=list)
    model_pricer: Any = None


def _make_on_market_resolved(registry: MarketRegistry) -> Callable:
    async def on_market_resolved(event: Dict) -> None:
        resolved_token = event.get("asset_id", "")
        slug = registry.mark_resolved(resolved_token)
        if slug:
            logger.info(
                f"Market resolved via WS: {slug} ({resolved_token[:16]}...)"
            )
            from infra import telegram
            telegram.fire(telegram.send(f"Market resolved: <b>{slug}</b>"))

    return on_market_resolved


def _make_on_slugs_changed(strategy_loop: StrategyLoop) -> Callable:
    async def on_slugs_changed(new_slugs: List[Dict]) -> None:
        for s in new_slugs:
            slug = s["slug"]
            if slug not in strategy_loop._series_slugs:
                strategy_loop._series_slugs.append(slug)

    return on_slugs_changed


def build_engine(cfg: AppConfig) -> Engine:
    """Construct every long-lived engine object. Pure wiring -- no I/O,
    no token resolution, no reconcile calls (those stay in engine/main.py's
    startup sequence, which runs after this returns)."""

    # ------------------------------------------------------------------
    # Db/repos
    # ------------------------------------------------------------------
    repos = repos_module  # module-level singletons (Phase 1); Engine.repos
                           # exposes the module itself as the repos bundle.

    # ------------------------------------------------------------------
    # limiter / executor
    # ------------------------------------------------------------------
    executor = ThreadPoolExecutor(max_workers=cfg.feed.executor_workers)

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

    # ------------------------------------------------------------------
    # provider
    # ------------------------------------------------------------------
    provider = make_provider(cfg, limiter=rl, executor=executor, repos=repos)

    # ------------------------------------------------------------------
    # risk
    # ------------------------------------------------------------------
    risk = RiskManager(
        daily_loss_pct=cfg.risk.daily_loss_breaker_pct,
        max_deployed_pct=cfg.risk.max_deployed_pct,
        per_market_cap_pct=cfg.risk.per_market_cap_pct,
    )

    # ------------------------------------------------------------------
    # order_manager
    # ------------------------------------------------------------------
    order_manager = OrderManager(cfg=cfg, provider=provider, risk=risk)

    # ------------------------------------------------------------------
    # registry -- shared mutable state populated by main.py's startup
    # sequence (token resolution) after build_engine returns. Single
    # owner (Phase 3): ws/pollers/strategy_loop all hold this same
    # MarketRegistry instance rather than reaching into a bare dict.
    # ------------------------------------------------------------------
    registry = MarketRegistry()

    # ------------------------------------------------------------------
    # model pricer (FIGARCH edge signal, strategy/model_pricer.py).
    # Always constructed -- pricer.enabled is hot-reloadable and re-read
    # on every evaluate(); construction imports nothing heavy (the
    # pricing engine loads lazily in the worker thread). Ladder computes
    # run on the shared REST executor.
    # ------------------------------------------------------------------
    model_pricer = ModelPricer(cfg.pricer, submit_fn=executor.submit)

    # ------------------------------------------------------------------
    # strategy_loop (OrderManager injected via constructor -- no
    # set_order_manager late setter)
    # ------------------------------------------------------------------
    strategy_loop = StrategyLoop(
        cfg=cfg, book_store=book_store, staleness=staleness, risk=risk,
        order_manager=order_manager, registry=registry,
        model_pricer=model_pricer,
    )

    # ------------------------------------------------------------------
    # reconciler (OrderManager injected via constructor -- no
    # reconciler._order_manager = late write)
    # ------------------------------------------------------------------
    reconciler = Reconciler(provider=provider, risk=risk, order_manager=order_manager)

    # ------------------------------------------------------------------
    # ws
    # ------------------------------------------------------------------
    resync_concurrency = min(cfg.feed.resync_concurrency, cfg.ratelimit.read_burst)
    ws = WsClient(
        book_store=book_store,
        staleness=staleness,
        ws_url=cfg.feed.ws_url,
        new_market_cb=None,
        market_resolved_cb=_make_on_market_resolved(registry),
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

    # ------------------------------------------------------------------
    # pollers / state publisher
    # ------------------------------------------------------------------
    pollers = Pollers(
        cfg=cfg,
        reconciler=reconciler,
        ws_client=ws,
        registry=registry,
        slugs_changed_cb=_make_on_slugs_changed(strategy_loop),
        order_manager=order_manager,
    )

    state_pub = StatePublisher(
        ws_client=ws,
        book_store=book_store,
        staleness=staleness,
        risk=risk,
        reconciler=reconciler,
    )

    return Engine(
        cfg=cfg,
        repos=repos,
        provider=provider,
        risk=risk,
        order_manager=order_manager,
        strategy_loop=strategy_loop,
        reconciler=reconciler,
        pollers=pollers,
        state_publisher=state_pub,
        ws_client=ws,
        book_store=book_store,
        staleness=staleness,
        executor=executor,
        registry=registry,
        model_pricer=model_pricer,
    )
