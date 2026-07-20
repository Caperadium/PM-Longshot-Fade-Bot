"""engine/pollers.py

Background polling tasks:
  - Bankroll reconcile + MATIC balance + USDC allowance (30s)
  - Resolution status + orders reconciliation (60s)
  - New-rung discovery for ladder markets (5 min)
  - Calibration data fetch for the dashboard Calibration tab (6h default)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from config.config_loader import AppConfig
from engine.registry import MarketRegistry
from persistence.repos import control_repo, decisions_repo

logger = logging.getLogger(__name__)


class Pollers:
    def __init__(
        self,
        cfg: AppConfig,
        reconciler,
        ws_client,
        registry: MarketRegistry,
        slugs_changed_cb: Optional[Callable] = None,
        order_manager=None,
    ) -> None:
        self._cfg = cfg
        self._reconciler = reconciler
        self._ws = ws_client
        self._registry = registry
        self._slugs_changed_cb = slugs_changed_cb
        self._order_manager = order_manager
        self._tasks: List[asyncio.Task] = []

    def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._bankroll_loop()),
            asyncio.create_task(self._resolution_loop()),
            asyncio.create_task(self._discovery_loop()),
            asyncio.create_task(self._maintenance_loop()),
            asyncio.create_task(self._calibration_loop()),
        ]

    def stop(self) -> None:
        for t in self._tasks:
            t.cancel()

    async def _bankroll_loop(self) -> None:
        while True:
            try:
                await self._reconciler._reconcile_bankroll()
                # USDC allowance (informational only — SELL orders need it)
                risk = self._reconciler._risk
                provider = self._reconciler._provider
                try:
                    if not provider.is_paper:
                        allowance = await provider.async_fetch_usdc_allowance()
                        if allowance < 1.0:
                            logger.warning(f"USDC.e allowance low: {allowance:.2f}")
                except Exception as e:
                    logger.warning(f"Allowance fetch error: {e}")
            except Exception as e:
                logger.error(f"Bankroll poll error: {e}")
            await asyncio.sleep(self._cfg.polling.bankroll_s)

    async def _resolution_loop(self) -> None:
        while True:
            try:
                await self._reconciler._reconcile_positions()
                await self._reconciler._reconcile_orders()
            except Exception as e:
                logger.error(f"Resolution poll error: {e}")
            await asyncio.sleep(self._cfg.polling.resolution_s)

    async def _discovery_loop(self) -> None:
        while True:
            await asyncio.sleep(self._cfg.polling.discovery_s)
            try:
                await self._discover_new_rungs()
                await self._discover_new_series()
            except Exception as e:
                logger.error(f"Discovery poll error: {e}")

    async def _maintenance_loop(self) -> None:
        """Hourly DB retention pruning.

        The decisions table gets one row per in-band candidate per 1s tick,
        which grows unbounded and fills a small VPS disk within months.
        Keep 14 days of decisions and processed control commands.
        """
        loop = asyncio.get_running_loop()
        while True:
            try:
                await loop.run_in_executor(None, self._prune_old_rows)
            except Exception as e:
                logger.error(f"Maintenance poll error: {e}")
            await asyncio.sleep(3600)

    @staticmethod
    def _prune_old_rows(retention_days: int = 14) -> None:
        n1 = decisions_repo.prune(retention_days)
        n2 = control_repo.prune(retention_days)
        if n1 or n2:
            logger.info(
                f"DB retention prune: {n1} decisions, {n2} control_commands "
                f"older than {retention_days}d removed"
            )

    async def _calibration_loop(self) -> None:
        """Keep DATA/historical_prices.csv fresh for the dashboard's
        Calibration tab (yes-resolution-rate vs implied-probability
        tracking). Runs ``update_calibration_data`` (backtest/historical.py)
        off the event loop: recent-window series discovery, fetch of any
        token not yet in the store, and a re-check of tokens fetched while
        still open so their resolution eventually gets stamped once Gamma
        reports the market closed.

        Runs once immediately at startup (so the tab has fresh-ish data
        without waiting a full interval), then sleeps
        ``cfg.polling.calibration_fetch_s`` between runs. An interval
        ``<= 0`` disables the fetch itself but keeps checking every 5min so
        a hot-reloaded config can re-enable it without an engine restart.
        Exceptions never escape this loop -- a calibration-fetch failure
        must not take down the engine.
        """
        failures = 0
        while True:
            interval = self._cfg.polling.calibration_fetch_s
            if interval <= 0:
                await asyncio.sleep(300)
                continue
            try:
                await asyncio.to_thread(self._run_calibration_update)
                failures = 0
            except Exception as e:
                failures += 1
                logger.error(f"Calibration fetch error ({failures} consecutive): {e}")
                if failures >= 5:
                    from infra import telegram
                    telegram.fire(
                        telegram.alert_reconcile_failures(
                            failures, "calibration_fetch", str(e)
                        )
                    )
            await asyncio.sleep(interval)

    def _run_calibration_update(self) -> None:
        """Blocking body of the calibration fetch, run via
        ``asyncio.to_thread`` from ``_calibration_loop``. Deferred import
        of ``backtest.historical`` mirrors ``_discover_new_rungs``'s
        deferred import of ``marketdata.rest_market`` -- keeps this
        module's own import graph light.
        """
        from backtest.historical import ContractPriceStore, update_calibration_data

        # Fresh store instance each cycle: picks up any rows the
        # dashboard's Backtest-tab fetch button wrote since the last
        # cycle (two-writer situation -- see update_calibration_data's
        # docstring for the last-writer-wins / self-healing tradeoff).
        store = ContractPriceStore()
        # ALL slugs.csv rows, including disabled ones -- the calibration
        # universe is intentionally broader than the live trading universe
        # (e.g. it should still track a series that's temporarily disabled).
        slugs = [s.slug for s in self._cfg.slugs]
        configs = {s.slug: s for s in self._cfg.slugs}

        stats = update_calibration_data(
            store, slugs, configs,
            progress=lambda m: logger.info(f"[calib] {m}"),
        )
        logger.info(f"Calibration update stats: {stats}")

    async def _discover_new_rungs(self) -> None:
        from marketdata.rest_market import discover_new_rungs
        from execution.provider import MarketInfo

        tracked_slugs = [s.slug for s in self._cfg.enabled_slugs()]
        kind_map = {s.slug: s.market_kind for s in self._cfg.slugs}

        loop = asyncio.get_event_loop()
        new_rungs = await loop.run_in_executor(
            None,
            lambda: discover_new_rungs(tracked_slugs, kind_map),
        )
        if not new_rungs:
            return

        for rung in new_rungs:
            slug = rung["slug"]
            if slug in self._registry:
                continue
            self._registry.add(slug, MarketInfo(
                slug=slug,
                condition_id=rung.get("condition_id", ""),
                token_id=rung["token_id"],
                outcome=rung.get("outcome", "No"),
                outcome_index=rung.get("outcome_index", 0),
                question="",
                end_date_iso=rung.get("end_date_iso", ""),
                active=rung.get("active", True),
                closed=False,
            ))
            await self._ws.subscribe_tokens([rung["token_id"]])
            logger.info(f"Discovered new rung: {slug}")

        if new_rungs and self._slugs_changed_cb:
            await self._slugs_changed_cb(new_rungs)

    async def _discover_new_series(self) -> None:
        from marketdata.rest_market import (
            discover_series_markets,
            _derive_series_filter,
            parse_series_date,
        )
        from execution.provider import MarketInfo

        loop = asyncio.get_event_loop()
        today = datetime.now(timezone.utc).date()
        added_all: List[Dict] = []

        for slug_row in self._cfg.enabled_slugs():
            if slug_row.market_kind not in ("series", "btc_daily"):
                continue
            series_filter = slug_row.series_filter or _derive_series_filter(slug_row.slug)
            from_date = max(
                parse_series_date(slug_row.series_from_date),
                today - timedelta(days=7),
            )

            new_children = await loop.run_in_executor(
                None,
                lambda: discover_series_markets(
                    base_slug=slug_row.slug,
                    series_filter=series_filter,
                    from_date=from_date,
                    to_date=today + timedelta(days=30),
                    progress=lambda msg: logger.info(msg),
                ),
            )

            for child in new_children:
                if child["slug"] in self._registry:
                    continue
                if any(
                    mi.token_id == child["token_id"]
                    for _, mi in self._registry.active_items()
                ):
                    continue
                self._registry.add(child["slug"], MarketInfo(
                    slug=child["slug"],
                    condition_id=child.get("condition_id", "") or child["token_id"],
                    token_id=child["token_id"],
                    outcome="No",
                    outcome_index=0,
                    question=child.get("question", ""),
                    end_date_iso=child.get("end_date", ""),
                    active=child.get("active", True),
                    closed=bool(child.get("resolution", "")),
                ))
                await self._ws.subscribe_tokens([child["token_id"]])
                added_all.append(child)

            if added_all:
                logger.info(
                    f"Discovered {len(added_all)} new series markets "
                    f"for {slug_row.slug}"
                )

        # Notify upstream so StrategyLoop._series_slugs gets updated
        if added_all and self._slugs_changed_cb:
            await self._slugs_changed_cb(added_all)
