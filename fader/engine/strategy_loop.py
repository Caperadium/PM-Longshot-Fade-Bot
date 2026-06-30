"""engine/strategy_loop.py

Core strategy decision loop. Runs every decision_interval_s.

Applies the 11-filter stack per NO contract per enabled slug.
On pass: calls OrderManager to place an order.
Every pass/reject writes a decisions row.

Filter stack (all must pass, all configurable):
  1. NO ask in [band_low, band_high]
  2. DTE in [min_dte, max_dte]
  3. Continuously in band >= min_time_in_band_s
  4. Market 24h volume >= min_24h_volume
  5. Market cumulative volume >= min_total_volume
  6. Book depth at NO touch >= min_book_depth (USD notional)
  7. Freshness: last ws update <= max_staleness_seconds and feed not gap-halted
  8. No existing OPEN position on this exact contract
  9. Per-market exposure cap not breached
  10. Total deployed cap not breached
  11. Daily-loss circuit breaker not tripped
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from config.config_loader import AppConfig
from engine.risk import RiskManager, get_open_notional
from execution.sizing import MIN_EFFECTIVE_NOTIONAL, make_sizing_fn
from marketdata.book_state import BookStore
from marketdata.staleness import StalenessTracker
from marketdata.rest_market import fetch_volumes, dte as compute_dte
from persistence.decision_log import log_entered, log_rejected

logger = logging.getLogger(__name__)


class StrategyLoop:
    """
    Autonomous decision loop.

    token_map: {slug: MarketInfo}  — resolved by reconciler/startup
    order_manager: OrderManager    — placed import at runtime to avoid circular
    bankroll: float                — updated by pollers
    """

    def __init__(
        self,
        cfg: AppConfig,
        book_store: BookStore,
        staleness: StalenessTracker,
        risk: RiskManager,
    ) -> None:
        self._cfg = cfg
        self._books = book_store
        self._staleness = staleness
        self._risk = risk
        self._token_map: Dict[str, Any] = {}  # slug -> MarketInfo
        self._order_manager = None
        self._bankroll: float = 0.0
        self._volume_cache: Dict[str, Dict[str, float]] = {}
        self._volume_cache_ts: Dict[str, float] = {}
        self._volume_ttl_s: float = 300.0
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._series_slugs: List[str] = []
        self._tick_count: int = 0

    def set_token_map(self, token_map: Dict[str, Any]) -> None:
        self._token_map = token_map

    def set_order_manager(self, order_manager) -> None:
        self._order_manager = order_manager

    def set_bankroll(self, bankroll: float) -> None:
        self._bankroll = bankroll

    def set_series_slugs(self, slugs: List[str]) -> None:
        self._series_slugs = slugs

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            tick_start = time.monotonic()
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Strategy loop error: {e}", exc_info=True)

            elapsed = time.monotonic() - tick_start
            sleep = max(0.0, self._cfg.feed.decision_interval_s - elapsed)
            await asyncio.sleep(sleep)

    async def _tick(self) -> None:
        cfg = self._cfg
        # Check gap halt first (paper mode: never halt — stale books are usable)
        if cfg.mode != "paper" and self._staleness.check_gap_halt():
            return

        # Load deployment state once per tick
        total_deployed, by_slug = get_open_notional()

        # --- Main slug loop (slugs.csv entries with per-slug config) ---
        for slug_row in cfg.enabled_slugs():
            slug = slug_row.slug
            market_info = self._token_map.get(slug)
            if market_info is None:
                continue
            if not market_info.active or market_info.closed:
                continue
            token_id = market_info.token_id
            book = self._books.get(token_id)

            band_low, band_high = cfg.band_for_slug(slug)
            if book is None or book.best_ask is None:
                log_rejected(slug, token_id, "no_book", {"band_low": band_low})
                continue
            ask_f = float(book.best_ask)
            if not (band_low <= ask_f <= band_high):
                log_rejected(slug, token_id, "ask_out_of_band", {
                    "no_ask": ask_f, "band_low": band_low, "band_high": band_high,
                })
                continue

            dte_val = await self._get_dte(slug)
            if dte_val is None or not (cfg.strategy.min_dte <= dte_val <= cfg.strategy.max_dte):
                log_rejected(slug, token_id, "dte_out_of_range", {
                    "dte": dte_val, "min_dte": cfg.strategy.min_dte,
                    "max_dte": cfg.strategy.max_dte,
                })
                continue

            notional = cfg.size_for_slug(slug)
            delta, _ = await self._evaluate_and_enter(
                slug=slug, token_id=token_id, book=book,
                band_low=band_low, band_high=band_high,
                notional=notional, market_info=market_info,
                total_deployed=total_deployed, by_slug=by_slug,
                ask_f=ask_f, log_decisions=True,
            )
            total_deployed += delta

        # --- Series-expanded slugs (no per-slug overrides — use globals) ---
        for slug in self._series_slugs:
            mi = self._token_map.get(slug)
            if mi is None or mi.closed or not mi.active:
                continue
            token_id = mi.token_id
            book = self._books.get(token_id)
            if book is None or book.best_ask is None:
                continue

            ask_f = float(book.best_ask)
            band_low, band_high = cfg.strategy.band_low, cfg.strategy.band_high
            if not (band_low <= ask_f <= band_high):
                continue

            dte_val = self._dte_from_end_date(mi.end_date_iso)
            if dte_val is None or not (cfg.strategy.min_dte <= dte_val <= cfg.strategy.max_dte):
                continue

            notional = cfg.strategy.order_notional_usd
            delta, _ = await self._evaluate_and_enter(
                slug=slug, token_id=token_id, book=book,
                band_low=band_low, band_high=band_high,
                notional=notional, market_info=mi,
                total_deployed=total_deployed, by_slug=by_slug,
                ask_f=ask_f, log_decisions=True,
            )
            total_deployed += delta

        # Prune closed slugs from series list every 60 ticks
        self._tick_count += 1
        if self._tick_count % 60 == 0:
            self._series_slugs = [
                s for s in self._series_slugs
                if not (self._token_map.get(s) and self._token_map[s].closed)
            ]

    async def _evaluate_and_enter(
        self,
        slug: str,
        token_id: str,
        book,
        band_low: float,
        band_high: float,
        notional: float,
        market_info,
        total_deployed: float,
        by_slug: Dict[str, float],
        ask_f: float,
        log_decisions: bool = True,
    ) -> Tuple[float, float]:
        """Shared filter evaluation (filters 3-11) for main + series loops.

        Filters 1-2 (ask in band, DTE) are pre-checked by caller.
        Returns (delta_deployed, delta_market) — (0.0, 0.0) on reject.
        """
        cfg = self._cfg
        filters: Dict[str, Any] = {}

        # -- Filter 3: continuously in band --
        # Paper mode: skip the min-time-in-band gate. Longshot-tail markets
        # trade thinly, so they rarely accrue enough continuous in-band WS
        # time; the gate would starve paper runs of entries.
        time_in_band = book.time_in_band()
        if cfg.mode != "paper" and not book.is_in_band_long_enough(cfg.strategy.min_time_in_band_s):
            if log_decisions:
                filters["time_in_band_s"] = time_in_band
                filters["min_time_in_band_s"] = cfg.strategy.min_time_in_band_s
                log_rejected(slug, token_id, "not_in_band_long_enough", filters)
            return (0.0, 0.0)

        # -- Filters 4 & 5: volumes --
        vols = await self._get_volumes(slug)
        if vols["volume_24h"] < cfg.filters.min_24h_volume:
            if log_decisions:
                filters.update(vols)
                filters["min_24h_volume"] = cfg.filters.min_24h_volume
                log_rejected(slug, token_id, "low_24h_volume", filters)
            return (0.0, 0.0)
        if vols["volume_total"] < cfg.filters.min_total_volume:
            if log_decisions:
                filters.update(vols)
                filters["min_total_volume"] = cfg.filters.min_total_volume
                log_rejected(slug, token_id, "low_total_volume", filters)
            return (0.0, 0.0)

        # -- Filter 6: book depth at NO touch --
        ask_depth_usd = float(book.ask_depth_usd)
        if cfg.filters.min_book_depth > 0 and ask_depth_usd < cfg.filters.min_book_depth:
            if log_decisions:
                filters["ask_depth_usd"] = ask_depth_usd
                filters["min_book_depth"] = cfg.filters.min_book_depth
                log_rejected(slug, token_id, "insufficient_depth", filters)
            return (0.0, 0.0)

        # -- Filter 7: staleness / gap --
        # Paper mode: skip staleness — markets that don't trade actively
        # (i.e. the longshot tail) get no WS deltas and go stale fast.
        if self._cfg.mode != "paper" and self._staleness.is_stale(token_id):
            if log_decisions:
                log_rejected(slug, token_id, "stale_data", {"stale": True})
            return (0.0, 0.0)

        # -- Filter 8: no existing OPEN position --
        if self._has_open_position(token_id):
            if log_decisions:
                log_rejected(slug, token_id, "position_already_open", {})
            return (0.0, 0.0)

        # -- Filters 9, 10, 11: risk --
        alpha = cfg.strategy.alpha
        if alpha != 0.0:
            fn = make_sizing_fn(alpha, band_low, band_high)
            notional = max(MIN_EFFECTIVE_NOTIONAL, notional * fn(ask_f))
        market_deployed = by_slug.get(slug, 0.0)
        allowed, risk_reason = self._risk.allow_entry(
            slug, notional, self._bankroll, total_deployed, market_deployed
        )
        if not allowed:
            if log_decisions:
                log_rejected(slug, token_id, "risk_cap", {"risk_reason": risk_reason})
            return (0.0, 0.0)

        if self._risk.check_breaker_against_bankroll(self._bankroll):
            if log_decisions:
                log_rejected(slug, token_id, "circuit_breaker", {})
            return (0.0, 0.0)

        # -- All filters passed: enter --
        if self._order_manager is None:
            logger.warning("OrderManager not set; skipping entry")
            return (0.0, 0.0)

        await self._order_manager.enter(
            slug=slug,
            token_id=token_id,
            book=book,
            notional=notional,
            filters={"band_low": band_low, "band_high": band_high},
            market_info=market_info,
        )
        return (notional, notional)

    def _has_open_position(self, token_id: str) -> bool:
        from infra.db import get_connection
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT 1 FROM positions WHERE token_id=? AND status='OPEN' LIMIT 1",
                (token_id,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    @staticmethod
    def _dte_from_end_date(end_date_iso: str) -> Optional[float]:
        """Compute DTE from MarketInfo.end_date_iso. No Gamma call."""
        if not end_date_iso:
            return None
        try:
            end = datetime.fromisoformat(end_date_iso.rstrip("Z"))
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            delta = end - datetime.now(timezone.utc)
            return max(0.0, delta.total_seconds() / 86400)
        except ValueError:
            return None

    async def _get_dte(self, slug: str) -> Optional[float]:
        # Fast path: compute from cached MarketInfo
        mi = self._token_map.get(slug)
        if mi and mi.end_date_iso:
            dte = self._dte_from_end_date(mi.end_date_iso)
            if dte is not None:
                return dte
        # Fallback: Gamma API
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, lambda: compute_dte(slug))
        except Exception:
            return None

    async def _get_volumes(self, slug: str) -> Dict[str, float]:
        now = time.monotonic()
        cached_ts = self._volume_cache_ts.get(slug, 0.0)
        if now - cached_ts < self._volume_ttl_s and slug in self._volume_cache:
            return self._volume_cache[slug]
        loop = asyncio.get_running_loop()
        try:
            vols = await loop.run_in_executor(None, lambda: fetch_volumes(slug))
        except Exception:
            vols = {"volume_24h": 0.0, "volume_total": 0.0}
        self._volume_cache[slug] = vols
        self._volume_cache_ts[slug] = now
        return vols
