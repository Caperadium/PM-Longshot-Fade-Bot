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
from typing import Any, Dict, List, Optional, Tuple

from config.config_loader import AppConfig
from engine.registry import MarketRegistry
from engine.risk import RiskManager
from execution.sizing import MIN_EFFECTIVE_NOTIONAL, make_sizing_fn
from marketdata.book_state import BookStore
from marketdata.staleness import StalenessTracker
from marketdata.rest_market import fetch_volumes, dte as compute_dte
from persistence.decision_log import log_entered, log_rejected
from persistence.repos import positions_repo
from strategy.filters import EntrySnapshot, FilterParams, evaluate_entry, evaluate_pregate

logger = logging.getLogger(__name__)


class StrategyLoop:
    """
    Autonomous decision loop.

    registry: MarketRegistry       — owns {slug: MarketInfo}, resolved by
                                      reconciler/startup (Phase 3: single
                                      owner replacing the shared token_map
                                      dict)
    order_manager: OrderManager    — injected via constructor (Phase 2:
                                      no more late set_order_manager setter)
    bankroll: float                — updated by pollers
    """

    def __init__(
        self,
        cfg: AppConfig,
        book_store: BookStore,
        staleness: StalenessTracker,
        risk: RiskManager,
        order_manager=None,
        registry: Optional[MarketRegistry] = None,
        model_pricer=None,
    ) -> None:
        self._cfg = cfg
        self._books = book_store
        self._staleness = staleness
        self._risk = risk
        self._registry = registry if registry is not None else MarketRegistry()
        self._order_manager = order_manager
        self._model_pricer = model_pricer  # strategy.model_pricer.ModelPricer or None
        self._bankroll: float = 0.0
        self._bankroll_fn = None  # optional live source (e.g. reconciler.bankroll)
        self._bankroll_view_fn = None  # optional BankrollView source (age logging)
        self._bankroll_poll_s: float = 30.0  # for the age-staleness threshold
        self._volume_cache: Dict[str, Dict[str, float]] = {}
        self._volume_cache_ts: Dict[str, float] = {}
        self._volume_ttl_s: float = 300.0
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._series_slugs: List[str] = []
        self._tick_count: int = 0

    def load_markets(self, markets: Dict[str, Any]) -> None:
        """Replace the registry's contents from a plain {slug: MarketInfo}
        mapping (test convenience / bulk-load entry point). Internally
        backed by MarketRegistry (Phase 3, single-owner state) instead of a
        bare shared dict."""
        self._registry = MarketRegistry()
        for slug, mi in markets.items():
            self._registry.add(slug, mi)

    def set_registry(self, registry: MarketRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> MarketRegistry:
        return self._registry

    def set_bankroll(self, bankroll: float) -> None:
        self._bankroll = bankroll

    def set_bankroll_source(self, fn) -> None:
        """Live bankroll callable — read each tick so risk caps and the
        breaker track the poller-reconciled balance instead of the value
        captured at startup."""
        self._bankroll_fn = fn

    def set_bankroll_view_source(self, fn, poll_interval_s: float = 30.0) -> None:
        """Optional second bankroll source returning a BankrollView
        (value, as_of). Used ONLY to log bankroll_age_s in the risk-cap
        decision filters dict when acting on a value older than 2x the
        bankroll poll interval -- the float bankroll plumbing above is
        untouched and remains the actual value risk caps compute against."""
        self._bankroll_view_fn = fn
        self._bankroll_poll_s = poll_interval_s

    @property
    def bankroll(self) -> float:
        if self._bankroll_fn is not None:
            try:
                return float(self._bankroll_fn())
            except Exception:
                return self._bankroll
        return self._bankroll

    def _bankroll_age_s(self) -> Optional[float]:
        """Seconds since the last bankroll reconcile, or None if no
        bankroll_view source is wired (paper mode / tests) or it raised."""
        if self._bankroll_view_fn is None:
            return None
        try:
            view = self._bankroll_view_fn()
            if view.as_of <= 0.0:
                return None  # never reconciled yet
            return time.monotonic() - view.as_of
        except Exception:
            return None

    def set_series_slugs(self, slugs: List[str]) -> None:
        self._series_slugs = slugs

    def _build_params(self, band_low: float, band_high: float) -> FilterParams:
        """FilterParams for this tick's mode + per-slug band. Live mode
        checks staleness and min-time-in-band; paper mode carves out BOTH
        (thinly-traded longshot-tail markets rarely accrue continuous
        in-band WS time or fresh WS deltas -- the gates would starve paper
        runs of entries). missing_dte='reject': live is fail-closed on a
        missing DTE (dte_out_of_range), unlike backtest's fail-open skip."""
        cfg = self._cfg
        is_paper = cfg.mode == "paper"
        return FilterParams(
            band_low=band_low, band_high=band_high,
            min_dte=cfg.strategy.min_dte, max_dte=cfg.strategy.max_dte,
            min_time_in_band_s=cfg.strategy.min_time_in_band_s,
            min_24h_volume=cfg.filters.min_24h_volume,
            min_total_volume=cfg.filters.min_total_volume,
            min_book_depth=cfg.filters.min_book_depth,
            check_staleness=not is_paper,
            check_time_in_band=not is_paper,
            missing_dte="reject",
        )

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return  # already running (idempotent for the 'start' command)
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

        # Load deployment state once per tick. Resting limit notional counts
        # against the caps too — unfilled limits have no position row yet.
        total_deployed, by_slug = positions_repo.deployed_total()
        if self._order_manager is not None:
            resting_total, resting_by_slug = self._order_manager.resting_exposure()
            total_deployed += resting_total
            for s, n in resting_by_slug.items():
                by_slug[s] = by_slug.get(s, 0.0) + n

        # --- Main slug loop (slugs.csv entries with per-slug config) ---
        for slug_row in cfg.enabled_slugs():
            slug = slug_row.slug
            market_info = self._registry.get(slug)
            if market_info is None:
                continue
            if not market_info.active or market_info.closed:
                continue
            token_id = market_info.token_id
            book = self._books.get(token_id)

            band_low, band_high = cfg.band_for_slug(slug)
            best_ask = float(book.best_ask) if (book is not None and book.best_ask is not None) else None
            params = self._build_params(band_low, band_high)

            if best_ask is None:
                log_rejected(slug, token_id, "no_book", {"band_low": band_low})
                continue

            dte_val = await self._get_dte(slug)
            pregate = evaluate_pregate(best_ask, dte_val, params)
            if not pregate.passed:
                log_rejected(slug, token_id, pregate.reason, pregate.detail)
                continue
            ask_f = best_ask

            notional = cfg.size_for_slug(slug)
            delta, _ = await self._evaluate_and_enter(
                slug=slug, token_id=token_id, book=book,
                band_low=band_low, band_high=band_high,
                notional=notional, market_info=market_info,
                total_deployed=total_deployed, by_slug=by_slug,
                ask_f=ask_f, dte_val=dte_val, params=params, log_decisions=True,
            )
            total_deployed += delta

        # --- Series-expanded slugs (no per-slug overrides — use globals) ---
        # NOTE: pregate rejects (band/DTE) here are NOT logged -- unlike the
        # main loop above. Thousands of series markets are evaluated every
        # tick; logging every reject would flood the decisions table (the
        # retention pruner exists because of decision volume). This
        # asymmetry is intentional and predates Phase 4 -- preserved as-is.
        for slug in self._series_slugs:
            mi = self._registry.get(slug)
            if mi is None or mi.closed or not mi.active:
                continue
            token_id = mi.token_id
            book = self._books.get(token_id)
            best_ask = float(book.best_ask) if (book is not None and book.best_ask is not None) else None
            if best_ask is None:
                continue

            band_low, band_high = cfg.strategy.band_low, cfg.strategy.band_high
            params = self._build_params(band_low, band_high)
            dte_val = self._dte_from_end_date(mi.end_date_iso)
            pregate = evaluate_pregate(best_ask, dte_val, params)
            if not pregate.passed:
                continue
            ask_f = best_ask

            notional = cfg.strategy.order_notional_usd
            delta, _ = await self._evaluate_and_enter(
                slug=slug, token_id=token_id, book=book,
                band_low=band_low, band_high=band_high,
                notional=notional, market_info=mi,
                total_deployed=total_deployed, by_slug=by_slug,
                ask_f=ask_f, dte_val=dte_val, params=params, log_decisions=True,
            )
            total_deployed += delta

        # Prune closed slugs from series list every 60 ticks
        self._tick_count += 1
        if self._tick_count % 60 == 0:
            self._series_slugs = [
                s for s in self._series_slugs
                if not (self._registry.get(s) and self._registry.get(s).closed)
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
        dte_val: Optional[float],
        params: FilterParams,
        log_decisions: bool = True,
    ) -> Tuple[float, float]:
        """Shared filter evaluation (filters 3-11) for main + series loops.

        Filters 1-2 (ask in band, DTE) were already checked by the caller
        via evaluate_pregate; dte_val is passed through unchanged so
        evaluate_entry's internal re-check of filters 1-2 is a no-op
        re-confirmation, not a second Gamma call. Filters 3-8 run here via
        evaluate_entry (shared pure core, strategy/filters.py) once the
        EntrySnapshot's REST-backed fields (volumes) are fetched -- lazily,
        only after pregate passes, so a full snapshot is never built for
        every active market on every tick (that would be an API storm on
        every 300s volume-cache expiry).
        Returns (delta_deployed, delta_market) — (0.0, 0.0) on reject.
        """
        cfg = self._cfg

        # -- Model pricer (optional, BTC-only, fail-open): evaluated HERE,
        # right after pregate, not after the full 1-8 pass -- evaluate()
        # registers the strike and warms the expiry's ladder cache, so it
        # must run every tick for every in-band market. Evaluating only on
        # a full filter pass would register a strike at the exact moment
        # the contract enters (and filter 8 then blocks re-evaluation), so
        # nearly every entry would be tagged naive forever. Log-only mode
        # attaches model_p_yes / model_edge_no to the decision (entered or
        # rejected); the veto itself is enforced AFTER filters 1-8 pass so
        # a contract failing volume/staleness still logs its real reason.
        # A None verdict (disabled, non-BTC, no cached ladder yet, stale
        # BTC data, engine failure) never blocks entry. --
        model_fields: Dict[str, Any] = {}
        if self._model_pricer is not None:
            try:
                verdict = self._model_pricer.evaluate(slug, ask_f, dte_val)
            except Exception as e:
                logger.warning(f"Model pricer evaluate failed for {slug}: {e}")
                verdict = None
            if verdict is not None:
                model_fields = dict(verdict)

        # -- Filters 4 & 5 REST inputs: volumes (cheap fields already
        # passed pregate; fetch the expensive ones only now) --
        vols = await self._get_volumes(slug)

        snapshot = EntrySnapshot(
            best_ask=ask_f,
            dte=dte_val,
            seconds_in_band=book.time_in_band(),
            volume_24h=vols["volume_24h"],
            volume_total=vols["volume_total"],
            ask_depth_usd=float(book.ask_depth_usd),
            is_stale=self._staleness.is_stale(token_id),
            has_open_position=self._has_open_position(token_id),
        )
        result = evaluate_entry(snapshot, params)
        if not result.passed:
            if log_decisions:
                log_rejected(
                    slug, token_id, result.reason,
                    {**result.detail, **model_fields},
                )
            return (0.0, 0.0)

        # -- Model veto (after filters 1-8 so reject reasons stay real) --
        if self._model_pricer is not None and model_fields:
            if self._model_pricer.should_veto(model_fields):
                if log_decisions:
                    detail = dict(model_fields)
                    detail["min_edge"] = self._model_pricer.min_edge
                    log_rejected(slug, token_id, "model_edge_low", detail)
                return (0.0, 0.0)

        # -- Filters 9, 10, 11: risk --
        alpha = cfg.strategy.alpha
        if alpha != 0.0:
            fn = make_sizing_fn(alpha, band_low, band_high)
            notional = max(MIN_EFFECTIVE_NOTIONAL, notional * fn(ask_f))
        market_deployed = by_slug.get(slug, 0.0)
        bankroll = self.bankroll  # live source when wired (set_bankroll_source)
        allowed, risk_reason = self._risk.allow_entry(
            slug, notional, bankroll, total_deployed, market_deployed
        )
        if not allowed:
            if log_decisions:
                risk_filters: Dict[str, Any] = {"risk_reason": risk_reason}
                age_s = self._bankroll_age_s()
                if age_s is not None and age_s > 2 * self._bankroll_poll_s:
                    risk_filters["bankroll_age_s"] = round(age_s, 1)
                log_rejected(slug, token_id, "risk_cap", risk_filters)
            return (0.0, 0.0)

        if self._risk.check_breaker_against_bankroll(bankroll):
            if log_decisions:
                log_rejected(slug, token_id, "circuit_breaker", {})
            return (0.0, 0.0)

        # -- All filters passed: enter --
        await self._order_manager.enter(
            slug=slug,
            token_id=token_id,
            book=book,
            notional=notional,
            filters={"band_low": band_low, "band_high": band_high, **model_fields},
            market_info=market_info,
        )
        return (notional, notional)

    def _has_open_position(self, token_id: str) -> bool:
        return positions_repo.has_open(token_id)

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
        mi = self._registry.get(slug)
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
