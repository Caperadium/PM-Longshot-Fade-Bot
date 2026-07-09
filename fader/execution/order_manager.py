"""execution/order_manager.py

Order lifecycle manager:
  - spread <= 1c: market order at fixed notional (FOK)
  - spread >  1c: limit at mid; requote on >=0.5c mid move (cancel-replace)
  - TTL 5 min; re-evaluate or cancel if price exits band
  - Tracks all resting orders in-memory + DB
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from config.config_loader import AppConfig
from execution.idempotency import make_key, is_already_submitted
from execution.sizing import compute_shares_and_notional
from marketdata.rest_market import _derive_series_filter
from persistence.decision_log import log_entered, log_rejected
from persistence.repos import db, orders_repo, positions_repo

logger = logging.getLogger(__name__)


def _log_entered(slug, token_id, filters, order_id, idempotency_key) -> None:
    if not log_entered(slug, token_id, filters, order_id, idempotency_key):
        logger.warning(
            f"log_entered failed to persist decision row for {slug} ({order_id})"
        )


def _log_rejected(slug, token_id, reason, filters) -> None:
    if not log_rejected(slug, token_id, reason, filters):
        logger.warning(
            f"log_rejected failed to persist decision row for {slug} ({reason})"
        )


class RestingOrder:
    __slots__ = (
        "order_id", "idempotency_key", "slug", "token_id", "price",
        "size", "notional", "placed_at", "ttl_expires_at", "last_mid",
        "unverified",
    )

    def __init__(
        self,
        order_id: str,
        idempotency_key: str,
        slug: str,
        token_id: str,
        price: float,
        size: float,
        notional: float,
        placed_at: float,
        ttl_s: int,
        mid: float,
        unverified: bool = False,
    ) -> None:
        self.order_id = order_id
        self.idempotency_key = idempotency_key
        self.slug = slug
        self.token_id = token_id
        self.price = price
        self.size = size
        self.notional = notional
        self.placed_at = placed_at
        self.ttl_expires_at = placed_at + ttl_s
        self.last_mid = mid
        # Set when this row was rehydrated from DB but could not be
        # confirmed against the live API (fetch_open_orders() returned
        # None -- API unavailable, not "no orders"). Re-verified on each
        # requote tick instead of being dropped or trusted blindly.
        self.unverified = unverified


class OrderManager:
    def __init__(self, cfg: AppConfig, provider, risk=None) -> None:
        self._cfg = cfg
        self._provider = provider
        self._risk = risk  # RiskManager, for TOCTOU breaker check
        self._resting: Dict[str, RestingOrder] = {}  # token_id -> order
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Startup rehydrate (FIX 3) — must run after construction, before any
    # background task starts, so _resting is fully populated before the
    # requote loop's first pass.
    # ------------------------------------------------------------------

    async def rehydrate_resting(self) -> int:
        """Repopulate in-memory _resting from DB PENDING LIMIT orders that
        are still live on the exchange.

        _resting is memory-only. After a restart, a live resting LIMIT
        order sits in DB as PENDING and is seen live by the reconciler
        (stays PENDING) but is absent from _resting -> no TTL/band-exit/
        requote management -> orphan until filled or manually cancelled.
        This closes that gap. Returns the count of orders rehydrated.

        Paper mode: DB read only, no API verify -- PaperProvider.place_order
        always returns status=FILLED (never PENDING), so in practice there
        are no PENDING LIMIT rows to find; this path exists so a stray row
        (e.g. left over from a live/paper mode switch) is still surfaced
        rather than silently ignored, matching main.py's existing
        `if cfg.mode != "paper"` gate around the startup call site (this
        method itself no longer hard-codes a paper no-op).

        DB read failure (the SELECT itself raising, e.g. a locked or
        corrupt file) is fatal at startup: fire a telegram alert and
        re-raise so main.py's startup sequence aborts rather than running
        with an empty, unmanaged resting-order set. The supervisor
        (systemd Restart=always / run_engine_supervised.py) bounds the
        resulting crash-loop.
        """
        try:
            rows = orders_repo.pending_limit_orders()
        except Exception as e:
            logger.error(f"rehydrate_resting: DB read failed — aborting startup: {e}")
            from infra import telegram
            await telegram.alert_error("rehydrate_resting (DB read)", str(e))
            raise

        if self._provider.is_paper:
            # No API to verify against; paper orders never rest anyway.
            return 0

        live = await self._provider.async_fetch_open_orders()
        # fetch_open_orders() returns None on API error (Phase 2) --
        # distinct from an empty list ("confirmed no live orders"). On
        # None we can't confirm any DB row against the exchange, so every
        # PENDING LIMIT row is rehydrated but flagged unverified=True;
        # requote_check re-verifies each tick instead of dropping them
        # (dropping would silently stop managing a live order).
        api_unavailable = live is None
        live_by_id = {} if api_unavailable else {
            o["order_id"]: o for o in live if o.get("order_id")
        }
        if api_unavailable:
            logger.warning(
                "rehydrate_resting: fetch_open_orders failed — "
                "keeping DB rows unverified, will re-verify on requote tick"
            )

        count = 0
        for row in rows:
            order_id = row["order_id"]
            if order_id.startswith("SIM-") or order_id.startswith("FAKE-"):
                continue  # paper-mode simulated; never live
            token_id = row["token_id"]
            if not api_unavailable and order_id not in live_by_id:
                continue  # not live — left to the reconciler's mark_vanished path
            if token_id in self._resting:
                # Rows are created_at DESC: newest PENDING limit per token wins.
                continue
            price = float(row["price"])
            size = float(row["size"])
            self._resting[token_id] = RestingOrder(
                order_id=order_id,
                idempotency_key=row["idempotency_key"],
                slug=row["slug"],
                token_id=token_id,
                price=price,
                size=size,
                notional=price * size,
                placed_at=time.monotonic(),
                ttl_s=self._cfg.orders.limit_ttl_s,
                mid=price,  # book may not be loaded yet; requote loop recomputes
                unverified=api_unavailable,
            )
            count += 1

        if count:
            logger.info(f"Rehydrated {count} resting limit order(s) from DB+live API")
        return count

    # ------------------------------------------------------------------
    # Entry (called by strategy loop when all filters pass)
    # ------------------------------------------------------------------

    async def enter(
        self,
        slug: str,
        token_id: str,
        book,  # OrderBook
        notional: float,
        filters: Dict[str, Any],
        market_info=None,  # MarketInfo, for position insert
    ) -> None:
        async with self._lock:
            # Don't double-enter
            if token_id in self._resting:
                return

            # TOCTOU breaker check (may have tripped between strategy
            # filter pass and lock acquisition)
            if self._risk and self._risk.breaker_tripped:
                _log_rejected(slug, token_id, "circuit_breaker_toctou", filters)
                return

            ask = book.best_ask
            bid = book.best_bid
            if ask is None:
                return

            spread_cents = float(book.spread_cents or 99)
            threshold = self._cfg.orders.spread_market_threshold_c

            if spread_cents <= threshold:
                await self._place_market(slug, token_id, ask, notional, filters, market_info)
            else:
                mid = book.mid
                if mid is None:
                    mid = ask
                await self._place_limit(
                    slug, token_id, float(mid), float(ask), notional, filters,
                    market_info=market_info,
                )

    async def _place_market(
        self,
        slug: str,
        token_id: str,
        ask: Decimal,
        notional: float,
        filters: Dict[str, Any],
        market_info=None,  # MarketInfo, for position insert
    ) -> None:
        price = float(ask)
        size, actual_notional = compute_shares_and_notional(notional, price)
        idem_key = make_key(slug, token_id, "BUY", price, size, "market")
        if is_already_submitted(idem_key):
            return

        result = await self._provider.async_place_order(
            token_id, "BUY", price, size, "MARKET"
        )

        if result.status == "DUPLICATE":
            self._handle_duplicate(
                slug, token_id, "BUY", price, size, "MARKET", idem_key, filters,
            )
            return

        if result.status == "REJECTED":
            logger.error(f"Market order failed for {slug}: {result.error}")
            from infra import telegram
            telegram.fire(telegram.alert_exchange_rejection(slug, result.error or ""))
            return

        order_id = result.order_id

        # Gate position insert: only when we have a real order_id
        if order_id is not None:
            # Paper mode has no exchange to fill against and the reconciler
            # skips SIM- orders, so a PENDING row would stay PENDING forever.
            # A simulated market order is an instant fill by definition
            # (PaperProvider.place_order always returns status=FILLED).
            # Order status update + position insert are atomic (one
            # transaction) so a mid-write crash can't leave a FILLED/PENDING
            # order with no matching position row.
            self._save_order_and_insert_position(
                order_id, idem_key, slug, token_id, price, size, "MARKET",
                result.status,  # "FILLED" or "PENDING"
                actual_notional, market_info,
            )
            _log_entered(slug, token_id, filters, order_id, idem_key)
            # Position row is now visible to _has_open_position on the next
            # tick (closes the 60s reconciler gap).
        else:
            # No order_id and not a duplicate → honest API gap
            order_id = idem_key
            self._save_order(
                order_id, idem_key, slug, token_id, price, size, "MARKET",
                status="UNKNOWN",
            )
            _log_entered(slug, token_id, filters, order_id, idem_key)
            # Do NOT insert position row — we aren't sure the order hit
            # the exchange.

        logger.info(
            f"Market order placed: {slug} {size:.4f}@{price:.4f} -> {order_id}"
        )
        from infra import telegram
        telegram.fire(telegram.alert_position_opened(slug, price, actual_notional))

    async def _place_limit(
        self,
        slug: str,
        token_id: str,
        mid: float,
        ask: float,
        notional: float,
        filters: Dict[str, Any],
        market_info=None,  # MarketInfo, for position insert (paper fill only)
    ) -> None:
        price = round(mid, 4)
        if price <= 0 or price >= 1:
            price = round(ask, 4)
        size, actual_notional = compute_shares_and_notional(notional, price)
        idem_key = make_key(slug, token_id, "BUY", price, size, "limit")
        if is_already_submitted(idem_key):
            return

        result = await self._provider.async_place_order(
            token_id, "BUY", price, size, "LIMIT"
        )

        if result.status == "DUPLICATE":
            self._handle_duplicate(
                slug, token_id, "BUY", price, size, "LIMIT", idem_key, filters,
            )
            return

        if result.status == "REJECTED":
            logger.error(f"Limit order failed for {slug}: {result.error}")
            from infra import telegram
            telegram.fire(telegram.alert_exchange_rejection(slug, result.error or ""))
            return

        order_id = result.order_id or idem_key

        if result.status == "FILLED":
            # Paper mode: instant fill, no resting order to track. Live
            # limit orders never return FILLED from place_order today
            # (the API confirms acceptance, not fill), so this path is
            # paper-only in practice, but the dispatch itself no longer
            # branches on provider.is_paper.
            self._save_order_and_insert_position(
                order_id, idem_key, slug, token_id, price, size, "LIMIT", "FILLED",
                actual_notional, market_info,
            )
            _log_entered(slug, token_id, filters, order_id, idem_key)
            logger.info(
                f"Simulated fill: {slug} {size:.4f}@{price:.4f} -> {order_id}"
            )
            return

        # PENDING (order_id may be missing -> falls back to idem_key above):
        # resting limit order to track.
        self._save_order(order_id, idem_key, slug, token_id, price, size, "LIMIT")
        _log_entered(slug, token_id, filters, order_id, idem_key)
        self._resting[token_id] = RestingOrder(
            order_id=order_id,
            idempotency_key=idem_key,
            slug=slug,
            token_id=token_id,
            price=price,
            size=size,
            notional=notional,
            placed_at=time.monotonic(),
            ttl_s=self._cfg.orders.limit_ttl_s,
            mid=mid,
        )
        logger.info(f"Limit order placed: {slug} {size:.4f}@{price:.4f} -> {order_id}")

    def _handle_duplicate(
        self,
        slug: str,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str,
        idem_key: str,
        filters: Dict[str, Any],
    ) -> None:
        """DUPLICATE dispatch: recover the real order_id via
        find_order_by_params; on failure store as UNKNOWN (no position
        insert). Shared by _place_market and _place_limit so both the
        success=false-body duplicate shape and the exception-string
        duplicate shape (see provider.OrderResult) get the same recovery
        attempt."""
        recovered = self._provider.find_order_by_params(token_id, side, size)
        if recovered and recovered.get("order_id"):
            real_id = recovered["order_id"]
            self._save_order(
                real_id, idem_key, slug, token_id, price, size, order_type,
                status="PENDING",
            )
            _log_entered(slug, token_id, filters, real_id, idem_key)
            logger.info(f"Order (duplicate): {slug} recovered as {real_id}")
            return
        # Fall through — store as UNKNOWN
        order_id = idem_key
        self._save_order(
            order_id, idem_key, slug, token_id, price, size, order_type,
            status="UNKNOWN",
        )
        _log_entered(slug, token_id, filters, order_id, idem_key)
        logger.info(f"Order (duplicate, unrecovered): {slug} -> UNKNOWN")

    # ------------------------------------------------------------------
    # Requote loop (called by main loop periodically)
    # ------------------------------------------------------------------

    async def _reverify_unverified(self) -> None:
        """Re-check any _resting rows flagged unverified against the live
        API. Runs once per requote_check tick, before the TTL/band/requote
        pass, so a confirmed-gone order doesn't get managed as if it were
        still live and a confirmed-live order stops being retried.

        On a fresh API failure (None again), rows simply stay unverified
        and are retried next tick -- no change.
        """
        async with self._lock:
            pending_ids = [
                tid for tid, o in self._resting.items() if o.unverified
            ]
        if not pending_ids:
            return

        live = await self._provider.async_fetch_open_orders()
        if live is None:
            return  # still unavailable — retry again next tick
        live_ids = {o["order_id"] for o in live if o.get("order_id")}

        async with self._lock:
            for token_id in pending_ids:
                order = self._resting.get(token_id)
                if order is None:
                    continue
                if order.order_id in live_ids:
                    order.unverified = False
                else:
                    # Confirmed gone — same handling as the reconciler's
                    # mark_vanished path: pop so re-entry isn't blocked.
                    self._resting.pop(token_id, None)
                    self._update_order_status(
                        order.order_id, "UNKNOWN", "vanished_from_api"
                    )
                    logger.info(
                        f"Unverified resting order {order.order_id[:16]} "
                        f"confirmed gone on re-verify — marked UNKNOWN"
                    )

    async def requote_check(self, book_store) -> None:
        """Iterate resting orders; cancel-replace if mid moved >= requote threshold."""
        if self._provider.is_paper:
            return  # paper mode fills instantly, no resting orders
        cfg = self._cfg
        now = time.monotonic()
        to_cancel: List[str] = []

        await self._reverify_unverified()

        async with self._lock:
            for token_id, order in list(self._resting.items()):
                book = book_store.get(token_id)
                if book is None:
                    continue

                # TTL expired
                if now > order.ttl_expires_at:
                    to_cancel.append(token_id)
                    logger.info(f"Limit TTL expired for {order.slug}; cancelling")
                    continue

                # Price exited band
                ask = book.best_ask
                if ask is None:
                    continue
                band_low, band_high = cfg.band_for_slug(order.slug)
                if not (band_low <= float(ask) <= band_high):
                    to_cancel.append(token_id)
                    logger.info(f"Price exited band for {order.slug}; cancelling limit")
                    continue

                # Mid moved enough to requote?
                mid = book.mid
                if mid is None:
                    continue
                mid_move_c = abs(float(mid) - order.last_mid) * 100
                if mid_move_c >= cfg.orders.requote_move_c:
                    await self._requote(order, book)

        # Cancel expired/out-of-band
        for token_id in to_cancel:
            await self._cancel_resting(token_id, "ttl_or_band_exit")

    async def _requote(self, order: RestingOrder, book) -> None:
        if not await self._cancel_resting(order.token_id, "requote"):
            # Cancel failed — old order may still be live. Placing a
            # replacement would risk two live orders (double exposure).
            return
        mid = float(book.mid) if book.mid else float(book.best_ask)
        notional = self._cfg.size_for_slug(order.slug)
        filters: Dict = {"requote": True}
        await self._place_limit(order.slug, order.token_id, mid, float(book.best_ask), notional, filters)

    async def _cancel_resting(self, token_id: str, reason: str) -> bool:
        """Cancel a resting order. Returns True when the cancel was accepted.

        On cancel failure the order is re-tracked in _resting so TTL/requote
        keeps managing it and the next pass retries the cancel; if the order
        is actually gone on the exchange the reconciler's mark_vanished pops
        it within one resolution cycle.
        """
        order = self._resting.pop(token_id, None)
        if order is None:
            return True
        result = await self._provider.async_cancel_order(order.order_id)
        if not result.get("success"):
            self._resting[token_id] = order
            self._update_order_status(
                order.order_id, "UNKNOWN", f"cancel_failed:{reason}"
            )
            logger.warning(
                f"Cancel FAILED for {order.order_id} ({reason}): "
                f"{result.get('error')} — order re-tracked, will retry"
            )
            return False
        self._update_order_status(order.order_id, "CANCELLED", reason)
        logger.info(f"Cancelled {order.order_id} ({reason})")
        return True

    def resting_exposure(self) -> Tuple[float, Dict[str, float]]:
        """(total_notional, {slug: notional}) of in-flight resting limit
        orders. Counted against the deployed caps so unfilled limits can't
        over-commit the bankroll (they don't create position rows until
        they fill)."""
        total = 0.0
        by_slug: Dict[str, float] = {}
        for o in list(self._resting.values()):
            total += o.notional
            by_slug[o.slug] = by_slug.get(o.slug, 0.0) + o.notional
        return total, by_slug

    async def cancel_resting_for_disabled_slugs(self) -> int:
        """Cancel resting limit orders whose slug is no longer enabled.

        Called after a config hot-reload so disabling a slug in slugs.csv
        (or the dashboard multiselect) pulls its live limit orders instead
        of leaving them to die on TTL. Series children match by
        series_filter substring — same rule as discovery and backtest
        expansion. Returns the number of cancels accepted.
        """
        enabled_exact: set = set()
        enabled_filters: List[str] = []
        for row in self._cfg.enabled_slugs():
            if row.market_kind in ("series", "btc_daily"):
                enabled_filters.append(
                    (row.series_filter or _derive_series_filter(row.slug)).lower()
                )
            else:
                enabled_exact.add(row.slug)

        def _enabled(slug: str) -> bool:
            # Discovery lowercases slugs for comparison but stores original
            # case (P4) -- lowercase both sides so mixed-case slugs still
            # match a lowercase filter.
            slug_lower = slug.lower()
            return slug in enabled_exact or any(
                f in slug_lower for f in enabled_filters
            )

        async with self._lock:
            to_cancel = [
                tid for tid, o in self._resting.items() if not _enabled(o.slug)
            ]
        n = 0
        for tid in to_cancel:
            try:
                if await self._cancel_resting(tid, "slug_disabled"):
                    n += 1
            except Exception as e:
                logger.warning(f"cancel_resting_for_disabled_slugs {tid[:16]}: {e}")
        if n:
            logger.info(f"Cancelled {n} resting order(s) on disabled slug(s)")
        return n

    def mark_filled(self, order_id: str, token_id: str) -> None:
        """Called by reconciler when an order becomes filled."""
        self._resting.pop(token_id, None)
        self._update_order_status(order_id, "FILLED", None)

    def mark_vanished(self, token_id: str) -> None:
        """Called by reconciler when an order vanished from live API.
        Pops from _resting so re-entry isn't blocked during the TTL window."""
        order = self._resting.pop(token_id, None)
        if order:
            self._update_order_status(order.order_id, "UNKNOWN", "vanished_from_api")

    # ------------------------------------------------------------------
    # Graceful shutdown (FIX 5)
    # ------------------------------------------------------------------

    async def cancel_all_resting(self, reason: str = "shutdown") -> None:
        """Cancel every resting limit order before the engine exits.

        Snapshots keys under the lock, then cancels sequentially —
        shutdown volume is tiny (one resting order per token), so a
        racing gather-and-pop isn't worth the complexity.

        Cancel-state semantics: _cancel_resting marks DB status CANCELLED
        once the provider cancel call returns (success or not — matches
        existing _cancel_resting behavior). Orders not yet processed if
        the caller times out stay PENDING in DB and live on-exchange;
        rehydrate_resting recovers them on next boot.
        """
        if self._provider.is_paper:
            self._resting.clear()
            return
        async with self._lock:
            token_ids = list(self._resting.keys())
        for tid in token_ids:
            try:
                await self._cancel_resting(tid, reason)
            except Exception as e:
                logger.warning(f"cancel_all_resting {tid[:16]}: {e}")

    # ------------------------------------------------------------------
    # Close-all (emergency)
    # ------------------------------------------------------------------

    async def close_all(self) -> None:
        """Cancel all resting orders, then close every open position.

        Live mode: market-sells each open position. Pre-flight checks
        USDC.e allowance is sufficient for SELL transfers; aborts with
        Telegram alert if allowance is too low.

        Paper mode: there is no venue to sell into, and with paper
        positions now persisting across restarts (D3) nothing else would
        ever close them — mark every OPEN row CLOSED directly at
        realized_pnl=0.0 (exit-at-entry; any other mark would be invented)
        instead of running the allowance/sell path below (M1).
        """
        await self._provider.async_cancel_all()
        self._resting.clear()

        if self._provider.is_paper:
            n = positions_repo.bulk_close_paper()
            logger.info(f"[PAPER] close_all: closed {n} open position(s)")
            return

        rows = positions_repo.open_for_close()

        if not rows:
            logger.info("close_all: no open positions")
            return

        # Pre-flight: check USDC.e allowance
        total_notional = sum(float(r["notional"]) for r in rows)
        try:
            allowance = await self._provider.async_fetch_usdc_allowance()
        except Exception as e:
            logger.warning(f"close_all allowance check failed: {e}")
            allowance = None
        if allowance is not None and allowance < total_notional:
            msg = (
                f"close_all ABORTED: USDC.e allowance {allowance:.2f} "
                f"< total notional {total_notional:.2f}.  Approve USDC.e on "
                f"Polymarket CTF Exchange before closing positions."
            )
            logger.error(msg)
            from infra import telegram
            telegram.fire(telegram.send(f"<b>{msg}</b>"))
            return

        failures = 0
        for row in rows:
            result = await self._provider.async_market_sell(row["token_id"], row["size"])
            if not result.get("success"):
                failures += 1
                logger.error(
                    f"close_all sell failed for {row['token_id'][:16]}: "
                    f"{result.get('error')}"
                )
        if failures:
            from infra import telegram
            telegram.fire(telegram.send(
                f"<b>close_all: {failures}/{len(rows)} sells FAILED</b> — "
                f"positions remain open, check dashboard"
            ))

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _save_order(
        self,
        order_id: str,
        idem_key: str,
        slug: str,
        token_id: str,
        price: float,
        size: float,
        order_type: str,
        status: str = "PENDING",
    ) -> None:
        try:
            orders_repo.insert(order_id, idem_key, slug, token_id, price, size, order_type, status=status)
        except ValueError:
            # Application-level status validation (OrdersRepo.insert) --
            # this is a programming error, not a DB fault; let it propagate
            # as before (original _save_order raised before its try/except).
            raise
        except Exception as e:
            logger.error(f"_save_order: {e}")

    def _insert_position(
        self,
        slug: str,
        token_id: str,
        price: float,
        size: float,
        notional: float,
        order_id: str,
        decision_id: str,
        market_info,
    ) -> None:
        """Immediately insert an OPEN position row after a market fill.

        Uses the canonical position_id = {user}:{condition_id}:{outcome_index}
        so the reconciler's INSERT OR IGNORE will skip this row if it sees
        the same position later. Without market_info (e.g. paper limit
        fills before that plumbing existed) condition_id/outcome_index are
        unknown for every market, which would collide all paper limit
        fills onto the single id f"{user}::0" (P3/D6) — token_id is unique
        per market+outcome, so fall back to that instead.
        """
        row = self._position_row(slug, token_id, price, size, notional, order_id, decision_id, market_info)
        try:
            positions_repo.insert_open(row)
        except Exception as e:
            logger.error(f"_insert_position: {e}")

    @staticmethod
    def _position_row(
        slug: str,
        token_id: str,
        price: float,
        size: float,
        notional: float,
        order_id: str,
        decision_id: str,
        market_info,
    ) -> Dict[str, Any]:
        import os
        user = os.getenv("POLYMARKET_USER_ADDRESS", "")
        if market_info:
            position_id = f"{user}:{market_info.condition_id}:{market_info.outcome_index}"
        else:
            position_id = f"{user}:{token_id}:0"
        cid = market_info.condition_id if market_info else ""
        now = datetime.now(timezone.utc).isoformat()
        return {
            "position_id": position_id,
            "slug": slug,
            "condition_id": cid,
            "token_id": token_id,
            "entry_price": price,
            "size": size,
            "notional": notional,
            "opened_at": now,
            "entry_order_id": order_id,
            "entry_decision_id": decision_id,
        }

    def _save_order_and_insert_position(
        self,
        order_id: str,
        idem_key: str,
        slug: str,
        token_id: str,
        price: float,
        size: float,
        order_type: str,
        status: str,
        notional: float,
        market_info,
    ) -> None:
        """Atomically record a fill: order status update + position insert
        in one transaction (Phase 1 of the architecture refactor -- was
        two separate connections/commits, leaving a window where an order
        could be marked FILLED/PENDING with no matching position row if
        the process died between the two writes)."""
        row = self._position_row(slug, token_id, price, size, notional, order_id, idem_key, market_info)
        try:
            with db.transaction() as conn:
                orders_repo.insert(
                    order_id, idem_key, slug, token_id, price, size, order_type,
                    status=status, conn=conn,
                )
                positions_repo.insert_open(row, conn=conn)
        except Exception as e:
            logger.error(f"_save_order_and_insert_position: {e}")

    def _update_order_status(
        self,
        order_id: str,
        status: str,
        cancel_reason: Optional[str],
    ) -> None:
        orders_repo.update_status(order_id, status, cancel_reason)
