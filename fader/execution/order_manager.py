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
from typing import Any, Dict, List, Optional

from config.config_loader import AppConfig
from execution.idempotency import make_key, is_already_submitted
from execution.sizing import compute_shares_and_notional
from persistence.decision_log import log_entered, log_rejected

logger = logging.getLogger(__name__)


class RestingOrder:
    __slots__ = (
        "order_id", "idempotency_key", "slug", "token_id", "price",
        "size", "notional", "placed_at", "ttl_expires_at", "last_mid",
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


class OrderManager:
    def __init__(self, cfg: AppConfig, provider, risk=None) -> None:
        self._cfg = cfg
        self._provider = provider
        self._risk = risk  # RiskManager, for TOCTOU breaker check
        self._resting: Dict[str, RestingOrder] = {}  # token_id -> order
        self._lock = asyncio.Lock()

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
                log_rejected(slug, token_id, "circuit_breaker_toctou", filters)
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
                await self._place_limit(slug, token_id, float(mid), float(ask), notional, filters)

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
        if not result.get("success"):
            # Check for server-side duplicate
            error_msg = result.get("error", "")
            from execution.idempotency import is_duplicate_error
            if is_duplicate_error(error_msg):
                # Recover real order_id from open orders
                recovered = self._provider.find_order_by_params(
                    token_id, "BUY", size
                )
                if recovered and recovered.get("order_id"):
                    real_id = recovered["order_id"]
                    self._save_order(
                        real_id, idem_key, slug, token_id, price, size, "MARKET",
                        status="PENDING",
                    )
                    log_entered(slug, token_id, filters, real_id, idem_key)
                    logger.info(
                        f"Market order (duplicate): {slug} recovered as {real_id}"
                    )
                    return
                # Fall through — store as UNKNOWN
                order_id = idem_key
                self._save_order(
                    order_id, idem_key, slug, token_id, price, size, "MARKET",
                    status="UNKNOWN",
                )
                log_entered(slug, token_id, filters, order_id, idem_key)
                logger.info(
                    f"Market order (duplicate, unrecovered): {slug} -> UNKNOWN"
                )
                return

            logger.error(f"Market order failed for {slug}: {result.get('error')}")
            from infra import telegram
            asyncio.create_task(
                telegram.alert_exchange_rejection(slug, result.get("error", ""))
            )
            return

        order_id = result.get("order_id")
        is_simulated = result.get("simulated", False)

        # Gate position insert: only when we have a real order_id
        if order_id is not None:
            self._save_order(
                order_id, idem_key, slug, token_id, price, size, "MARKET",
                status="PENDING",
            )
            log_entered(slug, token_id, filters, order_id, idem_key)
            # Immediately insert position row so _has_open_position returns
            # True on the next tick (closes the 60s reconciler gap).
            self._insert_position(
                slug, token_id, price, size, actual_notional,
                order_id, idem_key, market_info,
            )
        else:
            # No order_id and not a duplicate → honest API gap
            order_id = idem_key
            self._save_order(
                order_id, idem_key, slug, token_id, price, size, "MARKET",
                status="UNKNOWN",
            )
            log_entered(slug, token_id, filters, order_id, idem_key)
            # Do NOT insert position row — we aren't sure the order hit
            # the exchange.

        logger.info(
            f"Market order placed: {slug} {size:.4f}@{price:.4f} -> {order_id}"
        )
        from infra import telegram
        asyncio.create_task(telegram.alert_position_opened(slug, price, actual_notional))

    async def _place_limit(
        self,
        slug: str,
        token_id: str,
        mid: float,
        ask: float,
        notional: float,
        filters: Dict[str, Any],
    ) -> None:
        # Paper mode: no matching engine exists to fill limit orders.
        # Simulate immediate fill so positions track and re-entry is blocked.
        if self._provider._mode == "paper":
            price = round(mid, 4)
            if price <= 0 or price >= 1:
                price = round(ask, 4)
            size, actual_notional = compute_shares_and_notional(notional, price)
            idem_key = make_key(slug, token_id, "BUY", price, size, "paper_fill")
            if is_already_submitted(idem_key):
                return
            order_id = idem_key  # no real order ID in paper mode
            self._save_order(order_id, idem_key, slug, token_id, price, size, "LIMIT",
                             status="FILLED")
            log_entered(slug, token_id, filters, order_id, idem_key)
            self._insert_position(
                slug, token_id, price, size, actual_notional,
                order_id, idem_key, None,
            )
            logger.info(
                f"[PAPER] Simulated fill: {slug} {size:.4f}@{price:.4f} -> {order_id}"
            )
            return

        price = round(mid, 4)
        if price <= 0 or price >= 1:
            price = round(ask, 4)
        size, _ = compute_shares_and_notional(notional, price)
        idem_key = make_key(slug, token_id, "BUY", price, size, "limit")
        if is_already_submitted(idem_key):
            return

        result = await self._provider.async_place_order(
            token_id, "BUY", price, size, "LIMIT"
        )
        if not result.get("success"):
            logger.error(f"Limit order failed for {slug}: {result.get('error')}")
            from infra import telegram
            asyncio.create_task(
                telegram.alert_exchange_rejection(slug, result.get("error", ""))
            )
            return

        order_id = result.get("order_id") or idem_key
        self._save_order(order_id, idem_key, slug, token_id, price, size, "LIMIT")
        log_entered(slug, token_id, filters, order_id, idem_key)
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

    # ------------------------------------------------------------------
    # Requote loop (called by main loop periodically)
    # ------------------------------------------------------------------

    async def requote_check(self, book_store) -> None:
        """Iterate resting orders; cancel-replace if mid moved >= requote threshold."""
        if self._provider._mode == "paper":
            return  # paper mode fills instantly, no resting orders
        cfg = self._cfg
        now = time.monotonic()
        to_cancel: List[str] = []

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
        await self._cancel_resting(order.token_id, "requote")
        mid = float(book.mid) if book.mid else float(book.best_ask)
        notional = self._cfg.size_for_slug(order.slug)
        filters: Dict = {"requote": True}
        await self._place_limit(order.slug, order.token_id, mid, float(book.best_ask), notional, filters)

    async def _cancel_resting(self, token_id: str, reason: str) -> None:
        order = self._resting.pop(token_id, None)
        if order is None:
            return
        result = await self._provider.async_cancel_order(order.order_id)
        self._update_order_status(order.order_id, "CANCELLED", reason)
        logger.info(f"Cancelled {order.order_id} ({reason})")

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
    # Close-all (emergency)
    # ------------------------------------------------------------------

    async def close_all(self) -> None:
        """Cancel all resting orders, then market-sell every open position.

        Pre-flight: checks USDC.e allowance is sufficient for SELL transfers.
        Aborts with Telegram alert if allowance is too low.
        """
        await self._provider.async_cancel_all()
        self._resting.clear()

        from infra.db import get_connection
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT token_id, size, notional FROM positions WHERE status='OPEN'"
            ).fetchall()
        finally:
            conn.close()

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
            asyncio.create_task(telegram.send(f"<b>{msg}</b>"))
            return

        for row in rows:
            await self._provider.async_market_sell(row["token_id"], row["size"])

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
        # Application-level enforcement for existing DBs (CHECK constraint
        # only applied on CREATE TABLE IF NOT EXISTS for fresh deploys).
        VALID = frozenset({"PENDING", "FILLED", "CANCELLED", "FAILED", "UNKNOWN"})
        if status not in VALID:
            raise ValueError(f"Invalid order status: {status!r}  (valid: {sorted(VALID)})")
        from infra.db import get_connection
        now = datetime.now(timezone.utc).isoformat()
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO orders
                  (order_id, idempotency_key, slug, token_id, side, type, price, size,
                   status, created_at)
                VALUES (?, ?, ?, ?, 'BUY', ?, ?, ?, ?, ?)
                """,
                (order_id, idem_key, slug, token_id, order_type, price, size, status, now),
            )
            conn.commit()
        except Exception as e:
            logger.error(f"_save_order: {e}")
        finally:
            conn.close()

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
        the same position later.
        """
        import os
        user = os.getenv("POLYMARKET_USER_ADDRESS", "")
        cid = market_info.condition_id if market_info else ""
        oidx = market_info.outcome_index if market_info else 0
        position_id = f"{user}:{cid}:{oidx}"

        from infra.db import get_connection
        now = datetime.now(timezone.utc).isoformat()
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO positions
                  (position_id, slug, condition_id, token_id, outcome,
                   entry_price, size, notional, status, opened_at, source,
                   entry_order_id, entry_decision_id)
                VALUES (?, ?, ?, ?, 'No', ?, ?, ?, 'OPEN', ?, 'ENGINE_FILL', ?, ?)
                """,
                (position_id, slug, cid, token_id, price, size, notional, now,
                 order_id, decision_id),
            )
            conn.commit()
        except Exception as e:
            logger.error(f"_insert_position: {e}")
        finally:
            conn.close()

    def _update_order_status(
        self,
        order_id: str,
        status: str,
        cancel_reason: Optional[str],
    ) -> None:
        from infra.db import get_connection
        conn = get_connection()
        try:
            conn.execute(
                "UPDATE orders SET status=?, cancel_reason=? WHERE order_id=?",
                (status, cancel_reason, order_id),
            )
            conn.commit()
        finally:
            conn.close()
