"""engine/reconciler.py

On startup and on interval: reconcile orders/positions/bankroll/resolution
against Polymarket API as ground truth.

Resolution PnL: held NO pays payout in {0,1}.
  realized_pnl = (payout - entry_price) * size
Prefer API's booked PnL when present; formula is fallback/cross-check.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from engine.risk import RiskManager
from marketdata.rest_market import (
    CALL_DELAY_S,
    _resolution_from_outcome_prices,
    fetch_market_metadata,
)
from persistence.repos import db, engine_state_repo, orders_repo, positions_repo

# Cap on open paper positions polled per resolution cycle (D4). ~20 open
# positions today; 100 leaves headroom without risking a slow Gamma sweep
# blocking the 60s resolution poller for multiple cycles.
MAX_PAPER_POLL_PER_CYCLE = 100

# Phase 6, item 1: consecutive-failure escalation. Both _reconcile_positions
# (live) and _reconcile_paper_resolutions (paper) are mutually exclusive per
# mode and represent the same "positions reconcile" surface, so they share
# one counter. After this many consecutive misses, fire a telegram alert;
# any success resets the counter to 0.
RECONCILE_FAILURE_ALERT_THRESHOLD = 5

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class BankrollView:
    """Bankroll value + the monotonic timestamp it was reconciled at.

    Phase 3 (Single-owner state): Reconciler.bankroll STAYS a bare float
    -- five existing call sites consume it as a number and two fail
    silently if handed an object (see reconciler.bankroll_view docstring
    below). BankrollView is an ADDITIONAL, separate read for callers that
    want to reason about staleness (e.g. strategy_loop logging
    bankroll_age_s in the risk-cap decision filters dict) without
    touching the float plumbing at all.
    """
    value: float
    as_of: float  # time.monotonic() timestamp of the last reconcile


class Reconciler:
    def __init__(self, provider, risk: RiskManager, order_manager=None) -> None:
        self._provider = provider
        self._risk = risk
        self._order_manager = order_manager
        self._bankroll: float = 0.0
        self._bankroll_as_of: float = 0.0
        # Phase 6, item 1: consecutive reconcile-failure counter (positions/
        # paper-resolutions surface -- the two are mutually exclusive per
        # mode and share this counter).
        self._reconcile_failures: int = 0
        # Bugfix plan Bug 2: SEPARATE counter for the order-reconcile
        # surface (_reconcile_orders' fetch_open_orders()->None path). Kept
        # parallel to, not merged with, _reconcile_failures above -- orders
        # and positions/paper-resolutions are independent failure surfaces
        # (e.g. the order API can be down while positions reconcile fine).
        self._order_reconcile_failures: int = 0

    @property
    def bankroll(self) -> float:
        """STAYS a bare float -- do not change this type. Five call sites
        consume it as a number and two fail SILENTLY if handed anything
        else: main.py's set_bankroll(...)/set_bankroll_source(lambda: ...)
        call sites, state_publisher's json.dumps (would raise every 2s),
        and worst, strategy_loop.bankroll's `float(self._bankroll_fn())`
        wrapped in a bare `except: return self._bankroll` -- a non-float
        source would silently freeze the reported bankroll at the
        startup-captured value forever with the test suite still green.
        See bankroll_view for a value+timestamp pair on a separate path."""
        return self._bankroll

    @property
    def bankroll_view(self) -> "BankrollView":
        """(value, as_of) pair for staleness-aware consumers. Additive --
        does not replace the float `bankroll` property above."""
        return BankrollView(value=self._bankroll, as_of=self._bankroll_as_of)

    @property
    def reconcile_failures(self) -> int:
        """Current consecutive-failure count (Phase 6, item 1)."""
        return self._reconcile_failures

    @property
    def order_reconcile_failures(self) -> int:
        """Current consecutive-failure count for the order-reconcile
        surface (Bugfix plan Bug 2) -- separate from reconcile_failures
        above, which covers positions/paper-resolutions."""
        return self._order_reconcile_failures

    def _record_order_reconcile_success(self) -> None:
        """Reset the order-reconcile failure counter and publish it
        whenever it changes, mirroring _record_reconcile_success."""
        if self._order_reconcile_failures != 0:
            self._order_reconcile_failures = 0
            engine_state_repo.publish("order_reconcile_failures", 0)

    def _record_order_reconcile_failure(self) -> None:
        """Bump the order-reconcile consecutive-failure counter, publish
        it, and fire the existing telegram escalation alert once the
        threshold is hit -- mirrors _record_reconcile_failure exactly,
        including >= (fire-every-cycle-at-or-above-threshold) semantics."""
        self._order_reconcile_failures += 1
        engine_state_repo.publish(
            "order_reconcile_failures", self._order_reconcile_failures
        )
        if self._order_reconcile_failures >= RECONCILE_FAILURE_ALERT_THRESHOLD:
            from infra import telegram
            telegram.fire(
                telegram.alert_reconcile_failures(
                    self._order_reconcile_failures,
                    "reconcile_orders",
                    "fetch_open_orders returned None",
                )
            )

    def _record_reconcile_success(self) -> None:
        """Reset the consecutive-failure counter and publish it whenever it
        changes (avoids a write on every healthy cycle when it's already 0)."""
        if self._reconcile_failures != 0:
            self._reconcile_failures = 0
            engine_state_repo.publish("reconcile_failures", 0)

    def _record_reconcile_failure(self, context: str, error: Exception) -> None:
        """Bump the consecutive-failure counter, publish it, and fire a
        telegram alert once the escalation threshold is hit."""
        self._reconcile_failures += 1
        engine_state_repo.publish("reconcile_failures", self._reconcile_failures)
        if self._reconcile_failures >= RECONCILE_FAILURE_ALERT_THRESHOLD:
            from infra import telegram
            telegram.fire(
                telegram.alert_reconcile_failures(
                    self._reconcile_failures, context, str(error)
                )
            )

    async def full_reconcile(self) -> None:
        """Run on startup + periodically to re-sync with API.

        Paper positions PERSIST across restarts (D3): they have no on-chain
        resolution path of their own, so _reconcile_positions' paper branch
        delegates to _reconcile_paper_resolutions, which polls Gamma per
        open position's slug and closes resolved ones with real PnL.
        Unresolved positions stay OPEN — the bot still holds them. There is
        no more "clean slate" zero-close on startup; delete fader.db for a
        fresh paper session.
        """
        # Positions before orders: order reconcile decides FILLED-vs-UNKNOWN
        # by looking for an open position on the order's token.
        await asyncio.gather(
            self._reconcile_bankroll(),
            self._reconcile_positions(),
        )
        await self._reconcile_orders()

    # ------------------------------------------------------------------
    # Bankroll
    # ------------------------------------------------------------------

    async def _reconcile_bankroll(self) -> None:
        balance = await self._provider.async_fetch_usdc_balance()
        self._bankroll = balance
        self._bankroll_as_of = time.monotonic()
        engine_state_repo.publish("bankroll", balance)
        logger.info(f"Bankroll reconciled: ${balance:.2f}")
        # Kick breaker check with fresh bankroll
        if self._risk.check_breaker_against_bankroll(balance):
            logger.warning("Circuit breaker tripped after bankroll reconcile")

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def _reconcile_orders(self) -> None:
        """Sync open orders from API; mark vanished orders UNKNOWN.  Reap stale UNKNOWN.

        In paper mode this is a no-op: paper orders have SIM- IDs with no API counterpart.

        fetch_open_orders() returns None when the API call itself failed
        (Phase 2) -- previously it returned [] on error, indistinguishable
        from "no orders", which made an API outage mass-mark every PENDING
        order UNKNOWN. Skip the whole reconcile cycle on None instead.

        Bugfix plan Bug 2: that None-skip previously had no counter, no
        engine_state signal, and no telegram alert -- a persistent order-API
        outage produced warnings forever with no escalation. Now mirrors
        the existing positions/paper-resolutions escalation mechanism via
        a SEPARATE counter (_order_reconcile_failures): increments and
        alerts (>= threshold, every cycle) on None, resets on a completed
        cycle. Paper mode returns before the fetch, so the counter never
        moves in paper mode (there is no order API in paper) -- matches
        the existing paper carve-out for _reconcile_failures.
        """
        if self._provider.is_paper:
            return
        live_orders = await self._provider.async_fetch_open_orders()
        if live_orders is None:
            logger.warning("_reconcile_orders: fetch_open_orders failed — skipping cycle")
            self._record_order_reconcile_failure()
            return
        live_ids = {o["order_id"] for o in live_orders if o.get("order_id")}

        now_iso = _utc_now()
        # 1. Reaper: UNKNOWN > 1 hour -> CANCELLED.
        #
        # Phase 6, item 8: this UPDATE previously ran on a shared connection
        # that was never committed (Db.connect(), no conn.commit()) -- a
        # silent no-op preserved verbatim through Phases 1-5 specifically to
        # avoid a behavior change mid-refactor. Fixed deliberately here: call
        # reap_stale_unknown with no conn so OrdersRepo opens its own
        # connection and commits. Intentional behavior change -- stale
        # UNKNOWN orders now actually get marked CANCELLED after the 1h TTL.
        orders_repo.reap_stale_unknown(now_iso)

        conn = db.connect()
        rows = []  # safe default if SQL raises before assignment
        try:
            # 2. Find pending / previously-unseen orders not in live API
            rows = orders_repo.not_terminal(conn=conn)
        finally:
            conn.close()

        for row in rows:
            oid = row["order_id"]
            token_id = row["token_id"]
            if oid.startswith("SIM-") or oid.startswith("FAKE-"):
                continue  # paper mode simulated
            if oid not in live_ids:
                # An open position on this token means the order filled
                # (positions reconcile runs before this). Otherwise: not
                # live and no position — don't guess FILLED, mark UNKNOWN.
                if _has_open_position(token_id):
                    if self._order_manager:
                        self._order_manager.mark_filled(oid, token_id)
                    else:
                        _update_order_status(oid, "FILLED")
                    logger.info(
                        f"Order {oid[:16]} no longer live + position open — FILLED"
                    )
                    continue
                _update_order_status(oid, "UNKNOWN")
                # Also pop from OrderManager._resting so re-entry isn't blocked
                if self._order_manager:
                    self._order_manager.mark_vanished(token_id)
                logger.info(
                    f"Order {oid[:16]} no longer live — marked UNKNOWN"
                )

        # Full pass completed (live list obtained, cycle ran) -- reset the
        # order-reconcile failure counter. No try/except is added around
        # this method's body (per bugfix plan: raise semantics stay
        # identical, counter increments ONLY on the None-skip path above).
        self._record_order_reconcile_success()

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def _reconcile_positions(self) -> None:
        """
        Import open positions from Data API; check closed positions for resolution PnL.

        Paper positions have no Data-API counterpart (importing from the
        real API would pull stale live positions into the paper session),
        so paper mode delegates to Gamma-based resolution polling instead.
        """
        if self._provider.is_paper:
            await self._reconcile_paper_resolutions()
            return
        api_positions = await self._provider.async_fetch_open_positions()
        api_closed = await asyncio.get_running_loop().run_in_executor(
            None, self._provider.fetch_all_closed_positions
        )

        # Index API data by conditionId+outcome (same key as our position_id)
        user = _get_env("POLYMARKET_USER_ADDRESS")
        api_open_ids = set()

        try:
            with db.transaction() as conn:
                for pos in api_positions:
                    pos_id = _gen_position_id(pos, user)
                    api_open_ids.add(pos_id)
                    # Upsert if not in our DB
                    existing = conn.execute(
                        "SELECT 1 FROM positions WHERE position_id=?", (pos_id,)
                    ).fetchone()
                    if not existing:
                        _import_position(conn, pos, pos_id, user)

                # Mark positions closed if they appear in closed-positions API
                for cpos in api_closed:
                    pos_id = _gen_position_id(cpos, user)
                    if pos_id not in api_open_ids:
                        pnl = float(cpos.get("realizedPnl", 0) or 0)
                        conn.execute(
                            """
                            UPDATE positions SET status='CLOSED', realized_pnl=?,
                                resolved_at=?
                            WHERE position_id=? AND status='OPEN'
                            """,
                            (pnl, _utc_now(), pos_id),
                        )
                        if conn.execute(
                            "SELECT changes()"
                        ).fetchone()[0] > 0:
                            self._risk.record_pnl_event(pnl, conn)
                            logger.info(f"Position {pos_id[:24]} closed; PnL={pnl:.4f}")
                            from infra import telegram
                            slug = cpos.get("slug", cpos.get("marketSlug", "?"))
                            telegram.fire(telegram.alert_position_resolved(slug, pnl))
            self._record_reconcile_success()
        except Exception as e:
            logger.error(f"reconcile_positions: {e}", exc_info=True)
            self._record_reconcile_failure("reconcile_positions", e)

    async def _reconcile_paper_resolutions(self) -> None:
        """Paper-mode resolution polling (D1/D3).

        Paper positions have no Data-API record; the only ground truth for
        "has this market resolved, and who won" is Gamma /markets?slug=.
        Poll each open paper position's slug (capped, rate-limited — D4),
        derive the winner, and close positions whose market has resolved
        with real PnL (D2) — same risk/alert path as the live close (D5).
        Never raises: this runs inside the 60s resolution poller and must
        not take the loop down on a bad response.
        """
        try:
            rows = positions_repo.open_for_paper_poll(MAX_PAPER_POLL_PER_CYCLE)

            if not rows:
                self._record_reconcile_success()
                return

            # requests is sync — run the Gamma sweep off the event loop.
            def _fetch_all() -> Dict[str, Optional[Dict[str, Any]]]:
                out: Dict[str, Optional[Dict[str, Any]]] = {}
                for i, row in enumerate(rows):
                    if i > 0:
                        time.sleep(CALL_DELAY_S)
                    out[row["position_id"]] = fetch_market_metadata(row["slug"])
                return out

            meta_by_pos = await asyncio.get_running_loop().run_in_executor(
                None, _fetch_all
            )

            with db.transaction() as conn:
                for row in rows:
                    meta = meta_by_pos.get(row["position_id"])
                    if not meta:
                        continue  # market metadata unavailable — leave OPEN
                    # Gamma returns outcomes/outcomePrices as JSON strings.
                    raw_outcomes = meta.get("outcomes", "[]")
                    outcomes = (
                        json.loads(raw_outcomes)
                        if isinstance(raw_outcomes, str)
                        else raw_outcomes
                    )
                    winner = _resolution_from_outcome_prices(
                        outcomes, meta.get("outcomePrices"), meta.get("closed")
                    )
                    if not winner:
                        continue  # not resolved / unusable data — leave OPEN

                    held = (row["outcome"] or "").strip().upper()
                    payout = 1.0 if held == winner else 0.0
                    entry_price = float(row["entry_price"])
                    size = float(row["size"])
                    pnl = (payout - entry_price) * size

                    n = positions_repo.close(
                        row["position_id"], pnl, _utc_now(), conn=conn,
                    )
                    if n > 0:
                        self._risk.record_pnl_event(pnl, conn)
                        logger.info(
                            f"[PAPER] Position {row['position_id'][:24]} "
                            f"resolved ({winner}); PnL={pnl:.4f}"
                        )
                        from infra import telegram
                        telegram.fire(
                            telegram.alert_position_resolved(row["slug"], pnl)
                        )
            self._record_reconcile_success()
        except Exception as e:
            logger.error(f"reconcile_paper_resolutions: {e}", exc_info=True)
            self._record_reconcile_failure("reconcile_paper_resolutions", e)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _gen_position_id(pos: Dict[str, Any], user: str) -> str:
    cid = pos.get("conditionId", "")
    oidx = pos.get("outcomeIndex", 0)
    return f"{user}:{cid}:{oidx}"


def _import_position(conn, pos: Dict, pos_id: str, user: str) -> None:
    now = _utc_now()
    conn.execute(
        """
        INSERT OR IGNORE INTO positions
          (position_id, slug, condition_id, token_id, outcome, entry_price,
           size, notional, status, opened_at, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, 'RECONCILE_IMPORT')
        """,
        (
            pos_id,
            pos.get("slug", pos.get("marketSlug", "")),
            pos.get("conditionId", ""),
            # Data-API /positions returns the ERC1155 token id under "asset".
            pos.get("asset", pos.get("asset_id", pos.get("tokenId", ""))),
            pos.get("outcome", ""),
            float(pos.get("avgPrice", pos.get("curPrice", 0)) or 0),
            float(pos.get("size", pos.get("totalBought", 0)) or 0),
            float(pos.get("value", pos.get("currentValue", 0)) or 0),
            now,
        ),
    )


def _has_open_position(token_id: str) -> bool:
    return positions_repo.has_open(token_id)


def _update_order_status(order_id: str, status: str) -> None:
    orders_repo.set_status(order_id, status)


def _set_state(key: str, value: Any) -> None:
    engine_state_repo.publish(key, value)


def _get_env(key: str) -> str:
    import os
    return os.getenv(key, "")
