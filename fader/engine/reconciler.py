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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from engine.risk import RiskManager
from marketdata.rest_market import (
    CALL_DELAY_S,
    _resolution_from_outcome_prices,
    fetch_market_metadata,
)

# Cap on open paper positions polled per resolution cycle (D4). ~20 open
# positions today; 100 leaves headroom without risking a slow Gamma sweep
# blocking the 60s resolution poller for multiple cycles.
MAX_PAPER_POLL_PER_CYCLE = 100

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Reconciler:
    def __init__(self, provider, risk: RiskManager, order_manager=None) -> None:
        self._provider = provider
        self._risk = risk
        self._order_manager = order_manager
        self._bankroll: float = 0.0

    @property
    def bankroll(self) -> float:
        return self._bankroll

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
        _set_state("bankroll", balance)
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
        """
        if self._provider._mode == "paper":
            return
        from infra.db import get_connection
        live_orders = await self._provider.async_fetch_open_orders()
        live_ids = {o["order_id"] for o in live_orders if o.get("order_id")}

        conn = get_connection()
        rows = []  # safe default if SQL raises before assignment
        try:
            now_iso = _utc_now()
            # 1. Reaper: UNKNOWN > 1 hour → CANCELLED
            conn.execute(
                """
                UPDATE orders SET status='CANCELLED', cancel_reason='unknown_ttl'
                WHERE status='UNKNOWN'
                  AND created_at < datetime(?, '-3600 seconds')
                """,
                (now_iso,),
            )

            # 2. Find pending / previously-unseen orders not in live API
            rows = conn.execute(
                "SELECT order_id, status, token_id FROM orders "
                "WHERE status NOT IN ('FILLED','CANCELLED','FAILED','UNKNOWN')"
            ).fetchall()
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
        if self._provider._mode == "paper":
            await self._reconcile_paper_resolutions()
            return
        from infra.db import get_connection
        api_positions = await self._provider.async_fetch_open_positions()
        api_closed = await asyncio.get_running_loop().run_in_executor(
            None, self._provider.fetch_all_closed_positions
        )

        # Index API data by conditionId+outcome (same key as our position_id)
        user = _get_env("POLYMARKET_USER_ADDRESS")
        api_open_ids = set()

        conn = get_connection()
        try:
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

            conn.commit()
        except Exception as e:
            logger.error(f"reconcile_positions: {e}", exc_info=True)
        finally:
            conn.close()

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
            from infra.db import get_connection

            conn = get_connection()
            try:
                rows = conn.execute(
                    "SELECT position_id, slug, outcome, entry_price, size "
                    "FROM positions WHERE status='OPEN' LIMIT ?",
                    (MAX_PAPER_POLL_PER_CYCLE,),
                ).fetchall()
            finally:
                conn.close()

            if not rows:
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

            conn = get_connection()
            try:
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

                    cur = conn.execute(
                        """
                        UPDATE positions SET status='CLOSED', realized_pnl=?,
                            resolved_at=?
                        WHERE position_id=? AND status='OPEN'
                        """,
                        (pnl, _utc_now(), row["position_id"]),
                    )
                    if cur.rowcount > 0:
                        self._risk.record_pnl_event(pnl, conn)
                        logger.info(
                            f"[PAPER] Position {row['position_id'][:24]} "
                            f"resolved ({winner}); PnL={pnl:.4f}"
                        )
                        from infra import telegram
                        telegram.fire(
                            telegram.alert_position_resolved(row["slug"], pnl)
                        )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"reconcile_paper_resolutions: {e}", exc_info=True)


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


def _update_order_status(order_id: str, status: str) -> None:
    from infra.db import get_connection
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE orders SET status=? WHERE order_id=?", (status, order_id)
        )
        conn.commit()
    finally:
        conn.close()


def _set_state(key: str, value: Any) -> None:
    from infra.db import execute_write
    now = _utc_now()
    execute_write(
        "INSERT OR REPLACE INTO engine_state (key, value_json, updated_at) VALUES (?, ?, ?)",
        (key, json.dumps(value), now),
    )


def _get_env(key: str) -> str:
    import os
    return os.getenv(key, "")
