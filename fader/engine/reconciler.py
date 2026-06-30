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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from engine.risk import RiskManager

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

        In paper mode: close any leftover OPEN positions from previous sessions
        (paper positions have no on-chain resolution path) so the dashboard
        shows a clean slate.
        """
        if self._provider._mode == "paper":
            from infra.db import get_connection
            conn = get_connection()
            try:
                now_iso = _utc_now()
                result = conn.execute(
                    "UPDATE positions SET status='CLOSED', realized_pnl=0.0, "
                    "resolved_at=? WHERE status='OPEN'",
                    (now_iso,),
                )
                n = result.rowcount
                if n > 0:
                    conn.commit()
                    logger.info(
                        f"Paper mode startup: closed {n} stale position(s) "
                        f"from previous sessions"
                    )
            except Exception as e:
                logger.error(f"Paper startup cleanup error: {e}")
            finally:
                conn.close()
        await asyncio.gather(
            self._reconcile_bankroll(),
            self._reconcile_orders(),
            self._reconcile_positions(),
        )

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
                # Not live — don't guess FILLED. Mark UNKNOWN.
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

        In paper mode this is a no-op: paper positions have no on-chain counterpart
        and the real API would import stale live positions into the paper session.
        """
        if self._provider._mode == "paper":
            return
        from infra.db import get_connection
        api_positions = await self._provider.async_fetch_open_positions()
        api_closed = await asyncio.get_event_loop().run_in_executor(
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
                        self._risk.record_pnl_event(pnl)
                        logger.info(f"Position {pos_id[:24]} closed; PnL={pnl:.4f}")
                        from infra import telegram
                        slug = cpos.get("slug", cpos.get("marketSlug", "?"))
                        asyncio.create_task(
                            telegram.alert_position_resolved(slug, pnl)
                        )

            conn.commit()
        except Exception as e:
            logger.error(f"reconcile_positions: {e}", exc_info=True)
        finally:
            conn.close()


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
            pos.get("asset_id", pos.get("tokenId", "")),
            pos.get("outcome", ""),
            float(pos.get("avgPrice", pos.get("curPrice", 0)) or 0),
            float(pos.get("size", pos.get("totalBought", 0)) or 0),
            float(pos.get("value", pos.get("currentValue", 0)) or 0),
            now,
        ),
    )


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
    from infra.db import get_connection
    now = _utc_now()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO engine_state (key, value_json, updated_at)
            VALUES (?, ?, ?)
            """,
            (key, json.dumps(value), now),
        )
        conn.commit()
    finally:
        conn.close()


def _get_env(key: str) -> str:
    import os
    return os.getenv(key, "")
