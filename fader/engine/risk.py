"""engine/risk.py

Daily-loss circuit breaker, max-deployed cap, per-market cap.
All caps key off cash bankroll (USDC.e on-chain).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from infra.db import get_connection

logger = logging.getLogger(__name__)


def _safe_fire_alert(coro) -> None:
    """Schedule an async coroutine from a potentially sync context.
    Delegates to infra.telegram.fire(), which retains a task ref (GC-safe)
    and falls back to a background thread when no event loop is running."""
    from infra import telegram
    telegram.fire(coro)


class RiskManager:
    """
    Manages:
      - Daily-loss circuit breaker (resets midnight UTC, manual reset also allowed).
      - Max deployed % cap across all open positions.
      - Per-market cap.
      - MATIC balance gate (gas exhaustion guard).
    """

    def __init__(
        self,
        daily_loss_pct: float = 5.0,
        max_deployed_pct: float = 100.0,
        per_market_cap_pct: float = 5.0,
        matic_min_balance: float = 0.5,
    ) -> None:
        self._daily_loss_pct = daily_loss_pct
        self._max_deployed_pct = max_deployed_pct
        self._per_market_pct = per_market_cap_pct
        self._matic_min_balance = matic_min_balance
        self._matic_balance: float = 0.0
        self._breaker_tripped = False
        self._tripped_day: Optional[str] = None
        self._last_matic_alert_ts: float = 0.0

    def update_params(
        self,
        daily_loss_pct: float,
        max_deployed_pct: float,
        per_market_cap_pct: float,
        matic_min_balance: Optional[float] = None,
    ) -> None:
        self._daily_loss_pct = daily_loss_pct
        self._max_deployed_pct = max_deployed_pct
        self._per_market_pct = per_market_cap_pct
        if matic_min_balance is not None:
            self._matic_min_balance = matic_min_balance

    def set_matic_balance(self, balance: float) -> None:
        self._matic_balance = balance

    @property
    def breaker_tripped(self) -> bool:
        # Daily breaker: auto-clears at UTC midnight. The DB row is keyed by
        # day so a new day starts untripped; this clears the sticky in-memory
        # flag to match (previously it persisted until manual reset/restart).
        if (
            self._breaker_tripped
            and self._tripped_day
            and self._tripped_day != self.today_utc()
        ):
            self._breaker_tripped = False
            self._tripped_day = None
            logger.info("Circuit breaker auto-reset (UTC day rollover)")
        return self._breaker_tripped

    # ------------------------------------------------------------------
    # Breaker
    # ------------------------------------------------------------------

    def today_utc(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def record_pnl_event(self, pnl_delta: float) -> None:
        """Add realized PnL (positive=win, negative=loss) to today's tally."""
        day = self.today_utc()
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO circuit_breaker (day, realized_pnl)
                VALUES (?, ?)
                ON CONFLICT(day) DO UPDATE SET
                    realized_pnl = realized_pnl + excluded.realized_pnl
                """,
                (day, pnl_delta),
            )
            conn.commit()
        finally:
            conn.close()
        self._check_breaker(day)

    def _check_breaker(self, day: str) -> None:
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT realized_pnl, tripped FROM circuit_breaker WHERE day = ?",
                (day,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return
        pnl = row["realized_pnl"]
        if row["tripped"]:
            self._breaker_tripped = True
            self._tripped_day = day
            return
        # We need cash bankroll to compute %; caller must check allow_entry() which
        # uses the last-known bankroll from engine state. Here we just mark.
        # The actual % check is done in allow_entry().
        self._breaker_tripped = bool(row["tripped"])

    def check_breaker_against_bankroll(self, cash_bankroll: float) -> bool:
        """
        Return True (trip) if today's loss exceeds daily_loss_pct of bankroll.
        Also persists the trip flag.
        """
        if cash_bankroll <= 0:
            return False
        day = self.today_utc()
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT realized_pnl, tripped FROM circuit_breaker WHERE day = ?",
                (day,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return False
        if row["tripped"]:
            self._breaker_tripped = True
            self._tripped_day = day
            return True
        pnl = row["realized_pnl"]
        loss_pct = (-pnl / cash_bankroll) * 100 if pnl < 0 else 0.0
        if loss_pct >= self._daily_loss_pct:
            self._trip(day)
            return True
        return False

    def _trip(self, day: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO circuit_breaker (day, tripped, tripped_at)
                VALUES (?, 1, ?)
                ON CONFLICT(day) DO UPDATE SET tripped=1, tripped_at=excluded.tripped_at
                """,
                (day, now),
            )
            conn.commit()
        finally:
            conn.close()
        self._breaker_tripped = True
        self._tripped_day = day
        logger.error(f"Circuit breaker TRIPPED for {day}")
        from infra import telegram
        _safe_fire_alert(telegram.alert_breaker_tripped(self._daily_loss_pct, day))

    def reset_breaker(self) -> None:
        day = self.today_utc()
        now = datetime.now(timezone.utc).isoformat()
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO circuit_breaker (day, tripped, reset_at)
                VALUES (?, 0, ?)
                ON CONFLICT(day) DO UPDATE SET tripped=0, reset_at=excluded.reset_at
                """,
                (day, now),
            )
            conn.commit()
        finally:
            conn.close()
        self._breaker_tripped = False
        self._tripped_day = None
        logger.info(f"Circuit breaker reset for {day}")

    # ------------------------------------------------------------------
    # Deployment caps
    # ------------------------------------------------------------------

    def allow_entry(
        self,
        slug: str,
        order_notional: float,
        cash_bankroll: float,
        deployed_notional: float,
        market_deployed_notional: float,
    ) -> Tuple[bool, str]:
        """
        Returns (allowed, reason).
        Checks:
          1. Breaker not tripped.
          2. Total deployed + order <= max_deployed_pct of cash.
          3. Market deployed + order <= per_market_cap_pct of cash.
        """
        # MATIC gate removed — CLOB orders are off-chain signed messages, no gas needed.
        if self.breaker_tripped:  # property: handles UTC day rollover
            return False, "circuit_breaker_tripped"
        if cash_bankroll <= 0:
            return False, "zero_bankroll"
        max_total = cash_bankroll * self._max_deployed_pct / 100.0
        if deployed_notional + order_notional > max_total:
            return False, (
                f"max_deployed_cap: deployed={deployed_notional:.2f}"
                f" + order={order_notional:.2f} > cap={max_total:.2f}"
            )
        max_mkt = cash_bankroll * self._per_market_pct / 100.0
        if market_deployed_notional + order_notional > max_mkt:
            return False, (
                f"per_market_cap({slug}): {market_deployed_notional:.2f}"
                f" + {order_notional:.2f} > {max_mkt:.2f}"
            )
        return True, "ok"

    def _maybe_alert_matic(self) -> None:
        """Send low-MATIC Telegram alert with 1-hour cooldown."""
        now = __import__("time").monotonic()
        if now - self._last_matic_alert_ts < 3600:
            return
        self._last_matic_alert_ts = now
        from infra import telegram
        _safe_fire_alert(
            telegram.send(
                f"<b>Low MATIC</b>\n"
                f"Balance: {self._matic_balance:.4f} MATIC\n"
                f"Minimum: {self._matic_min_balance} MATIC\n"
                f"New entries halted."
            )
        )


def get_open_notional() -> Tuple[float, Dict[str, float]]:
    """
    Return (total_deployed, {slug: deployed}) from open positions table.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT slug, notional FROM positions WHERE status='OPEN'"
        ).fetchall()
    finally:
        conn.close()
    total = 0.0
    by_slug: Dict[str, float] = {}
    for row in rows:
        n = float(row["notional"])
        total += n
        by_slug[row["slug"]] = by_slug.get(row["slug"], 0.0) + n
    return total, by_slug
