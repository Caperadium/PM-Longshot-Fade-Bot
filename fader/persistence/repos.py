"""persistence/repos.py

Typed repository layer over infra/db.py. All engine-side SQL lives here
(Phase 1 of the architecture refactor -- see temp/implementation-plan.md).

Rules:
  - SQL is moved VERBATIM from call sites; no query rewrites in this phase.
  - Every repo method takes an optional `conn`. When a conn is passed, the
    method executes on it and does NOT commit or close (the caller owns the
    transaction lifecycle). When conn is None, a connection is opened, used,
    committed, and closed per call -- matching current per-call behavior.
    Modeled on engine/risk.py's pre-existing record_pnl_event(conn=None).
  - Db.transaction() opens one connection with isolation_level=None
    (autocommit off pysqlite's implicit-BEGIN machinery) and issues an
    explicit BEGIN IMMEDIATE ... COMMIT/ROLLBACK, so callers get a real
    all-or-nothing transaction across multiple statements.

No PmSyncRepo: pm_trades/pm_closed_positions/pm_sync_metadata have zero call
sites outside table creation in infra/db.py; those tables are left alone.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from infra.db import get_connection

logger = logging.getLogger(__name__)

# Bugfix plan (temp/bugfix-plan.md), Bug 1: reap_stale_unknown's TTL window,
# in seconds. Was an inline "-3600 seconds" fed to SQLite's datetime()
# function; now a named constant consumed by a Python-side cutoff
# computation (see reap_stale_unknown for why).
UNKNOWN_ORDER_TTL_S = 3600


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Db:
    """Owns transaction lifecycle for multi-statement writes.

    Opens connections with isolation_level=None (disables pysqlite's
    implicit BEGIN-on-first-write) and issues an explicit BEGIN IMMEDIATE
    so the transaction acquires the write lock up front; COMMIT on success,
    ROLLBACK on any exception.
    """

    def connect(self) -> sqlite3.Connection:
        """Plain connection factory, no transaction semantics -- for call
        sites that only need to share one connection across multiple
        READ-ONLY repo calls without Db.transaction()'s commit-on-success
        overhead (e.g. engine/reconciler.py._reconcile_orders' not_terminal
        lookup).

        Historical note: this same shared connection used to also carry
        the order-reaper's stale-UNKNOWN-to-CANCELLED UPDATE, which never
        got a conn.commit() call -- a silent no-op present since the
        initial commit and preserved verbatim through Phases 1-5
        specifically to avoid a behavior change mid-refactor. Phase 6,
        item 8 fixed it deliberately: the reaper now goes through
        OrdersRepo.reap_stale_unknown() with no conn, so it opens its own
        connection and commits."""
        return get_connection()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = get_connection()
        conn.isolation_level = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            # If BEGIN itself failed there is no transaction to roll back;
            # a bare ROLLBACK would raise and mask the original error.
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
        finally:
            conn.close()


class PositionsRepo:
    def has_open(self, token_id: str, conn: Optional[sqlite3.Connection] = None) -> bool:
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            row = conn.execute(
                "SELECT 1 FROM positions WHERE token_id=? AND status='OPEN' LIMIT 1",
                (token_id,),
            ).fetchone()
            return row is not None
        finally:
            if own_conn:
                conn.close()

    def open_positions(self, conn: Optional[sqlite3.Connection] = None) -> List[sqlite3.Row]:
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            return conn.execute(
                "SELECT slug, notional FROM positions WHERE status='OPEN'"
            ).fetchall()
        finally:
            if own_conn:
                conn.close()

    def open_for_close(self, conn: Optional[sqlite3.Connection] = None) -> List[sqlite3.Row]:
        """(token_id, size, notional) of every OPEN position, for
        order_manager.close_all's live-mode market-sell path."""
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            return conn.execute(
                "SELECT token_id, size, notional FROM positions WHERE status='OPEN'"
            ).fetchall()
        finally:
            if own_conn:
                conn.close()

    def deployed_by_slug(
        self, conn: Optional[sqlite3.Connection] = None
    ) -> Dict[str, float]:
        _, by_slug = self.deployed_total(conn=conn)
        return by_slug

    def deployed_total(
        self, conn: Optional[sqlite3.Connection] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Return (total_deployed, {slug: deployed}) from open positions.

        Absorbs engine.risk.get_open_notional's query verbatim; that
        function becomes a thin delegate to this method (kept for
        provider.py/strategy_loop.py call sites until Phase 2 repoints
        them).
        """
        rows = self.open_positions(conn=conn)
        total = 0.0
        by_slug: Dict[str, float] = {}
        for row in rows:
            n = float(row["notional"])
            total += n
            by_slug[row["slug"]] = by_slug.get(row["slug"], 0.0) + n
        return total, by_slug

    def open_count(self, conn: Optional[sqlite3.Connection] = None) -> int:
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM positions WHERE status='OPEN'"
            ).fetchone()[0]
        finally:
            if own_conn:
                conn.close()

    def open_notional(self, conn: Optional[sqlite3.Connection] = None) -> float:
        """Sum of notional across OPEN positions (deployed capital)."""
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            return float(conn.execute(
                "SELECT COALESCE(SUM(notional), 0) FROM positions WHERE status='OPEN'"
            ).fetchone()[0])
        finally:
            if own_conn:
                conn.close()

    def realized_pnl_total(self, conn: Optional[sqlite3.Connection] = None) -> float:
        """Sum of realized_pnl across all CLOSED positions."""
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            return float(conn.execute(
                "SELECT COALESCE(SUM(realized_pnl), 0) FROM positions "
                "WHERE status='CLOSED'"
            ).fetchone()[0])
        finally:
            if own_conn:
                conn.close()

    def realized_pnl_today(self, conn: Optional[sqlite3.Connection] = None) -> float:
        """Sum of realized_pnl for positions resolved today (UTC).

        date() tolerates the ISO variants writers use for resolved_at
        (space or 'T' separator, optional offset suffix)."""
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            return float(conn.execute(
                "SELECT COALESCE(SUM(realized_pnl), 0) FROM positions "
                "WHERE status='CLOSED' AND realized_pnl IS NOT NULL "
                "AND date(resolved_at) = date('now')"
            ).fetchone()[0])
        finally:
            if own_conn:
                conn.close()

    def open_for_paper_poll(
        self, limit: int, conn: Optional[sqlite3.Connection] = None
    ) -> List[sqlite3.Row]:
        """OPEN positions capped at `limit`, for the paper-mode resolution
        poller's per-cycle Gamma sweep (reconciler._reconcile_paper_resolutions)."""
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            return conn.execute(
                "SELECT position_id, slug, outcome, entry_price, size "
                "FROM positions WHERE status='OPEN' LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            if own_conn:
                conn.close()

    def insert_open(self, row: Dict[str, Any], conn: Optional[sqlite3.Connection] = None) -> None:
        """INSERT OR IGNORE a new OPEN position row.

        `row` keys: position_id, slug, condition_id, token_id, entry_price,
        size, notional, opened_at, entry_order_id, entry_decision_id.
        outcome/status/source match the ENGINE_FILL insert used by
        order_manager._insert_position (outcome fixed to 'No', status
        'OPEN', source 'ENGINE_FILL').
        """
        own_conn = conn is None
        if own_conn:
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
                (
                    row["position_id"], row["slug"], row["condition_id"],
                    row["token_id"], row["entry_price"], row["size"],
                    row["notional"], row["opened_at"], row["entry_order_id"],
                    row["entry_decision_id"],
                ),
            )
            if own_conn:
                conn.commit()
        finally:
            if own_conn:
                conn.close()

    def close(
        self,
        position_id: str,
        realized_pnl: float,
        resolved_at: str,
        conn: Optional[sqlite3.Connection] = None,
    ) -> int:
        """UPDATE a position to CLOSED with realized_pnl. Returns rowcount
        (0 if the position wasn't OPEN or didn't exist)."""
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            cur = conn.execute(
                """
                UPDATE positions SET status='CLOSED', realized_pnl=?,
                    resolved_at=?
                WHERE position_id=? AND status='OPEN'
                """,
                (realized_pnl, resolved_at, position_id),
            )
            if own_conn:
                conn.commit()
            return cur.rowcount
        finally:
            if own_conn:
                conn.close()

    def bulk_close_paper(self, conn: Optional[sqlite3.Connection] = None) -> int:
        """close_all paper branch: mark every OPEN position CLOSED at
        realized_pnl=0.0 (exit-at-entry; no venue to sell into in paper
        mode). Returns rowcount."""
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            now = _utc_now()
            cur = conn.execute(
                "UPDATE positions SET status='CLOSED', realized_pnl=0.0, "
                "resolved_at=? WHERE status='OPEN'",
                (now,),
            )
            if own_conn:
                conn.commit()
            return cur.rowcount
        finally:
            if own_conn:
                conn.close()


class OrdersRepo:
    def by_idem_key(
        self, key: str, conn: Optional[sqlite3.Connection] = None
    ) -> Optional[sqlite3.Row]:
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            return conn.execute(
                "SELECT status FROM orders WHERE idempotency_key = ?", (key,)
            ).fetchone()
        finally:
            if own_conn:
                conn.close()

    def insert(
        self,
        order_id: str,
        idem_key: str,
        slug: str,
        token_id: str,
        price: Optional[float],
        size: float,
        order_type: str,
        status: str = "PENDING",
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        VALID = frozenset({"PENDING", "FILLED", "CANCELLED", "FAILED", "UNKNOWN"})
        if status not in VALID:
            raise ValueError(f"Invalid order status: {status!r}  (valid: {sorted(VALID)})")
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            now = _utc_now()
            conn.execute(
                """
                INSERT OR IGNORE INTO orders
                  (order_id, idempotency_key, slug, token_id, side, type, price, size,
                   status, created_at)
                VALUES (?, ?, ?, ?, 'BUY', ?, ?, ?, ?, ?)
                """,
                (order_id, idem_key, slug, token_id, order_type, price, size, status, now),
            )
            if own_conn:
                conn.commit()
        finally:
            if own_conn:
                conn.close()

    def pending_count(self, conn: Optional[sqlite3.Connection] = None) -> int:
        """Count of PENDING orders (state_publisher's published metric)."""
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM orders WHERE status='PENDING'"
            ).fetchone()[0]
        finally:
            if own_conn:
                conn.close()

    def update_status(
        self,
        order_id: str,
        status: str,
        cancel_reason: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            conn.execute(
                "UPDATE orders SET status=?, cancel_reason=? WHERE order_id=?",
                (status, cancel_reason, order_id),
            )
            if own_conn:
                conn.commit()
        finally:
            if own_conn:
                conn.close()

    def pending_limit_orders(
        self, conn: Optional[sqlite3.Connection] = None
    ) -> List[sqlite3.Row]:
        """PENDING LIMIT orders, newest first (rehydrate_resting)."""
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            return conn.execute(
                """
                SELECT order_id, idempotency_key, slug, token_id, price, size, created_at
                FROM orders
                WHERE status='PENDING' AND type='LIMIT'
                ORDER BY created_at DESC
                """
            ).fetchall()
        finally:
            if own_conn:
                conn.close()

    def not_terminal(
        self, conn: Optional[sqlite3.Connection] = None
    ) -> List[sqlite3.Row]:
        """Orders not yet FILLED/CANCELLED/FAILED/UNKNOWN (reconciler order
        sync)."""
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            return conn.execute(
                "SELECT order_id, status, token_id FROM orders "
                "WHERE status NOT IN ('FILLED','CANCELLED','FAILED','UNKNOWN')"
            ).fetchall()
        finally:
            if own_conn:
                conn.close()

    def set_status(
        self, order_id: str, status: str, conn: Optional[sqlite3.Connection] = None
    ) -> None:
        """Status-only UPDATE (no cancel_reason column touch) -- distinct
        from update_status(), which also writes cancel_reason. Matches
        engine/reconciler.py's module-level _update_order_status verbatim."""
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            conn.execute(
                "UPDATE orders SET status=? WHERE order_id=?", (status, order_id)
            )
            if own_conn:
                conn.commit()
        finally:
            if own_conn:
                conn.close()

    def reap_stale_unknown(
        self, now_iso: str, conn: Optional[sqlite3.Connection] = None
    ) -> None:
        """UNKNOWN orders older than UNKNOWN_ORDER_TTL_S (1 hour) -> CANCELLED
        (reconciler reaper).

        Bugfix plan Bug 1: created_at is written by _utc_now() as
        datetime.isoformat() ("...T07:05:11.231837+00:00" -- 'T' separator,
        microseconds, +00:00 offset). The previous SQL compared that string
        directly against SQLite's datetime(?, '-3600 seconds'), which
        returns "YYYY-MM-DD HH:MM:SS" (space separator, no offset, no
        microseconds). Raw string collation put 'T' (0x54) > ' ' (0x20), so
        any same-UTC-date created_at compared GREATER than the cutoff
        regardless of time-of-day -- the reap only fired once the date
        prefix itself differed, making the effective TTL ~1 day instead of
        1 hour.

        Fix: compute the cutoff in Python via datetime.fromisoformat/
        timedelta and compare it, as a string, against created_at directly
        -- no SQLite date parsing involved (SQLite's parser only documents
        up to 3 fractional-second digits; a parse failure returns NULL,
        which would make the WHERE clause silently false -- the same
        silent-no-reap failure mode, just better hidden). This works
        because every created_at is written by the single _utc_now()
        writer in one uniform format, and cutoff_iso is produced by the
        same isoformat() path, so both sides of '<' share that format and
        uniform ISO-8601 strings order correctly under lexicographic
        comparison.

        Known edges (accepted, not fixed here):
          - A hypothetical legacy row whose created_at used SQLite's
            space-separated format would compare less-than any same-day
            ISO cutoff and get reaped immediately. Only UNKNOWN-status rows
            are eligible, and an early reap of a stale-unknown order is
            benign for this edge; no migration needed.
          - datetime.isoformat() omits the ".ffffff" microsecond block
            entirely when microsecond == 0 (~1-in-10^6 writes). Mixed
            fractional/non-fractional created_at strings still order
            correctly to the second (the wobble is sub-second, always
            harmless for a 1h TTL) -- but the "uniform format" rationale
            above is not absolute, so this is called out explicitly rather
            than silently relied on.
        """
        cutoff_iso = (
            datetime.fromisoformat(now_iso) - timedelta(seconds=UNKNOWN_ORDER_TTL_S)
        ).isoformat()
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            conn.execute(
                """
                UPDATE orders SET status='CANCELLED', cancel_reason='unknown_ttl'
                WHERE status='UNKNOWN'
                  AND created_at < ?
                """,
                (cutoff_iso,),
            )
            if own_conn:
                conn.commit()
        finally:
            if own_conn:
                conn.close()


class BreakerRepo:
    def day_state(
        self, day: str, conn: Optional[sqlite3.Connection] = None
    ) -> Optional[sqlite3.Row]:
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            return conn.execute(
                "SELECT realized_pnl, tripped FROM circuit_breaker WHERE day = ?",
                (day,),
            ).fetchone()
        finally:
            if own_conn:
                conn.close()

    def record_pnl_event(
        self, day: str, pnl_delta: float, conn: Optional[sqlite3.Connection] = None
    ) -> None:
        own_conn = conn is None
        if own_conn:
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
            if own_conn:
                conn.commit()
        finally:
            if own_conn:
                conn.close()

    def trip(self, day: str, conn: Optional[sqlite3.Connection] = None) -> None:
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            now = _utc_now()
            conn.execute(
                """
                INSERT INTO circuit_breaker (day, tripped, tripped_at)
                VALUES (?, 1, ?)
                ON CONFLICT(day) DO UPDATE SET tripped=1, tripped_at=excluded.tripped_at
                """,
                (day, now),
            )
            if own_conn:
                conn.commit()
        finally:
            if own_conn:
                conn.close()

    def reset(self, day: str, conn: Optional[sqlite3.Connection] = None) -> None:
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            now = _utc_now()
            conn.execute(
                """
                INSERT INTO circuit_breaker (day, tripped, reset_at)
                VALUES (?, 0, ?)
                ON CONFLICT(day) DO UPDATE SET tripped=0, reset_at=excluded.reset_at
                """,
                (day, now),
            )
            if own_conn:
                conn.commit()
        finally:
            if own_conn:
                conn.close()


class DecisionsRepo:
    def append(
        self,
        slug: str,
        token_id: Optional[str],
        decision: str,
        reason: str,
        filters: Dict[str, Any],
        order_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> bool:
        """Insert a decisions row. Returns True on success, False on
        failure (caller may log a warning on False -- new in this phase,
        log_decision previously swallowed the exception silently)."""
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            ts = _utc_now()
            conn.execute(
                """
                INSERT INTO decisions
                  (ts, slug, token_id, decision, reason, filters_json, order_id, idempotency_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts, slug, token_id, decision, reason,
                    json.dumps(filters, default=str),
                    order_id, idempotency_key,
                ),
            )
            if own_conn:
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"DecisionsRepo.append failed: {e}")
            return False
        finally:
            if own_conn:
                conn.close()

    def prune(self, retention_days: int, conn: Optional[sqlite3.Connection] = None) -> int:
        cutoff = f"-{retention_days} days"
        sql = "DELETE FROM decisions WHERE ts < datetime('now', ?)"
        if conn is not None:
            return conn.execute(sql, (cutoff,)).rowcount
        from infra.db import execute_write
        return execute_write(sql, (cutoff,))


class ControlRepo:
    def pending(self, conn: Optional[sqlite3.Connection] = None) -> List[sqlite3.Row]:
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            return conn.execute(
                "SELECT id, command, args_json FROM control_commands WHERE status='PENDING' ORDER BY id"
            ).fetchall()
        finally:
            if own_conn:
                conn.close()

    def mark_done(
        self, cmd_id: int, result: str, conn: Optional[sqlite3.Connection] = None
    ) -> None:
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            conn.execute(
                "UPDATE control_commands SET status='DONE', result_json=? WHERE id=?",
                (json.dumps({"result": result}), cmd_id),
            )
            if own_conn:
                conn.commit()
        finally:
            if own_conn:
                conn.close()

    def issue(
        self,
        command: str,
        args: Optional[Dict] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            now = _utc_now()
            conn.execute(
                "INSERT INTO control_commands (ts, command, args_json, status) VALUES (?, ?, ?, 'PENDING')",
                (now, command, json.dumps(args or {})),
            )
            if own_conn:
                conn.commit()
        finally:
            if own_conn:
                conn.close()

    def prune(self, retention_days: int, conn: Optional[sqlite3.Connection] = None) -> int:
        cutoff = f"-{retention_days} days"
        sql = (
            "DELETE FROM control_commands WHERE status != 'PENDING' "
            "AND ts < datetime('now', ?)"
        )
        if conn is not None:
            return conn.execute(sql, (cutoff,)).rowcount
        from infra.db import execute_write
        return execute_write(sql, (cutoff,))


class EngineStateRepo:
    def publish(self, key: str, value: Any, conn: Optional[sqlite3.Connection] = None) -> None:
        own_conn = conn is None
        if own_conn:
            from infra.db import execute_write
            now = _utc_now()
            execute_write(
                "INSERT OR REPLACE INTO engine_state (key, value_json, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, default=str), now),
            )
            return
        now = _utc_now()
        conn.execute(
            "INSERT OR REPLACE INTO engine_state (key, value_json, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, default=str), now),
        )

    def get(self, key: str, conn: Optional[sqlite3.Connection] = None) -> Any:
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            row = conn.execute(
                "SELECT value_json FROM engine_state WHERE key=?", (key,)
            ).fetchone()
            return json.loads(row["value_json"]) if row else None
        finally:
            if own_conn:
                conn.close()


class ConfigKVRepo:
    def all_items(self, conn: Optional[sqlite3.Connection] = None) -> List[sqlite3.Row]:
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            return conn.execute("SELECT key, value FROM config_kv").fetchall()
        finally:
            if own_conn:
                conn.close()

    def get_keys(
        self, keys: List[str], conn: Optional[sqlite3.Connection] = None
    ) -> List[sqlite3.Row]:
        """Fetch only the given keys (config_loader.apply_config_kv_overrides
        filters to a known-key allowlist)."""
        if not keys:
            return []
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            return conn.execute(
                "SELECT key, value FROM config_kv WHERE key IN ({})".format(
                    ",".join("?" for _ in keys)
                ),
                keys,
            ).fetchall()
        finally:
            if own_conn:
                conn.close()

    def get(
        self, key: str, default: Any = None, conn: Optional[sqlite3.Connection] = None
    ) -> Any:
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            row = conn.execute(
                "SELECT value FROM config_kv WHERE key=?", (key,)
            ).fetchone()
            if row:
                return json.loads(row["value"])
            return default
        finally:
            if own_conn:
                conn.close()

    def set(self, key: str, value: Any, conn: Optional[sqlite3.Connection] = None) -> None:
        own_conn = conn is None
        if own_conn:
            conn = get_connection()
        try:
            now = _utc_now()
            conn.execute(
                "INSERT OR REPLACE INTO config_kv (key, value, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), now),
            )
            if own_conn:
                conn.commit()
        finally:
            if own_conn:
                conn.close()


# ------------------------------------------------------------------
# Module-level default instances.
#
# Repos are stateless wrappers around infra.db.get_connection(), which
# reads infra.db.DB_PATH at call time -- so these singletons transparently
# follow infra.db.set_db_path() the same way direct get_connection() calls
# did before this phase (tests rely on this). Constructor injection lands
# in Phase 2; this phase accepts module-level defaults where threading a
# repo through every constructor would be pure churn (risk, decision_log).
# ------------------------------------------------------------------

db = Db()
positions_repo = PositionsRepo()
orders_repo = OrdersRepo()
breaker_repo = BreakerRepo()
decisions_repo = DecisionsRepo()
control_repo = ControlRepo()
engine_state_repo = EngineStateRepo()
config_kv_repo = ConfigKVRepo()
