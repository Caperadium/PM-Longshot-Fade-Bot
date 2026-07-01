"""infra/db.py

SQLite persistence for the fader bot.

Design:
- Connection-per-operation (WAL; safe for asyncio + Streamlit concurrency).
- All new tables from Section 6 of the implementation plan.
- Reuses pm_trades, pm_closed_positions, pm_sync_metadata from source repo pattern.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "fader.db"


def set_db_path(path: Path) -> None:
    global DB_PATH
    DB_PATH = path


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=8000;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all tables. Idempotent."""
    conn = get_connection()
    try:
        c = conn.cursor()

        # ------------------------------------------------------------------
        # positions: open and closed contract positions
        # source: ENGINE_FILL (bot opened) | RECONCILE_IMPORT (found on API)
        # ------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                slug TEXT NOT NULL,
                condition_id TEXT,
                token_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                entry_price REAL NOT NULL,
                size REAL NOT NULL,
                notional REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'OPEN'
                    CHECK (status IN ('OPEN','CLOSED')),
                opened_at TEXT NOT NULL,
                resolved_at TEXT,
                realized_pnl REAL,
                source TEXT NOT NULL DEFAULT 'ENGINE_FILL'
                    CHECK (source IN ('ENGINE_FILL','RECONCILE_IMPORT')),
                entry_order_id TEXT,
                entry_decision_id TEXT
            )
        """)

        # ------------------------------------------------------------------
        # orders: every order placed or attempted
        # ------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                slug TEXT NOT NULL,
                token_id TEXT NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
                type TEXT NOT NULL CHECK (type IN ('LIMIT','MARKET')),
                price REAL,
                size REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING'
                    CHECK (status IN ('PENDING','FILLED','CANCELLED','FAILED','UNKNOWN')),
                created_at TEXT NOT NULL,
                ttl_expires_at TEXT,
                last_requote_at TEXT,
                cancel_reason TEXT,
                raw_json TEXT
            )
        """)

        # ------------------------------------------------------------------
        # decisions: mandatory structured decision log
        # ------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                slug TEXT NOT NULL,
                token_id TEXT,
                decision TEXT NOT NULL CHECK (decision IN ('ENTERED','REJECTED')),
                reason TEXT,
                filters_json TEXT,
                order_id TEXT,
                idempotency_key TEXT
            )
        """)

        # ------------------------------------------------------------------
        # config_kv: dashboard-written live param overrides
        # ------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS config_kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                updated_by TEXT NOT NULL DEFAULT 'dashboard'
            )
        """)

        # ------------------------------------------------------------------
        # control_commands: dashboard -> engine IPC
        # ------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS control_commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                command TEXT NOT NULL,
                args_json TEXT,
                status TEXT NOT NULL DEFAULT 'PENDING'
                    CHECK (status IN ('PENDING','DONE','ERROR')),
                result_json TEXT
            )
        """)

        # ------------------------------------------------------------------
        # engine_state: key-value snapshot published by state_publisher
        # ------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS engine_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # ------------------------------------------------------------------
        # circuit_breaker: daily loss tracking
        # ------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS circuit_breaker (
                day TEXT PRIMARY KEY,
                realized_pnl REAL NOT NULL DEFAULT 0.0,
                tripped INTEGER NOT NULL DEFAULT 0,
                tripped_at TEXT,
                reset_at TEXT
            )
        """)

        # ------------------------------------------------------------------
        # pm_trades: raw trades from CLOB (reconciliation)
        # ------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS pm_trades (
                trade_id TEXT PRIMARY KEY,
                user_address TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                market_slug TEXT,
                market_id TEXT,
                condition_id TEXT,
                token_id TEXT,
                side TEXT,
                price REAL,
                size REAL,
                notional REAL,
                fee REAL,
                maker_taker TEXT,
                order_id TEXT,
                raw_json TEXT
            )
        """)

        # ------------------------------------------------------------------
        # pm_closed_positions: Data-API closed positions (realized PnL source)
        # ------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS pm_closed_positions (
                position_id TEXT PRIMARY KEY,
                user_address TEXT NOT NULL,
                condition_id TEXT,
                market_slug TEXT,
                title TEXT,
                outcome TEXT,
                outcome_index INTEGER,
                avg_price REAL,
                size REAL,
                total_bought REAL,
                realized_pnl REAL,
                cur_price REAL,
                resolved_at TEXT,
                end_date TEXT,
                raw_json TEXT
            )
        """)

        # ------------------------------------------------------------------
        # pm_sync_metadata: idempotent incremental sync cursors
        # ------------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS pm_sync_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT NOT NULL
            )
        """)

        # Indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_positions_slug ON positions(slug)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_positions_token ON positions(token_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_orders_slug ON orders(slug)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_orders_idem ON orders(idempotency_key)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_decisions_slug ON decisions(slug)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_control_status ON control_commands(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_pm_trades_ts ON pm_trades(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_pm_closed_user ON pm_closed_positions(user_address)")

        conn.commit()
        logger.info(f"DB initialized at {DB_PATH}")
    except Exception as e:
        logger.error(f"DB init failed: {e}")
        raise
    finally:
        conn.close()


def execute_query(query: str, params: tuple = (), fetch: bool = False) -> Optional[list]:
    conn = get_connection()
    try:
        c = conn.cursor()
        c.execute(query, params)
        if fetch:
            return c.fetchall()
        conn.commit()
        return None
    finally:
        conn.close()


def execute_many(query: str, params_list: list) -> None:
    conn = get_connection()
    try:
        c = conn.cursor()
        c.executemany(query, params_list)
        conn.commit()
    finally:
        conn.close()


def execute_write(query: str, params: tuple = (), retries: int = 3, base_sleep: float = 0.2) -> int:
    """Single-statement write with retry ONLY on lock/busy. Returns rowcount.

    Engine + dashboard both write WAL concurrently; a write that's still
    waiting past busy_timeout raises sqlite3.OperationalError("database is
    locked"). Retries with backoff only for locked/busy — any other
    OperationalError (corruption, constraint, etc.) propagates immediately,
    as does any non-OperationalError exception.
    """
    last: Optional[Exception] = None
    for attempt in range(retries):
        conn = get_connection()
        try:
            cur = conn.execute(query, params)
            conn.commit()
            return cur.rowcount
        except sqlite3.OperationalError as e:
            last = e
            msg = str(e).lower()
            if "locked" not in msg and "busy" not in msg:
                raise  # corruption/constraint/etc -> fail loud, do NOT retry
            time.sleep(base_sleep * (2 ** attempt))
        finally:
            conn.close()
    logger.warning(f"execute_write exhausted retries: {last}")
    return 0
