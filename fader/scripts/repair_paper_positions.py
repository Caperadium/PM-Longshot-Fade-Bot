"""scripts/repair_paper_positions.py

One-off DB repair for the P3 paper-mode position drop.

Before the D6 fix in execution.order_manager._insert_position, EVERY
paper limit fill computed position_id = f"{user}::0" (market_info was
always None on that path), so `INSERT OR IGNORE` kept only the first
fill and silently dropped the rest. Observed in the live paper DB: 56
FILLED orders but 20 position rows, one collided "...::0" row, and
_has_open_position() returning False for the dropped positions caused
repeated re-entry into the same market.

This script:
  1. Renames the collided position_id `{user}::0` -> `{user}:{token_id}:0`
     (matches the new D6 scheme used going forward).
  2. Backfills a position row for every FILLED BUY order that has no
     matching position row (joined on entry_order_id).

CAVEAT: multiple FILLED orders on the same token (the P3 duplicate
re-entries caused by the dropped position rows) collapse to ONE position
row via `INSERT OR IGNORE` on the D6 position_id -- acceptable: those
duplicate orders were phantom double-entries the 11-filter stack would
have blocked had the positions table stayed intact; keeping one position
per market matches intended behaviour. The new Gamma-based resolution
poller (engine.reconciler._reconcile_paper_resolutions) then closes
resolved rows with real PnL on the next engine run.

DO NOT run this while the engine or dashboard is live -- stop both
processes first, they share the same SQLite file.

Reads POLYMARKET_USER_ADDRESS from fader/.env, same mechanism as
engine/main.py (python-dotenv). Takes no arguments.

Run:
    python fader/scripts/repair_paper_positions.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_FADER_ROOT = Path(__file__).resolve().parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))

from dotenv import load_dotenv


def _counts(conn) -> tuple[int, int]:
    total = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    open_ = conn.execute(
        "SELECT COUNT(*) FROM positions WHERE status='OPEN'"
    ).fetchone()[0]
    return total, open_


def main() -> None:
    load_dotenv(_FADER_ROOT / ".env")  # load .env from fader/, same as engine/main.py
    user = os.getenv("POLYMARKET_USER_ADDRESS", "")
    if not user:
        # Paper mode runs without credentials; the engine then builds
        # position_ids with an empty user prefix (":{cid}:{oidx}"), so the
        # repair must use the same empty prefix to match existing rows.
        print("POLYMARKET_USER_ADDRESS not set -- paper DB, using empty user prefix")

    from infra.db import get_connection

    conn = get_connection()
    try:
        before_total, before_open = _counts(conn)
        print(f"Before: {before_total} position row(s) total, {before_open} OPEN")

        # 1. Fix the collided row: {user}::0 -> {user}:{token_id}:0
        cur = conn.execute(
            """
            UPDATE positions SET position_id = :user || ':' || token_id || ':0'
            WHERE position_id = :user || '::0'
            """,
            {"user": user},
        )
        renamed = cur.rowcount
        print(f"Renamed {renamed} collided position row(s)")

        # 2. Insert missing positions for FILLED BUY orders with no position row.
        #    Join on entry_order_id (engine-inserted rows store it); orders
        #    imported without a matching position get a fresh D6-style id.
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO positions
              (position_id, slug, condition_id, token_id, outcome, entry_price, size,
               notional, status, opened_at, source, entry_order_id)
            SELECT :user || ':' || o.token_id || ':0', o.slug, '', o.token_id, 'No',
                   o.price, o.size, o.price * o.size, 'OPEN', o.created_at,
                   'ENGINE_FILL', o.order_id
            FROM orders o
            LEFT JOIN positions p ON p.entry_order_id = o.order_id
            LEFT JOIN positions pt ON pt.token_id = o.token_id
            WHERE o.status = 'FILLED' AND o.side = 'BUY'
              AND p.position_id IS NULL AND pt.position_id IS NULL
            """,
            {"user": user},
        )
        inserted = cur.rowcount
        print(f"Inserted {inserted} missing position row(s)")

        conn.commit()

        after_total, after_open = _counts(conn)
        print(f"After: {after_total} position row(s) total, {after_open} OPEN")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
