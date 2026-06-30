"""execution/idempotency.py

Local idempotency key store backed by orders.idempotency_key UNIQUE.

Key: deterministic hash(slug, token_id, side, price_cents, size_cents, bucket).
Treats INVALID_ORDER_DUPLICATED as success (order already live).
On restart: reconciler reads open orders from API as ground truth.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from infra.db import get_connection

logger = logging.getLogger(__name__)


def make_key(
    slug: str,
    token_id: str,
    side: str,
    price: float,
    size: float,
    bucket: str = "entry",
) -> str:
    """Deterministic idempotency key."""
    price_cents = round(price * 10000)
    size_cents = round(size * 100)
    payload = json.dumps(
        [slug, token_id, side.upper(), price_cents, size_cents, bucket],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def is_already_submitted(key: str) -> bool:
    """True if this key exists in orders table with a non-failed status."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT status FROM orders WHERE idempotency_key = ?", (key,)
        ).fetchone()
        if row is None:
            return False
        # If previously failed/cancelled, allow retry
        return row["status"] not in ("FAILED", "CANCELLED")
    finally:
        conn.close()


def is_duplicate_error(error_msg: str) -> bool:
    """Return True if API error indicates server-side duplicate (treat as success)."""
    return "INVALID_ORDER_DUPLICATED" in str(error_msg).upper()
