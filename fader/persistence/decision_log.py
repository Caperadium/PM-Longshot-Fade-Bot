"""persistence/decision_log.py

Mandatory structured decision log.
Every filter evaluation (pass OR reject) writes one decisions row.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from infra.db import get_connection

logger = logging.getLogger(__name__)


def log_decision(
    slug: str,
    token_id: Optional[str],
    decision: str,  # "ENTERED" | "REJECTED"
    reason: str,
    filters: Dict[str, Any],
    order_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
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
        conn.commit()
    except Exception as e:
        logger.error(f"log_decision failed: {e}")
    finally:
        conn.close()


def log_entered(
    slug: str,
    token_id: str,
    filters: Dict[str, Any],
    order_id: str,
    idempotency_key: str,
) -> None:
    log_decision(slug, token_id, "ENTERED", "all_filters_passed",
                 filters, order_id, idempotency_key)


def log_rejected(
    slug: str,
    token_id: Optional[str],
    reason: str,
    filters: Dict[str, Any],
) -> None:
    log_decision(slug, token_id, "REJECTED", reason, filters)
