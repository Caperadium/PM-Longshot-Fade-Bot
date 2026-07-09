"""persistence/decision_log.py

Mandatory structured decision log.
Every filter evaluation (pass OR reject) writes one decisions row.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from persistence.repos import decisions_repo

logger = logging.getLogger(__name__)


def log_decision(
    slug: str,
    token_id: Optional[str],
    decision: str,  # "ENTERED" | "REJECTED"
    reason: str,
    filters: Dict[str, Any],
    order_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> bool:
    """Insert a decisions row. Returns True on success, False on failure
    (previously swallowed the exception silently and returned None;
    callers that care now get an honest signal -- see order_manager's
    warning-on-False, Phase 1 of the architecture refactor)."""
    return decisions_repo.append(
        slug, token_id, decision, reason, filters, order_id, idempotency_key,
    )


def log_entered(
    slug: str,
    token_id: str,
    filters: Dict[str, Any],
    order_id: str,
    idempotency_key: str,
) -> bool:
    return log_decision(slug, token_id, "ENTERED", "all_filters_passed",
                         filters, order_id, idempotency_key)


def log_rejected(
    slug: str,
    token_id: Optional[str],
    reason: str,
    filters: Dict[str, Any],
) -> bool:
    return log_decision(slug, token_id, "REJECTED", reason, filters)
