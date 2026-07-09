"""tests/helpers.py

Shared test helpers for Phase 2 of the architecture refactor
(temp/implementation-plan.md). Provider.place_order now returns a typed
OrderResult instead of a dict, so tests that mock async_place_order need a
convenient constructor instead of hand-rolling dicts with "success"/
"simulated" keys everywhere.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from execution.provider import OrderResult


def make_order_result(
    success: bool = True,
    status: str = "FILLED",
    order_id: Optional[str] = "SIM-1",
    filled_price: Optional[float] = None,
    error: Optional[str] = None,
    raw: Optional[Dict[str, Any]] = None,
) -> OrderResult:
    """Build an OrderResult for mocked async_place_order return values.

    Defaults to a successful paper-style FILLED result; override kwargs
    for PENDING/REJECTED/DUPLICATE/UNKNOWN cases.
    """
    return OrderResult(
        success=success,
        status=status,
        order_id=order_id,
        filled_price=filled_price,
        error=error,
        raw=raw or {},
    )
