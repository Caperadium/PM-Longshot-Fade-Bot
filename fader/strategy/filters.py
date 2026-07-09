"""strategy/filters.py

Pure, shared filter core for the anti-longshot entry stack (filters 1-8).
No I/O, no logging, no DB access -- this module only evaluates numbers
against thresholds and returns a result. Used by BOTH the live engine
(engine/strategy_loop.py) and the backtest engine (backtest/engine.py) so
there is exactly one implementation of "does this contract qualify."

Filters 9-11 (per-market cap, total-deployed cap, circuit breaker) are
NOT here -- they are stateful (bankroll, cumulative deployed notional,
daily PnL) and stay in engine/risk.py + engine/strategy_loop.py.

Two entry points, matching how the live caller must fetch data lazily:
  - evaluate_pregate(best_ask, dte, params): filters 1-2 only (band, DTE).
    Cheap -- no volumes/depth/staleness inputs required. Callers use this
    to decide whether the expensive REST-backed EntrySnapshot fields
    (volumes, depth) are even worth fetching.
  - evaluate_entry(snapshot, params): the full filters 1-8 stack, in
    current order, given a fully-populated EntrySnapshot.

Per-field None semantics (normative -- see temp/implementation-plan.md
Phase 4):
  - best_ask=None            -> reject "no_book" (fail-closed, both engines)
  - dte=None                 -> reject "dte_out_of_range" when
                                 params.missing_dte == "reject" (live);
                                 filter SKIPPED (pass-through, fail-open)
                                 when params.missing_dte == "skip" (backtest,
                                 matches today's backtest fail-open behavior)
  - volume_24h=None           -> filter skipped, recorded in `skipped`
  - volume_total=None         -> filter skipped, recorded in `skipped`
  - ask_depth_usd=None        -> filter skipped, recorded in `skipped`
  - is_stale=None             -> filter skipped, recorded in `skipped`

There is NO generic "None means pass" rule -- each field's policy is
independently spelled out above. Applying a blanket rule would silently
flip live's fail-closed DTE handling to fail-open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

MissingDtePolicy = Literal["reject", "skip"]


@dataclass(frozen=True)
class FilterParams:
    band_low: float
    band_high: float
    min_dte: float
    max_dte: float
    min_time_in_band_s: float
    min_24h_volume: float
    min_total_volume: float
    min_book_depth: float
    check_staleness: bool = True
    check_time_in_band: bool = True
    missing_dte: MissingDtePolicy = "reject"


@dataclass(frozen=True)
class EntrySnapshot:
    best_ask: Optional[float]
    dte: Optional[float]
    seconds_in_band: Optional[float]
    volume_24h: Optional[float]
    volume_total: Optional[float]
    ask_depth_usd: Optional[float]
    is_stale: Optional[bool]
    has_open_position: bool


@dataclass(frozen=True)
class FilterResult:
    passed: bool
    reason: str
    detail: Dict
    skipped: Tuple[str, ...] = ()


def _pregate_dte_reject(dte: Optional[float], p: FilterParams) -> bool:
    """True if the DTE check should REJECT given current dte/policy.

    dte=None: reject only under the "reject" policy (live fail-closed).
    Under "skip" (backtest), a None dte never rejects -- the filter is
    treated as not evaluated and the row proceeds (fail-open), matching
    backtest/engine.py's `if dte is not None: check range` today.
    """
    if dte is None:
        return p.missing_dte == "reject"
    return not (p.min_dte <= dte <= p.max_dte)


def evaluate_pregate(
    best_ask: Optional[float],
    dte: Optional[float],
    p: FilterParams,
) -> FilterResult:
    """Filters 1-2 only: NO ask in band, then DTE in range.

    Cheap -- does not require volumes/depth/staleness. Callers fetch those
    only after this passes (avoids an API storm on every tick).
    """
    if best_ask is None:
        return FilterResult(passed=False, reason="no_book", detail={})

    if not (p.band_low <= best_ask <= p.band_high):
        return FilterResult(
            passed=False,
            reason="ask_out_of_band",
            detail={
                "no_ask": best_ask,
                "band_low": p.band_low,
                "band_high": p.band_high,
            },
        )

    if _pregate_dte_reject(dte, p):
        return FilterResult(
            passed=False,
            reason="dte_out_of_range",
            detail={"dte": dte, "min_dte": p.min_dte, "max_dte": p.max_dte},
        )

    skipped: Tuple[str, ...] = ("dte",) if (dte is None and p.missing_dte == "skip") else ()
    return FilterResult(passed=True, reason="all_filters_passed", detail={}, skipped=skipped)


def evaluate_entry(s: EntrySnapshot, p: FilterParams) -> FilterResult:
    """Full filters 1-8 stack, in current order, against a fully built
    EntrySnapshot. Filters 9-11 (risk caps, breaker) are evaluated
    separately by the caller (stateful, live-only)."""
    skipped: list = []

    # -- Filters 1-2: ask in band, DTE --
    pregate = evaluate_pregate(s.best_ask, s.dte, p)
    if not pregate.passed:
        return pregate
    skipped.extend(pregate.skipped)

    # -- Filter 3: continuously in band --
    if p.check_time_in_band:
        seconds_in_band = s.seconds_in_band
        if seconds_in_band is None or seconds_in_band < p.min_time_in_band_s:
            return FilterResult(
                passed=False,
                reason="not_in_band_long_enough",
                detail={
                    "time_in_band_s": seconds_in_band,
                    "min_time_in_band_s": p.min_time_in_band_s,
                },
                skipped=tuple(skipped),
            )

    # -- Filter 4: 24h volume --
    if s.volume_24h is None:
        skipped.append("min_24h_volume")
    elif s.volume_24h < p.min_24h_volume:
        return FilterResult(
            passed=False,
            reason="low_24h_volume",
            detail={
                "volume_24h": s.volume_24h,
                "volume_total": s.volume_total,
                "min_24h_volume": p.min_24h_volume,
            },
            skipped=tuple(skipped),
        )

    # -- Filter 5: cumulative volume --
    if s.volume_total is None:
        skipped.append("min_total_volume")
    elif s.volume_total < p.min_total_volume:
        return FilterResult(
            passed=False,
            reason="low_total_volume",
            detail={
                "volume_24h": s.volume_24h,
                "volume_total": s.volume_total,
                "min_total_volume": p.min_total_volume,
            },
            skipped=tuple(skipped),
        )

    # -- Filter 6: book depth at NO touch --
    if s.ask_depth_usd is None:
        skipped.append("min_book_depth")
    elif p.min_book_depth > 0 and s.ask_depth_usd < p.min_book_depth:
        return FilterResult(
            passed=False,
            reason="insufficient_depth",
            detail={
                "ask_depth_usd": s.ask_depth_usd,
                "min_book_depth": p.min_book_depth,
            },
            skipped=tuple(skipped),
        )

    # -- Filter 7: staleness / gap --
    if p.check_staleness:
        if s.is_stale is None:
            skipped.append("stale_data")
        elif s.is_stale:
            return FilterResult(
                passed=False,
                reason="stale_data",
                detail={"stale": True},
                skipped=tuple(skipped),
            )

    # -- Filter 8: no existing OPEN position --
    if s.has_open_position:
        return FilterResult(
            passed=False,
            reason="position_already_open",
            detail={},
            skipped=tuple(skipped),
        )

    return FilterResult(passed=True, reason="all_filters_passed", detail={}, skipped=tuple(skipped))
