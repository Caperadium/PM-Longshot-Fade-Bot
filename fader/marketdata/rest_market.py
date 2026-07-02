"""marketdata/rest_market.py

REST-based market data operations:
- /book snapshots (ws resync)
- Gamma /markets?slug= for token resolution
- Ladder-rung discovery (new binary markets)
- Volumes from Gamma metadata
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import requests

from execution.provider import MarketInfo

logger = logging.getLogger(__name__)

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
CLOB_API_BASE = "https://clob.polymarket.com"
DATA_API_BASE = "https://data-api.polymarket.com"
DEFAULT_TIMEOUT = 15
MAX_RETRIES = 3
CALL_DELAY_S = 0.05

# Progress callback: receives a human-readable status line.
ProgressFn = Callable[[str], None]


def _get(url: str, params: Optional[Dict] = None, timeout: int = DEFAULT_TIMEOUT) -> Any:
    last_err = None
    for retry in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = 2 ** (retry + 1)
                logger.warning(f"429 from {url}; wait {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            if retry < MAX_RETRIES - 1:
                time.sleep(2 ** retry)
    raise last_err or RuntimeError(f"GET {url} failed")


def fetch_order_book_snapshot(token_id: str) -> Optional[Dict[str, Any]]:
    """REST /book snapshot for ws resync."""
    try:
        data = _get(f"{CLOB_API_BASE}/book", params={"token_id": token_id})
        return {
            "bids": data.get("bids", []),
            "asks": data.get("asks", []),
        }
    except Exception as e:
        logger.warning(f"book snapshot for {token_id[:16]}: {e}")
        return None


def fetch_market_metadata(slug: str) -> Optional[Dict[str, Any]]:
    """
    Gamma /markets?slug= for a single slug.
    Returns the first matching market dict, or None.

    Gamma excludes closed markets from a plain slug query, so a miss is
    retried with closed=true — required by paper resolution polling, which
    exists precisely to find positions whose market has closed.
    """
    try:
        data = _get(GAMMA_MARKETS_URL, params={"slug": slug})
        markets = data if isinstance(data, list) else data.get("markets", [])
        if not markets:
            time.sleep(CALL_DELAY_S)
            data = _get(GAMMA_MARKETS_URL, params={"slug": slug, "closed": "true"})
            markets = data if isinstance(data, list) else data.get("markets", [])
        return markets[0] if markets else None
    except Exception as e:
        logger.warning(f"fetch_market_metadata({slug}): {e}")
        return None


def fetch_volumes(slug: str) -> Dict[str, float]:
    """
    Return volume24h and volumeNum from Gamma market metadata.
    """
    meta = fetch_market_metadata(slug)
    if not meta:
        return {"volume_24h": 0.0, "volume_total": 0.0}
    return {
        "volume_24h": float(meta.get("volume24hr", meta.get("volume24h", 0)) or 0),
        "volume_total": float(meta.get("volumeNum", meta.get("volume", 0)) or 0),
    }


def get_market_end_date(slug: str) -> Optional[datetime]:
    """Return the market end date as UTC datetime, or None."""
    meta = fetch_market_metadata(slug)
    if not meta:
        return None
    end_str = meta.get("endDateIso") or meta.get("endDate") or ""
    if not end_str:
        return None
    try:
        dt = datetime.fromisoformat(end_str.rstrip("Z"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def dte(slug: str) -> Optional[float]:
    """Days to expiry for a slug. Returns None if unknown."""
    end = get_market_end_date(slug)
    if end is None:
        return None
    now = datetime.now(timezone.utc)
    delta = end - now
    return max(0.0, delta.total_seconds() / 86400)


def discover_new_rungs(
    tracked_slugs: List[str],
    market_kind_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Discovery poller: find new binary rungs for ladder markets.
    Queries Gamma for markets related to known ladder slugs' event slugs.
    Returns list of {"slug": ..., "token_id": ..., "outcome": "No"} dicts.
    """
    new_rungs: List[Dict[str, Any]] = []
    seen = set(tracked_slugs)

    for slug in tracked_slugs:
        if market_kind_map.get(slug) != "ladder":
            continue
        # For ladder markets, find the event and look for sibling rungs
        try:
            # Try Gamma events endpoint to find ladder siblings
            meta = fetch_market_metadata(slug)
            if not meta:
                continue
            event_id = meta.get("eventId")
            if not event_id:
                continue
            # Fetch all markets for this event
            siblings = _get(GAMMA_MARKETS_URL, params={"event_id": event_id}) or []
            for mkt in siblings:
                s = mkt.get("slug", "")
                if s and s not in seen:
                    # Check it has a NO token
                    raw_tokens = mkt.get("clobTokenIds", "[]")
                    raw_outcomes = mkt.get("outcomes", "[]")
                    tids = json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
                    outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
                    for i, o in enumerate(outcomes):
                        if o.strip().lower() == "no" and i < len(tids):
                            new_rungs.append({
                                "slug": s,
                                "condition_id": mkt.get("conditionId", ""),
                                "token_id": tids[i],
                                "outcome": o,
                                "outcome_index": i,
                                "end_date_iso": mkt.get("endDateIso", ""),
                                "active": bool(mkt.get("active", True)),
                            })
                            seen.add(s)
                            break
        except Exception as e:
            logger.warning(f"discover_new_rungs for {slug}: {e}")

    return new_rungs


def validate_slug_exists(slug: str) -> bool:
    """Quick check that a slug resolves to a market on Gamma."""
    return fetch_market_metadata(slug) is not None


# ------------------------------------------------------------------
# Series market discovery
# ------------------------------------------------------------------

def parse_series_date(val: str) -> date:
    """Parse YYYY-MM-DD series_from_date, defaulting to 2024-01-01."""
    try:
        return datetime.strptime(val, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return date(2024, 1, 1)


def _derive_series_filter(base_slug: str) -> str:
    """Fallback: strip trailing '-on' from base slug to get filter keyword.

    Only used when slugs.csv ``series_filter`` column is empty.
    ``bitcoin-above-on`` → ``bitcoin-above``
    ``highest-temperature-in-seoul-on`` → ``highest-temperature-in-seoul``
    """
    if base_slug.endswith("-on"):
        return base_slug[:-3]
    return base_slug


def _series_date_slug_groups(
    base_slug: str,
    from_date: date,
    to_date: date,
) -> Generator[List[str], None, None]:
    """Yield [legacy_slug, year_slug] pairs for every date in range.

    Polymarket uses two event-slug formats split at a mid-2026 transition:
      legacy:  {base_slug}-{month}-{day}
      current: {base_slug}-{month}-{day}-{year}
    Both are generated per date; Gamma deduplicates if they map to the same
    event.
    """
    start = from_date
    end = to_date
    current = start
    while current <= end:
        month = current.strftime("%B").lower()
        day = current.day
        year = current.year
        yield [
            f"{base_slug}-{month}-{day}",
            f"{base_slug}-{month}-{day}-{year}",
        ]
        current += timedelta(days=1)


def _resolution_from_outcome_prices(
    outcomes: Any, raw_prices: Any, closed: Any
) -> str:
    """Derive the winning outcome ("YES"/"NO") from a closed market.

    Gamma reports ``outcomePrices`` like ["1","0"] aligned to ``outcomes``
    (["Yes","No"]); the "1" marks the resolved winner. Returns "" if the
    market is not closed or the data is unusable (engine then leaves the
    position open).
    """
    if not closed:
        return ""
    try:
        prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
        if not prices or len(prices) != len(outcomes):
            return ""
        win_idx = max(range(len(prices)), key=lambda i: float(prices[i]))
        return outcomes[win_idx].strip().upper()
    except Exception:
        return ""


def discover_series_markets(
    base_slug: str,
    series_filter: str,
    from_date: date,
    to_date: Optional[date] = None,
    progress: Optional[ProgressFn] = None,
) -> List[Dict[str, str]]:
    """Discover all individual binary markets for a daily series via Gamma /events.

    Iterates every date in [from_date, to_date], queries Gamma /events with
    both the legacy and year-suffixed event-slug formats, and collects every
    individual binary market's NO token whose slug contains ``series_filter``.

    Args:
        base_slug: Virtual series slug (e.g. ``"highest-temperature-in-seoul-on"``).
        series_filter: Substring that identifies child market slugs
                       (e.g. ``"highest-temperature-in-seoul"``).
        from_date: First date to scan (from slugs.csv).
        to_date: Last date to scan (defaults to yesterday).

    Returns list of ``{"slug", "token_id", "end_date", "resolution",
    "condition_id", "active", "question"}`` dicts.
    """
    seen_tokens: set = set()
    results: List[Dict[str, str]] = []

    start = from_date
    end = to_date or (datetime.now(timezone.utc).date() - timedelta(days=1))
    n_dates = (end - start).days + 1
    if progress:
        progress(
            f"Discovering series '{base_slug}': scanning {n_dates} dates "
            f"({start} -> {end}) via Gamma /events..."
        )

    total = 0
    for slug_group in _series_date_slug_groups(base_slug, start, end):
        total += 1
        if progress and (total % 30 == 0 or total == n_dates):
            progress(
                f"  discovery {total}/{n_dates} dates scanned, "
                f"{len(results)} markets found so far"
            )
        try:
            resp = requests.get(
                GAMMA_EVENTS_URL,
                params=[("slug", s) for s in slug_group],
                timeout=15,
            )
            resp.raise_for_status()
            events = resp.json()
            if not isinstance(events, list):
                events = []
        except Exception as e:
            logger.debug(f"Gamma /events {slug_group[0]}: {e}")
            time.sleep(CALL_DELAY_S)
            continue

        for evt in events:
            for mkt in evt.get("markets", []):
                mkt_slug = mkt.get("slug", "")
                if not mkt_slug or series_filter not in mkt_slug.lower():
                    continue
                raw_tokens = mkt.get("clobTokenIds", "[]")
                raw_outcomes = mkt.get("outcomes", "[]")
                token_ids = (
                    json.loads(raw_tokens)
                    if isinstance(raw_tokens, str)
                    else raw_tokens
                )
                outcomes = (
                    json.loads(raw_outcomes)
                    if isinstance(raw_outcomes, str)
                    else raw_outcomes
                )
                for i, o in enumerate(outcomes):
                    if o.strip().lower() == "no" and i < len(token_ids):
                        tid = token_ids[i]
                        if tid not in seen_tokens:
                            seen_tokens.add(tid)
                            results.append({
                                "slug": mkt_slug,
                                "token_id": tid,
                                "end_date": mkt.get("endDateIso") or mkt.get("endDate") or "",
                                "resolution": _resolution_from_outcome_prices(
                                    outcomes,
                                    mkt.get("outcomePrices"),
                                    mkt.get("closed"),
                                ),
                                "condition_id": mkt.get("conditionId", ""),
                                "active": bool(mkt.get("active", True)),
                                "question": mkt.get("question", ""),
                            })
                        break

        time.sleep(CALL_DELAY_S)

    if progress:
        progress(
            f"Discovery complete: scanned {total} dates, found {len(results)} markets."
        )
    return results
