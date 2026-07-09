"""engine/registry.py

MarketRegistry: owns the former `token_map` shared dict (Phase 3 of the
architecture refactor, temp/implementation-plan.md -- Single-owner state).

Before this module, {slug: MarketInfo} was a plain dict built in
engine/startup.py, stored on the Engine dataclass, and mutated in place
by engine/pollers.py (discovery) and read by engine/strategy_loop.py and
the ws market_resolved callback (engine/build.py). No single object owned
writes; every caller reached into the dict directly.

INVARIANT: no `await` inside any MarketRegistry method. Callers that
resolve new markets over the network (pollers' discovery loops,
startup's resolve_markets) do all I/O first, then call add()/get()
synchronously to mutate the registry. This keeps the registry safe to
call from sync callback bodies (e.g. the ws market_resolved callback)
without making it a coroutine, and keeps mutation atomic from the
perspective of the single-threaded asyncio event loop.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class MarketRegistry:
    def __init__(self) -> None:
        self._by_slug: Dict[str, object] = {}  # slug -> MarketInfo

    def get(self, slug: str):
        """Return the MarketInfo for slug, or None if not resolved."""
        return self._by_slug.get(slug)

    def add(self, slug: str, market_info) -> None:
        """Register or replace the MarketInfo for slug."""
        self._by_slug[slug] = market_info

    def mark_resolved(self, token_id: str) -> Optional[str]:
        """Mark the MarketInfo whose token_id matches as closed/inactive.

        Returns the matching slug (truthy) or None if no match was found
        -- callers that only need a found/not-found signal can treat the
        return value as a bool. Linear scan -- registry size is bounded by
        tracked slugs (hundreds, not thousands of live token ids at once),
        matching the dict-scan this replaces.
        """
        for slug, mi in self._by_slug.items():
            if mi.token_id == token_id:
                mi.active = False
                mi.closed = True
                return slug
        return None

    def active_items(self) -> List[Tuple[str, object]]:
        """Snapshot copy of (slug, MarketInfo) pairs. A list() copy, not a
        live view, so callers iterating it are safe from concurrent
        mutation (a discovery-loop add() while strategy_loop iterates)."""
        return list(self._by_slug.items())

    def slugs(self) -> List[str]:
        """Snapshot copy of registered slugs."""
        return list(self._by_slug.keys())

    def __contains__(self, slug: str) -> bool:
        return slug in self._by_slug

    def __len__(self) -> int:
        return len(self._by_slug)
