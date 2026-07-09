"""engine/startup.py

Startup-sequence helpers extracted from engine/main.py (Phase 2 of the
architecture refactor, temp/implementation-plan.md) to keep main.py under
the target line budget. Pure startup logic -- token resolution + volume
cache pre-warm -- no object construction (that's engine/build.py) and no
task/signal lifecycle (that stays in main.py).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from engine.registry import MarketRegistry
from execution.provider import MarketInfo
from marketdata.rest_market import (
    discover_series_markets,
    _derive_series_filter,
    parse_series_date,
    fetch_volumes,
)

logger = logging.getLogger(__name__)


async def resolve_markets(cfg, provider, loop) -> Tuple[MarketRegistry, List[str]]:
    """Resolve token_ids for all enabled slugs (binary/ladder + series
    expansion). Returns (registry, series_slugs)."""
    registry = MarketRegistry()
    series_slugs: List[str] = []
    today = datetime.now(timezone.utc).date()

    for slug_row in cfg.enabled_slugs():
        if slug_row.market_kind in ("series", "btc_daily"):
            # --- Series expansion path ---
            series_filter = slug_row.series_filter or _derive_series_filter(slug_row.slug)
            from_date = parse_series_date(slug_row.series_from_date)
            start = max(from_date, today - timedelta(days=7))
            forward = min(cfg.strategy.max_dte + 3, 30)
            end = today + timedelta(days=forward)

            try:
                children = await loop.run_in_executor(
                    None,
                    lambda: discover_series_markets(
                        base_slug=slug_row.slug,
                        series_filter=series_filter,
                        from_date=start,
                        to_date=end,
                        progress=lambda msg: logger.info(msg),
                    ),
                )
            except Exception as e:
                logger.error(f"Could not discover series {slug_row.slug}: {e} — skipping")
                continue

            for child in children:
                # Dedup by token_id
                if any(
                    mi.token_id == child["token_id"]
                    for _, mi in registry.active_items()
                ):
                    continue
                cid = child.get("condition_id", "") or child["token_id"]
                registry.add(child["slug"], MarketInfo(
                    slug=child["slug"],
                    condition_id=cid,
                    token_id=child["token_id"],
                    outcome="No",
                    outcome_index=0,
                    question=child.get("question", ""),
                    end_date_iso=child.get("end_date", ""),
                    active=child.get("active", True),
                    closed=bool(child.get("resolution", "")),
                ))
                series_slugs.append(child["slug"])
            logger.info(
                f"Series {slug_row.slug}: expanded to {len(children)} markets "
                f"(filter='{series_filter}', {start} -> {end})"
            )
        else:
            # --- Existing binary/ladder path (unchanged) ---
            try:
                market_info = await loop.run_in_executor(
                    None, lambda s=slug_row.slug: provider.resolve_no_token(s)
                )
                registry.add(slug_row.slug, market_info)
                logger.info(
                    f"Resolved {slug_row.slug} -> NO token {market_info.token_id[:16]}..."
                )
            except Exception as e:
                logger.error(f"Could not resolve {slug_row.slug}: {e} — skipping")

    # Warn about resolved/closed slugs in config
    for slug, mi in registry.active_items():
        if mi.closed:
            logger.warning(f"Slug {slug!r} is CLOSED — consider removing from slugs.csv")
        elif not mi.active:
            logger.warning(f"Slug {slug!r} is INACTIVE — will be skipped at runtime")

    if len(registry) == 0:
        logger.warning("No slugs resolved. Engine will run but strategy loop idle.")

    return registry, series_slugs


async def prewarm_volume_cache(strategy_loop, series_slugs: List[str], loop) -> None:
    """Pre-warm volume cache for series slugs (avoids 500+ cold Gamma calls
    on first tick)."""
    if not series_slugs:
        return
    logger.info(f"Pre-warming volume cache for {len(series_slugs)} series markets...")
    batch_size = 10
    for i in range(0, len(series_slugs), batch_size):
        batch = series_slugs[i:i + batch_size]
        tasks = [
            loop.run_in_executor(None, lambda s=slug: fetch_volumes(s))
            for slug in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for slug, result in zip(batch, results):
            if isinstance(result, dict):
                strategy_loop._volume_cache[slug] = result
                strategy_loop._volume_cache_ts[slug] = time.monotonic()
            else:
                logger.warning(f"Volume pre-warm failed for {slug}: {result!r}")
        if i + batch_size < len(series_slugs):
            await asyncio.sleep(0.1)
    logger.info(f"Volume cache pre-warm complete ({len(series_slugs)} slugs)")
