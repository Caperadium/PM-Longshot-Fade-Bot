# Slug Fix Summary

## Problem

The fader bot's `slugs.csv` initially had a date-specific BTC market slug
(`bitcoin-above-on-june-27-2026`) copied from the V2 pricing program. This
only worked for a single day and made backtesting across historical dates
impossible.

## Root Cause

The V2 Polymarket BTC Pricing Program uses event-level slug *patterns*
(`bitcoin-above-on-{month}-{day}-{year}`) which it resolves dynamically via
the Gamma `/events` API. The fader bot needed the same dynamic discovery
capability rather than a hardcoded single-date slug.

## Fix

### 1. `fader/config/slugs.csv`

Replaced the date-specific slug with a series identifier:

| Before | After |
|--------|-------|
| `bitcoin-above-on-june-27-2026` | `bitcoin-above-on` |
| `market_kind=ladder` | `market_kind=btc_daily` |

The slug `bitcoin-above-on` acts as a series key. It never resolves directly
to a market; instead it triggers automatic discovery of all historical markets.

### 2. `fader/backtest/historical.py`

Added three new components:

**`_btc_date_slug_groups(from_date, to_date)`**
Generates `[legacy_slug, year_slug]` pairs for every date in the range.
Polymarket uses two event-slug formats:
- Legacy: `bitcoin-above-on-{month}-{day}` (pre-2026 dates)
- Current: `bitcoin-above-on-{month}-{day}-{year}` (2026+ dates)

Both are emitted per date so the Gamma query catches either format.

**`discover_btc_daily_markets(from_date, to_date)`**
Iterates every date from 2024-01-01 to yesterday, queries `Gamma /events`
with both slug formats, extracts every individual binary market slug and its
NO token ID from each event's markets array. Deduplicates by token ID.
Returns `[{slug, token_id, end_date}, ...]`.

**Updated `fetch_and_store(slugs, ..., slug_kinds)`**
Now accepts an optional `slug_kinds` dict (`{slug: market_kind}`). When a
slug has kind `btc_daily` (or equals `bitcoin-above-on` directly), it calls
`discover_btc_daily_markets()` to expand the series into all individual
market/token pairs before fetching CLOB price history. Non-btc_daily slugs
resolve as before via `Gamma /markets?slug=`.

### 3. `fader/dashboard/backtest_page.py`

Updated the "Fetch/Refresh Historical Prices" button handler to pass
`slug_kinds = {s.slug: s.market_kind for s in cfg.slugs}` into
`fetch_and_store`, so the expansion triggers automatically when the BTC
series slug is selected.

## How to Use

1. In the backtest dashboard, ensure `bitcoin-above-on` appears in the
   slug multiselect (it will after at least one fetch populates the store
   with individual market slugs).
2. Click **Fetch/Refresh Historical Prices**.
3. The bot scans ~900 date-slug pairs (Jan 1 2024 through yesterday),
   discovers every BTC daily market, and pulls CLOB price history for each
   NO token. This takes several minutes on first run.
4. After fetching, individual market slugs (e.g.
   `bitcoin-above-94k-on-november-15`) appear in the multiselect for
   backtesting across all available dates.

## Files Changed

- `fader/config/slugs.csv`
- `fader/backtest/historical.py`
- `fader/dashboard/backtest_page.py`
