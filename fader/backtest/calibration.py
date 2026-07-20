"""backtest/calibration.py

Market calibration: implied YES probability vs actual YES resolution rate.

The bot buys NO contracts on Polymarket in the price band [0.80, 0.95]
under the thesis that longshot YES contracts are systematically overpriced
(equivalently: the corresponding NO contracts are systematically
underpriced relative to their true win probability). This module measures
whether that bias actually holds in the historical data, independent of
any particular backtest run.

NO-price -> YES-implied conversion: a market's two outcomes are
complementary, so ``yes_implied = 1 - no_price``. The bot's trading band
on the NO side, [band_low, band_high] = [0.80, 0.95], corresponds to the
YES-implied longshot tail [1 - band_high, 1 - band_low] = [0.05, 0.20].
The thesis holds in a given implied-probability bucket if the actual
observed YES resolution rate (``yes_rate``) is LOWER than the mean implied
probability in that bucket (``mean_implied``) -- i.e. ``edge = mean_implied
- yes_rate > 0`` means the market over-priced the longshot, which is
exactly the mispricing the bot is faring against.

This module is pure and I/O-free: it consumes the DataFrame produced by
``backtest.historical.ContractPriceStore.snapshot()`` (see that module,
not imported here to keep this module free of any store/network
dependency) and returns DataFrames/dicts. No network calls, no file
reads, no Streamlit. Only pandas and the standard library are used --
the Wilson score interval is implemented by hand (no scipy).

Input snapshot columns (all strings, as stored in the CSV-backed price
store): ``slug``, ``token_id``, ``date`` (YYYY-MM-DD, the day of the price
observation), ``price`` (NO-token daily mid, stringified float),
``resolution`` (``"YES"``/``"NO"``/``""`` -- stamped only on the FINAL
observed day for a given token; it is a market-level fact, not a
per-row one), ``end_date`` (ISO date/timestamp string; only the first 10
characters, ``YYYY-MM-DD``, are used), ``fetched_at`` (ISO timestamp,
unused here). There is one row per ``(token_id, date)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Output columns of build_observations(), used both to build result rows
# and as the schema for the empty-input short-circuit.
_OBSERVATION_COLUMNS: Tuple[str, ...] = (
    "slug",
    "token_id",
    "end_date",
    "obs_date",
    "obs_dte",
    "no_price",
    "yes_implied",
    "resolution",
    "yes_won",
)

_BUCKET_COLUMNS: Tuple[str, ...] = (
    "bucket_mid",
    "n",
    "mean_implied",
    "yes_rate",
    "edge",
    "wilson_low",
    "wilson_high",
)

_MONTHLY_COLUMNS: Tuple[str, ...] = (
    "month",
    "n",
    "avg_no_price",
    "no_win_rate",
    "edge_pp",
)

_BOT_BUCKET_COLUMNS: Tuple[str, ...] = (
    "bucket_mid",
    "n",
    "avg_entry",
    "win_rate",
    "edge_pp",
    "wilson_low",
    "wilson_high",
)


@dataclass(frozen=True)
class CalibrationParams:
    """Parameters controlling how price observations are selected/bucketed.

    dte_days: how many days before end_date to sample the NO price at
        (the fixed "days to expiry" observation point).
    tolerance_days: max allowed |actual observation date - target date|
        for a token to be included; tokens with no row inside this window
        are dropped rather than approximated.
    bucket_width: width of the implied-probability buckets used by
        bucket_calibration()/bot_trade_calibration() (e.g. 0.05 = 5 pts).
    window_days: if set, only include tokens whose end_date falls within
        the last ``window_days`` days of ``as_of`` (or "today" in UTC).
    """

    dte_days: int = 4
    tolerance_days: int = 1
    bucket_width: float = 0.05
    window_days: Optional[int] = None


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    ``k`` successes out of ``n`` trials, at the confidence level implied
    by ``z`` (default 1.96 ~= 95%). Returns ``(0.0, 0.0)`` when ``n == 0``
    (undefined proportion) rather than raising or returning NaN, so
    callers can render a CI band unconditionally.
    """
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4 * n)) / n)
    low = (center - margin) / denom
    high = (center + margin) / denom
    return (max(0.0, low), min(1.0, high))


def _parse_date(value) -> Optional[date]:
    """Parse a stored date-ish value into a ``datetime.date``.

    Accepts plain ``date``/``datetime`` objects defensively, but the
    normal input is a string -- only the first 10 characters
    (``YYYY-MM-DD``) are used, since ``end_date`` may be a full ISO
    timestamp. Returns ``None`` on anything empty or unparseable.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if len(s) < 10:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def build_observations(
    snapshot_df: pd.DataFrame,
    params: CalibrationParams,
    as_of: Optional[date] = None,
) -> pd.DataFrame:
    """Reduce a raw price snapshot to one calibration observation per token.

    For each token: the resolution is the (single) non-empty
    ``resolution`` value found on ANY of the token's rows -- the store
    stamps it only on the final observed day, and since resolution is a
    market-level fact (not tied to that particular row), joining it back
    onto an earlier price observation for the same token is intentional
    and is exactly what this function does. Tokens with no non-empty
    resolution value on any row are excluded (still open / unresolved).

    The "end_date" used for a token is the first parseable end_date found
    among its rows (rows with empty/unparseable end_date are ignored for
    this purpose; a token with no parseable end_date at all is dropped).

    The observation itself is the token's row whose price is inside the
    open interval (0, 1) -- 0 and 1 are degenerate/resolved prices, not
    genuine market quotes -- and whose date is closest to
    ``end_date - dte_days``. Ties are broken by earlier date. If the
    closest candidate is still more than ``tolerance_days`` away from the
    target, the token is dropped (no interpolation/extrapolation).

    ``window_days`` (if set on ``params``) restricts to tokens whose
    end_date is >= ``(as_of or today in UTC) - window_days``.

    Returns a DataFrame with one row per surviving token and columns
    slug, token_id, end_date (date), obs_date (date), obs_dte (int),
    no_price (float), yes_implied (float), resolution (str), yes_won
    (bool). Empty/missing input returns an empty DataFrame with these
    columns.
    """
    if snapshot_df is None or snapshot_df.empty:
        return pd.DataFrame(columns=list(_OBSERVATION_COLUMNS))

    df = snapshot_df.copy()
    df["_end_date_parsed"] = df["end_date"].apply(_parse_date)
    df["_obs_date_parsed"] = df["date"].apply(_parse_date)
    df["_price_parsed"] = pd.to_numeric(df["price"], errors="coerce")

    rows: List[Dict] = []
    for token_id, grp in df.groupby("token_id", sort=False):
        res_vals = grp["resolution"].fillna("").astype(str).str.strip()
        res_vals = res_vals[res_vals != ""]
        if res_vals.empty:
            continue
        resolution = res_vals.iloc[0]

        valid_end = grp["_end_date_parsed"].dropna()
        if valid_end.empty:
            continue
        end_date = valid_end.iloc[0]

        target = end_date - timedelta(days=params.dte_days)

        cand = grp[grp["_obs_date_parsed"].notna() & grp["_price_parsed"].notna()]
        cand = cand[(cand["_price_parsed"] > 0.0) & (cand["_price_parsed"] < 1.0)]
        if cand.empty:
            continue

        cand = cand.copy()
        cand["_dist"] = cand["_obs_date_parsed"].apply(lambda d: abs((d - target).days))
        cand = cand.sort_values(
            by=["_dist", "_obs_date_parsed"], kind="mergesort"
        )
        best = cand.iloc[0]
        if int(best["_dist"]) > params.tolerance_days:
            continue

        no_price = float(best["_price_parsed"])
        obs_date = best["_obs_date_parsed"]
        rows.append(
            {
                "slug": best["slug"],
                "token_id": token_id,
                "end_date": end_date,
                "obs_date": obs_date,
                "obs_dte": (end_date - obs_date).days,
                "no_price": no_price,
                "yes_implied": 1.0 - no_price,
                "resolution": resolution,
                "yes_won": resolution == "YES",
            }
        )

    result = pd.DataFrame(rows, columns=list(_OBSERVATION_COLUMNS))

    if params.window_days is not None and not result.empty:
        cutoff = (as_of if as_of is not None else datetime.now(timezone.utc).date())
        cutoff = cutoff - timedelta(days=params.window_days)
        result = result[result["end_date"] >= cutoff].reset_index(drop=True)

    return result


def _bucket_index(values: pd.Series, bucket_width: float) -> pd.Series:
    """Floor-divide into buckets of width ``bucket_width``, clipped to
    ``[0, n_buckets - 1]``. A small epsilon guards against float
    imprecision landing exact bucket-boundary values one bucket low
    (e.g. ``0.15 / 0.05`` is ``2.9999999999999996`` in binary float)."""
    n_buckets = max(1, int(round(1.0 / bucket_width)))
    idx = values.apply(lambda v: int(math.floor(v / bucket_width + 1e-9)))
    return idx.clip(lower=0, upper=n_buckets - 1)


def bucket_calibration(obs: pd.DataFrame, bucket_width: float = 0.05) -> pd.DataFrame:
    """Bucket ``yes_implied`` over [0, 1) into width-``bucket_width`` bins.

    Per non-empty bucket: bucket_mid, n, mean_implied, yes_rate (mean of
    yes_won), edge (= mean_implied - yes_rate; positive means the market
    over-priced YES in that bucket, i.e. the fade thesis holds), plus a
    Wilson CI (wilson_low/wilson_high) on yes_rate. Buckets with no
    observations are simply absent from the output (not zero-filled).
    """
    if obs is None or obs.empty:
        return pd.DataFrame(columns=list(_BUCKET_COLUMNS))

    df = obs[(obs["yes_implied"] >= 0.0) & (obs["yes_implied"] < 1.0)].copy()
    if df.empty:
        return pd.DataFrame(columns=list(_BUCKET_COLUMNS))

    df["_bucket_idx"] = _bucket_index(df["yes_implied"], bucket_width)

    rows: List[Dict] = []
    for bidx, grp in df.groupby("_bucket_idx"):
        n = len(grp)
        mean_implied = float(grp["yes_implied"].mean())
        yes_rate = float(grp["yes_won"].mean())
        k = int(grp["yes_won"].sum())
        wlow, whigh = wilson_interval(k, n)
        rows.append(
            {
                "bucket_mid": bidx * bucket_width + bucket_width / 2.0,
                "n": n,
                "mean_implied": mean_implied,
                "yes_rate": yes_rate,
                "edge": mean_implied - yes_rate,
                "wilson_low": wlow,
                "wilson_high": whigh,
            }
        )

    result = pd.DataFrame(rows, columns=list(_BUCKET_COLUMNS))
    return result.sort_values("bucket_mid").reset_index(drop=True)


def band_summary(obs: pd.DataFrame, band_low: float, band_high: float) -> Dict[str, float]:
    """NO-side headline calibration restricted to the bot's trading band.

    Restricts to rows whose ``no_price`` is in ``[band_low, band_high]``
    (both ends inclusive). Returns n, avg_no_price, no_win_rate (= mean of
    NOT yes_won -- the NO contract's actual win rate), edge_pp (=
    no_win_rate - avg_no_price; positive means NO is systematically
    underpriced relative to its win rate), and a Wilson CI
    (wilson_low/wilson_high) on no_win_rate. All zero when there are no
    observations in the band (including empty ``obs``).
    """
    zeros = {
        "n": 0,
        "avg_no_price": 0.0,
        "no_win_rate": 0.0,
        "edge_pp": 0.0,
        "wilson_low": 0.0,
        "wilson_high": 0.0,
    }
    if obs is None or obs.empty:
        return zeros

    band = obs[(obs["no_price"] >= band_low) & (obs["no_price"] <= band_high)]
    n = len(band)
    if n == 0:
        return zeros

    avg_no_price = float(band["no_price"].mean())
    no_win = ~band["yes_won"].astype(bool)
    no_win_rate = float(no_win.mean())
    k = int(no_win.sum())
    wlow, whigh = wilson_interval(k, n)
    return {
        "n": n,
        "avg_no_price": avg_no_price,
        "no_win_rate": no_win_rate,
        "edge_pp": no_win_rate - avg_no_price,
        "wilson_low": wlow,
        "wilson_high": whigh,
    }


def monthly_edge(obs: pd.DataFrame, band_low: float, band_high: float) -> pd.DataFrame:
    """Monthly trend of the band edge -- "does the bias still exist".

    Same band restriction as ``band_summary`` (no_price in
    ``[band_low, band_high]`` inclusive), grouped by the calendar month of
    ``end_date`` (a ``"YYYY-MM"`` string in the ``month`` column). Per
    month: n, avg_no_price, no_win_rate, edge_pp. Sorted by month
    ascending.
    """
    if obs is None or obs.empty:
        return pd.DataFrame(columns=list(_MONTHLY_COLUMNS))

    band = obs[(obs["no_price"] >= band_low) & (obs["no_price"] <= band_high)].copy()
    if band.empty:
        return pd.DataFrame(columns=list(_MONTHLY_COLUMNS))

    band["month"] = band["end_date"].apply(lambda d: "%04d-%02d" % (d.year, d.month))

    rows: List[Dict] = []
    for month, grp in band.groupby("month"):
        no_win = ~grp["yes_won"].astype(bool)
        avg_no_price = float(grp["no_price"].mean())
        no_win_rate = float(no_win.mean())
        rows.append(
            {
                "month": month,
                "n": len(grp),
                "avg_no_price": avg_no_price,
                "no_win_rate": no_win_rate,
                "edge_pp": no_win_rate - avg_no_price,
            }
        )

    result = pd.DataFrame(rows, columns=list(_MONTHLY_COLUMNS))
    return result.sort_values("month").reset_index(drop=True)


def bot_trade_calibration(positions_df: pd.DataFrame, bucket_width: float = 0.05) -> pd.DataFrame:
    """Calibration of the bot's OWN realized trades (biased sample).

    This is conditioned on all of the live/backtest entry filters, so it
    is NOT a market-wide calibration estimate -- callers should label it
    as such. Input columns: entry_price (float or numeric string),
    realized_pnl (float or numeric string). The bot always holds NO, and
    "win" is not stored directly, so it is derived as ``realized_pnl >
    0``. Rows with missing/unparseable entry_price or realized_pnl
    (including closed-but-unsettled rows where realized_pnl is None) are
    dropped.

    Bucketed by entry_price (width ``bucket_width``): bucket_mid, n,
    avg_entry (the breakeven win rate implied by the average entry
    price), win_rate, edge_pp (= win_rate - avg_entry), and a Wilson CI
    (wilson_low/wilson_high) on win_rate.
    """
    if positions_df is None or positions_df.empty:
        return pd.DataFrame(columns=list(_BOT_BUCKET_COLUMNS))

    df = positions_df.copy()
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce")
    df = df.dropna(subset=["entry_price", "realized_pnl"])
    if df.empty:
        return pd.DataFrame(columns=list(_BOT_BUCKET_COLUMNS))

    df["_win"] = df["realized_pnl"] > 0
    df["_bucket_idx"] = _bucket_index(df["entry_price"], bucket_width)

    rows: List[Dict] = []
    for bidx, grp in df.groupby("_bucket_idx"):
        n = len(grp)
        avg_entry = float(grp["entry_price"].mean())
        win_rate = float(grp["_win"].mean())
        k = int(grp["_win"].sum())
        wlow, whigh = wilson_interval(k, n)
        rows.append(
            {
                "bucket_mid": bidx * bucket_width + bucket_width / 2.0,
                "n": n,
                "avg_entry": avg_entry,
                "win_rate": win_rate,
                "edge_pp": win_rate - avg_entry,
                "wilson_low": wlow,
                "wilson_high": whigh,
            }
        )

    result = pd.DataFrame(rows, columns=list(_BOT_BUCKET_COLUMNS))
    return result.sort_values("bucket_mid").reset_index(drop=True)


def filter_by_series(obs: pd.DataFrame, series_filter: str) -> pd.DataFrame:
    """Restrict observations to slugs containing ``series_filter``.

    Substring match, same semantics as
    ``dashboard.backtest_page._expand_series_slugs`` (matches anywhere in
    the slug, not just as a prefix -- some series ship children with a
    leading qualifier like "will-the-...").
    """
    if obs is None or obs.empty:
        return obs.copy() if obs is not None else pd.DataFrame(columns=list(_OBSERVATION_COLUMNS))
    mask = obs["slug"].astype(str).str.contains(series_filter, regex=False)
    return obs[mask].reset_index(drop=True)
