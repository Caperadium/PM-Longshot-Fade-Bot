"""execution/sizing.py

Share sizing helpers. Always rounds DOWN to preserve notional <= stake.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

# Minimum effective notional in USD. Prevents zero-size orders when
# alpha tilt pushes the multiplier near zero at band extremes.
MIN_EFFECTIVE_NOTIONAL = 1.00


def make_sizing_fn(
    alpha: float,
    band_low: float,
    band_high: float,
) -> Callable[[float], float]:
    """Return a sizing function for a given tilt a.

    f(p) = 1 + a * (p - p_mid) / (p_range / 2)

    Args:
        alpha: Tilt ∈ [-1, 1]. -1 = all weight at band_low, +1 = at band_high.
        band_low: Lower bound of entry band.
        band_high: Upper bound of entry band.

    Returns:
        Callable that maps fill_price -> notional multiplier.
    """
    p_mid = (band_high + band_low) / 2.0
    half_range = (band_high - band_low) / 2.0

    def _fn(fill_price: float) -> float:
        if half_range <= 0:
            return 1.0
        raw = 1.0 + alpha * (fill_price - p_mid) / half_range
        return raw

    return _fn


def compute_shares_and_notional(stake_usd: float, price: float) -> Tuple[float, float]:
    """
    Compute shares (round-DOWN to 2dp) and resulting notional.

    Returns:
        (size_shares, notional_usd)  where notional_usd <= stake_usd
    """
    if price <= 0 or price >= 1:
        raise ValueError(f"price must be in (0,1), got {price}")
    if stake_usd <= 0:
        raise ValueError(f"stake_usd must be > 0, got {stake_usd}")

    size_shares = math.floor((stake_usd / price) * 100) / 100
    notional_usd = size_shares * price
    return size_shares, notional_usd
