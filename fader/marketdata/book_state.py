"""marketdata/book_state.py

In-memory order book per contract + min-time-in-band tracker.

Band tracker watches the NO best ask. When ask enters [band_low, band_high]:
  - Start timer at that event timestamp.
  - If ask leaves band: reset timer to None.
  - Eligible when now - band_entry_ts >= min_time_in_band_s.
Driven by ws book/price_change events (not trade prints).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Level:
    price: Decimal
    size: Decimal


@dataclass
class OrderBook:
    token_id: str
    bids: List[Level] = field(default_factory=list)
    asks: List[Level] = field(default_factory=list)
    last_update_ts: float = 0.0  # monotonic seconds

    # Band-entry tracker
    band_entry_ts: Optional[float] = None  # monotonic ts when ask entered band

    @property
    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Optional[Decimal]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[Decimal]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_cents(self) -> Optional[float]:
        s = self.spread
        return float(s) * 100 if s is not None else None

    @property
    def ask_depth_usd(self) -> Decimal:
        if not self.asks:
            return Decimal(0)
        lvl = self.asks[0]
        return lvl.price * lvl.size

    def apply_snapshot(self, bids_raw: List[Dict], asks_raw: List[Dict]) -> None:
        self.bids = sorted(
            [Level(Decimal(b["price"]), Decimal(b["size"]))
             for b in bids_raw if b.get("price") and b.get("size")],
            key=lambda x: x.price, reverse=True,
        )
        self.asks = sorted(
            [Level(Decimal(a["price"]), Decimal(a["size"]))
             for a in asks_raw if a.get("price") and a.get("size")],
            key=lambda x: x.price,
        )
        self.last_update_ts = time.monotonic()

    def apply_delta(self, side: str, price_str: str, size_str: str) -> None:
        price = Decimal(price_str)
        size = Decimal(size_str)
        levels = self.bids if side.upper() == "BUY" else self.asks
        reverse = side.upper() == "BUY"

        if size == 0:
            levels[:] = [l for l in levels if l.price != price]
        else:
            # Update or insert
            found = False
            for l in levels:
                if l.price == price:
                    l.size = size
                    found = True
                    break
            if not found:
                levels.append(Level(price, size))
            levels.sort(key=lambda x: x.price, reverse=reverse)

        self.last_update_ts = time.monotonic()

    def update_band_tracker(
        self, band_low: float, band_high: float
    ) -> None:
        """Call after any book update to maintain band_entry_ts."""
        ask = self.best_ask
        if ask is None:
            self.band_entry_ts = None
            return
        ask_f = float(ask)
        in_band = band_low <= ask_f <= band_high
        if in_band:
            if self.band_entry_ts is None:
                self.band_entry_ts = time.monotonic()
        else:
            self.band_entry_ts = None

    def time_in_band(self) -> Optional[float]:
        """Seconds the NO ask has been continuously in-band. None if not in band."""
        if self.band_entry_ts is None:
            return None
        return time.monotonic() - self.band_entry_ts

    def is_in_band_long_enough(self, min_time_s: int) -> bool:
        t = self.time_in_band()
        return t is not None and t >= min_time_s


class BookStore:
    """Per-contract OrderBook registry."""

    def __init__(self) -> None:
        self._books: Dict[str, OrderBook] = {}

    def get(self, token_id: str) -> Optional[OrderBook]:
        return self._books.get(token_id)

    def get_or_create(self, token_id: str) -> OrderBook:
        if token_id not in self._books:
            self._books[token_id] = OrderBook(token_id=token_id)
        return self._books[token_id]

    def all_token_ids(self) -> List[str]:
        return list(self._books.keys())

    def snapshot(self, token_id: str, bids: List[Dict], asks: List[Dict]) -> None:
        book = self.get_or_create(token_id)
        book.apply_snapshot(bids, asks)

    def delta(self, token_id: str, side: str, price: str, size: str) -> None:
        book = self.get_or_create(token_id)
        book.apply_delta(side, price, size)

    def update_band_trackers(self, band_low: float, band_high: float) -> None:
        for book in self._books.values():
            book.update_band_tracker(band_low, band_high)
