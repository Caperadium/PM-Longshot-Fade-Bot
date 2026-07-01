"""execution/provider.py

Polymarket REST provider for the fader bot.

Adapted from V2-Polymarket-BTC-Pricing-Program/polymarket/provider_polymarket.py.
Changes vs source:
- Generic token resolver via Gamma /markets?slug= (no BTC-specific parsing)
- NO token selection: outcome label case-insensitively == "No"
- EOA-only (no proxy/signature_type=2 path)
- All calls go through async RateLimiter
- Paper mode short-circuits place/cancel (simulate against live book)
- cancel_order / cancel_all added
- INVALID_ORDER_DUPLICATED treated as success
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, List, Optional, Tuple

import requests

from infra.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

DATA_API_BASE = "https://data-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
POLYGON_RPC_DEFAULT = "https://polygon-rpc.com"


def _polygon_rpc_url() -> str:
    """Polygon RPC endpoint. Override with POLYGON_RPC_URL env var.

    The public default (polygon-rpc.com) is frequently rate-limited / returns
    401, which silently zeroes on-chain balance + allowance reads. In live
    mode a zeroed balance means allow_entry() returns 'zero_bankroll' and the
    bot never trades — so a working RPC (Alchemy/Infura/etc.) is required.
    """
    return os.getenv("POLYGON_RPC_URL", "").strip() or POLYGON_RPC_DEFAULT
USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
USDC_E_DECIMALS = 6
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3


class MarketInfo:
    """Resolved market metadata."""
    __slots__ = ("slug", "condition_id", "token_id", "outcome", "outcome_index",
                 "question", "end_date_iso", "active", "closed")

    def __init__(
        self, slug: str, condition_id: str, token_id: str, outcome: str,
        outcome_index: int, question: str, end_date_iso: str,
        active: bool, closed: bool,
    ) -> None:
        self.slug = slug
        self.condition_id = condition_id
        self.token_id = token_id
        self.outcome = outcome
        self.outcome_index = outcome_index
        self.question = question
        self.end_date_iso = end_date_iso
        self.active = active
        self.closed = closed


class Provider:
    """
    Polymarket REST provider for the fader bot.

    paper mode: place_order / cancel_order / cancel_all log + return
                simulated responses without hitting the venue.
    live mode:  all calls go through the CLOB API via rate limiter.
    """

    def __init__(
        self,
        limiter: RateLimiter,
        mode: str = "paper",
        executor=None,
        paper_bankroll_usdc: float = 0.0,
    ) -> None:
        self._limiter = limiter
        self._mode = mode
        self._executor = executor
        self._paper_bankroll_usdc = paper_bankroll_usdc
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self._session.mount("https://", adapter)
        self._clob_client = None  # lazy init (requires private key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        json_body: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> Any:
        last_err = None
        for retry in range(MAX_RETRIES):
            try:
                resp = self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                    headers=headers,
                    timeout=timeout,
                )
                if resp.status_code == 429:
                    wait = 2 ** (retry + 1)
                    logger.warning(f"429 from {url}; wait {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_err = e
                if retry < MAX_RETRIES - 1:
                    time.sleep(2 ** retry)
        raise last_err or RuntimeError("Request failed")

    async def _aget(self, url: str, params: Optional[Dict] = None) -> Any:
        await self._limiter.acquire("read")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._request("GET", url, params=params),
        )

    def _get_clob_client(self):
        """Lazy-init py_clob_client for signing (live mode only)."""
        if self._clob_client is not None:
            return self._clob_client
        pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        if not pk:
            raise RuntimeError("POLYMARKET_PRIVATE_KEY not set")
        from py_clob_client.client import ClobClient
        client = ClobClient(host=CLOB_API_BASE, chain_id=137, key=pk)
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        self._clob_client = client
        return client

    # ------------------------------------------------------------------
    # Market resolution
    # ------------------------------------------------------------------

    def resolve_no_token(self, slug: str) -> MarketInfo:
        """
        Fetch market metadata and return the NO token.

        Calls Gamma /markets?slug=<slug>, finds outcome case-insensitively
        equal to "No", returns MarketInfo. Fails loudly if no NO outcome
        (non-binary market — should not be in slugs.csv).
        """
        params = {"slug": slug}
        data = self._request("GET", GAMMA_MARKETS_URL, params=params)
        markets = data if isinstance(data, list) else data.get("markets", [])
        if not markets:
            raise ValueError(f"No market found for slug={slug!r}")
        mkt = markets[0]
        raw_tokens = mkt.get("clobTokenIds", "[]")
        raw_outcomes = mkt.get("outcomes", "[]")
        token_ids: List[str] = (
            json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
        )
        outcomes: List[str] = (
            json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
        )
        for i, outcome in enumerate(outcomes):
            if outcome.strip().lower() == "no" and i < len(token_ids):
                return MarketInfo(
                    slug=slug,
                    condition_id=mkt.get("conditionId", ""),
                    token_id=token_ids[i],
                    outcome=outcome,
                    outcome_index=i,
                    question=mkt.get("question", ""),
                    end_date_iso=mkt.get("endDateIso", mkt.get("endDate", "")),
                    active=bool(mkt.get("active", True)),
                    closed=bool(mkt.get("closed", False)),
                )
        raise ValueError(
            f"Slug {slug!r} has no 'No' outcome — non-binary market, skip"
        )

    async def async_resolve_no_token(self, slug: str) -> MarketInfo:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, lambda: self.resolve_no_token(slug)
        )

    # ------------------------------------------------------------------
    # Balance / account
    # ------------------------------------------------------------------

    def fetch_usdc_balance(self) -> float:
        """On-chain USDC.e balance for the EOA address.

        In paper mode returns paper_bankroll_usdc minus deployed notional
        so bankroll reflects spend as positions open.
        """
        if self._mode == "paper":
            from engine.risk import get_open_notional
            total_deployed, _ = get_open_notional()
            return max(0.0, self._paper_bankroll_usdc - total_deployed)
        user_address = os.getenv("POLYMARKET_USER_ADDRESS", "")
        if not user_address:
            return 0.0
        try:
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider(_polygon_rpc_url()))
            abi = [{"inputs": [{"name": "account", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "", "type": "uint256"}],
                    "type": "function"}]
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(USDC_E_ADDRESS), abi=abi
            )
            raw = contract.functions.balanceOf(
                Web3.to_checksum_address(user_address)
            ).call()
            return raw / (10 ** USDC_E_DECIMALS)
        except Exception as e:
            logger.error(f"fetch_usdc_balance failed: {e}")
            return 0.0

    async def async_fetch_usdc_balance(self) -> float:
        await self._limiter.acquire("read")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.fetch_usdc_balance)

    def fetch_open_positions(self) -> List[Dict[str, Any]]:
        user = os.getenv("POLYMARKET_USER_ADDRESS", "")
        if not user:
            return []
        return self._request("GET", f"{DATA_API_BASE}/positions", params={"user": user}) or []

    async def async_fetch_open_positions(self) -> List[Dict[str, Any]]:
        await self._limiter.acquire("read")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.fetch_open_positions)

    def fetch_all_closed_positions(self, max_positions: int = 1000) -> List[Dict[str, Any]]:
        user = os.getenv("POLYMARKET_USER_ADDRESS", "")
        if not user:
            return []
        all_pos: List[Dict] = []
        offset = 0
        page_size = 50
        while len(all_pos) < max_positions:
            page = self._request(
                "GET",
                f"{DATA_API_BASE}/closed-positions",
                params={"user": user, "limit": page_size, "offset": offset,
                        "sortBy": "TIMESTAMP", "sortDirection": "DESC"},
            ) or []
            all_pos.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return all_pos

    def fetch_open_orders(self) -> List[Dict[str, Any]]:
        try:
            client = self._get_clob_client()
            orders = client.get_orders() or []
            result = []
            for o in orders:
                if isinstance(o, dict):
                    result.append({
                        "order_id": o.get("id"),
                        "token_id": o.get("asset_id"),
                        "side": o.get("side"),
                        "price": float(o.get("price", 0)),
                        "size": float(o.get("original_size", 0)),
                        "size_matched": float(o.get("size_matched", 0)),
                        "status": o.get("status"),
                    })
            return result
        except Exception as e:
            logger.error(f"fetch_open_orders: {e}")
            return []

    async def async_fetch_open_orders(self) -> List[Dict[str, Any]]:
        await self._limiter.acquire("read")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.fetch_open_orders)

    # ------------------------------------------------------------------
    # Order book / price
    # ------------------------------------------------------------------

    def fetch_order_book(self, token_id: str) -> Optional[Dict[str, Any]]:
        try:
            data = self._request(
                "GET", f"{CLOB_API_BASE}/book", params={"token_id": token_id}
            )
            bids = sorted(
                [{"price": Decimal(b["price"]), "size": Decimal(b["size"])}
                 for b in data.get("bids", []) if b.get("price") and b.get("size")],
                key=lambda x: x["price"], reverse=True,
            )
            asks = sorted(
                [{"price": Decimal(a["price"]), "size": Decimal(a["size"])}
                 for a in data.get("asks", []) if a.get("price") and a.get("size")],
                key=lambda x: x["price"],
            )
            best_bid = bids[0]["price"] if bids else None
            best_ask = asks[0]["price"] if asks else None
            return {
                "bids": bids,
                "asks": asks,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": (best_ask - best_bid) if best_bid and best_ask else None,
                "ask_depth_usd": (asks[0]["price"] * asks[0]["size"]) if asks else Decimal(0),
            }
        except Exception as e:
            logger.warning(f"fetch_order_book({token_id[:16]}): {e}")
            return None

    # ------------------------------------------------------------------
    # Order submission / cancellation
    # ------------------------------------------------------------------

    def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "LIMIT",
    ) -> Dict[str, Any]:
        """
        Place order. In paper mode: simulate without hitting venue.
        Returns {"success": bool, "order_id": str|None, "error": str|None,
                 "simulated": bool}
        """
        if self._mode == "paper":
            sim_id = f"SIM-{int(time.time()*1000)}"
            logger.info(
                f"[PAPER] place_order {side} {size:.4f}@{price:.4f} "
                f"token={token_id[:16]} -> {sim_id}"
            )
            return {"success": True, "order_id": sim_id, "error": None, "simulated": True}

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            client = self._get_clob_client()
            otype = OrderType.FOK if order_type == "MARKET" else OrderType.GTC
            args = OrderArgs(token_id=token_id, price=price, size=size, side=side)
            resp = client.create_and_post_order(args)
            order_id = None
            if isinstance(resp, dict):
                # A 200 response can still carry success=false + errorMsg
                # (e.g. FOK killed). Treat it as a rejection, not a fill.
                if resp.get("success") is False:
                    err = resp.get("errorMsg") or "order rejected (success=false)"
                    logger.error(f"place_order rejected: {err}")
                    return {"success": False, "order_id": None, "error": err,
                            "simulated": False}
                order_id = resp.get("orderID") or resp.get("order_id")
            elif hasattr(resp, "orderID"):
                order_id = resp.orderID
            logger.info(f"place_order: {side} {size:.4f}@{price:.4f} -> {order_id}")
            return {"success": True, "order_id": order_id, "error": None, "simulated": False}
        except Exception as e:
            err = str(e)
            from execution.idempotency import is_duplicate_error
            if is_duplicate_error(err):
                logger.info(f"Duplicate order (server-side dedup): {err}")
                return {"success": True, "order_id": None, "error": None, "simulated": False}
            logger.error(f"place_order failed: {err}")
            return {"success": False, "order_id": None, "error": err, "simulated": False}

    async def async_place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "LIMIT",
    ) -> Dict[str, Any]:
        await self._limiter.acquire("write")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.place_order(token_id, side, price, size, order_type),
        )

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        if self._mode == "paper":
            logger.info(f"[PAPER] cancel_order {order_id}")
            return {"success": True, "simulated": True}
        try:
            client = self._get_clob_client()
            resp = client.cancel(order_id)
            return {"success": True, "response": resp, "simulated": False}
        except Exception as e:
            logger.error(f"cancel_order({order_id}): {e}")
            return {"success": False, "error": str(e), "simulated": False}

    async def async_cancel_order(self, order_id: str) -> Dict[str, Any]:
        await self._limiter.acquire("write")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, lambda: self.cancel_order(order_id)
        )

    def cancel_all(self) -> Dict[str, Any]:
        if self._mode == "paper":
            logger.info("[PAPER] cancel_all")
            return {"success": True, "simulated": True}
        try:
            client = self._get_clob_client()
            resp = client.cancel_all()
            return {"success": True, "response": resp, "simulated": False}
        except Exception as e:
            logger.error(f"cancel_all: {e}")
            return {"success": False, "error": str(e), "simulated": False}

    async def async_cancel_all(self) -> Dict[str, Any]:
        await self._limiter.acquire("write")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.cancel_all)

    # ------------------------------------------------------------------
    # Market sell (close-all emergency)
    # ------------------------------------------------------------------

    async def async_market_sell(self, token_id: str, size: float) -> Dict[str, Any]:
        """Market-sell a position at best bid (SELL side)."""
        book = self.fetch_order_book(token_id)
        if not book or not book.get("best_bid"):
            return {"success": False, "error": "No bid available"}
        price = float(book["best_bid"])
        return await self.async_place_order(token_id, "SELL", price, size, "MARKET")

    # ------------------------------------------------------------------
    # MATIC balance
    # ------------------------------------------------------------------

    def fetch_matic_balance(self) -> float:
        """On-chain MATIC balance for the EOA address (for gas)."""
        user_address = os.getenv("POLYMARKET_USER_ADDRESS", "")
        if not user_address:
            return 0.0
        try:
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider(_polygon_rpc_url()))
            raw = w3.eth.get_balance(Web3.to_checksum_address(user_address))
            return raw / 10**18
        except Exception as e:
            logger.error(f"fetch_matic_balance failed: {e}")
            return 0.0

    async def async_fetch_matic_balance(self) -> float:
        await self._limiter.acquire("read")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.fetch_matic_balance)

    # ------------------------------------------------------------------
    # USDC.e allowance (for SELL orders / close_all)
    # ------------------------------------------------------------------

    def _check_allowance(self, spender_address: str) -> float:
        """Return USDC.e allowance for a single spender. Returns 0 on error."""
        user_address = os.getenv("POLYMARKET_USER_ADDRESS", "")
        if not user_address:
            return 0.0
        try:
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider(_polygon_rpc_url()))
            abi = [
                {
                    "constant": True,
                    "inputs": [
                        {"name": "owner", "type": "address"},
                        {"name": "spender", "type": "address"},
                    ],
                    "name": "allowance",
                    "outputs": [{"name": "", "type": "uint256"}],
                    "type": "function",
                }
            ]
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(USDC_E_ADDRESS), abi=abi
            )
            raw = contract.functions.allowance(
                Web3.to_checksum_address(user_address),
                Web3.to_checksum_address(spender_address),
            ).call()
            return raw / (10 ** USDC_E_DECIMALS)
        except Exception as e:
            logger.warning(f"allowance check for {spender_address[:10]}: {e}")
            return 0.0

    def fetch_usdc_allowance(self) -> float:
        """
        Total USDC.e allowance across both known Polymarket spender
        contracts (CTF Exchange + NegRiskAdapter).  Returns the larger.

        In paper mode returns 0.0 — no on-chain allowance needed.
        """
        if self._mode == "paper":
            return 0.0
        a1 = self._check_allowance(CTF_EXCHANGE_ADDRESS)
        a2 = self._check_allowance(NEG_RISK_ADAPTER)
        result = max(a1, a2)
        if result > 0:
            logger.info(f"USDC.e allowance: {result:.2f}")
        return result

    async def async_fetch_usdc_allowance(self) -> float:
        await self._limiter.acquire("read")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.fetch_usdc_allowance)

    # ------------------------------------------------------------------
    # Order recovery (server-side duplicate)
    # ------------------------------------------------------------------

    def find_order_by_params(
        self, token_id: str, side: str, size: float
    ) -> Optional[Dict[str, Any]]:
        """
        Recover a real order_id from open orders matching token_id, side,
        and original_size.  Uses original_size (not current size) because
        partial fills can shrink a resting order between the duplicate
        response and this recovery call.
        """
        try:
            client = self._get_clob_client()
            orders = client.get_orders() or []
            for o in orders:
                if not isinstance(o, dict):
                    continue
                o_token = o.get("asset_id") or o.get("token_id")
                o_side = o.get("side")
                o_orig_size = float(o.get("original_size", 0))
                if (
                    o_token == token_id
                    and (o_side or "").upper() == side.upper()
                    and abs(o_orig_size - size) < 0.01  # 1-share tolerance
                ):
                    return {
                        "order_id": o.get("id"),
                        "token_id": o_token,
                        "side": o_side,
                        "price": float(o.get("price", 0)),
                        "original_size": o_orig_size,
                        "size_matched": float(o.get("size_matched", 0)),
                        "status": o.get("status"),
                    }
        except Exception as e:
            logger.warning(f"find_order_by_params: {e}")
        return None


def snap_to_tick(price: Decimal, tick_size: Decimal, side: str) -> Decimal:
    ticks = price / tick_size
    if side.upper() == "BUY":
        snapped = ticks.quantize(Decimal("1"), rounding=ROUND_UP)
    else:
        snapped = ticks.quantize(Decimal("1"), rounding=ROUND_DOWN)
    return snapped * tick_size
