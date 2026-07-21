"""infra/telegram.py

Telegram alerting: heartbeat every N minutes + event alerts.
Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from env.
Uses raw HTTP (no heavy library dependency at runtime).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_BOT_TOKEN: Optional[str] = None
_CHAT_ID: Optional[str] = None
_ENABLED: bool = False

# Fire-and-forget task registry — prevents GC of in-flight alert tasks
# (asyncio only holds a weak reference to tasks not retained elsewhere).
_bg_tasks: set = set()


def fire(coro) -> None:
    """Fire-and-forget a telegram coroutine; retains a ref (GC-safe).
    Falls back to a daemon thread if no event loop is running."""
    try:
        t = asyncio.create_task(coro)
        _bg_tasks.add(t)
        t.add_done_callback(_bg_tasks.discard)
    except RuntimeError:
        import threading

        def _run():
            try:
                asyncio.run(coro)
            except Exception:
                pass

        threading.Thread(target=_run, daemon=True).start()


def configure(enabled: bool = True) -> None:
    global _BOT_TOKEN, _CHAT_ID, _ENABLED
    _BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    _CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    _ENABLED = enabled and bool(_BOT_TOKEN) and bool(_CHAT_ID)
    if enabled and not _ENABLED:
        logger.warning("Telegram enabled in config but TOKEN/CHAT_ID not set in env")


def _send_sync(text: str) -> bool:
    if not _ENABLED:
        return False
    try:
        url = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if not resp.ok:
            logger.warning(f"Telegram send failed: {resp.status_code} {resp.text[:100]}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Telegram error: {e}")
        return False


async def send(text: str) -> bool:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _send_sync, text)


async def alert_start(mode: str) -> None:
    await send(f"<b>Fader bot started</b>\nMode: {mode}")


async def alert_stop(reason: str = "") -> None:
    await send(f"<b>Fader bot stopped</b>\n{reason}")


async def alert_breaker_tripped(loss_pct: float, day: str) -> None:
    await send(
        f"<b>CIRCUIT BREAKER TRIPPED</b>\n"
        f"Day: {day}\nLoss: {loss_pct:.2f}%"
    )


async def alert_breaker_reset(day: str) -> None:
    await send(f"Circuit breaker reset for {day}")


async def alert_position_opened(slug: str, price: float, notional: float) -> None:
    await send(
        f"Position opened: <b>{slug}</b>\n"
        f"NO ask={price:.3f}  notional=${notional:.2f}"
    )


async def alert_position_resolved(slug: str, pnl: float) -> None:
    sign = "+" if pnl >= 0 else ""
    await send(f"Position resolved: <b>{slug}</b>\nPnL: {sign}{pnl:.2f} USDC")


async def alert_ws_disconnect(reason: str) -> None:
    await send(f"<b>WS disconnected</b>: {reason}")


async def alert_ws_reconnect(suppressed: int = 0) -> None:
    """suppressed = reconnects since the last alert that were throttled
    (WsClient fires at most one of these per RECONNECT_ALERT_MIN_INTERVAL_S)."""
    if suppressed > 0:
        await send(f"WS reconnected ({suppressed} earlier reconnects unreported)")
    else:
        await send("WS reconnected")


async def alert_gap_halt(duration_s: float) -> None:
    await send(f"<b>GAP HALT</b>: no book updates for {duration_s:.0f}s")


async def alert_exchange_rejection(slug: str, reason: str) -> None:
    await send(f"Order rejected for <b>{slug}</b>: {reason}")


async def alert_error(context: str, error: str) -> None:
    await send(f"<b>Error in {context}</b>:\n<code>{error[:300]}</code>")


async def alert_reconcile_failures(count: int, context: str, error: str) -> None:
    """Escalation alert (Phase 6, item 1): fired when a reconciler path
    (positions, paper resolutions, or order reconcile -- see `context`)
    has failed this many consecutive cycles in a row. Distinct from
    alert_error so this specific failure class is greppable/filterable
    in chat history."""
    await send(
        f"<b>RECONCILE FAILING</b> ({count} consecutive misses)\n"
        f"Context: {context}\n<code>{error[:300]}</code>"
    )


def format_bankroll_message(
    bankroll: float,
    open_positions: int,
    deployed: float,
    pnl_today: float,
    pnl_total: float,
) -> str:
    """Reply body for the /bankroll command."""
    return (
        "<b>Fader status</b>\n"
        f"Bankroll: ${bankroll:,.2f}\n"
        f"Open positions: {open_positions}\n"
        f"Deployed: ${deployed:,.2f}\n"
        f"PnL today: {pnl_today:+,.2f} USDC\n"
        f"PnL total: {pnl_total:+,.2f} USDC"
    )


def _get_updates_sync(offset: Optional[int], timeout: int) -> list:
    """Blocking getUpdates long poll. Returns [] when disabled or on error."""
    if not _ENABLED:
        return []
    try:
        url = f"https://api.telegram.org/bot{_BOT_TOKEN}/getUpdates"
        params: dict = {"timeout": timeout, "allowed_updates": '["message"]'}
        if offset is not None:
            params["offset"] = offset
        resp = requests.get(url, params=params, timeout=timeout + 15)
        if not resp.ok:
            logger.warning(
                f"Telegram getUpdates failed: {resp.status_code} {resp.text[:100]}"
            )
            return []
        return resp.json().get("result", [])
    except Exception as e:
        logger.warning(f"Telegram getUpdates error: {e}")
        return []


class CommandListenerTask:
    """Answer inbound commands from the configured chat via getUpdates.

    Commands: /bankroll -> reply with stats_fn() (async, returns the HTML
    message body). Messages from any other chat id are ignored. This is
    the only getUpdates consumer in the codebase -- Telegram allows one
    consumer per bot, and none may run while a webhook is set.
    """

    def __init__(self, stats_fn) -> None:
        self._stats_fn = stats_fn
        self._task: Optional[asyncio.Task] = None
        self._offset: Optional[int] = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        if not _ENABLED:
            return
        loop = asyncio.get_running_loop()
        # Drain the backlog so a restart doesn't replay commands sent
        # while the engine was down.
        backlog = await loop.run_in_executor(None, _get_updates_sync, None, 0)
        if backlog:
            self._offset = backlog[-1]["update_id"] + 1
        while True:
            try:
                updates = await loop.run_in_executor(
                    None, _get_updates_sync, self._offset, 30
                )
                for u in updates:
                    self._offset = u["update_id"] + 1
                    await self._handle(u)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Telegram command loop error: {e}")
                await asyncio.sleep(5)

    async def _handle(self, update: dict) -> None:
        msg = update.get("message") or {}
        chat_id = str((msg.get("chat") or {}).get("id", ""))
        text = (msg.get("text") or "").strip()
        if chat_id != str(_CHAT_ID):
            return
        if not text.startswith("/bankroll"):
            return
        try:
            reply = await self._stats_fn()
        except Exception as e:
            logger.warning(f"/bankroll stats error: {e}")
            reply = f"<b>Error building status</b>:\n<code>{str(e)[:200]}</code>"
        await send(reply)


class HeartbeatTask:
    """Send a periodic heartbeat message."""

    def __init__(self, interval_minutes: int = 15) -> None:
        self._interval = interval_minutes * 60
        self._task: Optional[asyncio.Task] = None
        self._start_time = time.time()

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            uptime_h = (time.time() - self._start_time) / 3600
            await send(
                f"Fader heartbeat -- uptime {uptime_h:.1f}h"
            )
