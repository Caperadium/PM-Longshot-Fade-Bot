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


async def alert_ws_reconnect() -> None:
    await send("WS reconnected")


async def alert_gap_halt(duration_s: float) -> None:
    await send(f"<b>GAP HALT</b>: no book updates for {duration_s:.0f}s")


async def alert_exchange_rejection(slug: str, reason: str) -> None:
    await send(f"Order rejected for <b>{slug}</b>: {reason}")


async def alert_error(context: str, error: str) -> None:
    await send(f"<b>Error in {context}</b>:\n<code>{error[:300]}</code>")


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
