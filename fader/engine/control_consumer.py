"""engine/control_consumer.py

Polls control_commands table every control_poll_s.
Applies: stop, start, restart, close_all, breaker_reset, config_reload, slug_add, slug_remove.
Bounds emergency stop/close-all latency to ~1s.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from persistence.repos import control_repo

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ControlConsumer:
    def __init__(
        self,
        poll_s: float = 1.0,
        stop_engine_cb: Optional[Callable] = None,
        close_all_cb: Optional[Callable] = None,
        breaker_reset_cb: Optional[Callable] = None,
        config_reload_cb: Optional[Callable] = None,
        slug_change_cb: Optional[Callable] = None,
    ) -> None:
        self._poll_s = poll_s
        self._cbs: Dict[str, Optional[Callable]] = {
            "stop": stop_engine_cb,
            "start": stop_engine_cb,  # same dispatcher; restarts strategy loop
            "restart": stop_engine_cb,  # same dispatcher; graceful shutdown + exit 42
            "close_all": close_all_cb,
            "breaker_reset": breaker_reset_cb,
            "config_reload": config_reload_cb,
            "slug_add": slug_change_cb,
            "slug_remove": slug_change_cb,
        }
        self._task = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        while True:
            try:
                await self._process_pending()
            except Exception as e:
                logger.error(f"ControlConsumer error: {e}")
            await asyncio.sleep(self._poll_s)

    async def _process_pending(self) -> None:
        rows = control_repo.pending()

        for row in rows:
            cmd_id = row["id"]
            command = row["command"]
            try:
                args = json.loads(row["args_json"] or "{}")
            except Exception:
                args = {}

            result = await self._dispatch(command, args)
            _mark_done(cmd_id, result)

    async def _dispatch(self, command: str, args: Dict[str, Any]) -> str:
        cb = self._cbs.get(command)
        if cb is None:
            logger.warning(f"Unknown command: {command}")
            return "unknown_command"
        try:
            if asyncio.iscoroutinefunction(cb):
                await cb(command, args)
            else:
                cb(command, args)
            logger.info(f"Command executed: {command}")
            return "ok"
        except Exception as e:
            logger.error(f"Command {command} failed: {e}")
            return f"error: {e}"


def _mark_done(cmd_id: int, result: str) -> None:
    control_repo.mark_done(cmd_id, result)


def issue_command(command: str, args: Optional[Dict] = None) -> None:
    """Utility: write a command from dashboard or test code."""
    control_repo.issue(command, args)
