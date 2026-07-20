"""tests/test_telegram_bankroll.py

Unit tests for the /bankroll telegram command pieces:
  - infra.telegram.format_bankroll_message
  - infra.telegram.CommandListenerTask._handle (auth + dispatch)
  - PositionsRepo.open_notional / realized_pnl_total / realized_pnl_today
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

_FADER_ROOT = Path(__file__).parent.parent
if str(_FADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FADER_ROOT))


class TestFormatBankrollMessage(unittest.TestCase):
    def test_contents(self):
        from infra import telegram
        msg = telegram.format_bankroll_message(
            bankroll=928.39, open_positions=7, deployed=175.0,
            pnl_today=-3.5, pnl_total=41.25,
        )
        self.assertIn("$928.39", msg)
        self.assertIn("Open positions: 7", msg)
        self.assertIn("$175.00", msg)
        self.assertIn("-3.50", msg)
        self.assertIn("+41.25", msg)

    def test_thousands_separator(self):
        from infra import telegram
        msg = telegram.format_bankroll_message(
            bankroll=12345.6, open_positions=0, deployed=0.0,
            pnl_today=0.0, pnl_total=1234.5,
        )
        self.assertIn("$12,345.60", msg)
        self.assertIn("+1,234.50", msg)


class TestCommandListenerHandle(unittest.TestCase):
    """_handle: only the configured chat gets answers, only /bankroll."""

    def setUp(self):
        from infra import telegram
        self._telegram = telegram
        self._orig = (telegram._BOT_TOKEN, telegram._CHAT_ID, telegram._ENABLED)
        telegram._BOT_TOKEN = "tok"
        telegram._CHAT_ID = "12345"
        telegram._ENABLED = True

    def tearDown(self):
        t = self._telegram
        t._BOT_TOKEN, t._CHAT_ID, t._ENABLED = self._orig

    def _run(self, update, stats_reply="STATS"):
        from infra import telegram

        sent = []

        async def fake_send(text):
            sent.append(text)
            return True

        async def stats_fn():
            return stats_reply

        task = telegram.CommandListenerTask(stats_fn)
        with patch.object(telegram, "send", fake_send):
            asyncio.run(task._handle(update))
        return sent

    def test_bankroll_from_authorized_chat_answers(self):
        sent = self._run({
            "update_id": 1,
            "message": {"chat": {"id": 12345}, "text": "/bankroll"},
        })
        self.assertEqual(sent, ["STATS"])

    def test_bot_suffix_accepted(self):
        sent = self._run({
            "update_id": 2,
            "message": {"chat": {"id": 12345}, "text": "/bankroll@FaderBot"},
        })
        self.assertEqual(sent, ["STATS"])

    def test_wrong_chat_ignored(self):
        sent = self._run({
            "update_id": 3,
            "message": {"chat": {"id": 999}, "text": "/bankroll"},
        })
        self.assertEqual(sent, [])

    def test_other_text_ignored(self):
        sent = self._run({
            "update_id": 4,
            "message": {"chat": {"id": 12345}, "text": "hello"},
        })
        self.assertEqual(sent, [])

    def test_stats_error_reports_instead_of_crashing(self):
        from infra import telegram

        sent = []

        async def fake_send(text):
            sent.append(text)
            return True

        async def stats_fn():
            raise RuntimeError("db exploded")

        task = telegram.CommandListenerTask(stats_fn)
        with patch.object(telegram, "send", fake_send):
            asyncio.run(task._handle({
                "update_id": 5,
                "message": {"chat": {"id": 12345}, "text": "/bankroll"},
            }))
        self.assertEqual(len(sent), 1)
        self.assertIn("db exploded", sent[0])


class TestBankrollRepoMethods(unittest.TestCase):
    db_name = "test_fader_tg_bankroll.db"

    def setUp(self):
        os.environ["POLYMARKET_USER_ADDRESS"] = "0xTEST_USER"
        self.db_path = _FADER_ROOT / "tests" / self.db_name
        from infra.db import set_db_path, init_db
        set_db_path(self.db_path)
        if self.db_path.exists():
            self.db_path.unlink()
        init_db()

    def tearDown(self):
        if self.db_path.exists():
            self.db_path.unlink()

    @staticmethod
    def _insert(repo, pid, notional):
        repo.insert_open({
            "position_id": pid, "slug": "s1", "condition_id": "c1",
            "token_id": f"0x{pid}", "entry_price": 0.85, "size": 10.0,
            "notional": notional, "opened_at": "2026-01-01T00:00:00Z",
            "entry_order_id": f"o-{pid}", "entry_decision_id": f"ik-{pid}",
        })

    def test_open_notional_and_pnl_sums(self):
        from persistence.repos import PositionsRepo
        repo = PositionsRepo()
        self.assertEqual(repo.open_notional(), 0.0)
        self.assertEqual(repo.realized_pnl_total(), 0.0)
        self.assertEqual(repo.realized_pnl_today(), 0.0)

        self._insert(repo, "p1", 8.5)
        self._insert(repo, "p2", 4.5)
        self._insert(repo, "p3", 2.0)
        self.assertAlmostEqual(repo.open_notional(), 15.0)

        # Close p1 today (UTC), p2 on an old date; p3 stays OPEN.
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        repo.close("p1", realized_pnl=1.5, resolved_at=now_iso)
        repo.close("p2", realized_pnl=-4.0, resolved_at="2020-01-02T00:00:00Z")

        self.assertAlmostEqual(repo.open_notional(), 2.0)
        self.assertAlmostEqual(repo.realized_pnl_total(), -2.5)
        self.assertAlmostEqual(repo.realized_pnl_today(), 1.5)


if __name__ == "__main__":
    unittest.main()
